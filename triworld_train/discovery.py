"""Discover converted task bundles in a project tree without reading their data.

Mapping files mark a bundle boundary. Once found, its image/source directories
are not traversed. Existing non-recursive converter discovery stays unchanged.
"""
from dataclasses import dataclass
import codecs
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata


@dataclass(frozen=True)
class BundleSource:
    path: Path
    relative_path: str
    dataset_id: str


_MARKER = "robotwin_episode_mapping.json"
_IGNORED = {
    ".git", ".venv", ".pytest_cache", "__pycache__", "venv", "node_modules",
    "$recycle.bin", "system volume information",
}
_RESERVED_OUTPUT = {
    "episodes", "dataset.json", "windows.jsonl", "normalizer.json", "report.json",
    "batch.json", "batch_manifest.json", "batch_report.json", "batch_state.json",
}
_WINDOWS_DEVICES = {"con", "prn", "aux", "nul", "clock$", "conin$", "conout$"}
_WINDOWS_DEVICES.update("com" + str(number) for number in range(1, 10))
_WINDOWS_DEVICES.update("lpt" + str(number) for number in range(1, 10))
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _name_key(value):
    """Use a conservative portable key for case/Unicode name collisions."""
    return unicodedata.normalize("NFKC", value).casefold()


def _portable_basename(value):
    if (not value or value in {".", ".."} or value.startswith(".")
            or value.endswith((".", " ")) or len(value.encode("utf-8")) > 240
            or re.fullmatch(r"[\w.-]+", value) is None):
        return False
    key = _name_key(value)
    return key not in _RESERVED_OUTPUT and key.split(".", 1)[0] not in _WINDOWS_DEVICES


def _slug(value):
    value = re.sub(r"[^\w.-]+", "-", unicodedata.normalize("NFKC", value)).strip(" .-")
    while len(value.encode("utf-8")) > 80:
        value = value[:-1]
    if not value:
        value = "bundle"
    if not _portable_basename(value):
        value = "bundle-" + value
    return value


def _within(path, ancestor):
    return path == ancestor or ancestor in path.parents


def _items(value):
    if value is None:
        return []
    return [value] if isinstance(value, (str, os.PathLike)) else list(value)


def _mapping_file(directory):
    marker = directory / _MARKER
    try:
        attributes = marker.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise OSError(error.errno, "Cannot inspect bundle marker: " + str(marker), str(marker)) from error
    return stat.S_ISREG(attributes.st_mode) and not getattr(attributes, "st_file_attributes", 0) & _REPARSE_POINT


def _metadata_kind(directory, filename):
    """Read at most 64 KiB and inspect only a top-level JSON kind field.

    Generated manifests put kind before their potentially large episode list.
    Decode individual prefix values, so a large/truncated trailing list need
    not be loaded. A kind string inside a nested object or quoted text does not
    classify an unrelated project directory as generated training output.
    """
    path = directory / filename
    try:
        attributes = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise OSError(error.errno, "Cannot inspect generated-data marker: " + str(path), str(path)) from error
    if not stat.S_ISREG(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & _REPARSE_POINT:
        return None
    try:
        with path.open("rb") as handle:
            prefix = handle.read(64 * 1024)
    except OSError as error:
        raise OSError(error.errno, "Cannot read generated-data marker: " + str(path), str(path)) from error
    try:
        # A 64 KiB cut may end inside a Chinese UTF-8 character. Defer only
        # that incomplete final character, while rejecting invalid bytes.
        text = codecs.getincrementaldecoder("utf-8-sig")("strict").decode(prefix, final=False)
    except UnicodeDecodeError:
        return None
    decoder = json.JSONDecoder()
    cursor = len(text) - len(text.lstrip())
    if cursor >= len(text) or text[cursor] != "{":
        return None
    cursor += 1
    try:
        while cursor < len(text):
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if cursor >= len(text) or text[cursor] == "}":
                return None
            key, cursor = decoder.raw_decode(text, cursor)
            if not isinstance(key, str):
                return None
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if cursor >= len(text) or text[cursor] != ":":
                return None
            cursor += 1
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            value, cursor = decoder.raw_decode(text, cursor)
            if key == "kind":
                return value if isinstance(value, str) else None
            cursor += len(text[cursor:]) - len(text[cursor:].lstrip())
            if cursor >= len(text) or text[cursor] != ",":
                return None
            cursor += 1
    except (ValueError, TypeError):
        return None
    return None


def _generated_tree(directory):
    return (_metadata_kind(directory, "dataset.json") == "triview_window_training"
            or _metadata_kind(directory, "batch_state.json") == "triview_batch_state")


def _allocate_ids(root, paths):
    relative = {path: (path.name if path == root else path.relative_to(root).as_posix()) for path in paths}
    # An immediate child keeps its established simple name. Nested identities
    # always include their relative-path digest, so adding another task later
    # does not rename an already completed nested task during batch --resume.
    immediate = {}
    for path in paths:
        if path == root or path.parent == root:
            immediate.setdefault(_name_key(path.name), []).append(path)
    result, allocated = [], {}
    for path in paths:
        name, identity = path.name, relative[path]
        nested = path != root and path.parent != root
        ambiguous = len(immediate.get(_name_key(name), [])) > 1
        if _portable_basename(name) and not nested and not ambiguous:
            dataset_id = name
        else:
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            dataset_id = (name if _portable_basename(name) else _slug(name)) + "--" + suffix
        key = _name_key(dataset_id)
        if key in allocated:
            raise ValueError("Portable dataset ID collision between {!r} and {!r}: {!r}".format(
                allocated[key], identity, dataset_id))
        allocated[key] = identity
        result.append(BundleSource(path=path, relative_path=identity, dataset_id=dataset_id))
    return result


def _select(sources, tasks):
    requested = _items(tasks)
    if not requested:
        return sources
    selected = set()
    for task in requested:
        identifier = str(task).replace("\\", "/")
        relative = PurePosixPath(identifier)
        if (not identifier or relative.is_absolute() or ".." in relative.parts
                or ":" in identifier or relative.as_posix() == "."):
            raise ValueError("Task identifiers must be relative bundle paths: " + identifier)
        identifier = relative.as_posix()
        exact = [source for source in sources if source.relative_path == identifier]
        if exact:
            matches = exact
        else:
            matches = [source for source in sources if _name_key(source.relative_path) == _name_key(identifier)]
            if not matches and "/" not in identifier:
                matches = [source for source in sources if _name_key(source.path.name) == _name_key(identifier)]
        if not matches:
            raise ValueError("Unknown bundle identifier {!r}; available: {}".format(
                identifier, ", ".join(source.relative_path for source in sources) or "(none)"))
        if len(matches) > 1:
            raise ValueError("Ambiguous bundle basename/path {!r}; choose its relative path: {}".format(
                identifier, ", ".join(source.relative_path for source in matches)))
        selected.add(matches[0].relative_path)
    return [source for source in sources if source.relative_path in selected]


def discover_tree(source_root, tasks=None, exclude_paths=()):
    """Return converted bundles in deterministic relative-path order.

    ``tasks`` accepts exact relative paths, or a uniquely matching basename.
    Relative ``exclude_paths`` are interpreted against the working directory;
    each excluded directory and all descendants are omitted. Links/junctions
    and infrastructure/dot directories are skipped. Recognized generated data
    and batch-output trees are pruned using small JSON metadata prefixes. An explicitly supplied root
    is resolved first and can itself be a bundle. Other filesystem errors fail
    with the offending path rather than silently omitting training data.
    """
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError("Source root must be a directory: " + str(root))
    excluded = [Path(path).expanduser().resolve() for path in _items(exclude_paths)]
    stack, found = [root], []
    while stack:
        directory = stack.pop()
        if any(_within(directory, path) for path in excluded):
            continue
        if _mapping_file(directory):
            found.append(directory)
            continue
        if _generated_tree(directory):
            continue
        try:
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda entry: (_name_key(entry.name), entry.name))
        except OSError as error:
            raise OSError(error.errno, "Cannot scan project directory: " + str(directory), str(directory)) from error
        children = []
        for entry in entries:
            if entry.name.startswith(".") or _name_key(entry.name) in _IGNORED:
                continue
            candidate = Path(entry.path)
            if any(_within(candidate, path) for path in excluded):
                continue
            try:
                attributes = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise OSError(error.errno, "Cannot inspect project entry: " + str(candidate), str(candidate)) from error
            if not stat.S_ISDIR(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & _REPARSE_POINT:
                continue
            resolved = candidate.resolve(strict=True)
            if not _within(resolved, root):
                raise ValueError("Project directory escaped the source root: " + str(candidate))
            children.append(resolved)
        stack.extend(reversed(children))
    found.sort(key=lambda path: (_name_key(path.relative_to(root).as_posix()), path.relative_to(root).as_posix()))
    return _select(_allocate_ids(root, found), tasks)
