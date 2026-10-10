"""Small portable file helpers."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file(base, value):
    path = Path(value)
    base = Path(base).resolve()
    result = (base / path).resolve()
    if path.is_absolute() or result == base or base not in result.parents:
        raise ValueError(f"Path must remain inside dataset: {value}")
    if not result.is_file():
        raise FileNotFoundError(result)
    return result
