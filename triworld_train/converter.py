"""Read existing converted bundles and build self-contained training datasets."""
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import uuid

import h5py
import numpy as np
from PIL import Image

from .io import read_json, write_json, sha256, relative_file
from .windows import history_indices, window_starts

VIEWS = ("head", "left", "right")
ACTION_KEY = "joint_action/vector"


@dataclass
class EpisodeData:
    """Hook input/output. Images are ordered file paths, actions are [T,A]."""
    actions: np.ndarray
    prompt: str
    images: dict
    metadata: dict


def load_transform(spec):
    if callable(spec) or spec is None:
        return spec
    filename, separator, function = str(spec).rpartition(":")
    if not separator or not function:
        raise ValueError("transform requires path/to/file.py:function")
    path = Path(filename).resolve()
    module_spec = importlib.util.spec_from_file_location("triworld_user_transform", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"Cannot load transform: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    result = getattr(module, function)
    if not callable(result):
        raise TypeError("Transform must be callable")
    return result


def discover_bundles(source_root, tasks=None):
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    bundles = [root] if (root / "robotwin_episode_mapping.json").is_file() else sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "robotwin_episode_mapping.json").is_file()], key=lambda p: p.name)
    if tasks:
        requested = set(tasks)
        bundles = [p for p in bundles if p.name in requested]
        missing = requested - {p.name for p in bundles}
        if missing:
            raise ValueError(f"Unknown bundle names: {sorted(missing)}")
    if not bundles:
        raise ValueError(f"No converted RoboTwin bundles under {root}")
    if len({p.name.casefold() for p in bundles}) != len(bundles):
        raise ValueError("Bundle names collide on Windows")
    for bundle in bundles:
        if not re.fullmatch(r"[\w.-]+", bundle.name) or bundle.name in {".", "..", "episodes"}:
            raise ValueError(f"Unsupported bundle name: {bundle.name}")
    return bundles


def _image_paths(bundle, episode, count, image_source):
    variants = [("png", "lossless"), ("jpg", "gt_dataset")] if image_source == "auto" else [(image_source, "lossless" if image_source == "png" else "gt_dataset")]
    if image_source not in ("auto", "png", "jpg"):
        raise ValueError("image_source must be auto, png or jpg")
    for extension, directory in variants:
        images = {}
        valid = True
        for view in VIEWS:
            folder = bundle / directory / episode / view / "frames"
            paths = sorted(folder.glob(f"frame_*.{extension}"))
            expected = [f"frame_{i:05d}.{extension}" for i in range(count)]
            if [p.name for p in paths] != expected:
                valid = False
                break
            images[view] = paths
        if valid:
            return images, extension
    raise ValueError(f"No complete synchronized {image_source} image set: {bundle.name}/{episode}")


def _read_episode(bundle, entry, action_keys, expected_action_dim, image_source):
    episode = entry.get("new_episode")
    if not isinstance(episode, str) or re.fullmatch(r"episode\d+", episode) is None:
        raise ValueError("Invalid mapped episode ID")
    action_path = bundle / "test_dataset" / "data" / (episode + ".hdf5")
    with h5py.File(action_path, "r") as handle:
        arrays = []
        for key in action_keys:
            if key not in handle or not isinstance(handle[key], h5py.Dataset):
                raise ValueError(f"Missing action dataset {key}: {action_path}")
            values = np.asarray(handle[key])
            if values.ndim == 1:
                values = values[:, None]
            if values.ndim != 2 or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values) or not np.isfinite(values).all():
                raise ValueError(f"Invalid action values/shape for {key}: {action_path}")
            arrays.append(values)
        available_fields = []
        def list_numeric(name, obj):
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                available_fields.append({"key": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
        handle.visititems(list_numeric)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError("Action fields have different time lengths")
    actions = np.concatenate(arrays, axis=1)
    if actions.shape[1] != expected_action_dim:
        raise ValueError(f"Expected action_dim={expected_action_dim}, got {actions.shape[1]}: {episode}")
    if entry.get("frame_count", len(actions)) != len(actions):
        raise ValueError(f"Mapping/actions frame count mismatch: {episode}")
    instruction_path = bundle / "test_dataset" / "instructions" / (episode + ".json")
    instruction = read_json(instruction_path)
    prompt = instruction.get("instruction") if isinstance(instruction, dict) else instruction
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Missing selected instruction: {episode}")
    if entry.get("instruction", prompt) != prompt:
        raise ValueError(f"Mapping/instruction mismatch: {episode}")
    images, extension = _image_paths(bundle, episode, len(actions), image_source)
    metadata = {"bundle": bundle.name, "episode": episode, "source_episode": entry.get("source_episode", episode),
                "task": entry.get("task", bundle.name), "config_name": entry.get("config_name"),
                "source_action_sha256": sha256(action_path), "image_encoding": extension,
                "instruction_selection": entry.get("instruction_selection"), "arm_evidence": entry.get("arm_evidence"),
                "source_action_keys": list(action_keys), "available_numeric_fields": available_fields, "transformed": False}
    return EpisodeData(actions, prompt, images, metadata), action_path


def _validate_episode(data, expected_action_dim):
    if not isinstance(data, EpisodeData):
        raise TypeError("Transform must return EpisodeData")
    a = np.asarray(data.actions)
    if a.ndim != 2 or a.shape[1] != expected_action_dim or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) or not np.isfinite(a).all():
        raise ValueError("Transform actions must be finite [T,action_dim] numeric arrays")
    if not isinstance(data.prompt, str) or not data.prompt.strip():
        raise ValueError("Transform must provide a nonempty prompt")
    if set(data.images) != set(VIEWS) or any(len(data.images[v]) != len(a) for v in VIEWS):
        raise ValueError("All three image sequences must match action length")
    data.actions = a
    return data


def fit_normalizer(sequences):
    if not sequences or any(np.iscomplexobj(x) for x in sequences):
        raise ValueError("Normalizer requires nonempty real-valued actions")
    values = np.concatenate([np.asarray(x, dtype=np.float32) for x in sequences], axis=0)
    if not np.isfinite(values).all():
        raise ValueError("Actions overflow float32 model input")
    bounds = np.quantile(values, [0.01, 0.99], axis=0).astype(np.float32)
    return {"schema": 1, "method": "bwm_p01_p99_clip", "q01": bounds[0].tolist(), "q99": bounds[1].tolist(),
            "eps": 1e-8, "constant_policy": "zero", "statistics": {"unique_frames": len(values), "episode_count": len(sequences), "overlap_weighted": False}}


def inspect(source_root, tasks=None, limit=0, horizon=48, stride=8, history_frames=9,
            image_source="auto", action_keys=(ACTION_KEY,), expected_action_dim=14):
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    window_starts(horizon + 1, horizon, stride)
    result = {"bundles": [], "episode_count": 0, "frame_count": 0, "window_count": 0, "horizon": horizon, "stride": stride}
    history_indices(1, history_frames)
    for bundle in discover_bundles(source_root, tasks):
        entries = read_json(bundle / "robotwin_episode_mapping.json")["entries"]
        entries = sorted(entries, key=lambda e: int(e["new_episode"][7:]))
        if limit:
            entries = entries[:limit]
        row = {"bundle": bundle.name, "episode_count": len(entries), "frame_count": 0, "window_count": 0, "image_encodings": []}
        for entry in entries:
            data, _ = _read_episode(bundle, entry, action_keys, expected_action_dim, image_source)
            starts = window_starts(len(data.actions), horizon, stride)
            row["frame_count"] += len(data.actions)
            row["window_count"] += len(starts)
            row["image_encodings"].append(data.metadata["image_encoding"])
            if "available_numeric_fields" not in row:
                row["available_numeric_fields"] = data.metadata["available_numeric_fields"]
                row["action_dim"] = data.actions.shape[1]
                row["action_dtype"] = str(data.actions.dtype)
        row["image_encodings"] = sorted(set(row["image_encodings"]))
        result["bundles"].append(row)
        for key in ("episode_count", "frame_count", "window_count"):
            result[key] += row[key]
    result["image_decode_verified"] = False
    return result


def _store_episode(path, data, action_source):
    shape = None
    with h5py.File(path, "x") as output:
        output.attrs["schema_version"] = 1
        output.attrs["view_order"] = json.dumps(VIEWS)
        output.attrs["prompt"] = data.prompt
        output.create_dataset(ACTION_KEY, data=data.actions, compression="gzip", shuffle=True)
        # Keep other numeric source channels available for future adapters.
        with h5py.File(action_source, "r") as source:
            def copy_numeric(name, obj):
                if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number):
                    values = np.asarray(obj)
                    if values.ndim < 1 or len(values) != len(data.actions) or np.iscomplexobj(values) or not np.isfinite(values).all():
                        raise ValueError(f"Source numeric field is not frame aligned/finite: {name}")
                    output.create_dataset("source_numeric/" + name, data=values, compression="gzip", shuffle=True)
            source.visititems(copy_numeric)
        for view in VIEWS:
            dataset = output.create_dataset("frames/" + view, (len(data.actions),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for index, source_path in enumerate(data.images[view]):
                content = Path(source_path).read_bytes()
                with Image.open(io.BytesIO(content)) as picture:
                    picture.load()
                    current = (picture.height, picture.width, picture.mode)
                    if current[2] != "RGB" or (shape is not None and current != shape):
                        raise ValueError(f"Expected synchronized RGB images of identical dimensions: {source_path}")
                    shape = current
                dataset[index] = np.frombuffer(content, dtype=np.uint8)
        output.attrs["image_shape"] = shape[:2]
    return list(shape[:2])


def _write_manifest(root, episodes, windows, settings, sequences, bundle_names):
    windows_path = root / "windows.jsonl"
    windows_path.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in windows), encoding="utf-8")
    manifest = {"schema_version": 1, "kind": "triview_window_training", "split": "train", "source_role": "training",
                "view_order": list(VIEWS), "action_key": ACTION_KEY, "action_alignment": "frame_aligned", "time_basis": "frame_index", "fps": None,
                "settings": settings, "episodes": episodes, "windows_path": "windows.jsonl", "windows_sha256": sha256(windows_path),
                "episode_count": len(episodes), "window_count": len(windows), "bundles": bundle_names,
                "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(root / "normalizer.json", fit_normalizer(sequences))
    manifest["normalizer_sha256"] = sha256(root / "normalizer.json")
    write_json(root / "dataset.json", manifest)
    report = {"episode_count": len(episodes), "frame_count": sum(e["frame_count"] for e in episodes), "window_count": len(windows),
              "horizon": settings["horizon"], "stride": settings["stride"], "history_frames": settings["history_frames"],
              "train_count": len(windows), "validation_count": 0, "test_count": 0,
              "last_48_or_horizon_frames_used": True, "all_future_frames_covered": True,
              "image_decode_verified": True, "normalizer_uses_unique_timesteps": True, "bundles": bundle_names,
              "storage_bytes": sum((root / e["file"]).stat().st_size for e in episodes),
              "settings": settings, "episodes": [{"episode_id": e["episode_id"], "frame_count": e["frame_count"],
                 "window_count": sum(w["episode_id"] == e["episode_id"] for w in windows),
                 "last_future_start": e["frame_count"] - settings["horizon"], "last_target_frame": e["frame_count"] - 1} for e in episodes]}
    write_json(root / "report.json", report)
    return report


def convert(source_root, output_root, horizon=48, stride=8, history_frames=9, image_source="auto",
            action_keys=(ACTION_KEY,), expected_action_dim=14, transform=None, tasks=None, limit=0):
    history_indices(1, history_frames)
    window_starts(horizon + 1, horizon, stride)
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if not action_keys or type(expected_action_dim) is not int or expected_action_dim < 1:
        raise ValueError("Provide action_keys and a positive expected_action_dim")
    bundles = discover_bundles(source_root, tasks)
    source = Path(source_root).resolve()
    target = Path(output_root).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {target}; choose a new output directory")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Output must be outside the source tree")
    hook = load_transform(transform)
    settings = {"horizon": horizon, "stride": stride, "history_frames": history_frames, "history_packing": "bwm_anchor_recent_v1",
                "image_source": image_source, "action_keys": list(action_keys), "expected_action_dim": expected_action_dim,
                "transform": str(transform) if transform else None, "action_padding": "hold", "image_resize": False}
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / ("." + target.name + ".partial-" + uuid.uuid4().hex)
    stage.mkdir()
    all_episodes, all_windows, all_sequences = [], [], []
    try:
        for bundle in bundles:
            root = stage / bundle.name
            (root / "episodes").mkdir(parents=True)
            entries = read_json(bundle / "robotwin_episode_mapping.json")["entries"]
            entries = sorted(entries, key=lambda e: int(e["new_episode"][7:]))
            if limit:
                entries = entries[:limit]
            if not entries or len({e["new_episode"] for e in entries}) != len(entries):
                raise ValueError("Mapping must contain nonempty unique episodes")
            episodes, windows, sequences = [], [], []
            for number, entry in enumerate(entries, 1):
                data, action_source = _read_episode(bundle, entry, action_keys, expected_action_dim, image_source)
                provenance = dict(data.metadata)
                if hook:
                    data = hook(data)
                data = _validate_episode(data, expected_action_dim)
                provenance["transformed"] = hook is not None
                if hook:
                    provenance["transform_metadata"] = data.metadata
                    if isinstance(transform, str):
                        transform_file = Path(transform.rpartition(":")[0]).resolve()
                        provenance["transform_sha256"] = sha256(transform_file)
                # Same frame count preserves source_numeric semantics for hooks.
                if len(data.actions) != entry.get("frame_count", len(data.actions)):
                    raise ValueError("Transform cannot change frame count; configure a new source bundle for resampling")
                starts = window_starts(len(data.actions), horizon, stride)
                episode_id = bundle.name + "__" + entry["new_episode"]
                filename = "episodes/" + entry["new_episode"] + ".h5"
                image_shape = _store_episode(root / filename, data, action_source)
                source_uid = "sha256:" + provenance["source_action_sha256"]
                record = {"episode_id": episode_id, "source_uid": source_uid, "split": "train", "source_role": "training",
                          "task": provenance["task"], "bundle": bundle.name, "file": filename, "file_sha256": sha256(root / filename),
                          "frame_count": len(data.actions), "action_dim": expected_action_dim, "action_dtype": str(data.actions.dtype),
                          "image_shape": image_shape, "prompt": data.prompt, "provenance": provenance}
                episodes.append(record)
                sequences.append(data.actions)
                for start in starts:
                    windows.append({"sample_id": episode_id + f"__f{start:06d}", "episode_id": episode_id, "source_uid": source_uid,
                                    "split": "train", "bundle": bundle.name, "task": record["task"], "future_start": start, "horizon": horizon,
                                    "history_indices": history_indices(start, history_frames), "target_end_exclusive": start + horizon})
                print(f"[{bundle.name}] {number}/{len(entries)} {entry['new_episode']}: {len(data.actions)} frames -> {len(starts)} windows", flush=True)
            _write_manifest(root, episodes, windows, settings, sequences, [bundle.name])
            all_episodes.extend([{**e, "file": bundle.name + "/" + e["file"]} for e in episodes])
            all_windows.extend(windows)
            all_sequences.extend(sequences)
        report = _write_manifest(stage, all_episodes, all_windows, settings, all_sequences, [b.name for b in bundles])
        stage.rename(target)
        return {"output_root": str(target), **report}
    except BaseException:
        if stage.resolve().parent == target.parent and stage.name.startswith("." + target.name + ".partial-"):
            shutil.rmtree(stage)
        raise
