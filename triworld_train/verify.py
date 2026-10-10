"""Validate manifests, windows, archive hashes and optionally every RGB frame."""
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .converter import ACTION_KEY, VIEWS, fit_normalizer
from .io import read_json, relative_file, sha256, write_json
from .windows import history_indices, window_starts


def validate_index(manifest, rows):
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "triview_window_training":
        raise ValueError("Unsupported dataset schema")
    contract = {"split": "train", "source_role": "training", "view_order": list(VIEWS), "action_key": ACTION_KEY,
                "action_alignment": "frame_aligned", "time_basis": "frame_index", "fps": None}
    if any(manifest.get(k) != v for k, v in contract.items()):
        raise ValueError("Dataset does not satisfy the native training contract")
    settings = manifest["settings"]
    episodes = manifest["episodes"]
    if not episodes or len(episodes) != manifest["episode_count"]:
        raise ValueError("Empty or mismatched episode count")
    by_id = {e["episode_id"]: e for e in episodes}
    if len(by_id) != len(episodes) or len(rows) != manifest["window_count"]:
        raise ValueError("Duplicate episode or mismatched window count")
    samples = set()
    observed = {key: [] for key in by_id}
    for row in rows:
        episode = by_id.get(row["episode_id"])
        if episode is None or row["sample_id"] in samples:
            raise ValueError("Unknown episode or duplicate window sample")
        samples.add(row["sample_id"])
        if row["split"] != "train" or row["source_uid"] != episode["source_uid"]:
            raise ValueError("Window split/source UID mismatch")
        start, horizon = row["future_start"], row["horizon"]
        if type(start) is not int or start < 1 or horizon != settings["horizon"] or start + horizon > episode["frame_count"]:
            raise ValueError("Window exceeds real frames or has incorrect horizon")
        if row["history_indices"] != history_indices(start, settings["history_frames"]) or row["target_end_exclusive"] != start + horizon:
            raise ValueError("Window history/target indices are inconsistent; future leakage possible")
        observed[episode["episode_id"]].append(start)
    for episode in episodes:
        if episode["split"] != "train" or episode["source_role"] != "training" or episode["action_dim"] != settings["expected_action_dim"]:
            raise ValueError("Invalid episode training contract")
        if observed[episode["episode_id"]] != window_starts(episode["frame_count"], settings["horizon"], settings["stride"]):
            raise ValueError("Window plan has missing, repeated or out-of-order targets")


def verify(dataset_root, deep=False):
    root = Path(dataset_root).resolve()
    manifest = read_json(root / "dataset.json")
    index = relative_file(root, manifest["windows_path"])
    if sha256(index) != manifest["windows_sha256"]:
        raise ValueError("Window index SHA-256 mismatch")
    normalizer_path = relative_file(root, "normalizer.json")
    if sha256(normalizer_path) != manifest["normalizer_sha256"]:
        raise ValueError("Normalizer SHA-256 mismatch")
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    validate_index(manifest, rows)
    decoded = 0
    sequences = []
    for episode in manifest["episodes"]:
        path = relative_file(root, episode["file"])
        if sha256(path) != episode["file_sha256"]:
            raise ValueError(f"Episode SHA-256 mismatch: {path}")
        with h5py.File(path, "r") as handle:
            actions = np.asarray(handle[ACTION_KEY])
            if actions.shape != (episode["frame_count"], episode["action_dim"]) or str(actions.dtype) != episode["action_dtype"] or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action archive: {path}")
            if handle.attrs["prompt"] != episode["prompt"]:
                raise ValueError("Archive/manifest prompt mismatch")
            sequences.append(actions)
            for view in VIEWS:
                frames = handle["frames/" + view]
                if len(frames) != len(actions):
                    raise ValueError(f"Image/action length mismatch: {path}/{view}")
                if deep:
                    for encoded in frames:
                        with Image.open(io.BytesIO(encoded.tobytes())) as picture:
                            picture.load()
                            if picture.mode != "RGB" or [picture.height, picture.width] != episode["image_shape"]:
                                raise ValueError(f"Invalid image mode/shape: {path}/{view}")
                            decoded += 1
        print(f"Verified {episode['episode_id']}", flush=True)
    expected = fit_normalizer(sequences)
    actual = read_json(normalizer_path)
    if actual != expected:
        raise ValueError("Normalizer does not match unique training timesteps")
    return {"passed": True, "episode_count": len(sequences), "window_count": len(rows),
            "frame_count": sum(len(a) for a in sequences), "decoded_images": decoded, "deep": deep,
            "all_future_frames_covered": True, "tail_window_verified": True, "all_splits_train": True}


def export_npz(dataset_root, output_root, limit=0, image_size=None):
    from .dataset import WindowDataset
    dataset = WindowDataset(dataset_root, image_size=image_size)
    target = Path(output_root).resolve()
    source = Path(dataset_root).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite NPZ output: {target}")
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("NPZ output must be outside the source dataset tree")
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be nonnegative")
    verify(source)
    target.mkdir(parents=True)
    count = min(limit, len(dataset)) if limit else len(dataset)
    for i in range(count):
        item = dataset[i]
        details = {k: v for k, v in item.items() if k not in ("history_video", "future_video", "packed_actions")}
        np.savez_compressed(target / f"sample_{i:08d}.npz", history_video=item["history_video"], future_video=item["future_video"],
                            packed_actions=item["packed_actions"], metadata_json=json.dumps(details, ensure_ascii=False))
    write_json(target / "export.json", {"count": count, "image_size": image_size, "dataset_sha256": sha256(source / "dataset.json"), "allow_pickle": False})
    return {"output_root": str(target), "sample_count": count}
