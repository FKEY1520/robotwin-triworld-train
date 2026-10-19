"""Recursive, independently checkpointed bundle conversion and aggregate publishing."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import h5py
import numpy as np

from .converter import (ACTION_KEY, _convert_bundle, _load_entries, _read_episode,
                        fit_normalizer, load_transform)
from .io import read_json, relative_file, sha256
from .windows import history_indices, window_starts

STATE = "batch_state.json"
REPORT = "batch_report.json"
RECEIPT = ".batch_source.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_fingerprint(bundle, settings, limit):
    """Hash metadata/actions; image identities use relative name, size and mtime_ns."""
    digest = hashlib.sha256()
    digest.update(sha256(bundle / "robotwin_episode_mapping.json").encode("ascii"))
    for entry in _load_entries(bundle, limit, check_inventory=True):
        data, _ = _read_episode(bundle, entry, settings["action_keys"], settings["expected_action_dim"], settings["image_source"])
        instruction = bundle / "test_dataset" / "instructions" / (entry["new_episode"] + ".json")
        digest.update(_digest([entry["new_episode"], data.metadata["source_action_sha256"], sha256(instruction)]).encode("ascii"))
        for view, paths in data.images.items():
            for path in paths:
                stat = path.stat()
                digest.update(_digest([view, path.relative_to(bundle).as_posix(), stat.st_size, stat.st_mtime_ns]).encode("ascii"))
    return {"schema": 1, "method": "mapping_actions_instructions_sha256_and_image_size_mtime_ns", "digest": digest.hexdigest()}


def _remove_stage(stage, target):
    # Only a directory created by this invocation directly inside the output root.
    if stage.exists():
        if stage.resolve().parent != target or not stage.name.startswith(".batch-partial-") or stage.is_symlink():
            raise ValueError("Refusing to clean an unexpected staging path")
        shutil.rmtree(stage)


def _withdraw_aggregate(root):
    """Keep old metadata as backups, with no stale aggregate training entry."""
    for name in ("dataset.json", "windows.jsonl", "normalizer.json", "report.json"):
        path = root / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected aggregate metadata path: {path}")
            path.replace(root / ("." + name + ".previous"))


def _publish_aggregate(root, sources, settings):
    """Stream the complete window index; retain one episode inventory and raw stats."""
    episodes, sequences, dataset_summaries, episode_summaries = [], [], [], []
    storage_bytes = 0
    source_uids = Counter()
    window_count = 0
    temporary = root / (".windows.jsonl.partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            for source in sources:
                child = root / source.dataset_id
                manifest = read_json(child / "dataset.json")
                child_report = read_json(child / "report.json")
                dataset_summaries.append({"dataset_id": source.dataset_id, "source_relative_path": source.relative_path,
                                          "episode_count": manifest["episode_count"], "window_count": manifest["window_count"]})
                for record in manifest["episodes"]:
                    path = relative_file(child, record["file"])
                    episodes.append({**record, "file": source.dataset_id + "/" + record["file"]})
                    with h5py.File(path, "r") as handle:
                        sequences.append(np.asarray(handle[ACTION_KEY]))
                    storage_bytes += path.stat().st_size
                    source_uids[record["source_uid"]] += 1
                episode_summaries.extend(child_report["episodes"])
                with relative_file(child, manifest["windows_path"]).open(encoding="utf-8") as windows:
                    for line in windows:
                        if line.strip():
                            output.write(line.rstrip("\r\n") + "\n")
                            window_count += 1
        temporary.replace(root / "windows.jsonl")
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(root / "normalizer.json", fit_normalizer(sequences))
    manifest = {"schema_version": 1, "kind": "triview_window_training", "split": "train", "source_role": "training",
                "view_order": ["head", "left", "right"], "action_key": ACTION_KEY, "action_alignment": "frame_aligned",
                "time_basis": "frame_index", "fps": None, "settings": settings, "episodes": episodes,
                "windows_path": "windows.jsonl", "windows_sha256": sha256(root / "windows.jsonl"),
                "normalizer_sha256": sha256(root / "normalizer.json"), "episode_count": len(episodes),
                "window_count": window_count, "bundles": [s.dataset_id for s in sources], "created_at": _now()}
    frame_count = sum(e["frame_count"] for e in episodes)
    report = {"episode_count": len(episodes), "frame_count": frame_count, "window_count": window_count,
              "horizon": settings["horizon"], "stride": settings["stride"], "history_frames": settings["history_frames"],
              "train_count": window_count, "validation_count": 0, "test_count": 0,
              "last_48_or_horizon_frames_used": True, "all_future_frames_covered": True,
              "image_decode_verified": True, "normalizer_uses_unique_timesteps": True,
              "bundles": manifest["bundles"], "storage_bytes": storage_bytes, "settings": settings,
              "episodes": episode_summaries, "datasets": dataset_summaries,
              "repeated_source_uid_count": sum(count - 1 for count in source_uids.values() if count > 1)}
    _atomic_json(root / "report.json", report)
    # The aggregate becomes readable only after every child and other metadata exist.
    _atomic_json(root / "dataset.json", manifest)
    return report


def batch_convert(source_root, output_root, horizon=48, stride=8, history_frames=9,
                  image_source="auto", action_keys=(ACTION_KEY,), expected_action_dim=14,
                  transform=None, tasks=None, limit=0, resume=False):
    from .discovery import discover_tree
    from .verify import verify

    history_indices(1, history_frames)
    window_starts(horizon + 1, horizon, stride)
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    if not action_keys or type(expected_action_dim) is not int or expected_action_dim < 1:
        raise ValueError("Provide action_keys and a positive expected_action_dim")
    source = Path(source_root).resolve()
    raw_target = Path(output_root)
    if raw_target.is_symlink():
        raise ValueError("Batch output cannot be a symbolic link")
    target = raw_target.resolve()
    if target == source or target in source.parents or source in target.parents:
        raise ValueError("Batch output must be outside the source tree")
    sources = discover_tree(source, tasks=tasks, exclude_paths=(target,))
    if not sources:
        raise ValueError(f"No converted RoboTwin bundles under {source}")
    settings = {"horizon": horizon, "stride": stride, "history_frames": history_frames,
                "history_packing": "bwm_anchor_recent_v1", "image_source": image_source,
                "action_keys": list(action_keys), "expected_action_dim": expected_action_dim,
                "transform": str(transform) if transform else None, "action_padding": "hold", "image_resize": False}
    config = {"settings": settings, "limit": limit}
    if isinstance(transform, str):
        config["transform_sha256"] = sha256(Path(transform.rpartition(":")[0]).resolve())
    config_digest = _digest(config)
    if target.exists():
        if not resume:
            raise FileExistsError(f"Batch output already exists: {target}; use --resume for a matching managed batch")
        marker = target / STATE
        if not marker.is_file() or marker.is_symlink():
            raise ValueError("--resume requires an output directory created by batch")
        state = read_json(marker)
        if state.get("schema") != 1 or state.get("kind") != "triview_batch_state" or state.get("source_root") != str(source):
            raise ValueError("Batch output belongs to another source root or schema")
        if state.get("config_sha256") != config_digest:
            raise ValueError("Batch configuration changed; choose a new output directory")
    else:
        target.mkdir(parents=True, exist_ok=False)
        state = {"schema": 1, "kind": "triview_batch_state", "source_root": str(source),
                 "config": config, "config_sha256": config_digest, "tasks": {}, "created_at": _now()}
        _atomic_json(target / STATE, state)
    lock = target / ".batch.lock"
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write(str(os.getpid()) + "\n")
    except FileExistsError as exc:
        raise ValueError("Batch output is locked; confirm the previous process has exited before removing .batch.lock") from exc
    records = [{"dataset_id": s.dataset_id, "source_relative_path": s.relative_path, "status": "pending"} for s in sources]
    report = {"schema": 1, "kind": "triview_batch_report", "source_root": str(source), "output_root": str(target),
              "recursive": True, "settings": settings, "limit": limit, "total_bundles": len(sources),
              "status": "running", "passed": False, "aggregate_ready": False, "tasks": records}

    def save():
        complete = [r for r in records if r["status"] in {"completed", "skipped"}]
        report.update(completed_count=len(complete), failed_count=sum(r["status"] == "failed" for r in records),
                      skipped_count=sum(r["status"] == "skipped" for r in records),
                      converted_count=sum(r["status"] == "completed" for r in records), updated_at=_now())
        for field in ("episode_count", "frame_count", "window_count"):
            report[field] = sum(r.get(field, 0) for r in complete)
        _atomic_json(target / STATE, state)
        _atomic_json(target / REPORT, report)

    try:
        # Keep the previous manifest as a backup, never expose a stale ready entry.
        _withdraw_aggregate(target)
        save()
        hook = load_transform(transform)
        for number, (item, record) in enumerate(zip(sources, records), 1):
            child = target / item.dataset_id
            stage = target / (".batch-partial-" + uuid.uuid4().hex)
            print(f"[BATCH {number}/{len(sources)}] {item.relative_path} -> {item.dataset_id}", flush=True)
            try:
                fingerprint = _source_fingerprint(item.path, settings, limit)
                identity = {"schema": 1, "dataset_id": item.dataset_id, "source_relative_path": item.relative_path,
                            "source_fingerprint": fingerprint, "config_sha256": config_digest}
                if child.exists():
                    if child.is_symlink() or not (child / RECEIPT).is_file():
                        raise ValueError("Existing child is not a completed managed batch dataset")
                    receipt = read_json(child / RECEIPT)
                    expected = {key: receipt.get(key) for key in identity}
                    if expected != identity:
                        raise ValueError("Source changed or identity differs; choose a new output version instead of overwriting completed data")
                    if receipt.get("dataset_sha256") != sha256(child / "dataset.json"):
                        raise ValueError("Completed dataset manifest changed")
                    checked = verify(child)
                    record.update(status="skipped", **{k: checked[k] for k in ("episode_count", "frame_count", "window_count")})
                    print(f"[BATCH] verified and reused {item.dataset_id}", flush=True)
                else:
                    _, _, _, child_report = _convert_bundle(item.path, stage, settings, hook, transform,
                                                            dataset_id=item.dataset_id, limit=limit, check_inventory=True,
                                                            source_relative_path=item.relative_path)
                    if _source_fingerprint(item.path, settings, limit) != fingerprint:
                        raise ValueError("Source changed while this bundle was being converted")
                    checked = verify(stage)
                    _atomic_json(stage / RECEIPT, {**identity, "dataset_sha256": sha256(stage / "dataset.json")})
                    stage.rename(child)
                    record.update(status="completed", **{k: child_report[k] for k in ("episode_count", "frame_count", "window_count")})
                state["tasks"][item.relative_path] = {**record, "source_fingerprint": fingerprint}
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                state["tasks"][item.relative_path] = dict(record)
                print(f"[BATCH FAILED] {item.relative_path}: {exc}", flush=True)
            finally:
                _remove_stage(stage, target)
            save()
        if not any(r["status"] == "failed" for r in records):
            try:
                aggregate = _publish_aggregate(target, sources, settings)
                report.update(status="complete", passed=True, aggregate_ready=True,
                              storage_bytes=aggregate["storage_bytes"], repeated_source_uid_count=aggregate["repeated_source_uid_count"])
            except Exception as exc:
                _withdraw_aggregate(target)
                report.update(status="partial", aggregate_error=f"{type(exc).__name__}: {exc}")
        else:
            report["status"] = "partial"
        save()
        return report
    except BaseException:
        report["status"] = "partial"
        save()
        raise
    finally:
        lock.unlink(missing_ok=True)
