#!/usr/bin/env python3
"""将重叠三视角训练窗口编码成原 LRZ TriViewWan 的 schema 2 缓存。

不改变模型或原模型项目；需要已有真实 Wan T5/VAE 权重和原项目依赖。
--extract-only 可在登录节点安全解出单文件部署包，不执行部署控制器。
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def extract_model(model_file: Path, destination: Path) -> Path:
    """只读取两个字面量并检查 ZIP；绝不 import/exec 用户部署控制器。"""
    parsed = ast.parse(model_file.read_text(encoding="utf-8-sig"), filename=str(model_file))
    constants = {}
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in {"PAYLOAD_B64", "PAYLOAD_SHA256"}:
                    constants[name.id] = ast.literal_eval(node.value)
    if set(constants) != {"PAYLOAD_B64", "PAYLOAD_SHA256"}:
        raise ValueError("单文件必须包含 PAYLOAD_B64 与 PAYLOAD_SHA256 字面量。")
    encoded, expected = constants["PAYLOAD_B64"], constants["PAYLOAD_SHA256"]
    if not isinstance(encoded, str) or len(encoded) > 70 * 1024**2 or not isinstance(expected, str):
        raise ValueError("单文件部署包字面量类型或大小不合法。")
    content = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ValueError("内置 ZIP SHA256 不匹配。")
    destination = destination.expanduser().resolve()
    files = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) > 2000 or sum(entry.file_size for entry in entries) > 50 * 1024**2:
            raise ValueError("内置代码包超过 2000 文件或 50 MiB 上限。")
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename
                    or ":" in entry.filename or not relative.parts or relative.parts[0] != "TriViewWan"
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f"部署包路径不安全：{entry.filename}")
            if entry.is_dir():
                continue
            if entry.filename in files:
                raise ValueError(f"部署包包含重复文件：{entry.filename}")
            files[entry.filename] = archive.read(entry)
    if "TriViewWan/triviwan/train.py" not in files or "TriViewWan/triviwan/history.py" not in files:
        raise ValueError("部署包缺少原模型训练或历史模块。")
    receipt = destination / "model_source.json"
    if destination.exists():
        if destination.is_symlink() or not receipt.is_file():
            raise FileExistsError("解包目录已存在且无本脚本记录，请改用新目录。")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous.get("payload_sha256") != actual:
            raise FileExistsError("解包目录属于不同代码版本，请改用新目录。")
        for name, value in files.items():
            path = destination / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
                raise ValueError(f"已解包的模型代码被修改或缺失：{path}")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for name, value in files.items():
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
            if path.suffix == ".sh":
                path.chmod(0o755)
        write_json(receipt, {"schema_version": 1, "payload_sha256": actual,
                             "controller_sha256": sha256(model_file), "file_count": len(files),
                             "controller": str(model_file.resolve())})
    return destination / "TriViewWan"


def resolve_model(args) -> Path:
    if args.model_file:
        if not args.model_extract_dir:
            raise ValueError("--model-file 必须同时指定 --model-extract-dir。")
        root = extract_model(args.model_file.expanduser().resolve(), args.model_extract_dir)
    else:
        root = args.model_root.expanduser().resolve()
    required = ("triviwan/train.py", "triviwan/cache.py", "triviwan/history.py", "triviwan/normalization.py")
    if any(not (root / name).is_file() for name in required):
        raise FileNotFoundError(f"--model-root 必须指向原模型项目（包含 triviwan）：{root}")
    return root


def validate_sample(sample, torch, history_frames, expected_horizon, action_dim):
    """模型送入前确认三路、帧数、动作及时间契约，拒绝静默裁切。"""
    history = torch.as_tensor(sample["history_video"], dtype=torch.float32)
    future = torch.as_tensor(sample["future_video"], dtype=torch.float32)
    actions = torch.as_tensor(sample["packed_actions"], dtype=torch.float32)
    horizon = int(sample.get("horizon", future.shape[2]))
    if (history.ndim != 5 or tuple(history.shape[:3]) != (3, 3, history_frames)
            or future.ndim != 5 or tuple(future.shape[:3]) != (3, 3, horizon)
            or tuple(history.shape[-2:]) != tuple(future.shape[-2:])
            or actions.shape != (history_frames + horizon, action_dim)):
        raise ValueError(f"窗口张量形状不符：{sample.get('sample_id')}")
    if horizon != expected_horizon:
        raise ValueError(f"窗口 horizon={horizon}，配置要求 {expected_horizon}。")
    for tensor in (history, future, actions):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("窗口包含 NaN/Inf。")
    for tensor in (history, future):
        if float(tensor.min()) < -1.00001 or float(tensor.max()) > 1.00001:
            raise ValueError("窗口 RGB 必须已归一化为 [-1,1]。")
    metadata = sample["metadata"]
    if metadata.get("split") != "train" or metadata.get("action_alignment") != "frame_aligned":
        raise ValueError("只接受 train、frame_aligned 窗口。")
    return history, future, actions, horizon


def prepare(config_path: Path, model_root: Path) -> dict:
    if sys.version_info < (3, 11):
        raise RuntimeError("原模型编码/训练使用 tomllib，需 Python 3.11+；单独解包支持 Python 3.10+。")
    import torch
    from triworld_train.dataset import WindowDataset
    from triworld_train.verify import verify

    # 显式原项目优先，防止当前环境恰好装有另一个同名模型包。
    sys.path.insert(0, str(model_root))
    from triviwan.data import get_action_contract
    from triviwan.flow import latent_weights, pad_actions, pad_video
    from triviwan.history import history_indices, resolved_history
    from triviwan.normalization import fit_actions
    from triviwan.runtime import FrozenEncoders, fingerprint, read_config, source_provenance

    import triviwan
    if Path(triviwan.__file__).resolve().parent != model_root / "triviwan":
        raise RuntimeError("导入的 triviwan 并非 --model-root 中的原模型。")
    config = read_config(config_path)
    settings = config["data"]
    if settings.get("split", "train") != "train":
        raise ValueError("此入口仅建立 train 缓存。")
    if settings.get("action_padding", "hold") != "hold":
        raise ValueError("当前 joint_action/vector 为绝对关节目标，action_padding 必须 hold。")
    history_settings = resolved_history(config)
    history_frames = history_settings["num_frames"]
    horizon = settings["window_future"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 4 or horizon % 4:
        raise ValueError("data.window_future 必须是至少 4 的四的倍数。")
    if any(isinstance(settings[key], bool) or not isinstance(settings[key], int)
           or settings[key] <= 0 or settings[key] % 32 for key in ("height", "width")):
        raise ValueError("data.height/width 必须为 32 的正倍数。")
    manifest = Path(settings["manifest"]).expanduser().resolve()
    if manifest.name != "dataset.json":
        raise ValueError("data.manifest 必须指向本项目的 dataset.json。")
    dataset_root = manifest.parent
    windows_path = dataset_root / "windows.jsonl"
    manifest_digest, windows_digest = fingerprint(manifest), fingerprint(windows_path)
    # 逐源文件摘要、完整窗口计划、原始统计和 all-train 契约先验收，再加载昂贵编码器。
    verification = verify(dataset_root)
    dataset = WindowDataset(dataset_root, image_size=(settings["height"], settings["width"]), as_torch=True)
    if not len(dataset):
        raise ValueError("没有训练窗口。")
    output = Path(settings["cache_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "BUILDING.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                         "config": str(config_path.resolve()), "count": len(dataset)})
    try:
        # 独立拟合原模型 FP32 百分位规则。完整源轨迹仅出现一次；重叠窗口不改变统计权重。
        normalizer = fit_actions(torch.as_tensor(values, dtype=torch.float32) for values in dataset.episode_actions())
        provenance = source_provenance(config["paths"]["wan_source"])
        encoders = FrozenEncoders(config["paths"], config["runtime"].get("device", "cuda"))
        null_context = encoders.encode_text("")
        rows = []
        previous_prompt = None
        context = None
        contract = None
        for position in range(len(dataset)):
            sample = dataset[position]
            history, future, actions, count = validate_sample(
                sample, torch, history_frames, horizon, config["model"]["action_dim"])
            absolute_start = int(sample["future_start"])
            if list(sample["history_indices"]) != history_indices(absolute_start, history_frames):
                raise ValueError("窗口历史索引与原模型全局首帧加最近观测规则不一致。")
            current_contract = get_action_contract(sample["metadata"])
            if contract is not None and current_contract != contract:
                raise ValueError("同一训练缓存不能混合不同动作或时间契约。")
            contract = current_contract
            prompt = sample["prompt"]
            if prompt != previous_prompt or context is None:
                context = encoders.encode_text(prompt)
                previous_prompt = prompt
            clip = torch.cat((history, future), dim=2)
            payload = {"schema": 2, "clean": encoders.encode_video(pad_video(clip)),
                       "first": encoders.encode_video(history), "context": context,
                       "null_context": null_context,
                       "actions": pad_actions(actions[0], actions[1:], settings.get("action_padding", "hold")),
                       "weights": latent_weights(count, history_frames=history_frames),
                       "horizon": count, "history_frames": history_frames}
            if payload["first"].shape[2] != (history_frames - 1) // 4 + 1:
                raise ValueError("真实 VAE 历史 latent 帧数与原模型不一致。")
            cache_id = f"{position:08d}.pt"
            temporary = output / (cache_id + ".partial")
            torch.save(payload, temporary)
            temporary.rename(output / cache_id)
            rows.append({"path": cache_id, "episode_id": sample["episode_id"],
                         "sample_id": sample["sample_id"], "source_uid": sample["source_uid"],
                         "start": absolute_start - 1, "future_start": absolute_start,
                         "history_indices": list(sample["history_indices"]), "horizon": count, "split": "train"})
            print(f"CACHE {position + 1}/{len(dataset)} {sample['sample_id']} future_start={absolute_start}", flush=True)
        if fingerprint(manifest) != manifest_digest or fingerprint(windows_path) != windows_digest:
            raise RuntimeError("编码期间清单或窗口索引发生变化；缓存未发布。")
        verify(dataset_root)  # 编码结束时再次确认源 HDF5 内容，不能只检查小型清单。
        metadata = {"schema": 2, "manifest_sha256": manifest_digest, "windows_sha256": windows_digest,
                    "data": settings, "history": history_settings, "checkpoint": config["paths"]["checkpoint"],
                    "wan_source": provenance, "action_contract": contract, "count": len(rows),
                    "adapter": "triworld_train_overlap_v1", "model_root": str(model_root),
                    "dataset_verification": verification,
                    "model_code_sha256": {name: sha256(model_root / "triviwan" / name)
                                          for name in ("train.py", "history.py", "flow.py", "normalization.py")},
                    "normalizer_fit": "unique_full_training_episodes_once"}
        write_json(output / "action_normalizer.json", normalizer.state_dict())
        write_json(output / "metadata.json", metadata)
        # index.jsonl 是可训练入口，只有所有真实编码与元数据成功后才原子发布。
        temporary_index = output / "index.jsonl.partial"
        with temporary_index.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary_index.rename(output / "index.jsonl")
        (output / "BUILDING.json").unlink()
        print(f"CACHE_READY={output / 'index.jsonl'}", flush=True)
        return metadata
    except Exception as error:
        try:
            write_json(output / "FAILED.json", {"error_type": type(error).__name__, "error": str(error),
                                               "published": (output / "index.jsonl").exists()})
        except OSError as report_error:
            print(f"FAILED_REPORT_UNAVAILABLE={report_error}", file=sys.stderr, flush=True)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="与后续原生训练完全相同的 TOML 配置")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-root", type=Path, help="原 TriViewWan 项目根目录（包含 triviwan）")
    source.add_argument("--model-file", type=Path, help="已有 TriViewWan_LRZ.py 单文件部署包")
    parser.add_argument("--model-extract-dir", type=Path, help="--model-file 的独立解包目录，已有同版本可核对复用")
    parser.add_argument("--extract-only", action="store_true", help="只准备模型代码，无需 torch/GPU，不提交作业")
    args = parser.parse_args(argv)
    if not args.extract_only and args.config is None:
        parser.error("编码必须提供 --config；只解包时使用 --extract-only。")
    try:
        model_root = resolve_model(args)
        print(f"MODEL_ROOT={model_root}", flush=True)
        if not args.extract_only:
            prepare(args.config.expanduser().resolve(), model_root)
        return 0
    except Exception as error:
        print(f"CACHE_FAILED={type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
