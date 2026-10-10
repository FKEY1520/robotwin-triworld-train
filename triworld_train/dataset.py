"""Read fixed future windows without duplicating episode images on disk."""
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .converter import VIEWS, ACTION_KEY
from .io import read_json, relative_file, sha256
from .verify import validate_index


class WindowDataset:
    def __init__(self, root, image_size=None, as_torch=False):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root / "dataset.json")
        if self.manifest.get("kind") != "triview_window_training" or self.manifest.get("split") != "train":
            raise ValueError("Expected a TriViewWan window training dataset")
        self.episodes = {e["episode_id"]: e for e in self.manifest["episodes"]}
        self.windows = [json.loads(line) for line in relative_file(self.root, self.manifest["windows_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
        if sha256(relative_file(self.root, self.manifest["windows_path"])) != self.manifest["windows_sha256"]:
            raise ValueError("Window index SHA-256 mismatch; regenerate dataset after edits")
        validate_index(self.manifest, self.windows)
        if not self.windows:
            raise ValueError("Dataset is empty")
        if image_size is not None and (len(image_size) != 2 or any(type(n) is not int or n < 1 for n in image_size)):
            raise ValueError("image_size must be (positive height, positive width)")
        self.image_size, self.as_torch = image_size, as_torch

    def __len__(self):
        return len(self.windows)

    def episode_actions(self):
        for episode in self.manifest["episodes"]:
            with h5py.File(relative_file(self.root, episode["file"]), "r") as handle:
                yield np.asarray(handle[ACTION_KEY])

    def __getitem__(self, index):
        row = self.windows[index]
        record = self.episodes[row["episode_id"]]
        start, horizon = row["future_start"], row["horizon"]
        history = row["history_indices"]
        indices = history + list(range(start, start + horizon))
        with h5py.File(relative_file(self.root, record["file"]), "r") as handle:
            actions = np.asarray(handle[ACTION_KEY], dtype=np.float32)[indices]
            # Decode repeated early-history indices once per view.
            videos = []
            for view in VIEWS:
                pictures = {}
                for frame in set(indices):
                    with Image.open(io.BytesIO(handle["frames/" + view][frame].tobytes())) as picture:
                        if self.image_size is not None:
                            picture = picture.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BICUBIC)
                        pictures[frame] = np.asarray(picture, dtype=np.float32).transpose(2, 0, 1) / np.float32(127.5) - np.float32(1.0)
                videos.append(np.stack([pictures[f] for f in indices], axis=1))
            video = np.stack(videos)
        result = {"episode_id": row["episode_id"], "sample_id": row["sample_id"], "source_uid": row["source_uid"],
                  "future_start": start, "horizon": horizon, "history_indices": history,
                  "history_video": video[:, :, :len(history)].copy(), "future_video": video[:, :, len(history):].copy(),
                  "packed_actions": actions, "prompt": record["prompt"],
                  "metadata": {"split": "train", "source_role": "training", "view_order": VIEWS, "action_key": ACTION_KEY,
                               "action_alignment": "frame_aligned", "time_basis": "frame_index", "fps": None,
                               "horizon": horizon, "bundle": record["bundle"], "task": record["task"]}}
        if self.as_torch:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("as_torch=True requires PyTorch; use the existing LRZ model environment") from exc
            for key in ("history_video", "future_video", "packed_actions"):
                result[key] = torch.from_numpy(result[key])
        return result
