"""Small real PNG/HDF5 fixtures test conversion and the loader together."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from triworld_train.converter import convert, fit_normalizer
from triworld_train.dataset import WindowDataset
from triworld_train.verify import export_npz, verify


def make_bundle(root: Path, lengths=(66,), action_dim=14, *, image_format="png"):
    """Create a relocated converter output; original absolute paths are stale."""
    entries = []
    actions = []
    for number, length in enumerate(lengths, start=1):
        episode = f"episode{number}"
        action = (np.arange(length, dtype=np.float64)[:, None]
                  + np.arange(action_dim, dtype=np.float64)[None, :] / 100)
        actions.append(action)
        numeric = root / "test_dataset" / "data" / f"{episode}.hdf5"
        numeric.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(numeric, "w") as handle:
            handle.create_dataset("joint_action/vector", data=action)
            handle.create_dataset("joint_action/left_arm", data=action[:, :6])
            handle.create_dataset("joint_action/left_gripper", data=action[:, 6])
            handle.create_dataset("joint_action/right_arm", data=action[:, 7:13])
            handle.create_dataset("joint_action/right_gripper", data=action[:, 13])
            for arm in ("left", "right"):
                endpose = np.zeros((length, 7), dtype=np.float64)
                endpose[:, 3] = 1
                handle.create_dataset(f"endpose/{arm}_endpose", data=endpose)
                handle.create_dataset(f"endpose/{arm}_gripper", data=np.ones(length))
        instructions = root / "test_dataset" / "instructions" / f"{episode}.json"
        instructions.parent.mkdir(parents=True, exist_ok=True)
        instructions.write_text(json.dumps({"instruction": "Pick up the bottle with the left arm."}),
                                encoding="utf-8")
        for view_index, view in enumerate(("head", "left", "right")):
            folder = root / ("lossless" if image_format == "png" else "gt_dataset") / episode / view / "frames"
            folder.mkdir(parents=True, exist_ok=True)
            for frame in range(length):
                rgb = np.empty((3, 4, 3), dtype=np.uint8)
                rgb[:] = [frame, view_index * 70, frame + 10]
                Image.fromarray(rgb).save(folder / f"frame_{frame:05d}.{image_format}")
        entries.append({
            "new_episode": episode, "source_episode": f"episode{number - 1}",
            "task": "fixture_task", "config_name": "aloha-agilex_clean_50",
            "frame_count": length, "instruction": "Pick up the bottle with the left arm.",
            "source_hdf5_rel": f"test_dataset/data/{episode}.hdf5",
            "source_instruction_json_rel": f"test_dataset/instructions/{episode}.json",
            "source_hdf5": f"Z:/unavailable/source/{episode}.hdf5",
            "source_instruction_json": f"Z:/unavailable/instructions/{episode}.json",
            "view_counts": dict.fromkeys(("head", "left", "right"), length),
            "view_shapes_hwc": dict.fromkeys(("head", "left", "right"), [3, 4, 3]),
            "lossless_frames_rel": f"lossless/{episode}" if image_format == "png" else None,
            "instruction_selection": {"policy": "consistent", "selected": {"json_path": "$.seen[3]"}},
            "scene_info": {"info": {"{a}": "left"}},
        })
    (root / "robotwin_episode_mapping.json").write_text(
        json.dumps({"schema_version": 2, "entries": entries}), encoding="utf-8")
    return actions


def content_snapshot(root):
    return {str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in root.rglob("*") if file.is_file()}


class _FixtureCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="triworld-train-test-")
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.output = self.base / "training"

    def tearDown(self):
        self.temp.cleanup()

    def run_conversion(self, **kwargs):
        return convert(self.source, self.output, **kwargs)


class PipelineTests(_FixtureCase):
    def test_loader_shapes_pixel_order_and_frame_aligned_actions(self):
        original = make_bundle(self.source)[0]
        before = content_snapshot(self.source)
        self.run_conversion()
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        self.assertEqual(len(dataset), 4)  # starts 1, 9, 17, 18
        first = dataset[0]
        self.assertEqual(first["history_video"].shape, (3, 3, 9, 3, 4))
        self.assertEqual(first["future_video"].shape, (3, 3, 48, 3, 4))
        self.assertEqual(first["packed_actions"].shape, (57, 14))
        self.assertEqual(first["future_video"].dtype, np.float32)
        np.testing.assert_array_equal(first["history_indices"], np.zeros(9, dtype=int))
        np.testing.assert_allclose(first["packed_actions"], original[[0] * 9 + list(range(1, 49))], atol=1e-6)
        np.testing.assert_allclose(first["future_video"][0, 0, :, 0, 0],
                                   np.arange(1, 49) / 127.5 - 1, atol=1e-6)
        np.testing.assert_allclose(first["future_video"][:, 1, 0, 0, 0],
                                   np.arange(3) * 70 / 127.5 - 1, atol=1e-6)
        self.assertEqual(content_snapshot(self.source), before)
        last = dataset[len(dataset) - 1]
        np.testing.assert_allclose(last["packed_actions"][-48:], original[-48:], atol=1e-6)
        np.testing.assert_allclose(last["future_video"][0, 0, -1, 0, 0], 65 / 127.5 - 1, atol=1e-6)

    def test_overlapping_windows_reference_the_same_global_timesteps(self):
        make_bundle(self.source)
        self.run_conversion()
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        first, second = dataset[0], dataset[1]
        np.testing.assert_array_equal(first["future_video"][:, :, 8:], second["future_video"][:, :, :40])
        np.testing.assert_array_equal(first["packed_actions"][-40:], second["packed_actions"][9:49])
        np.testing.assert_array_equal(second["history_indices"], np.arange(9))

    def test_dataset_is_self_contained_after_sources_are_deleted_and_output_moves(self):
        make_bundle(self.source, lengths=(49, 66))
        self.run_conversion()
        moved = self.base / "moved" / "portable_training"
        moved.parent.mkdir()
        shutil.move(str(self.output), str(moved))
        shutil.rmtree(self.source)
        dataset = WindowDataset(moved, image_size=None, as_torch=False)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset[0]["future_video"].shape[2], 48)
        self.assertEqual(dataset[4]["packed_actions"].shape, (57, 14))
        manifest = json.loads((moved / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["episodes"]), 2)
        self.assertTrue((moved / "windows.jsonl").is_file())
        self.assertEqual(len({record["file"] for record in manifest["episodes"]}), 2)
        for record in manifest["episodes"]:
            self.assertTrue((moved / record["file"]).is_file())

    def test_episode_actions_iterates_each_original_timestep_once(self):
        originals = make_bundle(self.source, lengths=(49, 66))
        self.run_conversion(stride=1)
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        arrays = list(dataset.episode_actions())
        self.assertEqual(len(arrays), 2)
        self.assertEqual(sum(len(array) for array in arrays), 115)
        for expected, actual in zip(originals, arrays):
            np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_normalizer_statistics_ignore_overlapping_window_repetition(self):
        originals = make_bundle(self.source, lengths=(49, 66))
        self.run_conversion(stride=1)
        normalizer = json.loads((self.output / "normalizer.json").read_text(encoding="utf-8"))
        unique_actions = np.concatenate(originals, axis=0)
        self.assertEqual(normalizer["method"], "bwm_p01_p99_clip")
        self.assertEqual(normalizer["statistics"]["unique_frames"], 115)
        self.assertEqual(normalizer["statistics"]["episode_count"], 2)
        self.assertFalse(normalizer["statistics"]["overlap_weighted"])
        np.testing.assert_allclose(normalizer["q01"], np.quantile(unique_actions, 0.01, axis=0), atol=1e-6)
        np.testing.assert_allclose(normalizer["q99"], np.quantile(unique_actions, 0.99, axis=0), atol=1e-6)

    def test_transform_edits_data_without_modifying_source(self):
        original = make_bundle(self.source, lengths=(49,))[0]
        before = content_snapshot(self.source)

        def edit(episode):
            return replace(episode, actions=episode.actions + 2, prompt="Custom training prompt.",
                           metadata={**episode.metadata, "transform_audit": "plus_two"})

        self.run_conversion(transform=edit)
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        np.testing.assert_allclose(dataset[0]["packed_actions"][-48:], original[1:] + 2, atol=1e-6)
        self.assertEqual(content_snapshot(self.source), before)
        manifest = json.loads((self.output / "dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["episodes"][0]["prompt"], "Custom training prompt.")

    def test_transform_cannot_break_frame_action_alignment(self):
        make_bundle(self.source, lengths=(49,))

        def truncate(episode):
            return replace(episode, actions=episode.actions[:-1])

        with self.assertRaises(ValueError):
            self.run_conversion(transform=truncate)
        self.assertFalse((self.output / "dataset.json").exists())

    def test_custom_action_fields_preserve_arm_and_gripper_order(self):
        original = make_bundle(self.source, lengths=(49,))[0]
        self.run_conversion(action_keys=("joint_action/left_arm", "joint_action/left_gripper",
                                         "joint_action/right_arm", "joint_action/right_gripper"))
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        np.testing.assert_allclose(dataset[0]["packed_actions"][-48:], original[1:], atol=1e-6)

    def test_loader_refuses_episode_paths_outside_the_dataset(self):
        make_bundle(self.source, lengths=(49,))
        self.run_conversion()
        path = self.output / "dataset.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["episodes"][0]["file"] = "../external.h5"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError):
            WindowDataset(self.output, image_size=None, as_torch=False)[0]

    def test_multiple_bundles_with_same_episode_names_remain_distinct(self):
        make_bundle(self.source / "task_one", lengths=(49,))
        make_bundle(self.source / "task_two", lengths=(49,))
        self.run_conversion()
        manifest = json.loads((self.output / "dataset.json").read_text(encoding="utf-8"))
        records = manifest["episodes"]
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["file"] for record in records}), 2)
        for record in records:
            self.assertTrue((self.output / record["file"]).is_file())
        self.assertEqual(len(WindowDataset(self.output, image_size=None, as_torch=False)), 2)

    def test_jpeg_fallback_without_lossless_folder(self):
        make_bundle(self.source, lengths=(49,), image_format="jpg")
        self.run_conversion(image_source="auto")
        dataset = WindowDataset(self.output, image_size=None, as_torch=False)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["future_video"].shape, (3, 3, 48, 3, 4))

    def test_existing_destination_is_never_overwritten(self):
        make_bundle(self.source, lengths=(49,))
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing data", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_conversion()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing data")

    def test_short_episode_requires_a_real_48_frame_future(self):
        make_bundle(self.source, lengths=(48,))
        with self.assertRaises(ValueError):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_missing_view_frame_is_rejected(self):
        make_bundle(self.source, lengths=(49,))
        (self.source / "lossless/episode1/right/frames/frame_00048.png").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_corrupt_image_is_rejected(self):
        make_bundle(self.source, lengths=(49,))
        (self.source / "lossless/episode1/head/frames/frame_00048.png").write_bytes(b"broken PNG")
        with self.assertRaises((ValueError, OSError)):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_nonfinite_actions_are_rejected(self):
        make_bundle(self.source, lengths=(49,))
        with h5py.File(self.source / "test_dataset/data/episode1.hdf5", "r+") as handle:
            handle["joint_action/vector"][48, 2] = np.nan
        with self.assertRaises(ValueError):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_complex_source_actions_are_rejected(self):
        make_bundle(self.source, lengths=(49,))
        with h5py.File(self.source / "test_dataset/data/episode1.hdf5", "r+") as handle:
            actions = handle["joint_action/vector"][:].astype(np.complex64) + 2j
            del handle["joint_action/vector"]
            handle.create_dataset("joint_action/vector", data=actions)
        with self.assertRaises(ValueError):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())

    def test_complex_transform_actions_are_rejected(self):
        make_bundle(self.source, lengths=(49,))

        def add_imaginary(episode):
            return replace(episode, actions=episode.actions.astype(np.complex64) + 2j)

        with self.assertRaises(ValueError):
            self.run_conversion(transform=add_imaginary)
        self.assertFalse((self.output / "dataset.json").exists())

    def test_standalone_normalizer_rejects_complex_actions(self):
        actions = np.ones((49, 14), dtype=np.complex64) * (1 + 2j)
        with self.assertRaises(ValueError):
            fit_normalizer([actions])

    def test_action_dimensions_cannot_silently_change(self):
        make_bundle(self.source, lengths=(49,), action_dim=15)
        with self.assertRaises(ValueError):
            self.run_conversion(expected_action_dim=14)
        self.assertFalse((self.output / "dataset.json").exists())

    def test_numeric_image_length_mismatch_is_rejected(self):
        make_bundle(self.source, lengths=(49,))
        with h5py.File(self.source / "test_dataset/data/episode1.hdf5", "r+") as handle:
            original = handle["joint_action/vector"][:]
            del handle["joint_action/vector"]
            handle.create_dataset("joint_action/vector", data=original[:-1])
        with self.assertRaises(ValueError):
            self.run_conversion()
        self.assertFalse((self.output / "dataset.json").exists())


class VerificationTests(_FixtureCase):
    def build(self, lengths=(66,)):
        make_bundle(self.source, lengths=lengths)
        self.run_conversion()

    def load_manifest(self):
        return json.loads((self.output / "dataset.json").read_text(encoding="utf-8"))

    def save_manifest(self, manifest):
        (self.output / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")

    def replace_rows_and_hash(self, rows):
        path = self.output / "windows.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = self.load_manifest()
        manifest["windows_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["window_count"] = len(rows)
        self.save_manifest(manifest)

    def test_deep_verification_decodes_every_view_frame(self):
        self.build(lengths=(49, 66))
        report = verify(self.output, deep=True)
        self.assertTrue(report["passed"])
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["frame_count"], 115)
        self.assertEqual(report["decoded_images"], 345)
        self.assertTrue(report["tail_window_verified"])

    def test_changed_window_bytes_fail_integrity_check(self):
        self.build()
        with (self.output / "windows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify(self.output)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            WindowDataset(self.output)

    def test_rehashed_future_leaking_history_is_still_rejected(self):
        self.build()
        rows = [json.loads(line) for line in (self.output / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["history_indices"][-1] = rows[0]["future_start"]
        self.replace_rows_and_hash(rows)
        with self.assertRaisesRegex(ValueError, "history|leakage"):
            verify(self.output)
        with self.assertRaisesRegex(ValueError, "history|leakage"):
            WindowDataset(self.output)

    def test_rehashed_missing_tail_window_is_still_rejected(self):
        self.build()
        rows = [json.loads(line) for line in (self.output / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
        self.replace_rows_and_hash(rows[:-1])
        with self.assertRaisesRegex(ValueError, "missing|plan"):
            verify(self.output)

    def test_rehashed_but_wrong_normalizer_is_still_rejected(self):
        self.build()
        path = self.output / "normalizer.json"
        normalizer = json.loads(path.read_text(encoding="utf-8"))
        normalizer["q01"][0] += 1
        path.write_text(json.dumps(normalizer), encoding="utf-8")
        manifest = self.load_manifest()
        manifest["normalizer_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Normalizer"):
            verify(self.output)

    def test_modified_episode_archive_fails_integrity_check(self):
        self.build()
        manifest = self.load_manifest()
        path = self.output / manifest["episodes"][0]["file"]
        with h5py.File(path, "r+") as handle:
            handle["joint_action/vector"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify(self.output)

    def test_truncated_archive_is_rejected_before_decoding(self):
        self.build()
        path = self.output / self.load_manifest()["episodes"][0]["file"]
        original = path.read_bytes()
        path.write_bytes(original[:len(original) // 2])
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify(self.output, deep=True)

    def test_npz_export_is_readable_without_pickle(self):
        self.build(lengths=(49,))
        target = self.base / "npz_export"
        report = export_npz(self.output, target, limit=1, image_size=(6, 8))
        self.assertEqual(report["sample_count"], 1)
        with np.load(target / "sample_00000000.npz", allow_pickle=False) as archive:
            self.assertEqual(archive["history_video"].shape, (3, 3, 9, 6, 8))
            self.assertEqual(archive["future_video"].shape, (3, 3, 48, 6, 8))
            self.assertEqual(archive["packed_actions"].shape, (57, 14))
            metadata = json.loads(str(archive["metadata_json"]))
            self.assertEqual(metadata["future_start"], 1)


if __name__ == "__main__":
    unittest.main()
