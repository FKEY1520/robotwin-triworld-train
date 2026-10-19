"""End-to-end recursive batches, resumability and incomplete-output isolation."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from test_pipeline import content_snapshot, make_bundle
from triworld_train.batch import batch_convert
from triworld_train.cli import main
from triworld_train.dataset import WindowDataset
from triworld_train.verify import verify


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def archive_snapshot(root):
    return {str(path.relative_to(root)): (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
            for path in root.rglob("*.h5")}


class RecursiveBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="triworld-batch-test-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.source = self.base / "source"
        self.output = self.base / "training"

    def make_tree(self):
        arrays = []
        for relative, length in (("alpha", 49), ("group/beta", 66), ("group/deeper/gamma", 65)):
            arrays.extend(make_bundle(self.source / relative, lengths=(length,)))
        return arrays

    def run_batch(self, **kwargs):
        return batch_convert(self.source, self.output, **kwargs)

    def task_by_relative(self, report, relative):
        return next(task for task in report["tasks"] if task["source_relative_path"] == relative)

    def damage_beta_image(self):
        path = self.source / "group/beta/lossless/episode1/head/frames/frame_00065.png"
        path.write_bytes(b"corrupt PNG data")
        return path

    def assert_no_ready_aggregate(self):
        for name in ("dataset.json", "windows.jsonl", "normalizer.json", "report.json"):
            self.assertFalse((self.output / name).exists(), name)
        with self.assertRaises((ValueError, FileNotFoundError)):
            WindowDataset(self.output)

    def test_all_nested_tasks_complete_with_one_archive_per_episode(self):
        arrays = self.make_tree()
        before = content_snapshot(self.source)
        report = self.run_batch()
        self.assertTrue(report["passed"])
        self.assertTrue(report["aggregate_ready"])
        self.assertEqual(report["status"], "complete")
        self.assertEqual((report["total_bundles"], report["completed_count"], report["failed_count"]), (3, 3, 0))
        self.assertEqual((report["episode_count"], report["frame_count"], report["window_count"]), (3, 180, 8))
        self.assertEqual(len(list(self.output.rglob("*.h5"))), 3)
        self.assertFalse((self.output / "episodes").exists())
        self.assertEqual(content_snapshot(self.source), before)
        manifest = read_json(self.output / "dataset.json")
        self.assertEqual(len({entry["file"] for entry in manifest["episodes"]}), 3)
        for task in report["tasks"]:
            child = self.output / task["dataset_id"]
            self.assertEqual(verify(child)["episode_count"], 1)
            self.assertTrue((child / "episodes/episode1.h5").is_file())
            self.assertFalse((child / task["dataset_id"]).exists())
        checked = verify(self.output, deep=True)
        self.assertEqual(checked["decoded_images"], 540)
        normalizer = read_json(self.output / "normalizer.json")
        unique = np.concatenate(arrays).astype(np.float32)
        self.assertEqual(normalizer["statistics"]["unique_frames"], 180)
        self.assertFalse(normalizer["statistics"]["overlap_weighted"])
        np.testing.assert_allclose(normalizer["q01"], np.quantile(unique, 0.01, axis=0), atol=1e-6)
        np.testing.assert_allclose(normalizer["q99"], np.quantile(unique, 0.99, axis=0), atol=1e-6)

    def test_full_episode_future_coverage_and_global_anchor_are_preserved(self):
        self.make_tree()
        self.run_batch()
        dataset = WindowDataset(self.output)
        for episode in dataset.manifest["episodes"]:
            rows = [row for row in dataset.windows if row["episode_id"] == episode["episode_id"]]
            used = {frame for row in rows for frame in range(row["future_start"], row["target_end_exclusive"])}
            self.assertEqual(used, set(range(1, episode["frame_count"])))
            self.assertEqual(rows[-1]["future_start"], episode["frame_count"] - 48)
            for row in rows:
                self.assertEqual(row["history_indices"][0], 0)
                self.assertTrue(all(index < row["future_start"] for index in row["history_indices"]))
        last = dataset[-1]
        self.assertEqual(last["future_video"].shape, (3, 3, 48, 3, 4))
        self.assertEqual(last["packed_actions"].shape, (57, 14))

    def test_source_root_can_itself_be_one_bundle(self):
        make_bundle(self.source, lengths=(49, 66))
        report = self.run_batch()
        self.assertEqual(report["total_bundles"], 1)
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["window_count"], 5)
        self.assertEqual(verify(self.output)["episode_count"], 2)

    def test_nested_duplicate_names_generate_distinct_portable_task_datasets(self):
        make_bundle(self.source / "one/same_name", lengths=(49,))
        make_bundle(self.source / "two/same_name", lengths=(66,))
        report = self.run_batch()
        identifiers = [task["dataset_id"] for task in report["tasks"]]
        self.assertEqual(len(set(identifiers)), 2)
        manifest = read_json(self.output / "dataset.json")
        self.assertEqual(len({entry["episode_id"] for entry in manifest["episodes"]}), 2)
        for identifier in identifiers:
            self.assertTrue((self.output / identifier / "dataset.json").is_file())
            self.assertEqual(verify(self.output / identifier)["episode_count"], 1)

    def test_moved_aggregate_remains_readable_after_source_is_deleted(self):
        self.make_tree()
        self.run_batch()
        moved = self.base / "moved_training"
        self.output.rename(moved)
        self.assertTrue(self.source.resolve().is_relative_to(self.base))
        shutil.rmtree(self.source)
        self.assertEqual(len(WindowDataset(moved)), 8)
        self.assertEqual(WindowDataset(moved)[-1]["packed_actions"].shape, (57, 14))
        self.assertEqual(verify(moved, deep=True)["decoded_images"], 540)

    def test_damaged_bundle_does_not_stop_other_tasks_or_publish_partial_aggregate(self):
        self.make_tree()
        self.damage_beta_image()
        report = self.run_batch()
        self.assertFalse(report["passed"])
        self.assertFalse(report["aggregate_ready"])
        self.assertEqual(report["status"], "partial")
        self.assertEqual((report["completed_count"], report["failed_count"]), (2, 1))
        failed = self.task_by_relative(report, "group/beta")
        self.assertEqual(failed["status"], "failed")
        self.assertTrue(failed["error"])
        saved = read_json(self.output / "batch_report.json")
        self.assertFalse(saved["passed"])
        self.assert_no_ready_aggregate()
        for task in report["tasks"]:
            if task["status"] != "failed":
                self.assertEqual(verify(self.output / task["dataset_id"])["episode_count"], 1)

    def test_resume_skips_unchanged_completed_archives_without_rewriting(self):
        self.make_tree()
        self.run_batch()
        before = archive_snapshot(self.output)
        resumed = self.run_batch(resume=True)
        self.assertTrue(resumed["passed"])
        self.assertEqual((resumed["skipped_count"], resumed["converted_count"], resumed["failed_count"]), (3, 0, 0))
        self.assertEqual(archive_snapshot(self.output), before)

    def test_resume_retries_failed_task_and_preserves_completed_tasks(self):
        self.make_tree()
        damaged = self.damage_beta_image()
        failed = self.run_batch()
        self.assertFalse(failed["passed"])
        before = archive_snapshot(self.output)
        Image.new("RGB", (4, 3), color=(65, 0, 75)).save(damaged)
        resumed = self.run_batch(resume=True)
        self.assertTrue(resumed["passed"])
        self.assertEqual((resumed["skipped_count"], resumed["converted_count"], resumed["failed_count"]), (2, 1, 0))
        after = archive_snapshot(self.output)
        for relative, evidence in before.items():
            self.assertEqual(after[relative], evidence)
        self.assertTrue((self.output / "dataset.json").is_file())
        self.assertEqual(verify(self.output)["episode_count"], 3)

    def test_resume_rejects_changed_completed_source_instead_of_silently_skipping(self):
        self.make_tree()
        self.run_batch()
        before = archive_snapshot(self.output)
        with h5py.File(self.source / "alpha/test_dataset/data/episode1.hdf5", "r+") as handle:
            handle["joint_action/vector"][0, 0] += 1
        resumed = self.run_batch(resume=True)
        self.assertFalse(resumed["passed"])
        self.assertEqual(resumed["failed_count"], 1)
        self.assertEqual(self.task_by_relative(resumed, "alpha")["status"], "failed")
        self.assertEqual(archive_snapshot(self.output), before)
        self.assert_no_ready_aggregate()

    def test_resume_new_nested_duplicate_keeps_old_task_ids_and_archives(self):
        make_bundle(self.source / "one/same_name", lengths=(49,))
        make_bundle(self.source / "other", lengths=(49,))
        first = self.run_batch()
        identifier = self.task_by_relative(first, "one/same_name")["dataset_id"]
        before = archive_snapshot(self.output)
        make_bundle(self.source / "two/same_name", lengths=(66,))
        resumed = self.run_batch(resume=True)
        self.assertTrue(resumed["passed"])
        self.assertEqual((resumed["skipped_count"], resumed["converted_count"]), (2, 1))
        self.assertEqual(self.task_by_relative(resumed, "one/same_name")["dataset_id"], identifier)
        after = archive_snapshot(self.output)
        for relative, evidence in before.items():
            self.assertEqual(after[relative], evidence)
        self.assertEqual(verify(self.output)["episode_count"], 3)

    def test_resume_configuration_changes_require_a_new_output_directory(self):
        self.make_tree()
        self.run_batch()
        before = content_snapshot(self.output)
        with self.assertRaises(ValueError):
            self.run_batch(resume=True, horizon=24)
        self.assertEqual(content_snapshot(self.output), before)

    def test_resume_does_not_accept_a_different_source_root(self):
        self.make_tree()
        self.run_batch()
        other = self.base / "different_source"
        make_bundle(other / "other_task", lengths=(49,))
        before = content_snapshot(self.output)
        with self.assertRaises(ValueError):
            batch_convert(other, self.output, resume=True)
        self.assertEqual(content_snapshot(self.output), before)

    def test_unmapped_numeric_episode_is_reported_instead_of_silently_omitted(self):
        self.make_tree()
        make_bundle(self.source / "alpha", lengths=(49, 66))
        mapping_path = self.source / "alpha/robotwin_episode_mapping.json"
        mapping = read_json(mapping_path)
        mapping["entries"] = mapping["entries"][:1]
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
        report = self.run_batch()
        self.assertFalse(report["passed"])
        self.assertEqual((report["completed_count"], report["failed_count"]), (2, 1))
        self.assertEqual(self.task_by_relative(report, "alpha")["status"], "failed")
        self.assert_no_ready_aggregate()

    def test_resume_refuses_unmanaged_existing_directories(self):
        self.make_tree()
        self.output.mkdir()
        (self.output / "keep.txt").write_text("unrelated user data", encoding="utf-8")
        before = content_snapshot(self.output)
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_batch(resume=True)
        self.assertEqual(content_snapshot(self.output), before)

    def test_existing_batch_requires_explicit_resume(self):
        self.make_tree()
        self.run_batch()
        before = content_snapshot(self.output)
        with self.assertRaises((ValueError, FileExistsError)):
            self.run_batch()
        self.assertEqual(content_snapshot(self.output), before)

    def test_resume_never_silently_skips_a_corrupt_completed_archive(self):
        self.make_tree()
        first = self.run_batch()
        target = self.output / first["tasks"][0]["dataset_id"] / "episodes/episode1.h5"
        target.write_bytes(target.read_bytes()[:80])
        resumed = self.run_batch(resume=True)
        self.assertLessEqual(resumed["skipped_count"], 2)
        if resumed["passed"]:
            self.assertEqual(resumed["converted_count"], 1)
            self.assertTrue(verify(self.output)["passed"])
        else:
            self.assertGreaterEqual(resumed["failed_count"], 1)
            self.assert_no_ready_aggregate()

    def test_cli_defaults_recurse_and_process_all_episodes(self):
        self.make_tree()
        make_bundle(self.source / "group/beta", lengths=(66, 49))
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["batch", "--source-root", str(self.source), "--output-root", str(self.output)])
        self.assertEqual(result, 0)
        report = read_json(self.output / "batch_report.json")
        self.assertEqual(report["total_bundles"], 3)
        self.assertEqual(report["episode_count"], 4)
        self.assertEqual(report["window_count"], 9)

    def test_cli_returns_two_when_a_task_failed_but_other_tasks_completed(self):
        self.make_tree()
        self.damage_beta_image()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = main(["batch", "--source-root", str(self.source), "--output-root", str(self.output)])
        self.assertEqual(result, 2)
        report = read_json(self.output / "batch_report.json")
        self.assertEqual((report["completed_count"], report["failed_count"]), (2, 1))
        self.assert_no_ready_aggregate()

    def test_recursive_inspect_plans_all_nested_tasks_without_creating_output(self):
        self.make_tree()
        standard_output = io.StringIO()
        with redirect_stdout(standard_output), redirect_stderr(io.StringIO()):
            result = main(["inspect", "--source-root", str(self.source), "--recursive"])
        self.assertEqual(result, 0)
        report = json.loads(standard_output.getvalue())
        self.assertEqual((report["episode_count"], report["frame_count"], report["window_count"]), (3, 180, 8))
        self.assertFalse(report["image_decode_verified"])
        self.assertFalse(self.output.exists())

    def test_recursive_inspect_empty_source_returns_error_without_success_report(self):
        self.source.mkdir()
        standard_output, standard_error = io.StringIO(), io.StringIO()
        with redirect_stdout(standard_output), redirect_stderr(standard_error):
            result = main(["inspect", "--source-root", str(self.source), "--recursive"])
        self.assertEqual(result, 2)
        self.assertEqual(standard_output.getvalue(), "")
        self.assertIn("ERROR:", standard_error.getvalue())
        self.assertIn("No converted", standard_error.getvalue())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
