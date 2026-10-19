"""Recursive bundle discovery must stay bounded and identify tasks unambiguously."""
from pathlib import Path
import tempfile
import unittest

from triworld_train.discovery import discover_tree


def mark_bundle(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "robotwin_episode_mapping.json").write_text('{"entries": []}', encoding="utf-8")
    return path.resolve()


class RecursiveDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="triworld-discovery-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()

    def test_discovers_all_nested_bundles_and_stops_at_each_marker(self):
        expected = {mark_bundle(self.root / "alpha"), mark_bundle(self.root / "group" / "beta"),
                    mark_bundle(self.root / "group" / "deeper" / "gamma")}
        mark_bundle(self.root / "alpha" / "source_data" / "not_another_task")
        mark_bundle(self.root / "alpha" / "lossless" / "also_not_another_task")
        found = discover_tree(self.root)
        self.assertEqual({bundle.path for bundle in found}, expected)
        self.assertEqual(len({bundle.dataset_id.casefold() for bundle in found}), 3)

    def test_source_root_itself_can_be_one_complete_bundle(self):
        mark_bundle(self.root)
        mark_bundle(self.root / "source_data" / "nested_marker")
        found = discover_tree(self.root)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path, self.root.resolve())
        self.assertEqual(found[0].dataset_id, self.root.name)

    def test_skips_hidden_environment_and_generated_dataset_trees(self):
        expected = mark_bundle(self.root / "real_task")
        for folder in (".git", ".venv", "__pycache__", ".hidden"):
            mark_bundle(self.root / folder / "ignore_me")
        generated = self.root / "already_converted_training"
        generated.mkdir()
        (generated / "dataset.json").write_text('{"kind": "triview_window_training"}', encoding="utf-8")
        mark_bundle(generated / "must_not_be_reinterpreted_as_source")
        self.assertEqual([item.path for item in discover_tree(self.root)], [expected])

    def test_explicit_exclusions_prune_whole_subtrees(self):
        expected = mark_bundle(self.root / "keep")
        ignored = self.root / "archive"
        mark_bundle(ignored / "deeper" / "ignore")
        found = discover_tree(self.root, exclude_paths=(ignored,))
        self.assertEqual([item.path for item in found], [expected])

    def test_incomplete_managed_batch_tree_is_not_rediscovered(self):
        expected = mark_bundle(self.root / "real_task")
        generated = self.root / "partial_training"
        generated.mkdir()
        (generated / "batch_state.json").write_text('{"kind":"triview_batch_state"}', encoding="utf-8")
        mark_bundle(generated / "copied_input_metadata")
        self.assertEqual([item.path for item in discover_tree(self.root)], [expected])

    def test_large_unicode_generated_manifest_is_recognized_from_its_prefix(self):
        expected = mark_bundle(self.root / "real_task")
        generated = self.root / "large_training"
        generated.mkdir()
        prefix = '{"kind":"triview_window_training","padding":"'
        if (65536 - len(prefix.encode("utf-8"))) % 3 == 0:
            prefix += "x"
        # The bounded 64 KiB read ends inside a UTF-8 character, after kind.
        (generated / "dataset.json").write_text(prefix + "中" * 30000 + '"}', encoding="utf-8")
        mark_bundle(generated / "copied_input_metadata")
        self.assertEqual([item.path for item in discover_tree(self.root)], [expected])

    def test_nested_kind_string_does_not_hide_a_real_project_subtree(self):
        project = self.root / "project"
        project.mkdir()
        (project / "dataset.json").write_text(
            '{"metadata":{"kind":"triview_window_training"},"description":"project metadata"}', encoding="utf-8")
        expected = mark_bundle(project / "real_source")
        self.assertEqual([item.path for item in discover_tree(self.root)], [expected])

    def test_duplicate_basenames_have_distinct_stable_ids(self):
        first = mark_bundle(self.root / "set_one" / "same_name")
        initial = discover_tree(self.root)[0]
        mark_bundle(self.root / "set_two" / "same_name")
        found = discover_tree(self.root)
        self.assertEqual(len({item.dataset_id for item in found}), 2)
        after = next(item for item in found if item.path == first)
        self.assertEqual(initial.dataset_id, after.dataset_id)
        selected = discover_tree(self.root, tasks=("set_one/same_name",))
        self.assertEqual([item.dataset_id for item in selected], [initial.dataset_id])
        with self.assertRaisesRegex(ValueError, "ambiguous|Ambiguous|multiple"):
            discover_tree(self.root, tasks=("same_name",))

    def test_unique_simple_name_and_exact_relative_name_select_same_bundle(self):
        expected = mark_bundle(self.root / "group" / "unique")
        mark_bundle(self.root / "other")
        simple = discover_tree(self.root, tasks=("unique",))
        exact = discover_tree(self.root, tasks=("group/unique",))
        self.assertEqual(simple, exact)
        self.assertEqual(simple[0].path, expected)
        with self.assertRaises(ValueError):
            discover_tree(self.root, tasks=("does_not_exist",))

    def test_empty_source_returns_no_bundles_and_missing_source_is_rejected(self):
        self.assertEqual(discover_tree(self.root), [])
        with self.assertRaises((ValueError, FileNotFoundError)):
            discover_tree(self.root / "missing")

    def test_symlink_loop_is_not_followed(self):
        expected = mark_bundle(self.root / "real")
        link = self.root / "loop"
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Directory symlinks unavailable: {exc}")
        self.assertEqual([item.path for item in discover_tree(self.root)], [expected])


if __name__ == "__main__":
    unittest.main()
