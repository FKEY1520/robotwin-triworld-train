"""CPU-only integrity tests; native history audit uses an optional model checkout."""
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_lrz_cache import extract_model, sha256
from triworld_train.windows import history_indices


def make_controller(path, files=None, *, checksum=None):
    files = files or {"TriViewWan/triviwan/train.py": b"# original train\n",
                      "TriViewWan/triviwan/history.py": b"# original history\n"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    content = buffer.getvalue()
    digest = checksum or hashlib.sha256(content).hexdigest()
    # A controller side effect must never execute during extraction.
    path.write_text("raise RuntimeError('controller must not execute')\n"
                    f"PAYLOAD_B64 = {base64.b64encode(content).decode()!r}\n"
                    f"PAYLOAD_SHA256 = {digest!r}\n", encoding="utf-8")
    return digest


class ModelExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_literal_extract_reuse_and_receipt_without_controller_execution(self):
        source, target = self.root / "deploy.py", self.root / "native"
        digest = make_controller(source)
        model = extract_model(source, target)
        self.assertEqual(model, target / "TriViewWan")
        self.assertEqual((model / "triviwan/train.py").read_bytes(), b"# original train\n")
        self.assertEqual(extract_model(source, target), model)
        receipt = json.loads((target / "model_source.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["payload_sha256"], digest)
        self.assertEqual(receipt["controller_sha256"], sha256(source))

    def test_checksum_mismatch_writes_nothing(self):
        source, target = self.root / "deploy.py", self.root / "native"
        make_controller(source, checksum="0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            extract_model(source, target)
        self.assertFalse(target.exists())

    def test_parent_traversal_writes_nothing(self):
        source, target = self.root / "deploy.py", self.root / "native"
        make_controller(source, {"TriViewWan/../../escape.py": b"bad"})
        with self.assertRaises(ValueError):
            extract_model(source, target)
        self.assertFalse(target.exists())
        self.assertFalse((self.root / "escape.py").exists())

    def test_absolute_and_windows_paths_rejected(self):
        for name in ("/TriViewWan/test.py", "C:/TriViewWan/test.py", "TriViewWan\\test.py"):
            with self.subTest(name=name):
                source = self.root / "deploy.py"
                make_controller(source, {name: b"bad"})
                with self.assertRaises(ValueError):
                    extract_model(source, self.root / "native")

    def test_existing_unknown_directory_preserved(self):
        source, target = self.root / "deploy.py", self.root / "native"
        make_controller(source)
        target.mkdir()
        marker = target / "user.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            extract_model(source, target)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_modified_model_refused_without_rewrite(self):
        source, target = self.root / "deploy.py", self.root / "native"
        make_controller(source)
        model = extract_model(source, target)
        changed = model / "triviwan/train.py"
        changed.write_text("# user edits", encoding="utf-8")
        with self.assertRaises(ValueError):
            extract_model(source, target)
        self.assertEqual(changed.read_text(encoding="utf-8"), "# user edits")

    def test_different_payload_refused_without_rewrite(self):
        source, target = self.root / "deploy.py", self.root / "native"
        make_controller(source)
        model = extract_model(source, target)
        before = (model / "triviwan/train.py").read_bytes()
        make_controller(source, {"TriViewWan/triviwan/train.py": b"other",
                                 "TriViewWan/triviwan/history.py": b"other"})
        with self.assertRaises(FileExistsError):
            extract_model(source, target)
        self.assertEqual((model / "triviwan/train.py").read_bytes(), before)


@unittest.skipUnless(os.environ.get("TRIVIEW_MODEL_ROOT"), "Set TRIVIEW_MODEL_ROOT to audit the original model checkout")
class NativeHistoryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ["TRIVIEW_MODEL_ROOT"]) / "triviwan/history.py"
        module = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        names = {"_positive_integer", "_history_size", "history_indices"}
        functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
        if {node.name for node in functions} != names:
            raise AssertionError("Native history module has changed its helper interface")
        namespace = {}
        # Only pure indexing functions are compiled. No torch import or model execution.
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        cls.native = staticmethod(namespace["history_indices"])

    def test_synthetic_boundaries_match_native(self):
        for frames in (1, 5, 9, 13):
            for start in range(1, 1001):
                self.assertEqual(history_indices(start, frames), self.native(start, frames))

    def test_invalid_history_matches_native_rejection(self):
        for start, frames in ((0, 9), (-1, 9), (True, 9), (1, 0), (1, 2), (1, 8), (1, True)):
            with self.subTest(start=start, frames=frames):
                with self.assertRaises(ValueError):
                    history_indices(start, frames)
                with self.assertRaises(ValueError):
                    self.native(start, frames)

    def test_real_windows_match_native_when_dataset_given(self):
        root = os.environ.get("TRIWORLD_DATASET_ROOT")
        if not root:
            self.skipTest("Set TRIWORLD_DATASET_ROOT to check real converted windows")
        root = Path(root)
        manifest = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        count = 0
        with (root / manifest["windows_path"]).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.assertEqual(row["history_indices"], self.native(row["future_start"], manifest["settings"]["history_frames"]))
                    count += 1
        self.assertEqual(count, manifest["window_count"])
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
