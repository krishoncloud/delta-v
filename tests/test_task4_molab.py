"""Offline persistence checks; no HF token or network requests are used."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


NOTEBOOK = Path(__file__).resolve().parents[1] / "research" / "molab_task4.py"


class FakeHub:
    def __init__(self):
        self.files = {}
        self.uploads = 0
        self.corrupt_remote = False
        self.fail_upload = False

    def dataset_info(self, repo):
        return SimpleNamespace(sha="commit1")

    def upload_folder(self, **kwargs):
        if self.fail_upload:
            raise ConnectionError("offline")
        self.uploads += 1
        root = Path(kwargs["folder_path"])
        for path in root.rglob("*"):
            if path.is_file():
                self.files[f"{kwargs['path_in_repo']}/{path.relative_to(root)}"] = path.read_bytes()
        return SimpleNamespace(oid="commit2")

    def get_paths_info(self, repo, paths, **kwargs):
        result = []
        for name in paths:
            if name not in self.files:
                continue
            blob = self.files[name]
            blob_id = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
            lfs = {"sha256": hashlib.sha256(blob).hexdigest()} if name.endswith(".bin") else None
            if self.corrupt_remote:
                blob_id = "bad"
                if lfs:
                    lfs["sha256"] = "bad"
            result.append(SimpleNamespace(path=name, size=len(blob), lfs=lfs, blob_id=blob_id))
        return result


class TestTask4Save(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("task4_molab", NOTEBOOK)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, cls.definitions = module.app.run()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "run1"
        self.root.mkdir()
        (self.root / "report.md").write_text("measured result\n")
        (self.root / "model.bin").write_bytes(b"model-state")
        manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.iterdir()}
        (self.root / "artifact_manifest.json").write_text(json.dumps(manifest))
        self.hub = FakeHub()

    def save(self):
        return self.definitions["save_run_to_hf"](self.root, "test-only", api=self.hub)

    def test_success_and_idempotent_retry_verify_git_and_lfs_hashes(self):
        self.assertEqual(self.save()["verified_files"], 3)
        self.assertTrue(self.root.with_suffix(".hf-receipt.json").exists())
        self.save()
        self.assertEqual(self.hub.uploads, 1)

    def test_corrupt_local_artifact_is_not_uploaded(self):
        (self.root / "model.bin").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.save()
        self.assertEqual(self.hub.uploads, 0)

    def test_untracked_file_is_not_uploaded(self):
        (self.root / "unexpected.txt").write_text("must not publish")
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.save()
        self.assertEqual(self.hub.uploads, 0)

    def test_remote_corruption_never_reports_success(self):
        self.hub.corrupt_remote = True
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.save()
        self.assertFalse(self.root.with_suffix(".hf-receipt.json").exists())

    def test_partial_existing_run_is_not_overwritten(self):
        self.hub.files["research/task4-outcomes/v1/runs/run1/report.md"] = b"partial"
        with self.assertRaises(RuntimeError):
            self.save()
        self.assertEqual(self.hub.uploads, 0)

    def test_failed_upload_preserves_local_results_for_retry(self):
        self.hub.fail_upload = True
        with self.assertRaises(ConnectionError):
            self.save()
        self.assertTrue((self.root / "model.bin").exists())
        self.hub.fail_upload = False
        self.assertEqual(self.save()["verified_files"], 3)

    def test_missing_token_fails_before_network(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            self.definitions["preflight_hf"]("")


if __name__ == "__main__":
    unittest.main()
