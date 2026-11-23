import tempfile
import unittest
from pathlib import Path

from config import ModelConfig
from models.model_manager import ModelError, ModelManager, ModelNotFoundError, validate_weights_path


class WeightsValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_missing_file_gives_actionable_error(self):
        with self.assertRaises(ModelNotFoundError) as cm:
            validate_weights_path(self.tmp / "nope.pt", "plate")
        self.assertIn("nope.pt", str(cm.exception))
        self.assertIn("COCO model cannot detect plates", str(cm.exception))

    def test_wrong_extension_and_empty_file(self):
        bad = self.tmp / "m.onnx"; bad.write_bytes(b"x")
        with self.assertRaises(ModelError):
            validate_weights_path(bad, "vehicle")
        empty = self.tmp / "e.pt"; empty.write_bytes(b"")
        with self.assertRaises(ModelError):
            validate_weights_path(empty, "vehicle")

    def test_directory_rejected(self):
        d = self.tmp / "dir.pt"; d.mkdir()
        with self.assertRaises(ModelError):
            validate_weights_path(d, "vehicle")

    def test_valid_file_accepted(self):
        ok = self.tmp / "ok.pt"; ok.write_bytes(b"weights")
        self.assertEqual(validate_weights_path(ok, "vehicle"), ok.resolve())


class ModelManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        (self.tmp / "v.pt").write_bytes(b"v")
        (self.tmp / "p.pt").write_bytes(b"p")
        self.cfg = ModelConfig(vehicle_weights=self.tmp / "v.pt", plate_weights=self.tmp / "p.pt")
        self.loads = []

    def loader(self, path):
        self.loads.append(path.name)
        return object()

    def test_lazy_and_cached(self):
        mm = ModelManager(self.cfg, self.loader)
        self.assertEqual(self.loads, [])                 # nothing loaded at construction
        a = mm.get("vehicle"); b = mm.get("vehicle")
        self.assertIs(a, b)
        self.assertEqual(self.loads, ["v.pt"])           # loaded exactly once
        self.assertFalse(mm.is_loaded("plate"))

    def test_release_clears_cache(self):
        mm = ModelManager(self.cfg, self.loader)
        mm.get("plate"); mm.release()
        self.assertFalse(mm.is_loaded("plate"))
        mm.get("plate")
        self.assertEqual(self.loads, ["p.pt", "p.pt"])

    def test_loader_never_called_for_missing_weights(self):
        cfg = ModelConfig(vehicle_weights=self.tmp / "missing.pt", plate_weights=self.tmp / "p.pt")
        mm = ModelManager(cfg, self.loader)
        with self.assertRaises(ModelNotFoundError):
            mm.get("vehicle")
        self.assertEqual(self.loads, [])                 # i.e. no download attempt can happen

    def test_loader_failure_wrapped(self):
        def broken(path): raise RuntimeError("corrupt")
        with self.assertRaises(ModelError) as cm:
            ModelManager(self.cfg, broken).get("vehicle")
        self.assertIn("corrupt", str(cm.exception))

    def test_unknown_role(self):
        with self.assertRaises(ValueError):
            ModelManager(self.cfg, self.loader).get("tank")


if __name__ == "__main__":
    unittest.main()
