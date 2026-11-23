import json
import tempfile
import unittest
from pathlib import Path

from config import Config


class ConfigTests(unittest.TestCase):
    def test_json_roundtrip_with_overrides(self):
        d = Config().to_dict()
        d["video"]["frame_stride"] = 3
        d["ocr"]["engines"] = ["easyocr", "paddleocr"]
        d["models"]["vehicle_weights"] = "/tmp/x.pt"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(d, fh)
        cfg = Config.load(Path(fh.name))
        self.assertEqual(cfg.video.frame_stride, 3)
        self.assertEqual(cfg.ocr.engines, ("easyocr", "paddleocr"))
        self.assertEqual(cfg.models.vehicle_weights, Path("/tmp/x.pt"))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({"video": {"frame_strdie": 2}})

    def test_invalid_values_rejected(self):
        for patch in ({"video": {"frame_stride": 0}}, {"ocr": {"engines": ["invalid_engine"]}},
                      {"detection": {"plate_conf": 1.5}}, {"consensus": {"min_text_length": 20}}):
            with self.assertRaises(ValueError):
                Config.from_dict(patch)

    def test_default_config_is_valid(self):
        cfg = Config()
        cfg.validate()
        self.assertEqual(cfg.detection.vehicle_conf, 0.30)
        self.assertEqual(cfg.detection.plate_conf, 0.35)
        self.assertEqual(cfg.detection.plate_imgsz, 1280)

    def test_example_matches_portable_production_defaults(self):
        example = Path(__file__).resolve().parents[1] / "config.example.json"
        cfg = Config.load(example)
        self.assertEqual(cfg.detection.plate_conf, Config().detection.plate_conf)
        self.assertEqual(cfg.detection.vehicle_conf, Config().detection.vehicle_conf)
        self.assertEqual(cfg.detection.plate_imgsz, Config().detection.plate_imgsz)
        self.assertEqual(cfg.models.vehicle_weights, Path(__file__).resolve().parents[1] / "models/weights/yolov8n.pt")
        self.assertEqual(cfg.models.plate_weights, Path(__file__).resolve().parents[1] / "models/weights/license_plate_detector.pt")


if __name__ == "__main__":
    unittest.main()
