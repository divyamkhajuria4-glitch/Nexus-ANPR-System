import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from config import Config
from recognition.consensus import EvidenceStore, build_records
from records import OCRObservation, PipelineStats
from utils import exporters


class ExporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.cfg = Config()
        store = EvidenceStore(self.cfg)
        crop = np.full((30, 90, 3), 200, np.uint8)
        for key, vt, text in ((("vehicle", 1), 1, "MH12AB1234"), (("vehicle", 2), 2, "MH12AB1234")):
            inst = None
            for f in range(5):
                inst = store.observe_detection(key, vt, f, f / 25, 0.9, crop, 1.0)
                store.add_observations(inst, f, [OCRObservation(f, f / 25, "easyocr", "color", text + "\n", text, 0.9, 0.9)])
        store.finalize()
        self.store = store
        self.crops = exporters.save_crops(self.tmp / "crops", store.instances)
        self.records = build_records(store, self.cfg, self.crops)

    def test_json_contains_required_fields(self):
        p = exporters.export_json(self.tmp / "r.json", self.records, PipelineStats(frames_total=5),
                                  {"path": "x.mp4"}, self.cfg.to_dict())
        data = json.loads(p.read_text())
        self.assertEqual(len(data["records"]), 2)
        rec = data["records"][0]
        for key in ("record_id", "vehicle_track_id", "plate_instance_id", "observations", "final_text",
                    "confidence", "engines_used", "first_frame", "first_timestamp_s", "detection_confidence_mean",
                    "status", "uncertainty_reason", "support_frames"):
            self.assertIn(key, rec)
        self.assertEqual(rec["observations"][0]["raw_text"], "MH12AB1234\n")   # raw preserved exactly
        self.assertIn("notes", data)
        self.assertIsInstance(data["config"]["models"]["vehicle_weights"], str)

    def test_csv_has_one_row_per_record_even_with_identical_text(self):
        p = exporters.export_csv(self.tmp / "r.csv", self.records)
        with open(p, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["final_text"] for r in rows}, {"MH12AB1234"})
        self.assertEqual({r["vehicle_track_id"] for r in rows}, {"1", "2"})

    def test_observations_csv_keeps_every_read(self):
        p = exporters.export_observations_csv(self.tmp / "o.csv", self.records)
        with open(p, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 10)
        self.assertIn("\\n", rows[0]["raw_text"])
        self.assertEqual(rows[0]["engine"], "easyocr")

    def test_crops_saved_and_linked(self):
        self.assertEqual(len(self.crops), 2)
        for rec in self.records:
            self.assertTrue((self.tmp / rec.crop_file).is_file())

    def test_empty_records_export(self):
        p = exporters.export_csv(self.tmp / "e.csv", [])
        self.assertTrue(p.is_file())
        exporters.export_json(self.tmp / "e.json", [], PipelineStats(), {}, {})


if __name__ == "__main__":
    unittest.main()
