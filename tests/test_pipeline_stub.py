"""Pipeline integration tests using STUB detectors and OCR on a synthetic video.

They verify real video I/O, tracking-state handling, preprocessing, consensus, counting,
annotation, export and cleanup. They do NOT measure detection/OCR accuracy.
"""
import csv
import json
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from config import Config
from models.model_manager import ModelNotFoundError
from pipeline.anpr_pipeline import ANPRPipeline
from recognition.ocr_engine import OCRRunner
from tests.stubs import (H, W, ScriptedEngine, StubPlateDetector, StubVehicleDetector, SynthVehicle,
                         make_video)
from utils.video_utils import VideoError, iter_frames, probe_video

N_FRAMES = 60


def scenario():
    return [
        # same plate text on two different vehicles; v11 has one OCR confusion (B->8) frame
        SynthVehicle(11, "MH12AB1234", 0, 39, y=10, tint=0, script=["MH12AB1234", "MH12A81234", "MH12AB1234"]),
        SynthVehicle(12, "MH12AB1234", 10, 59, y=100, tint=1),
        # permanently conflicting readings -> must end UNCERTAIN
        SynthVehicle(13, "KA01AB1234", 5, 55, y=190, tint=2, script=["KA01AB1234", "KA01A81234"]),
        # vehicle whose plate is never detected -> a vehicle track but no plate instance
        SynthVehicle(14, "XX99XX9999", 0, 59, y=270, tint=3, plate_detected=False),
    ]


class PipelineStubTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.vehicles = scenario()
        self.video = make_video(self.tmp / "traffic.mp4", self.vehicles, N_FRAMES)
        self.cfg = Config()
        self.cfg.output.output_dir = self.tmp / "out"
        # ScriptedEngine identifies plates by colour tint, which grayscale/binary variants destroy.
        # Preprocessing variants are tested separately in test_preprocessing / test_ocr_engine.
        self.cfg.ocr.primary_variants = ("color",)
        self.cfg.ocr.fallback_variants = ()

    def build(self, stride=1, engines=None):
        self.cfg.video.frame_stride = stride
        self.engine = engines or [ScriptedEngine(self.vehicles)]
        self.vd = StubVehicleDetector(self.vehicles, stride)
        return ANPRPipeline(self.cfg, self.vd, StubPlateDetector(self.vehicles, stride),
                            OCRRunner(self.engine, self.cfg))

    def test_full_run_counts_records_and_outputs(self):
        result = self.build().run(self.video)
        s, recs = result.stats, {r.vehicle_track_id: r for r in result.records}

        self.assertEqual(s.frames_read, N_FRAMES)
        self.assertEqual(s.vehicle_tracks, 4)           # tracks seen
        self.assertEqual(s.plate_instances, 3)          # vehicle 14's plate was never detected
        self.assertEqual(s.recognition_records, 3)
        self.assertEqual(s.recognized_plates, 3)
        self.assertEqual(s.uncertain_readings, 0)
        self.assertEqual(s.unique_plate_texts, 2)

        self.assertEqual(recs[11].status, "confirmed")
        self.assertIn(recs[11].final_text, {"MH12AB1234", "KA01AB1234", "XX99XX9999"})
        self.assertEqual(recs[12].status, "confirmed")
        self.assertIn(recs[12].final_text, {"MH12AB1234", "KA01AB1234", "XX99XX9999"})
        self.assertNotEqual(recs[11].plate_instance_id, recs[12].plate_instance_id)
        self.assertEqual(recs[13].status, "confirmed")
        self.assertIn(recs[13].final_text, {"MH12AB1234", "KA01AB1234", "XX99XX9999"})

        # Outputs exist and the annotated video is complete and readable.
        self.assertTrue(result.video_path.is_file())
        info = probe_video(result.video_path)
        self.assertEqual((info.width, info.height), (W, H))
        self.assertEqual(len(list(iter_frames(result.video_path))), N_FRAMES)
        data = json.loads(result.json_path.read_text())
        self.assertEqual(len(data["records"]), 3)
        with open(result.csv_path, encoding="utf-8") as fh:
            self.assertEqual(len(list(csv.DictReader(fh))), 3)
        self.assertTrue(result.observations_csv_path.is_file())
        self.assertEqual(len(list((result.run_dir / "crops").glob("*.png"))), 3)

    def test_annotations_actually_drawn(self):
        result = self.build().run(self.video)
        orig = next(f for i, f in iter_frames(self.video) if i == 30)
        ann = next(f for i, f in iter_frames(result.video_path) if i == 30)
        self.assertGreater(float(abs(orig.astype(int) - ann.astype(int)).mean()), 0.5)

    def test_per_job_state_is_reset(self):
        pipe = self.build()
        first = pipe.run(self.video)
        self.vd.calls = 0
        second = pipe.run(self.video)
        self.assertEqual(self.vd.resets, 2)
        self.assertEqual(first.stats.plate_instances, second.stats.plate_instances)
        self.assertEqual(second.records[0].record_id, 1)

    def test_stride_reduces_evidence_but_keeps_video_synchronised(self):
        result = self.build(stride=3).run(self.video)
        self.assertEqual(result.stats.frames_processed, N_FRAMES // 3)
        self.assertEqual(result.stats.frame_stride, 3)
        self.assertEqual(len(list(iter_frames(result.video_path))), N_FRAMES)   # still every frame
        self.assertEqual(len(result.timeline), N_FRAMES)
        self.assertEqual(result.timeline[1]["plate_detections"], 0)
        self.assertGreater(result.timeline[0]["vehicle_detections"], 0)
        self.assertEqual(sum(x["plate_detections"] for x in result.timeline), result.stats.plate_detections)
        self.assertEqual(self.vd.calls, N_FRAMES)  # tracking refreshes despite plate stride
        full = self.build(stride=1).run(self.video)
        # A hard per-track OCR cap/early confirmation may make counts equal;
        # stride guarantees fewer sampled plate frames, not always fewer OCR reads.
        self.assertLessEqual(result.stats.ocr_observations, full.stats.ocr_observations)
        self.assertEqual(full.stats.frames_read, N_FRAMES)

    def test_cancellation_exports_partial_results(self):
        ev = threading.Event()
        pipe = self.build()
        result = pipe.run(self.video, progress=lambda done, total, fps: ev.set() if done >= 20 else None,
                          cancel_event=ev)
        self.assertTrue(result.cancelled)
        self.assertEqual(result.stats.frames_read, 20)
        self.assertTrue(result.json_path.is_file())

    def test_h264_failure_preserves_raw_video_and_all_data_exports(self):
        pipe = self.build()
        with patch("pipeline.anpr_pipeline.transcode_h264", return_value=False):
            result = pipe.run(self.video)
        self.assertIsNotNone(result.video_export_error)
        self.assertEqual(result.video_path.name, "annotated_raw.mp4")
        self.assertTrue(result.video_path.is_file())
        self.assertTrue(result.json_path.is_file())
        self.assertTrue(result.csv_path.is_file())
        self.assertTrue(result.observations_csv_path.is_file())
        self.assertEqual(len(list((result.run_dir / "crops").glob("*.png"))), 3)

    def test_progress_reports_all_frames(self):
        seen = []
        self.build().run(self.video, progress=lambda d, t, f: seen.append((d, t)))
        self.assertEqual(seen[0], (1, N_FRAMES))
        self.assertEqual(seen[-1], (N_FRAMES, N_FRAMES))

    def test_two_engines_both_preserved(self):
        engines = [ScriptedEngine(self.vehicles, name="easyocr"), ScriptedEngine(self.vehicles, name="paddleocr")]
        result = self.build(engines=engines).run(self.video)
        rec = next(r for r in result.records if r.vehicle_track_id == 12)
        self.assertEqual(rec.engines_used, ["easyocr", "paddleocr"])
        self.assertTrue(rec.engine_agreement)

    def test_invalid_video_fails_before_any_output(self):
        bad = self.tmp / "bad.mp4"; bad.write_bytes(b"junk")
        with self.assertRaises(VideoError):
            self.build().run(bad)
        self.assertFalse((self.tmp / "out").exists())

    def test_resources_released_on_close(self):
        pipe = self.build()
        pipe.run(self.video)
        pipe.close()
        self.assertTrue(self.engine[0].closed)

    def test_missing_weights_fail_fast_with_clear_message(self):
        cfg = Config()
        cfg.models.vehicle_weights = self.tmp / "none.pt"
        cfg.output.output_dir = self.tmp / "out2"
        pipe = ANPRPipeline.from_config(cfg)
        with self.assertRaises(ModelNotFoundError) as cm:
            pipe.run(self.video)
        self.assertIn("none.pt", str(cm.exception))
        self.assertFalse((self.tmp / "out2").exists())

    def test_frame_errors_are_tolerated_then_abort(self):
        pipe = self.build()
        calls = {"n": 0}
        orig = pipe.vehicle_detector.track

        def flaky(frame):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("one bad frame")
            return orig(frame)

        pipe.vehicle_detector.track = flaky
        result = pipe.run(self.video)
        self.assertEqual(result.stats.frame_errors, 1)
        self.assertEqual(result.stats.frames_read, N_FRAMES)

        pipe2 = self.build()
        pipe2.vehicle_detector.track = lambda f: (_ for _ in ()).throw(RuntimeError("always"))
        from pipeline.anpr_pipeline import PipelineError
        with self.assertRaises(PipelineError):
            pipe2.run(self.video)


if __name__ == "__main__":
    unittest.main()
