"""REAL end-to-end test: actual traffic video + actual local weights + actual OCR engines.

Skipped automatically unless everything is present. Place a recorded traffic video at
input/sample.mp4 (or set NEXUS_E2E_VIDEO) and the two weight files in models/weights/.
It asserts structural correctness only - it cannot assert accuracy without ground truth.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from config import PROJECT_ROOT, Config

VIDEO = Path(os.environ.get("NEXUS_E2E_VIDEO", PROJECT_ROOT / "input" / "sample.mp4"))


def _missing() -> str:
    cfg = Config()
    for mod in ("torch", "ultralytics", "easyocr"):
        if importlib.util.find_spec(mod) is None:
            return f"{mod} not installed"
    for p in (cfg.models.vehicle_weights, cfg.models.plate_weights, VIDEO):
        if not Path(p).is_file():
            return f"missing file: {p}"
    return ""


@unittest.skipIf(_missing(), _missing() or "ready")
class RealEndToEndTests(unittest.TestCase):
    def test_real_video_produces_outputs(self):
        from pipeline.anpr_pipeline import ANPRPipeline
        from utils.video_utils import iter_frames, probe_video

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        cfg = Config()
        cfg.output.output_dir = tmp
        cfg.video.max_frames = int(os.environ.get("NEXUS_E2E_FRAMES", "150"))
        pipe = ANPRPipeline.from_config(cfg)
        try:
            result = pipe.run(VIDEO)
        finally:
            pipe.close()
        self.assertGreater(result.stats.frames_read, 0)
        self.assertTrue(result.video_path.is_file())
        self.assertEqual(len(list(iter_frames(result.video_path))), result.stats.frames_read)
        data = json.loads(result.json_path.read_text())
        self.assertEqual(len(data["records"]), result.stats.recognition_records)
        self.assertEqual(probe_video(result.video_path).fps > 0, True)
        print("\nREAL E2E summary:", result.stats)


if __name__ == "__main__":
    unittest.main()
