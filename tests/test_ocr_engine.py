import sys
import unittest
from unittest import mock

import numpy as np

from config import Config
from recognition.ocr_engine import (EasyOCREngine, OCREngine, OCRRunner, OCRUnavailableError,
                                    PaddleOCREngine, parse_paddle_output)
from recognition.text_processor import TextFragment


class FakeEngine(OCREngine):
    def __init__(self, name, text="MH12AB1234", conf=0.9, fail=False, fail_load=False):
        super().__init__()
        self.name, self.text, self.conf, self.fail, self.fail_load = name, text, conf, fail, fail_load
        self.seen_shapes = []

    def _load(self):
        if self.fail_load:
            raise OCRUnavailableError("no models")

    def _read(self, image_bgr):
        self.seen_shapes.append(image_bgr.shape)
        if self.fail:
            raise RuntimeError("engine crashed")
        return [TextFragment(self.text, self.conf, 10, 10, 20)]


CROP = np.full((30, 100, 3), 200, np.uint8)


class RunnerTests(unittest.TestCase):
    def test_observation_preserves_engine_variant_frame_and_raw(self):
        r = OCRRunner([FakeEngine("easyocr", text="mh 12 ab 1234")], Config())
        obs = r.read_plate(CROP, 42, 1.68, 0.77)
        self.assertEqual({o.variant for o in obs}, {"best_clear_frame"})
        o = obs[0]
        self.assertEqual((o.frame_index, o.engine, o.det_confidence), (42, "easyocr", 0.77))
        self.assertEqual((o.raw_text, o.normalized_text, o.ocr_confidence), ("mh 12 ab 1234", "MH12AB1234", 0.9))
        self.assertAlmostEqual(o.timestamp_s, 1.68)

    def test_reads_only_one_clear_frame_variant(self):
        strong = FakeEngine("easyocr", conf=0.95)
        weak = FakeEngine("easyocr", conf=0.2)
        strong_obs = OCRRunner([strong], Config()).read_plate(CROP, 0, 0, 0.9)
        weak_obs = OCRRunner([weak], Config()).read_plate(CROP, 0, 0, 0.9)
        self.assertEqual(len(strong_obs), 1)
        self.assertEqual(len(weak_obs), 1)
        self.assertEqual({o.variant for o in strong_obs}, {"best_clear_frame"})

    def test_engine_failures_are_independent(self):
        good, bad = FakeEngine("easyocr"), FakeEngine("paddleocr", fail=True)
        r = OCRRunner([bad, good], Config())
        obs = r.read_plate(CROP, 1, 0.04, 0.9)
        self.assertTrue(obs)
        self.assertEqual({o.engine for o in obs}, {"easyocr"})
        self.assertGreater(r.error_counts["paddleocr"], 0)

    def test_blank_reads_produce_no_observation(self):
        r = OCRRunner([FakeEngine("easyocr", text="  -- ")], Config())
        self.assertEqual(r.read_plate(CROP, 0, 0, 0.9), [])

    def test_warmup_disables_failed_engine_but_continues(self):
        r = OCRRunner([FakeEngine("paddleocr", fail_load=True), FakeEngine("easyocr")], Config())
        r.warmup()
        self.assertEqual([e.name for e in r.active_engines], ["easyocr"])

    def test_warmup_raises_when_nothing_available(self):
        with self.assertRaises(OCRUnavailableError):
            OCRRunner([FakeEngine("easyocr", fail_load=True)], Config()).warmup()

    def test_grayscale_variants_are_sent_to_engines_as_3_channel(self):
        eng = FakeEngine("easyocr", conf=0.1)
        OCRRunner([eng], Config()).read_plate(CROP, 0, 0, 0.9)
        self.assertTrue(all(len(s) == 3 and s[2] == 3 for s in eng.seen_shapes))


class PaddleParseTests(unittest.TestCase):
    def test_v2_format(self):
        raw = [[[[[0, 0], [60, 0], [60, 20], [0, 20]], ("MH12", 0.97)],
                [[[70, 0], [130, 0], [130, 20], [70, 20]], ("AB1234", 0.91)]]]
        frags = parse_paddle_output(raw)
        self.assertEqual([f.text for f in frags], ["MH12", "AB1234"])
        self.assertAlmostEqual(frags[0].confidence, 0.97)

    def test_v3_format(self):
        raw = [{"rec_texts": ["MH12AB1234"], "rec_scores": [0.88], "rec_polys": [[[0, 0], [90, 0], [90, 20], [0, 20]]]}]
        frags = parse_paddle_output(raw)
        self.assertEqual((frags[0].text, round(frags[0].confidence, 2)), ("MH12AB1234", 0.88))

    def test_none_and_malformed(self):
        self.assertEqual(parse_paddle_output(None), [])
        self.assertEqual(parse_paddle_output([None]), [])
        self.assertEqual(parse_paddle_output([[None, "junk"]]), [])


class EngineAvailabilityTests(unittest.TestCase):
    def test_missing_easyocr_package_is_clear_error(self):
        with mock.patch.dict(sys.modules, {"easyocr": None}):
            with self.assertRaises(OCRUnavailableError):
                EasyOCREngine(Config()).ensure_loaded()

    def test_missing_paddle_package_is_clear_error(self):
        with mock.patch.dict(sys.modules, {"paddleocr": None}):
            with self.assertRaises(OCRUnavailableError):
                PaddleOCREngine(Config()).ensure_loaded()

    def test_easyocr_download_disabled_by_default(self):
        captured = {}

        class FakeReader:
            def __init__(self, langs, **kw): captured.update(kw)

        fake = mock.MagicMock(); fake.Reader = FakeReader
        with mock.patch.dict(sys.modules, {"easyocr": fake}):
            EasyOCREngine(Config()).ensure_loaded()
        self.assertIs(captured["download_enabled"], False)


if __name__ == "__main__":
    unittest.main()
