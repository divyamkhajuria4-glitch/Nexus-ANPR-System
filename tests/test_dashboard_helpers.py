import unittest
import tempfile
import json
from pathlib import Path

from dashboard.job_utils import (effective_frame_total, matching_manual_review,
                                 render_data_table, render_timeline_chart)


class FakeStreamlit:
    def __init__(self, error=None):
        self.error = error
        self.warnings = []
        self.chart_calls = 0
        self.data_calls = 0
        self.json_calls = []
        self.image_calls = []

    def line_chart(self, *_args, **_kwargs):
        self.chart_calls += 1
        if self.error:
            raise self.error

    def warning(self, message):
        self.warnings.append(message)

    def dataframe(self, *_args, **_kwargs):
        self.data_calls += 1
        if self.error:
            raise self.error

    def json(self, value):
        self.json_calls.append(value)

    def image(self, value, **kwargs):
        self.image_calls.append((value, kwargs))


class DashboardHelperTests(unittest.TestCase):
    def test_progress_total_respects_frame_cap(self):
        self.assertEqual(effective_frame_total(1632, 250), 250)

    def test_progress_total_does_not_exceed_short_source(self):
        self.assertEqual(effective_frame_total(100, 250), 100)

    def test_progress_total_uses_full_source_when_uncapped(self):
        self.assertEqual(effective_frame_total(1632, None), 1632)

    def test_incompatible_arrow_skips_streamlit_chart_and_uses_raster_fallback(self):
        st = FakeStreamlit(ImportError("pyarrow/NumPy ABI mismatch"))
        class Series:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        data = {"vehicle_detections": Series([0, 2, 1]), "plate_detections": Series([0, 1, 0])}
        self.assertTrue(render_timeline_chart(st, data))
        self.assertEqual(st.chart_calls, 0)
        self.assertEqual(len(st.image_calls), 1)
        self.assertEqual(st.warnings, [])

    def test_chart_success_is_reported(self):
        st = FakeStreamlit()
        class Series:
            def __init__(self, values): self.values = values
            def tolist(self): return self.values
        data = {"vehicle_detections": Series([1, 3]), "plate_detections": Series([0, 1])}
        self.assertTrue(render_timeline_chart(st, data))
        self.assertEqual(st.warnings, [])

    def test_data_table_falls_back_to_json_when_arrow_cannot_load(self):
        st = FakeStreamlit(ImportError("pyarrow/NumPy ABI mismatch"))
        rows = [{"text": "KAO3NU7762"}]
        self.assertFalse(render_data_table(st, rows))
        self.assertEqual(st.json_calls, [rows])

    def test_manual_review_applies_only_to_exact_crop_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            crop = Path(folder) / "crop.png"
            review = Path(folder) / "manual.json"
            crop.write_bytes(b"exact crop")
            review.write_text(json.dumps({"crop_sha256": "a8fe2b573e0eeefc01032a61b33fe1a061bf5ea6f9ddc7b8105f0e8b2a119d5f", "manual_status": "uncertain"}))
            # The expected digest is checked against the crop bytes, never text or plate ID.
            self.assertIsNone(matching_manual_review(crop, review))

    def test_manual_review_matches_sha256_bound_crop(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            crop = Path(folder) / "crop.png"
            review = Path(folder) / "manual.json"
            crop.write_bytes(b"reviewed crop")
            review.write_text(json.dumps({"crop_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(), "manual_status": "uncertain"}))
            self.assertEqual(matching_manual_review(crop, review)["manual_status"], "uncertain")

    def test_manual_review_selects_matching_crop_from_review_collection(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            crop = Path(folder) / "crop.png"
            review = Path(folder) / "manual.json"
            crop.write_bytes(b"release-run crop")
            review.write_text(json.dumps({"reviews": [
                {"crop_sha256": "not-this-crop", "manual_status": "confirmed"},
                {"crop_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                 "manual_status": "uncertain", "ambiguous_character": "O/0"}
            ]}))
            matched = matching_manual_review(crop, review)
            self.assertEqual(matched["manual_status"], "uncertain")
            self.assertEqual(matched["ambiguous_character"], "O/0")


if __name__ == "__main__":
    unittest.main()
