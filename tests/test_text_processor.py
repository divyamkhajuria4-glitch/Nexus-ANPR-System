import unittest

from recognition.text_processor import (TextFragment, is_indian_format, merge_fragments,
                                        normalize_text, process_text)


class NormalizationTests(unittest.TestCase):
    def test_whitespace_punctuation_case(self):
        self.assertEqual(normalize_text(" mh-12 ab.1234\n"), "MH12AB1234")

    def test_raw_preserved_separately(self):
        p = process_text("mh 12\nab 1234")
        self.assertEqual(p.raw, "mh 12\nab 1234")
        self.assertEqual(p.normalized, "MH12AB1234")

    def test_empty_none_like(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")


class FormatTests(unittest.TestCase):
    def test_indian_formats(self):
        for ok in ("MH12AB1234", "DL1CAB1234", "KA05MN9999", "22BH1234AA", "UP321234"):
            self.assertTrue(is_indian_format(ok), ok)

    def test_non_indian_not_rejected_by_processing(self):
        for other in ("ABC1234", "7ABC123", "B1234XYZ"):
            p = process_text(other)
            self.assertEqual(p.normalized, other)       # still produced, merely flagged
        self.assertFalse(is_indian_format("ABC1234"))


class MergeTests(unittest.TestCase):
    def test_two_line_plate_reading_order(self):
        frags = [TextFragment("1234", 0.8, 80, 70, 20), TextFragment("MH12", 0.9, 30, 20, 20),
                 TextFragment("AB", 0.6, 90, 22, 20)]
        raw, conf = merge_fragments(frags)
        self.assertEqual(raw, "MH12 AB\n1234")
        self.assertEqual(normalize_text(raw), "MH12AB1234")
        self.assertAlmostEqual(conf, (0.8 * 4 + 0.9 * 4 + 0.6 * 2) / 10)

    def test_single_line_left_to_right(self):
        frags = [TextFragment("1234", 0.9, 90, 10, 20), TextFragment("MH12", 0.9, 20, 11, 20)]
        self.assertEqual(merge_fragments(frags)[0], "MH12 1234")

    def test_empty(self):
        self.assertEqual(merge_fragments([]), ("", 0.0))
        self.assertEqual(merge_fragments([TextFragment("  ", 0.9, 0, 0, 5)]), ("", 0.0))


class CorrectionTests(unittest.TestCase):
    def test_only_standard_plate_slots_are_corrected(self):
        cases = (
            ("MH12AB1234", "MH12AB1234"),
            ("MH12A81234", "MH12AB1234"),
            ("KAO3NU7762", "KA03NU7762"),
            ("MHOZFJ9790", "MH02FJ9790"),
            ("HHOZFJ9790", "HH02FJ9790"),
            ("NOKA03NU7762", "KA03NU7762"),
            ("NO KA03NU7762", "KA03NU7762"),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize_text(source), expected)


if __name__ == "__main__":
    unittest.main()
