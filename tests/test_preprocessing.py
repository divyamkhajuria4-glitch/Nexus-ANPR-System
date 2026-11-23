import unittest

import cv2
import numpy as np

from recognition.preprocessing import (ALL_VARIANTS, adaptive_threshold, assess_quality, build_variants,
                                       correct_perspective, crop_quality_score, otsu_threshold,
                                       resize_for_ocr)


def plate_img(h=30, w=100):
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.putText(img, "AB12", (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    return img


class PreprocessingTests(unittest.TestCase):
    def test_all_variants_generated_and_valid(self):
        crop = plate_img()
        out = build_variants(crop, [v for v in ALL_VARIANTS if v != "perspective"])
        self.assertEqual(set(out), set(ALL_VARIANTS) - {"perspective"})
        for name, img in out.items():
            self.assertEqual(img.dtype, np.uint8, name)
            self.assertGreaterEqual(img.shape[0], 64, name)     # upscaled to min height

    def test_original_crop_not_modified(self):
        crop = plate_img()
        before = crop.copy()
        build_variants(crop, ["color", "clahe", "otsu", "adaptive", "sharpen"])
        self.assertTrue(np.array_equal(crop, before))

    def test_grayscale_input_supported(self):
        gray = cv2.cvtColor(plate_img(), cv2.COLOR_BGR2GRAY)
        self.assertIn("clahe", build_variants(gray, ["clahe"]))

    def test_unknown_variant_and_empty_crop(self):
        with self.assertRaises(ValueError):
            build_variants(plate_img(), ["magic"])
        self.assertEqual(build_variants(np.zeros((0, 0, 3), np.uint8), ["color"]), {})

    def test_resize_rules(self):
        self.assertEqual(resize_for_ocr(np.zeros((20, 60, 3), np.uint8), 64, 640).shape[:2], (64, 192))
        self.assertEqual(resize_for_ocr(np.zeros((100, 300, 3), np.uint8), 64, 640).shape[:2], (200, 600))
        self.assertLessEqual(resize_for_ocr(np.zeros((10, 600, 3), np.uint8), 64, 640).shape[1], 640)

    def test_thresholds_are_binary_with_light_majority(self):
        g = cv2.cvtColor(plate_img(), cv2.COLOR_BGR2GRAY)
        g = resize_for_ocr(g)
        for fn in (otsu_threshold, adaptive_threshold):
            b = fn(g)
            self.assertTrue(set(np.unique(b)) <= {0, 255})
            self.assertGreaterEqual(float((b > 127).mean()), 0.5)

    def test_inverted_plate_polarity_normalised(self):
        g = cv2.bitwise_not(resize_for_ocr(cv2.cvtColor(plate_img(), cv2.COLOR_BGR2GRAY)))
        self.assertGreaterEqual(float((otsu_threshold(g) > 127).mean()), 0.5)

    def test_perspective_corrects_visible_quadrilateral(self):
        canvas = np.full((200, 300, 3), 40, np.uint8)
        quad = np.array([[30, 40], [270, 20], [280, 170], [20, 180]], np.int32)
        cv2.fillPoly(canvas, [quad], (240, 240, 240))
        warped = correct_perspective(canvas, min_area_ratio=0.3)
        self.assertIsNotNone(warped)
        self.assertGreater(warped.shape[1], 200)

    def test_perspective_returns_none_without_quadrilateral(self):
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 255, (60, 120, 3), dtype=np.uint8)
        self.assertIsNone(correct_perspective(np.full((60, 120, 3), 128, np.uint8)))
        correct_perspective(noise)  # must not raise

    def test_quality_metrics(self):
        sharp, blurry = plate_img(), cv2.GaussianBlur(plate_img(), (9, 9), 3)
        self.assertGreater(assess_quality(sharp).sharpness, assess_quality(blurry).sharpness)
        self.assertGreater(crop_quality_score(sharp, 0.9), crop_quality_score(blurry, 0.9))

    def test_low_resolution_crop_handled(self):
        tiny = np.full((6, 20, 3), 200, np.uint8)
        out = build_variants(tiny, ["color", "clahe", "otsu", "adaptive"])
        self.assertEqual(len(out), 4)


if __name__ == "__main__":
    unittest.main()
