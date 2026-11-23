import unittest

import numpy as np

from detection.association import associate_plates
from detection.plate_extractor import clip_bbox, extract_plate_crop
from detection.tracker import IoUTracker
from models.plate_detector import parse_plate_results
from models.vehicle_detector import parse_vehicle_results
from records import BBox, PlateDet, VehicleDet
from tests.stubs import FakeBoxes, FakeResult


def veh(x1, y1, x2, y2, tid=1):
    return VehicleDet(BBox(x1, y1, x2, y2), 0.9, 2, "car", tid)


class ParseTests(unittest.TestCase):
    def test_vehicle_output_format_with_ids(self):
        boxes = FakeBoxes(np.array([[10, 20, 110, 90], [200, 50, 300, 150]], float),
                          np.array([0.9, 0.5]), np.array([2, 7]), np.array([5, 6]))
        out = parse_vehicle_results([FakeResult(boxes)])
        self.assertEqual([(v.track_id, v.class_name) for v in out], [(5, "car"), (6, "truck")])
        self.assertEqual(out[0].bbox, BBox(10, 20, 110, 90))
        self.assertAlmostEqual(out[1].confidence, 0.5)

    def test_unconfirmed_tracks_have_no_id(self):
        boxes = FakeBoxes(np.array([[0, 0, 50, 50]], float), np.array([0.8]), np.array([3]), None)
        out = parse_vehicle_results([FakeResult(boxes)])
        self.assertIsNone(out[0].track_id)
        self.assertEqual(out[0].class_name, "motorcycle")

    def test_empty_and_degenerate(self):
        self.assertEqual(parse_vehicle_results([]), [])
        self.assertEqual(parse_vehicle_results([FakeResult(None)]), [])
        zero = FakeBoxes(np.array([[5, 5, 5, 5]], float), np.array([0.9]), np.array([2]))
        self.assertEqual(parse_vehicle_results([FakeResult(zero)]), [])

    def test_plate_class_filter(self):
        boxes = FakeBoxes(np.array([[0, 0, 40, 15], [50, 0, 90, 15]], float),
                          np.array([0.7, 0.8]), np.array([0, 1]))
        self.assertEqual(len(parse_plate_results([FakeResult(boxes)])), 2)
        self.assertEqual(len(parse_plate_results([FakeResult(boxes)], class_ids=(1,))), 1)


class CropTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)

    def test_clip_to_image_bounds(self):
        c = clip_bbox(BBox(-10, -5, 250, 120), 200, 100)
        self.assertEqual((c.x1, c.y1, c.x2, c.y2), (0, 0, 200, 100))

    def test_clip_rejects_outside_nan_and_empty(self):
        self.assertIsNone(clip_bbox(BBox(300, 10, 350, 40), 200, 100))
        self.assertIsNone(clip_bbox(BBox(float("nan"), 0, 10, 10), 200, 100))
        self.assertIsNone(clip_bbox(BBox(10, 10, 10, 40), 200, 100))

    def test_valid_crop_is_independent_copy(self):
        crop = extract_plate_crop(self.frame, BBox(20, 30, 80, 50), 20, 8)
        self.assertEqual(crop.shape[:2], (20, 60))
        crop[:] = 0
        self.assertNotEqual(int(self.frame[30:50, 20:80].sum()), 0)

    def test_small_and_invalid_crops_rejected(self):
        self.assertIsNone(extract_plate_crop(self.frame, BBox(20, 30, 30, 50), 20, 8))   # too narrow
        self.assertIsNone(extract_plate_crop(self.frame, BBox(20, 30, 80, 34), 20, 8))   # too short
        self.assertIsNone(extract_plate_crop(np.zeros((0, 0, 3), np.uint8), BBox(0, 0, 50, 20), 10, 5))
        self.assertIsNone(extract_plate_crop(None, BBox(0, 0, 50, 20), 10, 5))

    def test_edge_crop_with_padding_is_clipped_not_crashing(self):
        crop = extract_plate_crop(self.frame, BBox(170, 80, 215, 110), 10, 5, pad_ratio=0.1)
        self.assertIsNotNone(crop)
        self.assertLessEqual(crop.shape[1], 200)


class AssociationTests(unittest.TestCase):
    def test_plate_inside_vehicle(self):
        a = associate_plates([PlateDet(BBox(40, 60, 90, 80), 0.9)], [veh(0, 0, 200, 100, 7)])
        self.assertEqual(a[0].vehicle.track_id, 7)

    def test_partially_visible_plate_still_matches(self):
        plate = PlateDet(BBox(150, 60, 230, 80), 0.9)          # 62.5% inside
        self.assertEqual(associate_plates([plate], [veh(0, 0, 200, 100, 3)])[0].vehicle.track_id, 3)

    def test_plate_without_vehicle_is_kept_unassociated(self):
        a = associate_plates([PlateDet(BBox(400, 300, 450, 320), 0.9)], [veh(0, 0, 200, 100)])
        self.assertEqual(len(a), 1)
        self.assertIsNone(a[0].vehicle)

    def test_nested_vehicles_prefer_smaller_box(self):
        plate = PlateDet(BBox(40, 40, 60, 50), 0.9)
        a = associate_plates([plate], [veh(0, 0, 300, 200, 1), veh(20, 20, 120, 80, 2)])
        self.assertEqual(a[0].vehicle.track_id, 2)

    def test_one_plate_per_vehicle_keeps_best_and_demotes_other(self):
        v = veh(0, 0, 200, 100, 9)
        strong, weak = PlateDet(BBox(10, 60, 60, 80), 0.9), PlateDet(BBox(100, 60, 150, 80), 0.4)
        a = associate_plates([weak, strong], [v], one_per_vehicle=True)
        self.assertEqual(len(a), 2)                             # nothing discarded
        self.assertIsNone(a[0].vehicle)
        self.assertEqual(a[1].vehicle.track_id, 9)

    def test_multiple_plates_associate_by_default(self):
        v = veh(0, 0, 200, 100, 9)
        plates = [PlateDet(BBox(10, 60, 60, 80), 0.9), PlateDet(BBox(100, 60, 150, 80), 0.8)]
        associations = associate_plates(plates, [v])
        self.assertEqual([item.vehicle.track_id for item in associations], [9, 9])


class IoUTrackerTests(unittest.TestCase):
    def test_stable_ids_and_new_ids(self):
        t = IoUTracker(0.3, max_age_frames=3)
        a = t.update([BBox(0, 0, 50, 20)], 0)
        b = t.update([BBox(3, 1, 53, 21)], 1)
        self.assertEqual(a, b)
        c = t.update([BBox(3, 1, 53, 21), BBox(300, 200, 350, 220)], 2)
        self.assertEqual(c[0], a[0])
        self.assertNotEqual(c[1], a[0])

    def test_expiry(self):
        t = IoUTracker(0.3, max_age_frames=2)
        first = t.update([BBox(0, 0, 50, 20)], 0)[0]
        t.update([], 10)
        self.assertNotEqual(t.update([BBox(0, 0, 50, 20)], 11)[0], first)


if __name__ == "__main__":
    unittest.main()
