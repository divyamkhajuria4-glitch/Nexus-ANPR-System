import unittest

from config import Config
from recognition.consensus import EvidenceStore, build_records
from records import OCRObservation


def feed(store, key, vtid, text, frames):
    inst = None
    for f in frames:
        inst = store.observe_detection(key, vtid, f, f / 25, 0.9, None, 0.0)
        store.add_observations(inst, f, [OCRObservation(f, f / 25, "easyocr", "color", text, text, 0.9, 0.9)])
    return inst


class CountingTests(unittest.TestCase):
    def test_identical_text_on_two_vehicles_stays_separate(self):
        store = EvidenceStore(Config())
        feed(store, ("vehicle", 1), 1, "MH12AB1234", range(0, 20))
        feed(store, ("vehicle", 2), 2, "MH12AB1234", range(5, 25))
        store.finalize()
        records = build_records(store, Config())
        c = store.counts()
        self.assertEqual(len(records), 2)                               # two recognition records
        self.assertEqual(c["plate_instances"], 2)                       # two plate instances
        self.assertEqual(c["recognized"], 2)
        self.assertEqual(c["unique_texts"], 1)                          # one unique text
        self.assertEqual({r.vehicle_track_id for r in records}, {1, 2})
        self.assertEqual(len({r.plate_instance_id for r in records}), 2)

    def test_frames_do_not_create_new_instances(self):
        store = EvidenceStore(Config())
        feed(store, ("vehicle", 1), 1, "MH12AB1234", range(100))
        self.assertEqual(store.counts()["plate_instances"], 1)
        self.assertGreater(store.counts()["ocr_observations"], 0)

    def test_orphan_and_vehicle_keys_do_not_collide(self):
        store = EvidenceStore(Config())
        feed(store, ("vehicle", 1), 1, "AAAA1111", range(4))
        feed(store, ("orphan", 1), None, "BBBB2222", range(4))
        recs = build_records(store, Config())
        self.assertEqual({r.association for r in recs}, {"vehicle", "unassociated"})
        self.assertEqual(store.counts()["plate_instances"], 2)

    def test_counter_buckets_sum_to_instances(self):
        store = EvidenceStore(Config())
        feed(store, ("vehicle", 1), 1, "AAAA1111", range(5))
        store.observe_detection(("vehicle", 2), 2, 0, 0.0, 0.9, None, 0.0)   # plate seen, never read
        store.finalize()
        c = store.counts()
        self.assertEqual(c["recognized"] + c["pending"] + c["uncertain"] + c["no_reading"], c["plate_instances"])
        self.assertEqual(c["no_reading"], 1)


if __name__ == "__main__":
    unittest.main()
