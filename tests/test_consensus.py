import unittest

from config import Config, ConsensusConfig
from recognition.consensus import EvidenceStore, build_records, compute_consensus
from records import OCRObservation, PlateStatus

CFG = ConsensusConfig()


def ob(frame, text, conf=0.9, engine="easyocr", variant="color", det=0.9, raw=None):
    return OCRObservation(frame, frame / 25, engine, variant, raw or text, text, conf, det)


class ConsensusTests(unittest.TestCase):
    def test_spec_example_confirms_majority_and_keeps_minority(self):
        obs = [ob(100, "MH12AB1234"), ob(101, "MH12AB1234"), ob(102, "MH12A81234"), ob(103, "MH12AB1234")]
        r = compute_consensus(obs, CFG)
        self.assertEqual(r.status, PlateStatus.CONFIRMED)
        self.assertEqual(r.text, "MH12AB1234")
        self.assertEqual({c.text for c in r.candidates}, {"MH12AB1234", "MH12A81234"})
        self.assertTrue(r.conflicting)
        self.assertEqual(r.observations_total, 4)

    def test_even_split_is_uncertain_and_no_final_text(self):
        obs = [ob(f, "KA01AB1234" if f % 2 else "KA01A81234") for f in range(10)]
        r = compute_consensus(obs, CFG, final=True)
        self.assertEqual(r.status, PlateStatus.UNCERTAIN)
        self.assertIsNone(r.text)
        self.assertIsNotNone(r.best_guess)

    def test_not_finalised_prematurely(self):
        obs = [ob(1, "MH12AB1234")]
        self.assertEqual(compute_consensus(obs, CFG).status, PlateStatus.PENDING)
        final = compute_consensus(obs, CFG, final=True)
        self.assertEqual(final.status, PlateStatus.UNCERTAIN)

    def test_confirms_exactly_at_threshold(self):
        obs = [ob(i, "MH12AB1234") for i in range(CFG.min_support_frames)]
        self.assertEqual(compute_consensus(obs, CFG).status, PlateStatus.CONFIRMED)

    def test_low_confidence_reads_preserved_but_do_not_vote(self):
        obs = [ob(i, "MH12AB1234", conf=0.1) for i in range(5)]
        r = compute_consensus(obs, CFG, final=True)
        self.assertEqual(r.status, PlateStatus.NO_READING)
        self.assertEqual((r.observations_total, r.observations_eligible), (5, 0))

    def test_wrong_length_excluded(self):
        obs = [ob(i, "AB1") for i in range(5)] + [ob(i, "A" * 20) for i in range(5)]
        self.assertEqual(compute_consensus(obs, CFG, final=True).status, PlateStatus.NO_READING)

    def test_many_variants_in_one_frame_count_once(self):
        obs = [ob(7, "MH12AB1234", variant=v) for v in ("color", "clahe", "otsu", "adaptive")]
        r = compute_consensus(obs, CFG)
        self.assertEqual(r.support_frames, 1)
        self.assertEqual(r.status, PlateStatus.PENDING)

    def test_two_engines_agreement_recorded(self):
        obs = [ob(i, "MH12AB1234", engine=e) for i in range(3) for e in ("easyocr", "paddleocr")]
        r = compute_consensus(obs, CFG)
        self.assertTrue(r.engine_agreement)
        self.assertEqual(r.engines_supporting, ("easyocr", "paddleocr"))

    def test_engines_disagreeing_stay_visible(self):
        obs = [ob(i, "MH12AB1234", engine="easyocr") for i in range(3)]
        obs += [ob(i, "MH12A81234", engine="paddleocr") for i in range(3)]
        r = compute_consensus(obs, CFG, final=True)
        self.assertEqual(r.status, PlateStatus.UNCERTAIN)
        self.assertIsNone(r.text)
        self.assertEqual(len(r.candidates), 2)

    def test_no_character_guessing_ever_changes_text(self):
        obs = [ob(i, "MH12AB1234") for i in range(3)] + [ob(9, "MH12A81234")]
        result = compute_consensus(obs, CFG, final=True)
        self.assertEqual(result.status, PlateStatus.CONFIRMED)
        self.assertEqual(result.text, "MH12AB1234")  # B/8 strings stay separate; no majority substitution
        self.assertEqual({c.text for c in result.candidates}, {"MH12AB1234", "MH12A81234"})

    def test_confusable_character_disagreements_never_confirm(self):
        cases = [("KA03NU7762", "KAO3NU7762"), ("MH00SB1111", "MH005B1111"),
                 ("MH00BB1111", "MH008B1111"), ("MH00IB1111", "MH001B1111"),
                 ("MH00ZB1111", "MH002B1111"), ("MH00GB1111", "MH006B1111")]
        for winner, rival in cases:
            with self.subTest(winner=winner, rival=rival):
                obs = [ob(i, winner) for i in range(4)] + [ob(10, rival)]
                result = compute_consensus(obs, CFG, final=True)
                self.assertEqual(result.status, PlateStatus.CONFIRMED)
                self.assertEqual(result.text, winner)
                self.assertEqual(result.best_guess, winner)

    def test_brand_only_text_cannot_be_confirmed(self):
        result = compute_consensus([ob(i, "SRFA") for i in range(5)], CFG, final=True)
        self.assertEqual(result.status, PlateStatus.UNCERTAIN)
        self.assertIsNone(result.text)
        self.assertIn("letter-and-digit", result.reason)

    def test_deterministic_tie_break(self):
        obs = [ob(i, "AAAA1111") for i in range(3)] + [ob(i + 10, "BBBB2222") for i in range(3)]
        a = compute_consensus(obs, CFG, final=True)
        b = compute_consensus(list(reversed(obs)), CFG, final=True)
        self.assertEqual([c.text for c in a.candidates], [c.text for c in b.candidates])

    def test_empty(self):
        self.assertEqual(compute_consensus([], CFG).status, PlateStatus.NO_READING)

    def test_stricter_config_changes_outcome(self):
        strict = ConsensusConfig(min_support_frames=5)
        obs = [ob(i, "MH12AB1234") for i in range(3)]
        self.assertEqual(compute_consensus(obs, strict, final=True).status, PlateStatus.UNCERTAIN)


class StoreLifecycleTests(unittest.TestCase):
    def test_ocr_stops_after_confirmed_and_caps(self):
        cfg = Config()
        cfg.ocr.stop_ocr_after_confirmed_frames = 4
        store = EvidenceStore(cfg)
        inst = store.observe_detection(("vehicle", 1), 1, 0, 0.0, 0.9, None, 0.0)
        for f in range(10):
            if not store.should_run_ocr(inst):
                break
            store.add_observations(inst, f, [ob(f, "MH12AB1234")])
        self.assertEqual(len(inst.ocr_frames), 4)
        self.assertEqual(inst.consensus.status, PlateStatus.CONFIRMED)

    def test_records_use_observation_evidence(self):
        store = EvidenceStore(Config())
        inst = store.observe_detection(("vehicle", 5), 5, 10, 0.4, 0.8, None, 0.0)
        for f in range(10, 14):
            store.add_observations(inst, f, [ob(f, "MH12AB1234", raw="MH 12 AB 1234")])
        store.finalize()
        rec = build_records(store, Config())[0]
        self.assertEqual((rec.final_text, rec.status, rec.vehicle_track_id), ("MH12AB1234", "confirmed", 5))
        self.assertEqual(rec.observations[0]["raw_text"], "MH 12 AB 1234")
        self.assertTrue(rec.format_valid_indian)


if __name__ == "__main__":
    unittest.main()
