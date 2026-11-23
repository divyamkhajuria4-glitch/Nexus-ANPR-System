"""Multi-frame consensus and the evidence store.

Design rules
------------
* Observations are never edited or dropped; weak ones simply do not vote.
* Votes are for WHOLE normalised strings. There is no fuzzy merging or character
  substitution, so ``MH12AB1234`` and ``MH12A81234`` stay two competing candidates.
* Within a frame each distinct text votes once per frame (best OCR confidence across
  engines/variants), so running many variants cannot inflate support.
* A winner needs: enough distinct agreeing frames, a large enough share of the total
  weight, and a clear margin over the runner-up. Otherwise the result is UNCERTAIN
  and no "final text" is produced.
* Identity is per plate instance (vehicle track), never per text.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from config import Config, ConsensusConfig
from recognition.text_processor import is_indian_format
from records import (CandidateSummary, ConsensusResult, OCRObservation, PlateStatus,
                     RecognitionRecord)

PlateKey = Tuple[str, int]  # ("vehicle", track_id) or ("orphan", plate_track_id)

_CONFUSABLES = (frozenset("O0"), frozenset("S5"), frozenset("B8"),
                frozenset("I1"), frozenset("Z2"), frozenset("G6"))


def _has_plate_like_structure(text: str) -> bool:
    """Reject ordinary words/signage and require generic plate alphanumeric shape."""
    text = text.strip().upper()
    if len(text) < 4 or len(text) > 12:
        return False
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    return letters >= 1 and digits >= 2


def _has_confusable_disagreement(candidates: Sequence["CandidateSummary"]) -> bool:
    """Keep exact OCR character disagreements uncertain when rival also matches plate format."""
    if len(candidates) < 2:
        return False
    lead = candidates[0].text
    lead_fmt = is_indian_format(lead)
    for other in candidates[1:]:
        if len(lead) != len(other.text):
            continue
        if lead_fmt and not is_indian_format(other.text):
            continue
        for left, right in zip(lead, other.text):
            if left != right and any(left in pair and right in pair for pair in _CONFUSABLES):
                return True
    return False


def is_eligible(obs: OCRObservation, cfg: ConsensusConfig) -> bool:
    return (cfg.min_text_length <= len(obs.normalized_text) <= cfg.max_text_length
            and obs.ocr_confidence >= cfg.min_vote_confidence
            and _has_plate_like_structure(obs.normalized_text))


def compute_consensus(observations: Sequence[OCRObservation], cfg: ConsensusConfig,
                      final: bool = False) -> ConsensusResult:
    total = len(observations)
    valid_format_and_conf = [
        o for o in observations
        if (cfg.min_text_length <= len(o.normalized_text) <= cfg.max_text_length
            and o.ocr_confidence >= cfg.min_vote_confidence)
    ]
    if not valid_format_and_conf:
        reason = "no OCR output yet" if total == 0 else f"{total} OCR reads, none met confidence or length criteria"
        return ConsensusResult(PlateStatus.NO_READING, None, None, 0.0, 0, 0, total, 0, reason=reason)

    eligible = [o for o in valid_format_and_conf if _has_plate_like_structure(o.normalized_text)]
    if not eligible:
        reason = f"{total} OCR reads, text lacks required letter-and-digit plate format"
        return ConsensusResult(PlateStatus.UNCERTAIN, None, None, 0.0, 0, 0, total, 0, reason=reason)

    per_frame: Dict[Tuple[int, str], List[Any]] = {}
    for o in eligible:
        entry = per_frame.setdefault((o.frame_index, o.normalized_text), [0.0, 0.0, set()])
        entry[0] = max(entry[0], o.ocr_confidence)
        entry[1] = max(entry[1], o.det_confidence)
        entry[2].add(o.engine)
        entry[0] = max(entry[0], o.ocr_confidence)
        entry[1] = max(entry[1], o.det_confidence)
        entry[2].add(o.engine)

    grouped: Dict[str, Dict[str, Any]] = {}
    for (frame, text), (ocr_c, det_c, engines) in per_frame.items():
        g = grouped.setdefault(text, {"score": 0.0, "frames": set(), "engines": set(), "ocr": []})
        g["score"] += ocr_c * (det_c if det_c > 0 else 1.0)
        g["frames"].add(frame)
        g["engines"] |= engines
        g["ocr"].append(ocr_c)

    grand_total = sum(g["score"] for g in grouped.values()) or 1e-9
    ranked = sorted(grouped.items(), key=lambda kv: (-kv[1]["score"], -len(kv[1]["frames"]), kv[0]))
    candidates = [CandidateSummary(text=t, score=g["score"], support_frames=len(g["frames"]),
                                   share=g["score"] / grand_total,
                                   engines=tuple(sorted(g["engines"])),
                                   mean_ocr_confidence=float(statistics.fmean(g["ocr"])))
                  for t, g in ranked]
    top = candidates[0]
    runner_share = candidates[1].share if len(candidates) > 1 else 0.0
    margin = top.share - runner_share
    voting_frames = len({f for f, _ in per_frame})
    conflicting = len(candidates) > 1
    ambiguous_positions = sorted({i + 1 for candidate in candidates[1:]
                                  if len(candidate.text) == len(top.text)
                                  for i, (a, b) in enumerate(zip(top.text, candidate.text)) if a != b})
    confidence = float(top.share * top.mean_ocr_confidence)

    structural = is_indian_format(top.text)
    # Confirm only an exact Indian registration string that repeats across frames,
    # clears the configured OCR threshold, and wins decisively. Keep OCR guesses out
    # of the final reading when any gate fails.
    meets = (structural and top.support_frames >= cfg.min_support_frames
             and top.mean_ocr_confidence >= cfg.min_vote_confidence
             and top.share >= cfg.min_weight_share and margin >= cfg.min_margin_share
             and not _has_confusable_disagreement(candidates))
    if meets:
        status, text, reason = PlateStatus.CONFIRMED, top.text, ""
    elif voting_frames < cfg.min_support_frames and not final:
        status, text, reason = PlateStatus.PENDING, None, "collecting evidence"
    elif not structural:
        status, text, reason = PlateStatus.UNCERTAIN, None, \
            "leading OCR candidate does not match an Indian registration format"
    elif _has_confusable_disagreement(candidates):
        status, text, reason = PlateStatus.UNCERTAIN, None, \
            "confusable character disagreement across OCR candidates"
    elif conflicting and (top.share < cfg.min_weight_share or margin < cfg.min_margin_share):
        status, text, reason = PlateStatus.UNCERTAIN, None, "conflicting readings without a clear winner"
    else:
        status, text, reason = PlateStatus.UNCERTAIN, None, \
            f"insufficient evidence ({top.support_frames} agreeing frames, need {cfg.min_support_frames})"

    return ConsensusResult(
        status=status, text=text, best_guess=top.text, confidence=confidence,
        support_frames=top.support_frames, voting_frames=voting_frames,
        observations_total=total, observations_eligible=len(eligible),
        candidates=candidates, engines_supporting=top.engines,
        engine_agreement=len(top.engines) > 1, conflicting=conflicting,
        ambiguous_positions=ambiguous_positions, reason=reason)


@dataclass
class PlateInstance:
    plate_instance_id: int
    key: PlateKey
    vehicle_track_id: Optional[int]
    first_frame: int
    first_ts: float
    last_frame: int
    last_ts: float
    detection_count: int = 0
    det_conf_sum: float = 0.0
    det_conf_max: float = 0.0
    observations: List[OCRObservation] = field(default_factory=list)
    ocr_frames: Set[int] = field(default_factory=set)
    best_crop: Optional[np.ndarray] = None
    best_crop_score: float = -1.0
    best_frame: int = 0
    best_ts: float = 0.0
    consensus: Optional[ConsensusResult] = None
    crops_by_text: Dict[str, Tuple[np.ndarray, float, int, float]] = field(default_factory=dict)


class EvidenceStore:
    """Per-job state: one PlateInstance per vehicle track (or orphan plate track)."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._instances: Dict[PlateKey, PlateInstance] = {}
        self._next_id = 1

    @property
    def instances(self) -> List[PlateInstance]:
        return sorted(self._instances.values(), key=lambda i: i.plate_instance_id)

    def observe_detection(self, key: PlateKey, vehicle_track_id: Optional[int], frame_index: int,
                          timestamp_s: float, det_conf: float, crop: Optional[np.ndarray],
                          crop_score: float) -> PlateInstance:
        inst = self._instances.get(key)
        if inst is None:
            inst = PlateInstance(self._next_id, key, vehicle_track_id, frame_index, timestamp_s,
                                 frame_index, timestamp_s)
            self._instances[key] = inst
            self._next_id += 1
        inst.last_frame, inst.last_ts = frame_index, timestamp_s
        inst.detection_count += 1
        inst.det_conf_sum += det_conf
        inst.det_conf_max = max(inst.det_conf_max, det_conf)
        if crop is not None and crop_score > inst.best_crop_score:
            inst.best_crop, inst.best_crop_score = crop, crop_score
            inst.best_frame, inst.best_ts = frame_index, timestamp_s
        return inst

    def should_run_ocr(self, inst: PlateInstance) -> bool:
        o = self._cfg.ocr
        n = len(inst.ocr_frames)
        if n >= o.max_ocr_frames_per_track:
            return False
        confirmed = inst.consensus is not None and inst.consensus.status == PlateStatus.CONFIRMED
        if confirmed:
            return n < o.stop_ocr_after_confirmed_frames
        # Spread the OCR budget across the whole track. Using the first N
        # detections misses plates that become legible later in the video.
        target = max(3, min(o.max_ocr_frames_per_track, int(np.ceil(
            o.max_ocr_frames_per_track * (1.0 - np.exp(
                -inst.detection_count / max(o.max_ocr_frames_per_track * 4.0, 1.0)))))))
        return n < target

    def add_observations(self, inst: PlateInstance, frame_index: int,
                         observations: Sequence[OCRObservation],
                         crop: Optional[np.ndarray] = None,
                         crop_score: float = -1.0,
                         timestamp_s: float = 0.0) -> None:
        inst.ocr_frames.add(frame_index)
        inst.observations.extend(observations)
        if crop is not None and crop_score > 0:
            for obs in observations:
                text = obs.normalized_text
                if text:
                    prev = inst.crops_by_text.get(text)
                    if prev is None or crop_score > prev[1]:
                        inst.crops_by_text[text] = (crop, crop_score, frame_index, timestamp_s)
        inst.consensus = compute_consensus(inst.observations, self._cfg.consensus, final=False)

    def deduplicate(self) -> None:
        """Merge fragmented/unassociated tracks of a plate into a confirmed or primary instance."""
        by_text: Dict[str, List[PlateInstance]] = {}
        for inst in list(self._instances.values()):
            if not inst.consensus:
                inst.consensus = compute_consensus(inst.observations, self._cfg.consensus, final=True)
            text = inst.consensus.text or inst.consensus.best_guess
            if text and is_indian_format(text) and _has_plate_like_structure(text):
                by_text.setdefault(text, []).append(inst)

        for text, group in by_text.items():
            if len(group) <= 1:
                continue

            confirmed_insts = [i for i in group if i.consensus and i.consensus.status == PlateStatus.CONFIRMED]
            uncertain_insts = [i for i in group if not (i.consensus and i.consensus.status == PlateStatus.CONFIRMED)]

            if confirmed_insts and uncertain_insts:
                primary = confirmed_insts[0]
                for secondary in uncertain_insts:
                    seen_obs = {(o.frame_index, o.engine, o.variant, o.normalized_text) for o in primary.observations}
                    for obs in secondary.observations:
                        key_t = (obs.frame_index, obs.engine, obs.variant, obs.normalized_text)
                        if key_t not in seen_obs:
                            primary.observations.append(obs)
                            seen_obs.add(key_t)

                    primary.ocr_frames |= secondary.ocr_frames
                    primary.first_frame = min(primary.first_frame, secondary.first_frame)
                    primary.last_frame = max(primary.last_frame, secondary.last_frame)
                    primary.first_ts = min(primary.first_ts, secondary.first_ts)
                    primary.last_ts = max(primary.last_ts, secondary.last_ts)
                    primary.detection_count += secondary.detection_count
                    primary.det_conf_sum += secondary.det_conf_sum
                    primary.det_conf_max = max(primary.det_conf_max, secondary.det_conf_max)

                    for t, crop_info in secondary.crops_by_text.items():
                        if t not in primary.crops_by_text or crop_info[1] > primary.crops_by_text[t][1]:
                            primary.crops_by_text[t] = crop_info

                    self._instances.pop(secondary.key, None)

                primary.consensus = compute_consensus(primary.observations, self._cfg.consensus, final=True)
            elif not confirmed_insts and len(uncertain_insts) > 1:
                primary = uncertain_insts[0]
                for secondary in uncertain_insts[1:]:
                    seen_obs = {(o.frame_index, o.engine, o.variant, o.normalized_text) for o in primary.observations}
                    for obs in secondary.observations:
                        key_t = (obs.frame_index, obs.engine, obs.variant, obs.normalized_text)
                        if key_t not in seen_obs:
                            primary.observations.append(obs)
                            seen_obs.add(key_t)

                    primary.ocr_frames |= secondary.ocr_frames
                    primary.first_frame = min(primary.first_frame, secondary.first_frame)
                    primary.last_frame = max(primary.last_frame, secondary.last_frame)
                    primary.first_ts = min(primary.first_ts, secondary.first_ts)
                    primary.last_ts = max(primary.last_ts, secondary.last_ts)
                    primary.detection_count += secondary.detection_count
                    primary.det_conf_sum += secondary.det_conf_sum
                    primary.det_conf_max = max(primary.det_conf_max, secondary.det_conf_max)

                    for t, crop_info in secondary.crops_by_text.items():
                        if t not in primary.crops_by_text or crop_info[1] > primary.crops_by_text[t][1]:
                            primary.crops_by_text[t] = crop_info

                    self._instances.pop(secondary.key, None)

                primary.consensus = compute_consensus(primary.observations, self._cfg.consensus, final=True)

    def finalize_crops(self) -> None:
        """Ensure the best crop matches the winning text, preventing drift to unrelated vehicles."""
        for inst in self._instances.values():
            if inst.consensus:
                target = inst.consensus.text or inst.consensus.best_guess
                if target and target in inst.crops_by_text:
                    crop, score, frame, ts = inst.crops_by_text[target]
                    inst.best_crop = crop
                    inst.best_crop_score = score
                    inst.best_frame = frame
                    inst.best_ts = ts

    def finalize(self) -> None:
        """Called once at the end: pending readings become uncertain, duplicates merge, crops finalize."""
        for inst in self._instances.values():
            inst.consensus = compute_consensus(inst.observations, self._cfg.consensus, final=True)
        self.deduplicate()
        self.finalize_crops()

    def counts(self) -> Dict[str, int]:
        by = {s: 0 for s in PlateStatus}
        texts: Set[str] = set()
        n_obs = 0
        for inst in self._instances.values():
            n_obs += len(inst.observations)
            status = inst.consensus.status if inst.consensus else PlateStatus.NO_READING
            by[status] += 1
            if status == PlateStatus.CONFIRMED and inst.consensus and inst.consensus.text:
                texts.add(inst.consensus.text)
        return {"plate_instances": len(self._instances), "ocr_observations": n_obs,
                "recognized": by[PlateStatus.CONFIRMED], "pending": by[PlateStatus.PENDING],
                "uncertain": by[PlateStatus.UNCERTAIN], "no_reading": by[PlateStatus.NO_READING],
                "unique_texts": len(texts)}


def _obs_dict(o: OCRObservation, cfg: ConsensusConfig) -> Dict[str, Any]:
    return {"frame": o.frame_index, "timestamp_s": round(o.timestamp_s, 3), "engine": o.engine,
            "variant": o.variant, "raw_text": o.raw_text, "normalized_text": o.normalized_text,
            "ocr_confidence": round(o.ocr_confidence, 4), "detection_confidence": round(o.det_confidence, 4),
            "eligible_for_vote": is_eligible(o, cfg)}


def build_records(store: EvidenceStore, cfg: Config,
                  crop_files: Optional[Dict[int, str]] = None) -> List[RecognitionRecord]:
    """One record per plate instance - identical texts on different vehicles stay separate."""
    crop_files = crop_files or {}
    records: List[RecognitionRecord] = []
    for n, inst in enumerate(store.instances, start=1):
        cons = inst.consensus or compute_consensus(inst.observations, cfg.consensus, final=True)
        engines_used = sorted({o.engine for o in inst.observations})
        guess = cons.best_guess

        records.append(RecognitionRecord(
            record_id=n, vehicle_track_id=inst.vehicle_track_id,
            plate_instance_id=inst.plate_instance_id,
            association="vehicle" if inst.key[0] == "vehicle" else "unassociated",
            final_text=cons.text,
            best_guess_text=guess if cons.status != PlateStatus.CONFIRMED else None,
            confidence=round(cons.confidence, 4), status=cons.status.value,
            uncertainty_reason=cons.reason, engines_used=engines_used,
            engines_supporting=list(cons.engines_supporting), engine_agreement=cons.engine_agreement,
            support_frames=cons.support_frames, voting_frames=cons.voting_frames,
            ocr_observation_count=cons.observations_total,
            eligible_observation_count=cons.observations_eligible,
            first_frame=inst.first_frame, last_frame=inst.last_frame,
            first_timestamp_s=round(inst.first_ts, 3), last_timestamp_s=round(inst.last_ts, 3),
            best_crop_frame=inst.best_frame, best_crop_timestamp_s=round(inst.best_ts, 3),
            detection_count=inst.detection_count,
            detection_confidence_mean=round(inst.det_conf_sum / max(inst.detection_count, 1), 4),
            detection_confidence_max=round(inst.det_conf_max, 4),
            format_valid_indian=is_indian_format(guess) if guess else None,
            candidates=[{"text": c.text, "support_frames": c.support_frames, "share": round(c.share, 4),
                         "engines": list(c.engines), "mean_ocr_confidence": round(c.mean_ocr_confidence, 4)}
                        for c in cons.candidates],
            ambiguous_positions=cons.ambiguous_positions,
            observations=[_obs_dict(o, cfg.consensus) for o in inst.observations],
            crop_file=crop_files.get(inst.plate_instance_id)))
    return records
