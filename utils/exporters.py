"""JSON / CSV / crop exports."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from recognition.consensus import PlateInstance
from records import PipelineStats, RecognitionRecord

logger = logging.getLogger(__name__)

NOTES = [
    "final_text is only filled when multi-frame consensus is CONFIRMED; best_guess_text is NOT a confirmed reading.",
    "confidence = winner's weight share x mean OCR confidence. It is a heuristic, not a calibrated probability.",
    "Identical plate text on different vehicle tracks stays as separate records; unique_plate_texts counts distinct confirmed strings.",
    "If ByteTrack switches a vehicle's ID, one physical vehicle can produce two records.",
    "Vehicle tracking runs on every decoded frame. Plate detection/OCR run only on every Nth frame when frame_stride > 1, reducing evidence without stale vehicle boxes.",
    "No accuracy is claimed: results are not compared against ground truth.",
]


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def export_json(path: Path, records: Sequence[RecognitionRecord], stats: PipelineStats,
                video_info: Dict[str, Any], config: Dict[str, Any],
                timeline: Sequence[Dict[str, Any]] = ()) -> Path:
    payload = {"video": video_info, "summary": to_jsonable(stats), "config": config,
               "notes": NOTES, "timeline": to_jsonable(timeline),
               "records": [to_jsonable(r) for r in records]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def _flat(record: RecognitionRecord) -> Dict[str, Any]:
    d = to_jsonable(record)
    d.pop("observations", None)
    d["candidates"] = json.dumps(d.get("candidates", []), ensure_ascii=False)
    for k in ("engines_used", "engines_supporting"):
        d[k] = ";".join(d.get(k, []))
    return d


def export_csv(path: Path, records: Sequence[RecognitionRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(_flat(records[0]).keys()) if records else [
        "record_id", "vehicle_track_id", "plate_instance_id", "final_text", "status"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in records:
            writer.writerow(_flat(r))
    return path


def export_observations_csv(path: Path, records: Sequence[RecognitionRecord]) -> Path:
    """Every raw OCR observation, linked to its record (nothing is dropped)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["record_id", "plate_instance_id", "vehicle_track_id", "frame", "timestamp_s", "engine",
              "variant", "raw_text", "normalized_text", "ocr_confidence", "detection_confidence",
              "eligible_for_vote"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for r in records:
            for o in r.observations:
                row = {"record_id": r.record_id, "plate_instance_id": r.plate_instance_id,
                       "vehicle_track_id": r.vehicle_track_id, **o}
                row["raw_text"] = str(row["raw_text"]).replace("\n", "\\n")
                writer.writerow(row)
    return path


def save_crops(directory: Path, instances: Sequence[PlateInstance]) -> Dict[int, str]:
    """Write each instance's best ORIGINAL (unprocessed) crop; returns id -> filename."""
    saved: Dict[int, str] = {}
    directory.mkdir(parents=True, exist_ok=True)
    for inst in instances:
        if inst.best_crop is None:
            continue
        veh = inst.vehicle_track_id if inst.vehicle_track_id is not None else "none"
        name = f"plate_{inst.plate_instance_id:04d}_vehicle_{veh}_frame_{inst.best_frame}.png"
        if cv2.imwrite(str(directory / name), inst.best_crop):
            saved[inst.plate_instance_id] = f"{directory.name}/{name}"
        else:
            logger.warning("Could not write crop %s", name)
    return saved
