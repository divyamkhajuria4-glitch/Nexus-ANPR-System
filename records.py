"""Shared data structures (kept dependency-free so every module can import them)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PlateStatus(str, Enum):
    CONFIRMED = "confirmed"      # enough agreeing evidence
    PENDING = "pending"          # not enough evidence yet (live state only)
    UNCERTAIN = "uncertain"      # conflicting or insufficient evidence at the end
    NO_READING = "no_reading"    # plate seen, no usable OCR output


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x1, self.y1, self.x2, self.y2))

    def intersection_area(self, other: "BBox") -> float:
        iw = min(self.x2, other.x2) - max(self.x1, other.x1)
        ih = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, iw) * max(0.0, ih)

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def fraction_inside(self, other: "BBox") -> float:
        """Fraction of THIS box's area that lies inside ``other``."""
        return self.intersection_area(other) / self.area if self.area > 0 else 0.0

    def clipped(self, width: int, height: int) -> "BBox":
        return BBox(
            min(max(self.x1, 0.0), float(width)),
            min(max(self.y1, 0.0), float(height)),
            min(max(self.x2, 0.0), float(width)),
            min(max(self.y2, 0.0), float(height)),
        )

    def to_int(self) -> Tuple[int, int, int, int]:
        return (int(math.floor(self.x1)), int(math.floor(self.y1)),
                int(math.ceil(self.x2)), int(math.ceil(self.y2)))


@dataclass(frozen=True)
class VehicleDet:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int]  # None until ByteTrack confirms the track


@dataclass(frozen=True)
class PlateDet:
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class OCRObservation:
    """One OCR read of one plate crop variant. Never modified after creation."""

    frame_index: int
    timestamp_s: float
    engine: str
    variant: str
    raw_text: str
    normalized_text: str
    ocr_confidence: float
    det_confidence: float


@dataclass(frozen=True)
class CandidateSummary:
    text: str
    score: float
    support_frames: int
    share: float
    engines: Tuple[str, ...]
    mean_ocr_confidence: float


@dataclass
class ConsensusResult:
    status: PlateStatus
    text: Optional[str]            # only set when CONFIRMED
    best_guess: Optional[str]      # leading candidate, NOT a confirmed reading
    confidence: float              # heuristic, not a calibrated probability
    support_frames: int
    voting_frames: int
    observations_total: int
    observations_eligible: int
    candidates: List[CandidateSummary] = field(default_factory=list)
    engines_supporting: Tuple[str, ...] = ()
    engine_agreement: bool = False
    conflicting: bool = False
    ambiguous_positions: List[int] = field(default_factory=list)  # 1-based positions
    reason: str = ""


@dataclass
class RecognitionRecord:
    record_id: int
    vehicle_track_id: Optional[int]
    plate_instance_id: int
    association: str                       # "vehicle" | "unassociated"
    final_text: Optional[str]              # only for CONFIRMED
    best_guess_text: Optional[str]         # never a confirmed reading
    confidence: float
    status: str
    uncertainty_reason: str
    engines_used: List[str]
    engines_supporting: List[str]
    engine_agreement: bool
    support_frames: int
    voting_frames: int
    ocr_observation_count: int
    eligible_observation_count: int
    first_frame: int
    last_frame: int
    first_timestamp_s: float
    last_timestamp_s: float
    best_crop_frame: int
    best_crop_timestamp_s: float
    detection_count: int
    detection_confidence_mean: float
    detection_confidence_max: float
    format_valid_indian: Optional[bool]
    candidates: List[Dict[str, Any]]
    ambiguous_positions: List[int]
    observations: List[Dict[str, Any]]
    crop_file: Optional[str] = None


@dataclass
class PipelineStats:
    frames_total: int = 0
    frames_read: int = 0
    frames_processed: int = 0
    vehicle_detections: int = 0
    plate_detections: int = 0
    frame_stride: int = 1
    processing_fps: float = 0.0
    vehicle_tracks: int = 0
    plate_instances: int = 0
    ocr_observations: int = 0
    recognition_records: int = 0
    recognized_plates: int = 0
    pending_readings: int = 0
    uncertain_readings: int = 0
    no_reading: int = 0
    unique_plate_texts: int = 0
    rejected_plate_crops: int = 0
    rejected_plate_geometry: int = 0
    frame_errors: int = 0
    cancelled: bool = False
