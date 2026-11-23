"""Central configuration for NEXUS ANPR.

Every tunable value lives here. Defaults are conservative; override them with a
JSON file (``--config``) or CLI flags rather than editing code.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

# COCO class ids: 2=car, 3=motorcycle, 5=bus, 7=truck
COCO_VEHICLE_CLASSES: Tuple[int, ...] = (2, 3, 5, 7)


@dataclass
class ModelConfig:
    """Locations of locally stored weights. Nothing is ever downloaded at run time."""

    vehicle_weights: Path = WEIGHTS_DIR / "yolov8n.pt"
    plate_weights: Path = WEIGHTS_DIR / "license_plate_detector.pt"
    device: str = "auto"  # auto | cpu | mps | cuda


@dataclass
class DetectionConfig:
    vehicle_conf: float = 0.30
    vehicle_iou: float = 0.5
    vehicle_classes: Tuple[int, ...] = COCO_VEHICLE_CLASSES
    vehicle_imgsz: int = 640
    # Small same-frame review favored precision at .20 over .10/.15; residual
    # branding/text false positives remain and must not be called ground truth.
    plate_conf: float = 0.35
    plate_iou: float = 0.45
    plate_imgsz: int = 1280  # real 4K frame diagnosis: 1280 retained small plates better than 960
    plate_class_ids: Optional[Tuple[int, ...]] = None  # None = accept every class
    min_plate_width: int = 20  # px, on the ORIGINAL frame
    min_plate_height: int = 8
    plate_pad_ratio: float = 0.20  # padding prevents edge characters from being cut
    association_min_overlap: float = 0.65  # fraction of plate area inside vehicle box
    one_plate_per_vehicle_per_frame: bool = False


@dataclass
class TrackingConfig:
    tracker_config: str = "bytetrack.yaml"  # Ultralytics' built-in ByteTrack config
    orphan_iou_threshold: float = 0.3  # IoU tracker for plates with no vehicle
    orphan_max_age_frames: int = 15


@dataclass
class OCRConfig:
    engines: Tuple[str, ...] = ("easyocr",)  # any of: easyocr, paddleocr
    languages: Tuple[str, ...] = ("en",)
    allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    allow_model_download: bool = False  # only `main.py prepare-ocr` sets this True
    easyocr_model_dir: Optional[Path] = None
    easyocr_use_gpu: bool = False
    primary_variants: Tuple[str, ...] = ("color", "clahe")
    fallback_variants: Tuple[str, ...] = ("sharpen", "bilateral", "otsu", "adaptive", "perspective")
    fallback_trigger_confidence: float = 0.60  # run fallback variants if best read is below this
    min_resize_height: int = 64
    max_resize_width: int = 640
    max_ocr_frames_per_track: int = 30
    stop_ocr_after_confirmed_frames: int = 10


@dataclass
class ConsensusConfig:
    min_text_length: int = 4
    max_text_length: int = 12
    min_vote_confidence: float = 0.60  # below this, OCR is uncertain and is not confirmed
    min_support_frames: int = 3  # exact text must recur across distinct video frames
    min_weight_share: float = 0.60  # winner's share of all voting weight
    min_margin_share: float = 0.25  # winner minus runner-up, as share of all weight


@dataclass
class VideoConfig:
    frame_stride: int = 1  # 1 = run detection/OCR on every frame
    output_fps: Optional[float] = None  # None = original FPS
    output_scale: float = 1.0
    max_frames: Optional[int] = None
    max_consecutive_frame_errors: int = 5


@dataclass
class OutputConfig:
    output_dir: Path = PROJECT_ROOT / "output"
    log_dir: Path = PROJECT_ROOT / "logs"
    save_crops: bool = True
    save_video: bool = True
    transcode_h264: bool = True  # needs ffmpeg on PATH; falls back to mp4v


@dataclass
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Raise ValueError for settings that cannot work."""
        if self.video.frame_stride < 1:
            raise ValueError("video.frame_stride must be >= 1")
        if not 0 < self.video.output_scale <= 4:
            raise ValueError("video.output_scale must be in (0, 4]")
        for name in ("vehicle_conf", "vehicle_iou", "plate_conf", "plate_iou",
                     "association_min_overlap"):
            if not 0.0 <= getattr(self.detection, name) <= 1.0:
                raise ValueError(f"detection.{name} must be within [0, 1]")
        bad = [e for e in self.ocr.engines if e not in ("easyocr", "paddleocr", "tesseract")]
        if bad or not self.ocr.engines:
            raise ValueError(f"ocr.engines must be a non-empty subset of easyocr/paddleocr, got {self.ocr.engines}")
        if self.consensus.min_text_length > self.consensus.max_text_length:
            raise ValueError("consensus.min_text_length > max_text_length")
        if self.consensus.min_support_frames < 1:
            raise ValueError("consensus.min_support_frames must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()
        _apply(cfg, data)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def _coerce(default: Any, value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(default, Path) or name.endswith(("_dir", "_weights")):
        path = Path(value).expanduser()
        if not path.is_absolute() and name in {"vehicle_weights", "plate_weights"}:
            path = PROJECT_ROOT / path
        return path
    if isinstance(default, tuple) or isinstance(value, list):
        return tuple(value)
    return value


def _apply(target: Any, data: Dict[str, Any]) -> None:
    known = {f.name for f in fields(target)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key '{key}' for {type(target).__name__}")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply(current, value)
        else:
            setattr(target, key, _coerce(current, value, key))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
