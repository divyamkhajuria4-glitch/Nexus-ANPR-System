"""Centralised, lazy, validated model loading.

Weights are NEVER downloaded here: the path is checked before Ultralytics sees
it (Ultralytics would otherwise try to fetch known names such as ``yolov8n.pt``).
"""
from __future__ import annotations

import gc
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from config import ModelConfig
from utils.device import empty_device_cache

logger = logging.getLogger(__name__)

ROLE_HINTS = {
    "vehicle": (
        "Vehicle detector weights are missing. Place a YOLO .pt file (for example an "
        "Ultralytics COCO model such as yolov8n.pt) at the configured path. Obtain it yourself "
        "from the official Ultralytics releases (https://github.com/ultralytics/assets/releases) "
        "or by running once on a networked machine: "
        "python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" and then moving the file."
    ),
    "plate": (
        "License-plate detector weights are missing. A COCO model cannot detect plates. You must "
        "supply a YOLO .pt model trained for license plates (a pretrained one from a source whose "
        "licence you have checked, or one you train yourself) and place it at the configured path."
    ),
}


class ModelError(RuntimeError):
    """Base class for model problems."""


class ModelNotFoundError(ModelError, FileNotFoundError):
    """Weights file missing; the message says exactly where to put it."""


def validate_weights_path(path: Path, role: str) -> Path:
    p = Path(path).expanduser()
    hint = ROLE_HINTS.get(role, "")
    if not p.exists():
        raise ModelNotFoundError(f"{role} weights not found at: {p}\n{hint}")
    if not p.is_file():
        raise ModelError(f"{role} weights path is not a file: {p}")
    if p.suffix.lower() != ".pt":
        raise ModelError(f"{role} weights must be a PyTorch .pt file, got '{p.suffix}': {p}")
    if p.stat().st_size == 0:
        raise ModelError(f"{role} weights file is empty: {p}")
    return p.resolve()


def _default_loader(path: Path) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelError("ultralytics is not installed. Run: pip install -r requirements.txt") from exc
    return YOLO(str(path))


class ModelManager:
    """Loads each detector once, on first use, and releases them on demand."""

    def __init__(self, config: ModelConfig, loader: Optional[Callable[[Path], Any]] = None) -> None:
        self._config = config
        self._loader = loader or _default_loader
        self._models: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _path_for(self, role: str) -> Path:
        if role == "vehicle":
            return self._config.vehicle_weights
        if role == "plate":
            return self._config.plate_weights
        raise ValueError(f"Unknown model role '{role}'")

    def validate_all(self) -> None:
        for role in ("vehicle", "plate"):
            validate_weights_path(self._path_for(role), role)

    def get(self, role: str) -> Any:
        with self._lock:
            if role in self._models:
                return self._models[role]
            path = validate_weights_path(self._path_for(role), role)
            logger.info("Loading %s model from %s", role, path)
            try:
                model = self._loader(path)
            except ModelError:
                raise
            except Exception as exc:
                raise ModelError(f"Failed to load {role} model from {path}: {exc}") from exc
            self._models[role] = model
            return model

    def is_loaded(self, role: str) -> bool:
        return role in self._models

    def release(self) -> None:
        with self._lock:
            self._models.clear()
        gc.collect()
        empty_device_cache()
        logger.info("Models released.")

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()
