"""Dedicated license-plate detector (requires user-supplied plate weights)."""
from __future__ import annotations

import logging
from typing import Any, List

import numpy as np

from config import Config
from models.model_manager import ModelManager
from models.vehicle_detector import to_numpy
from records import BBox, PlateDet
from utils.device import DeviceContext

logger = logging.getLogger(__name__)


def parse_plate_results(results: Any, class_ids: Any = None) -> List[PlateDet]:
    if not results:
        return []
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = to_numpy(boxes.xyxy).reshape(-1, 4)
    conf = to_numpy(boxes.conf).reshape(-1)
    cls = to_numpy(boxes.cls).reshape(-1).astype(int) if getattr(boxes, "cls", None) is not None else None
    out: List[PlateDet] = []
    for i in range(len(xyxy)):
        if class_ids is not None and cls is not None and int(cls[i]) not in class_ids:
            continue
        bbox = BBox(*[float(v) for v in xyxy[i]])
        if not bbox.is_finite() or bbox.area <= 0:
            continue
        out.append(PlateDet(bbox=bbox, confidence=float(conf[i])))
    return out


class PlateDetector:
    def __init__(self, manager: ModelManager, config: Config, device: DeviceContext) -> None:
        self._manager = manager
        self._cfg = config
        self._device = device

    def warmup(self) -> None:
        self._manager.get("plate")

    def reset(self) -> None:  # stateless; present so pipelines can treat detectors uniformly
        return None

    def detect(self, frame: np.ndarray) -> List[PlateDet]:
        model = self._manager.get("plate")
        det = self._cfg.detection

        def _run(device: str) -> Any:
            return model.predict(frame, conf=det.plate_conf, iou=det.plate_iou,
                                 imgsz=det.plate_imgsz, device=device, verbose=False)

        return parse_plate_results(self._device.run(_run), det.plate_class_ids)
