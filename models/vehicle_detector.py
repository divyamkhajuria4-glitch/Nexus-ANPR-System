"""Vehicle detection + ByteTrack tracking through Ultralytics' built-in tracker."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import numpy as np

from config import Config
from models.model_manager import ModelManager
from records import BBox, VehicleDet
from utils.device import DeviceContext

logger = logging.getLogger(__name__)


def to_numpy(x: Any) -> np.ndarray:
    """Accept torch tensors or array-likes."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def parse_vehicle_results(results: Any) -> List[VehicleDet]:
    """Convert an Ultralytics ``Results`` list into :class:`VehicleDet` objects."""
    if not results:
        return []
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = to_numpy(boxes.xyxy).reshape(-1, 4)
    conf = to_numpy(boxes.conf).reshape(-1)
    cls = to_numpy(boxes.cls).reshape(-1).astype(int)
    ids_raw = getattr(boxes, "id", None)
    ids = None if ids_raw is None else to_numpy(ids_raw).reshape(-1).astype(int)
    names = getattr(res, "names", None) or {}
    out: List[VehicleDet] = []
    for i in range(len(xyxy)):
        bbox = BBox(*[float(v) for v in xyxy[i]])
        if not bbox.is_finite() or bbox.area <= 0:
            continue
        cid = int(cls[i])
        out.append(VehicleDet(
            bbox=bbox,
            confidence=float(conf[i]),
            class_id=cid,
            class_name=str(names.get(cid, cid)),
            track_id=int(ids[i]) if ids is not None and i < len(ids) else None,
        ))
    return out


class VehicleDetector:
    """Detects cars/motorcycles/buses/trucks and assigns persistent ByteTrack IDs."""

    def __init__(self, manager: ModelManager, config: Config, device: DeviceContext) -> None:
        self._manager = manager
        self._cfg = config
        self._device = device

    def warmup(self) -> None:
        self._manager.get("vehicle")

    def reset(self) -> None:
        """Drop tracker state so a new video job starts with fresh track IDs."""
        if self._manager.is_loaded("vehicle"):
            model = self._manager.get("vehicle")
            if getattr(model, "predictor", None) is not None:
                model.predictor = None  # Ultralytics rebuilds predictor + tracker on next call

    def track(self, frame: np.ndarray) -> List[VehicleDet]:
        model = self._manager.get("vehicle")
        det, trk = self._cfg.detection, self._cfg.tracking

        def _run(device: str) -> Any:
            return model.track(
                frame, persist=True, tracker=trk.tracker_config,
                conf=det.vehicle_conf, iou=det.vehicle_iou,
                classes=list(det.vehicle_classes), imgsz=det.vehicle_imgsz,
                device=device, verbose=False,
            )

        return parse_vehicle_results(self._device.run(_run))
