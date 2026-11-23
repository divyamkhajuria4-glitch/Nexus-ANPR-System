"""Safe plate-crop extraction (coordinate clipping + invalid/small crop rejection)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from records import BBox


def clip_bbox(bbox: BBox, width: int, height: int) -> Optional[BBox]:
    """Clip to image bounds; None if the box is non-finite or has no area left."""
    if width <= 0 or height <= 0 or not bbox.is_finite():
        return None
    clipped = bbox.clipped(width, height)
    return clipped if clipped.area > 0 else None


def padded_clip(bbox: BBox, width: int, height: int, pad_ratio: float) -> Optional[BBox]:
    if not bbox.is_finite():
        return None
    px, py = bbox.width * pad_ratio, bbox.height * pad_ratio
    grown = BBox(bbox.x1 - px, bbox.y1 - py, bbox.x2 + px, bbox.y2 + py)
    return clip_bbox(grown, width, height)


def extract_plate_crop(frame: np.ndarray, bbox: BBox, min_width: int, min_height: int,
                       pad_ratio: float = 0.0) -> Optional[np.ndarray]:
    """Return an independent copy of the plate region, or None if unusable."""
    if frame is None or frame.size == 0 or frame.ndim < 2:
        return None
    h, w = frame.shape[:2]
    box = padded_clip(bbox, w, h, pad_ratio)
    if box is None:
        return None
    x1, y1, x2, y2 = box.to_int()
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if (x2 - x1) < min_width or (y2 - y1) < min_height:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or not math.isfinite(float(crop.mean())):
        return None
    return crop.copy()
