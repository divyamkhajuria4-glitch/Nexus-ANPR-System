"""Frame annotation. Uncertain readings are labelled as such, never shown as text."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from records import BBox, VehicleDet

GREEN = (60, 200, 60)
YELLOW = (0, 215, 255)
ORANGE = (0, 140, 255)
GREY = (170, 170, 170)
BLUE = (230, 160, 40)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Normalized vehicle label mapping for clean on-screen display
_VEHICLE_CLASS_MAP = {
    "motorcycle": "bike",
    "motorbike": "bike",
    "bicycle": "cycle",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "auto": "auto",
    "rickshaw": "auto",
    "autorickshaw": "auto",
    "auto_rickshaw": "auto",
    "auto-rickshaw": "auto",
}


def _format_vehicle_label(class_name: Optional[str]) -> str:
    if not class_name:
        return "vehicle"
    raw = str(class_name).strip().lower()
    return _VEHICLE_CLASS_MAP.get(raw, raw)


def _scale(frame: np.ndarray) -> Tuple[float, int, int]:
    h = frame.shape[0]
    font = max(0.45, h / 1080 * 0.75)
    # Increased thickness for crisp, prominent bounding boxes on all resolutions
    box_thick = max(2, int(round(h / 360)))
    text_thick = max(1, int(round(box_thick * 0.5)))
    return font, box_thick, text_thick


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int],
               font_scale: float, thickness: int, above: bool = True) -> None:
    (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, thickness)
    h, w = frame.shape[:2]
    pad = 4
    x = int(min(max(x, 0), max(w - tw - pad * 2, 0)))
    top = y - th - base - pad * 2 if above else y + 2
    if top < 0:
        top = y + 2
    top = int(min(top, max(h - th - base - pad * 2, 0)))
    cv2.rectangle(frame, (x, top), (x + tw + pad * 2, top + th + base + pad * 2), color, -1)
    cv2.putText(frame, text, (x + pad, top + th + pad), _FONT, font_scale, BLACK, thickness, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, vehicles: Sequence[VehicleDet],
               plates: Sequence[Tuple[BBox, Optional[str]]],
               hud_lines: Sequence[str]) -> np.ndarray:
    """Return an annotated copy with vehicle and plate boxes."""
    out = frame.copy()
    font, box_thick, text_thick = _scale(out)
    for v in vehicles:
        x1, y1, x2, y2 = v.bbox.to_int()
        # Single consistent blue color with increased thickness
        cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, box_thick)
        label = _format_vehicle_label(v.class_name)
        if label:
            draw_label(out, label, x1, y1, BLUE, font, text_thick, above=True)
    for bbox, text in plates:
        if text:
            color = GREEN
        else:
            color = ORANGE
        x1, y1, x2, y2 = bbox.to_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick + 1)
        if text:
            draw_label(out, text, x1, y2, color, font, text_thick, above=False)
    return out
