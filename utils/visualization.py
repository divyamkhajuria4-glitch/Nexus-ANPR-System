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


def _scale(frame: np.ndarray) -> Tuple[float, int]:
    h = frame.shape[0]
    font = max(0.45, h / 1080 * 0.8)
    return font, max(1, int(round(h / 540)))


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int],
               font_scale: float, thickness: int, above: bool = True) -> None:
    (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, thickness)
    h, w = frame.shape[:2]
    x = int(min(max(x, 0), max(w - tw - 4, 0)))
    top = y - th - base - 4 if above else y + 2
    if top < 0:
        top = y + 2
    top = int(min(top, max(h - th - base - 4, 0)))
    cv2.rectangle(frame, (x, top), (x + tw + 4, top + th + base + 4), color, -1)
    cv2.putText(frame, text, (x + 2, top + th + 1), _FONT, font_scale, BLACK, thickness, cv2.LINE_AA)


def draw_frame(frame: np.ndarray, vehicles: Sequence[VehicleDet],
               plates: Sequence[Tuple[BBox, Optional[str]]],
               hud_lines: Sequence[str]) -> np.ndarray:
    """Return an annotated copy with vehicle and plate boxes."""
    out = frame.copy()
    font, thick = _scale(out)
    for v in vehicles:
        x1, y1, x2, y2 = v.bbox.to_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, thick)
    for bbox, text in plates:
        if text:
            color = GREEN
        else:
            color = ORANGE
        x1, y1, x2, y2 = bbox.to_int()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick + 1)
        if text:
            draw_label(out, text, x1, y2, color, font, thick, above=False)
    return out
