"""Plate-crop preprocessing. Produces several OCR-ready variants; the original crop
is never modified. Which variant helps is decided by OCR evidence, not assumed."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ALL_VARIANTS = ("color", "gray", "clahe", "bilateral", "gaussian",
                "sharpen", "otsu", "adaptive", "perspective")


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def resize_for_ocr(img: np.ndarray, min_height: int = 64, max_width: int = 640) -> np.ndarray:
    """Upscale the crop 2x with cubic interpolation for the stable OCR pass."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Cannot resize an empty image")
    scale = 2.0
    if h * scale < min_height:
        scale = max(scale, min_height / h)
    if w * scale > max_width:
        scale = min(scale, max_width / w)
    if abs(scale - 1.0) < 1e-3:
        return img.copy()
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, new_size, interpolation=interp)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(gray)


def sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    return cv2.addWeighted(gray, 1.6, blur, -0.6, 0)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation, polarity-normalised so the background is the majority (white)."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary > 127)) < 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    block = max(11, (min(gray.shape[:2]) // 2) | 1)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 10)
    if float(np.mean(binary > 127)) < 0.5:
        binary = cv2.bitwise_not(binary)
    return binary


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).reshape(-1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]   # tl, br
    rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]   # tr, bl
    return rect


def correct_perspective(img: np.ndarray, min_area_ratio: float = 0.4) -> Optional[np.ndarray]:
    """Warp a clearly visible 4-corner plate outline to a rectangle.

    Returns None when no convex quadrilateral covering ``min_area_ratio`` of the crop
    is found (the usual case for tight detector crops) - callers then skip this variant.
    """
    gray = to_gray(img)
    h, w = gray.shape[:2]
    edges = cv2.dilate(cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150), None)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(cnt) < min_area_ratio * h * w:
            break
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = _order_points(approx.reshape(4, 2).astype(np.float32))
            tl, tr, br, bl = rect
            width = int(round(max(math.dist(br, bl), math.dist(tr, tl))))
            height = int(round(max(math.dist(tr, br), math.dist(tl, bl))))
            if width < 8 or height < 8:
                continue
            dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
            return cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect, dst), (width, height))
    return None


def build_variants(crop: np.ndarray, names: Sequence[str], min_height: int = 64,
                   max_width: int = 640) -> Dict[str, np.ndarray]:
    """Return ``{variant_name: image}`` for the requested variants (original untouched)."""
    if crop is None or crop.size == 0:
        return {}
    base = resize_for_ocr(crop, min_height, max_width)
    gray = to_gray(base)
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name not in ALL_VARIANTS:
            raise ValueError(f"Unknown preprocessing variant '{name}'. Valid: {ALL_VARIANTS}")
        if name == "color":
            out[name] = base
        elif name == "gray":
            out[name] = gray
        elif name == "clahe":
            out[name] = apply_clahe(gray)
        elif name == "bilateral":
            out[name] = cv2.bilateralFilter(gray, 9, 75, 75)
        elif name == "gaussian":
            out[name] = cv2.GaussianBlur(gray, (3, 3), 0)
        elif name == "sharpen":
            out[name] = sharpen(gray)
        elif name == "otsu":
            out[name] = otsu_threshold(gray)
        elif name == "adaptive":
            out[name] = adaptive_threshold(gray)
        elif name == "perspective":
            warped = correct_perspective(base)
            if warped is not None:
                out[name] = resize_for_ocr(warped, min_height, max_width)
    return out


@dataclass(frozen=True)
class CropQuality:
    width: int
    height: int
    sharpness: float  # variance of the Laplacian; higher = sharper
    contrast: float   # grayscale standard deviation


def assess_quality(crop: np.ndarray) -> CropQuality:
    gray = to_gray(crop)
    return CropQuality(
        width=int(gray.shape[1]), height=int(gray.shape[0]),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        contrast=float(gray.std()),
    )


def crop_quality_score(crop: np.ndarray, det_conf: float) -> float:
    """Heuristic used only to pick a representative crop to save; not used for voting."""
    q = assess_quality(crop)
    return float(q.sharpness * q.width * q.height * det_conf)
