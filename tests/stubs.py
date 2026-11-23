"""Test doubles + a synthetic video generator.

IMPORTANT: stubs return SCRIPTED detections/OCR text. Tests built on them prove the
pipeline mechanics (tracking-state handling, consensus, counting, export, video I/O),
NOT detection or OCR accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from records import BBox, PlateDet, VehicleDet
from recognition.ocr_engine import OCREngine
from recognition.text_processor import TextFragment

W, H, FPS = 640, 360, 25.0
CAR_W, CAR_H = 160, 90
PLATE_DX, PLATE_DY, PLATE_W, PLATE_H = 45, 55, 70, 22


@dataclass
class SynthVehicle:
    track_id: int
    text: str
    start: int
    end: int
    y: int
    x0: int = 20
    speed: int = 4
    tint: int = 0                       # unique plate-background tint index
    plate_detected: bool = True
    script: Optional[List[str]] = None  # what the stub OCR "reads", cycled per call

    def visible(self, f: int) -> bool:
        return self.start <= f <= self.end

    def car_box(self, f: int) -> BBox:
        x = self.x0 + self.speed * (f - self.start)
        return BBox(x, self.y, x + CAR_W, self.y + CAR_H)

    def plate_box(self, f: int) -> BBox:
        c = self.car_box(f)
        return BBox(c.x1 + PLATE_DX, c.y1 + PLATE_DY, c.x1 + PLATE_DX + PLATE_W, c.y1 + PLATE_DY + PLATE_H)

    @property
    def plate_bgr(self):
        return (250 - 30 * self.tint, 245, 245)


def make_video(path: Path, vehicles: Sequence[SynthVehicle], n_frames: int, fps: float = FPS) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (W, H))
    assert writer.isOpened(), "synthetic video writer failed"
    for f in range(n_frames):
        img = np.full((H, W, 3), 90, np.uint8)
        for v in vehicles:
            if not v.visible(f):
                continue
            cb, pb = v.car_box(f).to_int(), v.plate_box(f).to_int()
            cv2.rectangle(img, (cb[0], cb[1]), (cb[2], cb[3]), (120, 60, 40), -1)
            cv2.rectangle(img, (pb[0], pb[1]), (pb[2], pb[3]), v.plate_bgr, -1)
            cv2.putText(img, v.text[:8], (pb[0] + 2, pb[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        writer.write(img)
    writer.release()
    return path


class StubVehicleDetector:
    def __init__(self, vehicles: Sequence[SynthVehicle], stride: int = 1) -> None:
        self.vehicles, self.stride, self.calls = list(vehicles), stride, 0
        self.warmed = self.resets = 0

    def warmup(self) -> None: self.warmed += 1
    def reset(self) -> None: self.calls, self.resets = 0, self.resets + 1

    def track(self, frame: np.ndarray) -> List[VehicleDet]:
        # The pipeline tracks vehicles on every decoded frame, independent of
        # the plate/OCR stride.
        f = self.calls
        self.calls += 1
        return [VehicleDet(v.car_box(f), 0.9, 2, "car", v.track_id) for v in self.vehicles if v.visible(f)]


class StubPlateDetector:
    def __init__(self, vehicles: Sequence[SynthVehicle], stride: int = 1) -> None:
        self.vehicles, self.stride, self.calls = list(vehicles), stride, 0

    def warmup(self) -> None: pass
    def reset(self) -> None: self.calls = 0

    def detect(self, frame: np.ndarray) -> List[PlateDet]:
        f = self.calls * self.stride
        self.calls += 1
        return [PlateDet(v.plate_box(f), 0.88) for v in self.vehicles if v.visible(f) and v.plate_detected]


class ScriptedEngine(OCREngine):
    """Identifies the plate by its background tint and replays that vehicle's script."""
    name = "stub"

    def __init__(self, vehicles: Sequence[SynthVehicle], name: str = "stub", conf: float = 0.9) -> None:
        super().__init__()
        self.name = name
        self._by_tint = {v.plate_bgr[0]: v for v in vehicles}
        self._counts: Dict[int, int] = {}
        self._conf = conf
        self.closed = False

    def _load(self) -> None: pass

    def _read(self, image_bgr: np.ndarray) -> List[TextFragment]:
        # Use the crop centre rather than a percentile: lossy video codec
        # artifacts on text/edges must not select another synthetic vehicle.
        blue = float(np.mean(image_bgr[:, :, 0]))
        tint = min(self._by_tint, key=lambda t: abs(t - blue))
        veh = self._by_tint[tint]
        n = self._counts.get(tint, 0)
        self._counts[tint] = n + 1
        script = veh.script or [veh.text]
        return [TextFragment(script[n % len(script)], self._conf, 50.0, 10.0, 20.0)]

    def close(self) -> None:
        self.closed = True
        super().close()


@dataclass
class FakeBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    id: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class FakeResult:
    boxes: Optional[FakeBoxes]
    names: Dict[int, str] = field(default_factory=lambda: {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"})
