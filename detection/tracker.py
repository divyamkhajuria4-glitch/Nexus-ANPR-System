"""Tracking helpers.

Vehicle tracking uses ByteTrack as implemented by Ultralytics (``bytetrack.yaml``),
invoked from ``models/vehicle_detector.py`` so detection and tracking share one pass.

``IoUTracker`` below only gives stable IDs to plate detections that could not be
attached to any tracked vehicle ("orphan" plates), so they are not counted once per frame.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from records import BBox


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 15) -> None:
        self._thr = iou_threshold
        self._max_age = max_age_frames
        self._tracks: Dict[int, Tuple[BBox, int]] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, boxes: List[BBox], frame_index: int) -> List[int]:
        """Return one track id per input box (greedy highest-IoU matching)."""
        pairs = []
        for bi, box in enumerate(boxes):
            for tid, (tbox, _) in self._tracks.items():
                iou = box.iou(tbox)
                if iou >= self._thr:
                    pairs.append((iou, tid, bi))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        ids: List[int] = [-1] * len(boxes)
        used_tracks, used_boxes = set(), set()
        for _, tid, bi in pairs:
            if tid in used_tracks or bi in used_boxes:
                continue
            ids[bi] = tid
            used_tracks.add(tid)
            used_boxes.add(bi)
        for bi in range(len(boxes)):
            if ids[bi] == -1:
                ids[bi] = self._next_id
                self._next_id += 1
        for bi, tid in enumerate(ids):
            self._tracks[tid] = (boxes[bi], frame_index)
        stale = [t for t, (_, last) in self._tracks.items() if frame_index - last > self._max_age]
        for t in stale:
            del self._tracks[t]
        return ids
