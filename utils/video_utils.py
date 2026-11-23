"""Video I/O helpers: probing, frame iteration, writing, H.264 transcoding."""
from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
DEFAULT_FPS = 25.0


class VideoError(RuntimeError):
    """Raised for unreadable / invalid video input or output problems."""


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int  # container estimate; may be 0 for some streams


def probe_video(path: Path) -> VideoInfo:
    """Validate a video file and read its basic properties."""
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Video file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        logger.warning("Unusual video extension '%s'; trying to open anyway.", path.suffix)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV could not open video: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoError(f"Video has no readable frames: {path}")
            height, width = frame.shape[:2]
        if not math.isfinite(fps) or fps <= 0:
            logger.warning("Video reports invalid FPS (%s); assuming %.1f.", fps, DEFAULT_FPS)
            fps = DEFAULT_FPS
        return VideoInfo(path, width, height, fps, max(count, 0))
    finally:
        cap.release()


@contextmanager
def open_capture(path: Path) -> Iterator[cv2.VideoCapture]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"OpenCV could not open video: {path}")
        yield cap
    finally:
        cap.release()


def iter_frames(path: Path, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, BGR frame)`` sequentially."""
    with open_capture(path) as cap:
        idx = 0
        while max_frames is None or idx < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield idx, frame
            idx += 1


def _fourcc(code: str) -> int:
    fn = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return fn(*code)


def create_writer(path: Path, fps: float, size: Tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    """Open a VideoWriter; raises VideoError when the backend cannot."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), _fourcc(codec), float(fps), (int(size[0]), int(size[1])))
    if not writer.isOpened():
        writer.release()
        raise VideoError(f"Could not open video writer for {path} (codec={codec}, size={size}).")
    return writer


def transcode_h264(src: Path, dst: Path, timeout_s: int = 600) -> bool:
    """Re-encode to H.264/yuv420p so the file plays in browsers and QuickTime.

    Returns False (and leaves ``src`` untouched) if ffmpeg is missing or fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        logger.warning("ffmpeg not found; keeping the mp4v file (install ffmpeg: brew install ffmpeg).")
        return False
    dst = Path(dst)
    partial = dst.with_name(f".{dst.stem}.partial{dst.suffix}")
    partial.unlink(missing_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", "-an", str(partial)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg transcode failed: %s", exc)
        partial.unlink(missing_ok=True)
        return False
    if proc.returncode != 0 or not partial.is_file() or not _is_valid_h264(partial, Path(src)):
        logger.warning("ffmpeg transcode failed: %s", proc.stderr.strip()[:300])
        partial.unlink(missing_ok=True)
        return False
    os.replace(partial, dst)
    return True


def _is_valid_h264(path: Path, source: Optional[Path] = None) -> bool:
    """Require a readable H.264 video stream before publishing a transcode."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                    "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
                                   capture_output=True, text=True, timeout=30)
            if probe.returncode != 0 or "h264" not in probe.stdout.lower():
                return False
            decoded = probe_video(path)
            if decoded.frame_count <= 0:
                return False
            if source is not None:
                original = probe_video(source)
                expected_size = ((original.width // 2) * 2, (original.height // 2) * 2)
                if (decoded.width, decoded.height) != expected_size:
                    return False
                if original.frame_count and decoded.frame_count != original.frame_count:
                    return False
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False
    decoded = probe_video(path)
    return decoded.frame_count > 0


def estimated_video_space_bytes(width: int, height: int, frames: int) -> int:
    """Conservative allowance for raw annotated frames plus H.264 temporary output."""
    return max(256 * 1024 * 1024, int(width * height * max(frames, 0) * 0.04))


def ensure_output_space(path: Path, required_bytes: int) -> None:
    """Fail before processing if the target volume lacks safe working space."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise VideoError(f"Insufficient output disk space: need about {required_bytes:,} bytes, "
                         f"only {free:,} bytes are free at {path}.")
