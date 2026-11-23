"""NEXUS ANPR command-line interface.

    python main.py check                       # environment + weights report
    python main.py prepare-ocr                 # one-time OCR model download (needs internet)
    python main.py run --video input/traffic.mp4
"""
from __future__ import annotations

import os

# Must be set before torch is imported: lets unsupported ops fall back to CPU on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import importlib
import logging
import platform
import sys
from pathlib import Path
from typing import List, Optional

from config import Config
from utils.logging_setup import setup_logging

logger = logging.getLogger("nexus_anpr")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=Path, help="JSON config file (see README)")
    p.add_argument("--vehicle-weights", type=Path)
    p.add_argument("--plate-weights", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--engines", nargs="+", choices=["easyocr", "paddleocr", "tesseract"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-anpr", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process a recorded video")
    _add_common(run)
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--stride", type=int, help="process every Nth frame (default 1)")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--vehicle-conf", type=float)
    run.add_argument("--plate-conf", type=float)
    run.add_argument("--no-crops", action="store_true", help="do not save plate crop images")
    run.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    run.add_argument("--verbose", action="store_true")

    chk = sub.add_parser("check", help="report Python/package/device/weights status")
    _add_common(chk)

    prep = sub.add_parser("prepare-ocr", help="one-time, explicit download of OCR models")
    _add_common(prep)
    return parser


def load_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()
    if args.vehicle_weights:
        cfg.models.vehicle_weights = args.vehicle_weights
    if args.plate_weights:
        cfg.models.plate_weights = args.plate_weights
    if args.device:
        cfg.models.device = args.device
    if args.engines:
        cfg.ocr.engines = tuple(args.engines)
    if getattr(args, "output_dir", None):
        cfg.output.output_dir = args.output_dir
    if getattr(args, "stride", None):
        cfg.video.frame_stride = args.stride
    if getattr(args, "max_frames", None):
        cfg.video.max_frames = args.max_frames
    if getattr(args, "vehicle_conf", None) is not None:
        cfg.detection.vehicle_conf = args.vehicle_conf
    if getattr(args, "plate_conf", None) is not None:
        cfg.detection.plate_conf = args.plate_conf
    if getattr(args, "no_crops", False):
        cfg.output.save_crops = False
    if getattr(args, "no_video", False):
        cfg.output.save_video = False
    cfg.validate()
    return cfg


def cmd_check(cfg: Config) -> int:
    from models.model_manager import ModelError, validate_weights_path
    from utils.device import select_device

    print(f"Python {platform.python_version()} on {platform.platform()}")
    missing = 0
    for mod in ("cv2", "numpy", "torch", "ultralytics", "easyocr", "paddleocr", "streamlit", "pytest"):
        try:
            m = importlib.import_module(mod)
            print(f"  [ok]      {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            required = mod in ("cv2", "numpy", "torch", "ultralytics") or mod in cfg.ocr.engines
            print(f"  [{'MISSING' if required else 'optional'}] {mod}: {exc.__class__.__name__}")
            missing += 1 if required else 0
    print(f"Device selected: {select_device(cfg.models.device)}")
    for role, path in (("vehicle", cfg.models.vehicle_weights), ("plate", cfg.models.plate_weights)):
        try:
            print(f"  [ok]      {role} weights: {validate_weights_path(path, role)}")
        except ModelError as exc:
            print(f"  [MISSING] {role} weights: {exc}")
            missing += 1
    import shutil
    print(f"  [{'ok' if shutil.which('ffmpeg') else 'optional'}] ffmpeg (H.264 re-encode): {shutil.which('ffmpeg')}")
    return 1 if missing else 0


def cmd_prepare_ocr(cfg: Config) -> int:
    from recognition.ocr_engine import build_engines
    cfg.ocr.allow_model_download = True
    print("Downloading/initialising OCR models (internet required, one time only)...")
    ok = True
    for eng in build_engines(cfg):
        try:
            eng.ensure_loaded()
            print(f"  [ok] {eng.name}")
        except Exception as exc:
            ok = False
            print(f"  [FAILED] {eng.name}: {exc}")
    return 0 if ok else 1


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from models.model_manager import ModelError
    from pipeline.anpr_pipeline import ANPRPipeline, PipelineError
    from recognition.ocr_engine import OCRUnavailableError
    from utils.video_utils import VideoError

    pipeline = ANPRPipeline.from_config(cfg)
    last = {"pct": -1}

    def progress(done: int, total: int, fps: float) -> None:
        pct = int(100 * done / total) if total else 0
        if pct != last["pct"] and (pct % 5 == 0 or done == total):
            last["pct"] = pct
            print(f"\r  {done}/{total or '?'} frames ({pct}%)  {fps:.1f} fps", end="", flush=True)

    try:
        result = pipeline.run(args.video, progress=progress)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except (ModelError, OCRUnavailableError, VideoError, PipelineError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        pipeline.close()

    s = result.stats
    print("\n\n=== NEXUS ANPR summary ===")
    print(f"Frames read/processed : {s.frames_read}/{s.frames_processed} (stride {s.frame_stride})")
    print(f"Vehicle tracks        : {s.vehicle_tracks}")
    print(f"Plate instances       : {s.plate_instances}")
    print(f"OCR observations      : {s.ocr_observations}")
    print(f"Recognized (confirmed): {s.recognized_plates}")
    print(f"Uncertain / no reading: {s.uncertain_readings} / {s.no_reading}")
    print(f"Unique plate texts    : {s.unique_plate_texts}")
    print(f"Processing FPS        : {s.processing_fps}")
    for r in result.records:
        shown = r.final_text or (f"UNCERTAIN — {r.best_guess_text}" if r.best_guess_text
                                 else f"UNCERTAIN ({r.uncertainty_reason})")
        print(f"  record {r.record_id:>3} | vehicle {str(r.vehicle_track_id):>5} | {shown}")
    print(f"\nOutput folder: {result.run_dir}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args)
    except (ValueError, OSError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg.output.log_dir, logging.DEBUG if getattr(args, "verbose", False) else logging.INFO)
    if args.command == "check":
        return cmd_check(cfg)
    if args.command == "prepare-ocr":
        return cmd_prepare_ocr(cfg)
    return cmd_run(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
