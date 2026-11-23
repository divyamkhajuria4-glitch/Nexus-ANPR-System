"""The end-to-end ANPR pipeline: video in -> annotated video + JSON/CSV out."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import cv2
import numpy as np

from config import Config
from detection.association import associate_plates
from detection.plate_extractor import extract_plate_crop
from detection.tracker import IoUTracker
from models.model_manager import ModelManager
from models.plate_detector import PlateDetector
from models.vehicle_detector import VehicleDetector
from recognition.consensus import EvidenceStore, PlateKey, build_records
from recognition.ocr_engine import OCRRunner, build_engines
from recognition.preprocessing import crop_quality_score
from records import (BBox, ConsensusResult, PipelineStats, PlateDet, PlateStatus,
                     RecognitionRecord, VehicleDet)
from utils import exporters
from utils.device import DeviceContext, select_device
from utils.video_utils import (VideoError, create_writer, ensure_output_space,
                               estimated_video_space_bytes, probe_video, transcode_h264)
from utils.visualization import draw_frame

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, float], None]  # (frames_done, frames_total, processing_fps)


class PipelineError(RuntimeError):
    pass


class VehicleDetectorLike(Protocol):
    def warmup(self) -> None: ...
    def reset(self) -> None: ...
    def track(self, frame: np.ndarray) -> List[VehicleDet]: ...


class PlateDetectorLike(Protocol):
    def warmup(self) -> None: ...
    def reset(self) -> None: ...
    def detect(self, frame: np.ndarray) -> List[PlateDet]: ...


@dataclass
class FrameState:
    vehicles: List[VehicleDet] = field(default_factory=list)
    plates: List[Tuple[BBox, PlateKey]] = field(default_factory=list)
    raw_plates: List[PlateDet] = field(default_factory=list)


@dataclass
class PipelineResult:
    records: List[RecognitionRecord]
    stats: PipelineStats
    run_dir: Path
    video_path: Optional[Path]
    json_path: Path
    csv_path: Path
    observations_csv_path: Path
    cancelled: bool = False
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    video_export_error: Optional[str] = None


class ANPRPipeline:
    def __init__(self, config: Config, vehicle_detector: VehicleDetectorLike,
                 plate_detector: PlateDetectorLike, ocr_runner: OCRRunner,
                 model_manager: Optional[ModelManager] = None) -> None:
        config.validate()
        self.config = config
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_runner = ocr_runner
        self.model_manager = model_manager

    @classmethod
    def from_config(cls, config: Config) -> "ANPRPipeline":
        config.validate()
        device = select_device(config.models.device)
        logger.info("Using device: %s", device)
        ctx = DeviceContext(device)
        manager = ModelManager(config.models)
        return cls(config, VehicleDetector(manager, config, ctx), PlateDetector(manager, config, ctx),
                   OCRRunner(build_engines(config, device), config), manager)

    # ------------------------------------------------------------------ setup
    def preflight(self) -> None:
        """Fail early (before touching the video) on missing weights / OCR engines."""
        if self.model_manager is not None:
            self.model_manager.validate_all()
        self.vehicle_detector.warmup()
        self.plate_detector.warmup()
        self.ocr_runner.warmup()

    def _run_dir(self, video: Path, output_dir: Optional[Path]) -> Path:
        base = Path(output_dir) if output_dir else self.config.output.output_dir
        run_dir = base / f"{video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # ------------------------------------------------------------- per frame
    def _process_frame(self, frame: np.ndarray, idx: int, ts: float, store: EvidenceStore,
                       orphan_tracker: IoUTracker, track_ids: set, stats: PipelineStats,
                       detect_plates: bool = True) -> FrameState:
        det = self.config.detection
        vehicles = self.vehicle_detector.track(frame)
        for v in vehicles:
            stats.vehicle_detections += 1
            if v.track_id is not None:
                track_ids.add(v.track_id)
        # Vehicle boxes are refreshed on every decoded frame so annotated video
        # never displays stale positions. Expensive plate detection follows stride.
        if not detect_plates:
            return FrameState(vehicles=vehicles)
        plates = self.plate_detector.detect(frame)
        stats.plate_detections += len(plates)
        logger.info("FRAME %d ts=%.3f -> vehicles=%d plate_detections=%d", idx, ts, len(vehicles), len(plates))
        # The reviewed tail-light false positive had a 0.70 width/height ratio.
        # Keep it in raw detections for overlays/logging, but exclude implausibly
        # narrow/tall geometry from OCR and instance creation.
        ocr_eligible_plates = [p for p in plates if p.bbox.height > 0
                               and 0.8 <= p.bbox.width / p.bbox.height <= 8.0]
        stats.rejected_plate_geometry += len(plates) - len(ocr_eligible_plates)
        assocs = associate_plates(ocr_eligible_plates, vehicles, det.association_min_overlap,
                                  det.one_plate_per_vehicle_per_frame)

        orphan_idx = [i for i, a in enumerate(assocs) if a.vehicle is None or a.vehicle.track_id is None]
        orphan_ids = orphan_tracker.update([assocs[i].plate.bbox for i in orphan_idx], idx)
        orphan_map = dict(zip(orphan_idx, orphan_ids))

        state = FrameState(vehicles=vehicles, raw_plates=plates)
        valid_crops = 0
        rejected_crops = 0
        ocr_observations = 0
        for i, a in enumerate(assocs):
            key: PlateKey
            vtid: Optional[int]
            if i in orphan_map:
                key, vtid = ("orphan", orphan_map[i]), None
            else:
                vtid = a.vehicle.track_id  # type: ignore[union-attr]
                key = ("vehicle", vtid)  # type: ignore[arg-type]
            crop = extract_plate_crop(frame, a.plate.bbox, det.min_plate_width, det.min_plate_height,
                                      det.plate_pad_ratio)
            if crop is None:
                stats.rejected_plate_crops += 1
                rejected_crops += 1
                continue
            valid_crops += 1
            inst = store.observe_detection(key, vtid, idx, ts, a.plate.confidence, crop,
                                           crop_quality_score(crop, a.plate.confidence))
            # Read distinct frames so consensus can confirm a plate from repeated
            # exact readings instead of trying to infer consensus from one crop.
            if idx not in inst.ocr_frames and store.should_run_ocr(inst):
                observations = self.ocr_runner.read_plate(crop, idx, ts, a.plate.confidence)
                store.add_observations(inst, idx, observations, crop,
                                       crop_quality_score(crop, a.plate.confidence), ts)
                stats.ocr_observations += len(observations)
                ocr_observations += len(observations)
            state.plates.append((a.plate.bbox, key))
        logger.info("FRAME %d -> associations=%d valid_crops=%d rejected_crops=%d ocr_observations=%d consensus_instances=%d",
                    idx, len(assocs), valid_crops, rejected_crops, ocr_observations, len(store.instances))
        return state

    def _consensus_for(self, store: EvidenceStore, key: PlateKey) -> Optional[ConsensusResult]:
        for inst in store.instances:
            if inst.key == key:
                return inst.consensus
        return None

    @staticmethod
    def _hud(store: EvidenceStore, n_tracks: int, fps: float) -> List[str]:
        c = store.counts()
        return [f"Vehicle tracks: {n_tracks}   Plate instances: {c['plate_instances']}",
                f"Recognized: {c['recognized']}   Uncertain: {c['uncertain']}   Unique texts: {c['unique_texts']}",
                f"Processing FPS: {fps:.1f}"]

    # ------------------------------------------------------------------- run
    def run(self, video_path: Path, output_dir: Optional[Path] = None,
            progress: Optional[ProgressCallback] = None,
            cancel_event: Optional[threading.Event] = None) -> PipelineResult:
        cfg = self.config
        video_path = Path(video_path)
        info = probe_video(video_path)  # raises VideoError on bad input
        self.preflight()
        run_dir = self._run_dir(video_path, output_dir)
        expected_frames = min(info.frame_count, cfg.video.max_frames) if cfg.video.max_frames else info.frame_count
        if cfg.output.save_video and expected_frames:
            ensure_output_space(run_dir, estimated_video_space_bytes(
                int(info.width * cfg.video.output_scale), int(info.height * cfg.video.output_scale), expected_frames))

        # Fresh state for every job: tracker IDs, orphan IDs, evidence.
        self.vehicle_detector.reset()
        self.plate_detector.reset()
        store = EvidenceStore(cfg)
        orphan_tracker = IoUTracker(cfg.tracking.orphan_iou_threshold, cfg.tracking.orphan_max_age_frames)
        track_ids: set = set()
        stats = PipelineStats(frames_total=info.frame_count, frame_stride=cfg.video.frame_stride)
        if cfg.video.frame_stride > 1:
            logger.warning("frame_stride=%d: detection/OCR run on every %dth frame only, so fewer OCR "
                           "observations are collected per vehicle.", cfg.video.frame_stride, cfg.video.frame_stride)

        scale = cfg.video.output_scale
        out_size = (max(2, int(round(info.width * scale))), max(2, int(round(info.height * scale))))
        out_fps = cfg.video.output_fps or info.fps
        raw_video = run_dir / "annotated_raw.mp4"
        final_video = run_dir / "annotated.mp4"

        cap = cv2.VideoCapture(str(video_path))
        state = FrameState()
        frame_states: List[FrameState] = []
        timeline: List[Dict[str, Any]] = []
        consecutive_errors = 0
        idx = 0
        t_start = time.perf_counter()
        try:
            if not cap.isOpened():
                raise VideoError(f"OpenCV could not open video: {video_path}")
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    stats.cancelled = True
                    logger.warning("Cancelled at frame %d; exporting partial results.", idx)
                    break
                if cfg.video.max_frames is not None and idx >= cfg.video.max_frames:
                    break
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                stats.frames_read += 1
                ts = idx / info.fps
                sample_for_plates = idx % cfg.video.frame_stride == 0
                vehicles_before, plates_before = stats.vehicle_detections, stats.plate_detections
                try:
                    state = self._process_frame(frame, idx, ts, store, orphan_tracker, track_ids, stats,
                                                detect_plates=sample_for_plates)
                    if sample_for_plates:
                        stats.frames_processed += 1
                    consecutive_errors = 0
                except (VideoError, MemoryError):
                    raise
                except Exception as exc:
                    # Never render a previous frame's detections after a failed
                    # detector call; an empty overlay is safer than stale boxes.
                    state = FrameState()
                    stats.frame_errors += 1
                    consecutive_errors += 1
                    logger.exception("Frame %d failed: %s", idx, exc)
                    if consecutive_errors > cfg.video.max_consecutive_frame_errors:
                        raise PipelineError(
                            f"Aborting: {consecutive_errors} consecutive frame failures. Last error: {exc}") from exc
                frame_states.append(state)
                timeline.append({"frame": idx, "timestamp_s": round(ts, 3),
                                 "vehicle_detections": stats.vehicle_detections - vehicles_before,
                                 "plate_detections": stats.plate_detections - plates_before,
                                 "ocr_observations": stats.ocr_observations})
                elapsed = max(time.perf_counter() - t_start, 1e-9)
                fps_now = stats.frames_processed / elapsed
                idx += 1
                if progress is not None:
                    progress(idx, info.frame_count or idx, fps_now)
        finally:
            cap.release()

        if stats.frames_read == 0:
            raise VideoError(f"No frames could be read from {video_path}")

        # Final crop pass: recover evidence for uncertain/no-reading instances.
        for instance in store.instances:
            if instance.best_crop is None:
                continue
            if instance.consensus is not None and instance.consensus.status == PlateStatus.CONFIRMED:
                continue
            extra = self.ocr_runner.read_plate(
                instance.best_crop,
                instance.best_frame,
                instance.best_ts,
                instance.det_conf_max,
            )
            if extra:
                store.add_observations(instance, instance.best_frame, extra,
                                       instance.best_crop, instance.best_crop_score, instance.best_ts)

        store.finalize()

        # Render only after every track has a final consensus. This prevents early
        # frames from flashing temporary NO READING/UNCERTAIN labels before a later
        # clear plate crop confirms the text.
        if cfg.output.save_video:
            render_cap = cv2.VideoCapture(str(video_path))
            render_writer = None
            try:
                if not render_cap.isOpened():
                    raise VideoError(f"OpenCV could not reopen video for final annotation: {video_path}")
                render_writer = create_writer(raw_video, out_fps, out_size)
                confirmed = {
                    inst.key: inst.consensus.text
                    for inst in store.instances
                    if inst.consensus is not None
                    and inst.consensus.status == PlateStatus.CONFIRMED
                    and inst.consensus.text
                }
                for frame_state in frame_states:
                    ok, frame = render_cap.read()
                    if not ok or frame is None:
                        raise VideoError("Could not reread all frames for final video annotation")
                    plates = [(bbox, confirmed.get(key))
                              for bbox, key in frame_state.plates]
                    annotated = draw_frame(frame, frame_state.vehicles, plates, [])
                    if out_size != (info.width, info.height):
                        annotated = cv2.resize(annotated, out_size, interpolation=cv2.INTER_AREA)
                    render_writer.write(annotated)
            finally:
                render_cap.release()
                if render_writer is not None:
                    render_writer.release()

        elapsed = max(time.perf_counter() - t_start, 1e-9)
        stats.processing_fps = round(stats.frames_processed / elapsed, 2)

        crop_files = exporters.save_crops(run_dir / "crops", store.instances) if cfg.output.save_crops else {}
        records = build_records(store, cfg, crop_files)
        c = store.counts()
        stats.vehicle_tracks = len(track_ids)
        stats.plate_instances = c["plate_instances"]
        stats.ocr_observations = c["ocr_observations"]
        stats.recognition_records = len(records)
        stats.recognized_plates = c["recognized"]
        stats.pending_readings = c["pending"]
        stats.uncertain_readings = c["uncertain"]
        stats.no_reading = c["no_reading"]
        stats.unique_plate_texts = c["unique_texts"]

        video_meta = {"path": str(video_path), "width": info.width, "height": info.height,
                      "fps": info.fps, "frame_count_reported": info.frame_count,
                      "frames_read": stats.frames_read}
        json_path = exporters.export_json(run_dir / "results.json", records, stats, video_meta,
                                          cfg.to_dict(), timeline=timeline)
        csv_path = exporters.export_csv(run_dir / "results.csv", records)
        obs_path = exporters.export_observations_csv(run_dir / "observations.csv", records)
        video_out: Optional[Path] = None
        video_export_error: Optional[str] = None
        if cfg.output.save_video and raw_video.is_file():
            if cfg.output.transcode_h264:
                if transcode_h264(raw_video, final_video):
                    raw_video.unlink(missing_ok=True)
                    video_out = final_video
                else:
                    # Keep the valid raw result, never publish a partial H.264 file.
                    video_out = raw_video
                    video_export_error = "H.264 conversion failed; raw MP4V video retained. JSON/CSV/OCR exports are complete."
                    logger.error(video_export_error)
            else:
                video_out = raw_video
        logger.info("Done: %d frames, %d vehicle tracks, %d plate instances, %d confirmed, %d uncertain. "
                    "Output: %s", stats.frames_read, stats.vehicle_tracks, stats.plate_instances,
                    stats.recognized_plates, stats.uncertain_readings, run_dir)
        return PipelineResult(records, stats, run_dir, video_out, json_path, csv_path, obs_path,
                              cancelled=stats.cancelled, timeline=timeline,
                              video_export_error=video_export_error)

    def close(self) -> None:
        """Release OCR engines and detector models."""
        self.ocr_runner.close()
        if self.model_manager is not None:
            self.model_manager.release()
