"""NEXUS ANPR Streamlit dashboard. Run with ``streamlit run dashboard/app.py``."""
from __future__ import annotations

import os
import sys
import json
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from config import PROJECT_ROOT, Config
from dashboard.job_utils import (effective_frame_total, list_saved_runs, load_saved_run,
                                 matching_manual_review, render_data_table, render_timeline_chart)
from models.model_manager import ModelError
from pipeline.anpr_pipeline import ANPRPipeline, PipelineError
from recognition.ocr_engine import OCRUnavailableError
from records import PlateStatus
from utils.logging_setup import setup_logging
from utils.video_utils import (SUPPORTED_SUFFIXES, VideoError, ensure_output_space,
                               estimated_video_space_bytes, probe_video)

st.set_page_config(page_title="NEXUS ANPR", page_icon="🚘", layout="wide")
setup_logging(PROJECT_ROOT / "logs")
st.markdown("""
<style>
.stApp {background:#0b1220;color:#e5edf7}
[data-testid="stSidebar"] {background:#101a2b;border-right:1px solid #24344b}
[data-testid="stMetric"] {background:#121f33;border:1px solid #263a56;padding:16px 18px;border-radius:12px}
[data-testid="stMetricLabel"] {color:#9db0c9}
.hero {padding:24px 28px;border:1px solid #263a56;border-radius:16px;background:linear-gradient(120deg,#13253d,#111827);margin-bottom:18px}
.hero h1 {margin:0;color:#f3f7fc;font-size:1.8rem}.hero p {margin:8px 0 0;color:#a9bbd2}
.section {border-bottom:1px solid #26364d;padding-bottom:6px;margin-top:24px}
.plate-card {background:#111e31;border:1px solid #293c57;border-radius:12px;padding:16px;margin:10px 0}
.confirmed {color:#5de0a2;font-weight:700}.uncertain {color:#ffbd69;font-weight:700}
</style>
<div class="hero"><h1>NEXUS ANPR</h1><p>Intelligent number plate recognition · recorded video analysis</p></div>
""", unsafe_allow_html=True)

defaults = Config()
with st.sidebar:
    st.markdown("### Control room")
    section = st.radio("Workspace", ["Overview", "Process video", "Plate results", "System status"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("#### Models and runtime")
    vehicle_w = st.text_input("Vehicle YOLO weights", str(defaults.models.vehicle_weights))
    plate_w = st.text_input("Plate YOLO weights", str(defaults.models.plate_weights))
    device = st.selectbox("Compute device", ["auto", "cpu", "cuda", "mps"], index=0)
    engines = st.multiselect("OCR engines", ["easyocr", "paddleocr"], default=["easyocr"])
    st.markdown("#### Detection settings")
    vconf = st.slider("Vehicle confidence", 0.05, 0.95, defaults.detection.vehicle_conf, 0.05)
    pconf = st.slider("Plate confidence", 0.05, 0.95, defaults.detection.plate_conf, 0.05)
    ocr_conf = st.slider("Minimum OCR vote confidence", 0.0, 1.0, defaults.consensus.min_vote_confidence, 0.05)
    stride = st.slider("Plate/OCR frame stride", 1, 10, defaults.video.frame_stride)
    max_frames = st.number_input("Maximum frames (0 = all)", 0, 2_000_000, 0, step=100)

cfg = Config()
cfg.models.vehicle_weights, cfg.models.plate_weights = Path(vehicle_w).expanduser(), Path(plate_w).expanduser()
cfg.models.device, cfg.ocr.engines = device, tuple(engines)
cfg.detection.vehicle_conf, cfg.detection.plate_conf = vconf, pconf
cfg.video.frame_stride, cfg.video.max_frames = int(stride), int(max_frames) or None
cfg.consensus.min_vote_confidence = ocr_conf

def package_status(name: str) -> bool:
    try:
        from importlib.util import find_spec
        return find_spec(name) is not None
    except (ImportError, ValueError):
        return False

def process_job(job: dict) -> None:
    """Run the shared pipeline in a worker; UI reads progress from this job object."""
    pipeline = None
    try:
        pipeline = ANPRPipeline.from_config(job["config"])
        def on_progress(done: int, total: int, fps: float) -> None:
            with job["lock"]:
                job["done"] = done
                job["total"] = effective_frame_total(total, job["config"].video.max_frames)
                job["fps"] = fps
        result = pipeline.run(job["video"], progress=on_progress, cancel_event=job["cancel"])
        with job["lock"]:
            job["result"] = result
            job["state"] = "complete_with_video_error" if result.video_export_error else "complete"
    except Exception as exc:
        with job["lock"]:
            job["error"], job["state"] = str(exc), "error"
    finally:
        if pipeline is not None:
            pipeline.close()

status_rows = [
    ("Vehicle YOLO weights", "AVAILABLE" if cfg.models.vehicle_weights.is_file() else "NOT CONFIGURED", str(cfg.models.vehicle_weights)),
    ("Dedicated plate YOLO weights", "AVAILABLE" if cfg.models.plate_weights.is_file() else "NOT CONFIGURED", str(cfg.models.plate_weights)),
    ("Ultralytics / ByteTrack runtime", "AVAILABLE" if package_status("ultralytics") else "UNAVAILABLE", "Loaded and exercised when processing starts"),
    ("EasyOCR", "AVAILABLE" if package_status("easyocr") else "UNAVAILABLE", "Lazy initialization during preflight"),
    ("PaddleOCR", "AVAILABLE" if package_status("paddleocr") else "UNAVAILABLE", "Optional engine"),
    ("PyTorch", "AVAILABLE" if package_status("torch") else "UNAVAILABLE", f"Requested device: {device}"),
]

st.markdown(f"### {section}")
if section in ("Overview", "Process video"):
    upload = st.file_uploader("Upload a recorded traffic video", type=[s.lstrip(".") for s in SUPPORTED_SUFFIXES])
    if upload:
        if Path(upload.name).suffix.lower() not in SUPPORTED_SUFFIXES:
            st.error("Unsupported video format.")
        else:
            temp_root = Path(tempfile.gettempdir()) / "nexus_anpr_uploads"
            temp_root.mkdir(parents=True, exist_ok=True)
            input_path = temp_root / upload.name
            input_path.write_bytes(upload.getbuffer())
            try:
                info = probe_video(input_path)
                duration = info.frame_count / info.fps if info.frame_count else 0
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Video", upload.name)
                m2.metric("Duration", f"{duration:.1f} s" if duration else "Unknown")
                m3.metric("Frame rate", f"{info.fps:.2f} FPS")
                m4.metric("Resolution", f"{info.width} × {info.height}")
                with st.expander("Preview source video"):
                    st.video(upload.getvalue())
                active_job = st.session_state.get("job")
                job_running = False
                if active_job is not None:
                    with active_job["lock"]:
                        job_running = active_job["state"] == "running"
                if job_running:
                    st.info("A video is already processing. Wait for it to finish or cancel it before starting another.")
                if st.button("Start processing", type="primary", disabled=not engines or job_running,
                             use_container_width=True):
                    try:
                        nframes = effective_frame_total(info.frame_count, cfg.video.max_frames)
                        ensure_output_space(cfg.output.output_dir, estimated_video_space_bytes(
                            int(info.width * cfg.video.output_scale), int(info.height * cfg.video.output_scale), nframes))
                        job = {"config": cfg, "video": input_path, "cancel": threading.Event(),
                               "lock": threading.Lock(), "state": "running", "done": 0,
                               "total": nframes, "fps": 0.0, "result": None, "error": None}
                        st.session_state["job"] = job
                        threading.Thread(target=process_job, args=(job,), daemon=True).start()
                    except VideoError as exc:
                        st.error(str(exc))
            except VideoError as exc:
                st.error(f"This file could not be opened as video: {exc}")
    else:
        st.info("Upload a supported recorded video to inspect its metadata and start a processing job.")

result = st.session_state.get("result")
job = st.session_state.get("job")

# Saved-run selector: lets the dashboard switch between multiple processed videos.
saved_runs = list_saved_runs(cfg.output.output_dir)
if saved_runs:
    run_labels = {str(run_dir): f"{run_dir.name}" for run_dir in saved_runs}
    selected_run = st.sidebar.selectbox(
        "Saved run",
        options=[str(run_dir) for run_dir in saved_runs],
        format_func=lambda value: run_labels[value],
        key="selected_run",
    )
    try:
        if selected_run:
            result = load_saved_run(Path(selected_run))
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        st.error(f"Could not load selected run: {exc}")

if job is not None:
    @st.fragment(run_every=0.75)
    def job_monitor() -> None:
        with job["lock"]:
            state, done, total, fps = job["state"], job["done"], job["total"], job["fps"]
            error, job_result = job["error"], job["result"]
        if state == "running":
            st.markdown("#### Processing job")
            st.progress(min(done / total, 1.0) if total else 0.0)
            st.caption(f"Frame {done:,} / {total or 'unknown'} · {fps:.1f} processing FPS")
            if st.button("Cancel processing", type="secondary"):
                job["cancel"].set()
                st.info("Cancellation requested. The pipeline will export partial results safely.")
        elif state in ("complete", "complete_with_video_error"):
            if job_result is not None and not job.get("saved_to_session"):
                st.session_state["result"] = job_result
                st.session_state["selected_run"] = str(job_result.run_dir)
                st.session_state["runtime_ready"] = True
                job["saved_to_session"] = True
                if job_result.video_export_error:
                    st.error(job_result.video_export_error)
                    st.success("Inference and JSON/CSV evidence completed; video export did not produce H.264.")
                else:
                    st.success("Processing complete. Results are available below.")
                st.rerun()
            if job_result is not None and job_result.cancelled:
                st.warning("Processing was cancelled; partial results are available below.")
        elif state == "error":
            st.session_state["runtime_ready"] = False
            st.error(f"Processing could not complete: {error}")
    job_monitor()

if result is not None and section in ("Overview", "Process video", "Plate results"):
    s = result.stats
    manual_review_path = PROJECT_ROOT / "validation_output" / "final_review" / "plate" / "manual_plate_review.json"
    manual_reviews = {}
    for item in result.records:
        item_crop = result.run_dir / item.crop_file if item.crop_file else None
        review = matching_manual_review(item_crop, manual_review_path) if item_crop else None
        if review:
            manual_reviews[item.plate_instance_id] = review
    st.markdown('<div class="section"><h3>Processing overview</h3></div>', unsafe_allow_html=True)
    metrics = [("Frames processed", s.frames_processed), ("Vehicle detections", s.vehicle_detections),
               ("Vehicle tracks", s.vehicle_tracks), ("Plate detections", s.plate_detections),
               ("Plate instances", s.plate_instances), ("Unique confirmed texts", s.unique_plate_texts),
               ("Confirmed", s.recognized_plates), ("Uncertain", s.uncertain_readings),
               ("No reading", s.no_reading)]
    for start in range(0, len(metrics), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, metrics[start:start + 4]):
            col.metric(label, f"{value:,}")
    st.caption(f"Frames read: {s.frames_read:,} · Plate/OCR sample frames: {s.frames_processed:,} · OCR observations: {s.ocr_observations:,} · Raw plate boxes excluded by geometry: {s.rejected_plate_geometry:,} · Processing: {s.processing_fps:.1f} FPS · Stride: {s.frame_stride}")
    if result.timeline:
        import pandas as pd
        chart = pd.DataFrame(result.timeline).set_index("frame")
        st.markdown("#### Detection activity by frame")
        render_timeline_chart(st, chart[["vehicle_detections", "plate_detections"]])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Annotated video")
        uncertain_video_records = [r for r in result.records
                                   if manual_reviews.get(r.plate_instance_id, {}).get("manual_status") == "uncertain"]
        if uncertain_video_records:
            ids = ", ".join(f"{r.plate_instance_id:04d}" for r in uncertain_video_records)
            st.warning(f"This video retains algorithmic consensus overlays. Manual review marks plate instance(s) {ids} UNCERTAIN; do not treat overlay text as verified ground truth.")
        if result.video_export_error:
            st.error(result.video_export_error)
            if result.video_path and result.video_path.is_file():
                st.download_button("Download raw annotated MP4V", result.video_path.read_bytes(),
                                   result.video_path.name, "video/mp4")
            st.caption("Browser playback is unavailable because a validated H.264 export was not produced.")
        elif result.video_path and result.video_path.is_file():
            video_bytes = result.video_path.read_bytes()
            st.video(video_bytes)
            st.download_button("Download annotated video", video_bytes, result.video_path.name, "video/mp4")
        else:
            st.info("Annotated video output is disabled or unavailable for this run.")
    with right:
        st.markdown("#### Recognition status")
        st.json({"Confirmed": s.recognized_plates, "Uncertain": s.uncertain_readings,
                 "No reading": s.no_reading, "Unique plate texts": s.unique_plate_texts,
                 "Separate plate instances": s.plate_instances})
        st.caption("Every plate instance remains a separate result, including identical confirmed text.")

    st.markdown('<div class="section"><h3>Final number plate readings</h3></div>', unsafe_allow_html=True)
    confirmed_texts = {r.final_text for r in result.records if r.status == PlateStatus.CONFIRMED.value and r.final_text}
    seen_confirmed_texts = set()
    for record in result.records:
        if record.status == PlateStatus.NO_READING.value:
            continue
        if record.status != PlateStatus.CONFIRMED.value and record.best_guess_text in confirmed_texts:
            continue
        if record.status == PlateStatus.CONFIRMED.value and record.final_text:
            if record.final_text in seen_confirmed_texts:
                continue
            seen_confirmed_texts.add(record.final_text)
        crop_path = result.run_dir / record.crop_file if record.crop_file else None
        manual_review = manual_reviews.get(record.plate_instance_id)
        manually_uncertain = bool(manual_review and manual_review.get("manual_status") == "uncertain")
        status = "MANUAL REVIEW UNCERTAIN" if manually_uncertain else record.status.upper()
        shown = ("MANUAL REVIEW: UNCERTAIN" if manually_uncertain else
                 record.final_text or (f"UNCERTAIN — {record.best_guess_text}"
                                       if record.best_guess_text else "UNCERTAIN"))
        css = "uncertain" if manually_uncertain or record.status != PlateStatus.CONFIRMED.value else "confirmed"
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**Instance {record.plate_instance_id:04d}** · Vehicle track {record.vehicle_track_id if record.vehicle_track_id is not None else 'unassociated'}")
            b.markdown(f'<span class="{css}">{status}</span>', unsafe_allow_html=True)
            st.markdown(f"### `{shown}`")
            if manually_uncertain:
                st.caption(f"Algorithmic consensus: {record.status.upper()}")
                st.warning(f"Manual review: UNCERTAIN · {manual_review.get('ambiguous_character', 'character ambiguity')}. No ground truth available.")
                if manual_review.get("disagreement_positions_1based"):
                    st.caption("Crop disagreement positions (1-based): " + ", ".join(map(str, manual_review["disagreement_positions_1based"])))
                st.caption("Pipeline support frames: " + ", ".join(map(str, manual_review.get("supporting_pipeline_frames_for_KA03NU7762", []))))
            st.caption(f"Plate detection confidence max {record.detection_confidence_max:.3f} · agreeing frames {record.support_frames} · first seen {record.first_timestamp_s:.2f}s")
            if crop_path and crop_path.is_file():
                st.image(str(crop_path), caption=f"Original plate crop · frame {record.best_crop_frame}", width=420)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download JSON results", result.json_path.read_bytes(), result.json_path.name, "application/json")
    d2.download_button("Download CSV results", result.csv_path.read_bytes(), result.csv_path.name, "text/csv")
    d3.download_button("Download OCR observations", result.observations_csv_path.read_bytes(), result.observations_csv_path.name, "text/csv")
    st.caption(f"Run output: {result.run_dir} · Per-observation CSV: {result.observations_csv_path.name}")

if section == "System status":
    st.markdown("Actual model loading occurs during pipeline preflight; file presence alone is not reported as READY.")
    for name, state, detail in status_rows:
        st.write(f"**{state}** · {name} — {detail}")
    st.write("**Pipeline runtime** · " + ("READY (preflight passed in the latest completed job)" if st.session_state.get("runtime_ready") else "NOT INITIALIZED"))
    st.info("A dedicated license plate YOLO model is required. The general COCO vehicle model cannot detect plates. Add the model at the configured path or choose another .pt file in the sidebar.")
