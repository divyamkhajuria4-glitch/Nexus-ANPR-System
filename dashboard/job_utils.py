"""Small helpers for displaying dashboard job progress accurately."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def _arrow_available() -> bool:
    """Return whether Streamlit's Arrow-backed widgets can initialize."""
    try:
        import pyarrow  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def effective_frame_total(source_total: int, max_frames: int | None) -> int:
    """Return the number of frames the current dashboard job will process."""
    if source_total <= 0:
        return max_frames or source_total
    if max_frames is not None and max_frames > 0:
        return min(source_total, max_frames)
    return source_total


def render_timeline_chart(st, chart_data) -> bool:
    """Render an interactive chart or an OpenCV raster fallback without Arrow."""
    if _arrow_available():
        try:
            st.line_chart(chart_data, height=220)
            return True
        except (ImportError, OSError):
            pass
    try:
        import cv2
        import numpy as np

        width, height = 960, 260
        left, right, top, bottom = 52, width - 24, 36, height - 34
        canvas = np.full((height, width, 3), (18, 31, 51), dtype=np.uint8)
        series = (("vehicle_detections", (230, 160, 40)), ("plate_detections", (60, 200, 60)))
        values = {name: [float(v) for v in chart_data[name].tolist()] for name, _ in series}
        peak = max((v for values_for_line in values.values() for v in values_for_line), default=1.0) or 1.0
        cv2.rectangle(canvas, (left, top), (right, bottom), (80, 96, 118), 1)
        for step in range(1, 4):
            y = top + (bottom - top) * step // 4
            cv2.line(canvas, (left, y), (right, y), (46, 63, 84), 1)
        for (name, color), line in zip(series, (values[n] for n, _ in series)):
            if not line:
                continue
            denom = max(len(line) - 1, 1)
            points = [(left + i * (right - left) // denom,
                       bottom - int(v / peak * (bottom - top))) for i, v in enumerate(line)]
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Vehicle detections", (left, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    series[0][1], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Plate detections", (left + 220, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    series[1][1], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Frame index", (right - 105, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (205, 218, 232), 1, cv2.LINE_AA)
        st.image(canvas, channels="BGR", caption="Detection activity by frame (static fallback)",
                 use_container_width=True)
    except (ImportError, OSError):
        st.warning("Detection timeline unavailable: install compatible PyArrow and NumPy versions.")
    return True


def render_data_table(st, rows, **kwargs) -> bool:
    """Render tabular evidence, falling back to JSON if Arrow cannot load."""
    if not _arrow_available():
        st.json(rows)
        return False
    try:
        st.dataframe(rows, **kwargs)
    except (ImportError, OSError):
        st.json(rows)
        return False
    return True


def matching_manual_review(crop_path: Path, review_path: Path) -> dict | None:
    """Load a manual disposition only when the reviewed crop bytes match."""
    try:
        payload = json.loads(review_path.read_text(encoding="utf-8"))
        reviews = payload.get("reviews", [payload])
        if not crop_path.is_file():
            return None
        actual = hashlib.sha256(crop_path.read_bytes()).hexdigest()
        return next((review for review in reviews if review.get("crop_sha256") == actual), None)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def list_saved_runs(output_dir: Path) -> list[Path]:
    """List completed runs newest first."""
    if not output_dir.is_dir():
        return []
    return sorted(
        (path for path in output_dir.iterdir()
         if path.is_dir() and (path / "results.json").is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def load_saved_run(run_dir: Path):
    """Reconstruct a PipelineResult from exported results.json."""
    from dataclasses import fields
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline.anpr_pipeline import PipelineResult
    from records import PipelineStats, RecognitionRecord
    from recognition.text_processor import normalize_text

    payload = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    record_fields = {field.name for field in fields(RecognitionRecord)}
    records = []
    for item in payload.get("records", []):
        record_data = {key: value for key, value in item.items() if key in record_fields}
        for key in ("final_text", "best_guess_text"):
            if record_data.get(key):
                record_data[key] = normalize_text(str(record_data[key]))
        for candidate in record_data.get("candidates", []):
            if candidate.get("text"):
                candidate["text"] = normalize_text(str(candidate["text"]))
        for observation in record_data.get("observations", []):
            normalized = observation.get("normalized_text") or observation.get("raw_text")
            if normalized:
                observation["normalized_text"] = normalize_text(str(normalized))
        records.append(RecognitionRecord(**record_data))

    stats_fields = {field.name for field in fields(PipelineStats)}
    summary = payload.get("summary", {})
    stats = PipelineStats(**{key: value for key, value in summary.items() if key in stats_fields})

    video_path = run_dir / "annotated.mp4"
    if not video_path.is_file():
        video_path = run_dir / "annotated_raw.mp4"
    if not video_path.is_file():
        video_path = None

    return PipelineResult(
        records=records,
        stats=stats,
        run_dir=run_dir,
        video_path=video_path,
        json_path=run_dir / "results.json",
        csv_path=run_dir / "results.csv",
        observations_csv_path=run_dir / "observations.csv",
        cancelled=bool(stats.cancelled),
        timeline=payload.get("timeline", []),
    )
