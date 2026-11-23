# NEXUS ANPR

Automatic Number Plate Recognition on **recorded traffic video**: YOLO vehicle detection,
ByteTrack tracking, dedicated plate detector, EasyOCR / PaddleOCR, and a multi-frame
consensus that refuses to guess. Outputs an annotated MP4 plus JSON/CSV results.

## Verification status (read this first)

| Item | Status |
|---|---|
| Unit + integration tests | **128 passed, 1 skipped** after release hardening (2026-11-22) |
| Synthetic video pipeline with stub detectors/OCR | **Passed**; exercises OpenCV video I/O, tracking calls, consensus, counting, annotation, exports and cleanup. It does not measure recognition accuracy. |
| CLI and models | `main.py check` reports the configured vehicle and plate weights available in this workspace. Both checkpoints loaded and were used in real-video inference. |
| Real traffic-video inference | **Verified on supplied 250-frame and 1,632-frame runs.** Full CLI run: 10,695 vehicle detections, 261 tracks, 1,776 plate detections, 71 candidate instances, and 2,512 OCR observations at the hardened 0.20 threshold. |
| Weight checkpoint loading | **Verified:** both local YOLO checkpoints load with Ultralytics; vehicle checkpoint exposes COCO classes and plate checkpoint exposes `license_plate`. This confirms loadability, not detector accuracy. |
| `tests/test_e2e.py` | **Skipped** because its optional fixture `input/sample.mp4` is not present. The supplied real video was validated separately from its external path. |
| Streamlit dashboard | **250-frame real-video smoke workflow verified** including result/evidence/crop rendering, annotated playback, and JSON/CSV/observation downloads. A separate a previous full dashboard inference reached 1,632 frames, but its H.264 conversion failed from disk exhaustion; full dashboard playback/export comparison remains incomplete. |
| Dashboard compatibility | Timeline/table helpers now preflight PyArrow and fall back to raster/JSON for the installed NumPy/PyArrow mismatch; regression tests pass. The final browser recheck could not reconnect to Streamlit. |
| OCR accuracy | **Not measured.** No ground-truth dataset was available. The `KA03NU7762` / `KAO3NU7762` O/0 ambiguity remains manually UNCERTAIN on the exact latest-run crop. |
| PaddleOCR | **Not installed / not run.** EasyOCR was used for the real-video runs. |

Tests built on stubs prove the *mechanics* (counting, consensus, exports, resource handling). They say nothing about detection or OCR quality.

## 1. Environment setup (macOS)

```bash
brew install python@3.11 ffmpeg          # ffmpeg = H.264 re-encode so the output plays in browsers/QuickTime
cd NEXUS_ANPR
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Dependencies

```bash
pip install -r requirements.txt
python main.py check                      # shows exactly what is missing
# optional second OCR engine (riskiest install; skip if it fights you):
# pip install -r requirements-paddle.txt
```

## 3. Models (nothing is ever downloaded during a run)

Put two files in `models/weights/` (paths configurable in `config.py` / `--vehicle-weights` / `--plate-weights`):

| File | What it is | Where it comes from |
|---|---|---|
| `yolov8n.pt` | COCO vehicle detector (cars, motorcycles, buses, trucks) | Get it yourself from the official Ultralytics releases, or on a networked machine run `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` and move the file into `models/weights/`. Any COCO-class YOLO `.pt` works if you point `--vehicle-weights` at it. |
| `license_plate_detector.pt` | Dedicated YOLO model with class 0 named `license_plate` | Included at `models/weights/license_plate_detector.pt`. It loads with the installed Ultralytics version. Its provenance/license and detection accuracy have not been independently established; validate it on your labeled traffic data before operational use. If class ids differ in another model, set `detection.plate_class_ids`. |

OCR models: EasyOCR needs its recognition models once. Run (internet required, deliberate, one time):

```bash
python main.py prepare-ocr
```

Runs afterwards stay offline (`download_enabled=False`).

## 4. Command line

```bash
python main.py check
python main.py run --video input/traffic.mp4
python main.py run --video input/traffic.mp4 --device mps --engines easyocr paddleocr --stride 2
python main.py run --video input/traffic.mp4 --max-frames 300 --no-crops
python main.py run --video input/traffic.mp4 --config my_config.json     # see config.example.json
```

Each run writes `output/<video>_<timestamp>/`:

| File | Content |
|---|---|
| `annotated.mp4` | Vehicle boxes and confirmed plate boxes/text, rendered after final consensus; no uncertain or no-reading plate overlays |
| `results.json` | Summary, config snapshot, per-frame detection timeline, notes, one record per plate instance with every OCR observation |
| `results.csv` | One row per record |
| `observations.csv` | Every raw OCR read (engine, variant, frame, raw text, confidences, vote eligibility) |
| `crops/` | Best original (unprocessed) crop per plate instance |

## 5. Dashboard (optional)

```bash
streamlit run app.py
```

The root `app.py` entry point delegates to `dashboard/app.py`. See
[INTERVIEW_PREP.md](INTERVIEW_PREP.md) for the architecture walkthrough, interview Q&A,
demo plan, and limitations.

## 6. Tests

```bash
python -m pytest -q                       # or: python -m unittest discover -s tests -t .
```

The suite uses `unittest.TestCase` classes, so both runners work. `test_e2e.py` is skipped until its optional fixture `input/sample.mp4` is provided; the real supplied video was validated separately. The current release source reports 128 passed, 1 skipped. The skipped case needs optional `input/sample.mp4`.

## 7. Real end-to-end check (do this first on your Mac)

```bash
cp /path/to/your_traffic_clip.mp4 input/sample.mp4     # a short clip, 10-30 s, plates reasonably visible
python main.py check
python -m pytest tests/test_e2e.py -q -s                # runs 150 frames by default (NEXUS_E2E_FRAMES=...)
python main.py run --video input/sample.mp4 --max-frames 300
open output/                                            # inspect annotated.mp4, results.csv
```

## 8. Design decisions

* **No character substitution.** `O`/`0`, `S`/`5`, `B`/`8`, `I`/`1`, `Z`/`2`, and `G`/`6` are never swapped. Visually confusable candidate disagreement keeps the record UNCERTAIN.
* **Whole-string voting**, one vote per distinct text per frame (best OCR confidence across engines/variants), weighted by `ocr_conf x plate_det_conf`.
* **CONFIRMED** needs an Indian registration format, at least `min_support_frames` exact agreeing frames, OCR confidence above the configured threshold, `min_weight_share` of total weight and `min_margin_share` over the runner-up (all in `config.py`). Confusable character conflicts remain `UNCERTAIN`; uncertain guesses are shown as candidates for review and never promoted to final readings.
* **Identity = vehicle track, not text.** Two vehicles with the same plate text give two records and one unique text.
* **Plates are never silently dropped:** unmatched plates become "unassociated" instances tracked by an IoU tracker.
* **Adaptive preprocessing:** `color` + `clahe` first; `sharpen`, `bilateral`, `otsu`, `adaptive`, `perspective` only when the best read is weak. EasyOCR is the primary engine; Tesseract adds an optional second opinion when its executable is installed, and PaddleOCR can be configured separately.
* **Early OCR stop** per track after confirmation (`stop_ocr_after_confirmed_frames`) and a hard cap (`max_ocr_frames_per_track`) for speed.
* **Stride:** every frame is read and written; vehicle detection/tracking refreshes on every frame, while plate detection/OCR run on every Nth frame. Skipped frames do not reuse old vehicle or plate boxes. Fewer plate/OCR opportunities result, and a warning is logged.
* **Per-job state:** tracker, orphan tracker and evidence store are recreated for each `run()`.

## 9. Known limitations

* If ByteTrack changes a vehicle's ID after occlusion, one vehicle can yield two records (never merged by text, by design).
* A plate first seen with no vehicle and later inside a tracked vehicle becomes two instances.
* Plate detection runs on the full frame (default confidence `0.20`, `imgsz=1280`). The reviewed threshold sample favored `0.20` for fewer false boxes than `0.10`/`0.15`, while also losing some genuine low-confidence sightings; this is a tradeoff, not measured accuracy. Very small/far plates may be missed. Cropping per vehicle is a possible improvement.
* `confidence` is a heuristic (winner's weight share x mean OCR confidence), not a calibrated probability.
* The optional Indian-format check is informational only; the character allowlist defaults to `A-Z0-9`.
* Timestamps are `frame_index / fps`.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `vehicle weights not found at ...` | Place the `.pt` file at that path (section 3) or pass `--vehicle-weights` |
| `COCO model cannot detect plates` message | You need a real plate-detection model |
| `EasyOCR could not start ... prepare-ocr` | Run `python main.py prepare-ocr` once with internet |
| ByteTrack / `lap` import error | `pip install lapx` |
| MPS error mid-run | Handled: the run falls back to CPU automatically; or use `--device cpu` |
| Annotated video won't play in the browser | Install ffmpeg (`brew install ffmpeg`); without it the file is mp4v-encoded |
| `pip` resolver conflicts | Relax the failing pin in `requirements.txt`; PaddleOCR is the usual culprit, so drop it |
| Slow on CPU | `--stride 2`, `--max-frames`, one OCR engine, lower `detection.plate_imgsz` |

## 11. OS and device notes

Create a Python 3.11 virtual environment on macOS, Windows, or Linux. Install the matching PyTorch build for the target OS/hardware using the official PyTorch install selector first; CUDA wheel choices vary by driver/runtime. Then install the remaining project requirements. Apple Silicon can use MPS when supported by the installed PyTorch build; `auto` selects an available device and MPS fallback is enabled. CPU is the portable option. Confirm the actual selection and missing components with `python main.py check`. PaddleOCR remains optional and has a separate dependency file; install it only in an environment where its PaddlePaddle requirements resolve.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the OS/device-matched PyTorch + torchvision build first.
python -m pip install -r requirements.txt
python main.py check
```

See [FINAL_REAL_VIDEO_TEST_REPORT.md](FINAL_REAL_VIDEO_TEST_REPORT.md), [AUDIT_REPORT.md](AUDIT_REPORT.md), [CHANGELOG.md](CHANGELOG.md), and [REAL_VIDEO_VALIDATION.md](REAL_VIDEO_VALIDATION.md) for exact run evidence, audit findings, changes, and reproduction steps.
