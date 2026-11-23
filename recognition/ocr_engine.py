"""OCR engines (EasyOCR, PaddleOCR) and the runner that feeds them preprocessing variants.

Engines load lazily and independently; a failure in one never stops the other.
No character "correction" happens here - raw output is preserved.
"""
from __future__ import annotations

import logging
import os
import csv
import io
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from config import Config
from records import OCRObservation
from recognition.preprocessing import build_variants, correct_perspective, ensure_bgr
from recognition.text_processor import TextFragment, merge_fragments, normalize_text

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    """An OCR engine cannot be initialised (missing package or missing local models)."""


class OCREngine(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self._loaded = False

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _read(self, image_bgr: np.ndarray) -> List[TextFragment]: ...

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    def read(self, image: np.ndarray) -> tuple[str, float]:
        """Return ``(raw_text, confidence)`` for one image variant."""
        self.ensure_loaded()
        return merge_fragments(self._read(ensure_bgr(image)))

    def close(self) -> None:
        self._loaded = False


def _frag(points: Any, text: str, conf: float) -> TextFragment:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    return TextFragment(text=str(text), confidence=float(conf),
                        cx=float(pts[:, 0].mean()), cy=float(pts[:, 1].mean()),
                        height=float(max(pts[:, 1].max() - pts[:, 1].min(), 1.0)))


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, config: Config, use_gpu: bool = False) -> None:
        super().__init__()
        self._cfg = config
        self._gpu = use_gpu
        self._reader: Any = None

    def _load(self) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise OCRUnavailableError("easyocr is not installed (pip install -r requirements.txt).") from exc
        o = self._cfg.ocr
        kwargs: Dict[str, Any] = dict(gpu=self._gpu, verbose=False, download_enabled=o.allow_model_download)
        if o.easyocr_model_dir:
            kwargs["model_storage_directory"] = str(o.easyocr_model_dir)
            kwargs["user_network_directory"] = str(Path(o.easyocr_model_dir) / "user_network")
        try:
            self._reader = easyocr.Reader(list(o.languages), **kwargs)
        except Exception as exc:
            raise OCRUnavailableError(
                f"EasyOCR could not start: {exc}\nIts recognition models are not downloaded automatically. "
                "Run once with internet access:  python main.py prepare-ocr"
            ) from exc

    def _read(self, image_bgr: np.ndarray) -> List[TextFragment]:
        allow = self._cfg.ocr.allowlist or None
        # Paragraph mode returns (box, text) pairs without recognition scores in
        # EasyOCR; keep line mode so confidence is available for consensus voting.
        results = self._reader.readtext(image_bgr, detail=1, paragraph=False, allowlist=allow)
        return [_frag(box, text, conf) for box, text, conf in results]

    def close(self) -> None:
        self._reader = None
        super().close()


class TesseractEngine(OCREngine):
    """Optional local Tesseract OCR engine; it never downloads models."""
    name = "tesseract"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._cfg = config
        self._binary: Optional[str] = None

    def _load(self) -> None:
        self._binary = shutil.which("tesseract")
        if not self._binary:
            raise OCRUnavailableError("Tesseract executable is not installed or not on PATH.")

    def _read(self, image_bgr: np.ndarray) -> List[TextFragment]:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise RuntimeError("Could not encode plate crop for Tesseract")
        allow = self._cfg.ocr.allowlist or "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        proc = subprocess.run(
            [self._binary or "tesseract", "stdin", "stdout", "--psm", "7",
             "-c", f"tessedit_char_whitelist={allow}", "tsv"],
            input=encoded.tobytes(), capture_output=True, timeout=20, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip()
                               or "Tesseract failed to read plate crop")
        rows = csv.DictReader(io.StringIO(proc.stdout.decode("utf-8", errors="replace")),
                              delimiter="\t")
        fragments: List[TextFragment] = []
        for row in rows:
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf", "-1"))
                left, top = float(row["left"]), float(row["top"])
                width, height = float(row["width"]), float(row["height"])
            except (KeyError, TypeError, ValueError):
                continue
            if not text or confidence < 0:
                continue
            points = [[left, top], [left + width, top],
                      [left + width, top + height], [left, top + height]]
            fragments.append(_frag(points, text, confidence / 100.0))
        return fragments


def _paddle_models_present() -> bool:
    home = Path(os.path.expanduser("~"))
    return any((home / d).exists() for d in (".paddleocr", ".paddlex"))


class PaddleOCREngine(OCREngine):
    """Supports PaddleOCR 2.x (``.ocr``) and 3.x (``.predict``) result formats."""
    name = "paddleocr"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self._cfg = config
        self._ocr: Any = None
        self._v3 = False

    def _load(self) -> None:
        try:
            import paddleocr
        except ImportError as exc:
            raise OCRUnavailableError("paddleocr is not installed (see requirements-paddle.txt).") from exc
        if not self._cfg.ocr.allow_model_download and not _paddle_models_present():
            raise OCRUnavailableError(
                "PaddleOCR models are not on disk and automatic downloads are disabled. "
                "Run once with internet access:  python main.py prepare-ocr --engines paddleocr")
        lang = self._cfg.ocr.languages[0]
        try:
            major = int(str(getattr(paddleocr, "__version__", "2")).split(".")[0])
            self._v3 = major >= 3
            if self._v3:
                self._ocr = paddleocr.PaddleOCR(lang=lang, use_doc_orientation_classify=False,
                                                use_doc_unwarping=False, use_textline_orientation=False)
            else:
                self._ocr = paddleocr.PaddleOCR(lang=lang, use_angle_cls=False, show_log=False)
        except Exception as exc:
            raise OCRUnavailableError(f"PaddleOCR could not start: {exc}") from exc

    def _read(self, image_bgr: np.ndarray) -> List[TextFragment]:
        raw = self._ocr.predict(image_bgr) if self._v3 else self._ocr.ocr(image_bgr, cls=False)
        return parse_paddle_output(raw)

    def close(self) -> None:
        self._ocr = None
        super().close()


def parse_paddle_output(raw: Any) -> List[TextFragment]:
    """Handle both PaddleOCR 3.x (dict-like with rec_texts) and 2.x (nested lists)."""
    frags: List[TextFragment] = []
    if raw is None:
        return frags
    for page in raw:
        if page is None:
            continue
        if isinstance(page, Mapping) and "rec_texts" in page:
            texts, scores = page["rec_texts"], page.get("rec_scores", [])
            polys = page.get("rec_polys")
            if polys is None:
                polys = page.get("dt_polys", [])
            for i, text in enumerate(texts):
                poly = polys[i] if i < len(polys) else [[0, 0], [1, 0], [1, 1], [0, 1]]
                frags.append(_frag(poly, text, scores[i] if i < len(scores) else 0.0))
        elif isinstance(page, (list, tuple)):
            for line in page:
                try:
                    points, (text, conf) = line
                except (TypeError, ValueError):
                    continue
                frags.append(_frag(points, text, conf))
    return frags


def build_engines(config: Config, device: str = "cpu") -> List[OCREngine]:
    engines: List[OCREngine] = []
    for name in config.ocr.engines:
        if name == "easyocr":
            engines.append(EasyOCREngine(config, use_gpu=config.ocr.easyocr_use_gpu))
        elif name == "paddleocr":
            engines.append(PaddleOCREngine(config))
        elif name == "tesseract":
            engines.append(TesseractEngine(config))
    return engines


class OCRRunner:
    """Runs every active engine over preprocessing variants of one plate crop."""

    def __init__(self, engines: Sequence[OCREngine], config: Config) -> None:
        self._engines = list(engines)
        self._cfg = config
        self._disabled: set[str] = set()
        self.error_counts: Dict[str, int] = {}

    @property
    def active_engines(self) -> List[OCREngine]:
        return [e for e in self._engines if e.name not in self._disabled]

    def warmup(self) -> None:
        """Initialise engines up-front so setup problems surface before frame 1."""
        for eng in self._engines:
            try:
                eng.ensure_loaded()
                logger.info("OCR engine ready: %s", eng.name)
            except Exception as exc:
                logger.error("OCR engine '%s' disabled: %s", eng.name, exc)
                self._disabled.add(eng.name)
        if not self.active_engines:
            raise OCRUnavailableError("No OCR engine could be initialised. See the errors above.")

    def _run_variants(self, variants: Dict[str, np.ndarray], frame_index: int,
                      timestamp_s: float, det_conf: float) -> List[OCRObservation]:
        obs: List[OCRObservation] = []
        for eng in self.active_engines:
            for vname, img in variants.items():
                try:
                    raw, conf = eng.read(img)
                except Exception as exc:  # one engine/variant failing must not stop the others
                    self.error_counts[eng.name] = self.error_counts.get(eng.name, 0) + 1
                    logger.warning("OCR error (%s, %s, frame %d): %s", eng.name, vname, frame_index, exc)
                    continue
                norm = normalize_text(raw)
                if not norm:
                    continue
                obs.append(OCRObservation(frame_index, timestamp_s, eng.name, vname,
                                          raw, norm, float(conf), float(det_conf)))
        return obs

    def read_plate(self, crop: np.ndarray, frame_index: int, timestamp_s: float,
                   det_conf: float) -> List[OCRObservation]:
        """Run OCR once on one corrected, enlarged, sharpened and contrast-enhanced crop."""
        o = self._cfg.ocr
        warped = correct_perspective(crop)
        clear_crop = warped if warped is not None else crop
        pre = dict(min_height=o.min_resize_height, max_width=o.max_resize_width)

        from recognition.preprocessing import apply_clahe, ensure_bgr, resize_for_ocr, sharpen, to_gray

        enlarged = resize_for_ocr(clear_crop, **pre)
        channels = cv2.split(enlarged)
        enhanced = [sharpen(apply_clahe(channel)) for channel in channels]
        final_crop = cv2.merge(enhanced)
        return self._run_variants({"best_clear_frame": final_crop},
                                  frame_index, timestamp_s, det_conf)

    def close(self) -> None:
        for eng in self._engines:
            try:
                eng.close()
            except Exception:  # pragma: no cover
                logger.debug("engine close failed", exc_info=True)
