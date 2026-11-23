"""Conservative OCR text handling.

Rules: upper-case, drop whitespace/punctuation, keep letters and digits exactly as
read. There is NO character substitution (O/0, B/8, I/1 ... stay as the OCR emitted
them) and nothing is ever invented or completed.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Sequence, Tuple

# Conservative Indian registrations: two-letter state/UT, two RTO digits, a
# series, then the four-digit serial. Delhi's legacy one-digit RTO layout is kept.
_INDIAN_STANDARD = re.compile(r"^(?:[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}|DL\d[A-Z]{1,3}\d{4})$")
_INDIAN_STATE_CODES = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML",
    "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR",
    "TS", "UK", "UP", "WB",
})
# Bharat (BH) series: YY BH NNNN X(X), e.g. 22BH1234AA.
_INDIAN_BH = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")
# Substring match for Indian plates, to find text embedded in OCR noise.
_INDIAN_SUBSTRING = re.compile(r"[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{1,4}")


@dataclass(frozen=True)
class TextFragment:
    """One text box returned by an OCR engine."""
    text: str
    confidence: float
    cx: float
    cy: float
    height: float


@dataclass(frozen=True)
class ProcessedText:
    raw: str
    normalized: str
    indian_format: bool


def normalize_text(raw: str) -> str:
    """Upper-case and remove separators, correcting OCR confusion by plate slot."""
    text = "".join(ch for ch in (raw or "").upper() if ch.isalnum())

    # Remove OCR noise around a valid plate, e.g. "NO KA03NU7762" -> "KA03NU7762".
    if len(text) > 10:
        match = _INDIAN_SUBSTRING.search(text)
        if match and len(match.group(0)) == 10:
            text = match.group(0)

    # Standard Indian plate layout: AA DD LL DDDD.
    if len(text) == 10:
        chars = list(text)
        digit_for_letter = {
            "O": "0", "I": "1", "L": "1", "Q": "0", "Z": "2",
            "S": "5", "G": "6", "B": "8", "A": "4", "D": "0", "T": "7"
        }
        letter_for_digit = {
            "0": "O", "1": "I", "2": "Z", "5": "S",
            "6": "G", "8": "B", "4": "A", "7": "T"
        }

        for index in (0, 1, 4, 5):
            if chars[index] in letter_for_digit:
                chars[index] = letter_for_digit[chars[index]]
        for index in (2, 3, 6, 7, 8, 9):
            if chars[index] in digit_for_letter:
                chars[index] = digit_for_letter[chars[index]]
        return "".join(chars)

    return text


def is_indian_format(normalized: str) -> bool:
    """Return true only for a conservative Indian registration pattern and prefix."""
    text = (normalized or "").upper()
    standard = bool(_INDIAN_STANDARD.fullmatch(text)
                    and text[:2] in _INDIAN_STATE_CODES)
    bharat = bool(_INDIAN_BH.fullmatch(text))
    return standard or bharat


def process_text(raw: str) -> ProcessedText:
    norm = normalize_text(raw)
    return ProcessedText(raw=raw, normalized=norm, indian_format=is_indian_format(norm))


def merge_fragments(fragments: Sequence[TextFragment]) -> Tuple[str, float]:
    """Join OCR boxes into reading order (top line first, left to right).

    Returns ``(raw_text, confidence)``; lines are separated by ``\\n`` in the raw text.
    Confidence is the mean weighted by alphanumeric length.
    """
    frags = [f for f in fragments if f.text and f.text.strip()]
    if not frags:
        return "", 0.0
    med_h = statistics.median(max(f.height, 1.0) for f in frags)
    ordered = sorted(frags, key=lambda f: f.cy)
    lines: list[list[TextFragment]] = []
    for frag in ordered:
        if lines:
            line_cy = sum(f.cy for f in lines[-1]) / len(lines[-1])
            if abs(frag.cy - line_cy) <= 0.6 * med_h:
                lines[-1].append(frag)
                continue
        lines.append([frag])
    raw_lines = [" ".join(f.text.strip() for f in sorted(line, key=lambda f: f.cx)) for line in lines]
    weights = [max(len(normalize_text(f.text)), 1) for f in frags]
    conf = sum(f.confidence * w for f, w in zip(frags, weights)) / sum(weights)
    return "\n".join(raw_lines), float(conf)
