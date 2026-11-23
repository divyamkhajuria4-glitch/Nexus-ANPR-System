"""Device selection (CUDA / Apple-Silicon MPS / CPU) with a safe CPU fallback."""
from __future__ import annotations

import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")
_UNSET = object()
T = TypeVar("T")


def _import_torch() -> Any:
    try:
        import torch  # lazy on purpose: importing torch is slow and optional for unit tests
        return torch
    except Exception as exc:  # ImportError or a broken install
        logger.warning("PyTorch could not be imported (%s); using CPU.", exc)
        return None


def _mps_available(torch: Any) -> bool:
    try:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend and backend.is_built() and backend.is_available())
    except Exception:
        return False


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def select_device(preference: str = "auto", torch_module: Any = _UNSET) -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"``.

    ``torch_module`` can be injected for tests. A requested accelerator that is
    unavailable falls back to CPU with a warning instead of crashing.
    """
    pref = (preference or "auto").lower()
    if pref not in VALID_DEVICES:
        raise ValueError(f"Unknown device '{preference}'. Choose one of {VALID_DEVICES}.")
    if pref == "cpu":
        return "cpu"

    torch = _import_torch() if torch_module is _UNSET else torch_module
    if torch is None:
        if pref != "auto":
            logger.warning("Requested device '%s' but PyTorch is unavailable; using CPU.", pref)
        return "cpu"

    if pref == "auto":
        if _cuda_available(torch):
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    available = _cuda_available(torch) if pref == "cuda" else _mps_available(torch)
    if available:
        return pref
    logger.warning("Requested device '%s' is not available; falling back to CPU.", pref)
    return "cpu"


class DeviceContext:
    """Holds the active device and retries once on CPU if an accelerator call fails."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.fell_back = False

    def fall_back_to_cpu(self, reason: str) -> bool:
        if self.device == "cpu":
            return False
        logger.warning("Inference on '%s' failed (%s). Switching to CPU for the rest of this run.",
                       self.device, reason)
        self.device = "cpu"
        self.fell_back = True
        return True

    def run(self, fn: Callable[[str], T]) -> T:
        try:
            return fn(self.device)
        except Exception as exc:
            if self.fall_back_to_cpu(str(exc)):
                return fn(self.device)
            raise


def empty_device_cache() -> None:
    """Release cached accelerator memory (best effort, never raises)."""
    try:
        import sys
        torch = sys.modules.get("torch")
        if torch is None:
            return
        if _cuda_available(torch):
            torch.cuda.empty_cache()
        if _mps_available(torch) and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover
        logger.debug("empty_device_cache failed", exc_info=True)
