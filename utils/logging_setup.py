"""Console + rotating-file logging."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FMT))
    root.addHandler(console)
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(Path(log_dir) / "nexus_anpr.log",
                                     maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FMT))
            root.addHandler(fh)
        except OSError as exc:  # read-only FS etc. - console logging still works
            root.warning("File logging disabled: %s", exc)
