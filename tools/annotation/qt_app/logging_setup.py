"""Centralized logging setup for the PyQt5 toolkit.

Single ``setup_logging()`` call from the entry point. Subsequent
``logging.getLogger(__name__)`` calls anywhere in the app produce
formatted lines on both console and a rolling file.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_LOG_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s :: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    quiet_libs: bool = True,
) -> Path:
    """Configure root logger. Returns the active log file path.

    Idempotent: safe to call twice (won't add duplicate handlers).
    """
    root = logging.getLogger()
    if any(getattr(h, "_qt_app_owned", False) for h in root.handlers):
        # already configured
        return Path(getattr(root, "_qt_app_log_path", ""))

    root.setLevel(level)

    fmt = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)

    # Console
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    ch._qt_app_owned = True  # type: ignore[attr-defined]
    root.addHandler(ch)

    # File (rolling)
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "qt_app.log"
    fh = RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=4, encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    fh._qt_app_owned = True  # type: ignore[attr-defined]
    root.addHandler(fh)
    root._qt_app_log_path = str(log_path)  # type: ignore[attr-defined]

    if quiet_libs:
        for name in ("PIL", "matplotlib", "torch", "urllib3", "PyQt5"):
            logging.getLogger(name).setLevel(logging.WARNING)

    root.info("Logging initialized. file=%s level=%s", log_path, logging.getLevelName(level))

    # Catch unhandled exceptions in the main thread.
    def _excepthook(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("uncaught").error("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _excepthook

    return log_path
