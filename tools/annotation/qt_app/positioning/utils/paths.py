"""Weight and output locations for the Breast Positioning tool.

All paths come from the toolkit configuration (``positioning.*`` keys in
``annotation.yaml`` or the matching environment variables); see ``toolkit_config``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

# Files smaller than this are treated as broken downloads (HTML error pages etc.).
MIN_WEIGHT_BYTES = 1_000_000


def _cfg():
    from toolkit_config import get_config

    return get_config()


def weight_paths() -> Dict[str, Path]:
    """``{"mlo": Path, "cc": Path}`` for the two YOLO-pose models."""
    cfg = _cfg()
    return {"mlo": cfg.positioning_mlo_weights, "cc": cfg.positioning_cc_weights}


def missing_weights() -> List[Path]:
    """Weight files that are absent or too small to be a real checkpoint."""
    return [
        p for p in weight_paths().values()
        if not p.is_file() or p.stat().st_size < MIN_WEIGHT_BYTES
    ]


def results_dir() -> Path:
    """Folder for saved reports / figures (created on demand by the callers)."""
    return _cfg().positioning_results_dir
