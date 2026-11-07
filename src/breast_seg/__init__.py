"""Mammography segmentation, landmarks and positioning assessment.

Submodules are imported lazily so that light users (``breast_seg.geometry``) do not pull in torch / ultralytics.
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.0.0.dev0"

_LAZY = {
    "Config": "breast_seg.config",
    "SegmentationModel": "breast_seg.model",
    "MLOAnalyzer": "breast_seg.analyzer",
    "ResultVisualizer": "breast_seg.visualizer",
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'breast_seg' has no attribute {name!r}")
