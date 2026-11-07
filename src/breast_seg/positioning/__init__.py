"""Positioning assessment: delta-D by two independent routes, graded criteria, and a written report.

Pure Python + numpy / OpenCV; no model framework is imported here.
"""
from .config import DEFAULT_CRITERIA, DEFAULT_SEVERE_MM, DEFAULT_THRESHOLD_MM, Rules, default_criteria
from .criteria import ACCEPTABLE, GOOD, NA, POOR, CriteriaGrader
from .labels import LabelError, apply_labels, clean_labels, clean_rules
from .measures import (
    WIDE_GAP_MM,
    compare,
    masks_from_polygons,
    measure_landmark,
    measure_mask,
    pair_verdict,
    summarise,
)
from .report import LANGUAGES, build_report
from .study import PositioningAssessor, labels_of, public_image

__all__ = [
    "ACCEPTABLE",
    "DEFAULT_CRITERIA",
    "DEFAULT_SEVERE_MM",
    "DEFAULT_THRESHOLD_MM",
    "GOOD",
    "LANGUAGES",
    "NA",
    "POOR",
    "WIDE_GAP_MM",
    "CriteriaGrader",
    "LabelError",
    "PositioningAssessor",
    "Rules",
    "apply_labels",
    "build_report",
    "clean_labels",
    "clean_rules",
    "compare",
    "default_criteria",
    "labels_of",
    "masks_from_polygons",
    "measure_landmark",
    "measure_mask",
    "pair_verdict",
    "public_image",
    "summarise",
]
