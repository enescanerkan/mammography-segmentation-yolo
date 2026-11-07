"""Thresholds of the positioning verdict and the criterion bands, with their defaults."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

# |PNL_MLO - D_CC| at or below this is acceptable positioning (the "10 mm rule").
DEFAULT_THRESHOLD_MM = 10.0
# Beyond this the pair is called clearly inadequate regardless of the above.
DEFAULT_SEVERE_MM = 15.0
THRESHOLD_BOUNDS_MM = (1.0, 60.0)
SEVERE_BOUNDS_MM = (1.0, 120.0)

# Positioning criteria beyond the delta. Bands are {good, acceptable}: a [low, high] range for "range" criteria,
# otherwise a single limit whose direction the criterion defines (lower-is-better or higher-is-better). The angle
# and nipple-profile limits follow EUREF / PGMI guidance; the millimetre ones follow the same 10 mm convention.
# The focus (variance of Laplacian) and exposure (saturated tissue fraction) bands were calibrated on one
# screening cohort (25th / 5th percentiles) and should be re-measured for a different vendor mix. Compression
# pressure is graded but ESTIMATED (see criteria.py) and never counts towards the score.
DEFAULT_CRITERIA: Dict[str, Dict[str, Any]] = {
    "pectoral_angle_deg": {"good": [20, 35], "acceptable": [15, 45]},
    "pectoral_below_nipple_mm": {"good": 0, "acceptable": -10},  # negative = pectoral ends above the nipple
    "nipple_profile_mm": {"good": 5, "acceptable": 10},
    "nipple_centred_mm": {"good": 10, "acceptable": 20},
    "symmetry_mm": {"good": 10, "acceptable": 20},
    "clipped_fraction": {"good": 0.005, "acceptable": 0.02},
    "sharpness": {"good": 820, "acceptable": 330},
    "compression_kpa": {"good": [8, 15], "acceptable": [5, 20]},
}


def default_criteria() -> Dict[str, Dict[str, Any]]:
    """A fresh copy of :data:`DEFAULT_CRITERIA` (safe to mutate)."""
    return copy.deepcopy(DEFAULT_CRITERIA)


@dataclass(frozen=True)
class Rules:
    """The delta-D verdict thresholds."""

    threshold_mm: float = DEFAULT_THRESHOLD_MM
    severe_mm: float = DEFAULT_SEVERE_MM

    def as_dict(self) -> Dict[str, float]:
        return {"threshold_mm": float(self.threshold_mm), "severe_mm": float(self.severe_mm)}
