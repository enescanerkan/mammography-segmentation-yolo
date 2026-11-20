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

# Positioning criteria beyond the delta. Bands are {good, acceptable}, a single limit whose direction the criterion
# defines. Only one criterion is graded; the pectoral angle, nipple centring and gantry angle are shown ungraded.
DEFAULT_CRITERIA: Dict[str, Dict[str, Any]] = {
    "pectoral_below_nipple_mm": {"good": 0, "acceptable": -10},  # negative = pectoral ends above the nipple
}

# Bands of criteria that are no longer graded (2026-11-20). A config written for the old set may still name them;
# they are ignored rather than refused.
RETIRED_CRITERIA = frozenset({"pectoral_angle_deg", "nipple_profile_mm", "nipple_centred_mm", "symmetry_mm",
                              "clipped_fraction", "sharpness", "compression_kpa"})


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
