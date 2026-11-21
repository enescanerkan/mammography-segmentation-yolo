"""Thresholds of the positioning verdict, with their defaults."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# |PNL_MLO - D_CC| at or below this is acceptable positioning (the "10 mm rule").
DEFAULT_THRESHOLD_MM = 10.0
# Beyond this the pair is called clearly inadequate regardless of the above.
DEFAULT_SEVERE_MM = 15.0
THRESHOLD_BOUNDS_MM = (1.0, 60.0)
SEVERE_BOUNDS_MM = (1.0, 120.0)

# Band keys of criteria that are no longer graded (2026-11-20); no criterion has a configurable band any more. A
# config written for an older set may still carry a ``criteria:`` section naming them: they are ignored, not refused.
RETIRED_CRITERIA = frozenset({"pectoral_angle_deg", "nipple_profile_mm", "nipple_centred_mm", "symmetry_mm",
                              "clipped_fraction", "sharpness", "compression_kpa", "pectoral_below_nipple_mm"})


@dataclass(frozen=True)
class Rules:
    """The delta-D verdict thresholds."""

    threshold_mm: float = DEFAULT_THRESHOLD_MM
    severe_mm: float = DEFAULT_SEVERE_MM

    def as_dict(self) -> Dict[str, float]:
        return {"threshold_mm": float(self.threshold_mm), "severe_mm": float(self.severe_mm)}
