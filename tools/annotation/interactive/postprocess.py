"""Post-processing strategies that transform a raw model mask.

Two concerns live here:

1. :class:`HardIgnoreMask` — "ignore points must NOT be covered". SAM's
   negative point is a *soft* prior (the model may keep the area). To
   guarantee exclusion we mechanically subtract a disk of radius ``R``
   around each ignore point from the mask, AFTER the model returns.

2. :class:`ClassCompetition` — Resolve overlap between the current target
   class mask and an excluding class mask **that was already drawn by the
   user**. Specifically: when the user is segmenting breast tissue on an
   MLO view, if they have already drawn a pectoral mask in this session,
   we mechanically subtract those pectoral pixels from the breast mask.

   This is intentionally **NOT** an auto-prediction: an earlier version
   ran a second SAM2 forward pass to guess pectoral, but because the
   fine-tuned decoder is box-only and class-agnostic, that auto-pectoral
   mask sometimes ended up segmenting the *breast* itself — which then
   got subtracted from the breast prediction, leaving it empty. The
   user-mask-driven approach is robust by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class HardIgnoreMask:
    """Subtract a disk of radius ``radius_px`` around each ignore point.

    Pixel coverage of one disk: ``round(pi * R^2)``. UI shows this so the
    user knows what they're subtracting.
    """

    radius_px: int = 20

    def disk_pixels(self) -> int:
        return int(round(np.pi * self.radius_px * self.radius_px))

    def apply(
        self,
        mask: np.ndarray,
        ignore_points: list[tuple[float, float]],
    ) -> np.ndarray:
        if not ignore_points or self.radius_px <= 0:
            return mask
        h, w = mask.shape[:2]
        kill = np.zeros((h, w), dtype=np.uint8)
        for x, y in ignore_points:
            cv2.circle(kill, (int(x), int(y)), self.radius_px, 1, thickness=-1)
        out = mask.astype(np.uint8).copy()
        out[kill.astype(bool)] = 0
        return out


class ClassCompetition:
    """Subtract a previously-drawn 'exclude' tissue from the current target.

    Typical use: target=breast tissue, exclude=pectoral (drawn earlier by
    the user). Behavior is a pure set difference; if no exclude mask is
    available, the target is returned unchanged.
    """

    @staticmethod
    def subtract(
        target: np.ndarray,
        exclude: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int]:
        """Return (new_mask, pixels_removed)."""
        if exclude is None or exclude.shape != target.shape or not exclude.any():
            return target, 0
        out = target.astype(np.uint8).copy()
        before = int(out.sum())
        out[exclude.astype(bool)] = 0
        return out, before - int(out.sum())

    @staticmethod
    def resolve_exclude(
        masks_by_tissue: Mapping[str, np.ndarray],
        current_tissue_key: str,
        exclude_for: Mapping[str, list[str]] | None = None,
    ) -> Optional[np.ndarray]:
        """Look up the OR of all 'exclude' tissue masks for the current tissue.

        ``exclude_for`` maps a tissue key to a list of tissue keys whose
        masks should be subtracted. Default policy: breast subtracts pectoral.
        """
        if exclude_for is None:
            exclude_for = {"breast": ["pectoral"]}
        keys = exclude_for.get(current_tissue_key, [])
        merged: Optional[np.ndarray] = None
        for k in keys:
            m = masks_by_tissue.get(k)
            if m is None or not isinstance(m, np.ndarray) or not m.any():
                continue
            merged = m.astype(bool) if merged is None else (merged | m.astype(bool))
        return merged.astype(np.uint8) if merged is not None else None
