"""PNL / CC-depth measurement for the annotation toolkit.

Turns the tissues labelled on one image (nipple, pectoral muscle, breast) into the
standard positioning measurements:

* **MLO view** (pectoral labelled): the pectoral line is fitted along the
  pectoral / breast boundary and a perpendicular is dropped from the nipple
  centroid to it; its length is the **PNL** (posterior nipple line).
* **CC view** (no pectoral): a horizontal line from the nipple centroid back to the
  chest-wall edge of the breast tissue, the **CC depth**.

All geometry comes from :mod:`breast_seg.geometry` (pectoral line by the package's
tangent rule, nipple centroid, PNL foot, CC depth). This module only extracts masks
from the toolkit's annotation store and packages the result for the canvas overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from toolkit_config import ensure_breast_seg_importable

ensure_breast_seg_importable()

from breast_seg.geometry import (  # noqa: E402  (needs the sys.path fallback above)
    CCDepthResult,
    PectoralLine,
    PNLResult,
    compute_cc_depth,
    compute_nipple_centroid,
    compute_pnl,
    fit_pectoral_line,
)

__all__ = [
    "AnalysisResult",
    "CCDepthResult",
    "PNLResult",
    "PectoralLine",
    "analyze_annotation",
    "compute_cc_depth",
    "compute_nipple_centroid",
    "compute_pnl",
    "fit_pectoral_line",
    "mask_for_tissue",
]

# Tissue keys as defined in ``interactive.prompts.TISSUE_PRESETS``.
NIPPLE_KEY = "nipple"
PECTORAL_KEY = "pectoral"
BREAST_KEY = "breast"


@dataclass
class AnalysisResult:
    """Everything the canvas needs to draw the overlay."""

    is_mlo: bool
    has_nipple: bool
    has_pectoral: bool
    pectoral_line: Optional[PectoralLine] = None
    pnl: Optional[PNLResult] = None
    cc_depth: Optional[CCDepthResult] = None
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.pnl is not None or self.cc_depth is not None


def mask_for_tissue(ann, tissue_key: str) -> Optional[np.ndarray]:
    """Return a binary ``H x W`` uint8 mask (0/255) for ``tissue_key``, or ``None`` if empty.

    Combines the stored union mask with the tissue's polygons, so a polygon-only
    label (not rasterised yet) still counts.
    """
    h, w = ann.image_hw
    out = np.zeros((h, w), dtype=np.uint8)

    stored = ann.masks.get(tissue_key)
    if stored is not None and stored.any():
        out = np.where(stored > 0, np.uint8(255), np.uint8(0))

    for poly in ann.polygons.get(tissue_key) or []:
        if len(poly) < 3:
            continue
        pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
        cv2.fillPoly(out, [pts], 255)

    return out if out.any() else None


def analyze_annotation(ann) -> AnalysisResult:
    """Compute the PNL (MLO) or the CC depth (CC) from one image's annotation.

    The view is inferred as in ``breast_seg``: a labelled pectoral muscle means MLO,
    otherwise CC.
    """
    image_shape = ann.image_hw

    nipple_mask = mask_for_tissue(ann, NIPPLE_KEY)
    breast_mask = mask_for_tissue(ann, BREAST_KEY)
    pectoral_mask = mask_for_tissue(ann, PECTORAL_KEY)

    has_nipple = nipple_mask is not None
    has_pectoral = pectoral_mask is not None
    result = AnalysisResult(is_mlo=has_pectoral, has_nipple=has_nipple, has_pectoral=has_pectoral)

    if not has_nipple:
        result.messages.append("Nipple is not labelled; the measurement needs the nipple.")
        return result

    nipple_center = compute_nipple_centroid(nipple_mask)
    if nipple_center is None:
        result.messages.append("Nipple mask is empty.")
        return result

    if breast_mask is None:
        what = "the pectoral line" if result.is_mlo else "the CC depth"
        result.messages.append(f"Breast tissue is not labelled; cannot draw {what}.")
        return result

    if result.is_mlo:
        line = fit_pectoral_line(pectoral_mask, breast_mask, image_shape)
        if line is None:
            result.messages.append("Could not fit the pectoral line.")
            return result
        result.pectoral_line = line
        result.pnl = compute_pnl(nipple_center, line)
    else:
        result.cc_depth = compute_cc_depth(nipple_center, breast_mask, image_shape)
        if result.cc_depth is None:
            result.messages.append("Could not compute the CC depth.")

    return result
