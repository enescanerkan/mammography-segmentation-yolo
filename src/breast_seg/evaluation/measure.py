"""Downstream measurements from a predicted label map (or predicted landmarks) against the reference landmarks.

Per image (original-grid millimetres):

* MLO: ``e_nipple``, ``e_pec_top``, ``e_pec_bot`` (distance of the fitted pectoral line's end points to the
  reference end points), ``pnl_mm`` and ``e_pnl = |pnl_mm - gt_pnl_mm|``.
* CC: ``e_nipple``, ``cc_mm`` (D_CC) and ``e_cc = |cc_mm - gt_cc_mm|``.

The reference value (``gt_pnl_mm`` / ``gt_cc_mm``) is always written, even when the model produced nothing, so all
models are compared on the same reference subset. ``e_pec_top`` compares two different definitions of the upper
pectoral point (the annotator's and the fitted line's extrapolated end); report it as descriptive, not as an error.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from breast_seg.evaluation.fallback import farthest_point, tissue_contour
from breast_seg.evaluation.landmarks import MIN_NIPPLE_PX, ImageGeometry, centroid, chest_wall_mm, distance_mm, pnl_mm


@dataclass(frozen=True)
class MeasureOptions:
    """``min_nipple_px``: smaller nipple masks count as not found. ``fallback``: measure at the farthest
    breast-tissue contour point when the nipple is missing (flagged ``fallback = True``)."""

    min_nipple_px: int = MIN_NIPPLE_PX
    fallback: bool = False


def _pectoral_line(label: np.ndarray):
    from breast_seg.geometry import fit_pectoral_line

    line = fit_pectoral_line((label == 1).astype(np.uint8), (label == 2).astype(np.uint8), label.shape)
    return None if line is None else (np.asarray(line.point_a, float), np.asarray(line.point_b, float))


def measure_mlo(label: np.ndarray, geom: ImageGeometry, gt: Optional[Mapping[str, np.ndarray]],
                options: MeasureOptions = MeasureOptions(),
                landmarks: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, object]:
    """MLO measurements. With ``landmarks`` (canvas ``nipple / pec_top / pec_bot``) the points come from a
    landmark model instead of the mask (the mask is then only used for the optional fallback)."""
    t, sp = geom.transform, geom.spacing_mm
    row: Dict[str, object] = {}
    if landmarks is not None:
        nip = landmarks.get("nipple")
        line = (landmarks["pec_top"], landmarks["pec_bot"]) if {"pec_top", "pec_bot"} <= set(landmarks) else None
    else:
        nip = centroid(label == 3, options.min_nipple_px)
        line = _pectoral_line(label)
    if gt is not None:
        row["gt_pnl_mm"] = pnl_mm(gt["nipple"], gt["pec_top"], gt["pec_bot"], sp)
        if nip is not None:
            row["e_nipple"] = distance_mm(t.to_original(nip), gt["nipple"], sp)
    if line is None:
        return row
    pa, pb = t.to_original(line[0]), t.to_original(line[1])
    point = t.to_original(nip) if nip is not None else None
    if gt is not None:
        row["e_pec_top"] = distance_mm(pa, gt["pec_top"], sp)
        row["e_pec_bot"] = distance_mm(pb, gt["pec_bot"], sp)
    if point is None and options.fallback:
        far = farthest_point(tissue_contour(label == 2), lambda q: pnl_mm(t.to_original(q), pa, pb, sp))
        if far is not None:
            point, row["fallback"] = t.to_original(far), True
    if point is not None:
        row["pnl_mm"] = pnl_mm(point, pa, pb, sp)
        if "gt_pnl_mm" in row:
            row["e_pnl"] = abs(row["pnl_mm"] - row["gt_pnl_mm"])
    return row


def measure_cc(label: np.ndarray, geom: ImageGeometry, gt_nipple: Optional[np.ndarray],
               options: MeasureOptions = MeasureOptions(),
               landmarks: Optional[Mapping[str, np.ndarray]] = None) -> Dict[str, object]:
    """CC measurements (D_CC to the image edge on the chest-wall side given by the laterality)."""
    t, sp, lat, w = geom.transform, geom.spacing_mm, geom.laterality, geom.width
    row: Dict[str, object] = {}
    nip = landmarks.get("nipple") if landmarks is not None else centroid(label == 3, options.min_nipple_px)
    if gt_nipple is not None:
        row["gt_cc_mm"] = chest_wall_mm(gt_nipple, lat, w, sp)
    point = t.to_original(nip) if nip is not None else None
    if point is not None and gt_nipple is not None:
        row["e_nipple"] = distance_mm(point, gt_nipple, sp)
    if point is None and options.fallback:
        far = farthest_point(tissue_contour(label == 2), lambda q: chest_wall_mm(t.to_original(q), lat, w, sp))
        if far is not None:
            point, row["fallback"] = t.to_original(far), True
    if point is not None:
        row["cc_mm"] = chest_wall_mm(point, lat, w, sp)
        if "gt_cc_mm" in row:
            row["e_cc"] = abs(row["cc_mm"] - row["gt_cc_mm"])
    return row
