"""Positioning measurements, computed twice per breast and then compared.

Two independent routes to the same number:

``landmark``  the pose model's points -> the PNL is the perpendicular from the nipple to the line through
              ``pectoral_top`` / ``pectoral_bottom``; on CC the distance from the nipple to the chest-wall edge.
``mask``      the segmentation model's contours -> the pectoral line is fitted to the pectoral / breast boundary,
              the nipple is the mask centroid.

Both work in the same 640 frame with the same millimetre scale, so their outputs are directly comparable.
Disagreement between them is the signal worth surfacing: it is what marks a study as needing a human look.

An *image* here is the per-view record the review app stores: ``view``, ``laterality``, ``chest_side``,
``mm_per_px``, ``landmarks`` (pose output), ``masks`` (class -> 640x640 binary) and, after measuring, ``measures``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .. import geometry
from .config import DEFAULT_SEVERE_MM, DEFAULT_THRESHOLD_MM

GOOD = "good"
BAD = "bad"
FRAME = 640

# Two routes landing this far apart on the same breast means one of them placed a landmark badly, even when both
# stay on the same side of the clinical threshold.
WIDE_GAP_MM = 5.0


def measure_landmark(image: Mapping[str, Any]) -> Dict[str, Any]:
    """PNL (MLO) or nipple-to-chest-wall (CC) from the pose keypoints."""
    points = image["landmarks"].get("points") or {}
    nipple = points.get("nipple")
    if not nipple:
        return _missing("nipple keypoint not detected")

    nipple = np.asarray(nipple, dtype=float)
    mm_per_px = image["mm_per_px"]

    if image["view"] == "CC":
        distance_px, edge = geometry.cc_chestwall_distance(nipple, image["chest_side"], FRAME)
        return _measured(distance_px, mm_per_px, nipple, edge)

    top, bottom = points.get("pectoral_top"), points.get("pectoral_bottom")
    if not top or not bottom:
        return _missing("pectoral keypoints not detected")
    distance_px, foot = geometry.perpendicular_distance(np.asarray(top, float), np.asarray(bottom, float), nipple)
    result = _measured(distance_px, mm_per_px, nipple, foot)
    result["line"] = [[round(float(v), 1) for v in top], [round(float(v), 1) for v in bottom]]
    return result


def measure_mask(image: Mapping[str, Any]) -> Dict[str, Any]:
    """The same two distances, derived from the segmentation masks instead."""
    masks = image["masks"]
    nipple_mask = masks.get("nipple")
    breast_mask = masks.get("breast-tissue")
    if nipple_mask is None:
        return _missing("no nipple mask")
    if breast_mask is None:
        return _missing("no breast tissue mask")

    centre = geometry.compute_nipple_centroid(nipple_mask)
    if centre is None:
        return _missing("nipple mask is empty")

    mm_per_px = image["mm_per_px"]
    shape = breast_mask.shape[:2]

    if image["view"] == "CC":
        depth = geometry.compute_cc_depth(centre, breast_mask, shape)
        if depth is None:
            return _missing("chest wall edge not found")
        return _measured(depth.distance_px, mm_per_px, centre, depth.edge_point)

    pectoral_mask = masks.get("pectoral")
    if pectoral_mask is None:
        return _missing("no pectoral mask")
    line = geometry.fit_pectoral_line(pectoral_mask, breast_mask, shape)
    if line is None:
        return _missing("could not fit a pectoral line")
    pnl = geometry.compute_pnl(centre, line)
    result = _measured(pnl.distance_px, mm_per_px, centre, pnl.foot_point)
    result["line"] = [[float(line.point_a[0]), float(line.point_a[1])],
                      [float(line.point_b[0]), float(line.point_b[1])]]
    return result


def pair_verdict(mlo: Optional[Mapping[str, Any]], cc: Optional[Mapping[str, Any]],
                 threshold_mm: float = DEFAULT_THRESHOLD_MM, severe_mm: float = DEFAULT_SEVERE_MM) -> Dict[str, Any]:
    """|PNL_MLO - D_CC| of one breast against the clinical threshold."""
    threshold, severe = float(threshold_mm), float(severe_mm)

    if mlo is None or cc is None:
        return {"ok": False, "reason": "needs both MLO and CC"}
    if not mlo.get("ok") or not cc.get("ok"):
        return {"ok": False, "reason": mlo.get("reason") or cc.get("reason") or "no measurement"}

    delta = abs(mlo["mm"] - cc["mm"])
    return {
        "ok": True,
        "pnl_mm": round(mlo["mm"], 2),
        "cc_mm": round(cc["mm"], 2),
        "delta_mm": round(delta, 2),
        "verdict": GOOD if delta <= threshold else BAD,
        "severe": bool(delta > severe),
        "threshold_mm": threshold,
    }


def compare(landmark: Mapping[str, Any], mask: Mapping[str, Any]) -> Dict[str, Any]:
    """How far apart the two routes landed, and whether they agree on the call."""
    if not landmark.get("ok") or not mask.get("ok"):
        available = [name for name, result in (("landmark", landmark), ("mask", mask)) if result.get("ok")]
        return {
            "state": "partial" if available else "none",
            "available": available,
            "note": "only one method produced a result" if available else "neither method produced a result",
        }

    gap = abs(landmark["delta_mm"] - mask["delta_mm"])
    agree = landmark["verdict"] == mask["verdict"]
    # Agreeing on the verdict is not the same as agreeing on the measurement: the two routes can land on the same
    # side of the threshold while one of them has clearly mislocated a landmark. A wide gap is worth a look either
    # way.
    wide = gap > WIDE_GAP_MM
    if not agree:
        note = "the methods disagree on the verdict — review this study"
    elif wide:
        note = f"same verdict, but the measurements differ by {gap:.1f} mm"
    else:
        note = "both methods agree"
    return {
        "state": "agree" if agree else "disagree",
        "gap_mm": round(gap, 2),
        "wide_gap": wide,
        "note": note,
    }


def masks_from_polygons(polygons: Optional[Mapping[str, Any]], size: int = FRAME) -> Dict[str, np.ndarray]:
    """Rasterise stored contours back into ``size`` x ``size`` binary masks.

    The polygon is the only shape that survives a round-trip through the editor, so measuring from it keeps the
    drawn outline and the reported millimetres in agreement — measuring from the model's original raster would let
    a corrected contour leave the number untouched.
    """
    import cv2

    masks: Dict[str, np.ndarray] = {}
    for name, entry in (polygons or {}).items():
        points = (entry or {}).get("points") or []
        if len(points) < 3:
            continue
        canvas = np.zeros((size, size), np.uint8)
        cv2.fillPoly(canvas, [np.asarray(points, dtype=np.float64).round().astype(np.int32)], 1)
        if canvas.any():
            masks[name] = canvas
    return masks


def summarise(images: List[Mapping[str, Any]], threshold_mm: float = DEFAULT_THRESHOLD_MM,
              severe_mm: float = DEFAULT_SEVERE_MM) -> Dict[str, Any]:
    """Per-laterality verdicts for both routes, plus their agreement, plus the study-level call."""
    by_side: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for image in images:
        by_side.setdefault(image["laterality"], {})[image["view"]] = image

    sides = []
    for laterality in ("L", "R"):
        views = by_side.get(laterality)
        if not views:
            continue
        landmark = pair_verdict(_route(views.get("MLO"), "landmark"),
                                _route(views.get("CC"), "landmark"), threshold_mm, severe_mm)
        mask = pair_verdict(_route(views.get("MLO"), "mask"),
                            _route(views.get("CC"), "mask"), threshold_mm, severe_mm)
        sides.append({
            "laterality": laterality,
            "views": sorted(views),
            "landmark": landmark,
            "mask": mask,
            "comparison": compare(landmark, mask),
        })

    verdicts = [side["landmark"]["verdict"] for side in sides if side["landmark"].get("ok")]
    verdicts += [side["mask"]["verdict"] for side in sides if side["mask"].get("ok")]
    return {
        "sides": sides,
        "overall": (BAD if BAD in verdicts else GOOD) if verdicts else None,
        "flagged": any(side["comparison"]["state"] != "agree" or side["comparison"].get("wide_gap")
                       for side in sides),
    }


# ── helpers ────────────────────────────────────────────────────────────────
def _route(image: Optional[Mapping[str, Any]], route: str) -> Optional[Mapping[str, Any]]:
    return None if image is None else image["measures"][route]


def _measured(distance_px: float, mm_per_px: float, start: Any, end: Any) -> Dict[str, Any]:
    return {
        "ok": True,
        "px": round(float(distance_px), 2),
        "mm": float(distance_px) * mm_per_px,
        "from": [round(float(start[0]), 1), round(float(start[1]), 1)],
        "to": [round(float(end[0]), 1), round(float(end[1]), 1)],
    }


def _missing(reason: str) -> Dict[str, Any]:
    return {"ok": False, "reason": reason}
