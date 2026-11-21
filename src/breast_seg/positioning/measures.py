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
from .numbers import fixed

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


def measure_mask(image: Mapping[str, Any], contour_fallback: bool = False) -> Dict[str, Any]:
    """The same two distances, derived from the segmentation masks instead.

    On an MLO the fitted pectoral line is returned as ``line`` whenever the pectoral and breast masks allow it, also
    when the distance itself cannot be measured (no nipple mask): the pectoral angle and the MLO quality need only
    the line, and take the nipple from the pose point then.

    ``contour_fallback``: when no nipple contour was found (after every search the engine runs), measure at the
    breast-tissue contour point that reaches farthest — from the pectoral line on an MLO, from the chest wall on a
    CC — instead of giving no measurement. Such a result is marked ``"estimated": True`` with ``"nipple":
    "contour"`` (where its start point came from) and ``"missing"`` (why no nipple was used). It enters the verdict
    like any other; a nipple contour, once found or drawn, is always measured from instead.
    """
    masks = image["masks"]
    nipple_mask = masks.get("nipple")
    breast_mask = masks.get("breast-tissue")

    line = None
    if image["view"] != "CC" and breast_mask is not None and masks.get("pectoral") is not None:
        line = geometry.fit_pectoral_line(masks["pectoral"], breast_mask, breast_mask.shape[:2])

    def missing(reason: str) -> Dict[str, Any]:
        result = _missing(reason)
        if line is not None:
            result["line"] = _line_points(line)
        return result

    if breast_mask is None:
        return missing("no nipple mask" if nipple_mask is None else "no breast tissue mask")

    centre = geometry.compute_nipple_centroid(nipple_mask) if nipple_mask is not None else None
    if centre is None:
        reason = "no nipple mask" if nipple_mask is None else "nipple mask is empty"
        if contour_fallback:
            return _contour_estimate(image, breast_mask, line, reason) or missing(reason)
        return missing(reason)

    mm_per_px = image["mm_per_px"]

    if image["view"] == "CC":
        depth = geometry.compute_cc_depth(centre, breast_mask, breast_mask.shape[:2])
        if depth is None:
            return missing("chest wall edge not found")
        return _measured(depth.distance_px, mm_per_px, centre, depth.edge_point)

    if masks.get("pectoral") is None:
        return missing("no pectoral mask")
    if line is None:
        return missing("could not fit a pectoral line")
    pnl = geometry.compute_pnl(centre, line)
    result = _measured(pnl.distance_px, mm_per_px, centre, pnl.foot_point)
    result["line"] = _line_points(line)
    return result


def _contour_estimate(image: Mapping[str, Any], breast_mask: np.ndarray, line: Any,
                      reason: str) -> Optional[Dict[str, Any]]:
    """The segmentation measurement at the breast-tissue contour point that reaches farthest, measured exactly as a
    nipple there would be; None when it cannot be taken (too little tissue, no pectoral line on an MLO, no chest-wall
    edge on a CC).

    The point is chosen as it was evaluated (``breast_seg.evaluation.fallback``): on an MLO the contour point farthest
    from the fitted pectoral line taken as an infinite line, on a CC the one farthest from the chest wall, the frame
    edge on the chest side. (Scored against the fitted *segment* instead, a point below the end of the muscle wins
    by its distance to that end, and the estimate starts at the bottom of the breast instead of its tip.)
    """
    mm_per_px = image["mm_per_px"]
    side = image.get("chest_side", "L")
    if image["view"] == "CC":
        far = geometry.anterior_point(breast_mask, side, None, FRAME)
        if far is None:
            return None
        point = (int(far[0]), int(far[1]))
        depth = geometry.compute_cc_depth(point, breast_mask, breast_mask.shape[:2])
        if depth is None:
            return None
        result = _measured(depth.distance_px, mm_per_px, point, depth.edge_point)
    else:
        far = geometry.anterior_point(breast_mask, side, line, FRAME) if line is not None else None
        if far is None:
            return None
        point = (int(far[0]), int(far[1]))
        pnl = geometry.compute_pnl(point, line)
        result = _measured(pnl.distance_px, mm_per_px, point, pnl.foot_point)
        result["line"] = _line_points(line)
    result.update(estimated=True, nipple="contour", missing=reason)
    return result


def _line_points(line: Any) -> List[List[float]]:
    return [[float(line.point_a[0]), float(line.point_a[1])], [float(line.point_b[0]), float(line.point_b[1])]]


def pair_verdict(mlo: Optional[Mapping[str, Any]], cc: Optional[Mapping[str, Any]],
                 threshold_mm: float = DEFAULT_THRESHOLD_MM, severe_mm: float = DEFAULT_SEVERE_MM) -> Dict[str, Any]:
    """|PNL_MLO - D_CC| of one breast against the clinical threshold."""
    threshold, severe = float(threshold_mm), float(severe_mm)

    if mlo is None or cc is None:
        return {"ok": False, "reason": "needs both MLO and CC"}
    if not mlo.get("ok") or not cc.get("ok"):
        return {"ok": False, "reason": mlo.get("reason") or cc.get("reason") or "no measurement"}

    # Unrounded, like the per-view measurements they come from: whatever shows a millimetre rounds it once
    # (numbers.py), so a card can never disagree with the value under its image.
    pnl, wall = float(mlo["mm"]), float(cc["mm"])
    delta = abs(pnl - wall)
    result = {
        "ok": True,
        "pnl_mm": pnl,
        "cc_mm": wall,
        "delta_mm": delta,
        "verdict": GOOD if delta <= threshold else BAD,
        "severe": bool(delta > severe),
        "threshold_mm": threshold,
    }
    # A view measured without its nipple (the contour fallback): the verdict stands, and says what it rests on.
    estimated = [view for view, measure in (("MLO", mlo), ("CC", cc)) if measure.get("estimated")]
    if estimated:
        result["estimated"] = estimated
    return result


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
        note = f"same verdict, but the measurements differ by {fixed(gap)} mm"
    else:
        note = "both methods agree"
    return {
        "state": "agree" if agree else "disagree",
        "gap_mm": gap,
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


def restore_precision(summary: Dict[str, Any], images: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Put the unrounded millimetres back into a summary stored with them rounded to two decimals (in place).

    Records written before the summary carried unrounded values showed 47.949 mm as 47.9 mm under the image and, from
    the summary's 47.95, as 48.0 mm on the card. The per-view measurements such a summary was built from are stored
    unrounded next to it, so its numbers are restored from them — only where they round to the stored ones, and
    without touching a verdict: those were decided from the same unrounded values.
    """
    by_side: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for image in images:
        by_side.setdefault(image.get("laterality"), {})[image.get("view")] = image

    for side in summary.get("sides") or []:
        views = by_side.get(side.get("laterality")) or {}
        deltas = []
        for route in ("landmark", "mask"):
            result = side.get(route) or {}
            mlo = ((views.get("MLO") or {}).get("measures") or {}).get(route) or {}
            cc = ((views.get("CC") or {}).get("measures") or {}).get(route) or {}
            if not (result.get("ok") and mlo.get("ok") and cc.get("ok")):
                continue
            pnl, wall = float(mlo["mm"]), float(cc["mm"])
            if round(pnl, 2) != round(float(result["pnl_mm"]), 2) or round(wall, 2) != round(float(result["cc_mm"]), 2):
                continue  # not the measurements this verdict was decided from: leave it as stored
            result.update(pnl_mm=pnl, cc_mm=wall, delta_mm=abs(pnl - wall))
            deltas.append(result["delta_mm"])
        comparison = side.get("comparison") or {}
        if len(deltas) == 2 and "gap_mm" in comparison:
            comparison["gap_mm"] = abs(deltas[0] - deltas[1])
    return summary


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
