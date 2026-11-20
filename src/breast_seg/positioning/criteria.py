"""Positioning criteria beyond delta-D.

Delta-D answers one question — is the posterior tissue on the CC consistent with the MLO. Next to it the app grades
one more thing a reader checks on the MLO, whether the pectoral muscle reaches the nipple level, and shows four
values as information only, with no grade: the pectoral angle (MLO), the MLO-alone verdict (MLO), how far the nipple
sits from the centre of the breast on the CC, and the gantry angle the header reports.

Removed on 2026-11-20 because the radiologists do not grade them and their bands were never checked against a
reader: compression force / thickness / pressure, dose, focus, exposure, nipple in profile, pectoral visible on CC,
inframammary fold, left/right symmetry, and the 0-100 criterion score.

The one remaining threshold is configuration (:data:`~breast_seg.positioning.config.DEFAULT_CRITERIA`).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .config import default_criteria

GOOD = "good"
ACCEPTABLE = "acceptable"
POOR = "poor"
NA = "na"
INFO = "info"  # a value shown for information: never graded, never counted

# MLO-alone verdict, the criterion of the MLO-only RAUNet paper (its evaluation.py): the foot of the perpendicular
# from the nipple to the infinite pectoral line must fall inside the image, with a margin of 0.25% of the longest side.
MLO_ALONE_MARGIN = 0.0025
MIN_NIPPLE_PX = 30  # a smaller nipple mask counts as missing (the pose nipple is used instead)

# Bumped whenever the set of criteria changes, so stored studies are regraded when they are opened again.
CRITERIA_VERSION = 3


def item(item_id: str, name: str, grade: str, value: Any = None, unit: str = "",
         detail: str = "", view_key: str = "", estimated: bool = False) -> Dict[str, Any]:
    """One graded criterion as stored in a study record."""
    return {"id": item_id, "name": name, "grade": grade, "value": value,
            "unit": unit, "detail": detail, "view_key": view_key, "estimated": estimated}


class CriteriaGrader:
    """Grades the per-image and per-study criteria against configurable bands.

    Args:
        bands: ``{criterion key: {"good": ..., "acceptable": ...}}``; missing keys grade as ``na``.
    """

    def __init__(self, bands: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        self.bands: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in (bands or default_criteria()).items()}

    # ── grading primitives ───────────────────────────────────────────────
    def _band(self, key: str) -> Dict[str, Any]:
        return self.bands.get(key) or {}

    def higher_is_better(self, value: float, key: str) -> str:
        band = self._band(key)
        if not band:
            return NA
        if value >= float(band["good"]):
            return GOOD
        if value >= float(band["acceptable"]):
            return ACCEPTABLE
        return POOR

    # ── per image ────────────────────────────────────────────────────────
    def for_image(self, image: Mapping[str, Any], masks: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Criteria of one view; ``image`` must already carry ``measures`` (for the fitted pectoral line)."""
        view_key = image["view_key"]
        mm = image["mm_per_px"]
        items: List[Optional[Dict[str, Any]]] = []

        breast = masks.get("breast-tissue")
        nipple = masks.get("nipple")
        nipple_point = nipple_position(image, nipple)

        if image["view"] == "MLO":
            items += self._mlo_items(image, nipple_point, mm, view_key)
            items.append(mlo_alone(image, nipple_position(image, nipple, MIN_NIPPLE_PX), view_key))
        else:
            items.append(self._nipple_centred(breast, nipple_point, mm, view_key))
        gantry = (image.get("physics") or {}).get("gantry_deg")
        if gantry is not None:
            items.append(item("gantry", "Gantry angle", INFO, round(gantry, 1), "°", "from the DICOM header", view_key))
        return [entry for entry in items if entry is not None]

    def _mlo_items(self, image: Mapping[str, Any], nipple_point: Optional[List[float]], mm: float,
                   view_key: str) -> List[Dict[str, Any]]:
        items = []
        line = ((image.get("measures") or {}).get("mask") or {}).get("line")

        if line:
            (ax, ay), (bx, by) = line
            angle = math.degrees(math.atan2(abs(bx - ax), max(abs(by - ay), 1e-6)))
            items.append(item("pectoral_angle", "Pectoral angle", INFO, round(angle, 1), "°",
                              "angle of the fitted pectoral line from the vertical", view_key))
            bottom_y = max(ay, by)
        else:
            items.append(item("pectoral_angle", "Pectoral angle", NA, None, "°", "no pectoral line", view_key))
            bottom_y = None

        if bottom_y is not None and nipple_point is not None:
            # Positive means the pectoral shadow reaches below the nipple level, which is the classic MLO adequacy
            # criterion.
            depth = (bottom_y - nipple_point[1]) * mm
            items.append(item("pectoral_depth", "Pectoral reaches nipple level",
                              self.higher_is_better(depth, "pectoral_below_nipple_mm"),
                              round(depth, 1), "mm", "positive = pectoral extends below the nipple", view_key))
        else:
            items.append(item("pectoral_depth", "Pectoral reaches nipple level", NA, None, "mm",
                              "needs the pectoral and the nipple", view_key))
        return items

    @staticmethod
    def _nipple_centred(breast: Optional[np.ndarray], nipple_point: Optional[List[float]], mm: float,
                        view_key: str) -> Dict[str, Any]:
        """CC: vertical distance of the nipple from the middle of the breast's vertical extent (information only).

        The extent is the first to the last image row holding breast-tissue mask; on a CC that is the breast's
        height at the chest wall, where it is widest. The nipple is the nipple-mask centroid, else the pose point.
        """
        if breast is None or nipple_point is None:
            return item("nipple_centred", "Nipple centred", NA, None, "mm", "needs the nipple and the breast", view_key)
        rows = np.where(breast.any(axis=1))[0]
        if not rows.size:
            return item("nipple_centred", "Nipple centred", NA, None, "mm", "no breast tissue", view_key)
        top, bottom = float(rows.min()), float(rows.max())
        offset_px = abs(nipple_point[1] - (top + bottom) / 2.0)
        share = 100.0 * offset_px / max((bottom - top) / 2.0, 1.0)
        return item("nipple_centred", "Nipple centred", INFO, round(offset_px * mm, 1), "mm",
                    f"distance from the middle of the breast's height ({share:.0f}% of the half-height)", view_key)

    # ── across the study ─────────────────────────────────────────────────
    def for_study(self, images: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Study-level criteria. None since the left/right symmetry checks were removed; kept as the hook."""
        return []

    @staticmethod
    def summarise(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """How many graded items are good / acceptable / poor (information items and estimates are not counted)."""
        counts = {GOOD: 0, ACCEPTABLE: 0, POOR: 0, NA: 0}
        for entry in items:
            if entry.get("estimated") or entry["grade"] not in counts:
                continue
            counts[entry["grade"]] += 1
        return {"counts": counts, "graded": counts[GOOD] + counts[ACCEPTABLE] + counts[POOR]}


def mlo_alone(image: Mapping[str, Any], nipple_point: Optional[List[float]], view_key: str) -> Dict[str, Any]:
    """MLO-alone verdict (information only): does the posterior nipple line meet the pectoral line inside the image?

    The nipple is dropped perpendicularly onto the fitted pectoral line, taken as an infinite line; the foot of that
    perpendicular is mapped back to the original pixel grid and must lie within the image, widened on every side by
    0.25% of its longest side. Inside means adequate, outside inadequate. It never changes the ΔD verdict.
    """
    name = "MLO alone"
    line = ((image.get("measures") or {}).get("mask") or {}).get("line")
    if not line or nipple_point is None:
        return item("mlo_alone", name, NA, None, "", "needs the pectoral line and the nipple", view_key)
    geometry = image.get("transform") or {}
    try:
        scale = float(geometry["scale"])
        width, height = float(geometry["original_width"]), float(geometry["original_height"])
        shift = np.array([geometry["pad_left"], geometry["pad_top"]], float)
        origin = np.array([geometry["crop_x1"], geometry["crop_y1"]], float)
    except (KeyError, TypeError, ValueError):
        return item("mlo_alone", name, NA, None, "", "no image geometry stored for this study", view_key)

    def original(point):
        return (np.asarray(point, float) - shift) / scale + origin

    a, b, nipple = original(line[0]), original(line[1]), original(nipple_point)
    direction = b - a
    if not direction.any():
        return item("mlo_alone", name, NA, None, "", "needs the pectoral line and the nipple", view_key)
    foot = a + float((nipple - a) @ direction / (direction @ direction)) * direction
    margin = MLO_ALONE_MARGIN * max(width, height)
    inside = -margin <= foot[0] < width + margin and -margin <= foot[1] < height + margin
    if inside:
        return item("mlo_alone", name, INFO, "adequate", "",
                    "the posterior nipple line meets the pectoral line inside the image", view_key)
    beyond_px = float(np.hypot(max(0.0, -foot[0], foot[0] - width), max(0.0, -foot[1], foot[1] - height)))
    spacing = float(geometry.get("pixel_spacing") or 0.0)
    beyond = f", {beyond_px * spacing:.1f} mm beyond the edge" if spacing > 0 else ""
    return item("mlo_alone", name, INFO, "inadequate", "",
                f"the posterior nipple line meets the pectoral line outside the image{beyond}", view_key)


def nipple_position(image: Mapping[str, Any], nipple_mask: Optional[np.ndarray],
                    min_pixels: int = 1) -> Optional[List[float]]:
    """The nipple, preferring the segmentation centroid and falling back to the pose point."""
    if nipple_mask is not None and int(np.count_nonzero(nipple_mask)) >= min_pixels:
        ys, xs = np.nonzero(nipple_mask)
        return [float(xs.mean()), float(ys.mean())]
    point = ((image.get("landmarks") or {}).get("points") or {}).get("nipple")
    return [float(point[0]), float(point[1])] if point else None
