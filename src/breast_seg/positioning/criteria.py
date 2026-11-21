"""Positioning criteria beyond delta-D.

Delta-D answers one question — is the posterior tissue on the CC consistent with the MLO. Next to it the app grades
one more thing on the MLO alone, the MLO quality: the posterior nipple line must meet the pectoral line inside the
image (the criterion of the MLO-only RAUNet paper). It is good or poor and never changes the ΔD verdict. Three values
are shown as information only, with no grade: the pectoral angle (MLO), how far the nipple sits from the centre of the
breast on the CC, and the gantry angle the header reports.

Removed on 2026-11-20 because the radiologists do not grade them and their bands were never checked against a
reader: compression force / thickness / pressure, dose, focus, exposure, nipple in profile, pectoral visible on CC,
pectoral reaching the nipple level, inframammary fold, left/right symmetry, and the 0-100 criterion score.

No criterion has a configurable band: the MLO quality is a fixed geometric rule and the rest is information.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from ..geometry import compute_nipple_centroid
from ..imaging.preprocess import Transform
from .numbers import fixed, rounded

GOOD = "good"
ACCEPTABLE = "acceptable"
POOR = "poor"
NA = "na"
INFO = "info"  # a value shown for information: never graded, never counted

# MLO quality, the criterion of the MLO-only RAUNet paper (its evaluation.py): the foot of the perpendicular from the
# nipple to the infinite pectoral line must fall inside the image, with a margin of 0.25% of the longest side.
MLO_QUALITY_MARGIN = 0.0025

# Bumped whenever the set of criteria or the shape of an item changes, so stored studies are regraded when they are
# opened again. 4: the single graded MLO quality. 5: items carry ``code`` and ``params``. 6: the MLO quality carries
# its ``basis`` (the line and the nipple it was judged from); no ``estimated`` flag, no study-level items. 7: the
# MLO quality takes the nipple the measurement takes (the mask centroid whatever the mask's size). 8: every number is
# rounded once, by one rule (numbers.py); an older record's summary gets its unrounded millimetres back when read.
CRITERIA_VERSION = 8


# How many decimals a parameter of an explanation is written with (numbers.fixed); the interface writes them the same.
PARAM_DIGITS = {"mm": 1, "share": 0}

# Every explanation a criterion can carry, by code. The record stores the code and its parameters next to the English
# sentence, so the interface translates by code (i18n.js ``detail_<code>``) instead of matching the English wording:
# rewording a sentence here cannot silently lose its translation, and a test checks that every code has one.
DETAILS: Dict[str, str] = {
    "pectoral_angle": "angle of the fitted pectoral line from the vertical",
    "no_pectoral_line": "no pectoral line",
    "mlo_inside": "the posterior nipple line meets the pectoral line inside the image",
    "mlo_outside": "the posterior nipple line meets the pectoral line outside the image",
    "mlo_outside_by": "the posterior nipple line meets the pectoral line outside the image, {mm} mm beyond the edge",
    "needs_line_and_nipple": "needs the pectoral line and the nipple",
    "no_geometry": "no image geometry stored for this study",
    "nipple_centred": "distance from the middle of the breast's height ({share}% of the half-height)",
    "needs_nipple_and_breast": "needs the nipple and the breast",
    "no_breast_tissue": "no breast tissue",
    "from_header": "from the DICOM header",
}


def item(item_id: str, name: str, grade: str, value: Any = None, unit: str = "",
         code: str = "", view_key: str = "", **params: float) -> Dict[str, Any]:
    """One criterion as stored in a study record; ``code`` (a key of :data:`DETAILS`) and ``params`` explain it."""
    digits = {key: PARAM_DIGITS.get(key, 1) for key in params}
    return {"id": item_id, "name": name, "grade": grade, "value": value, "unit": unit,
            "detail": DETAILS[code].format(**{key: fixed(params[key], digits[key]) for key in params}) if code else "",
            "code": code, "params": {key: rounded(params[key], digits[key]) for key in params},
            "view_key": view_key}


class CriteriaGrader:
    """The criteria of one view, and their counts."""

    def for_image(self, image: Mapping[str, Any], masks: Mapping[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Criteria of one view; ``image`` must already carry ``measures`` (for the fitted pectoral line)."""
        view_key = image["view_key"]
        mm = image["mm_per_px"]
        items: List[Optional[Dict[str, Any]]] = []

        breast = masks.get("breast-tissue")
        nipple = masks.get("nipple")
        nipple_point = nipple_position(image, nipple)

        if image["view"] == "MLO":
            items.append(self._pectoral_angle(image, view_key))
            items.append(mlo_quality(image, nipple_point, view_key))
        else:
            items.append(self._nipple_centred(breast, nipple_point, mm, view_key))
        gantry = (image.get("physics") or {}).get("gantry_deg")
        if gantry is not None:
            items.append(item("gantry", "Gantry angle", INFO, rounded(gantry, 1), "°", "from_header", view_key))
        return [entry for entry in items if entry is not None]

    @staticmethod
    def _pectoral_angle(image: Mapping[str, Any], view_key: str) -> Dict[str, Any]:
        """MLO: angle of the fitted pectoral line from the vertical (information only)."""
        line = ((image.get("measures") or {}).get("mask") or {}).get("line")
        if not line:
            return item("pectoral_angle", "Pectoral angle", NA, None, "°", "no_pectoral_line", view_key)
        (ax, ay), (bx, by) = line
        angle = math.degrees(math.atan2(abs(bx - ax), max(abs(by - ay), 1e-6)))
        return item("pectoral_angle", "Pectoral angle", INFO, rounded(angle, 1), "°", "pectoral_angle", view_key)

    @staticmethod
    def _nipple_centred(breast: Optional[np.ndarray], nipple_point: Optional[List[float]], mm: float,
                        view_key: str) -> Dict[str, Any]:
        """CC: vertical distance of the nipple from the middle of the breast's vertical extent (information only).

        The extent is the first to the last image row holding breast-tissue mask; on a CC that is the breast's
        height at the chest wall, where it is widest. The nipple is the nipple-mask centroid, else the pose point.
        """
        if breast is None or nipple_point is None:
            return item("nipple_centred", "Nipple centred", NA, None, "mm", "needs_nipple_and_breast", view_key)
        rows = np.where(breast.any(axis=1))[0]
        if not rows.size:
            return item("nipple_centred", "Nipple centred", NA, None, "mm", "no_breast_tissue", view_key)
        top, bottom = float(rows.min()), float(rows.max())
        offset_px = abs(nipple_point[1] - (top + bottom) / 2.0)
        share = 100.0 * offset_px / max((bottom - top) / 2.0, 1.0)
        return item("nipple_centred", "Nipple centred", INFO, rounded(offset_px * mm, 1), "mm", "nipple_centred",
                    view_key, share=share)

    @staticmethod
    def summarise(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """How many graded items are good / acceptable / poor (information items are not counted)."""
        counts = {GOOD: 0, ACCEPTABLE: 0, POOR: 0, NA: 0}
        for entry in items:
            if entry["grade"] in counts:
                counts[entry["grade"]] += 1
        return {"counts": counts, "graded": counts[GOOD] + counts[ACCEPTABLE] + counts[POOR]}


def mlo_quality(image: Mapping[str, Any], nipple_point: Optional[List[float]], view_key: str) -> Dict[str, Any]:
    """MLO quality, graded good or poor: does the posterior nipple line meet the pectoral line inside the image?

    The nipple is dropped perpendicularly onto the fitted pectoral line, taken as an infinite line; the foot of that
    perpendicular is mapped back to the original pixel grid and must lie within the image, widened on every side by
    0.25% of its longest side. Inside is good, outside poor. It never changes the ΔD verdict.

    A graded item carries its ``basis`` in the 640 frame — the pectoral ``line``, the ``nipple`` that was used (see
    :func:`nipple_position`) and the ``foot`` of the perpendicular — so the review screen can draw what the verdict
    rests on when the segmentation route itself measured nothing (no nipple mask: the pose point was used).
    """
    name = "MLO quality"
    line = ((image.get("measures") or {}).get("mask") or {}).get("line")
    if not line or nipple_point is None:
        return item("mlo_quality", name, NA, None, "", "needs_line_and_nipple", view_key)
    try:
        transform = Transform(**(image.get("transform") or {}))
        width, height = float(transform.original_width), float(transform.original_height)
        if not float(transform.scale) > 0:
            raise ValueError("scale")
        a, b, nipple = transform.to_original(np.asarray([line[0], line[1], nipple_point], float))
    except (TypeError, ValueError):
        return item("mlo_quality", name, NA, None, "", "no_geometry", view_key)

    direction = b - a
    if not direction.any():
        return item("mlo_quality", name, NA, None, "", "needs_line_and_nipple", view_key)
    along = float((nipple - a) @ direction / (direction @ direction))
    foot = a + along * direction
    margin = MLO_QUALITY_MARGIN * max(width, height)
    inside = -margin <= foot[0] < width + margin and -margin <= foot[1] < height + margin
    if inside:
        graded = item("mlo_quality", name, GOOD, "good", "", "mlo_inside", view_key)
    else:
        beyond_px = float(np.hypot(max(0.0, -foot[0], foot[0] - width), max(0.0, -foot[1], foot[1] - height)))
        spacing = float(transform.pixel_spacing or 0.0)
        if spacing > 0:
            graded = item("mlo_quality", name, POOR, "poor", "", "mlo_outside_by", view_key, mm=beyond_px * spacing)
        else:
            graded = item("mlo_quality", name, POOR, "poor", "", "mlo_outside", view_key)
    # The frame change is a scale and a shift, so the foot sits at the same fraction of the line in the 640 frame.
    start, end = np.asarray(line[0], float), np.asarray(line[1], float)
    graded["basis"] = {"line": [[round(float(v), 1) for v in start], [round(float(v), 1) for v in end]],
                       "nipple": [round(float(v), 1) for v in nipple_point],
                       "foot": [round(float(v), 1) for v in start + along * (end - start)]}
    return graded


def nipple_position(image: Mapping[str, Any], nipple_mask: Optional[np.ndarray]) -> Optional[List[float]]:
    """The nipple of a view: the centroid of the nipple mask when there is one, the pose point only when there is
    none.

    The centroid is the one the segmentation route measures PNL and the chest-wall distance from
    (:func:`breast_seg.geometry.compute_nipple_centroid`), whatever the size of the mask, so a criterion never
    judges another nipple than the measurement next to it.
    """
    centre = compute_nipple_centroid(nipple_mask) if nipple_mask is not None else None
    if centre is not None:
        return [float(centre[0]), float(centre[1])]
    point = ((image.get("landmarks") or {}).get("points") or {}).get("nipple")
    return [float(point[0]), float(point[1])] if point else None
