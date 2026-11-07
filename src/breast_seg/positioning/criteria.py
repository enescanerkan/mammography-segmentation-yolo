"""Positioning criteria beyond delta-D.

Delta-D answers one question — is the posterior tissue on the CC consistent with the MLO. A radiographer judging
a film asks several more, and most of them are already implied by the masks the segmentation model produced, so
they cost nothing extra: the pectoral's angle and how far down it reaches, whether the inframammary fold is open,
whether the nipple is in profile, whether the CC is centred. Two more come straight from the pixels: focus and
exposure.

Every threshold is configuration (:data:`~breast_seg.positioning.config.DEFAULT_CRITERIA`). They are guidance
values, not a standard — the numbers a service audits against are its own.
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

RANK = {GOOD: 0, ACCEPTABLE: 1, POOR: 2, NA: -1}


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

    def lower_is_better(self, value: float, key: str) -> str:
        band = self._band(key)
        if not band:
            return NA
        if value <= float(band["good"]):
            return GOOD
        if value <= float(band["acceptable"]):
            return ACCEPTABLE
        return POOR

    def higher_is_better(self, value: float, key: str) -> str:
        band = self._band(key)
        if not band:
            return NA
        if value >= float(band["good"]):
            return GOOD
        if value >= float(band["acceptable"]):
            return ACCEPTABLE
        return POOR

    def in_range(self, value: float, key: str) -> str:
        band = self._band(key)
        if not band:
            return NA
        good_low, good_high = band["good"]
        ok_low, ok_high = band["acceptable"]
        if good_low <= value <= good_high:
            return GOOD
        if ok_low <= value <= ok_high:
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
        pectoral = masks.get("pectoral")
        nipple_point = nipple_position(image, nipple)

        if image["view"] == "MLO":
            items += self._mlo_items(image, breast, pectoral, nipple_point, mm, view_key)
        else:
            items += self._cc_items(breast, pectoral, nipple_point, mm, view_key)

        items.append(self._nipple_profile(breast, nipple_point, image["chest_side"], mm, view_key))
        items += self._quality_items(image.get("quality") or {}, view_key)
        items += self._physics_items(image.get("physics") or {}, breast, mm, view_key)
        return [entry for entry in items if entry is not None]

    def _physics_items(self, physics: Mapping[str, Any], breast: Optional[np.ndarray], mm: float,
                       view_key: str) -> List[Dict[str, Any]]:
        """Compression pressure, and the header numbers it is built from.

        Pressure rather than force, because force alone is misleading: the same 80 N is firm on a small breast and
        slack on a large one. Pressure needs a contact area, and the segmentation already measures one.

        Marked ESTIMATED, and deliberately kept out of the score. The area divided by is the *segmented breast
        tissue*, not the paddle's true contact area — the two differ by an amount nobody has calibrated here. The
        number is useful for ranking and for spotting an obviously slack compression; it is not a device
        measurement.
        """
        force = physics.get("compression_n")
        if not force:
            return []

        items = [item("compression_force", "Compression force", NA, round(force, 1), "N",
                      "from the DICOM header", view_key)]

        thickness = physics.get("thickness_mm")
        if thickness:
            items.append(item("thickness", "Compressed thickness", NA, round(thickness, 0), "mm",
                              "from the DICOM header", view_key))

        if breast is not None and breast.any():
            area_mm2 = float(breast.sum()) * (mm ** 2)
            if area_mm2 > 0:
                kpa = force / (area_mm2 * 1e-6) / 1000.0
                items.append(item(
                    "compression_pressure", "Compression pressure",
                    self.in_range(kpa, "compression_kpa"), round(kpa, 1), "kPa",
                    f"estimated: force ÷ segmented tissue area ({area_mm2 / 100.0:.0f} cm²), not the paddle's "
                    "true contact area",
                    view_key, estimated=True))

        gantry = physics.get("gantry_deg")
        if gantry:
            items.append(item("gantry", "Gantry angle", NA, round(gantry, 1), "°",
                              "the angle the pectoral is expected to follow", view_key))

        dose = physics.get("entrance_dose_mgy")
        if dose:
            items.append(item("dose", "Entrance dose", NA, round(dose, 2), "mGy",
                              "from the DICOM header", view_key))
        return items

    def _mlo_items(self, image: Mapping[str, Any], breast: Optional[np.ndarray], pectoral: Optional[np.ndarray],
                   nipple_point: Optional[List[float]], mm: float, view_key: str) -> List[Dict[str, Any]]:
        items = []
        line = ((image.get("measures") or {}).get("mask") or {}).get("line")

        if line:
            (ax, ay), (bx, by) = line
            angle = math.degrees(math.atan2(abs(bx - ax), max(abs(by - ay), 1e-6)))
            items.append(item("pectoral_angle", "Pectoral angle", self.in_range(angle, "pectoral_angle_deg"),
                              round(angle, 1), "°", "20–35° expected", view_key))
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

        if breast is not None and pectoral is not None:
            rows = np.where(pectoral.any(axis=1))[0]
            if rows.size:
                below = breast[int(rows.max()):, :]
                open_fold = bool(below.sum() > 200)
                height_mm = round(float(below.sum()) / max(breast.shape[1], 1) * mm, 1)
                items.append(item("imf", "Inframammary fold open", GOOD if open_fold else POOR,
                                  height_mm if open_fold else 0.0, "mm",
                                  "breast tissue visible below the pectoral", view_key))
        return items

    def _cc_items(self, breast: Optional[np.ndarray], pectoral: Optional[np.ndarray],
                  nipple_point: Optional[List[float]], mm: float, view_key: str) -> List[Dict[str, Any]]:
        items = []
        if breast is not None and nipple_point is not None:
            rows = np.where(breast.any(axis=1))[0]
            if rows.size:
                centre = (float(rows.min()) + float(rows.max())) / 2.0
                offset = abs(nipple_point[1] - centre) * mm
                items.append(item("nipple_centred", "Nipple centred",
                                  self.lower_is_better(offset, "nipple_centred_mm"),
                                  round(offset, 1), "mm", "vertical distance from the breast's mid-line", view_key))
        items.append(item("cc_pectoral", "Pectoral visible on CC", GOOD if pectoral is not None else NA,
                          None, "", "a bonus criterion, not a requirement", view_key))
        return items

    def _nipple_profile(self, breast: Optional[np.ndarray], nipple_point: Optional[List[float]], chest_side: str,
                        mm: float, view_key: str) -> Dict[str, Any]:
        """How far the nipple sits from the breast's anterior edge in its own row band."""
        if breast is None or nipple_point is None:
            return item("nipple_profile", "Nipple in profile", NA, None, "mm",
                        "needs the nipple and the breast contour", view_key)

        x, y = nipple_point
        row = int(round(y))
        band = breast[max(0, row - 8):min(breast.shape[0], row + 9), :]
        columns = np.where(band.any(axis=0))[0]
        if not columns.size:
            return item("nipple_profile", "Nipple in profile", NA, None, "mm",
                        "no tissue at the nipple's level", view_key)

        anterior = float(columns.max()) if chest_side == "L" else float(columns.min())
        deviation = abs(anterior - x) * mm
        return item("nipple_profile", "Nipple in profile", self.lower_is_better(deviation, "nipple_profile_mm"),
                    round(deviation, 1), "mm", "distance from the anterior edge of the breast", view_key)

    def _quality_items(self, quality: Mapping[str, Any], view_key: str) -> List[Dict[str, Any]]:
        if not quality:
            return []
        sharpness = float(quality.get("sharpness") or 0.0)
        clipped = float(quality.get("clipped_fraction") or 0.0)
        return [
            item("sharpness", "Focus", self.higher_is_better(sharpness, "sharpness"),
                 round(sharpness, 1), "", "variance of Laplacian over the tissue", view_key),
            item("exposure", "Exposure", self.lower_is_better(clipped, "clipped_fraction"),
                 round(clipped * 100, 2), "%", "share of saturated tissue pixels", view_key),
        ]

    # ── across the study ─────────────────────────────────────────────────
    def for_study(self, images: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Left-versus-right comparisons, which no single image can answer."""
        by_key = {image["view_key"]: image for image in images}
        items = []
        for view, name in (("MLO", "PNL symmetry"), ("CC", "Chest-wall distance symmetry")):
            left = by_key.get(f"L-{view}")
            right = by_key.get(f"R-{view}")
            # A failed route still leaves a {"ok": False} dict behind, so the fallback has to test "ok" rather than
            # presence. Both sides come from the same route — comparing a mask millimetre with a landmark one would
            # measure the methods.
            measures = None
            for route in ("mask", "landmark"):
                pair = [((image or {}).get("measures") or {}).get(route) or {} for image in (left, right)]
                if all(m.get("ok") for m in pair):
                    measures = pair
                    break
            if measures is None:
                items.append(item(f"symmetry_{view.lower()}", name, NA, None, "mm", "needs both sides measured", ""))
                continue
            gap = abs(measures[0]["mm"] - measures[1]["mm"])
            items.append(item(f"symmetry_{view.lower()}", name, self.lower_is_better(gap, "symmetry_mm"),
                              round(gap, 1), "mm", f"left versus right on {view}", ""))
        return items

    @staticmethod
    def summarise(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
        """Counts, worst grade and a 0-100 score. The score counts only measurements, never estimates.

        An estimated value can still be graded and shown — it is worth seeing — but it must not move a number
        somebody will quote, because its uncertainty has not been characterised.
        """
        counts = {GOOD: 0, ACCEPTABLE: 0, POOR: 0, NA: 0}
        for entry in items:
            if entry.get("estimated"):
                continue
            counts[entry["grade"]] = counts.get(entry["grade"], 0) + 1
        graded = counts[GOOD] + counts[ACCEPTABLE] + counts[POOR]
        worst = POOR if counts[POOR] else (ACCEPTABLE if counts[ACCEPTABLE] else GOOD)
        return {
            "counts": counts,
            "graded": graded,
            "worst": worst if graded else NA,
            "score": round(100.0 * (counts[GOOD] + 0.5 * counts[ACCEPTABLE]) / graded, 0) if graded else None,
        }


def nipple_position(image: Mapping[str, Any], nipple_mask: Optional[np.ndarray]) -> Optional[List[float]]:
    """The nipple, preferring the segmentation centroid and falling back to the pose point."""
    if nipple_mask is not None and nipple_mask.any():
        ys, xs = np.nonzero(nipple_mask)
        return [float(xs.mean()), float(ys.mean())]
    point = ((image.get("landmarks") or {}).get("points") or {}).get("nipple")
    return [float(point[0]), float(point[1])] if point else None
