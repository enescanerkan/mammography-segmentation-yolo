"""From engine output to a study record, and back again after every edit.

A *study record* is the JSON the review app stores and serves: one entry per view (labels in the 640 frame, both
measurement routes, criteria) plus the per-breast verdicts, the study-level criteria and the rules they were
decided with. Every number in it is recomputed from the labels it currently carries, so an edited contour or a
moved threshold always moves the verdict.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional

from . import measures
from .config import Rules
from .criteria import CRITERIA_VERSION, CriteriaGrader

FRAME = 640

# The review grid's order: the right breast on the reader's left, MLO above CC. The report, the recent list and the
# page (main.js VIEW_ORDER, kept equal by a test) list views in it rather than in upload order.
VIEW_ORDER = ("R-MLO", "L-MLO", "R-CC", "L-CC")


def view_rank(view_key: Any) -> int:
    """Where a view goes in :data:`VIEW_ORDER` (anything else after it)."""
    return VIEW_ORDER.index(view_key) if view_key in VIEW_ORDER else len(VIEW_ORDER)


class PositioningAssessor:
    """Builds image entries and (re)computes measurements, criteria and verdicts of a study record.

    Args:
        rules: default delta-D thresholds (a record keeps its own once created).
        grader: criterion grader.
        contour_fallback: measure the segmentation route at the farthest breast-tissue contour point when a view
            has no nipple contour (:func:`measures.measure_mask`); such a measurement is marked estimated.
    """

    def __init__(self, rules: Optional[Rules] = None, grader: Optional[CriteriaGrader] = None,
                 contour_fallback: bool = False) -> None:
        self.rules = rules or Rules()
        self.grader = grader or CriteriaGrader()
        self.contour_fallback = bool(contour_fallback)

    def build_image(self, view_key: str, meta: Mapping[str, Any], analysis: Mapping[str, Any]) -> Dict[str, Any]:
        """One view's entry from the DICOM summary and :meth:`LandmarkEngine.analyse` output (measured)."""
        image = {
            "view_key": view_key,
            "view": meta["view"],
            "laterality": meta["laterality"],
            "chest_side": meta["chest_side"],
            "view_source": meta.get("view_source", "ViewPosition"),
            "laterality_source": meta.get("laterality_source", "ImageLaterality"),
            "mm_per_px": analysis["mm_per_px"],
            "pixel_spacing_known": meta["pixel_spacing_known"],
            "manufacturer": meta["manufacturer"] or meta["model_name"],
            "source_size": [meta["original_width"], meta["original_height"]],
            "landmarks": analysis["landmarks"],
            "polygons": analysis["polygons"],
            "lesions": analysis["lesions"],
            "physics": meta.get("physics") or {},
            "transform": analysis.get("transform"),  # 640 frame -> original pixels (the MLO-alone verdict needs it)
        }
        return self.remeasure(image)

    def remeasure(self, image: Dict[str, Any]) -> Dict[str, Any]:
        """(Re)compute both routes and every criterion from whatever labels the image now carries."""
        masks = measures.masks_from_polygons(image.get("polygons") or {}, FRAME)
        image["masks"] = masks
        image["measures"] = {
            "landmark": measures.measure_landmark(image),
            "mask": measures.measure_mask(image, self.contour_fallback),
        }
        image["criteria"] = self.grader.for_image(image, masks)
        image.pop("masks", None)
        return image

    def rules_of(self, record: Mapping[str, Any]) -> Dict[str, float]:
        """The record's rules, with the assessor defaults filling any gap."""
        rules = record.get("rules") or {}
        return {
            "threshold_mm": float(rules.get("threshold_mm", self.rules.threshold_mm)),
            "severe_mm": float(rules.get("severe_mm", self.rules.severe_mm)),
        }

    def recompute(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Remeasure every image, then the verdicts and the criteria (in place)."""
        rules = self.rules_of(record)
        for image in record["images"]:
            self.remeasure(image)
        record["summary"] = measures.summarise(record["images"], rules["threshold_mm"], rules["severe_mm"])
        record["criteria"] = self._criteria_block(record)
        record["rules"] = rules
        return record

    def regrade(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Recompute the criteria only, from the labels and measurements the record already holds (in place).

        For records saved under an older criteria set: the ΔD measurements, the verdicts and the rules are left
        exactly as stored — only the criteria lists and their counts are rebuilt, and the summary's millimetres,
        once stored rounded to two decimals, are restored from the per-view measurements they were computed from
        (:func:`measures.restore_precision`), so the card and the image show the same number.
        """
        for image in record["images"]:
            masks = measures.masks_from_polygons(image.get("polygons") or {}, FRAME)
            image["criteria"] = self.grader.for_image(image, masks)
        record["criteria"] = self._criteria_block(record)
        if isinstance(record.get("summary"), dict):
            measures.restore_precision(record["summary"], record["images"])
        return record

    def _criteria_block(self, record: Mapping[str, Any]) -> Dict[str, Any]:
        every = [entry for image in record["images"] for entry in image.get("criteria", [])]
        return {"summary": self.grader.summarise(every), "version": CRITERIA_VERSION}

    def new_record(self, study_id: str, images: List[Dict[str, Any]], problems: List[str],
                   models: Mapping[str, str], preview_size: int, created: Optional[float] = None) -> Dict[str, Any]:
        """A fresh, fully computed study record."""
        record = {
            "id": study_id,
            "created": time.time() if created is None else created,
            "images": [public_image(image) for image in images],
            "problems": list(problems),
            "rules": self.rules.as_dict(),
            "models": dict(models),
            "preview_size": int(preview_size),
        }
        return self.recompute(record)


def public_image(image: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop raw mask arrays — the polygons carry the same shape in a few kilobytes."""
    return {key: value for key, value in image.items() if key != "masks"}


def labels_of(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Just the coordinates, in the 640 frame — the shape a label pipeline reads."""
    return {
        "id": record["id"],
        "frame": {"width": FRAME, "height": FRAME},
        "images": [{
            "view_key": image["view_key"],
            "mm_per_px": image["mm_per_px"],
            "landmarks": image["landmarks"],
            "polygons": image["polygons"],
            "lesions": image["lesions"],
        } for image in record["images"]],
    }
