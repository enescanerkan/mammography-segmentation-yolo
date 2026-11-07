"""Validation of reviewer edits (labels and rules) before they touch a study record.

Only the shapes the editor draws are accepted; anything else in a request body is ignored, and a field with the
wrong shape is rejected rather than treated as empty — a labels update replaces a view's labels wholesale, so
silently emptying a malformed field would wipe them.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Mapping, Optional
from uuid import uuid4

from .config import SEVERE_BOUNDS_MM, THRESHOLD_BOUNDS_MM

FRAME = 640
EDITABLE_CLASSES = ("breast-tissue", "pectoral", "nipple")
EDITABLE_POINTS = ("nipple", "pectoral_top", "pectoral_bottom")
MAX_POLYGON_POINTS = 4000
MAX_LESIONS = 64


class LabelError(ValueError):
    """The request body is malformed; the message is user-facing."""


def clean_point(value: Any) -> Optional[List[float]]:
    """``[x, y]`` clamped to the frame and rounded to 0.01, or None when it is not a finite pair."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [round(min(max(x, 0.0), FRAME), 2), round(min(max(y, 0.0), FRAME), 2)]


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _of_type(value: Any, kind: type, default: Any, what: str) -> Any:
    """The field, or its default when absent / empty; a :class:`LabelError` when it has the wrong shape."""
    if not value:  # absent, or an empty [] / {} either way
        return default
    if not isinstance(value, kind):
        raise LabelError(f"'{what}' has the wrong type.")
    return value


def clean_polygon(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    raw_points = entry.get("points")
    points = [p for p in (clean_point(v) for v in (raw_points if isinstance(raw_points, list) else [])) if p]
    if len(points) < 3 or len(points) > MAX_POLYGON_POINTS:
        return None
    return {
        "points": points,
        "confidence": _number(entry.get("confidence") or 0.0, 0.0),
        "rescued": bool(entry.get("rescued")),
        "edited": bool(entry.get("edited")),
    }


def _clean_lesion(lesion: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(lesion, dict) or not isinstance(lesion.get("box"), list):
        return None
    box = [clean_point(lesion["box"][0:2]), clean_point(lesion["box"][2:4])]
    if not all(box):
        return None
    (x1, y1), (x2, y2) = box
    if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
        return None
    # The id is what the editor's instance list and its delete button address, so it has to survive the
    # round-trip; anything unrecognised gets a new one.
    raw_id = str(lesion.get("id") or "")[:40]
    return {
        "id": raw_id if raw_id.replace("-", "").replace("_", "").isalnum() else uuid4().hex[:12],
        "label": str(lesion.get("label") or "Mass")[:40],
        "box": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        "confidence": _number(lesion.get("confidence") or 1.0, 1.0),
        "edited": True,
    }


def clean_labels(payload: Any) -> Dict[str, Dict[str, Any]]:
    """``{"images": [{view_key, polygons, landmarks: {points}, lesions}]}`` -> ``{view_key: cleaned}``."""
    if not isinstance(payload, dict):
        raise LabelError("Expected an object with an 'images' list.")
    incoming = payload.get("images")
    if not isinstance(incoming, list) or not incoming:
        raise LabelError("No images in the request body.")

    cleaned: Dict[str, Dict[str, Any]] = {}
    for entry in incoming:
        if not isinstance(entry, dict):
            continue
        view_key = entry.get("view_key")
        if not isinstance(view_key, str):
            continue

        polygons = {}
        for name, shape in _of_type(entry.get("polygons"), dict, {}, "polygons").items():
            if name in EDITABLE_CLASSES:
                polygon = clean_polygon(shape)
                if polygon:
                    polygons[name] = polygon

        points = {}
        landmarks = _of_type(entry.get("landmarks"), dict, {}, "landmarks")
        for name, value in _of_type(landmarks.get("points"), dict, {}, "landmarks.points").items():
            if name in EDITABLE_POINTS:
                point = clean_point(value)
                if point:
                    points[name] = point

        lesions = []
        for lesion in _of_type(entry.get("lesions"), list, [], "lesions")[:MAX_LESIONS]:
            box = _clean_lesion(lesion)
            if box:
                lesions.append(box)

        cleaned[view_key] = {"polygons": polygons, "points": points, "lesions": lesions}
    return cleaned


def apply_labels(record: Dict[str, Any], cleaned: Mapping[str, Mapping[str, Any]]) -> int:
    """Replace the labels of every view named in ``cleaned``; returns how many views matched."""
    touched = 0
    for image in record["images"]:
        update = cleaned.get(image["view_key"])
        if update is None:
            continue
        image["polygons"] = update["polygons"]
        image["landmarks"] = {
            **(image.get("landmarks") or {}),
            "points": update["points"],
            "detected": bool(update["points"]),
            "edited": True,
        }
        image["lesions"] = update["lesions"]
        touched += 1
    if touched:
        record["edited"] = time.time()
    return touched


def clean_rules(current: Mapping[str, float], payload: Any) -> Dict[str, float]:
    """Merge ``{"threshold_mm"?, "severe_mm"?}`` into ``current`` within bounds; severe never below threshold."""
    if not isinstance(payload, dict):
        raise LabelError("Expected an object with threshold_mm and/or severe_mm.")
    rules = {"threshold_mm": float(current["threshold_mm"]), "severe_mm": float(current["severe_mm"])}
    for key, (low, high) in (("threshold_mm", THRESHOLD_BOUNDS_MM), ("severe_mm", SEVERE_BOUNDS_MM)):
        if key in payload and payload[key] is not None:
            try:
                value = float(payload[key])
            except (TypeError, ValueError):
                raise LabelError(f"{key} must be a number.") from None
            if not (low <= value <= high):
                raise LabelError(f"{key} must be between {low:g} and {high:g} mm.")
            rules[key] = value
    if rules["severe_mm"] < rules["threshold_mm"]:
        rules["severe_mm"] = rules["threshold_mm"]
    return rules
