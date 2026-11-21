"""Validation of reviewer edits (labels and rules) before they touch a study record.

Only the shapes the editor draws are accepted; anything else in a request body is ignored, and a field with the
wrong shape is rejected rather than treated as empty — a labels update replaces a view's labels wholesale, so
silently emptying a malformed field would wipe them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
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


def fingerprint(points: Any) -> str:
    """16 hex characters naming a contour's points (rounded to 0.01 px, as stored): the ``"cleaned"`` mark that
    "Smooth / clean labels" leaves on the contours it returned (:func:`breast_seg.positioning.label_cleanup.clean_view`).
    """
    rounded = [[round(float(x), 2), round(float(y), 2)] for x, y in points]
    return hashlib.sha1(json.dumps(rounded).encode("ascii")).hexdigest()[:16]


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
    cleaned = {
        "points": points,
        "confidence": _number(entry.get("confidence") or 0.0, 0.0),
        "rescued": bool(entry.get("rescued")),
        "brightened": bool(entry.get("brightened")),
        "edited": bool(entry.get("edited")),
    }
    # The fingerprint "Smooth / clean labels" leaves on the contours it returned (label_cleanup.clean_view).
    mark = entry.get("cleaned")
    if isinstance(mark, str) and re.fullmatch(r"[0-9a-f]{16}", mark):
        cleaned["cleaned"] = mark
    return cleaned


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


def _lesion_key(lesion: Any) -> Optional[tuple]:
    """What makes two lesions the same label: box, class and confidence, at the cleaned precision."""
    cleaned = _clean_lesion(lesion)
    if cleaned is None:
        return None
    return tuple(cleaned["box"]), cleaned["label"], round(cleaned["confidence"], 4)


def same_labels(image: Mapping[str, Any], update: Mapping[str, Any]) -> bool:
    """True when ``update`` (cleaned) carries exactly the view's stored labels, compared at the cleaned precision.

    A view sent back unchanged — an edit undone, a vertex dragged back — is not an edit and must not be marked as one.
    """
    stored_polygons = {}
    for name, shape in (image.get("polygons") or {}).items():
        polygon = clean_polygon(shape) if name in EDITABLE_CLASSES else None
        if polygon:
            stored_polygons[name] = polygon["points"]
    stored_points = {}
    for name, value in ((image.get("landmarks") or {}).get("points") or {}).items():
        point = clean_point(value) if name in EDITABLE_POINTS else None
        if point:
            stored_points[name] = point
    stored_lesions = [key for key in (_lesion_key(lesion) for lesion in image.get("lesions") or []) if key]
    return (stored_polygons == {name: shape["points"] for name, shape in update["polygons"].items()}
            and stored_points == dict(update["points"])
            and stored_lesions == [_lesion_key(lesion) for lesion in update["lesions"]])


def _reconcile(image: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """``update`` with everything the reviewer did not touch put back as it was stored.

    * A contour whose points equal the stored ones stays the stored contour, whatever flags came with it: a vertex
      dragged away and dropped back is not an edit, and a client cannot hand a model contour back as edited, change
      its confidence or drop the "rescued" / "found after brightening" marks that ask the reader to check it.
    * A contour whose points differ is the reviewer's now: it is no longer "rescued" or "found after brightening"
      (the reviewer has just set the position those marks ask about).
    * The ``"cleaned"`` mark of "Smooth / clean labels" is kept in both cases when it names the contour's points
      (it is a fingerprint of them, so a client cannot attach it to anything else).
    * A lesion sent back with the same box, class and confidence stays the stored one, with its id and without an
      ``edited`` mark; only a changed or a new one is taken as cleaned.
    """
    polygons = {}
    for name, polygon in update["polygons"].items():
        kept = (image.get("polygons") or {}).get(name)
        stored = clean_polygon(kept)
        mark = polygon.get("cleaned")
        if mark is not None and mark != fingerprint(polygon["points"]):
            polygon = {key: value for key, value in polygon.items() if key != "cleaned"}
            mark = None
        if stored is not None and stored["points"] == polygon["points"]:
            polygons[name] = {**kept, "cleaned": mark} if mark and kept.get("cleaned") != mark else kept
        else:
            polygons[name] = {**polygon, "rescued": False, "brightened": False}

    kept = [(key, lesion) for key, lesion in ((_lesion_key(lesion), lesion) for lesion in image.get("lesions") or [])
            if key]
    lesions = []
    for lesion in update["lesions"]:
        match = next((index for index, (key, _) in enumerate(kept) if key == _lesion_key(lesion)), None)
        lesions.append(kept.pop(match)[1] if match is not None else lesion)
    return {"polygons": polygons, "points": update["points"], "lesions": lesions}


def apply_labels(record: Dict[str, Any], cleaned: Mapping[str, Mapping[str, Any]]) -> int:
    """Replace the labels of every view named in ``cleaned``; returns how many views matched.

    A matched view whose labels are unchanged is left as it is, so it keeps its model-made status; in a changed
    view, what was not touched keeps its stored form (see :func:`_reconcile`).
    """
    touched = 0
    changed = 0
    for image in record["images"]:
        update = cleaned.get(image["view_key"])
        if update is None:
            continue
        touched += 1
        if same_labels(image, update):
            continue
        changed += 1
        update = _reconcile(image, update)
        image["polygons"] = update["polygons"]
        image["landmarks"] = {
            **(image.get("landmarks") or {}),
            "points": update["points"],
            "detected": bool(update["points"]),
            "edited": True,
        }
        image["lesions"] = update["lesions"]
    if changed:
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
