"""Clean and smooth the segmentation labels of one view (the editor's "Smooth / clean labels").

A label map is the 640 x 640 frame with 0 background, 1 pectoral muscle, 2 breast tissue and 3 nipple. The rules,
applied by :func:`clean_label_map`:

1. Pectoral muscle and breast tissue keep their largest 8-connected component, with its holes filled. The nipple
   keeps the component that touches the breast-tissue front edge (the largest of those within 2 px), or failing
   that the one nearest to the breast tissue; its holes are filled too.
2. Discarded pixels (islands, pixels of a class that lost) inside the foreground envelope take the label of the
   nearest kept class; outside it they become background. The envelope is the filled union of what was kept, plus
   the filled union of the input labels where it lies within 2 px of what was kept. So breast-tissue islands inside
   the pectoral muscle become pectoral, pectoral specks inside the breast become breast, and no gap is left at the
   border between them.
3. MLO only, with the image: the unlabelled strip between the pectoral muscle and the breast tissue that rule 1
   cannot fill because it is open to the image edge, within 25 px of both classes and brighter than 15 (of 255,
   i.e. tissue rather than air or padding), also goes to the nearest class.
4. Smoothing: every class's one-hot layer is blurred with a Gaussian (sigma 3 px, the nipple 2 px), the arg-max is
   taken, and rules 1-3 run once more for the small islands the smoothing can open.
5. A CC view has no pectoral class: pectoral pixels on a CC join the breast tissue.

:func:`clean_polygons` does the same on the stored contours of one view: they are painted into a label map in the
order the datasets use (breast tissue, then pectoral over it, then the nipple on top), cleaned, and each class that
changed is traced back into a contour the way the engine traces model masks.

:func:`clean_view` is the editor's button on top of that. Whether anything changed is decided on label maps: the
contours as sent and the cleaned contours are painted, and when the two maps are pixel-identical nothing changed
and the contours are kept exactly as sent. The smoothing (rule 4) is not idempotent: run again on its own result it
shaves another few pixels off every edge (10 to 100 per view on the public CMMD test images), and tracing adds a
pixel here and there. So every contour of a cleaned view carries a fingerprint of its points (``"cleaned"``); a view
whose contours all still match theirs is exactly what a clean produced, and cleaning it again is a no-op. Any edit
of a contour changes its points, and the view is then cleaned for real.

The rules need SciPy (``scipy.ndimage``), which every installation that runs the models has (Ultralytics depends on
it); it is imported only when a label map is cleaned.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from .labels import fingerprint

FRAME = 640
BACKGROUND, PECTORAL, BREAST, NIPPLE = 0, 1, 2, 3

#: Stored polygon class -> label value, in painting order: breast tissue first, pectoral over it, nipple on top
#: (the order of :data:`breast_seg.config.PAINT_ORDER`).
CLASS_VALUES: Dict[str, int] = {"breast-tissue": BREAST, "pectoral": PECTORAL, "nipple": NIPPLE}

#: Gaussian sigma (px) of each label value's one-hot layer in the smoothing step.
SIGMA: Dict[int, float] = {BACKGROUND: 3.0, PECTORAL: 3.0, BREAST: 3.0, NIPPLE: 2.0}
#: A nipple component this close to the breast tissue (px) counts as touching it.
TOUCH_PX = 2.0
#: MLO strip rule: unlabelled pixels within this distance (px) of both the pectoral muscle and the breast tissue ...
SEAM_PX = 25
#: ... and brighter than this (0-255 image) are tissue, not air or padding.
TISSUE_LEVEL = 15


def _ndimage() -> Any:
    from scipy import ndimage

    return ndimage


def _largest(mask: np.ndarray) -> np.ndarray:
    """The largest 8-connected component of ``mask`` (boolean; empty in, empty out)."""
    if not mask.any():
        return mask.astype(bool)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _nipple_component(nipple: np.ndarray, breast: np.ndarray) -> np.ndarray:
    """The nipple component to keep: the largest of those touching the breast tissue (within :data:`TOUCH_PX`),
    otherwise the nearest one (the larger on a tie); the largest when there is no breast tissue."""
    if not nipple.any():
        return nipple.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(nipple.astype(np.uint8), connectivity=8)
    if count == 2:
        return labels == 1
    if breast.any():
        distance = _ndimage().distance_transform_edt(~breast)
        candidates = sorted((float(distance[labels == k].min()), -int(stats[k, cv2.CC_STAT_AREA]), k)
                            for k in range(1, count))
        touching = [entry for entry in candidates if entry[0] <= TOUCH_PX]
        chosen = min(touching, key=lambda entry: entry[1]) if touching else candidates[0]
        return labels == chosen[2]
    return _largest(nipple)


def _components(label: np.ndarray, view: str, image: Optional[np.ndarray] = None) -> np.ndarray:
    """Rules 1, 2, 3 and 5 on one label map (a new array)."""
    ndimage = _ndimage()
    mlo = view == "MLO"
    pectoral = _largest(label == PECTORAL) if mlo else np.zeros_like(label, bool)
    breast = _largest(np.isin(label, (BREAST,) if mlo else (PECTORAL, BREAST)))
    pectoral = ndimage.binary_fill_holes(pectoral)
    breast = _largest(ndimage.binary_fill_holes(breast) & ~pectoral)
    nipple = ndimage.binary_fill_holes(_nipple_component(label == NIPPLE, breast))

    out = np.zeros_like(label)
    out[breast] = BREAST
    out[pectoral] = PECTORAL
    out[nipple] = NIPPLE

    kept = out > 0
    envelope = ndimage.binary_fill_holes(kept) | (
        ndimage.binary_fill_holes(label > 0) & ndimage.binary_dilation(kept, iterations=2))
    if image is not None and pectoral.any() and breast.any():
        near = ((ndimage.distance_transform_edt(~pectoral) <= SEAM_PX)
                & (ndimage.distance_transform_edt(~breast) <= SEAM_PX))
        envelope |= near & (image > TISSUE_LEVEL)
    empty = envelope & (out == 0)
    if empty.any() and kept.any():
        _, (rows, cols) = ndimage.distance_transform_edt(out == 0, return_indices=True)
        out[empty] = out[rows[empty], cols[empty]]
    return out


def smooth(label: np.ndarray) -> np.ndarray:
    """Rule 4 without the second pass: arg-max of the Gaussian-blurred one-hot layers."""
    layers = np.stack([cv2.GaussianBlur((label == value).astype(np.float32), (0, 0), SIGMA[value])
                       for value in range(4)])
    return layers.argmax(0).astype(np.uint8)


def _frame_image(image: Optional[np.ndarray], shape: Any) -> Optional[np.ndarray]:
    """``image`` as a grey ``uint8`` array of ``shape`` (a larger render of the same frame is area-reduced)."""
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.shape[:2] != tuple(shape):
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return image


def clean_label_map(label: np.ndarray, view: str, image: Optional[np.ndarray] = None) -> np.ndarray:
    """The cleaned and smoothed copy of ``label`` (rules 1-5 of the module docstring).

    Args:
        label: ``uint8`` label map (0 background, 1 pectoral, 2 breast tissue, 3 nipple).
        view: ``"MLO"``; any other value is treated as a CC (no pectoral class).
        image: the grey image of the same frame (0-255). Only the MLO strip rule reads it; without it that rule is
            skipped. A render of the frame at another size is resized to the label map first.
    """
    label = np.asarray(label, dtype=np.uint8)
    image = _frame_image(image, label.shape)
    once = _components(label, view, image)
    return _components(smooth(once), view, image)


def label_map(polygons: Mapping[str, Any], size: int = FRAME) -> np.ndarray:
    """Paint stored contours into a label map: breast tissue, then the pectoral muscle, then the nipple on top.

    Each contour is filled the way :func:`breast_seg.positioning.measures.masks_from_polygons` fills it for the
    measurements (vertices rounded to the nearest pixel); a contour of fewer than three points paints nothing.
    """
    out = np.zeros((size, size), np.uint8)
    for name, value in CLASS_VALUES.items():
        points = ((polygons or {}).get(name) or {}).get("points") or []
        if len(points) < 3:
            continue
        layer = np.zeros((size, size), np.uint8)
        cv2.fillPoly(layer, [np.asarray(points, dtype=np.float64).round().astype(np.int32)], 1)
        out[layer > 0] = value
    return out


def _is_cleaned(entry: Any) -> bool:
    return isinstance(entry, Mapping) and bool(entry.get("points")) and entry.get("cleaned") == fingerprint(
        entry["points"])


def clean_view(polygons: Mapping[str, Any], view: str, image: Optional[np.ndarray] = None,
               size: int = FRAME) -> Tuple[Dict[str, Any], List[str]]:
    """``(polygons, changed)``: the editor's "Smooth / clean labels" on one view.

    ``changed`` names the classes whose contour was replaced or removed. It is empty, and ``polygons`` are the
    contours exactly as given, when the label map of the cleaned contours is pixel-identical to the label map of the
    given ones, or when every contour is still the one an earlier clean returned (its ``"cleaned"`` fingerprint
    matches its points). Otherwise every contour returned carries its fingerprint.
    """
    source = dict(polygons or {})
    present = [name for name in CLASS_VALUES if name in source]
    if present and all(_is_cleaned(source[name]) for name in present):
        return source, []  # exactly what a clean returned: a second click changes nothing
    cleaned = clean_polygons(source, view, image, size)
    if np.array_equal(label_map(cleaned, size), label_map(source, size)):
        return source, []
    changed = [name for name in CLASS_VALUES if cleaned.get(name) != source.get(name)]
    for name in CLASS_VALUES:
        if name in cleaned:
            cleaned[name] = {**cleaned[name], "cleaned": fingerprint(cleaned[name]["points"])}
    return cleaned, changed


def clean_polygons(polygons: Mapping[str, Any], view: str, image: Optional[np.ndarray] = None,
                   size: int = FRAME) -> Dict[str, Dict[str, Any]]:
    """The stored contours of one view, cleaned by :func:`clean_label_map`.

    A class whose painted region the cleaning leaves exactly as it was is returned as given (same points, same
    flags), so labels that are already clean come back unchanged. A class that changed is a copy of its entry with
    the new contour (traced like a model mask, :func:`breast_seg.inference.largest_polygon`); its other fields are
    kept and ``edited`` is not set here. A class the cleaning removes (a CC pectoral, merged into the breast tissue;
    a nipple the smoothing erased) is left out. The only class that can appear is the breast tissue of a CC that had
    a pectoral contour and none for the breast: its pixels become breast tissue, as a new entry
    ``{"points", "confidence": 1.0, "rescued": False}`` (the shape of a contour drawn in the editor). Entries of
    other names are passed through untouched.
    """
    from ..inference.engine import largest_polygon

    source = dict(polygons or {})
    before = label_map(source, size)
    after = clean_label_map(before, view, image)

    out: Dict[str, Dict[str, Any]] = {name: entry for name, entry in source.items() if name not in CLASS_VALUES}
    for name, value in CLASS_VALUES.items():
        entry = source.get(name)
        region = after == value
        if np.array_equal(region, before == value):
            if entry is not None:
                out[name] = entry
            continue
        points = largest_polygon(region.astype(np.uint8)) if region.any() else []
        if len(points) < 3:
            continue  # the class is gone
        updated = copy.deepcopy(entry) if isinstance(entry, Mapping) else {"confidence": 1.0, "rescued": False}
        updated["points"] = points
        out[name] = updated
    return out
