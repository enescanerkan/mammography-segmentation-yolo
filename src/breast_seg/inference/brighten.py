"""Brightened nipple re-search: the third nipple step, for when the normal pass and the low-confidence rescue both miss.

On a dark image the nipple sits in the faint periphery of the breast and the segmentation model does not see it. The
engine then finds the anterior tip of the segmented breast (MLO: the contour point farthest from the fitted pectoral
line; CC: the point farthest from the chest-wall edge), cuts a window around it from the ORIGINAL pixels, re-windows
it linearly from the background level outside the breast to the 99.5th percentile of the breast pixels, pastes it
back into the 640 canvas at the same scale and position, and runs the segmentation model again at the rescue
confidence. Only a nipple whose centroid lies inside the pasted window counts. The pass adds a nipple contour and
nothing else: the breast-tissue and pectoral masks of the normal pass stay as they were.

An optional check, off unless ``max_gap_mm`` is given: the nipple's mask must then lie within that distance of the
breast-tissue mask, taking the normal pass and this brightened pass together (on a dark image the normal pass stops
short of the skin line, so its mask alone can end 10-20 mm before a correct nipple).

This is the variant chosen beforehand on the VinDr validation split (in-place window of 50 mm, linear window) out of
six (in-place / enlarged window x linear / gamma / CLAHE), evaluated with the ``seg_v4`` segmentation model. The
functions here are pure image work; the model pass is
:meth:`breast_seg.inference.engine.LandmarkEngine._brighten_nipple`.

The search is the caller's choice, not the registry's: it runs only for an engine built with a
:class:`BrightenSearch` (the review app does so through the ``nipple:`` section of its own config), so every other
user of the shared registry gets the landmarks it always got.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

from ..geometry import anterior_point, fit_pectoral_line
from ..imaging.dicom import display_pixels
from ..imaging.preprocess import TARGET_SIZE, Transform

MIN_WINDOW_PX = 16  # a window narrower than this on the original grid is not searched
MIN_PASTE_PX = 4  # nor one that lands almost entirely outside the 640 canvas


@dataclass(frozen=True)
class BrightenSearch:
    """Turns the brightened re-search on for one engine (``LandmarkEngine(..., brighten=BrightenSearch())``).

    ``window_mm``: side of the square re-windowed around the anterior tip. ``max_gap_mm``: the optional adjacency
    check, off when None (see the module docstring). The search runs on a segmentation model whose registry entry
    has ``nipple_rescue`` enabled, at that rescue's confidence, after the rescue has missed.
    """

    window_mm: float = 50.0
    max_gap_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if not float(self.window_mm) > 0:
            raise ValueError("window_mm must be more than zero")
        if self.max_gap_mm is not None and not float(self.max_gap_mm) >= 0:
            raise ValueError("max_gap_mm must be zero or more, or None")


def anterior_tip(masks: Mapping[str, np.ndarray], view: str, chest_side: str) -> Optional[np.ndarray]:
    """The breast-contour point farthest from the pectoral line (MLO) or from the chest-wall edge (CC, or an MLO
    whose line cannot be fitted), 640 frame (:func:`breast_seg.geometry.anterior_point`).

    The tip can be a short flat stretch of contour; the middle of the points within one pixel of the farthest is
    taken. None without a breast mask.
    """
    breast = masks.get("breast-tissue")
    if breast is None or not breast.any():
        return None
    line = None
    pectoral = masks.get("pectoral")
    if view == "MLO" and pectoral is not None:
        line = fit_pectoral_line(pectoral, breast, breast.shape)
    return anterior_point(breast, chest_side, line, TARGET_SIZE, min_px=1, within_px=1.0)


def distance_to(mask: np.ndarray) -> Optional[np.ndarray]:
    """Per pixel, the distance in pixels to the nearest pixel of ``mask`` (0 on it); None when the mask is empty.

    Computed once per image and handed to :func:`mask_gap_mm` for every candidate.
    """
    if not mask.any():
        return None
    return cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)


def mask_gap_mm(mask: np.ndarray, to_breast: Optional[np.ndarray], mm_per_px: float) -> float:
    """The shortest distance in mm from ``mask`` to the breast, given the breast's :func:`distance_to` map (0 where
    they overlap); infinite without either."""
    if to_breast is None or not mask.any():
        return math.inf
    return float(to_breast[mask > 0].min()) * float(mm_per_px)


def breast_in_box(breast640: np.ndarray, transform: Transform, box: Tuple[int, int, int, int]) -> np.ndarray:
    """The 640-frame breast mask sampled onto the original-pixel window ``box`` = (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    columns, rows = np.arange(x1, x2, dtype=float), np.arange(y1, y2, dtype=float)
    mx = np.round(transform.to_model(np.column_stack([columns, np.zeros_like(columns)]))[:, 0]).astype(int)
    my = np.round(transform.to_model(np.column_stack([np.zeros_like(rows), rows]))[:, 1]).astype(int)
    ok_x, ok_y = (mx >= 0) & (mx < TARGET_SIZE), (my >= 0) & (my < TARGET_SIZE)
    out = np.zeros((y2 - y1, x2 - x1), bool)
    out[np.ix_(ok_y, ok_x)] = breast640[np.ix_(my[ok_y], mx[ok_x])] > 0
    return out


def rewindow(crop: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Linear window from the background median (outside the breast) to the 99.5th percentile of the breast pixels."""
    crop = crop.astype(np.float32)
    background = crop[~inside]
    low = float(np.median(background)) if background.size > 100 else float(crop.min())
    high = float(np.percentile(crop[inside], 99.5)) if inside.sum() > 100 else float(crop.max())
    if high <= low:
        high = low + 1.0
    return (np.clip((crop - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def brightened_canvas(pixels: np.ndarray, meta: Mapping[str, Any], image640: np.ndarray, transform: Transform,
                      breast640: np.ndarray, tip640: np.ndarray, window_mm: float = 50.0
                      ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """The 640 canvas with a re-windowed ``window_mm`` square around the tip pasted in, and the pasted box (640
    frame, x1, y1, x2, y2); None when the window is too small or falls outside the canvas, or when the image has no
    usable pixel spacing or scale (a header that says 0 mm cannot size a window)."""
    spacing, scale = float(transform.pixel_spacing or 0.0), float(transform.scale or 0.0)
    if not (math.isfinite(spacing) and spacing > 0 and math.isfinite(scale) and scale > 0 and window_mm > 0):
        return None
    half = int(round(window_mm / 2 / spacing))
    tip = transform.to_original(np.asarray([tip640], float))[0]
    height, width = pixels.shape[:2]
    x1, y1 = max(0, int(tip[0]) - half), max(0, int(tip[1]) - half)
    x2, y2 = min(width, int(tip[0]) + half), min(height, int(tip[1]) + half)
    if x2 - x1 < MIN_WINDOW_PX or y2 - y1 < MIN_WINDOW_PX:
        return None
    inside = breast_in_box(breast640, transform, (x1, y1, x2, y2))
    # Only the window is turned to "tissue is bright" (a MONOCHROME1 file): inverting the whole full-resolution frame
    # would copy it for the sake of a few square centimetres. The inversion is then relative to the window's own
    # maximum instead of the frame's; that is a constant offset, which the linear window below removes exactly.
    window = rewindow(display_pixels(pixels[y1:y2, x1:x2], dict(meta)), inside)

    sx, sy = (int(round(v)) for v in transform.to_model(np.asarray([[x1, y1]], float))[0])
    small = cv2.resize(window, (max(1, int(round((x2 - x1) * transform.scale))),
                                max(1, int(round((y2 - y1) * transform.scale)))), interpolation=cv2.INTER_AREA)
    tx1, ty1 = max(0, sx), max(0, sy)
    tx2, ty2 = min(TARGET_SIZE, sx + small.shape[1]), min(TARGET_SIZE, sy + small.shape[0])
    if tx2 - tx1 < MIN_PASTE_PX or ty2 - ty1 < MIN_PASTE_PX:
        return None
    canvas = image640.copy()
    canvas[ty1:ty2, tx1:tx2] = small[ty1 - sy:ty2 - sy, tx1 - sx:tx2 - sx]
    return canvas, (tx1, ty1, tx2, ty2)
