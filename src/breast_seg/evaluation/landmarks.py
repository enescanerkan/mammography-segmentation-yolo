"""Landmark geometry in original-image (DICOM pixel) coordinates and the millimetre distances built on it.

Models see a 640x640 canvas made by cropping the breast region, scaling and padding; :class:`CanvasTransform`
maps between that canvas and the original pixel grid (the per-image metadata written by preprocessing:
``crop_x1, crop_y1, scale, pad_left, pad_top``). Distances are measured in original pixels and converted to
millimetres with the pixel spacing.

Definitions used throughout the evaluation:

* **PNL (MLO)**: perpendicular distance from the nipple to the *infinite* line through the two pectoral points
  (the foot is not clamped to the segment).
* **D_CC (CC)**: horizontal distance from the nipple to the chest-wall image edge: ``x`` for a left breast,
  ``width - x`` for a right breast. The breast mask does not enter this measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

#: A nipple mask smaller than this many pixels counts as "not found" (rejects degenerate specks that would
#: otherwise make a model look as if it always answered).
MIN_NIPPLE_PX = 30

Point = Union[Sequence[float], np.ndarray]


def _getter(row: Any):
    """Field access for a mapping, a ``pandas.Series`` or a named tuple."""
    if hasattr(row, "get"):
        return lambda k, d=None: row.get(k, d)
    return lambda k, d=None: getattr(row, k, d)


@dataclass(frozen=True)
class CanvasTransform:
    """Original pixel grid <-> model canvas: ``canvas = (orig - crop) * scale + pad``."""

    scale: float
    crop_x1: float
    crop_y1: float
    pad_left: float
    pad_top: float

    @classmethod
    def from_row(cls, row: Union[Mapping[str, Any], Any]) -> CanvasTransform:
        get = _getter(row)
        return cls(float(get("scale")), float(get("crop_x1")), float(get("crop_y1")),
                   float(get("pad_left")), float(get("pad_top")))

    def to_original(self, p: Point) -> np.ndarray:
        return np.array([(float(p[0]) - self.pad_left) / self.scale + self.crop_x1,
                         (float(p[1]) - self.pad_top) / self.scale + self.crop_y1])

    def to_canvas(self, p: Point) -> np.ndarray:
        return np.array([(float(p[0]) - self.crop_x1) * self.scale + self.pad_left,
                         (float(p[1]) - self.crop_y1) * self.scale + self.pad_top])


@dataclass(frozen=True)
class ImageGeometry:
    """Everything needed to measure on one image: transform, pixel spacing (mm), laterality, original width."""

    transform: CanvasTransform
    spacing_mm: float
    laterality: str = "L"
    width: float = 0.0

    @classmethod
    def from_row(cls, row: Union[Mapping[str, Any], Any]) -> ImageGeometry:
        get = _getter(row)
        return cls(CanvasTransform.from_row(row), float(get("pixel_spacing")), str(get("laterality", "L")).upper(),
                   float(get("original_width", 0.0) or 0.0))


def centroid(mask: np.ndarray, min_px: int = 1) -> Optional[np.ndarray]:
    """``(x, y)`` mean of the mask's pixels, or ``None`` below ``min_px`` pixels."""
    m = np.asarray(mask).astype(bool)
    if m.sum() < max(min_px, 1):
        return None
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def point_line_distance(p: Point, a: Point, b: Point) -> float:
    """Distance from ``p`` to the infinite line through ``a`` and ``b`` (``|p - a|`` for a degenerate line)."""
    a = np.asarray(a, float)
    v, w = np.asarray(b, float) - a, np.asarray(p, float) - a
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return float(np.linalg.norm(w))
    return float(abs(v[0] * w[1] - v[1] * w[0]) / n)


def foot_on_line(p: Point, a: Point, b: Point) -> np.ndarray:
    """Foot of the perpendicular from ``p`` on the infinite line ``a``-``b``."""
    a = np.asarray(a, float)
    v = np.asarray(b, float) - a
    vv = float(v @ v)
    if vv < 1e-12:
        return a.copy()
    return a + float((np.asarray(p, float) - a) @ v) / vv * v


def pnl_mm(nipple: Point, pec_a: Point, pec_b: Point, spacing_mm: float) -> float:
    """Posterior nipple line length in mm (infinite-line definition)."""
    return point_line_distance(nipple, pec_a, pec_b) * float(spacing_mm)


def chest_wall_mm(nipple: Point, laterality: str, width: float, spacing_mm: float) -> float:
    """CC nipple-to-chest-wall distance (image edge) in mm: ``x`` for L, ``|width - x|`` for R."""
    x = float(nipple[0])
    return (x if str(laterality).upper() == "L" else abs(float(width) - x)) * float(spacing_mm)


def distance_mm(p: Point, q: Point, spacing_mm: float) -> float:
    """Euclidean distance between two original-grid points in mm."""
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float))) * float(spacing_mm)


def landmark_errors_mm(pred: Mapping[str, Point], gt: Mapping[str, Point], spacing_mm: float) -> Mapping[str, float]:
    """``{name: error_mm}`` for every landmark present in both mappings (original-grid coordinates)."""
    return {k: distance_mm(pred[k], gt[k], spacing_mm) for k in gt if k in pred and pred[k] is not None}


def canvas_to_original(points: Mapping[str, Optional[Point]], transform: CanvasTransform) -> Mapping[str, Optional[np.ndarray]]:
    """Map a ``{name: canvas point}`` dict to the original grid (``None`` stays ``None``)."""
    return {k: (None if v is None else transform.to_original(v)) for k, v in points.items()}


def as_tuple(p: Optional[Point]) -> Optional[Tuple[float, float]]:
    return None if p is None else (float(p[0]), float(p[1]))
