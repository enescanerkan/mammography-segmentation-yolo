"""Coordinate mapping between the 640 canvas and the original image, and the compare pipeline's distances.

Thin static facade over :mod:`breast_seg.evaluation.landmarks`, kept so the pipeline code reads as before.
Metadata rows carry ``crop_x1, crop_y1, scale, pad_left, pad_top``.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from breast_seg.evaluation.landmarks import CanvasTransform, chest_wall_mm, foot_on_line, point_line_distance

Point = Tuple[float, float]


class GeometryUtils:
    @staticmethod
    def inverse_transform_640(x: float, y: float, row: Any) -> Point:
        """Canvas point -> original image point."""
        p = CanvasTransform.from_row(row).to_original((x, y))
        return float(p[0]), float(p[1])

    @staticmethod
    def dicom_to_640(x: float, y: float, row: Any) -> Point:
        """Original image point -> canvas point."""
        p = CanvasTransform.from_row(row).to_canvas((x, y))
        return float(p[0]), float(p[1])

    @staticmethod
    def pnl_infinite_line_mm(nipple: Point, pec_top: Point, pec_bottom: Point, pixel_spacing_mm: float) -> float:
        """Nipple distance to the infinite line through the two pectoral points, in mm."""
        return point_line_distance(nipple, pec_top, pec_bottom) * float(pixel_spacing_mm)

    @staticmethod
    def cc_chest_mm_from_nipple_dicom(nipple: Point, laterality: str, original_width: float, pixel_spacing_mm: float) -> float:
        """CC nipple-to-chest-wall (image edge) distance in mm."""
        return chest_wall_mm(nipple, laterality, original_width, pixel_spacing_mm)

    @staticmethod
    def foot_on_infinite_line(nipple: Point, p1: Point, p2: Point) -> Point:
        f = foot_on_line(nipple, p1, p2)
        return float(f[0]), float(f[1])

    @staticmethod
    def euclidean_mm_dicom(a: Point, b: Point, pixel_spacing_mm: float) -> float:
        return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))) * float(pixel_spacing_mm)
