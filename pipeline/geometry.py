"""
SOLID OOP: Geometry and Coordinate Transformations.
"""
from typing import Tuple
import numpy as np
import pandas as pd


class GeometryUtils:
    @staticmethod
    def inverse_transform_640(kp_x: float, kp_y: float, row: pd.Series) -> Tuple[float, float]:
        """Maps 640x640 processed coordinates back to full original DICOM coordinates."""
        ox = (kp_x - float(row["pad_left"])) / float(row["scale"]) + float(row["crop_x1"])
        oy = (kp_y - float(row["pad_top"])) / float(row["scale"]) + float(row["crop_y1"])
        return float(ox), float(oy)

    @staticmethod
    def dicom_to_640(ox: float, oy: float, row: pd.Series) -> Tuple[float, float]:
        """Maps original DICOM coordinates to the 640x640 inference padding box."""
        x = (ox - float(row["crop_x1"])) * float(row["scale"]) + float(row["pad_left"])
        y = (oy - float(row["crop_y1"])) * float(row["scale"]) + float(row["pad_top"])
        return float(x), float(y)

    @staticmethod
    def pnl_infinite_line_mm(nipple_dicom: Tuple[float, float], pec_top: Tuple[float, float], pec_bottom: Tuple[float, float], pixel_spacing_mm: float) -> float:
        """Calculates distance from nipple to the infinite line established by pectoralis points."""
        n = np.array(nipple_dicom, dtype=np.float64)
        p1 = np.array(pec_top, dtype=np.float64)
        p2 = np.array(pec_bottom, dtype=np.float64)
        v = p2 - p1
        vl = float(np.linalg.norm(v))
        if vl < 1e-8:
            return float(np.linalg.norm(n - p1)) * pixel_spacing_mm
        proj = p1 + np.dot(n - p1, v) / (vl * vl) * v
        return float(np.linalg.norm(n - proj)) * pixel_spacing_mm

    @staticmethod
    def cc_chest_mm_from_nipple_dicom(nipple_dicom: Tuple[float, float], laterality: str, original_width: float, pixel_spacing_mm: float) -> float:
        dist_px = float(nipple_dicom[0]) if laterality.upper() == "L" else abs(float(original_width) - nipple_dicom[0])
        return dist_px * float(pixel_spacing_mm)

    @staticmethod
    def foot_on_infinite_line(nipple: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        n = np.array(nipple, dtype=np.float64)
        a = np.array(p1, dtype=np.float64)
        b = np.array(p2, dtype=np.float64)
        v = b - a
        vl = float(np.dot(v, v))
        if vl < 1e-12:
            return float(a[0]), float(a[1])
        t = float(np.dot(n - a, v) / vl)
        proj = a + t * v
        return float(proj[0]), float(proj[1])

    @staticmethod
    def euclidean_mm_dicom(a: Tuple[float, float], b: Tuple[float, float], pixel_spacing_mm: float) -> float:
        return float(np.linalg.norm(np.array(a) - np.array(b))) * pixel_spacing_mm
