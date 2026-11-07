"""DICOM loading for the compare visualisations (VOI LUT, MONOCHROME1 inversion, 8-bit BGR)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np

DEFAULT_SPACING_MM = 0.085


class DicomUtils:
    @staticmethod
    def spacing_mm_from_ds(ds: Any) -> float:
        """``ImagerPixelSpacing`` (preferred) or ``PixelSpacing`` in mm; a default when both are missing."""
        if ds is None:
            return DEFAULT_SPACING_MM
        for tag in ("ImagerPixelSpacing", "PixelSpacing"):
            value = getattr(ds, tag, None)
            try:
                if value is not None and len(value) > 0:
                    return float(value[0])
            except (TypeError, ValueError):
                continue
        return DEFAULT_SPACING_MM

    @staticmethod
    def load_dicom_bgr(path: Path) -> Tuple[Optional[np.ndarray], float, Any]:
        """``(bgr uint8, spacing_mm, dataset)``; ``(None, default, None)`` when the file cannot be read."""
        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError:
            return None, DEFAULT_SPACING_MM, None
        if not Path(path).is_file():
            return None, DEFAULT_SPACING_MM, None
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = np.squeeze(np.asarray(ds.pixel_array))
            if arr.ndim == 3:
                arr = arr[0] if arr.shape[0] < arr.shape[1] else arr[:, :, 0]
            a = apply_voi_lut(arr, ds).astype(np.float32)
            if str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "") == "MONOCHROME1" and a.size:
                a = float(a.max()) - a
            lo, hi = float(a.min()), float(a.max())
            g8 = (np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8) if hi > lo + 1e-8
                  else np.zeros(a.shape, np.uint8))
            return cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR), DicomUtils.spacing_mm_from_ds(ds), ds
        except Exception:  # noqa: BLE001 - a broken file falls back to the PNG canvas
            return None, DEFAULT_SPACING_MM, None
