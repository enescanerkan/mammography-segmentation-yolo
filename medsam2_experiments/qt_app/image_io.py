"""Image loading + windowing helpers.

Supports PNG / JPEG / BMP / TIFF via OpenCV, and DICOM (.dcm / .dicom) via
pydicom (lazy import). DICOM frames are normalized to uint8 grayscale and
expanded to 3-channel RGB for canvas display.

``apply_window_level`` performs interactive brightness/contrast adjustment
(window/level) on top of the raw 16-bit DICOM data or the 8-bit RGB
fallback. The canvas keeps a reference to the unmodified array so the user
can scrub W/L without lossy re-quantization between adjustments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def load_image_rgb(path: Path) -> Optional[np.ndarray]:
    """Load any supported image as RGB uint8 ``H×W×3``.

    Returns ``None`` (and logs) on failure. Uses pydicom for ``.dcm``,
    ``cv2.imdecode`` (via ``np.fromfile`` for Unicode paths on Windows)
    for everything else.
    """
    path = Path(path)
    if not path.is_file():
        log.warning("load_image_rgb: not a file: %s", path)
        return None
    suffix = path.suffix.lower()
    try:
        if suffix in (".dcm", ".dicom"):
            return _load_dicom_rgb(path)
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            log.warning("load_image_rgb: empty file: %s", path)
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("load_image_rgb: cv2.imdecode failed: %s", path)
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        log.exception("load_image_rgb: unexpected error for %s", path)
        return None


def _load_dicom_rgb(path: Path) -> Optional[np.ndarray]:
    """Load a DICOM file → RGB uint8 ``H×W×3`` with correct display mapping.

    Order of operations follows the DICOM display pipeline:
      1. Modality LUT (Rescale slope/intercept)
      2. VOI LUT / Window-Level (clamps to display range)
      3. Photometric inversion for MONOCHROME1 (final step)

    Fallbacks for files that lack VOI tags use a robust 1-99 percentile
    clip so mammography studies don't end up looking washed-out white
    (the previous min-max normalization was sensitive to a single bright
    pixel pushing the whole image into the upper half of the histogram).
    """
    try:
        import pydicom  # type: ignore
        from pydicom.pixel_data_handlers.util import (  # type: ignore
            apply_modality_lut,
            apply_voi_lut,
        )
    except ImportError:
        log.error("pydicom is not installed — cannot load DICOM: %s", path)
        return None
    try:
        ds = pydicom.dcmread(str(path), force=True)
    except Exception:
        log.exception("pydicom.dcmread failed for %s", path)
        return None
    arr = getattr(ds, "pixel_array", None)
    if arr is None:
        log.error("DICOM has no pixel_array: %s", path)
        return None

    # ── Normalize array dimensionality ─────────────────────────────
    #   2D                  → single-frame monochrome (H, W)
    #   3D last dim ∈ {3,4} → single-frame color      (H, W, C)
    #   3D otherwise        → multi-frame monochrome  (N, H, W)  → arr[0]
    #   4D                  → multi-frame color       (N, H, W, C) → arr[0]
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        # First dim is the frame index. Take the first frame.
        arr = arr[0]

    is_color = arr.ndim == 3 and arr.shape[-1] in (3, 4)
    if is_color:
        # Already-rendered color DICOM — windowing / VOI LUT don't apply.
        # Just clamp to uint8 and skip the monochrome pipeline.
        c = arr[:, :, :3]
        if c.dtype != np.uint8:
            lo, hi = float(np.min(c)), float(np.max(c))
            if hi - lo < 1e-6:
                c = np.zeros_like(c, dtype=np.uint8)
            else:
                c = ((c.astype(np.float32) - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(c.astype(np.uint8))
        log.info("loaded DICOM %s shape=%s (color, no windowing)", path.name, rgb.shape)
        return rgb

    # ── Monochrome pipeline ────────────────────────────────────────
    # 1) Modality LUT — applies RescaleSlope/Intercept (or a real LUT).
    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr.astype(np.float32) * slope + intercept

    arr = arr.astype(np.float32)

    # 2) Windowing. Prefer the DICOM-provided VOI LUT / Window tags so
    #    mammography studies are shown the way the modality intended.
    windowed: Optional[np.ndarray] = None
    try:
        # apply_voi_lut returns float when VOI tags exist
        windowed = apply_voi_lut(arr.astype(np.int32), ds).astype(np.float32)
    except Exception:
        log.debug("apply_voi_lut failed for %s; falling back to W/L tags", path.name)

    if windowed is None:
        # Manual W/L from tags if present.
        try:
            wc_attr = ds.WindowCenter
            ww_attr = ds.WindowWidth
            wc = float(wc_attr[0] if hasattr(wc_attr, "__iter__") else wc_attr)
            ww = float(ww_attr[0] if hasattr(ww_attr, "__iter__") else ww_attr)
            ww = max(1.0, ww)
            lo = wc - ww / 2.0
            hi = wc + ww / 2.0
            windowed = np.clip(arr, lo, hi)
        except Exception:
            # Percentile clip — robust to bright outliers (calibration markers).
            lo_q, hi_q = np.percentile(arr, [1.0, 99.0])
            if hi_q - lo_q < 1e-6:
                lo_q, hi_q = float(arr.min()), float(arr.max())
            windowed = np.clip(arr, lo_q, hi_q)

    # 3) Normalize windowed values to 0–255.
    lo = float(np.min(windowed))
    hi = float(np.max(windowed))
    if hi - lo < 1e-6:
        norm = np.zeros_like(windowed, dtype=np.uint8)
    else:
        norm = ((windowed - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)

    # 4) Photometric inversion — applied AFTER windowing/normalization so
    #    the DICOM-supplied W/L still operates on the original intensities.
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        norm = 255 - norm

    rgb = np.stack([norm] * 3, axis=-1)
    log.info(
        "loaded DICOM %s shape=%s photometric=%s (windowed via %s)",
        path.name, rgb.shape, photometric,
        "VOI LUT" if windowed is arr else "VOI/percentile",
    )
    return rgb


def apply_window_level(
    rgb: np.ndarray,
    window: float,
    level: float,
) -> np.ndarray:
    """Apply brightness/contrast (level / window) to an RGB image.

    ``level`` ∈ roughly [-128, 128]  — additive brightness shift.
    ``window`` ∈ roughly [-128, 128]  — contrast adjustment (higher = more
    contrast). 0/0 = identity. Result is uint8 RGB.
    """
    if abs(window) < 1e-3 and abs(level) < 1e-3:
        return rgb
    out = rgb.astype(np.float32)
    contrast = 1.0 + (float(window) / 128.0)
    contrast = max(0.05, min(3.0, contrast))
    out = (out - 128.0) * contrast + 128.0 + float(level)
    return np.clip(out, 0, 255).astype(np.uint8)
