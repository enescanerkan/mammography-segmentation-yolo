"""Display renders and pixel-level quality numbers in the 640 model frame."""
from __future__ import annotations

from typing import Any, Dict, Mapping

import cv2
import numpy as np

from .preprocess import TARGET_SIZE, Transform, preprocess_pair

# 8-bit level below which a pixel of the baseline render counts as padding / air.
TISSUE_LEVEL = 12


def render_display(pixels: np.ndarray, meta: Mapping[str, Any], transform: Transform, size: int) -> np.ndarray:
    """The same framing as the 640 model input, rendered at ``size`` x ``size`` for reading.

    Built by warping the original pixels through the transform rather than by upscaling the 640 render —
    upscaling would add pixels without adding detail, and skin folds and motion blur are exactly what those pixels
    are for. The geometry is identical, so a label drawn in the 640 frame still lands correctly when the frame is
    drawn at any size. ``size <= 640`` returns the baseline render itself.
    """
    if size <= TARGET_SIZE:
        baseline, _, _ = preprocess_pair(pixels, meta)
        return baseline

    source = pixels
    if meta["photometric"] == "MONOCHROME1":
        source = source.max() - source

    # Normalise on the crop, the way the training preprocessing does; using the whole image would change the grey
    # levels the reviewer sees.
    crop = source[transform.crop_y1:transform.crop_y2, transform.crop_x1:transform.crop_x2]
    low, high = float(crop.min()), float(crop.max())
    if high - low < 1e-8:
        return np.zeros((size, size), np.uint8)
    normalised = np.clip((source - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)

    factor = size / float(TARGET_SIZE)
    scale = transform.scale * factor
    offset_x = (transform.pad_left - transform.crop_x1 * transform.scale) * factor
    offset_y = (transform.pad_top - transform.crop_y1 * transform.scale) * factor

    if scale < 1.0:
        # warpAffine has no area filter: asked for INTER_AREA it quietly samples bilinearly, which at a 0.3-0.5
        # reduction skips pixels instead of averaging them and aliases exactly the fine texture this render is
        # kept for. Reduce with a real area filter, then place the result by a sub-pixel shift chosen so the
        # mapping stays x' = scale * x + offset (resize maps pixel centres, the affine maps pixel corners — hence
        # the half-pixel terms).
        normalised = cv2.resize(normalised, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        offset_x += 0.5 - 0.5 * scale
        offset_y += 0.5 - 0.5 * scale
        scale = 1.0

    matrix = np.array([[scale, 0.0, offset_x], [0.0, scale, offset_y]], dtype=np.float64)
    return cv2.warpAffine(normalised, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def image_quality(baseline: np.ndarray) -> Dict[str, float]:
    """Two numbers a reader would otherwise have to eyeball: focus and exposure.

    Both are measured on the tissue only — the black padding would otherwise dilute the Laplacian variance and
    inflate the "not clipped" fraction.
    """
    tissue = baseline[baseline > TISSUE_LEVEL]
    if tissue.size < 1000:
        return {"sharpness": 0.0, "clipped_fraction": 0.0, "mean_level": 0.0}
    mask = (baseline > TISSUE_LEVEL).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)
    region = baseline[y:y + h, x:x + w]
    return {
        "sharpness": round(float(cv2.Laplacian(region, cv2.CV_64F).var()), 1),
        "clipped_fraction": round(float((tissue >= 254).mean()), 4),
        "mean_level": round(float(tissue.mean()), 1),
    }
