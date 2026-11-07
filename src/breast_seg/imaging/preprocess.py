"""The 640 model-input recipes the pose and segmentation checkpoints were trained on.

This is the inference side of the positioning training pipeline and must stay byte-for-byte identical to it:
VOI LUT (done by the DICOM reader), MONOCHROME1 inversion, threshold crop with 30 px padding,
normalise-after-crop, resize longest side to 640, optional wavelet enhancement, laterality-aware black pad.
One deliberate difference from training: the keypoint-guided bounding-box expansion needs ground-truth
landmarks, so it is skipped.

Two recipes share one geometry:

``baseline640``  the plain crop/resize/pad (segmentation and lesion models)
``wavelet640``   the same canvas with NLM -> db4 level-2 detail boost -> CLAHE applied before padding (pose)

so a point found by one model is directly comparable with a mask drawn by the other, and one
:class:`Transform` maps both back to the original pixels.

Keypoint order of the pose models: MLO = [nipple, pectoral_top, pectoral_bottom], CC = [nipple].
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

TARGET_SIZE = 640
CROP_PADDING = 30  # padding added around the detected breast bounding box
MIN_THRESHOLD = 15  # 8-bit level above which a pixel counts as tissue for the crop

RECIPES = ("baseline640", "wavelet640", "raw1024")


@dataclass
class Transform:
    """Maps original pixels <-> the 640x640 model input."""

    original_width: int
    original_height: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    pixel_spacing: float

    def to_original(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) points in 640 space -> (N, 2) points in original pixels."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] - self.pad_left) / self.scale + self.crop_x1
        out[:, 1] = (pts[:, 1] - self.pad_top) / self.scale + self.crop_y1
        return out

    def to_model(self, points: np.ndarray) -> np.ndarray:
        """(N, 2) points in original pixels -> (N, 2) points in 640 space (inverse of :meth:`to_original`)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        out = np.empty_like(pts)
        out[:, 0] = (pts[:, 0] - self.crop_x1) * self.scale + self.pad_left
        out[:, 1] = (pts[:, 1] - self.crop_y1) * self.scale + self.pad_top
        return out

    @property
    def mm_per_640px(self) -> float:
        """Physical size of one pixel of the 640x640 input."""
        return self.pixel_spacing / self.scale

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def wavelet_enhance(image: np.ndarray) -> np.ndarray:
    """Advanced wavelet strategy: NLM denoise -> db4 level-2 detail boost (1.3 / 1.6) -> CLAHE 2.5 / 8x8."""
    import pywt  # imported here so the geometry-only users of the package never need PyWavelets

    denoised = cv2.fastNlMeansDenoising(image, None, h=8, templateWindowSize=5, searchWindowSize=11)

    coeffs = pywt.wavedec2(denoised.astype(np.float64), "db4", level=2)
    boosted = [coeffs[0]]
    for level_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        factor = 1.3 if level_idx == 1 else 1.6
        boosted.append((cH * factor, cV * factor, cD * factor))
    reconstructed = pywt.waverec2(boosted, "db4")[: image.shape[0], : image.shape[1]]
    enhanced = np.clip(reconstructed, 0, 255).astype(np.uint8)

    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(enhanced)


def quick_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max to uint8; a constant image becomes all zeros."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def find_breast_bbox(preview: np.ndarray) -> Tuple[int, int, int, int]:
    """(x1, y1, x2, y2) of the tissue in an 8-bit preview, padded by :data:`CROP_PADDING`; the whole image if none."""
    h, w = preview.shape[:2]
    binary = (preview > MIN_THRESHOLD).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return 0, 0, w, h
    x, y, bw, bh = cv2.boundingRect(coords)
    if bw < 100 or bh < 100:
        return 0, 0, w, h
    return (max(0, x - CROP_PADDING), max(0, y - CROP_PADDING),
            min(w, x + bw + CROP_PADDING), min(h, y + bh + CROP_PADDING))


def chest_wall_side(ds: Any, pixels: Optional[np.ndarray] = None) -> str:
    """Which side of the *pixel data* the chest wall sits on ('L', 'R', or 'U' when it cannot be told).

    Not the same as ImageLaterality: many Hologic exports carry FieldOfViewHorizontalFlip=YES, i.e. the stored
    pixels are mirrored, so an R breast appears on the left of the image. Padding direction and the CC chest-wall
    reference must follow the pixels, not the anatomy label. ``ds`` is any object with DICOM-style attributes.
    """
    laterality = str(getattr(ds, "ImageLaterality", "")) or "U"
    flipped = str(getattr(ds, "FieldOfViewHorizontalFlip", "NO")).upper() == "YES"
    if laterality in ("L", "R"):
        if not flipped:
            return laterality
        return "R" if laterality == "L" else "L"

    # No laterality tag: fall back to where the tissue actually is.
    if pixels is None:
        return "U"
    half = pixels.shape[1] // 2
    threshold = pixels.mean() * 0.5
    return "L" if (pixels[:, :half] > threshold).sum() > (pixels[:, half:] > threshold).sum() else "R"


def preprocess(pixels: np.ndarray, meta: Mapping[str, Any], enhance: bool = True) -> Tuple[np.ndarray, Transform]:
    """Original pixels -> (640x640 uint8 model input, coordinate transform).

    ``meta`` needs ``photometric``, ``chest_side`` ('L' | 'R' | other = centred) and ``pixel_spacing`` (mm).
    """
    orig_h, orig_w = pixels.shape[:2]

    if meta["photometric"] == "MONOCHROME1":
        pixels = pixels.max() - pixels

    x1, y1, x2, y2 = find_breast_bbox(quick_normalize(pixels))
    cropped = quick_normalize(pixels[y1:y2, x1:x2])

    crop_h, crop_w = cropped.shape[:2]
    scale = TARGET_SIZE / max(crop_h, crop_w)
    resized = cv2.resize(cropped, (int(crop_w * scale), int(crop_h * scale)),
                         interpolation=cv2.INTER_LINEAR)
    enhanced = wavelet_enhance(resized) if enhance else resized

    final_h, final_w = enhanced.shape[:2]
    pad_h, pad_w = TARGET_SIZE - final_h, TARGET_SIZE - final_w
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    if meta["chest_side"] == "L":  # chest wall on the left -> pad right
        pad_left, pad_right = 0, pad_w
    elif meta["chest_side"] == "R":  # chest wall on the right -> pad left
        pad_left, pad_right = pad_w, 0
    else:
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(enhanced, pad_top, pad_bottom, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=0)

    return padded, Transform(
        original_width=orig_w, original_height=orig_h,
        crop_x1=x1, crop_y1=y1, crop_x2=x2, crop_y2=y2,
        scale=scale, pad_left=pad_left, pad_top=pad_top,
        pad_right=pad_right, pad_bottom=pad_bottom,
        pixel_spacing=meta["pixel_spacing"],
    )


def preprocess_pair(pixels: np.ndarray, meta: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, Transform]:
    """Original pixels -> (baseline 640, wavelet 640, transform).

    The two model families want different renderings of the *same* canvas: the pose checkpoints were trained on
    wavelet-enhanced images, the segmentation checkpoint on the plain ones. Geometry is identical, so one transform
    maps both back to the original pixels.
    """
    baseline, transform = preprocess(pixels, meta, enhance=False)
    padded = baseline.copy()

    # Enhancement runs before padding in training, so redo the pad on the enhanced crop.
    top, bottom = transform.pad_top, transform.pad_bottom
    left, right = transform.pad_left, transform.pad_right
    height, width = TARGET_SIZE - top - bottom, TARGET_SIZE - left - right
    crop = baseline[top:top + height, left:left + width]
    enhanced = wavelet_enhance(crop)
    wavelet = cv2.copyMakeBorder(enhanced, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=0)
    return padded, wavelet, transform
