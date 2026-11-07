"""Label clean-up applied when a dataset generation is built from raw annotations.

MLO: the pectoral muscle is reduced to its largest component, holes filled and lightly closed; the thin gap that
hand annotation leaves between breast tissue and pectoral muscle is added to the breast tissue (up to the muscle,
never over it). CC: there is no pectoral muscle; breast and nipple holes are filled.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import cv2
import numpy as np

PECTORAL, BREAST, NIPPLE = 0, 1, 2


@dataclass(frozen=True)
class CorrectionParams:
    """Kernel sizes of the clean-up (pixels, elliptical structuring elements)."""

    pectoral_close: int = 9
    gap_kernel: int = 41


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes of a binary mask: background not 4-connected to the image border becomes foreground.

    Same result as ``scipy.ndimage.binary_fill_holes`` with its default structure, without the SciPy dependency.
    """
    fg = (np.asarray(mask) > 0).astype(np.uint8)
    padded = np.pad(fg, 1, constant_values=0)
    reach = padded.copy()
    flood = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(reach, flood, (0, 0), 2, flags=4)
    return (reach[1:-1, 1:-1] != 2).astype(np.uint8)


def _fill(m: np.ndarray) -> np.ndarray:
    return fill_holes(m) if m.any() else m


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 8-connected component of a binary mask."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    sizes = [(lab == i).sum() for i in range(1, n)]
    return (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)


def fix_pectoral(pectoral: np.ndarray, params: CorrectionParams = CorrectionParams()) -> np.ndarray:
    """Largest component, holes filled, closed with a ``pectoral_close`` ellipse."""
    if not pectoral.any():
        return pectoral
    p = fill_holes(largest_component(pectoral)).astype(np.uint8)
    return cv2.morphologyEx(p, cv2.MORPH_CLOSE, _ellipse(params.pectoral_close)).astype(np.uint8)


def fill_gap_into_breast(breast: np.ndarray, pectoral: np.ndarray, params: CorrectionParams = CorrectionParams()) -> np.ndarray:
    """Add the breast/pectoral seam to the breast tissue.

    The union of both masks is closed and hole-filled; the added pixels that lie within ``gap_kernel`` of *both*
    masks form the seam. The seam joins the breast mask, which is hole-filled and then cut back by the pectoral
    mask, so it never overlaps the muscle. Without a pectoral mask only the holes are filled.
    """
    if not pectoral.any():
        return fill_holes(breast).astype(np.uint8)
    union = ((breast | pectoral) > 0).astype(np.uint8)
    ker = _ellipse(params.gap_kernel)
    closed = fill_holes(cv2.morphologyEx(union, cv2.MORPH_CLOSE, ker)).astype(np.uint8)
    added = (closed > 0) & (union == 0)
    seam = added & (cv2.dilate(breast, ker) > 0) & (cv2.dilate(pectoral, ker) > 0)
    filled = fill_holes((breast > 0) | seam).astype(np.uint8)
    return ((filled > 0) & (pectoral == 0)).astype(np.uint8)


def correct_mlo(masks: Dict[int, np.ndarray], params: CorrectionParams = CorrectionParams()) -> Dict[int, np.ndarray]:
    """MLO clean-up: pectoral fix, seam into breast, nipple holes filled."""
    p = fix_pectoral(masks[PECTORAL], params)
    return {PECTORAL: p, BREAST: fill_gap_into_breast(masks[BREAST], p, params), NIPPLE: _fill(masks[NIPPLE])}


def correct_cc(masks: Dict[int, np.ndarray], params: CorrectionParams = CorrectionParams()) -> Dict[int, np.ndarray]:
    """CC clean-up: no pectoral class; breast and nipple holes filled."""
    b = _fill(masks[BREAST])
    return {PECTORAL: np.zeros_like(b), BREAST: b, NIPPLE: _fill(masks[NIPPLE])}


def no_correction(masks: Dict[int, np.ndarray], params: CorrectionParams = CorrectionParams()) -> Dict[int, np.ndarray]:
    """Identity."""
    return masks


CORRECTIONS: Dict[str, Callable[..., Dict[int, np.ndarray]]] = {"mlo": correct_mlo, "cc": correct_cc, "none": no_correction}
