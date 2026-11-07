"""Fallback measurement point when no nipple is found.

The quantity of interest is how far the breast tissue reaches from the pectoral line (MLO) or the chest wall (CC);
the nipple is its marker. Without a marker the measurement can be taken at the breast-tissue contour point that
reaches farthest. It is a substitute only for a *missing* nipple, never a replacement for a found one.
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


def tissue_contour(tissue: np.ndarray, min_px: int = 200) -> Optional[np.ndarray]:
    """Largest external contour of the breast-tissue mask as ``(N, 2)`` float canvas points; ``None`` if the mask
    has fewer than ``min_px`` pixels."""
    if np.asarray(tissue).sum() < min_px:
        return None
    contours, _ = cv2.findContours(np.asarray(tissue).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)


def farthest_point(points: Optional[np.ndarray], measure: Callable[[np.ndarray], float]) -> Optional[np.ndarray]:
    """The point maximising ``measure`` (distance to the pectoral line, or to the chest wall)."""
    if points is None or len(points) == 0:
        return None
    return points[int(np.argmax([measure(p) for p in points]))]
