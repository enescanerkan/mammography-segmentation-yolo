"""Fallback measurement point when no nipple is found.

The quantity of interest is how far the breast tissue reaches from the pectoral line (MLO) or the chest wall (CC);
the nipple is its marker. Without a marker the measurement can be taken at the breast-tissue contour point that
reaches farthest. It is a substitute only for a *missing* nipple, never a replacement for a found one.

The two helpers live in :mod:`breast_seg.geometry`, where the review app's measurements use them too; they are
re-exported here for the evaluation code.
"""
from __future__ import annotations

from breast_seg.geometry import farthest_point, tissue_contour

__all__ = ["farthest_point", "tissue_contour"]
