"""Pure geometric computations for mammography analysis.

This module has NO side effects — no file I/O, no drawing.
It operates solely on numpy arrays and returns data structures.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class PectoralLine:
    """A fitted straight line along the pectoral muscle boundary edge.

    The line runs from the top to the bottom of the pectoral muscle,
    tracing its boundary with the breast tissue.
    """
    point_a: Tuple[int, int]  # (x, y) top
    point_b: Tuple[int, int]  # (x, y) bottom
    slope: float
    intercept: float
    boundary_points: Tuple[Tuple[int, int], ...] = ()

    @property
    def direction(self) -> np.ndarray:
        """Unit direction vector from point_a to point_b."""
        d = np.array(self.point_b, dtype=float) - np.array(self.point_a, dtype=float)
        norm = np.linalg.norm(d)
        return d / norm if norm > 0 else d


@dataclass(frozen=True)
class PNLResult:
    """Result of the Posterior Nipple Line (MLO: perpendicular to pectoral line).

    Attributes:
        nipple_center: (x, y) centroid of the nipple mask.
        foot_point: (x, y) perpendicular foot on the pectoral line.
        distance_px: PNL length in pixels.
    """
    nipple_center: Tuple[int, int]
    foot_point: Tuple[int, int]
    distance_px: float


@dataclass(frozen=True)
class CCDepthResult:
    """Result of the CC Depth measurement (nipple to breast edge).

    For CC views: horizontal line from nipple centroid to the farthest
    edge of the breast tissue.

    Attributes:
        nipple_center: (x, y) centroid of the nipple mask.
        edge_point: (x, y) farthest breast tissue edge point.
        distance_px: CC depth in pixels.
        breast_side: 'right' or 'left' indicating which side the breast is on.
    """
    nipple_center: Tuple[int, int]
    edge_point: Tuple[int, int]
    distance_px: float
    breast_side: str


# ── Boundary Edge Detection ───────────────────────────────


def _find_boundary_edge_points(
    pectoral_mask: np.ndarray,
    breast_mask: np.ndarray,
) -> Tuple[List[Tuple[int, int]], bool]:
    """Find the pectoral muscle edge points closest to breast tissue for every row.
    Returns the points and a boolean indicating if breast is on the right.
    """
    pec_bin = (pectoral_mask > 0).astype(np.uint8)
    brs_bin = (breast_mask > 0).astype(np.uint8)

    rows_with_pec = np.where(pec_bin.any(axis=1))[0]
    if len(rows_with_pec) < 2:
        return [], True

    top_row = rows_with_pec[0]
    bot_row = rows_with_pec[-1]

    pec_coords = np.argwhere(pec_bin > 0)
    brs_coords = np.argwhere(brs_bin > 0)
    if len(pec_coords) == 0 or len(brs_coords) == 0:
        return [], True

    pec_cx = pec_coords[:, 1].mean()
    brs_cx = brs_coords[:, 1].mean()
    breast_is_right = brs_cx > pec_cx

    edge_points = []
    for row in range(top_row, bot_row + 1):
        cols_with_pec = np.where(pec_bin[row, :])[0]
        if len(cols_with_pec) == 0:
            continue
        if breast_is_right:
            edge_col = cols_with_pec.max()
        else:
            edge_col = cols_with_pec.min()
        edge_points.append((int(edge_col), int(row)))

    return edge_points, breast_is_right


def fit_pectoral_line(
    pectoral_mask: np.ndarray,
    breast_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[PectoralLine]:
    """Fit a line along the pectoral muscle's boundary with breast tissue.
    Rule: Take top half outermost and bottom half outermost, draw line, translate to absolute outermost.
    """
    if pectoral_mask is None or breast_mask is None:
        return None

    edge_points, breast_is_right = _find_boundary_edge_points(pectoral_mask, breast_mask)
    if len(edge_points) < 2:
        return None

    # Ensure points are ordered top to bottom by y-coordinate
    edge_points.sort(key=lambda p: p[1])
    
    # Point A: Absolute topmost pixel of the pectoral muscle edge
    pt_top = edge_points[0]
    # Point B: Absolute bottommost pixel of the pectoral muscle edge
    pt_bot = edge_points[-1]

    # Narrow down the middle region search (Requested behavior).
    # Only search a very narrow band (Middle 10%) at the exact vertical center of the pectoral muscle.
    pectoral_height = pt_bot[1] - pt_top[1]
    mid_y = (pt_top[1] + pt_bot[1]) / 2.0
    threshold = pectoral_height * 0.05  # Middle 10% bounds: +/- 5%
    
    middle_points = [p for p in edge_points if abs(p[1] - mid_y) <= threshold]
    
    # Fallback to all points if the muscle is extremely short/small
    if not middle_points:
        middle_points = edge_points

    if breast_is_right:
        pt_max = max(middle_points, key=lambda p: p[0])
    else:
        pt_max = min(middle_points, key=lambda p: p[0])

    dx = pt_bot[0] - pt_top[0]
    dy = pt_bot[1] - pt_top[1]

    if abs(dy) < 1e-6:
        slope_orig = 0.0
    else:
        slope_orig = float(dx) / float(dy)

    # 1. Compute shifted bottom X coordinate using pure horizontal translation semantics
    shifted_bot_x = pt_max[0] + slope_orig * (pt_bot[1] - pt_max[1])

    # 2. Refined bottom anchor correction: If the bottom point plunges too deep into breast tissue,
    # pull it slightly back towards its original predicted bottom boundary.
    # (Weighted 50% averaging between the rigidly shifted point and the original true bottom edge)
    target_bot_x = (shifted_bot_x + pt_bot[0]) / 2.0

    # 3. Pivot/Tilt the line conceptually.
    # The line is strictly forced to pass through the outermost pt_max and drop down to target_bot_x.
    dy_bot = pt_bot[1] - pt_max[1]
    if abs(dy_bot) < 1e-6:
        slope_new = slope_orig
    else:
        slope_new = (target_bot_x - pt_max[0]) / dy_bot
        
    # Calculate interception using the newly pivoted slope
    intercept_new = pt_max[0] - slope_new * pt_max[1]

    h, w = image_shape
    
    y_top_clamped = pt_top[1]
    y_bot_clamped = pt_bot[1]
    
    # Reproject top and bottom X coordinates based on the pt_max pivot
    x_top_clamped = int(round(slope_new * y_top_clamped + intercept_new))
    x_bot_clamped = int(round(slope_new * y_bot_clamped + intercept_new))
    
    x_top_clamped = max(0, min(w - 1, x_top_clamped))
    x_bot_clamped = max(0, min(w - 1, x_bot_clamped))

    final_dx = x_bot_clamped - x_top_clamped
    if abs(final_dx) < 1e-6:
        slope = float('inf')
        intercept = 0.0
    else:
        slope = (y_bot_clamped - y_top_clamped) / final_dx
        intercept = y_top_clamped - slope * x_top_clamped

    return PectoralLine(
        point_a=(x_top_clamped, y_top_clamped),
        point_b=(x_bot_clamped, y_bot_clamped),
        slope=slope,
        intercept=intercept,
        boundary_points=(pt_top, pt_bot, pt_max),
    )


# ── PNL: Perpendicular Nipple Line (MLO) ──────────────────


def compute_nipple_centroid(nipple_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Compute the centroid of the nipple segmentation mask.

    Returns:
        (x, y) centroid or None if mask is empty.
    """
    if nipple_mask is None:
        return None

    moments = cv2.moments((nipple_mask > 0).astype(np.uint8))
    if moments["m00"] == 0:
        return None

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    return (cx, cy)


def compute_pnl(
    nipple_center: Tuple[int, int],
    pectoral_line: PectoralLine,
) -> PNLResult:
    """Compute the PNL — perpendicular from nipple to the pectoral line SEGMENT.

    The foot point is CLAMPED to the pectoral line segment [point_a, point_b],
    so it never extends beyond the actual pectoral muscle boundary.

    Args:
        nipple_center: (x, y) centroid of the nipple.
        pectoral_line: The fitted pectoral boundary line.

    Returns:
        PNLResult with foot point and perpendicular distance.
    """
    P = np.array(nipple_center, dtype=float)
    A = np.array(pectoral_line.point_a, dtype=float)
    B = np.array(pectoral_line.point_b, dtype=float)

    AB = B - A
    AP = P - A

    ab_dot = np.dot(AB, AB)
    if ab_dot < 1e-10:
        return PNLResult(
            nipple_center=nipple_center,
            foot_point=pectoral_line.point_a,
            distance_px=float(np.linalg.norm(P - A)),
        )

    t = np.dot(AP, AB) / ab_dot

    # CLAMP t to [0, 1] so the foot stays ON the pectoral line segment
    t = max(0.0, min(1.0, t))

    foot = A + t * AB
    distance = np.linalg.norm(P - foot)

    return PNLResult(
        nipple_center=nipple_center,
        foot_point=(int(round(foot[0])), int(round(foot[1]))),
        distance_px=float(distance),
    )


# ── CC Depth: Nipple to Breast Edge (CC view) ─────────────


def compute_cc_depth(
    nipple_center: Tuple[int, int],
    breast_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[CCDepthResult]:
    """Compute CC Depth — horizontal distance from nipple to breast tissue edge.

    For CC views (no pectoral muscle), draws a horizontal line from
    the nipple centroid to the farthest edge of the breast tissue.

    Algorithm:
        1. Determine which side the breast is on (left/right) based on
           breast tissue centroid relative to image center.
        2. In the nipple's row, find the farthest breast tissue pixel
           on that side.
        3. Draw a horizontal line from nipple to that edge.

    Args:
        nipple_center: (x, y) centroid of the nipple.
        breast_mask:   Binary mask (H, W) of the breast tissue.
        image_shape:   (height, width) of the original image.

    Returns:
        CCDepthResult or None if breast mask is empty.
    """
    if breast_mask is None:
        return None

    brs_bin = (breast_mask > 0).astype(np.uint8)
    h, w = image_shape
    nx, ny = nipple_center

    # Determine breast side from breast tissue centroid
    brs_coords = np.argwhere(brs_bin > 0)  # (row, col)
    if len(brs_coords) == 0:
        return None

    breast_cx = brs_coords[:, 1].mean()
    image_cx = w / 2.0
    breast_side = "right" if breast_cx > image_cx else "left"

    # Search for the farthest breast edge in the nipple's row
    # Use a band of rows around nipple for robustness
    row_band = 10
    row_start = max(0, ny - row_band)
    row_end = min(h, ny + row_band + 1)
    band = brs_bin[row_start:row_end, :]

    cols_with_breast = np.where(band.any(axis=0))[0]
    if len(cols_with_breast) == 0:
        return None

    if breast_side == "right":
        # Breast is on right → edge is the rightmost breast pixel
        edge_col = int(cols_with_breast.max())
    else:
        # Breast is on left → edge is the leftmost breast pixel
        edge_col = int(cols_with_breast.min())

    edge_point = (edge_col, ny)  # Horizontal line → same y as nipple
    distance = abs(edge_col - nx)

    return CCDepthResult(
        nipple_center=nipple_center,
        edge_point=edge_point,
        distance_px=float(distance),
        breast_side=breast_side,
    )
