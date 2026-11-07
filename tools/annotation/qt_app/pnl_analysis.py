"""PNL / CC-depth geometry for the annotation toolkit.

When the user has labelled the anatomical tissues (nipple, pectoral muscle,
breast) this module turns those masks into the standard positioning
measurements:

* **MLO view** (pectoral present): fit a line along the pectoral-muscle /
  breast boundary, then drop a perpendicular from the nipple centroid to it.
  The length of that perpendicular is the **PNL** (Posterior Nipple Line).
* **CC view** (no pectoral): a horizontal line from the nipple centroid back
  to the chest-wall edge of the breast tissue — the **CC depth**.

The geometry mirrors ``breast_seg/geometry.py`` (the module that produced the
``example_mlo.png`` / ``example_cc.png`` reference figures). It is vendored
here — rather than imported — so the Qt toolkit stays self-contained and
keeps working when bundled as a stand-alone executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Tissue keys as defined in ``interactive.prompts.TISSUE_PRESETS``.
NIPPLE_KEY = "nipple"
PECTORAL_KEY = "pectoral"
BREAST_KEY = "breast"


# ── Result containers ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PectoralLine:
    point_a: Tuple[int, int]  # (x, y) top
    point_b: Tuple[int, int]  # (x, y) bottom
    boundary_points: Tuple[Tuple[int, int], ...] = ()

    @property
    def direction(self) -> np.ndarray:
        d = np.array(self.point_b, dtype=float) - np.array(self.point_a, dtype=float)
        norm = np.linalg.norm(d)
        return d / norm if norm > 0 else d


@dataclass(frozen=True)
class PNLResult:
    nipple_center: Tuple[int, int]
    foot_point: Tuple[int, int]
    distance_px: float


@dataclass(frozen=True)
class CCDepthResult:
    nipple_center: Tuple[int, int]
    edge_point: Tuple[int, int]
    distance_px: float
    breast_side: str


@dataclass
class AnalysisResult:
    """Everything the canvas needs to draw the overlay."""

    is_mlo: bool
    has_nipple: bool
    has_pectoral: bool
    pectoral_line: Optional[PectoralLine] = None
    pnl: Optional[PNLResult] = None
    cc_depth: Optional[CCDepthResult] = None
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.pnl is not None or self.cc_depth is not None


# ── Mask extraction from an ImageAnnotation ───────────────────────────────


def mask_for_tissue(ann, tissue_key: str) -> Optional[np.ndarray]:
    """Return a binary ``H×W`` uint8 mask (0/255) for ``tissue_key``.

    Prefers the stored union mask; falls back to rasterising the tissue's
    polygons so a polygon-only label (not yet rasterised) still works.
    """
    h, w = ann.image_hw
    out = np.zeros((h, w), dtype=np.uint8)

    stored = ann.masks.get(tissue_key)
    if stored is not None and stored.any():
        out = np.where(stored > 0, np.uint8(255), np.uint8(0))

    polys = ann.polygons.get(tissue_key) or []
    if polys:
        for poly in polys:
            if len(poly) < 3:
                continue
            pts = np.array([[int(round(x)), int(round(y))] for x, y in poly], dtype=np.int32)
            cv2.fillPoly(out, [pts], 255)

    return out if out.any() else None


# ── Pectoral line fitting (mirrors breast_seg/geometry.py) ─────────────────


def _find_boundary_edge_points(
    pectoral_mask: np.ndarray, breast_mask: np.ndarray
) -> Tuple[List[Tuple[int, int]], bool]:
    pec_bin = (pectoral_mask > 0).astype(np.uint8)

    rows_with_pec = np.where(pec_bin.any(axis=1))[0]
    if len(rows_with_pec) < 2:
        return [], True

    top_row = rows_with_pec[0]
    bot_row = rows_with_pec[-1]

    pec_coords = np.argwhere(pec_bin > 0)
    brs_coords = np.argwhere((breast_mask > 0).astype(np.uint8) > 0)
    if len(pec_coords) == 0 or len(brs_coords) == 0:
        return [], True

    breast_is_right = brs_coords[:, 1].mean() > pec_coords[:, 1].mean()

    edge_points: List[Tuple[int, int]] = []
    for row in range(top_row, bot_row + 1):
        cols_with_pec = np.where(pec_bin[row, :])[0]
        if len(cols_with_pec) == 0:
            continue
        edge_col = cols_with_pec.max() if breast_is_right else cols_with_pec.min()
        edge_points.append((int(edge_col), int(row)))

    return edge_points, breast_is_right


def fit_pectoral_line(
    pectoral_mask: np.ndarray,
    breast_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[PectoralLine]:
    if pectoral_mask is None or breast_mask is None:
        return None

    edge_points, breast_is_right = _find_boundary_edge_points(pectoral_mask, breast_mask)
    if len(edge_points) < 2:
        return None

    edge_points.sort(key=lambda p: p[1])
    pt_top = edge_points[0]
    pt_bot = edge_points[-1]

    pectoral_height = pt_bot[1] - pt_top[1]
    mid_y = (pt_top[1] + pt_bot[1]) / 2.0
    threshold = pectoral_height * 0.05

    middle_points = [p for p in edge_points if abs(p[1] - mid_y) <= threshold] or edge_points
    pt_max = (max if breast_is_right else min)(middle_points, key=lambda p: p[0])

    dx = pt_bot[0] - pt_top[0]
    dy = pt_bot[1] - pt_top[1]
    slope_orig = 0.0 if abs(dy) < 1e-6 else float(dx) / float(dy)

    shifted_bot_x = pt_max[0] + slope_orig * (pt_bot[1] - pt_max[1])
    target_bot_x = (shifted_bot_x + pt_bot[0]) / 2.0

    dy_bot = pt_bot[1] - pt_max[1]
    slope_new = slope_orig if abs(dy_bot) < 1e-6 else (target_bot_x - pt_max[0]) / dy_bot
    intercept_new = pt_max[0] - slope_new * pt_max[1]

    h, w = image_shape
    y_top, y_bot = pt_top[1], pt_bot[1]
    x_top = int(round(slope_new * y_top + intercept_new))
    x_bot = int(round(slope_new * y_bot + intercept_new))
    x_top = max(0, min(w - 1, x_top))
    x_bot = max(0, min(w - 1, x_bot))

    return PectoralLine(
        point_a=(x_top, y_top),
        point_b=(x_bot, y_bot),
        boundary_points=(pt_top, pt_bot, pt_max),
    )


def compute_nipple_centroid(nipple_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    if nipple_mask is None:
        return None
    moments = cv2.moments((nipple_mask > 0).astype(np.uint8))
    if moments["m00"] == 0:
        return None
    return (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))


def compute_pnl(nipple_center: Tuple[int, int], pectoral_line: PectoralLine) -> PNLResult:
    P = np.array(nipple_center, dtype=float)
    A = np.array(pectoral_line.point_a, dtype=float)
    B = np.array(pectoral_line.point_b, dtype=float)

    AB = B - A
    ab_dot = np.dot(AB, AB)
    if ab_dot < 1e-10:
        return PNLResult(nipple_center, pectoral_line.point_a, float(np.linalg.norm(P - A)))

    t = max(0.0, min(1.0, float(np.dot(P - A, AB) / ab_dot)))
    foot = A + t * AB
    return PNLResult(
        nipple_center=nipple_center,
        foot_point=(int(round(foot[0])), int(round(foot[1]))),
        distance_px=float(np.linalg.norm(P - foot)),
    )


def compute_cc_depth(
    nipple_center: Tuple[int, int],
    breast_mask: np.ndarray,
    image_shape: Tuple[int, int],
) -> Optional[CCDepthResult]:
    if breast_mask is None:
        return None

    brs_bin = (breast_mask > 0).astype(np.uint8)
    h, w = image_shape
    nx, ny = nipple_center

    brs_coords = np.argwhere(brs_bin > 0)
    if len(brs_coords) == 0:
        return None

    breast_side = "right" if brs_coords[:, 1].mean() > w / 2.0 else "left"

    row_band = 10
    band = brs_bin[max(0, ny - row_band) : min(h, ny + row_band + 1), :]
    cols_with_breast = np.where(band.any(axis=0))[0]
    if len(cols_with_breast) == 0:
        return None

    edge_col = int(cols_with_breast.max()) if breast_side == "right" else int(cols_with_breast.min())
    return CCDepthResult(
        nipple_center=nipple_center,
        edge_point=(edge_col, ny),
        distance_px=float(abs(edge_col - nx)),
        breast_side=breast_side,
    )


# ── Top-level entry point ─────────────────────────────────────────────────


def analyze_annotation(ann) -> AnalysisResult:
    """Compute PNL (MLO) or CC depth (CC) from a single image's annotation.

    View is inferred the same way ``breast_seg`` does it: a labelled pectoral
    muscle ⇒ MLO, otherwise CC.
    """
    image_shape = ann.image_hw

    nipple_mask = mask_for_tissue(ann, NIPPLE_KEY)
    breast_mask = mask_for_tissue(ann, BREAST_KEY)
    pectoral_mask = mask_for_tissue(ann, PECTORAL_KEY)

    has_nipple = nipple_mask is not None
    has_pectoral = pectoral_mask is not None
    is_mlo = has_pectoral

    result = AnalysisResult(is_mlo=is_mlo, has_nipple=has_nipple, has_pectoral=has_pectoral)

    if not has_nipple:
        result.messages.append("Nipple etiketlenmemiş — ölçüm için meme ucu gerekli.")
        return result

    nipple_center = compute_nipple_centroid(nipple_mask)
    if nipple_center is None:
        result.messages.append("Nipple maskesi boş.")
        return result

    if is_mlo:
        if breast_mask is None:
            result.messages.append("Breast tissue etiketlenmemiş — pectoral hattı çizilemiyor.")
            return result
        line = fit_pectoral_line(pectoral_mask, breast_mask, image_shape)
        if line is None:
            result.messages.append("Pectoral hattı oturtulamadı.")
            return result
        result.pectoral_line = line
        result.pnl = compute_pnl(nipple_center, line)
    else:
        if breast_mask is None:
            result.messages.append("Breast tissue etiketlenmemiş — CC derinliği çizilemiyor.")
            return result
        result.cc_depth = compute_cc_depth(nipple_center, breast_mask, image_shape)
        if result.cc_depth is None:
            result.messages.append("CC derinliği hesaplanamadı.")

    return result
