"""qt_app.pnl_analysis delegates all geometry to breast_seg.geometry (synthetic masks only)."""

from __future__ import annotations

import cv2
import numpy as np

from breast_seg import geometry  # importable via conftest's src/ fallback when not pip-installed
from qt_app import pnl_analysis
from qt_app.session import ImageAnnotation

H, W = 400, 300


def _disk(cx: int, cy: int, r: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m


def _mlo_annotation() -> ImageAnnotation:
    """Pectoral wedge top-left (as a polygon), breast to its right (as a mask), nipple disk."""
    ann = ImageAnnotation(image_hw=(H, W))
    # A slightly convex pectoral edge so the tangent rule has a bulge to touch.
    ann.polygons["pectoral"] = [[(0, 0), (110, 0), (95, 120), (60, 220), (0, 260)]]
    breast = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(breast, [np.array([[0, 0], [260, 0], [280, 200], [240, 380], [0, 390]], np.int32)], 1)
    pec = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(pec, [np.array(ann.polygons["pectoral"][0], np.int32)], 1)
    ann.masks["breast"] = (breast & (1 - pec)).astype(np.uint8)
    ann.masks["nipple"] = _disk(250, 210, 6)
    return ann


def test_same_functions_as_the_package():
    assert pnl_analysis.fit_pectoral_line is geometry.fit_pectoral_line
    assert pnl_analysis.compute_pnl is geometry.compute_pnl
    assert pnl_analysis.compute_cc_depth is geometry.compute_cc_depth
    assert pnl_analysis.compute_nipple_centroid is geometry.compute_nipple_centroid


def test_mlo_pnl_matches_breast_seg_geometry():
    ann = _mlo_annotation()
    result = pnl_analysis.analyze_annotation(ann)
    assert result.ok and result.is_mlo and result.has_pectoral and result.has_nipple

    pec = pnl_analysis.mask_for_tissue(ann, "pectoral")
    brs = pnl_analysis.mask_for_tissue(ann, "breast")
    nip = pnl_analysis.mask_for_tissue(ann, "nipple")
    line = geometry.fit_pectoral_line(pec, brs, (H, W))
    pnl = geometry.compute_pnl(geometry.compute_nipple_centroid(nip), line)

    assert result.pectoral_line == line
    assert result.pnl == pnl
    assert result.pnl.distance_px > 0
    # foot lies on the fitted pectoral segment
    (ax, ay), (bx, by) = line.point_a, line.point_b
    fx, fy = result.pnl.foot_point
    assert min(ay, by) - 1 <= fy <= max(ay, by) + 1
    assert min(ax, bx) - 1 <= fx <= max(ax, bx) + 1


def test_cc_depth_matches_breast_seg_geometry():
    ann = ImageAnnotation(image_hw=(H, W))
    breast = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(breast, (0, 200), (200, 170), 0, -90, 90, 1, -1)  # breast against the left edge
    ann.masks["breast"] = breast
    ann.masks["nipple"] = _disk(190, 200, 5)

    result = pnl_analysis.analyze_annotation(ann)
    assert result.ok and not result.is_mlo
    expected = geometry.compute_cc_depth(geometry.compute_nipple_centroid(ann.masks["nipple"]), breast, (H, W))
    assert result.cc_depth == expected
    assert result.cc_depth.breast_side == "left"


def test_messages_when_labels_are_missing():
    ann = ImageAnnotation(image_hw=(H, W))
    ann.masks["breast"] = _disk(100, 200, 80)
    result = pnl_analysis.analyze_annotation(ann)
    assert not result.ok
    assert "Nipple" in result.messages[0]

    ann = ImageAnnotation(image_hw=(H, W))
    ann.masks["nipple"] = _disk(100, 200, 5)
    ann.masks["pectoral"] = _disk(20, 20, 15)
    result = pnl_analysis.analyze_annotation(ann)
    assert not result.ok and result.is_mlo
    assert "pectoral line" in result.messages[0]
