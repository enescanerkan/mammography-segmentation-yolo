"""Geometry: the landmark route (perpendicular, chest-wall distance) and the mask route (pectoral fit, PNL, depth)."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from breast_seg import geometry

REFERENCE_ENV = "BREAST_SEG_REFERENCE_PREPROCESS"


@pytest.fixture(scope="module")
def reference():
    folder = os.environ.get(REFERENCE_ENV, "").strip()
    if not folder or not (Path(folder) / "positioning_infer.py").is_file():
        pytest.skip(f"set {REFERENCE_ENV} to the folder of the reference positioning_infer.py")
    sys.path.insert(0, folder)
    try:
        return importlib.import_module("positioning_infer")
    finally:
        sys.path.remove(folder)


def test_perpendicular_distance_to_a_vertical_line():
    distance, foot = geometry.perpendicular_distance(np.array([10.0, 0.0]), np.array([10.0, 100.0]),
                                                     np.array([40.0, 30.0]))
    assert distance == pytest.approx(30.0)
    assert np.allclose(foot, [10.0, 30.0])


def test_perpendicular_distance_is_not_clamped_to_the_segment():
    distance, foot = geometry.perpendicular_distance(np.array([0.0, 0.0]), np.array([10.0, 0.0]),
                                                     np.array([50.0, 5.0]))
    assert distance == pytest.approx(5.0) and np.allclose(foot, [50.0, 0.0])


def test_perpendicular_distance_degenerate_line():
    distance, foot = geometry.perpendicular_distance(np.array([3.0, 4.0]), np.array([3.0, 4.0]), np.array([0.0, 0.0]))
    assert distance == pytest.approx(5.0) and np.allclose(foot, [3.0, 4.0])


def test_cc_chestwall_distance_follows_the_pixel_side():
    distance, edge = geometry.cc_chestwall_distance(np.array([200.0, 310.0]), "L")
    assert distance == 200.0 and np.allclose(edge, [0.0, 310.0])
    distance, edge = geometry.cc_chestwall_distance(np.array([200.0, 310.0]), "R")
    assert distance == 440.0 and np.allclose(edge, [640.0, 310.0])
    distance, _ = geometry.cc_chestwall_distance(np.array([200.0, 310.0]), "R", frame_size=1280)
    assert distance == 1080.0


def test_landmark_route_matches_reference(reference):
    rng = np.random.default_rng(7)
    for _ in range(50):
        p1, p2, point = rng.uniform(0, 640, (3, 2))
        ours = geometry.perpendicular_distance(p1, p2, point)
        theirs = reference.perpendicular_distance(p1, p2, point)
        assert ours[0] == theirs[0] and np.array_equal(ours[1], theirs[1])
        for side in ("L", "R"):
            ours = geometry.cc_chestwall_distance(point, side)
            theirs = reference.cc_chestwall_distance(point, side)
            assert ours[0] == theirs[0] and np.array_equal(ours[1], theirs[1])


# ── mask route ─────────────────────────────────────────────────────────────
def _masks(size=200):
    """Breast filling the left 150 columns, pectoral wedge in its top-left corner, chest wall on the left."""
    breast = np.zeros((size, size), np.uint8)
    breast[10:190, 0:150] = 1
    pectoral = np.zeros((size, size), np.uint8)
    cv2.fillPoly(pectoral, [np.array([[0, 10], [60, 10], [0, 120]], np.int32)], 1)
    pectoral &= breast
    nipple = np.zeros((size, size), np.uint8)
    cv2.circle(nipple, (145, 100), 4, 1, -1)
    return breast, pectoral, nipple


def test_pectoral_line_and_pnl():
    breast, pectoral, nipple = _masks()
    line = geometry.fit_pectoral_line(pectoral, breast, breast.shape)
    assert line is not None
    rows = np.where(pectoral.any(axis=1))[0]
    assert (line.point_a[1], line.point_b[1]) == (rows.min(), rows.max())  # topmost and bottommost pectoral rows
    centre = geometry.compute_nipple_centroid(nipple)
    assert centre == (145, 100)
    pnl = geometry.compute_pnl(centre, line)
    assert pnl.distance_px > 0
    # The foot lies on the pectoral segment.
    a, b, foot = np.array(line.point_a), np.array(line.point_b), np.array(pnl.foot_point)
    assert np.linalg.norm(a - foot) + np.linalg.norm(foot - b) == pytest.approx(np.linalg.norm(a - b), abs=1.5)


def test_pnl_foot_is_clamped_to_the_segment():
    line = geometry.PectoralLine(point_a=(0, 0), point_b=(0, 10), slope=float("inf"), intercept=0.0)
    pnl = geometry.compute_pnl((5, 50), line)
    assert pnl.foot_point == (0, 10) and pnl.distance_px == pytest.approx(np.hypot(5, 40))


def test_cc_depth():
    breast, _, nipple = _masks()
    depth = geometry.compute_cc_depth(geometry.compute_nipple_centroid(nipple), breast, breast.shape)
    assert depth.breast_side == "left" and depth.edge_point == (0, 100) and depth.distance_px == 145.0


def test_empty_masks():
    assert geometry.compute_nipple_centroid(np.zeros((10, 10), np.uint8)) is None
    assert geometry.compute_nipple_centroid(None) is None
    assert geometry.fit_pectoral_line(None, None, (10, 10)) is None
    assert geometry.compute_cc_depth((1, 1), np.zeros((10, 10), np.uint8), (10, 10)) is None
