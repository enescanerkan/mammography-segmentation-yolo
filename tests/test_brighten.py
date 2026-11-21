"""Image work of the brightened nipple re-search (breast_seg.inference.brighten), on synthetic arrays."""
from __future__ import annotations

import numpy as np
import pytest

from breast_seg.imaging.preprocess import Transform
from breast_seg.inference import brighten
from breast_seg.inference.brighten import (
    anterior_tip,
    breast_in_box,
    brightened_canvas,
    distance_to,
    mask_gap_mm,
    rewindow,
)


def _identity(size=640, spacing=0.1):
    return Transform(original_width=size, original_height=size, crop_x1=0, crop_y1=0, crop_x2=size, crop_y2=size,
                     scale=1.0, pad_left=0, pad_top=0, pad_right=0, pad_bottom=0, pixel_spacing=spacing)


def _half_disc(chest_side="L"):
    breast = np.zeros((640, 640), np.uint8)
    yy, xx = np.mgrid[:640, :640]
    wall = 0 if chest_side == "L" else 639
    breast[(xx - wall) ** 2 + (yy - 320) ** 2 <= 250 ** 2] = 1
    return breast


def test_the_cc_tip_is_the_point_farthest_from_the_chest_wall():
    assert np.allclose(anterior_tip({"breast-tissue": _half_disc("L")}, "CC", "L"), [250, 320], atol=1.0)
    assert np.allclose(anterior_tip({"breast-tissue": _half_disc("R")}, "CC", "R"), [389, 320], atol=1.0)
    assert anterior_tip({}, "CC", "L") is None


def test_the_mlo_tip_is_the_point_farthest_from_the_pectoral_line():
    breast = _half_disc("L")
    pectoral = np.zeros_like(breast)
    pectoral[70:300, 0:40] = 1                       # a vertical pectoral edge at x ~ 40
    tip = anterior_tip({"breast-tissue": breast, "pectoral": pectoral}, "MLO", "L")
    assert tip[0] > 240 and abs(tip[1] - 320) < 30


def test_rewindow_maps_the_background_to_black_and_the_breast_top_to_white():
    crop = np.full((40, 40), 100.0)
    crop[:, 20:] = np.linspace(110, 140, 20)[None, :]
    inside = np.zeros((40, 40), bool)
    inside[:, 20:] = True
    out = rewindow(crop, inside)
    assert out.dtype == np.uint8 and out[:, :20].max() == 0 and out[:, -1].min() >= 250


def test_breast_in_box_samples_the_640_mask_on_the_original_grid():
    breast = _half_disc("L")
    inside = breast_in_box(breast, _identity(), (200, 300, 260, 340))
    assert inside.shape == (40, 60) and inside[:, :40].all() and not inside[:, -5:].any()


def test_the_window_is_pasted_back_at_the_same_place_and_scale():
    transform = _identity(spacing=0.1)
    pixels = np.where(_half_disc("L") > 0, 120, 100).astype(np.uint16)
    image640 = np.where(_half_disc("L") > 0, 30, 0).astype(np.uint8)      # a dark render of the same breast
    canvas, (x1, y1, x2, y2) = brightened_canvas(pixels, {"photometric": "MONOCHROME2"}, image640, transform,
                                                 _half_disc("L"), np.array([250.0, 320.0]), window_mm=50.0)
    assert (x1, y1, x2, y2) == (0, 70, 500, 570)                            # 50 mm at 0.1 mm/px, cut at the edge
    assert canvas[320, 240] == 255 and canvas[320, 260] == 0                 # breast bright, background black
    assert (canvas[:70] == image640[:70]).all() and (canvas[:, 500:] == image640[:, 500:]).all()  # rest untouched
    assert brightened_canvas(pixels, {}, image640, transform, _half_disc("L"), np.array([250.0, 320.0]),
                             window_mm=1.0) is None                          # too small to search


def test_mask_gap_is_the_shortest_distance_to_the_breast():
    breast = np.zeros((640, 640), np.uint8)
    breast[100:500, :300] = 1
    nipple = np.zeros_like(breast)
    nipple[290:310, 295:320] = 1                       # overlaps the breast
    to_breast = distance_to(breast)                    # one map per image, shared by every candidate
    assert mask_gap_mm(nipple, to_breast, 0.3) == 0.0
    nipple[:] = 0
    nipple[290:310, 310:330] = 1                       # its nearest pixel is 11 px beyond the last breast column
    assert mask_gap_mm(nipple, to_breast, 0.3) == pytest.approx(3.3)
    assert distance_to(np.zeros_like(breast)) is None and mask_gap_mm(nipple, None, 0.3) == np.inf  # no breast
    assert mask_gap_mm(np.zeros_like(breast), to_breast, 0.3) == np.inf


def test_a_monochrome1_file_is_inverted_in_the_window_only(monkeypatch):
    """The brightened window of a MONOCHROME1 file is what the inverted whole frame would give, without inverting
    (and copying) the whole frame for it."""
    transform = _identity(spacing=0.1)
    rng = np.random.default_rng(3)
    tissue = np.where(_half_disc("L") > 0, 900, 200).astype(np.uint16) + rng.integers(0, 40, (640, 640), dtype=np.uint16)
    tissue[:40, :40] = 4000                              # the frame's brightest pixels lie outside the window
    stored = (4095 - tissue).astype(np.uint16)           # as a MONOCHROME1 file stores it: air bright, tissue dark
    image640 = np.where(_half_disc("L") > 0, 30, 0).astype(np.uint8)
    tip = np.array([250.0, 320.0])

    plain = brightened_canvas(tissue, {"photometric": "MONOCHROME2"}, image640, transform, _half_disc("L"), tip, 20.0)
    inverted_shapes = []
    display = brighten.display_pixels
    monkeypatch.setattr(brighten, "display_pixels",
                        lambda pixels, meta: inverted_shapes.append(pixels.shape) or display(pixels, meta))
    mono1 = brightened_canvas(stored, {"photometric": "MONOCHROME1"}, image640, transform, _half_disc("L"), tip, 20.0)
    assert inverted_shapes == [(200, 200)]               # the 20 mm window, not the 640 x 640 frame
    assert mono1[1] == plain[1] and (mono1[0] == plain[0]).all()
