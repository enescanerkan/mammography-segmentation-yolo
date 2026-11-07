"""Imaging: DICOM decoding with view / side resolution, the 640 recipes, display renders.

The recipes must stay byte-identical to the preprocessing the checkpoints were trained on. Point
``BREAST_SEG_REFERENCE_PREPROCESS`` at the folder holding that reference implementation (``positioning_infer.py``)
to run the parity tests; they are skipped otherwise.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from breast_seg.imaging import (
    DEFAULT_PIXEL_SPACING_MM,
    TARGET_SIZE,
    DicomError,
    Transform,
    chest_wall_side,
    display_pixels,
    find_breast_bbox,
    image_quality,
    preprocess,
    preprocess_pair,
    quick_normalize,
    read_dicom,
    render_display,
    wavelet_enhance,
)
from breast_seg.imaging.phantom import phantom_dicom, phantom_mammogram

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


def _meta(photometric="MONOCHROME2", chest_side="L", spacing=0.07):
    return {"photometric": photometric, "chest_side": chest_side, "pixel_spacing": spacing}


def _pixels(photometric, chest_side, size, seed=3):
    height, width = size
    image = phantom_mammogram(width, height, chest_side="R" if chest_side == "R" else "L", seed=seed)
    return image.max() - image if photometric == "MONOCHROME1" else image


CASES = [(p, side, size) for p in ("MONOCHROME2", "MONOCHROME1") for side in ("L", "R", "U")
         for size in ((1200, 900), (700, 1000))]


# ── parity with the training-time preprocessing ────────────────────────────
@pytest.mark.parametrize("photometric,side,size", CASES)
def test_preprocess_pair_matches_reference(reference, photometric, side, size):
    pixels, meta = _pixels(photometric, side, size), _meta(photometric, side)
    ours = preprocess_pair(pixels.copy(), meta)
    theirs = reference.preprocess_pair(pixels.copy(), meta)
    for a, b in zip(ours[:2], theirs[:2]):
        assert a.dtype == b.dtype and a.shape == b.shape == (TARGET_SIZE, TARGET_SIZE)
        assert np.array_equal(a, b)
    assert ours[2].as_dict() == theirs[2].as_dict()


@pytest.mark.parametrize("enhance", [False, True])
@pytest.mark.parametrize("photometric,side", [("MONOCHROME2", "L"), ("MONOCHROME1", "R"), ("MONOCHROME2", "U")])
def test_preprocess_matches_reference(reference, photometric, side, enhance):
    pixels, meta = _pixels(photometric, side, (1100, 850), seed=11), _meta(photometric, side)
    ours, ours_t = preprocess(pixels.copy(), meta, enhance=enhance)
    theirs, theirs_t = reference.preprocess(pixels.copy(), meta, enhance=enhance)
    assert np.array_equal(ours, theirs)
    assert ours_t.as_dict() == theirs_t.as_dict()


@pytest.mark.parametrize("shape", [(333, 517), (640, 451), (97, 120)])
def test_wavelet_enhance_matches_reference(reference, shape):
    rng = np.random.default_rng(shape[0])
    image = rng.integers(0, 256, shape, dtype=np.uint8)
    assert np.array_equal(wavelet_enhance(image), reference.wavelet_enhance(image))


def test_chest_wall_side_matches_reference(reference):
    pixels = phantom_mammogram(300, 400, chest_side="R")
    for laterality in ("L", "R", "", None):
        for flip in ("YES", "NO", None):
            ds = SimpleNamespace(**{k: v for k, v in (("ImageLaterality", laterality),
                                                      ("FieldOfViewHorizontalFlip", flip)) if v is not None})
            assert chest_wall_side(ds, pixels) == reference.chest_wall_side(ds, pixels)
            assert chest_wall_side(ds) == reference.chest_wall_side(ds)


def test_transform_and_constants_match_reference(reference):
    assert TARGET_SIZE == reference.TARGET_SIZE
    pixels, meta = _pixels("MONOCHROME2", "R", (900, 700)), _meta("MONOCHROME2", "R")
    _, transform = preprocess(pixels, meta, enhance=False)
    theirs = reference.Transform(**transform.as_dict())
    points = np.array([[0.0, 0.0], [320.5, 100.25], [639.0, 639.0]])
    assert np.array_equal(transform.to_original(points), theirs.to_original(points))
    assert transform.mm_per_640px == theirs.mm_per_640px


# ── recipes ────────────────────────────────────────────────────────────────
def test_transform_round_trip():
    pixels, meta = _pixels("MONOCHROME2", "L", (1200, 900)), _meta()
    _, transform = preprocess(pixels, meta, enhance=False)
    points = np.array([[10.0, 20.0], [400.0, 600.0]])
    assert np.allclose(transform.to_model(transform.to_original(points)), points)
    assert transform.mm_per_640px == pytest.approx(0.07 / transform.scale)


@pytest.mark.parametrize("side", ["L", "R"])
def test_padding_goes_away_from_the_chest_wall(side):
    pixels = phantom_mammogram(900, 1200, chest_side=side)
    image, transform = preprocess(pixels, _meta(chest_side=side), enhance=False)
    assert image.shape == (TARGET_SIZE, TARGET_SIZE) and image.dtype == np.uint8
    if side == "L":
        assert transform.pad_left == 0 and transform.pad_right > 0
        assert image[:, :4].max() > 0 and image[:, -4:].max() == 0
    else:
        assert transform.pad_right == 0 and transform.pad_left > 0
        assert image[:, -4:].max() > 0 and image[:, :4].max() == 0


def test_monochrome1_is_inverted_before_the_crop():
    plain = phantom_mammogram(600, 800)
    a, ta = preprocess(plain, _meta("MONOCHROME2"), enhance=False)
    b, tb = preprocess(plain.max() - plain, _meta("MONOCHROME1"), enhance=False)
    assert ta.as_dict() == tb.as_dict()
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_pair_shares_geometry():
    baseline, wavelet, transform = preprocess_pair(phantom_mammogram(600, 800), _meta())
    assert baseline.shape == wavelet.shape == (TARGET_SIZE, TARGET_SIZE)
    assert np.array_equal(baseline == 0, baseline == 0)
    pad = transform.pad_right
    assert pad > 0 and wavelet[:, -pad:].max() == 0  # the enhanced crop is re-padded with black


def test_helpers_on_degenerate_input():
    assert not quick_normalize(np.full((5, 5), 7.0)).any()
    assert find_breast_bbox(np.zeros((300, 200), np.uint8)) == (0, 0, 200, 300)
    assert chest_wall_side(SimpleNamespace()) == "U"


# ── DICOM ──────────────────────────────────────────────────────────────────
def test_read_dicom_header_facts():
    data = phantom_dicom("MLO", "L", attributes={"CompressionForce": 95.0, "BodyPartThickness": 51.0, "KVP": 29.0})
    pixels, meta = read_dicom(data, "upload/L-MLO.dcm")
    assert pixels.dtype == np.float32 and pixels.shape == (1200, 900)
    assert (meta["view"], meta["laterality"], meta["view_key"]) == ("MLO", "L", "L-MLO")
    assert (meta["view_source"], meta["laterality_source"]) == ("ViewPosition", "ImageLaterality")
    assert meta["pixel_spacing"] == pytest.approx(0.1) and meta["pixel_spacing_known"]
    assert meta["chest_side"] == "L"
    assert meta["physics"]["compression_n"] == 95.0 and meta["physics"]["kvp"] == 29.0
    assert meta["physics"]["exposure_uas"] is None


def test_read_dicom_keeps_no_identifiers():
    _, meta = read_dicom(phantom_dicom("CC", "R"), "R-CC.dcm")
    assert set(meta) == {
        "filename", "photometric", "laterality", "view", "view_key", "view_source", "laterality_source",
        "pixel_spacing", "pixel_spacing_known", "horizontal_flip", "chest_side", "manufacturer", "model_name",
        "original_width", "original_height", "physics"}
    assert not any("uid" in key.lower() or "patient" in key.lower() for key in meta)


def test_view_from_code_sequence():
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence

    item = Dataset()
    item.CodeValue = "399162004"
    item.CodeMeaning = "cranio-caudal"
    data = phantom_dicom("CC", "R", include_view=False, attributes={"ViewCodeSequence": Sequence([item])})
    _, meta = read_dicom(data, "x.dcm")
    assert (meta["view"], meta["view_source"]) == ("CC", "ViewCodeSequence")


def test_view_and_side_from_series_description():
    data = phantom_dicom("MLO", "L", include_view=False, include_laterality=False,
                         attributes={"SeriesDescription": "L MLO tomo-free"})
    _, meta = read_dicom(data, "IM0001.dcm")
    assert (meta["view"], meta["view_source"]) == ("MLO", "SeriesDescription")
    assert (meta["laterality"], meta["laterality_source"]) == ("L", "SeriesDescription")


@pytest.mark.parametrize("name,view,side,source", [
    ("case-042/RCC/IM0001.dcm", "CC", "R", "folder name"),
    ("export/L_MLO.dcm", "MLO", "L", "file name"),
    ("lmlo.dcm", "MLO", "L", "file name"),
    ("right/CC/IM7.dcm", "CC", "R", "folder name"),
])
def test_view_and_side_from_the_upload_path(name, view, side, source):
    data = phantom_dicom(view, side, include_view=False, include_laterality=False)
    _, meta = read_dicom(data, name)
    assert (meta["view"], meta["laterality"]) == (view, side)
    assert meta["view_source"] == source


def test_accession_is_not_a_view():
    data = phantom_dicom("CC", "L", include_view=False)
    with pytest.raises(DicomError, match="could not tell CC from MLO"):
        read_dicom(data, "accession.dcm")


def test_true_lateral_is_rejected():
    with pytest.raises(DicomError, match="only MLO and CC"):
        read_dicom(phantom_dicom("ML", "L"), "L-ML.dcm")


def test_missing_side_is_reported():
    data = phantom_dicom("MLO", "L", include_laterality=False)
    with pytest.raises(DicomError, match="could not tell left from right"):
        read_dicom(data, "IM0001.dcm")


def test_non_images_and_garbage():
    with pytest.raises(DicomError, match="skipped"):
        read_dicom(b"labels", "case/RMLO/labels.txt")
    with pytest.raises(DicomError, match="not readable as DICOM"):
        read_dicom(b"\x00" * 64, "broken.dcm")


def test_missing_spacing_falls_back_to_the_default():
    _, meta = read_dicom(phantom_dicom("CC", "L", spacing=None), "L-CC.dcm")
    assert meta["pixel_spacing"] == DEFAULT_PIXEL_SPACING_MM and not meta["pixel_spacing_known"]


def test_flipped_monochrome1_without_laterality_tag_follows_the_pixels():
    # An L breast stored mirrored: the chest wall is on the right of the pixel data.
    data = phantom_dicom("CC", "L", photometric="MONOCHROME1", flipped=True, include_laterality=False)
    pixels, meta = read_dicom(data, "LCC/IM1.dcm")
    assert meta["laterality"] == "L" and meta["chest_side"] == "R" and meta["horizontal_flip"]
    shown = display_pixels(pixels, meta)
    assert shown[:, -20:].mean() > shown[:, :20].mean()  # tissue bright after inversion, at the chest wall side


def test_flip_tag_moves_the_chest_side():
    _, meta = read_dicom(phantom_dicom("MLO", "R", flipped=True), "R-MLO.dcm")
    assert meta["chest_side"] == "L"


# ── renders ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("photometric", ["MONOCHROME2", "MONOCHROME1"])
def test_render_display_matches_the_model_frame(photometric):
    import cv2

    pixels = _pixels(photometric, "L", (1200, 900))
    meta = _meta(photometric)
    baseline, _, transform = preprocess_pair(pixels, meta)
    assert np.array_equal(render_display(pixels, meta, transform, 640), baseline)
    display = render_display(pixels, meta, transform, 1280)
    assert display.shape == (1280, 1280) and display.dtype == np.uint8
    reduced = cv2.resize(display, (640, 640), interpolation=cv2.INTER_AREA)
    assert np.abs(reduced.astype(float) - baseline.astype(float)).mean() < 6.0


def test_image_quality():
    assert image_quality(np.zeros((640, 640), np.uint8)) == {"sharpness": 0.0, "clipped_fraction": 0.0, "mean_level": 0.0}
    baseline, _, _ = preprocess_pair(phantom_mammogram(600, 800), _meta())
    quality = image_quality(baseline)
    assert quality["sharpness"] > 0 and 0 <= quality["clipped_fraction"] <= 1 and quality["mean_level"] > 12


def test_transform_is_a_plain_dataclass():
    transform = Transform(10, 20, 0, 0, 10, 20, 0.5, 0, 0, 0, 0, 0.1)
    assert transform.as_dict()["scale"] == 0.5
