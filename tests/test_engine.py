"""LandmarkEngine with injected fake models (no weights, CPU), plus one opt-in run on real weights.

The real-weights test reads the weights root from ``BREAST_SEG_WEIGHTS`` and is skipped when the files are absent.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from breast_seg.imaging import TARGET_SIZE, Transform, preprocess_pair, read_dicom
from breast_seg.imaging.phantom import phantom_dicom
from breast_seg.inference import EngineError, LandmarkEngine, landmark_record, largest_polygon
from breast_seg.inference.testing import fake_loader, placeholder_weights
from breast_seg.registry import WEIGHTS_ENV, Registry

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"


@pytest.fixture()
def registry(tmp_path):
    return Registry.load(MODELS_YAML, weights_root=placeholder_weights(MODELS_YAML, tmp_path))


def _engine(registry, **options):
    return LandmarkEngine(registry, device="cpu", preview_size=960, loader=fake_loader(**options)).load()


def _image(view="MLO", laterality="L", **kwargs):
    return read_dicom(phantom_dicom(view, laterality, **kwargs), f"{laterality}-{view}.dcm")


def test_loads_the_role_models(registry):
    engine = _engine(registry)
    status = engine.status()
    assert engine.ready and status["errors"] == {}
    assert status["loaded"] == ["lesion_det", "pose_cc", "pose_mlo", "seg_v4"]
    assert status["roles"] == {"pose_mlo": "pose_mlo", "pose_cc": "pose_cc", "segmentation": "seg_v4",
                               "detection": "lesion_det"}
    assert status["device"] == "cpu"
    assert engine.loaded_weights()["seg_v4"] == "breast_seg_v4_yolo26m.pt"


def test_analyse_mlo(registry):
    engine = _engine(registry)
    pixels, meta = _image("MLO", "L")
    out = engine.analyse(pixels, meta)
    assert set(out) == {"transform", "mm_per_px", "display", "quality", "landmarks", "masks", "polygons", "lesions"}
    assert out["display"].shape == (960, 960)
    assert out["mm_per_px"] == pytest.approx(Transform(**out["transform"]).mm_per_640px)
    landmarks = out["landmarks"]
    assert landmarks["detected"] and not landmarks["mirrored"] and landmarks["confidence"] == 0.9
    assert set(landmarks["points"]) == {"nipple", "pectoral_top", "pectoral_bottom"}
    assert set(out["polygons"]) == {"breast-tissue", "pectoral", "nipple"}
    for name, mask in out["masks"].items():
        assert mask.shape == (TARGET_SIZE, TARGET_SIZE) and mask.dtype == np.uint8
        polygon = out["polygons"][name]["points"]
        assert len(polygon) >= 3 and all(0 <= c <= TARGET_SIZE for point in polygon for c in point)
    assert out["lesions"][0]["label"] == "Mass" and len(out["lesions"][0]["box"]) == 4
    # every model call ran on the configured device, at the trained input size
    models = engine._models
    assert models["pose_mlo"].calls and models["seg_v4"].calls and models["lesion_det"].calls
    assert not models["pose_cc"].calls
    for model in models.values():
        assert all(call["device"] == "cpu" and call["imgsz"] == TARGET_SIZE for call in model.calls)


def test_cc_has_only_the_nipple_keypoint(registry):
    out = _engine(registry).analyse(*_image("CC", "R"))
    assert set(out["landmarks"]["points"]) == {"nipple"}


def test_nipple_rescue(registry):
    out = _engine(registry, nipple_conf=0.1).analyse(*_image("MLO", "R"))
    nipple = out["polygons"]["nipple"]
    assert nipple["rescued"] and nipple["confidence"] == 0.1
    assert out["polygons"]["breast-tissue"]["rescued"] is False


def test_nipple_rescue_disabled(tmp_path):
    raw = Registry.load(MODELS_YAML)
    models = [{"id": m.id, "task": m.task, "view": m.view, "recipe": m.recipe, "weights": m.weights,
               "enabled": m.enabled, "classes": list(m.classes), "keypoints": list(m.keypoints),
               "nipple_rescue": {"enabled": False}} for m in raw.models]
    registry = Registry.from_dict({"models": models}, weights_root=placeholder_weights(MODELS_YAML, tmp_path))
    out = _engine(registry, nipple_conf=0.1).analyse(*_image("MLO", "R"))
    assert "nipple" not in out["polygons"]


def test_mirror_fallback_maps_points_back(registry):
    pixels, meta = _image("MLO", "L")
    plain = _engine(registry).analyse(pixels, meta)["landmarks"]
    mirrored = _engine(registry, only_mirrored=True).analyse(pixels, meta)["landmarks"]
    assert mirrored["mirrored"] and not plain["mirrored"]
    for name in ("nipple", "pectoral_bottom"):
        assert np.allclose(mirrored["points"][name], plain["points"][name], atol=2.0)


def test_nothing_found(registry):
    out = _engine(registry, find_nothing=True).analyse(*_image("CC", "L"))
    assert out["landmarks"] == {"detected": False, "mirrored": False, "confidence": 0.0, "points": {}}


def test_missing_weights_are_reported_not_raised(tmp_path):
    registry = Registry.load(MODELS_YAML, weights_root=tmp_path)
    engine = LandmarkEngine(registry, loader=fake_loader()).load()
    assert not engine.ready
    expected = "weights file not found: segmentation/breast_seg_v4_yolo26m.pt (under the weights root)"
    assert engine.errors["seg_v4"] == expected
    assert str(tmp_path) not in str(engine.status())
    with pytest.raises(EngineError):
        engine.analyse(*_image())
    with pytest.raises(EngineError):
        engine.analyse_landmarks(*_image())


def test_loader_failure_is_recorded(registry):
    def broken(spec):
        raise OSError("corrupt checkpoint")

    engine = LandmarkEngine(registry, loader=broken).load()
    assert not engine.ready and engine.errors["pose_mlo"] == "failed to load: corrupt checkpoint"


def test_missing_role(tmp_path):
    registry = Registry.from_dict({"models": []}, weights_root=tmp_path)
    engine = LandmarkEngine(registry, loader=fake_loader()).load()
    assert not engine.ready and "no enabled segmentation model" in engine.errors["segmentation"]


def _crossview_summary(landmarks, polygons, transform):
    """The landmark record exactly as the cross-view project's adapter built it from the web app's engine."""
    t = Transform(**transform)
    pts = {k: t.to_original(np.array([p]))[0].round(1).tolist() for k, p in (landmarks.get("points") or {}).items()}
    shapes = {}
    for k, e in polygons.items():
        if e["points"]:
            shapes[k] = dict(p=t.to_original(np.array(e["points"], dtype=float)).round(1).tolist(), conf=e["confidence"],
                             rescued=e.get("rescued", False))
    return dict(points=pts, conf=landmarks.get("confidence"), polygons=shapes)


@pytest.mark.parametrize("view,laterality,flipped", [("MLO", "L", False), ("CC", "R", False), ("CC", "L", True)])
def test_analyse_landmarks_reproduces_the_crossview_record(registry, view, laterality, flipped):
    engine = _engine(registry, nipple_conf=0.1)
    pixels, meta = _image(view, laterality, flipped=flipped)
    record = engine.analyse_landmarks(pixels, meta)
    out = engine.analyse(pixels, meta)
    expected = _crossview_summary(out["landmarks"], out["polygons"], out["transform"])
    expected.update(chest_side=meta["chest_side"], lat=meta["laterality"], view=meta["view"],
                    W=int(pixels.shape[1]), H=int(pixels.shape[0]))
    assert record == expected
    assert set(record) == {"points", "conf", "polygons", "chest_side", "lat", "view", "W", "H"}
    assert record["polygons"]["nipple"]["rescued"] is True
    # original pixel coordinates: inside the image, not the 640 frame
    xs = [p[0] for p in record["polygons"]["breast-tissue"]["p"]]
    assert max(xs) <= record["W"] and max(xs) > TARGET_SIZE * 0.5


def test_landmark_record_skips_empty_polygons():
    transform = Transform(100, 100, 0, 0, 100, 100, 2.0, 0, 0, 0, 0, 0.1)
    record = landmark_record({"points": {"nipple": [20.0, 40.0]}, "confidence": 0.5},
                             {"nipple": {"points": [], "confidence": 0.3}}, transform)
    assert record == {"points": {"nipple": [10.0, 20.0]}, "conf": 0.5, "polygons": {}}


def test_largest_polygon():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1
    mask[30:45, 30:45] = 1
    polygon = largest_polygon(mask)
    assert min(p[0] for p in polygon) >= 30
    assert largest_polygon(np.zeros((5, 5), np.uint8)) == []


def test_importing_the_engine_does_not_import_ultralytics():
    code = ("import sys, breast_seg.inference, breast_seg.apps.positioning; "
            "print('ultralytics' in sys.modules, 'torch' in sys.modules)")
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert result.stdout.strip() == "False False"


# ── real weights (opt-in) ──────────────────────────────────────────────────
def _real_registry():
    root = os.environ.get(WEIGHTS_ENV, "").strip()
    if not root:
        pytest.skip(f"set {WEIGHTS_ENV} to the weights root to run against real weights")
    registry = Registry.load(MODELS_YAML, weights_root=root)
    needed = [registry.select("landmark", "MLO"), registry.select("landmark", "CC"), registry.select("segmentation")]
    if not all(spec is not None and spec.available for spec in needed):
        pytest.skip("weights not found under the weights root")
    return registry


@pytest.mark.weights
def test_real_weights_on_cpu():
    pytest.importorskip("ultralytics")
    engine = LandmarkEngine(_real_registry(), device="cpu", preview_size=640).load()
    assert engine.ready, engine.errors
    pixels, meta = _image("MLO", "L")
    out = engine.analyse(pixels, meta)
    assert out["display"].shape == (640, 640)
    assert set(out) >= {"landmarks", "polygons", "lesions", "quality"}
    record = engine.analyse_landmarks(pixels, meta)
    assert record["W"] == pixels.shape[1] and record["view"] == "MLO"
    baseline, _, _ = preprocess_pair(pixels, meta)
    assert baseline.shape == (TARGET_SIZE, TARGET_SIZE)
