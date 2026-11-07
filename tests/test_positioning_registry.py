"""Model registry: parsing, validation, weights-root resolution, role selection."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from breast_seg.registry import WEIGHTS_ENV, NippleRescue, Registry, RegistryError, resolve_weights_root

REPO = Path(__file__).resolve().parents[1]
MODELS_YAML = REPO / "configs" / "models.yaml"


def _registry(models, **top):
    return Registry.from_dict({"models": models, **top}, base_dir="/base")


POSE = {"id": "pose_mlo", "task": "landmark", "view": "MLO", "recipe": "wavelet640", "weights": "p/mlo.pt",
        "keypoints": ["nipple", "pectoral_top", "pectoral_bottom"]}


def test_committed_registry_parses(monkeypatch, tmp_path):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    registry = Registry.load(MODELS_YAML)
    ids = [m.id for m in registry.models]
    assert ids == ["pose_mlo", "pose_cc", "seg_v4", "seg_v5", "lesion_det", "medsam2"]
    assert registry.weights_root == REPO / "weights"  # weights_root: ../weights relative to configs/
    assert registry.select("landmark", "MLO").id == "pose_mlo"
    assert registry.select("landmark", "CC").id == "pose_cc"
    assert registry.select("segmentation").id == "seg_v4"  # seg_v5 is declared but disabled
    assert registry.select("detection").id == "lesion_det"
    assert registry.select("prompt_segmentation") is None  # medsam2 disabled
    seg = registry.get("seg_v4")
    assert seg.classes == ("pectoral", "breast-tissue", "nipple") and seg.nipple_rescue.enabled
    assert seg.nipple_rescue == NippleRescue(enabled=True, conf=0.02, max_mm=15.0)
    assert (seg.conf, seg.iou) == (0.25, 0.5)


def test_committed_configs_hold_no_machine_paths():
    for path in (MODELS_YAML, REPO / "configs" / "app.example.yaml"):
        text = path.read_text(encoding="utf-8")
        for value in _scalars(yaml.safe_load(text)):
            assert not (isinstance(value, str) and (value.startswith("/") or value[1:3] == ":\\")), value


def _scalars(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _scalars(value)
    elif isinstance(node, list):
        for value in node:
            yield from _scalars(value)
    else:
        yield node


def test_weights_root_precedence(monkeypatch, tmp_path):
    base = tmp_path / "configs"
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    assert resolve_weights_root(None, base) == base
    assert resolve_weights_root("../w", base) == (tmp_path / "w").resolve()
    monkeypatch.setenv(WEIGHTS_ENV, str(tmp_path / "env"))
    assert resolve_weights_root("../w", base) == tmp_path / "env"
    assert resolve_weights_root("../w", base, explicit=tmp_path / "cli") == tmp_path / "cli"


def test_weights_are_resolved_under_the_root(monkeypatch, tmp_path):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "mlo.pt").write_bytes(b"")
    registry = Registry.from_dict({"models": [POSE]}, weights_root=tmp_path)
    spec = registry.get("pose_mlo")
    assert spec.weights_path == tmp_path / "p" / "mlo.pt" and spec.available
    public = spec.to_dict()
    assert public["weights"] == "p/mlo.pt" and public["available"] and str(tmp_path) not in str(public)


def test_legacy_keys_are_accepted(monkeypatch):
    monkeypatch.delenv(WEIGHTS_ENV, raising=False)
    registry = _registry([POSE], root="/models", rules={"threshold_mm": 10}, preview={"size": 640})
    assert registry.weights_root == Path("/models")


@pytest.mark.parametrize("change,message", [
    ({"task": "pose"}, "task"),
    ({"view": "ML"}, "view"),
    ({"recipe": "raw640"}, "recipe"),
    ({"weights": "/abs/model.pt"}, "relative"),
    ({"colour": "red"}, "unknown keys"),
    ({"nipple_rescue": {"enabled": True, "radius": 3}}, "nipple_rescue"),
])
def test_invalid_entries(change, message):
    with pytest.raises(RegistryError, match=message):
        _registry([{**POSE, **change}])


def test_missing_field_and_duplicates():
    with pytest.raises(RegistryError, match="missing 'weights'"):
        _registry([{k: v for k, v in POSE.items() if k != "weights"}])
    with pytest.raises(RegistryError, match="duplicate"):
        _registry([POSE, POSE])
    with pytest.raises(RegistryError, match="unknown keys"):
        _registry([POSE], extra=1)


def test_select_respects_enabled_flags_and_views():
    seg = {"task": "segmentation", "recipe": "baseline640", "weights": "s.pt"}
    registry = _registry([
        {**POSE, "enabled": False},
        {**POSE, "id": "pose_any", "view": "both"},
        {**seg, "id": "seg_a", "enabled": False},
        {**seg, "id": "seg_b"},
    ])
    assert registry.select("landmark", "MLO").id == "pose_any"
    assert registry.select("landmark", "CC").id == "pose_any"
    assert registry.select("segmentation").id == "seg_b"
    assert [m.id for m in registry.active()] == ["pose_any", "seg_b"]
    assert registry.get("nope") is None
