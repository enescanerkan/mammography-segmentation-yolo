"""toolkit_config: defaults, YAML path rules, environment overrides."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import toolkit_config
from toolkit_config import TOOLKIT_ROOT, load_config


def test_defaults_are_relative_to_the_toolkit():
    cfg = load_config(environ={})  # no $ANNOTATION_CONFIG -> defaults (unless a local annotation.yaml exists)
    if cfg.source is not None:  # a developer's local annotation.yaml would be picked up
        pytest.skip("local annotation.yaml present")
    assert cfg.weights_root == TOOLKIT_ROOT / "weights"
    assert cfg.medsam2_repo == TOOLKIT_ROOT / "vendor" / "MedSAM"
    assert cfg.medsam2_checkpoint == TOOLKIT_ROOT / "weights" / "medsam2" / "medsam_model_best.pth"
    assert cfg.sam2_base_weights == TOOLKIT_ROOT / "weights" / "medsam2" / "sam2_hiera_tiny.pt"
    assert cfg.detector_weights == TOOLKIT_ROOT / "weights" / "detector" / "manual_detector.pt"
    assert cfg.positioning_mlo_weights.name == "mlo-yolo26-pose-advanced.pt"
    assert cfg.device == "auto"


def test_yaml_paths_resolve_against_file_and_weights_root(tmp_path):
    cfg_file = tmp_path / "conf" / "annotation.yaml"
    cfg_file.parent.mkdir()
    cfg_file.write_text(
        "weights_root: ../models\n"
        "medsam2:\n"
        "  checkpoint: runs/best.pth\n"
        "  repo: /opt/MedSAM\n"
        "positioning:\n"
        "  results_dir: out\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file, environ={})
    assert cfg.source == cfg_file.resolve()
    assert cfg.weights_root == (tmp_path / "models").resolve()
    assert cfg.medsam2_checkpoint == (tmp_path / "models" / "runs" / "best.pth").resolve()
    assert cfg.medsam2_repo == Path("/opt/MedSAM")
    assert cfg.positioning_results_dir == (tmp_path / "conf" / "out").resolve()
    # untouched keys keep their defaults, relative to the new weights_root
    assert cfg.detector_weights == (tmp_path / "models" / "detector" / "manual_detector.pt").resolve()


def test_environment_overrides_yaml(tmp_path):
    cfg_file = tmp_path / "annotation.yaml"
    cfg_file.write_text("medsam2:\n  checkpoint: from_yaml.pth\n  device: cpu\n", encoding="utf-8")
    env = {"MEDSAM2_CHECKPOINT": str(tmp_path / "from_env.pth"), "MEDSAM2_DEVICE": "cuda:1"}
    cfg = load_config(cfg_file, environ=env)
    assert cfg.medsam2_checkpoint == (tmp_path / "from_env.pth").resolve()
    assert cfg.device == "cuda:1"


def test_legacy_checkpoint_variable_is_accepted(tmp_path):
    env = {"MEDSAM2_INTERACTIVE_CKPT": str(tmp_path / "legacy.pth")}
    cfg = load_config(environ=env)
    assert cfg.medsam2_checkpoint == (tmp_path / "legacy.pth").resolve()


def test_unknown_key_warns(tmp_path, caplog):
    cfg_file = tmp_path / "annotation.yaml"
    cfg_file.write_text("medsam2:\n  chekpoint: typo.pth\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="toolkit_config"):
        load_config(cfg_file, environ={})
    assert "medsam2.chekpoint" in caplog.text


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml", environ={})


def test_missing_files_are_reported(isolated_config):
    cfg = toolkit_config.get_config()
    problems = cfg.missing_medsam2_files()
    assert len(problems) == 3
    assert any("medsam2.checkpoint" in p for p in problems)
    assert len(cfg.missing_positioning_files()) == 2


def test_resolve_device_passthrough():
    assert toolkit_config.resolve_device("cpu") == "cpu"
    assert toolkit_config.resolve_device("cuda:1") == "cuda:1"
    assert toolkit_config.resolve_device("auto") in ("cpu", "cuda")
