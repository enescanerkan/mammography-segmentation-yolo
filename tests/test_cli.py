"""The command line parses, wires the application overrides and checks the model registry."""
from pathlib import Path

import pytest

from breast_seg import __version__
from breast_seg.cli import _app_config, build_parser, main


def test_help_and_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and __version__ in capsys.readouterr().out
    for cmd in ("serve", "models", "build-dataset", "train", "cv", "evaluate", "compare"):
        with pytest.raises(SystemExit) as e:
            main([cmd, "--help"])
        assert e.value.code == 0


def test_serve_flags_become_config_overrides(tmp_path):
    args = build_parser().parse_args(["serve", "--weights", str(tmp_path), "--device", "cuda:1",
                                      "--data-dir", str(tmp_path / "d"), "--set", "rules.threshold_mm=12", "--port", "8123"])
    cfg = _app_config(args)
    assert Path(cfg.models.weights_root) == tmp_path.resolve() and cfg.models.device == "cuda:1"
    assert Path(cfg.storage.data_dir) == (tmp_path / "d").resolve() and cfg.rules.threshold_mm == 12
    assert args.port == 8123


def test_models_reports_missing_weights(tmp_path, capsys):
    (tmp_path / "pose.pt").write_bytes(b"x")
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "models:\n"
        "  - {id: pose_mlo, task: landmark, view: MLO, recipe: wavelet640, weights: pose.pt}\n"
        "  - {id: seg, task: segmentation, recipe: baseline640, weights: seg.pt}\n"
        "  - {id: old, task: segmentation, recipe: baseline640, weights: old.pt, enabled: false}\n")
    code = main(["models", "--weights", str(tmp_path), "--set", f"models.registry={reg}"])
    out = capsys.readouterr().out
    assert code == 1 and "MISSING" in out and "pose_mlo" in out
    (tmp_path / "seg.pt").write_bytes(b"x")
    assert main(["models", "--weights", str(tmp_path), "--set", f"models.registry={reg}"]) == 0  # disabled ones don't count
