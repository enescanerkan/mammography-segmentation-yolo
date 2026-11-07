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


def _registry(tmp_path, with_detector=False):
    for rel in ("positioning/mlo.pt", "positioning/cc.pt", "segmentation/seg.pt"):
        (tmp_path / "w" / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "w" / rel).write_bytes(b"x")
    (tmp_path / "configs").mkdir(exist_ok=True)
    (tmp_path / "configs" / "models.yaml").write_text(
        "models:\n"
        "  - {id: pose_mlo, task: landmark, view: MLO, recipe: wavelet640, weights: positioning/mlo.pt}\n"
        "  - {id: pose_cc, task: landmark, view: CC, recipe: wavelet640, weights: positioning/cc.pt}\n"
        "  - {id: seg, task: segmentation, recipe: baseline640, weights: segmentation/seg.pt}\n"
        "  - {id: det, task: detection, recipe: baseline640, weights: detector/det.pt}\n")
    (tmp_path / "configs" / "app.yaml").write_text("models:\n  registry: models.yaml\n  weights_root: ../w\nstorage:\n")
    return tmp_path / "configs" / "app.yaml"


def test_models_resolves_like_serve_and_needs_only_the_required_roles(tmp_path, monkeypatch, capsys):
    app = _registry(tmp_path)
    monkeypatch.chdir(tmp_path / "w")  # relative paths must follow the config file, not the working directory
    assert main(["models", "--config", str(app)]) == 0  # the optional detector is missing, the app can still run
    assert "MISSING" in capsys.readouterr().out
    (tmp_path / "w" / "segmentation" / "seg.pt").unlink()
    assert main(["models", "--config", str(app)]) == 1
    assert "segmentation: seg has no weights file" in capsys.readouterr().err


def test_flags_are_typed_values_and_empty_sections_are_fine(tmp_path):
    app = _registry(tmp_path)  # `storage:` is an empty section
    odd = tmp_path / "positioning data #2: copy"
    args = build_parser().parse_args(["serve", "--config", str(app), "--data-dir", str(odd), "--device", "0",
                                      "--set", "title=Review: Clinic A"])
    cfg = _app_config(args)
    assert Path(cfg.storage.data_dir) == odd.resolve() and cfg.models.device == "0"
    assert cfg.title == "Review: Clinic A"


def test_config_errors_are_one_line(tmp_path, capsys):
    app = _registry(tmp_path)
    assert main(["models", "--config", str(app), "--set", "storage.nope=1"]) == 2
    assert "error:" in capsys.readouterr().err


def test_compare_example_config_is_valid():
    from breast_seg.cli import _section
    from breast_seg.compare import CompareConfig
    from breast_seg.config import from_dict

    root = Path(__file__).resolve().parents[1]
    cfg = from_dict(CompareConfig, _section(str(root / "configs" / "compare.example.yaml"), "compare", ["limit=5"]), "compare")
    assert cfg.limit == 5 and cfg.wavelet
