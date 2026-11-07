"""Config machinery, every committed training / CV YAML, and the trainer registry."""
from pathlib import Path

import pytest
import yaml

from breast_seg.config import ConfigError, PathsConfig, apply_overrides, parse_override, read_yaml
from breast_seg.training import (
    AttUNetTrainConfig,
    NnUNetTrainConfig,
    SegFormerTrainConfig,
    UNetTrainConfig,
    YoloTrainConfig,
    available_families,
    build_trainer,
    get_trainer_class,
    load_cv_config,
    load_train_config,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = sorted(p for p in (ROOT / "configs/train").glob("*.yaml") if not p.name.startswith("_"))
FRAGMENTS = sorted((ROOT / "configs/train").glob("_*.yaml"))
CV = sorted(p for p in (ROOT / "configs/cv").glob("*.yaml") if not p.name.startswith("_"))
EXPECTED = {"yolo_seg": YoloTrainConfig, "yolo_pose": YoloTrainConfig, "yolo_detect": YoloTrainConfig,
            "segformer": SegFormerTrainConfig, "unet": UNetTrainConfig, "attunet": AttUNetTrainConfig,
            "nnunet": NnUNetTrainConfig}


def test_registry_lists_every_family():
    assert set(available_families()) == set(EXPECTED)
    for family, cfg_cls in EXPECTED.items():
        assert get_trainer_class(family).config_class is cfg_cls
    with pytest.raises(ConfigError, match="unknown family"):
        get_trainer_class("nope")


@pytest.mark.parametrize("path", TRAIN, ids=lambda p: p.name)
def test_every_train_config_loads_and_builds(path, tmp_path):
    cfg = load_train_config(path)
    assert isinstance(cfg, EXPECTED[cfg.family]) and cfg.name and cfg.export.filename
    trainer = build_trainer(cfg, PathsConfig(root=str(tmp_path)))
    assert trainer.run_dir == tmp_path / "runs" / cfg.name
    dataset_dir = getattr(getattr(cfg, "data", None), "dataset_dir", "")
    assert dataset_dir.startswith("${")                      # no machine path in a committed config


def test_names_are_unique_and_fragments_are_used():
    names = [load_train_config(p).name for p in TRAIN]
    assert len(names) == len(set(names))
    used = {str(read_yaml_raw(p).get("base")) for p in TRAIN + FRAGMENTS}
    for frag in FRAGMENTS:
        assert any(frag.name in u for u in used), frag.name


def read_yaml_raw(path):
    return yaml.safe_load(path.read_text()) or {}


def test_v4_recipe_values():
    c = load_train_config(ROOT / "configs/train/yolo26m_seg_v4.yaml")
    h = c.hyp
    assert (c.model, h.epochs, h.patience, h.batch, h.imgsz) == ("yolo26m-seg.pt", 150, 30, 16, 640)
    assert (h.optimizer, h.lr0, h.amp) == ("AdamW", 0.001, True)
    assert (h.fliplr, h.flipud, h.degrees, h.translate) == (0.5, 0.0, 8.0, 0.05)
    assert (h.scale, h.mosaic, h.close_mosaic) == (0.2, 0.5, 15)
    assert (h.hsv_h, h.hsv_s, h.hsv_v, h.save_period, c.seed) == (0.0, 0.0, 0.3, -1, 42)
    names = [t["name"] for t in c.albumentations.transforms]
    assert c.albumentations.enabled and names[:2] == ["CLAHE", "RandomBrightnessContrast"] and len(names) == 10
    v5 = load_train_config(ROOT / "configs/train/yolo26m_seg_v5.yaml")
    assert (v5.hyp.epochs, v5.hyp.patience) == (300, 50) and v5.data.dataset_dir.endswith("seg-dataset_v5")
    v8 = load_train_config(ROOT / "configs/train/yolov8l_seg.yaml")
    assert v8.model == "yolov8l-seg.pt" and v8.hyp.epochs == 300 and v8.albumentations.enabled
    pose = load_train_config(ROOT / "configs/train/pose_mlo_yolo26l.yaml")
    assert pose.data.kpt_shape == [3, 3] and pose.hyp.batch == 26 and pose.hyp.fliplr == 0.0
    nn = load_train_config(ROOT / "configs/train/nnunet_resenc_m.yaml")
    assert (nn.patch_size, nn.batch_size, nn.trainer) == ([640, 640], 12, "nnUNetTrainerMammo")


def test_inheritance_overrides_and_validation(tmp_path):
    (tmp_path / "a.yaml").write_text("family: unet\nname: a\noptim: {epochs: 5, lr: 0.1}\nseed: 1\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nname: b\noptim: {lr: 0.2}\n")
    c = load_train_config(tmp_path / "b.yaml", ["optim.batch=4", "device=cpu"])
    assert (c.name, c.seed, c.optim.epochs, c.optim.lr, c.optim.batch, c.device) == ("b", 1, 5, 0.2, 4, "cpu")
    with pytest.raises(ConfigError, match="unknown key"):
        load_train_config(tmp_path / "b.yaml", ["optim.lr_typo=1"])
    with pytest.raises(ConfigError, match="expected an integer"):
        load_train_config(tmp_path / "b.yaml", ["optim.epochs=many"])
    (tmp_path / "c.yaml").write_text("base: d.yaml\n")
    (tmp_path / "d.yaml").write_text("base: c.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        read_yaml(tmp_path / "c.yaml")
    assert parse_override("a.b=[1, 2]") == ("a.b", [1, 2])
    assert apply_overrides({"a": {"b": 1}}, {"a.c": None}) == {"a": {"b": 1, "c": None}}
    with pytest.raises(ConfigError):
        parse_override("novalue")


def test_paths_placeholders(tmp_path):
    f = tmp_path / "cfg" / "paths.yaml"
    f.parent.mkdir()
    f.write_text("root: ..\ndata_root: d\nextra: {pool: '${data_root}/pool'}\n")
    p = PathsConfig.load(f)
    assert p.resolve("${pool}/x") == tmp_path / "d/pool/x"
    assert p.dir("runs_dir") == tmp_path / "runs"
    with pytest.raises(ConfigError, match="unknown placeholder"):
        p.resolve("${nope}")
    f.write_text("extra: {runs_dir: x}\n")
    with pytest.raises(ConfigError):
        PathsConfig.load(f)
    example = PathsConfig.load(ROOT / "configs/paths.example.yaml")
    assert str(example.resolve("${mlo_labels}")).startswith(str(ROOT))


@pytest.mark.parametrize("path", CV, ids=lambda p: p.name)
def test_every_cv_config_loads(path):
    cv = load_cv_config(path)
    train = (path.parent / cv.train_config).resolve()
    cfg = load_train_config(train, cv.overrides)
    assert cv.n_folds == 10 and cv.val_offset == 1 and cv.evaluation is not None
    assert cfg.family in ("yolo_seg", "segformer", "nnunet")
    if cfg.family != "nnunet":
        assert (cfg.hyp.patience if cfg.family == "yolo_seg" else cfg.optim.patience) == 300


def test_trainer_rejects_wrong_config_type(tmp_path):
    cfg = load_train_config(ROOT / "configs/train/unet_plain.yaml")
    with pytest.raises(TypeError):
        get_trainer_class("yolo_seg")(cfg, PathsConfig(root=str(tmp_path)))
