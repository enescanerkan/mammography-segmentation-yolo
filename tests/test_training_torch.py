"""PyTorch families on CPU with tiny synthetic data (a two-epoch loop on 32x32 images; no real training)."""
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from breast_seg.config import PathsConfig  # noqa: E402
from breast_seg.training import SplitSpec, build_trainer  # noqa: E402
from breast_seg.training.nets import AttUNet, UNet, dice_loss, soft_argmax, upgrade_state_dict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPU = ["device=cpu", "workers=0", "optim.val_workers=0", "optim.epochs=2", "optim.patience=5", "augment=[]",
       "optim.amp_dtype=none"]


def semantic_dataset(root: Path, n: int = 6, size: int = 32):
    rows = []
    for i in range(n):
        split = "train" if i < 4 else "val"
        stem = f"cc_c{i}" if i % 2 else f"m{i}"
        img = np.full((size, size), 60 + 10 * i, np.uint8)
        mask = np.zeros((size, size), np.uint8)
        mask[:, : size // 2] = 2
        mask[: size // 4, : size // 4] = 1 if i % 2 == 0 else 2
        mask[14:18, 12:16] = 3
        for sub, arr in (("images", img), ("masks", mask)):
            (root / sub / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / sub / split / f"{stem}.png"), arr)
        rows.append({"stem": stem, "split": split})
    pd.DataFrame(rows).to_csv(root / "index.csv", index=False)
    lm = []
    for r in rows:
        lm.append({"stem": r["stem"], "role": "nipple", "x": 14.0, "y": 16.0})
        if not r["stem"].startswith("cc_"):
            lm += [{"stem": r["stem"], "role": "pec_top", "x": 2.0, "y": 0.0},
                   {"stem": r["stem"], "role": "pec_bot", "x": 0.0, "y": 8.0}]
    pd.DataFrame(lm).to_csv(root / "landmarks_canvas.csv", index=False)
    return rows


def test_nets_shapes_and_soft_argmax():
    x = torch.zeros(2, 1, 32, 32)
    assert UNet(1, 4, 4)(x).shape == (2, 4, 32, 32)
    seg, heat = AttUNet(4, 3, 4)(x)
    assert seg.shape == (2, 4, 32, 32) and heat.shape == (2, 3, 32, 32)
    h = torch.full((1, 1, 11, 21), -50.0)
    h[0, 0, 5, 15] = 50.0
    xy = soft_argmax(h)[0, 0]
    assert float(xy[0]) == pytest.approx(15 / 20, abs=1e-3) and float(xy[1]) == pytest.approx(0.5, abs=1e-3)
    y = torch.zeros(1, 4, 4, dtype=torch.long)
    perfect = torch.nn.functional.one_hot(y, 4).permute(0, 3, 1, 2).float() * 100
    assert float(dice_loss(perfect, y, 4)) < 0.6        # foreground absent in both: eps-smoothed


def test_legacy_checkpoint_names_are_upgraded():
    legacy = {"inis.0.0.weight": 1, "dip.0.weight": 2, "yukselt.0.weight": 3, "cikis.0.0.weight": 4, "bas.weight": 5,
              "kapi.0.wg.weight": 6, "seg_bas.bias": 7, "lm_bas.bias": 8}
    new = upgrade_state_dict(legacy)
    assert set(new) == {"down.0.0.weight", "bottom.0.weight", "up.0.weight", "dec.0.0.weight", "head.weight",
                        "gates.0.wg.weight", "seg_head.bias", "lm_head.bias"}
    assert set(upgrade_state_dict(UNet(1, 4, 4).state_dict())) == set(UNet(1, 4, 4).state_dict())


def test_unet_loop_predict_evaluate_export(tmp_path):
    semantic_dataset(tmp_path / "data/semantic_v4")
    t = build_trainer(ROOT / "configs/train/unet_plain.yaml", PathsConfig(root=str(tmp_path)),
                      [*CPU, "base_channels=4", "optim.batch=2"])
    res = t.fit()
    assert res.weights.is_file() and (t.run_dir / "last.pt").is_file()
    hist = pd.read_csv(t.run_dir / "history.csv")
    assert list(hist.epoch) == [1, 2] and {"mean_fg", "pectoral", "nipple"} <= set(hist.columns)
    out = t.predict([tmp_path / "data/semantic_v4/images/val/m4.png"], tmp_path / "pred")
    pred = cv2.imread(str(out[0]), cv2.IMREAD_GRAYSCALE)
    assert pred.shape == (32, 32) and set(np.unique(pred)) <= {0, 1, 2, 3}
    report = t.evaluate("val")
    assert set(report["dice"]) == {"MLO", "CC"}
    assert t.export().name == "unet_plain.pt"


def test_attunet_multitask_loop_and_landmarks(tmp_path):
    semantic_dataset(tmp_path / "data/semantic_v4")
    t = build_trainer(ROOT / "configs/train/attunet_multitask.yaml", PathsConfig(root=str(tmp_path)),
                      [*CPU, "base_channels=4", "optim.batch=2"])
    t.fit()
    hist = pd.read_csv(t.run_dir / "history.csv")
    assert "lm_px" in hist and hist.lm_px.notna().all()
    out = t.predict([tmp_path / "data/semantic_v4/images/val/m4.png"], tmp_path / "pred")
    lm = pd.read_csv(tmp_path / "pred/landmarks.csv")
    assert list(lm.role) == ["nipple", "pec_top", "pec_bot"] and lm.x.between(0, 32).all()
    assert any(p.suffix == ".png" for p in out)                   # multitask arm writes masks
    ev = t.evaluate("val")
    assert set(ev["landmark_px"]) <= {"nipple", "pec_top", "pec_bot"} and "dice" in ev


def test_semantic_use_split_writes_index(tmp_path):
    semantic_dataset(tmp_path / "src")
    t = build_trainer(ROOT / "configs/train/unet_plain.yaml", PathsConfig(root=str(tmp_path)), CPU)
    imgs = {"m0": tmp_path / "src/images/train/m0.png", "m4": tmp_path / "src/images/val/m4.png"}
    masks = {"m0": tmp_path / "src/masks/train/m0.png"}
    ft = t.use_split(SplitSpec({"m0": "train", "m4": "test"}, imgs, {}, masks), tmp_path / "fold")
    idx = pd.read_csv(tmp_path / "fold/index.csv", keep_default_na=False)
    assert list(idx.split) == ["train", "test"] and idx["mask"].iloc[1] == ""
    assert ft.rows("test").image.iloc[0].endswith("m4.png")
    with pytest.raises(ValueError):
        t.use_split(SplitSpec({"m4": "val"}, imgs, {}, masks), tmp_path / "fold2")


class TinySegformer(torch.nn.Module):
    """Stands in for ``SegformerForSemanticSegmentation``: logits at 1/4 resolution."""

    def __init__(self, num_labels=4):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_labels, 4, stride=4)

    def forward(self, pixel_values):
        return types.SimpleNamespace(logits=self.conv(pixel_values))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "weights.pt")

    @classmethod
    def from_pretrained(cls, name, num_labels=4, ignore_mismatched_sizes=False):
        m = cls(num_labels)
        f = Path(str(name)) / "weights.pt"
        if f.is_file():
            m.load_state_dict(torch.load(f, weights_only=True))
        return m


def test_segformer_loop_with_fake_transformers(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(SegformerForSemanticSegmentation=TinySegformer))
    semantic_dataset(tmp_path / "data/semantic_v4")
    t = build_trainer(ROOT / "configs/train/segformer_b3.yaml", PathsConfig(root=str(tmp_path)), [*CPU, "optim.batch=2"])
    res = t.fit()
    assert res.weights == t.run_dir / "best" and (res.weights / "weights.pt").is_file()
    out = t.predict([tmp_path / "data/semantic_v4/images/val/m4.png"], tmp_path / "pred")
    assert cv2.imread(str(out[0]), cv2.IMREAD_GRAYSCALE).shape == (32, 32)
    assert t.export().is_dir()
