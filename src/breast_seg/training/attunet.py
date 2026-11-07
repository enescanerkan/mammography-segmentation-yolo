"""``attunet`` family: Attention U-Net + CoordConv with a soft-argmax landmark head.

One network, two heads, two arms of the same recipe:

* ``w_seg = 0`` (landmark): only the landmark regression is trained (the segmentation head receives no gradient
  and its masks are meaningless);
* ``w_seg = 1`` (multitask): the segmentation head is trained as well, through the shared encoder/decoder.

The only difference between the arms is the segmentation loss: same split, augmentation, schedule, seed and the
same early-stopping criterion (landmark error), so the comparison isolates what the segmentation signal adds.

Landmarks: MLO ``nipple, pec_top, pec_bot``; CC ``nipple`` only. Missing points are masked out of the loss.
The landmark loss is the masked L1 distance of normalised coordinates; the monitored ``lm_px`` is the mean absolute
per-axis deviation in canvas pixels (``mean(|dx| + |dy|) / 2 * 640``), not a Euclidean distance.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from breast_seg.evaluation.gt import ROLE_ALIASES
from breast_seg.training.augment import build_transforms
from breast_seg.training.base import register_trainer
from breast_seg.training.common import dice_report, stems_of
from breast_seg.training.config import AttUNetTrainConfig
from breast_seg.training.semantic import TorchSemanticTrainer


@register_trainer("attunet")
class AttUNetTrainer(TorchSemanticTrainer):
    config_class = AttUNetTrainConfig
    config: AttUNetTrainConfig
    monitor = "lm_px"
    monitor_mode = "min"

    def landmarks(self) -> Dict[str, Dict[str, np.ndarray]]:
        from breast_seg.training.torch_data import landmark_lookup

        path = self.resolve(self.config.landmarks_csv)
        return landmark_lookup(pd.read_csv(path, dtype={"stem": str}), ROLE_ALIASES)

    def prepare(self) -> Path:
        out = super().prepare()
        if not self.resolve(self.config.landmarks_csv).is_file():
            raise FileNotFoundError(f"{self.config.name}: landmarks_csv not found")
        return out

    def to_input(self):
        from breast_seg.training.torch_data import grey01

        return grey01

    def build_model(self) -> Any:
        from breast_seg.training.nets import AttUNet

        return AttUNet(len(self.config.data.classes), len(self.config.keypoints), self.config.base_channels)

    def transform(self, train: bool):
        if not train or not self.config.augment:
            return None
        import albumentations as A

        return A.Compose(build_transforms(self.config.augment),
                         keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    def make_dataset(self, rows: pd.DataFrame, train: bool):
        from breast_seg.training.torch_data import LandmarkDataset

        ds = LandmarkDataset(rows, self.landmarks(), self.config.keypoints, self.to_input(), self.transform(train))
        if ds.dropped:
            print(f"  {ds.dropped} image(s) without landmarks left out", flush=True)
        return ds

    def forward(self, model: Any, x: Any) -> Any:
        return model(x)[0]

    def compute_loss(self, model: Any, batch: Sequence[Any]) -> Any:
        import torch.nn.functional as F

        from breast_seg.training.nets import dice_loss, soft_argmax

        x, y, k, valid = batch
        seg, heat = model(x)
        coords = soft_argmax(heat.float())
        lm = ((coords - k).abs().sum(-1) * valid).sum() / valid.sum().clamp(min=1)
        loss = self.config.w_lm * lm
        if self.config.w_seg > 0:
            n = len(self.config.data.classes)
            loss = loss + self.config.w_seg * (F.cross_entropy(seg.float(), y) + dice_loss(seg.float(), y, n))
        return loss

    def validate(self, model: Any, loader: Any, device: Any) -> Dict[str, float]:
        import torch

        from breast_seg.evaluation.masks import PooledDice
        from breast_seg.training.nets import soft_argmax

        acc = PooledDice(len(self.config.data.classes))
        err, count = 0.0, 0.0
        model.eval()
        with torch.no_grad():
            for x, y, k, valid in loader:
                x, k, valid = x.to(device), k.to(device), valid.to(device)
                with self.autocast(device):
                    seg, heat = model(x)
                acc.update(seg.float().argmax(1).cpu().numpy(), y.numpy())
                coords = soft_argmax(heat.float())
                err += float(((coords - k).abs().sum(-1) * valid).sum().item())
                count += float(valid.sum().item())
        dice = acc.result(dict(enumerate(self.config.data.classes)))
        size = 640.0
        return {"mean_fg": dice["mean_fg"], "nipple": dice.get("nipple", float("nan")),
                "pectoral": dice.get("pectoral", float("nan")), "lm_px": err / max(count, 1.0) * size / 2}

    def save(self, model: Any, path: Path) -> None:
        import torch

        torch.save({"model": model.state_dict(), "base_channels": self.config.base_channels,
                    "num_classes": len(self.config.data.classes), "keypoints": list(self.config.keypoints),
                    "w_seg": self.config.w_seg}, path)

    def load(self, weights: Optional[Path] = None) -> Any:

        from breast_seg.training.nets import AttUNet, load_checkpoint, upgrade_state_dict

        ck = load_checkpoint(weights or self.best_weights())
        n_kp = len(ck["keypoints"]) if "keypoints" in ck else ck.get("nokta", len(self.config.keypoints))
        model = AttUNet(ck.get("num_classes", ck.get("sinif", 4)), n_kp, ck.get("base_channels", ck.get("taban", 64)))
        model.load_state_dict(upgrade_state_dict(ck["model"]))
        return model

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None,
                write_masks: Optional[bool] = None) -> List[Path]:
        """Label maps (unless ``write_masks`` is false; default: only when ``w_seg > 0``) and ``landmarks.csv``
        (``stem, role, x, y, conf`` in canvas pixels; ``conf`` = peak of the heatmap softmax). With ``flip_tta`` the
        flipped heatmap probabilities are mirrored back and averaged before the expectation is taken."""
        import torch

        from breast_seg.evaluation.predictions import write_label_map
        from breast_seg.training.nets import expectation
        from breast_seg.training.torch_data import read_grey, stack_inputs

        write_masks = self.config.w_seg > 0 if write_masks is None else write_masks
        device = self.device()
        model = self.load(weights).to(device).eval()
        batch = self.config.optim.val_batch or self.config.optim.batch
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written, rows = [], []
        images = [Path(i) for i in images]
        with torch.no_grad():
            for i in range(0, len(images), batch):
                chunk = images[i:i + batch]
                imgs = [read_grey(str(p)) for p in chunk]
                x = stack_inputs(imgs, self.to_input()).to(device)
                with self.autocast(device):
                    seg, heat = model(x)
                seg_p = seg.float().softmax(1)
                b, k, h, w = heat.shape
                heat_p = heat.float().flatten(2).softmax(-1).view(b, k, h, w)
                if self.config.flip_tta:
                    with self.autocast(device):
                        seg2, heat2 = model(x.flip(-1))
                    seg_p = (seg_p + seg2.float().flip(-1).softmax(1)) / 2
                    heat_p = (heat_p + heat2.float().flatten(2).softmax(-1).view(b, k, h, w).flip(-1)) / 2
                coords = expectation(heat_p).cpu().numpy()
                conf = heat_p.flatten(2).max(-1).values.cpu().numpy()
                pred = seg_p.argmax(1).to(torch.uint8).cpu().numpy()
                for j, p in enumerate(chunk):
                    if write_masks:
                        written.append(write_label_map(out_dir / f"{p.stem}.png", pred[j]))
                    ih, iw = imgs[j].shape[:2]
                    for r, role in enumerate(self.config.keypoints):
                        rows.append({"stem": p.stem, "role": role, "x": float(coords[j, r, 0]) * iw,
                                     "y": float(coords[j, r, 1]) * ih, "conf": float(conf[j, r])})
        target = out_dir / "landmarks.csv"
        with open(target, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=["stem", "role", "x", "y", "conf"])
            wr.writeheader()
            wr.writerows(rows)
        return written + [target]

    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        """Mean Euclidean landmark error per role (canvas pixels) and, for the multitask arm, Dice by view."""
        from breast_seg.training.torch_data import read_grey

        rows = self.rows(split)
        out = self.run_dir / "predictions" / split
        self.predict([Path(p) for p in rows.image], out, weights, write_masks=self.config.w_seg > 0)
        pred = pd.read_csv(out / "landmarks.csv", dtype={"stem": str})
        ref = self.landmarks()
        errors: Dict[str, List[float]] = {}
        for r in pred.itertuples(index=False):
            g = ref.get(r.stem, {}).get(r.role)
            if g is not None:
                errors.setdefault(r.role, []).append(float(np.hypot(r.x - g[0], r.y - g[1])))
        result: Dict[str, Any] = {"landmark_px": {k: float(np.mean(v)) for k, v in errors.items()}}
        if self.config.w_seg > 0:
            masks = dict(zip(rows.stem, rows["mask"]))
            result["dice"] = dice_report(out, stems_of(rows.image), lambda s: read_grey(masks[s]) if masks.get(s) else None)
        return result
