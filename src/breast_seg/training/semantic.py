"""PyTorch semantic-segmentation families: ``segformer`` (pre-trained MiT encoder, transformers) and ``unet``
(plain U-Net from scratch). Both share one training loop: AdamW, polynomial learning-rate decay per iteration,
cross-entropy + soft Dice, bf16 autocast, validation every epoch with pooled per-class Dice, early stopping on the
mean foreground Dice, ``best`` and ``last`` checkpoints, horizontal-flip test-time augmentation at prediction."""
from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

import pandas as pd

from breast_seg.datasets.formats import semantic_paths
from breast_seg.training.augment import build_transforms, validate_specs
from breast_seg.training.base import SplitSpec, Trainer, TrainResult, register_trainer
from breast_seg.training.common import dice_report, stems_of, torch_device
from breast_seg.training.config import SegFormerTrainConfig, TorchTrainConfig, UNetTrainConfig


class TorchSemanticTrainer(Trainer):
    """Template of the PyTorch families; subclasses provide the network, its input encoding and checkpoints."""

    config_class = TorchTrainConfig
    config: TorchTrainConfig
    monitor: ClassVar[str] = "mean_fg"
    monitor_mode: ClassVar[str] = "max"
    weights_suffix: ClassVar[str] = ".pt"

    # ── data ───────────────────────────────────────────────────────────────────────────────────────────────

    @property
    def dataset_dir(self) -> Path:
        return self.resolve(self.config.data.dataset_dir)

    @property
    def index_path(self) -> Path:
        return self.dataset_dir / self.config.data.index

    def index(self) -> pd.DataFrame:
        if not self.index_path.is_file():
            raise FileNotFoundError(f"{self.config.name}: no {self.config.data.index} under data.dataset_dir")
        return semantic_paths(self.dataset_dir, pd.read_csv(self.index_path, dtype={"stem": str}))

    def rows(self, split: str) -> pd.DataFrame:
        df = self.index()
        return df[df.split == split].reset_index(drop=True)

    def prepare(self) -> Path:
        d = self.config.data
        df = self.index()
        for split in (d.train_split, d.val_split):
            if not (df.split == split).any():
                raise ValueError(f"{self.config.name}: split {split!r} is empty in {d.index}")
        validate_specs(self.config.augment)
        self.save_config()
        return self.index_path

    def use_split(self, spec: SplitSpec, work_dir: Path) -> TorchSemanticTrainer:
        """Write ``work_dir/index.csv`` with absolute image / mask paths of the fold and point the config at it."""
        rows = []
        for stem, split in sorted(spec.split_of.items()):
            if split in ("train", "val") and stem not in spec.masks:
                raise ValueError(f"{stem}: unlabelled image in {split}")
            rows.append({"stem": stem, "split": split, "image": str(spec.images[stem]), "mask": str(spec.masks.get(stem, ""))})
        work_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["stem", "split", "image", "mask"]).to_csv(work_dir / "index.csv", index=False)
        data = dataclasses.replace(self.config.data, dataset_dir=str(work_dir), index="index.csv")
        return type(self)(dataclasses.replace(self.config, data=data), self.paths)

    # ── hooks ──────────────────────────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def build_model(self) -> Any:
        """A freshly initialised network."""

    @abstractmethod
    def to_input(self):
        """``(H, W) uint8`` -> network input array (see :mod:`breast_seg.training.torch_data`)."""

    @abstractmethod
    def save(self, model: Any, path: Path) -> None:
        """Write a checkpoint."""

    @abstractmethod
    def load(self, weights: Optional[Path] = None) -> Any:
        """Network with the weights of ``weights`` (default :meth:`best_weights`)."""

    def forward(self, model: Any, x: Any) -> Any:
        """Segmentation logits at input resolution."""
        return model(x)

    def transform(self, train: bool):
        if not train or not self.config.augment:
            return None
        import albumentations as A

        return A.Compose(build_transforms(self.config.augment))

    def make_dataset(self, rows: pd.DataFrame, train: bool):
        from breast_seg.training.torch_data import SemanticDataset

        return SemanticDataset(rows, self.to_input(), self.transform(train))

    def compute_loss(self, model: Any, batch: Sequence[Any]) -> Any:
        import torch.nn.functional as F

        from breast_seg.training.nets import dice_loss

        x, y = batch[0], batch[1]
        logits = self.forward(model, x)
        return F.cross_entropy(logits, y) + dice_loss(logits.float(), y, len(self.config.data.classes))

    def validate(self, model: Any, loader: Any, device: Any) -> Dict[str, float]:
        import torch

        from breast_seg.evaluation.masks import PooledDice

        acc = PooledDice(len(self.config.data.classes))
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(device, non_blocking=True), batch[1]
                with self.autocast(device):
                    pred = self.forward(model, x).argmax(1)
                acc.update(pred.cpu().numpy(), y.numpy())
        return acc.result(dict(enumerate(self.config.data.classes)))

    # ── loop ───────────────────────────────────────────────────────────────────────────────────────────────

    def autocast(self, device: Any):
        import torch

        dt = self.config.optim.amp_dtype
        if dt == "none":
            return contextlib.nullcontext()
        return torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=getattr(torch, dt))

    def device(self):
        import torch

        return torch.device(torch_device(self.config.device))

    def best_weights(self) -> Path:
        return self.run_dir / f"{self.config.export.checkpoint}{self.weights_suffix}"

    def fit(self) -> TrainResult:
        self.prepare()
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        c, o = self.config, self.config.optim
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        device = self.device()
        run = self.run_dir
        train_ds = self.make_dataset(self.rows(c.data.train_split), train=True)
        val_ds = self.make_dataset(self.rows(c.data.val_split), train=False)
        tr = DataLoader(train_ds, batch_size=o.batch, shuffle=True, num_workers=c.workers, pin_memory=True,
                        drop_last=True, persistent_workers=c.workers > 0)
        va = DataLoader(val_ds, batch_size=o.val_batch or o.batch, shuffle=False, num_workers=o.val_workers,
                        pin_memory=True, persistent_workers=o.val_workers > 0)
        model = self.build_model().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=o.lr, weight_decay=o.weight_decay)
        total = max(o.epochs * len(tr), 1)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / total) ** o.poly_power)
        maximise = self.monitor_mode == "max"
        best, waiting, history, ep = (-math.inf if maximise else math.inf), 0, [], 0
        print(f"{c.name}: train {len(train_ds)} / val {len(val_ds)} | {o.epochs} epochs x {len(tr)} iterations | {device}",
              flush=True)
        for ep in range(1, o.epochs + 1):
            model.train()
            t0, running = time.time(), 0.0
            for batch in tr:
                batch = [t.to(device, non_blocking=True) for t in batch]
                with self.autocast(device):
                    loss = self.compute_loss(model, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                running += float(loss.item())
            m = self.validate(model, va, device)
            history.append({"epoch": ep, "loss": running / max(len(tr), 1), **m})
            pd.DataFrame(history).to_csv(run / "history.csv", index=False)
            score = m[self.monitor]
            improved = score > best if maximise else score < best
            if improved:
                best, waiting = score, 0
                self.save(model, run / f"best{self.weights_suffix}")
            else:
                waiting += 1
            print(f"epoch {ep}/{o.epochs} loss {history[-1]['loss']:.4f} {self.monitor} {score:.4f} "
                  f"{time.time() - t0:.0f}s{' *' if improved else ''}", flush=True)
            if waiting >= o.patience:
                print(f"early stop: no improvement for {o.patience} epochs (epoch {ep})", flush=True)
                break
        self.save(model, run / f"last{self.weights_suffix}")
        metrics = {f"best_{self.monitor}": best, "epochs_run": ep, "n_train": len(train_ds), "n_val": len(val_ds)}
        (run / "train_summary.json").write_text(json.dumps(metrics, indent=2))
        print(f"{c.name}: training finished -> {self.best_weights()}", flush=True)
        return TrainResult(run, self.best_weights(), metrics)

    # ── inference ──────────────────────────────────────────────────────────────────────────────────────────

    def predict_probabilities(self, model: Any, x: Any) -> Any:
        """Class probabilities (softmax), averaged with the horizontally flipped pass when ``flip_tta``."""
        logits = self.forward(model, x).float()
        prob = logits.softmax(1)
        if self.config.flip_tta:
            prob = (prob + self.forward(model, x.flip(-1)).float().softmax(1).flip(-1)) / 2
        return prob

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None) -> List[Path]:
        import torch

        from breast_seg.evaluation.predictions import write_label_map
        from breast_seg.training.torch_data import read_grey, stack_inputs

        device = self.device()
        model = self.load(weights).to(device).eval()
        batch = self.config.optim.val_batch or self.config.optim.batch
        written: List[Path] = []
        images = [Path(i) for i in images]
        with torch.no_grad():
            for i in range(0, len(images), batch):
                chunk = images[i:i + batch]
                x = stack_inputs([read_grey(str(p)) for p in chunk], self.to_input()).to(device)
                with self.autocast(device):
                    pred = self.predict_probabilities(model, x).argmax(1).to(torch.uint8).cpu().numpy()
                if len(pred) != len(chunk):
                    raise RuntimeError("prediction count does not match the input count")
                written += [write_label_map(Path(out_dir) / f"{p.stem}.png", m) for p, m in zip(chunk, pred)]
        return written

    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        """Mean per-image Dice by view on ``split`` (predicted with :meth:`predict`)."""
        from breast_seg.training.torch_data import read_grey

        rows = self.rows(split)
        out = self.run_dir / "predictions" / split
        self.predict([Path(p) for p in rows.image], out, weights)
        masks = dict(zip(rows.stem, rows["mask"]))
        return {"dice": dice_report(out, stems_of(rows.image), lambda s: read_grey(masks[s]) if masks.get(s) else None)}


@register_trainer("segformer")
class SegFormerTrainer(TorchSemanticTrainer):
    """SegFormer (MiT encoder from ``transformers``); grey input repeated to three ImageNet-normalised channels.
    The decoder emits logits at 1/4 resolution, upsampled bilinearly to the canvas."""

    config_class = SegFormerTrainConfig
    config: SegFormerTrainConfig
    weights_suffix = ""

    def to_input(self):
        from breast_seg.training.torch_data import imagenet3

        return imagenet3(self.config.image_mean, self.config.image_std)

    def build_model(self) -> Any:
        from transformers import SegformerForSemanticSegmentation

        return SegformerForSemanticSegmentation.from_pretrained(
            self.config.backbone, num_labels=len(self.config.data.classes), ignore_mismatched_sizes=True)

    def forward(self, model: Any, x: Any) -> Any:
        import torch.nn.functional as F

        logits = model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def save(self, model: Any, path: Path) -> None:
        model.save_pretrained(path)

    def load(self, weights: Optional[Path] = None) -> Any:
        from transformers import SegformerForSemanticSegmentation

        return SegformerForSemanticSegmentation.from_pretrained(str(weights or self.best_weights()))


@register_trainer("unet")
class UNetTrainer(TorchSemanticTrainer):
    """Plain U-Net from scratch on single-channel ``[0, 1]`` input (no ImageNet statistics)."""

    config_class = UNetTrainConfig
    config: UNetTrainConfig

    def to_input(self):
        from breast_seg.training.torch_data import grey01

        return grey01

    def build_model(self) -> Any:
        from breast_seg.training.nets import UNet

        return UNet(self.config.in_channels, len(self.config.data.classes), self.config.base_channels)

    def save(self, model: Any, path: Path) -> None:
        import torch

        torch.save({"model": model.state_dict(), "base_channels": self.config.base_channels,
                    "num_classes": len(self.config.data.classes), "in_channels": self.config.in_channels}, path)

    def load(self, weights: Optional[Path] = None) -> Any:

        from breast_seg.training.nets import UNet, load_checkpoint, upgrade_state_dict

        ck = load_checkpoint(weights or self.best_weights())
        base = ck.get("base_channels", ck.get("taban", self.config.base_channels))
        classes = ck.get("num_classes", ck.get("sinif", len(self.config.data.classes)))
        model = UNet(ck.get("in_channels", 1), classes, base)
        model.load_state_dict(upgrade_state_dict(ck["model"]))
        return model
