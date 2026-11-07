"""Mask quality: per-class Dice and IoU on semantic label maps (``0 bg / 1 pectoral / 2 breast-tissue / 3 nipple``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

from breast_seg.config import LABEL_VALUES

FOREGROUND: Dict[int, str] = {v: n for v, n in LABEL_VALUES.items() if v}


def dice_iou(pred: np.ndarray, gt: np.ndarray, classes: Mapping[int, str] = FOREGROUND) -> Dict[str, float]:
    """``{"dice_<name>": ..., "iou_<name>": ...}`` per image.

    A class absent from both maps is ``NaN`` (not 1): it carries no evidence and is skipped by ``nanmean``.
    """
    if pred is None or gt is None:
        raise ValueError("unreadable mask")
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction {pred.shape} != reference {gt.shape}")
    out: Dict[str, float] = {}
    for value, name in classes.items():
        p, g = pred == value, gt == value
        inter, total, union = float((p & g).sum()), float(p.sum() + g.sum()), float((p | g).sum())
        out[f"dice_{name}"] = 2 * inter / total if total else float("nan")
        out[f"iou_{name}"] = inter / union if union else float("nan")
    return out


@dataclass
class PooledDice:
    """Dataset-level Dice: intersections and sums pooled over every image before dividing (the training-time
    validation metric). ``mean_fg`` averages the foreground classes."""

    n_classes: int = len(LABEL_VALUES)
    inter: np.ndarray = field(default=None)  # type: ignore[assignment]
    pred_sum: np.ndarray = field(default=None)  # type: ignore[assignment]
    gt_sum: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.inter = np.zeros(self.n_classes)
        self.pred_sum = np.zeros(self.n_classes)
        self.gt_sum = np.zeros(self.n_classes)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        for c in range(self.n_classes):
            p, g = pred == c, gt == c
            self.inter[c] += float((p & g).sum())
            self.pred_sum[c] += float(p.sum())
            self.gt_sum[c] += float(g.sum())

    def result(self, names: Optional[Mapping[int, str]] = None) -> Dict[str, float]:
        names = names or LABEL_VALUES
        dice = 2 * self.inter / np.maximum(self.pred_sum + self.gt_sum, 1)
        out = {names.get(c, str(c)): float(dice[c]) for c in range(self.n_classes)}
        out["mean_fg"] = float(dice[1:].mean())
        return out
