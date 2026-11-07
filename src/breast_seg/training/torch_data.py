"""PyTorch datasets of the semantic families (imported only when training / predicting)."""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

InputFn = Callable[[np.ndarray], np.ndarray]


def grey01(img: np.ndarray) -> np.ndarray:
    """``(H, W) uint8`` -> ``(1, H, W) float32`` in ``[0, 1]`` (from-scratch networks)."""
    return (img.astype(np.float32) / 255.0)[None]


def imagenet3(mean: Sequence[float], std: Sequence[float]) -> InputFn:
    """``(H, W) uint8`` -> ``(3, H, W)`` grey repeated on three channels, ImageNet-normalised (pre-trained encoders)."""
    m, s = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def fn(img: np.ndarray) -> np.ndarray:
        x = np.repeat(img[..., None].astype(np.float32) / 255.0, 3, axis=2)
        return ((x - m) / s).transpose(2, 0, 1)

    return fn


def read_grey(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"unreadable image {path}")
    return img


class SemanticDataset(Dataset):
    """Rows of a semantic index (``stem, image, mask``); ``transform`` is an Albumentations ``Compose`` or None."""

    def __init__(self, rows: pd.DataFrame, to_input: InputFn, transform=None):
        self.rows = rows.reset_index(drop=True)
        self.to_input, self.transform = to_input, transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        img, msk = read_grey(r["image"]), read_grey(r["mask"])
        if self.transform is not None:
            out = self.transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"]
        return torch.from_numpy(np.ascontiguousarray(self.to_input(img))), torch.from_numpy(msk.astype(np.int64))


class LandmarkDataset(SemanticDataset):
    """Semantic rows plus canvas landmarks: returns ``(x, mask, keypoints (K, 2) in [0, 1], valid (K,))``.

    Missing roles and points pushed off the canvas by the augmentation are invalid and masked out of the loss.
    The ``transform`` must be composed with ``KeypointParams(format="xy", remove_invisible=False)``.
    """

    def __init__(self, rows: pd.DataFrame, landmarks: Mapping[str, Mapping[str, np.ndarray]], roles: Sequence[str],
                 to_input: InputFn, transform=None):
        keep = [s in landmarks and len(landmarks[s]) > 0 for s in rows["stem"]]
        super().__init__(rows[keep], to_input, transform)
        self.dropped = int(len(keep) - sum(keep))
        self.landmarks, self.roles = landmarks, list(roles)

    def __getitem__(self, i: int):
        r = self.rows.iloc[i]
        img, msk = read_grey(r["image"]), read_grey(r["mask"])
        d = self.landmarks[r["stem"]]
        kp = [tuple(map(float, d[k])) if k in d else (-1.0, -1.0) for k in self.roles]
        valid = np.array([k in d for k in self.roles], np.float32)
        if self.transform is not None:
            out = self.transform(image=img, mask=msk, keypoints=kp)
            img, msk, kp = out["image"], out["mask"], out["keypoints"]
        h, w = img.shape[:2]
        k = np.asarray(kp, np.float32).reshape(-1, 2)
        inside = (k[:, 0] >= 0) & (k[:, 0] < w) & (k[:, 1] >= 0) & (k[:, 1] < h)
        valid = valid * inside.astype(np.float32)
        k = np.stack([k[:, 0] / w, k[:, 1] / h], 1)
        return (torch.from_numpy(np.ascontiguousarray(self.to_input(img))), torch.from_numpy(msk.astype(np.int64)),
                torch.from_numpy(k.astype(np.float32)), torch.from_numpy(valid))


def landmark_lookup(table: pd.DataFrame, aliases: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, np.ndarray]]:
    """``stem, role, x, y`` -> ``{stem: {role: (x, y)}}`` with optional role aliases."""
    aliases = aliases or {}
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for r in table.itertuples(index=False):
        out.setdefault(str(r.stem), {})[aliases.get(r.role, r.role)] = np.array([float(r.x), float(r.y)])
    return out


def stack_inputs(images: List[np.ndarray], to_input: InputFn) -> torch.Tensor:
    return torch.from_numpy(np.stack([to_input(i) for i in images]))
