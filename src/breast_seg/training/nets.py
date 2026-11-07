"""Network definitions of the from-scratch PyTorch families (imported only when such a model is built).

* :class:`UNet` - the plain U-Net (Ronneberger et al., 2015): double 3x3 conv + BatchNorm + ReLU, max-pool down,
  transposed-conv up, concatenated skips. No attention, no residuals, no pre-training.
* :class:`AttUNet` - Attention U-Net (Oktay et al., 2018 gates on the skips) with CoordConv input channels and two
  heads: semantic segmentation and one heatmap per landmark, read out by :func:`soft_argmax`.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    """(3x3 conv -> BN -> ReLU) x 2."""

    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _widths(base: int) -> List[int]:
    return [base, base * 2, base * 4, base * 8]


def _align(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class UNet(nn.Module):
    """Four downsamplings, a bottleneck, four upsamplings with skip concatenation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 64):
        super().__init__()
        g = _widths(base_channels)
        self.down = nn.ModuleList()
        prev = in_channels
        for k in g:
            self.down.append(DoubleConv(prev, k))
            prev = k
        self.pool = nn.MaxPool2d(2)
        self.bottom = DoubleConv(g[-1], g[-1] * 2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = g[-1] * 2
        for k in reversed(g):
            self.up.append(nn.ConvTranspose2d(prev, k, 2, stride=2))
            self.dec.append(DoubleConv(k * 2, k))
            prev = k
        self.head = nn.Conv2d(g[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, block, s in zip(self.up, self.dec, reversed(skips)):
            x = _align(up(x), s)
            x = block(torch.cat([s, x], dim=1))
        return self.head(x)


class AttentionGate(nn.Module):
    """Gate a skip connection with the decoder signal (Oktay et al., 2018)."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.wg = nn.Conv2d(gate_channels, inter_channels, 1, bias=True)
        self.wx = nn.Conv2d(skip_channels, inter_channels, 1, bias=True)
        self.psi = nn.Sequential(nn.Conv2d(inter_channels, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return skip * self.psi(F.relu(self.wg(gate) + self.wx(skip), inplace=True))


class AttUNet(nn.Module):
    """Attention U-Net + CoordConv; returns ``(segmentation logits, landmark heatmaps)``."""

    def __init__(self, num_classes: int = 4, num_keypoints: int = 3, base_channels: int = 64):
        super().__init__()
        g = _widths(base_channels)
        self.down = nn.ModuleList()
        prev = 3  # grey + x + y coordinate channels
        for k in g:
            self.down.append(DoubleConv(prev, k))
            prev = k
        self.pool = nn.MaxPool2d(2)
        self.bottom = DoubleConv(g[-1], g[-1] * 2)
        self.up = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.dec = nn.ModuleList()
        prev = g[-1] * 2
        for k in reversed(g):
            self.up.append(nn.ConvTranspose2d(prev, k, 2, stride=2))
            self.gates.append(AttentionGate(k, k, max(k // 2, 8)))
            self.dec.append(DoubleConv(k * 2, k))
            prev = k
        self.seg_head = nn.Conv2d(g[0], num_classes, 1)
        self.lm_head = nn.Conv2d(g[0], num_keypoints, 1)

    @staticmethod
    def add_coordinates(x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xx, yy], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.add_coordinates(x)
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottom(x)
        for up, gate, block, s in zip(self.up, self.gates, self.dec, reversed(skips)):
            x = _align(up(x), s)
            x = block(torch.cat([gate(s, x), x], 1))
        return self.seg_head(x), self.lm_head(x)


def soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """``(B, K, H, W)`` heatmaps -> ``(B, K, 2)`` normalised ``(x, y)`` (spatial softmax expectation)."""
    b, k, h, w = heatmaps.shape
    p = heatmaps.flatten(2).softmax(-1).view(b, k, h, w)
    return expectation(p)


def expectation(prob: torch.Tensor) -> torch.Tensor:
    """Expected normalised ``(x, y)`` of already normalised ``(B, K, H, W)`` probability maps."""
    _, _, h, w = prob.shape
    xs = torch.linspace(0, 1, w, device=prob.device).view(1, 1, 1, w)
    ys = torch.linspace(0, 1, h, device=prob.device).view(1, 1, h, 1)
    return torch.stack([(prob * xs).sum((2, 3)), (prob * ys).sum((2, 3))], -1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over the foreground classes, pooled over the batch (class 0 = background is excluded)."""
    prob = logits.softmax(1)
    oh = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * oh).sum(dims)[1:]
    denom = prob.sum(dims)[1:] + oh.sum(dims)[1:]
    return 1.0 - ((2 * inter + eps) / (denom + eps)).mean()


#: Parameter-name prefixes of checkpoints written by the original research scripts -> names used here.
LEGACY_PREFIXES = (("inis.", "down."), ("dip.", "bottom."), ("yukselt.", "up."), ("cikis.", "dec."),
                   ("kapi.", "gates."), ("seg_bas.", "seg_head."), ("lm_bas.", "lm_head."), ("bas.", "head."))


def upgrade_state_dict(state: dict) -> dict:
    """Rename the parameters of a legacy checkpoint so it loads into :class:`UNet` / :class:`AttUNet`."""
    out = {}
    for key, value in state.items():
        for old, new in LEGACY_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        out[key] = value
    return out


def load_checkpoint(path) -> dict:
    """``torch.load`` restricted to tensors and plain values (the checkpoints written here hold nothing else)."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # torch < 1.13 has no weights_only
        return torch.load(str(path), map_location="cpu")
