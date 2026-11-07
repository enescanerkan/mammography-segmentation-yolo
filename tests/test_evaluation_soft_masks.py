"""soft_masks with a fake YOLO26 segmentation head (no weights)."""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from breast_seg.soft_masks import soft_masks  # noqa: E402


class FakeNet(torch.nn.Module):
    def __init__(self, det, protos):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.det, self.protos = det, protos

    def forward(self, x):
        assert x.shape == (1, 3, 64, 64) and float(x.max()) <= 1.0
        return (self.det[None], self.protos[None]), None


def test_soft_masks_per_class_probabilities():
    c = 2
    protos = torch.zeros(c, 16, 16)
    protos[0] = 4.0                                     # coefficient 1 on this prototype -> sigmoid(4) inside the box
    protos[1, :8] = 6.0
    det = torch.tensor([
        [0, 0, 32, 64, 0.9, 1, 1.0, 0.0],              # breast tissue, left half
        [0, 0, 64, 64, 0.8, 2, 0.0, 1.0],              # nipple class, top rows only
        [0, 0, 64, 64, 0.1, 0, 1.0, 0.0],              # pectoral below the confidence threshold
    ])
    model = SimpleNamespace(model=FakeNet(det, protos), names={0: "pectoral", 1: "breast-tissue", 2: "nipple"})
    out = soft_masks(model, np.zeros((64, 64), np.uint8), conf=0.25, size=64)
    assert out.shape == (3, 64, 64) and out.dtype == np.float32
    assert out[0].max() == 0                            # no instance emitted
    assert out[1, 10, 10] == pytest.approx(1 / (1 + np.exp(-4)), abs=1e-3) and out[1, 10, 50] == 0   # cropped to box
    assert out[2, 2, 30] > 0.99 and out[2, 60, 30] == pytest.approx(0.5, abs=1e-3)
