"""The common prediction format: one 640x640 ``uint8`` label map per image (``0 bg, 1 pectoral, 2 breast-tissue,
3 nipple``), file name ``<stem>.png``. Semantic models write it directly; instance models are flattened here."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from breast_seg.config import PAINT_ORDER


@dataclass
class FlattenStats:
    """Counters of :func:`instances_to_label_map` (per label value)."""

    rescued: Dict[int, int] = field(default_factory=lambda: {v: 0 for _, v in PAINT_ORDER})
    dropped: Dict[int, int] = field(default_factory=lambda: {v: 0 for _, v in PAINT_ORDER})
    empty: int = 0

    def add(self, other: FlattenStats) -> None:
        for v in self.rescued:
            self.rescued[v] += other.rescued.get(v, 0)
            self.dropped[v] += other.dropped.get(v, 0)
        self.empty += other.empty


def instances_to_label_map(masks: Optional[np.ndarray], classes: Optional[np.ndarray], confs: Optional[np.ndarray],
                           shape: Tuple[int, int], conf: float = 0.25, floor: float = 0.01, is_cc: bool = False,
                           paint_order: Sequence[Tuple[int, int]] = PAINT_ORDER) -> Tuple[np.ndarray, FlattenStats]:
    """Flatten instance masks ``(N, h, w)`` into a label map of ``shape``.

    Instances with confidence >= ``conf`` are unioned per class and painted in ``paint_order``. If a class has no
    instance above ``conf``, its single most confident instance is taken back when it is above ``floor``
    (semantic models have no confidence gate; every mammogram has a nipple). The rescue is not applied to the
    pectoral class of a CC view, which has no pectoral muscle. ``floor = 0`` disables the rescue. Inference must
    therefore run at ``min(conf, floor)``.
    """
    h, w = shape
    label = np.zeros((h, w), np.uint8)
    stats = FlattenStats()
    if masks is None or classes is None or len(masks) == 0:
        stats.empty = 1
        return label, stats
    classes = np.asarray(classes).astype(int)
    confs = np.asarray(confs, float)
    for yolo_cls, value in paint_order:
        idx = np.nonzero((classes == yolo_cls) & (confs >= conf))[0]
        if len(idx) == 0:
            allow = floor > 0 and not (is_cc and value == 1)
            cand = np.nonzero((classes == yolo_cls) & (confs >= floor))[0] if allow else np.array([], int)
            if len(cand):
                idx = cand[[int(np.argmax(confs[cand]))]]
                stats.rescued[value] += 1
            else:
                if not (is_cc and value == 1):
                    stats.dropped[value] += 1
                continue
        layer = np.zeros((h, w), np.uint8)
        for k in idx:
            m = np.asarray(masks[k], np.float32)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            layer |= (m > 0.5).astype(np.uint8)
        label[layer > 0] = value
    return label, stats


def write_label_map(path: Union[str, Path], label: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(label, np.uint8)):
        raise OSError(f"could not write {path.name}")
    return path


def read_label_map(path: Union[str, Path]) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
