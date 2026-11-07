"""Helpers shared by several trainers."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

from breast_seg.datasets.formats import link_or_copy, view_of
from breast_seg.evaluation.masks import FOREGROUND, dice_iou
from breast_seg.evaluation.predictions import read_label_map


def dice_report(pred_dir: Path, stems: Iterable[str],
                reference: Callable[[str], Optional[np.ndarray]]) -> Dict[str, Dict[str, float]]:
    """Mean per-image Dice by view (``{"MLO": {class: dice}, "CC": {...}}``) of label maps in ``pred_dir``.

    Classes absent from both maps are skipped (NaN); the pectoral class is not reported on CC views.
    """
    acc: Dict[str, Dict[str, List[float]]] = {"MLO": {}, "CC": {}}
    for stem in stems:
        pred, ref = read_label_map(pred_dir / f"{stem}.png"), reference(stem)
        if pred is None or ref is None:
            continue
        view = view_of(stem)
        for k, v in dice_iou(pred, ref).items():
            if k.startswith("dice_") and not np.isnan(v):
                acc[view].setdefault(k[5:], []).append(v)
    return {view: {n: float(np.mean(vals)) for n, vals in d.items() if not (view == "CC" and n == FOREGROUND[1])}
            for view, d in acc.items()}


def materialise_split(root: Path, split_of: Mapping[str, str], files: Mapping[str, Path], sub: str, suffix: str,
                      mode: str = "symlink") -> None:
    """Link ``files[stem]`` to ``root/<sub>/<split>/<stem><suffix>`` for every stem that has a file."""
    for stem, split in split_of.items():
        if stem in files:
            link_or_copy(files[stem], root / sub / split / f"{stem}{suffix}", mode)


def torch_device(device: str) -> str:
    """``"0"`` -> ``"cuda:0"``; ``"cpu"`` / ``"cuda:1"`` unchanged."""
    d = str(device)
    return f"cuda:{d}" if d.isdigit() else d


@dataclass
class CommandRunner:
    """Runs external commands (the nnU-Net CLI). Swap it for a recorder in tests."""

    dry_run: bool = False
    history: List[List[str]] = field(default_factory=list)

    def __call__(self, cmd: Sequence[str], env: Optional[Mapping[str, str]] = None, log: Optional[Path] = None) -> int:
        cmd = [str(c) for c in cmd]
        self.history.append(cmd)
        if self.dry_run:
            return 0
        import os

        full_env = {**os.environ, **(env or {})}
        if log is None:
            return subprocess.run(cmd, env=full_env, check=True).returncode
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as fh:
            return subprocess.run(cmd, env=full_env, check=True, stdout=fh, stderr=subprocess.STDOUT).returncode


def stems_of(images: Sequence[Union[str, Path]]) -> List[str]:
    return [Path(p).stem for p in images]
