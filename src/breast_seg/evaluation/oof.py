"""Out-of-fold aggregation: every image is predicted by the one fold model that never saw it during training."""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union


@dataclass
class OOFReport:
    """Coverage of a merged out-of-fold prediction directory."""

    output_dir: Path
    n_images: int = 0
    missing: List[str] = field(default_factory=list)
    foreign: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def merge_oof_predictions(fold_dirs: Mapping[int, Union[str, Path]], output_dir: Union[str, Path],
                          expected: Optional[Mapping[str, int]] = None, suffix: str = ".png",
                          strict: bool = True) -> OOFReport:
    """Copy each fold's predictions into ``output_dir``.

    With ``expected`` (stem -> fold) only the stems a fold owns are taken from it; a stem found in another fold's
    directory is reported in ``foreign`` and ignored, a stem without a prediction in ``missing``. ``strict`` raises
    when a stem is missing.
    """
    out = Path(output_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    report = OOFReport(out)
    taken: Dict[str, int] = {}
    for fold, d in sorted(fold_dirs.items()):
        for f in sorted(Path(d).glob(f"*{suffix}")):
            stem = f.name[: -len(suffix)]
            if expected is not None and expected.get(stem) != fold:
                report.foreign.append(stem)
                continue
            if stem in taken:
                raise ValueError(f"{stem} predicted by folds {taken[stem]} and {fold}")
            shutil.copyfile(f, out / f.name)
            taken[stem] = fold
    report.n_images = len(taken)
    if expected is not None:
        report.missing = sorted(set(expected) - set(taken))
        if strict and report.missing:
            raise FileNotFoundError(f"{len(report.missing)} image(s) have no out-of-fold prediction")
    return report
