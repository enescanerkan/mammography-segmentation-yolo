"""Scoring a directory of predictions (common label-map format) end to end.

    evaluator = SegmentationEvaluator.from_config(load_evaluation_config("configs/cv/cv10_yolo26m.yaml"), paths)
    result = evaluator.evaluate(preds_dir, stems)          # per image, per pair, summary
    result.save(out_dir, name)

Two layers are reported and should be read together: mask quality (per-class Dice / IoU by view) and the
downstream measurements against the reference landmarks (mm errors and the PNL decision with both scorings).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd

from breast_seg.config import ConfigError, PathsConfig, apply_overrides, from_dict, read_yaml
from breast_seg.datasets.formats import view_of
from breast_seg.evaluation.gt import GroundTruth, landmarks_by_stem
from breast_seg.evaluation.landmarks import MIN_NIPPLE_PX, ImageGeometry
from breast_seg.evaluation.masks import FOREGROUND, dice_iou
from breast_seg.evaluation.measure import MeasureOptions, measure_cc, measure_mlo
from breast_seg.evaluation.pairs import pairs_from_fold_table, pairs_from_labels, score_pairs
from breast_seg.evaluation.predictions import read_label_map
from breast_seg.evaluation.rule import decision_summary, ref_is_good

ERROR_COLUMNS = ("e_nipple", "e_pec_top", "e_pec_bot", "e_pnl", "e_cc")


@dataclass
class EvaluationConfig:
    """Inputs of an evaluation (all paths may use ``${placeholders}``).

    ``masks_dir``: semantic dataset root (``masks/<split>/<stem>.png`` + ``index.csv``) providing reference masks;
    optional. ``pairs``: ``labels`` builds the pairs from the label tables (``pairs_split`` filters their ``Split``
    column), ``fold_table`` reads them from ``pairs_csv`` (pair-level fold assignment).
    """

    mlo_labels: str = ""
    cc_labels: str = ""
    mlo_metadata: str = ""
    cc_metadata: str = ""
    metadata_key: str = "sop_uid"
    masks_dir: Optional[str] = None
    pairs: str = "labels"
    pairs_split: Optional[str] = None
    pairs_csv: Optional[str] = None
    cc_prefix: str = "cc_"
    min_nipple_px: int = MIN_NIPPLE_PX
    fallback: bool = False
    n_total: Optional[int] = None

    def __post_init__(self) -> None:
        if self.pairs not in ("labels", "fold_table", "none"):
            raise ConfigError("evaluation.pairs must be labels, fold_table or none")
        if self.pairs == "fold_table" and not self.pairs_csv:
            raise ConfigError("evaluation.pairs_csv is required with pairs: fold_table")


def load_evaluation_config(path: Union[str, Path], overrides=None, key: str = "evaluation") -> EvaluationConfig:
    """Read an evaluation section: the ``evaluation:`` mapping of a (CV) config, or a file that is only that."""
    data = apply_overrides(read_yaml(path), overrides)
    section = data.get(key, data) if isinstance(data, Mapping) else data
    try:
        return from_dict(EvaluationConfig, section, key)
    except ConfigError as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


@dataclass
class EvaluationResult:
    views: pd.DataFrame
    pairs: pd.DataFrame
    summary: Dict[str, object] = field(default_factory=dict)

    def save(self, out_dir: Union[str, Path], name: str) -> Dict[str, Path]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {"views": out / f"{name}_views.csv", "pairs": out / f"{name}_pairs.csv", "summary": out / f"{name}_summary.json"}
        self.views.to_csv(files["views"], index=False)
        self.pairs.to_csv(files["pairs"], index=False)
        files["summary"].write_text(json.dumps(self.summary, indent=2, default=float))
        return files


def _describe(s: pd.Series) -> Dict[str, float]:
    s = s.dropna()
    return {"mean": float(s.mean()), "median": float(s.median()), "p90": float(s.quantile(0.9)), "n": int(len(s))}


class SegmentationEvaluator:
    """Per-image mask metrics + measurements, per-pair decisions and a summary for one prediction directory."""

    def __init__(self, gt: GroundTruth, mlo_meta: pd.DataFrame, cc_meta: pd.DataFrame,
                 gt_mask: Optional[Callable[[str], Optional[Path]]] = None, pairs: Optional[pd.DataFrame] = None,
                 options: MeasureOptions = MeasureOptions(), cc_prefix: str = "cc_", n_total: Optional[int] = None,
                 landmarks: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None):
        self.gt, self.options, self.cc_prefix, self.n_total = gt, options, cc_prefix, n_total
        self.mlo_meta, self.cc_meta = mlo_meta, cc_meta
        self.gt_mask = gt_mask or (lambda stem: None)
        self.pairs = pairs
        self.landmarks = landmarks

    @classmethod
    def from_config(cls, cfg: EvaluationConfig, paths: Optional[PathsConfig] = None,
                    landmarks_csv: Optional[Union[str, Path]] = None) -> SegmentationEvaluator:
        paths = paths or PathsConfig.load()
        r = paths.resolve
        gt = GroundTruth.from_csv(r(cfg.mlo_labels), r(cfg.cc_labels))
        key = cfg.metadata_key

        def meta(p: str) -> pd.DataFrame:
            df = pd.read_csv(r(p), dtype={key: str})
            return df.set_index(key)

        gt_mask = None
        if cfg.masks_dir:
            root = r(cfg.masks_dir)
            split_of = pd.read_csv(root / "index.csv", dtype={"stem": str}).set_index("stem").split.to_dict()

            def gt_mask(stem: str) -> Optional[Path]:
                sp = split_of.get(stem)
                return root / "masks" / sp / f"{stem}.png" if sp else None

        if cfg.pairs == "labels":
            pairs = pairs_from_labels(gt.mlo_table, gt.cc_table, cfg.pairs_split, cfg.cc_prefix)
        elif cfg.pairs == "fold_table":
            pairs = pairs_from_fold_table(pd.read_csv(r(cfg.pairs_csv), dtype=str), cc_prefix=cfg.cc_prefix)
        else:
            pairs = None
        lm = landmarks_by_stem(pd.read_csv(landmarks_csv, dtype={"stem": str})) if landmarks_csv else None
        return cls(gt, meta(cfg.mlo_metadata), meta(cfg.cc_metadata), gt_mask, pairs,
                   MeasureOptions(cfg.min_nipple_px, cfg.fallback), cfg.cc_prefix, cfg.n_total, lm)

    def _sop(self, stem: str) -> str:
        return stem[len(self.cc_prefix):] if stem.startswith(self.cc_prefix) else stem

    def measure_image(self, stem: str, pred: np.ndarray) -> Dict[str, object]:
        """Mask metrics (if a reference mask exists) and measurements of one image."""
        view = view_of(stem, self.cc_prefix)
        sop = self._sop(stem)
        row: Dict[str, object] = {"stem": stem, "view": view, "missing": False}
        mpath = self.gt_mask(stem)
        if mpath is not None and Path(mpath).is_file():
            row.update(dice_iou(pred, read_label_map(mpath)))
        meta = self.cc_meta if view == "CC" else self.mlo_meta
        if sop not in meta.index:
            return row
        geom = ImageGeometry.from_row(meta.loc[sop])
        lm = self.landmarks.get(stem, {}) if self.landmarks is not None else None
        if view == "MLO":
            row.update(measure_mlo(pred, geom, self.gt.mlo.get(sop), self.options, lm))
        else:
            row.update(measure_cc(pred, geom, self.gt.cc.get(sop), self.options, lm))
        return row

    def views(self, preds_dir: Union[str, Path], stems: Iterable[str]) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for stem in stems:
            pred = read_label_map(Path(preds_dir) / f"{stem}.png")
            if pred is None:
                rows.append({"stem": stem, "view": view_of(stem, self.cc_prefix), "missing": True})
                continue
            rows.append(self.measure_image(stem, pred))
        return pd.DataFrame(rows)

    def summary(self, views: pd.DataFrame, pairs: pd.DataFrame) -> Dict[str, object]:
        ok = views[~views.missing] if "missing" in views else views
        missing = int(views["missing"].sum()) if "missing" in views else 0
        out: Dict[str, object] = {"n_images": int(len(views)), "n_missing": missing, "dice": {}, "landmark_mm": {}}
        for view in ("MLO", "CC"):
            d = ok[ok.view == view]
            out["dice"][view] = {n: float(d[f"dice_{n}"].mean()) for n in FOREGROUND.values()
                                 if f"dice_{n}" in d and d[f"dice_{n}"].notna().any()}
            out["landmark_mm"][view] = {c: _describe(d[c]) for c in ERROR_COLUMNS if c in d and d[c].notna().any()}
        if len(pairs):
            answered = pairs.dropna(subset=["delta_mm"])
            out["decision"] = decision_summary(pairs.delta_mm, ref_is_good(pairs.truth), self.n_total)
            out["delta_mae_mm"] = _describe((answered.delta_mm - answered.gt_delta_mm).abs())
        return out

    def evaluate(self, preds_dir: Union[str, Path], stems: Optional[Iterable[str]] = None) -> EvaluationResult:
        """Score ``preds_dir``. ``stems`` defaults to every image named by the pairs (or every PNG present)."""
        if stems is None:
            if self.pairs is not None:
                stems = list(dict.fromkeys(list(self.pairs.mlo_stem) + list(self.pairs.cc_stem)))
            else:
                stems = sorted(p.stem for p in Path(preds_dir).glob("*.png"))
        views = self.views(preds_dir, stems)
        pairs = score_pairs(views.set_index("stem"), self.pairs) if self.pairs is not None else pd.DataFrame()
        return EvaluationResult(views, pairs, self.summary(views, pairs))
