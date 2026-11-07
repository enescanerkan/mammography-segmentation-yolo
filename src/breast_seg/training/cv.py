"""k-fold cross-validation for any registered family (``configs/cv/*.yaml``).

Folds are read, never re-drawn: a fold table (one row per image, or one row per MLO/CC pair) fixes every image's
fold so that different models are compared case by case on the same folds. Protocol of fold ``i``: test = fold
``i``, validation = fold ``(i + val_offset) % k``, train = the rest. Unlabelled images of the table are predicted
in their own test fold only. Each fold model predicts only its test fold; the out-of-fold predictions are merged
into one directory and scored once.

    runner = CrossValidationRunner.from_yaml("configs/cv/cv10_yolo26m.yaml")
    runner.run()                 # train + predict every fold, merge, evaluate
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

import pandas as pd

from breast_seg.config import ConfigError, PathsConfig, apply_overrides, from_dict, read_yaml
from breast_seg.datasets.formats import semantic_paths
from breast_seg.datasets.splits import SPLITS, FoldAssignment, FoldSplit, folds_from_pairs
from breast_seg.evaluation.evaluator import EvaluationConfig, EvaluationResult, SegmentationEvaluator
from breast_seg.evaluation.oof import OOFReport, merge_oof_predictions
from breast_seg.evaluation.rule import FLAT_RULE, TOLERANCE_RULE, ref_is_good
from breast_seg.training.base import SplitSpec, Trainer, build_trainer, load_train_config


@dataclass
class FoldTableConfig:
    """The fold table. ``format: images`` has one row per image (``stem_column``, ``fold_column`` and optional
    group / labelled columns); ``format: pairs`` one row per breast with the MLO and CC image ids, converted with the
    dataset naming (``cc_prefix`` for CC)."""

    csv: str = ""
    format: str = "images"
    stem_column: str = "stem"
    fold_column: str = "fold"
    group_column: Optional[str] = None
    labelled_column: Optional[str] = None
    mlo_column: str = "SOPInstanceUID_mlo"
    cc_column: str = "SOPInstanceUID_cc"
    cc_prefix: str = "cc_"

    def __post_init__(self) -> None:
        if not self.csv:
            raise ConfigError("folds.csv is required")
        if self.format not in ("images", "pairs"):
            raise ConfigError("folds.format must be images or pairs")


@dataclass
class GroupTableConfig:
    """Image key -> group (study) tables used to place images missing from the fold table into the fold of their
    study; ``prefixes`` are the stem prefixes under which a key may appear (``""`` MLO, ``"cc_"`` CC)."""

    csv: List[str] = field(default_factory=list)
    key: str = "SOPInstanceUID"
    group: str = "StudyInstanceUID"
    prefixes: List[str] = field(default_factory=lambda: ["", "cc_"])


@dataclass
class CVDataConfig:
    """Where the fold images come from: a YOLO dataset (images + polygon labels), a semantic dataset (images +
    label maps) and a folder of images that have no label (predicted only)."""

    yolo_dir: Optional[str] = None
    semantic_dir: Optional[str] = None
    unlabelled_images: Optional[str] = None
    group_table: Optional[GroupTableConfig] = None

    def __post_init__(self) -> None:
        if not self.yolo_dir and not self.semantic_dir:
            raise ConfigError("data: give yolo_dir and / or semantic_dir")


@dataclass
class CVConfig:
    name: str = ""
    description: str = ""
    train_config: str = ""
    overrides: Dict[str, Any] = field(default_factory=dict)
    folds: Optional[FoldTableConfig] = None
    n_folds: int = 10
    val_offset: int = 1
    run_folds: Optional[List[int]] = None
    data: Optional[CVDataConfig] = None
    work_dir: str = "${runs_dir}/cv"
    predictions_dir: str = "${predictions_dir}"
    results_dir: str = "${results_dir}"
    skip_finished: bool = True
    evaluation: Optional[EvaluationConfig] = None

    def __post_init__(self) -> None:
        if not self.name or not self.train_config:
            raise ConfigError("name and train_config are required")
        if self.folds is None or self.data is None:
            raise ConfigError("folds and data are required")
        if self.n_folds < 2:
            raise ConfigError("n_folds must be >= 2")
        if self.run_folds and any(not 0 <= f < self.n_folds for f in self.run_folds):
            raise ConfigError("run_folds outside 0..n_folds-1")


def load_cv_config(path: Union[str, Path], overrides: Union[None, Mapping[str, Any], Iterable[str]] = None) -> CVConfig:
    data = apply_overrides(read_yaml(path), overrides)
    try:
        return from_dict(CVConfig, data)
    except ConfigError as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


@dataclass
class CVReport:
    folds_run: List[int]
    oof: Optional[OOFReport] = None
    evaluation: Optional[EvaluationResult] = None


class CrossValidationRunner:
    """Materialises, trains, predicts, merges and scores the folds of one :class:`CVConfig`."""

    def __init__(self, config: CVConfig, paths: Optional[PathsConfig] = None, config_dir: Union[str, Path] = ".",
                 trainer_factory: Callable[..., Trainer] = build_trainer):
        self.config = config
        self.paths = paths or PathsConfig.load()
        self.config_dir = Path(config_dir)
        self.trainer_factory = trainer_factory
        self._assignment: Optional[FoldAssignment] = None
        self._files: Optional[Dict[str, Dict[str, Path]]] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path], overrides=None, paths: Optional[PathsConfig] = None) -> CrossValidationRunner:
        return cls(load_cv_config(path, overrides), paths, Path(path).resolve().parent)

    # ── inputs ─────────────────────────────────────────────────────────────────────────────────────────────

    def train_config_path(self) -> Path:
        raw = self.paths.expand(self.config.train_config)
        p = Path(raw)
        return p if p.is_absolute() else (self.config_dir / p).resolve()

    def base_config(self):
        return load_train_config(self.train_config_path(), self.config.overrides)

    def files(self) -> Dict[str, Dict[str, Path]]:
        """``{"images": {stem: path}, "labels": {...}, "masks": {...}}`` over the configured sources."""
        if self._files is not None:
            return self._files
        d, r = self.config.data, self.paths.resolve
        images: Dict[str, Path] = {}
        labels: Dict[str, Path] = {}
        masks: Dict[str, Path] = {}
        if d.yolo_dir:
            root = r(d.yolo_dir)
            for split in SPLITS:
                for img in sorted((root / "images" / split).glob("*.png")):
                    images.setdefault(img.stem, img)
                    lbl = root / "labels" / split / f"{img.stem}.txt"
                    if lbl.is_file():
                        labels[img.stem] = lbl
        if d.semantic_dir:
            root = r(d.semantic_dir)
            idx = semantic_paths(root, pd.read_csv(root / "index.csv", dtype={"stem": str}))
            for s, img, m in zip(idx.stem, idx.image, idx["mask"]):
                images.setdefault(s, Path(img))
                masks[s] = Path(m)
        self._files = {"images": images, "labels": labels, "masks": masks}
        return self._files

    def assignment(self) -> FoldAssignment:
        if self._assignment is not None:
            return self._assignment
        f = self.config.folds
        table = pd.read_csv(self.paths.resolve(f.csv), dtype=str)
        if f.format == "pairs":
            table = folds_from_pairs(table, f.mlo_column, f.cc_column, f.fold_column, f.group_column, "", f.cc_prefix)
            cols = ("stem", "fold", "group" if f.group_column else None, None)
        else:
            cols = (f.stem_column, f.fold_column, f.group_column, f.labelled_column)
        table[cols[1]] = table[cols[1]].astype(int)
        lab_col = cols[3]
        a = FoldAssignment.from_frame(table, *cols, n_folds=self.config.n_folds, val_offset=self.config.val_offset)
        files = self.files()
        gt = self.config.data.group_table
        if gt is not None:
            group_of: Dict[str, str] = {}
            for p in gt.csv:
                df = pd.read_csv(self.paths.resolve(p), usecols=[gt.key, gt.group], dtype=str)
                for k, g in zip(df[gt.key], df[gt.group]):
                    for prefix in gt.prefixes:
                        group_of.setdefault(f"{prefix}{k}", g)
            for s in list(a.fold_of):
                if s not in a.group_of and s in group_of:
                    a.group_of[s] = group_of[s]
            a.extend_by_group(sorted(files["images"]), group_of)
        if lab_col is None:
            a.labelled = {s: (s in files["labels"] or s in files["masks"]) for s in a.fold_of}
        self._assignment = a
        return a

    def unlabelled_image(self, stem: str) -> Optional[Path]:
        if not self.config.data.unlabelled_images:
            return None
        p = self.paths.resolve(self.config.data.unlabelled_images) / f"{stem}.png"
        return p if p.is_file() else None

    def split(self, fold: int) -> FoldSplit:
        return self.assignment().split(fold)

    def spec(self, fold: int) -> SplitSpec:
        """The fold's :class:`SplitSpec`; raises if an image of the fold cannot be found."""
        fs, files = self.split(fold), self.files()
        split_of = fs.split_of()
        images = {}
        for s in split_of:
            p = files["images"].get(s) or self.unlabelled_image(s)
            if p is None:
                raise FileNotFoundError(f"fold {fold}: no image for one of its stems")
            images[s] = p
        return SplitSpec(split_of, images, {s: files["labels"][s] for s in split_of if s in files["labels"]},
                         {s: files["masks"][s] for s in split_of if s in files["masks"]}, fold)

    # ── execution ──────────────────────────────────────────────────────────────────────────────────────────

    @property
    def work_dir(self) -> Path:
        return self.paths.resolve(self.config.work_dir) / self.config.name

    def fold_name(self, fold: int) -> str:
        return f"{self.config.name}_fold{fold}"

    def fold_prediction_dir(self, fold: int) -> Path:
        return self.paths.resolve(self.config.predictions_dir) / self.fold_name(fold)

    def folds_to_run(self) -> List[int]:
        return list(self.config.run_folds) if self.config.run_folds else list(range(self.config.n_folds))

    def fold_trainer(self, fold: int) -> Trainer:
        base = self.base_config()
        cfg = dataclasses.replace(base, name=self.fold_name(fold))
        trainer = self.trainer_factory(cfg, self.paths)
        return trainer.use_split(self.spec(fold), self.work_dir / f"fold{fold}")

    def prepare(self) -> List[Trainer]:
        """Materialise every fold to run (and the shared nnU-Net dataset); no training."""
        folds = self.folds_to_run()
        trainers = [self.fold_trainer(f) for f in folds]
        shared = self.trainer_factory(dataclasses.replace(self.base_config(), name=self.config.name), self.paths)
        if hasattr(shared, "prepare_folds"):
            shared.prepare_folds([self.spec(f) for f in range(self.config.n_folds)])
        else:
            for t in trainers:
                t.prepare()
        (self.work_dir).mkdir(parents=True, exist_ok=True)
        self.assignment().to_frame().to_csv(self.work_dir / "assignment.csv", index=False)
        return trainers

    def run_fold(self, fold: int, train: bool = True, predict: bool = True) -> Path:
        trainer = self.fold_trainer(fold)
        marker = self.work_dir / f"fold{fold}"
        marker.mkdir(parents=True, exist_ok=True)
        if train and not (self.config.skip_finished and (marker / ".trained").is_file()):
            trainer.fit()
            (marker / ".trained").touch()
        out = self.fold_prediction_dir(fold)
        if predict and not (self.config.skip_finished and (out / ".done").is_file()):
            spec = self.spec(fold)
            trainer.predict([spec.images[s] for s in spec.stems("test")], out)
            (out / ".done").touch()
        return out

    def merge(self) -> OOFReport:
        a = self.assignment()
        dirs = {f: self.fold_prediction_dir(f) for f in range(self.config.n_folds)}
        return merge_oof_predictions(dirs, self.paths.resolve(self.config.predictions_dir) / self.config.name, a.fold_of)

    def evaluate(self, preds_dir: Optional[Path] = None) -> EvaluationResult:
        """Score the merged out-of-fold predictions; adds a per-fold decision table to the summary."""
        if self.config.evaluation is None:
            raise ConfigError(f"{self.config.name}: no evaluation section")
        a = self.assignment()
        ev = SegmentationEvaluator.from_config(self.config.evaluation, self.paths)
        if ev.pairs is not None:
            keep = ev.pairs.mlo_stem.isin(a.fold_of) & ev.pairs.cc_stem.isin(a.fold_of)
            ev.pairs = ev.pairs[keep].reset_index(drop=True)
            ev.pairs["fold"] = ev.pairs.mlo_stem.map(a.fold_of).to_numpy()
        preds = preds_dir or self.paths.resolve(self.config.predictions_dir) / self.config.name
        result = ev.evaluate(preds, sorted(a.fold_of))
        result.views["fold"] = result.views.stem.map(a.fold_of)
        if len(result.pairs):
            per_fold = []
            for f, g in result.pairs.groupby("fold"):
                good = ref_is_good(g.truth)
                per_fold.append({"fold": int(f), "n": int(len(g)),
                                 "flat_10mm": FLAT_RULE.matrix(g.delta_mm, good)["accuracy"],
                                 "tolerance_1mm": TOLERANCE_RULE.matrix(g.delta_mm, good)["accuracy"]})
            result.summary["per_fold"] = per_fold
        result.save(self.paths.resolve(self.config.results_dir), self.config.name)
        return result

    def run(self, train: bool = True, predict: bool = True, evaluate: bool = True) -> CVReport:
        folds = self.folds_to_run()
        self.prepare()
        for f in folds:
            self.run_fold(f, train, predict)
        report = CVReport(folds)
        if predict and len(folds) == self.config.n_folds:
            report.oof = self.merge()
            if evaluate and self.config.evaluation is not None:
                report.evaluation = self.evaluate()
        (self.work_dir / "cv_report.json").write_text(json.dumps(
            {"folds_run": folds, "merged": report.oof.n_images if report.oof else None}, indent=2))
        return report


__all__ = ["CVConfig", "CVDataConfig", "CVReport", "CrossValidationRunner", "FoldTableConfig", "GroupTableConfig",
           "load_cv_config"]
