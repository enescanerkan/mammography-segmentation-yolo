"""``nnunet`` family: nnU-Net v2 (residual-encoder M preset) driven through its command-line tools.

:meth:`NnUNetTrainer.prepare` converts the semantic dataset to the raw nnU-Net layout, plans with the ResEnc-M
planner, derives the project plans (the preset's network with a full-image ``patch_size`` and its ``batch_size``),
preprocesses and installs the mammography trainer variants. :meth:`fit` runs ``nnUNetv2_train``; :meth:`predict`
runs ``nnUNetv2_predict``, which already writes ``<stem>.png`` label maps in the common format.

The three nnU-Net locations (raw, preprocessed, results) come from the paths config and are exported as the
``nnUNet_raw`` / ``nnUNet_preprocessed`` / ``nnUNet_results`` environment variables of every command.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from breast_seg.datasets.formats import NnUNetExporter, link_or_copy, semantic_paths
from breast_seg.training.base import SplitSpec, Trainer, TrainResult, register_trainer
from breast_seg.training.common import CommandRunner, dice_report, stems_of
from breast_seg.training.config import NnUNetTrainConfig

VARIANTS_FILE = Path(__file__).with_name("nnunet_variants.py")


def install_trainer_variants(target_dir: Optional[Path] = None) -> Path:
    """Copy the mammography trainer classes into nnU-Net's ``variants`` package (where it looks for trainers)."""
    if target_dir is None:
        import nnunetv2

        target_dir = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / "variants"
    target = Path(target_dir) / "nnUNetTrainerMammo.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(VARIANTS_FILE, target)
    return target


def derive_plans(src: Path, dst: Path, plans_name: str, patch_size: Optional[Sequence[int]],
                 batch_size: Optional[int], configuration: str = "2d") -> Dict[str, Any]:
    """Copy a plans JSON under a new ``plans_name`` with a different patch / batch size; the network topology and
    the preprocessing identifier stay those of the source plans."""
    plans = json.loads(Path(src).read_text())
    plans["plans_name"] = plans_name
    conf = plans["configurations"][configuration]
    if patch_size is not None:
        conf["patch_size"] = [int(v) for v in patch_size]
    if batch_size is not None:
        conf["batch_size"] = int(batch_size)
    Path(dst).write_text(json.dumps(plans, indent=4))
    return plans


@register_trainer("nnunet")
class NnUNetTrainer(Trainer):
    config_class = NnUNetTrainConfig
    config: NnUNetTrainConfig

    def __init__(self, config: NnUNetTrainConfig, paths=None, runner: Optional[CommandRunner] = None):
        super().__init__(config, paths)
        self.runner = runner or CommandRunner()

    # ── locations ──────────────────────────────────────────────────────────────────────────────────────────

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.config.nnunet_raw)

    @property
    def preprocessed_dir(self) -> Path:
        return self.resolve(self.config.nnunet_preprocessed)

    @property
    def results_dir(self) -> Path:
        return self.resolve(self.config.nnunet_results)

    @property
    def exporter(self) -> NnUNetExporter:
        c = self.config
        return NnUNetExporter(self.raw_dir, self.preprocessed_dir, c.dataset_id, c.dataset_name, c.data.description)

    @property
    def dataset_folder(self) -> str:
        return self.exporter.dataset_folder

    @property
    def model_dir(self) -> Path:
        c = self.config
        return self.results_dir / self.dataset_folder / f"{c.trainer}__{c.plans}__{c.configuration}"

    @property
    def fold_dir(self) -> Path:
        return self.model_dir / f"fold_{self.config.fold}"

    def env(self) -> Dict[str, str]:
        return {"nnUNet_raw": str(self.raw_dir), "nnUNet_preprocessed": str(self.preprocessed_dir),
                "nnUNet_results": str(self.results_dir)}

    def run(self, *cmd: Any, log: Optional[str] = None) -> int:
        return self.runner([str(c) for c in cmd], self.env(), self.run_dir / log if log else None)

    # ── steps ──────────────────────────────────────────────────────────────────────────────────────────────

    def convert(self) -> Path:
        """Semantic dataset -> raw nnU-Net dataset (fold 0 = its own train/val split, ``cv_folds`` more by group)."""
        d = self.config.data
        root = self.resolve(d.dataset_dir)
        index = semantic_paths(root, pd.read_csv(root / d.index, dtype={"stem": str}))
        groups = dict(zip(index.stem, index["group"].fillna(""))) if "group" in index else None
        return self.exporter.export(index, d.cv_folds, d.cv_seed, groups)

    def plan(self) -> None:
        c = self.config
        did = str(c.dataset_id)
        pre = self.preprocessed_dir / self.dataset_folder
        if c.transfer_plans_from is not None:
            self.run("nnUNetv2_extract_fingerprint", "-d", did, "-np", c.num_processes, log="nnunet_fingerprint.log")
            self.run("nnUNetv2_move_plans_between_datasets", "-s", c.transfer_plans_from, "-t", did,
                     "-sp", c.plans, "-tp", c.plans, log="nnunet_plan.log")
        else:
            cmd = ["nnUNetv2_plan_and_preprocess", "-d", did, "-c", c.configuration, "-np", c.num_processes]
            if c.verify_dataset_integrity:
                cmd.append("--verify_dataset_integrity")
            self.run(*cmd, log="nnunet_plan.log")
            self.run("nnUNetv2_plan_experiment", "-d", did, "-pl", c.planner, log="nnunet_plan.log")
            if c.plans != c.base_plans:
                derive_plans(pre / f"{c.base_plans}.json", pre / f"{c.plans}.json", c.plans, c.patch_size, c.batch_size,
                             c.configuration)
        self.run("nnUNetv2_preprocess", "-d", did, "-plans_name", c.plans, "-c", c.configuration,
                 "-np", c.num_processes, log="nnunet_preprocess.log")

    def prepare(self) -> Path:
        """Convert (unless the raw dataset exists and ``reuse_raw``), plan, preprocess, install the trainer."""
        self.save_config()
        ds = self.raw_dir / self.dataset_folder
        if not (self.config.data.reuse_raw and (ds / "dataset.json").is_file()):
            self.convert()
        self.plan()
        if self.config.install_trainer and not self.runner.dry_run:
            install_trainer_variants()
        return ds

    def fit(self) -> TrainResult:
        c = self.config
        cmd = ["nnUNetv2_train", c.dataset_id, c.configuration, c.fold, "-p", c.plans, "-tr", c.trainer]
        if c.npz:
            cmd.append("--npz")
        if (self.fold_dir / "checkpoint_latest.pth").is_file() and not (self.fold_dir / "checkpoint_final.pth").is_file():
            cmd.append("--c")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run(*cmd, log=f"nnunet_train_fold{c.fold}.log")
        return TrainResult(self.fold_dir, self.best_weights())

    def best_weights(self) -> Path:
        return self.fold_dir / self.config.checkpoint

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None,
                input_dir: Optional[Path] = None) -> List[Path]:
        """``nnUNetv2_predict`` on ``images`` (linked as ``<stem>_0000.png`` into a temporary folder) or directly on
        an nnU-Net ``input_dir``. ``weights`` may name another checkpoint file of the fold."""
        c = self.config
        out_dir = Path(out_dir)
        tmp = None
        if input_dir is None:
            tmp = Path(tempfile.mkdtemp(prefix="nnunet_in_"))
            for p in images:
                link_or_copy(p, tmp / f"{Path(p).stem}_0000.png")
            input_dir = tmp
        cmd = ["nnUNetv2_predict", "-i", input_dir, "-o", out_dir, "-d", c.dataset_id, "-c", c.configuration,
               "-f", c.fold, "-p", c.plans, "-tr", c.trainer, "-chk", Path(weights).name if weights else c.checkpoint,
               "--disable_progress_bar"]
        if c.disable_tta:
            cmd.append("--disable_tta")
        try:
            self.run(*cmd, log="nnunet_predict.log")
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
        return [out_dir / f"{Path(p).stem}.png" for p in images]

    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        """Predict the semantic dataset's ``split`` and report mean per-image Dice by view."""
        from breast_seg.training.torch_data import read_grey

        d = self.config.data
        root = self.resolve(d.dataset_dir)
        index = semantic_paths(root, pd.read_csv(root / d.index, dtype={"stem": str}))
        rows = index[index.split == split]
        out = self.run_dir / "predictions" / split
        self.predict([Path(p) for p in rows.image], out, weights)
        masks = dict(zip(rows.stem, rows["mask"]))
        return {"dice": dice_report(out, stems_of(rows.image), lambda s: read_grey(masks[s]) if masks.get(s) else None)}

    def export(self, dest: Optional[Path] = None, weights: Optional[Path] = None) -> Path:
        """``nnUNetv2_export_model_to_zip`` of this trainer / plans / fold (default ``<export.dir>/<filename>``)."""
        c = self.config
        if dest is None:
            dest = self.resolve(c.export.dir) / (c.export.filename or f"{c.name}.zip")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.run("nnUNetv2_export_model_to_zip", "-d", c.dataset_id, "-o", dest, "-c", c.configuration,
                 "-tr", c.trainer, "-p", c.plans, "-f", c.fold, "-chk", Path(weights).name if weights else c.checkpoint,
                 log="nnunet_export.log")
        return Path(dest)

    # ── cross-validation ───────────────────────────────────────────────────────────────────────────────────

    def prepare_folds(self, specs: Sequence[SplitSpec]) -> Path:
        """One raw dataset for every fold: all labelled images in ``imagesTr``, ``splits_final.json`` with one
        train/val entry per fold, and ``imagesTest_fold<i>`` holding fold ``i``'s test images (unlabelled ones
        included). Then plan and preprocess once."""
        self.save_config()
        images: Dict[str, Path] = {}
        masks: Dict[str, Path] = {}
        for spec in specs:
            images.update(spec.images)
            masks.update(spec.masks)
        rows = [{"stem": s, "split": "train", "image": str(images[s]), "mask": str(masks.get(s, ""))} for s in sorted(images)]
        index = pd.DataFrame(rows, columns=["stem", "split", "image", "mask"])
        splits = [{"train": spec.stems("train"), "val": spec.stems("val")} for spec in specs]
        tests = {f"imagesTest_fold{i}": spec.stems("test") for i, spec in enumerate(specs)}
        ds = self.exporter.export(index, splits=splits, extra_test_dirs=tests)
        self.plan()
        if self.config.install_trainer and not self.runner.dry_run:
            install_trainer_variants()
        return ds

    def use_split(self, spec: SplitSpec, work_dir: Path) -> NnUNetTrainer:
        """The trainer of fold ``spec.fold`` of the dataset made by :meth:`prepare_folds`."""
        return type(self)(dataclasses.replace(self.config, fold=spec.fold), self.paths, self.runner)

    def fold_test_dir(self, fold: int) -> Path:
        return self.raw_dir / self.dataset_folder / f"imagesTest_fold{fold}"
