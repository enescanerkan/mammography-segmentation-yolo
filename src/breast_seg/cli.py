"""``breast-seg`` command line.

    breast-seg serve [--config configs/app.yaml] [--host 127.0.0.1] [--port 8000] [--device cpu] [--weights DIR]
    breast-seg models [--config configs/app.yaml] [--weights DIR]
    breast-seg build-dataset configs/datasets/seg_v5.yaml
    breast-seg train configs/train/yolo26m_seg_v5.yaml [--set epochs=10] [--prepare-only]
    breast-seg export configs/train/yolo26m_seg_v5.yaml [--dest weights/segmentation/yolo26m_seg_v5.pt]
    breast-seg cv configs/cv/cv10_yolo26m.yaml [--fold 3] [--prepare-only] [--no-train --no-predict]
    breast-seg evaluate --config configs/cv/cv10_yolo26m.yaml --preds DIR --out DIR --name NAME
    breast-seg compare configs/compare.yaml          (start from configs/compare.example.yaml)

Every YAML-driven command takes ``--set key.sub=value`` (repeatable; for ``evaluate`` and ``compare`` the keys are
those of the ``evaluation:`` / ``compare:`` section) and ``--paths`` (machine paths file, default
``$BREAST_SEG_PATHS`` or ``configs/paths.yaml``). Heavy dependencies are imported inside each command, so ``--help``
works with the core install. Configuration mistakes end with a one-line message and exit code 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

from breast_seg import __version__

# Roles the review app cannot run without (breast_seg.inference.engine.REQUIRED_ROLES): task and view per role.
_REQUIRED_ROLES = {"pose_mlo": ("landmark", "MLO"), "pose_cc": ("landmark", "CC"), "segmentation": ("segmentation", None)}


def _paths(args: argparse.Namespace):
    from breast_seg.config import PathsConfig

    return PathsConfig.load(args.paths)


def _print_json(obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    print(json.dumps(obj, indent=2, default=str))


def _section(path: str, key: str, overrides):
    """The ``key:`` section of a YAML file (or the whole file when it has none), ``base:`` resolved, overrides applied."""
    from breast_seg.config import apply_overrides, read_yaml

    data = read_yaml(path)
    section = data.get(key, data) if isinstance(data, dict) else data
    return apply_overrides(section, overrides)


# ── application ──────────────────────────────────────────────────────────────
def _app_config(args: argparse.Namespace):
    """App config from ``--config`` + ``--set``; the path / device flags are set as typed values, never re-parsed."""
    from breast_seg.apps.positioning.config import load_app_config

    cfg = load_app_config(args.config, list(args.set or []))
    if getattr(args, "weights", None):
        cfg.models.weights_root = str(Path(args.weights).expanduser().resolve())
    if getattr(args, "device", None):
        cfg.models.device = str(args.device)
    if getattr(args, "data_dir", None):
        cfg.storage.data_dir = str(Path(args.data_dir).expanduser().resolve())
    cfg.validate()
    return cfg


def cmd_serve(args: argparse.Namespace) -> int:
    from breast_seg.apps.positioning.api import serve

    serve(_app_config(args), host=args.host, port=args.port)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from breast_seg.registry import Registry

    cfg = _app_config(args)
    registry = Registry.load(cfg.registry_path, weights_root=cfg.weights_root)  # resolved exactly as `serve` does
    rows = [(m.id, m.task, m.view, "yes" if m.enabled else "no", "yes" if m.available else "MISSING", m.weights)
            for m in registry.models]
    head = ("id", "task", "view", "enabled", "weights", "path (relative to the weights root)")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())
    print(f"\nweights root: {registry.weights_root}")
    missing = []
    for role, (task, view) in _REQUIRED_ROLES.items():
        spec = registry.select(task, view)
        if spec is None:
            missing.append(f"{role}: no enabled {task} model" + (f" for {view}" if view else ""))
        elif not spec.available:
            missing.append(f"{role}: {spec.id} has no weights file ({spec.weights})")
    for m in missing:
        print(f"required for the app — {m}", file=sys.stderr)
    if not missing:
        print("the review app has every model it needs (optional models may still be missing)")
    return 1 if missing else 0


# ── data and training ────────────────────────────────────────────────────────
def cmd_build_dataset(args: argparse.Namespace) -> int:
    from breast_seg.datasets import DatasetAuditError, DatasetBuilder, load_dataset_config

    builder = DatasetBuilder(load_dataset_config(args.config, args.set), paths=_paths(args))
    try:
        report = builder.build(strict=not args.no_strict)
    except DatasetAuditError as exc:
        for p in exc.problems:
            print(f"problem: {p}", file=sys.stderr)
        print(f"dataset audit failed with {len(exc.problems)} problem(s); data.yaml and the exports were not written",
              file=sys.stderr)
        return 1
    print(f"dataset written to {report.output_dir}")
    for split, counter in report.counts.items():
        print(f"  {split}: {dict(counter)}")
    if report.skipped:
        print(f"  skipped: {dict(report.skipped)}")
    for p in report.problems:
        print(f"problem: {p}", file=sys.stderr)
    if not report.ok:
        print("the audit found problems: images and labels are written, data.yaml and the exports are not",
              file=sys.stderr)
    return 0 if report.ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    from breast_seg.training import build_trainer

    trainer = build_trainer(args.config, paths=_paths(args), overrides=args.set)
    if args.prepare_only:
        print(f"prepared {trainer.prepare()}")
        return 0
    if not type(trainer).prepares_in_fit:
        trainer.prepare()
    _print_json(trainer.fit())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from breast_seg.training import build_trainer

    trainer = build_trainer(args.config, paths=_paths(args), overrides=args.set)
    print(trainer.export(Path(args.dest) if args.dest else None, Path(args.weights) if args.weights else None))
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    from breast_seg.training.cv import CrossValidationRunner

    runner = CrossValidationRunner.from_yaml(args.config, overrides=args.set, paths=_paths(args))
    train, predict, evaluate = not args.no_train, not args.no_predict, not args.no_evaluate
    if args.prepare_only:
        for t in runner.prepare():
            print(f"prepared {t.run_dir}")
        return 0
    if args.fold is not None:
        if train:
            runner.prepare()  # nnU-Net plans and preprocesses the shared dataset here, not in fit()
        print(f"fold {args.fold} predictions in {runner.run_fold(args.fold, train=train, predict=predict)}")
        return 0
    evaluation = None
    if train or predict:
        report = runner.run(train=train, predict=predict, evaluate=evaluate)
        print(f"folds run: {report.folds_run}")
        evaluation = report.evaluation
    if not predict:  # reuse the stored fold predictions: merge them and score the merged set
        oof = runner.merge()
        print(f"merged out-of-fold predictions: {oof.n_images} images")
        if evaluate and runner.config.evaluation is not None:
            evaluation = runner.evaluate()
    if evaluation is not None and args.out:
        for kind, path in evaluation.save(args.out, args.name).items():
            print(f"{kind}: {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from breast_seg.config import from_dict
    from breast_seg.evaluation import EvaluationConfig, SegmentationEvaluator

    cfg = from_dict(EvaluationConfig, _section(args.config, "evaluation", args.set), "evaluation")
    result = SegmentationEvaluator.from_config(cfg, paths=_paths(args)).evaluate(Path(args.preds))
    for kind, path in result.save(args.out, args.name).items():
        print(f"{kind}: {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from breast_seg.compare import CompareConfig, run_compare
    from breast_seg.config import from_dict

    cfg = from_dict(CompareConfig, _section(args.config, "compare", args.set), "compare")
    table = run_compare(cfg, paths=_paths(args))
    print(f"{len(table)} pairs compared")
    return 0


# ── parser ───────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="breast-seg", description="Mammography segmentation, landmarks and positioning review.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def yaml_cmd(name: str, fn, help_: str, config_positional: bool = True) -> argparse.ArgumentParser:
        s = sub.add_parser(name, help=help_, description=help_)
        if config_positional:
            s.add_argument("config", help="YAML config")
        s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
        s.add_argument("--paths", help="machine paths file (default: $BREAST_SEG_PATHS or configs/paths.yaml)")
        s.set_defaults(func=fn)
        return s

    for name, fn, help_ in (("serve", cmd_serve, "run the positioning review web application"),
                            ("models", cmd_models, "list the model registry and check that the app's weights exist")):
        s = sub.add_parser(name, help=help_, description=help_)
        s.add_argument("--config", help="application config (default: built-in defaults; see configs/app.example.yaml)")
        s.add_argument("--weights", help="weights root (overrides models.weights_root and $BREAST_SEG_WEIGHTS)")
        s.add_argument("--device", help='"cpu", or a GPU such as "0" or "cuda:1"')
        s.add_argument("--data-dir", help="where studies are kept (renders and labels; DICOMs are never stored)")
        s.add_argument("--set", action="append", metavar="KEY=VALUE", help="override an application config value")
        if name == "serve":
            s.add_argument("--host", help="listen address (default from the config: 127.0.0.1)")
            s.add_argument("--port", type=int, help="port (default from the config: 8000)")
        s.set_defaults(func=fn)

    s = yaml_cmd("build-dataset", cmd_build_dataset, "export a training dataset (YOLO / semantic / nnU-Net) from a config")
    s.add_argument("--no-strict", action="store_true",
                   help="report audit problems and exit 1 instead of stopping with an error (images and labels are "
                        "written either way; data.yaml and the exports only when the audit passes)")
    s = yaml_cmd("train", cmd_train, "train one model from a training config")
    s.add_argument("--prepare-only", action="store_true", help="write the run directory and data files, do not train")
    s = yaml_cmd("export", cmd_export, "copy a trained model's weights to its release name (export.dir / export.filename)")
    s.add_argument("--weights", help="weights to export (default: the run's selected checkpoint)")
    s.add_argument("--dest", help="destination file (default from the config's export section)")
    s = yaml_cmd("cv", cmd_cv, "k-fold cross-validation: train, predict out-of-fold, merge and evaluate")
    s.add_argument("--fold", type=int, help="run only this fold (train + predict)")
    s.add_argument("--prepare-only", action="store_true", help="prepare every fold, do not train")
    s.add_argument("--no-train", action="store_true", help="reuse the trained fold weights")
    s.add_argument("--no-predict", action="store_true", help="reuse the stored fold predictions: merge and evaluate them")
    s.add_argument("--no-evaluate", action="store_true", help="skip the evaluation of the merged predictions")
    s.add_argument("--out", help="also write the evaluation tables here")
    s.add_argument("--name", default="cv", help="file name prefix of the evaluation tables")
    s = yaml_cmd("evaluate", cmd_evaluate, "evaluate predicted masks against the ground truth (Dice, landmarks, PNL rule)",
                 config_positional=False)
    s.add_argument("--config", required=True, help="YAML with an `evaluation:` section (e.g. a cv config), or only that")
    s.add_argument("--preds", required=True, help="directory of predicted masks")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--name", default="eval", help="file name prefix")
    yaml_cmd("compare", cmd_compare, "ground truth vs pose landmarks vs segmentation on the compare set")
    return p


def _known_errors() -> Tuple[type, ...]:
    """Errors that mean "your input is wrong", reported as one line instead of a traceback."""
    from breast_seg.apps.positioning.config import ConfigError as AppConfigError
    from breast_seg.config import ConfigError
    from breast_seg.registry import RegistryError

    return ConfigError, AppConfigError, RegistryError, FileNotFoundError


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        if os.environ.get("BREAST_SEG_DEBUG") or not isinstance(exc, _known_errors()):
            raise
        print(f"breast-seg {args.command}: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
