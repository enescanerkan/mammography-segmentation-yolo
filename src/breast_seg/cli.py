"""``breast-seg`` command line.

    breast-seg serve [--config configs/app.yaml] [--host 0.0.0.0] [--port 8000] [--device cpu] [--weights DIR]
    breast-seg models [--config configs/app.yaml] [--weights DIR]
    breast-seg build-dataset configs/datasets/seg_v5.yaml
    breast-seg train configs/train/yolo26m_seg_v5.yaml [--set epochs=10] [--prepare-only]
    breast-seg cv configs/cv/cv10_yolo26m.yaml [--fold 3] [--prepare-only]
    breast-seg evaluate --config configs/cv/cv10_yolo26m.yaml --preds DIR --out DIR --name NAME
    breast-seg compare configs/compare.yaml

Every YAML-driven command takes ``--set key.sub=value`` (repeatable) and ``--paths`` (machine paths file, default
``$BREAST_SEG_PATHS`` or ``configs/paths.yaml``). Heavy dependencies are imported inside each command, so ``--help``
works with the core install.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

from breast_seg import __version__


def _paths(args: argparse.Namespace):
    from breast_seg.config import PathsConfig

    return PathsConfig.load(args.paths)


def _print_json(obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    print(json.dumps(obj, indent=2, default=str))


# ── application ──────────────────────────────────────────────────────────────
def _app_config(args: argparse.Namespace):
    from breast_seg.apps.positioning.config import load_app_config

    overrides = list(args.set or [])
    if getattr(args, "weights", None):
        overrides.append(f"models.weights_root={Path(args.weights).resolve()}")
    if getattr(args, "device", None):
        overrides.append(f"models.device={args.device}")
    if getattr(args, "data_dir", None):
        overrides.append(f"storage.data_dir={Path(args.data_dir).resolve()}")
    return load_app_config(args.config, overrides)


def cmd_serve(args: argparse.Namespace) -> int:
    from breast_seg.apps.positioning.api import serve

    serve(_app_config(args), host=args.host, port=args.port)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from breast_seg.registry import Registry

    cfg = _app_config(args)
    registry = Registry.load(cfg.models.registry, weights_root=cfg.models.weights_root)
    rows = [(m.id, m.task, m.view, "yes" if m.enabled else "no", "yes" if Path(m.weights_path).is_file() else "MISSING",
             m.weights) for m in registry.models]
    head = ("id", "task", "view", "enabled", "weights", "path (relative to the weights root)")
    widths = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(str(v).ljust(w) for v, w in zip(r, widths)).rstrip())
    return 0 if all(Path(m.weights_path).is_file() for m in registry.models if m.enabled) else 1


# ── data and training ────────────────────────────────────────────────────────
def cmd_build_dataset(args: argparse.Namespace) -> int:
    from breast_seg.datasets import DatasetBuilder, load_dataset_config

    report = DatasetBuilder(load_dataset_config(args.config, args.set), paths=_paths(args)).build(strict=not args.no_strict)
    print(f"dataset written to {report.output_dir}")
    for split, counter in report.counts.items():
        print(f"  {split}: {dict(counter)}")
    if report.skipped:
        print(f"  skipped: {dict(report.skipped)}")
    for p in report.problems:
        print(f"  problem: {p}", file=sys.stderr)
    return 0 if report.ok else 1


def cmd_train(args: argparse.Namespace) -> int:
    from breast_seg.training import build_trainer

    trainer = build_trainer(args.config, paths=_paths(args), overrides=args.set)
    if args.prepare_only:
        print(f"prepared {trainer.prepare()}")
        return 0
    _print_json(trainer.fit())
    return 0


def cmd_cv(args: argparse.Namespace) -> int:
    from breast_seg.training.cv import CrossValidationRunner

    runner = CrossValidationRunner.from_yaml(args.config, overrides=args.set, paths=_paths(args))
    if args.prepare_only:
        for t in runner.prepare():
            print(f"prepared {t.run_dir}")
        return 0
    if args.fold is not None:
        preds = runner.run_fold(args.fold, train=not args.no_train, predict=not args.no_predict)
        print(f"fold {args.fold} predictions in {preds}")
        return 0
    report = runner.run(train=not args.no_train, predict=not args.no_predict, evaluate=not args.no_evaluate)
    print(f"folds run: {report.folds_run}")
    if report.evaluation is not None and args.out:
        for kind, path in report.evaluation.save(args.out, args.name).items():
            print(f"{kind}: {path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from breast_seg.evaluation import SegmentationEvaluator, load_evaluation_config

    cfg = load_evaluation_config(args.config, args.set)
    result = SegmentationEvaluator.from_config(cfg, paths=_paths(args)).evaluate(Path(args.preds))
    for kind, path in result.save(args.out, args.name).items():
        print(f"{kind}: {path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    import yaml

    from breast_seg.compare import CompareConfig, run_compare
    from breast_seg.config import apply_overrides

    raw = yaml.safe_load(Path(args.config).read_text()) or {}
    raw = apply_overrides(raw.get("compare", raw), args.set)
    table = run_compare(CompareConfig(**raw), paths=_paths(args))
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
                            ("models", cmd_models, "list the model registry and check that the weights exist")):
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
    s.add_argument("--no-strict", action="store_true", help="write the dataset even when the audit finds problems")
    s = yaml_cmd("train", cmd_train, "train one model from a training config")
    s.add_argument("--prepare-only", action="store_true", help="write the run directory and data files, do not train")
    s = yaml_cmd("cv", cmd_cv, "k-fold cross-validation: train, predict out-of-fold, merge and evaluate")
    s.add_argument("--fold", type=int, help="run only this fold (train + predict)")
    s.add_argument("--prepare-only", action="store_true", help="prepare every fold, do not train")
    s.add_argument("--no-train", action="store_true", help="reuse the trained fold weights")
    s.add_argument("--no-predict", action="store_true", help="reuse the fold predictions")
    s.add_argument("--no-evaluate", action="store_true", help="skip the evaluation of the merged predictions")
    s.add_argument("--out", help="write the evaluation tables here")
    s.add_argument("--name", default="cv", help="file name prefix of the evaluation tables")
    s = yaml_cmd("evaluate", cmd_evaluate, "evaluate predicted masks against the ground truth (Dice, landmarks, PNL rule)",
                 config_positional=False)
    s.add_argument("--config", required=True, help="YAML with an `evaluation:` section (e.g. a cv config)")
    s.add_argument("--preds", required=True, help="directory of predicted masks")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--name", default="eval", help="file name prefix")
    yaml_cmd("compare", cmd_compare, "ground truth vs pose landmarks vs segmentation on the compare set")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())


__all__: List[str] = ["build_parser", "main"]
