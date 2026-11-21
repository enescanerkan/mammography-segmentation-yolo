"""End-to-end runtime of the released pipelines: one interleaved session per device (Table 11 of the paper).

All pipelines are timed in ONE process on the same pairs, from the two DICOM files of a pair to its adequacy
decision, one pair at a time. Every pair runs every configuration; the order of the configurations rotates from
pair to pair (fixed seed, :mod:`breast_seg.benchmark.protocol`), so whatever else the machine does hits all of them
alike. After ``warmup`` pairs, all pairs are passed ``passes`` times. The process is pinned to the ``threads``
least-loaded physical cores with as many threads; a GPU session checks its card for foreign compute processes at
the start, every ``card_check_every`` pairs and at the end and stops (exit code 4) if it finds one.

Files are read once before the session (warm page cache) and again before every pair; every configuration reads
and decodes its own DICOMs. Outputs, both under ``--out``:

* ``<name>.csv``: one row per pair x configuration (stage times, PNL, D_CC, ΔD, decision, nipple sources, load);
* ``<name>.json``: the summary (:mod:`breast_seg.benchmark.summary`): seconds per pair (median [IQR]), pairs per
  minute, every pass on its own, the spread between passes, the disturbed passes, model sizes, and the checks of
  the outputs against stored predictions (``--stored``) and against a reference ΔD (``reference_delta_mm``).

Only the standard library, the YAML loader and the protocol module are imported here at module level; numpy,
OpenCV and torch are imported after the process is pinned.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from breast_seg.benchmark import protocol
from breast_seg.benchmark.protocol import CRAUNET, UNBATCHED, InvalidMeasurement, Pair, Setting
from breast_seg.config import ConfigError, PathsConfig, apply_overrides, from_dict, read_yaml

DEFAULT_SEED = 20261120
#: Table 11: every pipeline in its evaluated setting and, where it has one, without flipping.
TABLE11 = ["craunet", "segformer_b3_tta", "segformer_b3_no_tta", "nnunet_tta", "nnunet_no_tta", "yolo26m",
           "unet_tta", "unet_no_tta"]
DICOM_SUFFIXES = (".dcm", ".dicom")
PROGRESS_EVERY = 10
ERRORS_BEFORE_DROP = 3


@dataclass
class PipelineEntry:
    """A pipeline of this package: its training config (weights by the config's conventions) or explicit weights
    (``weights``; nnU-Net: the trained model folder)."""

    train_config: str = ""
    weights: str = ""

    def __post_init__(self) -> None:
        if not self.train_config:
            raise ConfigError("train_config is required")


@dataclass
class CraunetEntry:
    """The external CRAUNet baseline (:mod:`breast_seg.benchmark.craunet`); skipped while ``code_dir`` is empty."""

    code_dir: str = ""
    weights_dir: str = ""
    fold: int = 0
    mlo_weights: str = ""
    cc_weights: str = ""


@dataclass
class RuntimeConfig:
    pipelines: Dict[str, PipelineEntry] = field(default_factory=dict)
    craunet: CraunetEntry = field(default_factory=CraunetEntry)
    configurations: List[str] = field(default_factory=lambda: list(TABLE11))
    passes: int = 2
    warmup: int = 5
    seed: int = DEFAULT_SEED
    threads: int = 8
    core_sampling_s: float = 5.0
    card_check_every: int = 20
    allowed_pids: List[int] = field(default_factory=list)
    batched: bool = True
    flops: bool = True

    def __post_init__(self) -> None:
        if CRAUNET in self.pipelines:
            raise ConfigError("'craunet' is configured under craunet:, not under pipelines:")
        if self.passes < 1 or self.warmup < 0 or self.threads < 1 or self.card_check_every < 1:
            raise ConfigError("passes, threads and card_check_every must be >= 1, warmup >= 0")
        if len(set(self.configurations)) != len(self.configurations):
            raise ConfigError("configurations: duplicate names")


def load_runtime_config(path: Path, overrides: Optional[Sequence[str]] = None) -> RuntimeConfig:
    data = apply_overrides(read_yaml(path), overrides)
    try:
        return from_dict(RuntimeConfig, data)
    except ConfigError as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


def _resolve(raw: str, base: Path, paths: PathsConfig) -> Path:
    p = Path(paths.expand(raw)).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def train_config_path(entry: PipelineEntry, config_dir: Path, paths: PathsConfig) -> Path:
    return _resolve(entry.train_config, config_dir, paths)


def families(cfg: RuntimeConfig, config_dir: Path, paths: PathsConfig) -> Dict[str, str]:
    """Pipeline -> model family, read from its training config (YAML only)."""
    out = {}
    for name, entry in cfg.pipelines.items():
        fam = read_yaml(train_config_path(entry, config_dir, paths)).get("family")
        if not fam:
            raise ConfigError(f"pipelines.{name}: its training config has no family")
        out[name] = str(fam)
    return out


def resolve_configurations(cfg: RuntimeConfig, fams: Mapping[str, str]) -> Tuple[Dict[str, Tuple[str, Setting]], Dict[str, str]]:
    """Requested configurations -> (pipeline, setting), and the skipped ones with the reason."""
    available: Dict[str, Tuple[str, Setting]] = {}
    for name, fam in [*fams.items(), (CRAUNET, CRAUNET)]:
        for conf, setting in protocol.pipeline_settings(name, fam, cfg.batched).items():
            available[conf] = (name, setting)
    out, skipped = {}, {}
    for conf in cfg.configurations:
        if conf not in available:
            raise ConfigError(f"unknown configuration {conf!r} (available: {', '.join(sorted(available))})")
        if available[conf][0] == CRAUNET and not cfg.craunet.code_dir:
            skipped[conf] = "CRAUNet is not configured (give its released code with --craunet-code and its weights)"
            continue
        out[conf] = available[conf]
    return out, skipped


# ── pairs ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _dicom_index(root: Path) -> Dict[str, Path]:
    """``<name>`` -> path of every ``<name>.dcm`` / ``<name>.dicom`` under ``root``."""
    out: Dict[str, Path] = {}
    for d, _, files in os.walk(root):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in DICOM_SUFFIXES:
                out.setdefault(stem, Path(d) / f)
    return out


def read_pairs(path: Path, dicom_root: Optional[Path] = None, first: int = 0) -> List[Pair]:
    """Pairs from a CSV with ``study, side`` and either ``mlo_dicom, cc_dicom`` (paths; relative ones against
    ``dicom_root`` or the CSV's folder) or ``mlo_sop, cc_sop`` (files named ``<SOPInstanceUID>.dcm|.dicom``
    anywhere under ``dicom_root``). An optional ``reference_delta_mm`` column enables the accuracy check."""
    path = Path(path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if first:
        rows = rows[:first]
    if not rows:
        raise ConfigError(f"{path.name}: no pairs")
    cols = set(rows[0])
    if not {"study", "side"} <= cols:
        raise ConfigError(f"{path.name}: needs the columns study and side")
    if {"mlo_dicom", "cc_dicom"} <= cols:
        base = Path(dicom_root) if dicom_root else path.parent
        find = lambda r, v: (Path(r[f"{v}_dicom"]) if Path(r[f"{v}_dicom"]).is_absolute()  # noqa: E731
                             else base / r[f"{v}_dicom"])
    elif {"mlo_sop", "cc_sop"} <= cols:
        if not dicom_root:
            raise ConfigError(f"{path.name}: SOP Instance UIDs need --dicom-root")
        index = _dicom_index(Path(dicom_root))
        find = lambda r, v: index.get(r[f"{v}_sop"], Path(dicom_root) / f"{r[f'{v}_sop']}.dcm")  # noqa: E731
    else:
        raise ConfigError(f"{path.name}: needs mlo_dicom / cc_dicom or mlo_sop / cc_sop columns")
    pairs, missing = [], []
    for i, r in enumerate(rows):
        mlo, cc = find(r, "mlo"), find(r, "cc")
        missing += [str(p) for p in (mlo, cc) if not p.is_file()]
        side = str(r["side"]).strip().upper()
        if side not in ("L", "R"):
            raise ConfigError(f"{path.name} row {i + 1}: side must be L or R, not {r['side']!r}")
        ref = str(r.get("reference_delta_mm") or "").strip()
        pairs.append(Pair(i, str(r["study"]).strip(), side, str(mlo), str(cc),
                          float(ref) if ref and ref.lower() != "nan" else float("nan")))
    if missing:
        raise FileNotFoundError(f"{len(missing)} DICOM file(s) not found, e.g. {missing[:2]}")
    return pairs


def load_stored(path: Path, fold: Optional[int] = None):
    """Stored per-pair results of one model (``study`` or ``StudyInstanceUID``, ``side``, ``delta_mm`` or
    ``delta``) as a series indexed by (study, side). With a ``fold`` column and ``fold`` given, only that fold's
    rows (CRAUNet's per-case table lists every fold's test pairs, each from its own fold's model)."""
    import pandas as pd

    d = pd.read_csv(path, dtype={"study": str, "StudyInstanceUID": str, "side": str})
    study = next((c for c in ("study", "StudyInstanceUID") if c in d), None)
    delta = next((c for c in ("delta_mm", "delta") if c in d), None)
    if study is None or delta is None or "side" not in d:
        raise ConfigError(f"{Path(path).name}: needs study (or StudyInstanceUID), side and delta_mm (or delta)")
    if fold is not None and "fold" in d:
        d = d[d["fold"] == fold]
    return d.set_index([study, "side"])[delta].astype(float).rename_axis(["study", "side"])


# ── session ───────────────────────────────────────────────────────────────────────────────────────────────────

class Session:
    """One interleaved timing session. ``pipelines`` hold the (not yet loaded) models; ``device`` provides sync,
    memory and thread control (:class:`breast_seg.benchmark.pipelines.Device`)."""

    def __init__(self, cfg: RuntimeConfig, pairs: Sequence[Pair], configurations: Mapping[str, Tuple[str, Setting]],
                 pipelines: Mapping[str, Any], device: Any, out_dir: Path, name: str, setup: Mapping[str, Any],
                 guard: Optional[protocol.GpuGuard] = None, stored: Optional[Mapping[str, Tuple[Any, str]]] = None,
                 skipped: Optional[Mapping[str, str]] = None, command: str = "", overwrite: bool = False):
        self.cfg, self.pairs, self.configurations = cfg, list(pairs), dict(configurations)
        self.pipelines, self.device, self.guard = dict(pipelines), device, guard
        self.out_dir, self.name, self.setup = Path(out_dir), name, dict(setup)
        self.stored, self.skipped, self.command = dict(stored or {}), dict(skipped or {}), command
        if (self.out_dir / f"{name}.json").exists() and not overwrite:
            raise FileExistsError(f"{name}.json exists in the output folder; give another --name or --overwrite")

    # -- invalid sessions
    def _invalid(self, exc: InvalidMeasurement, rows: Optional[List[Dict[str, Any]]]) -> Path:
        print(f"MEASUREMENT INVALID, STOPPED: {exc.reason}: {exc.detail}", flush=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = self.out_dir / f"{self.name}_INVALID_{time.strftime('%m%d_%H%M%S')}"
        if rows:
            _write_csv(stem.with_suffix(".csv"), rows)
        stem.with_suffix(".json").write_text(json.dumps(
            {"valid": False, "reason": exc.reason, "detail": exc.detail, "card": self.guard.info if self.guard else None,
             "start": self.setup.get("start"), "end": protocol.machine_snapshot(), "command": self.command},
            indent=1, default=str))
        return stem.with_suffix(".json")

    def _time(self, pipe: Any, pair: Pair, setting: Setting):
        from breast_seg.benchmark.pipelines import read_dicom_file

        dev = self.device
        dev.sync()
        a = time.perf_counter()
        raw = (read_dicom_file(pair.mlo), read_dicom_file(pair.cc))
        b = time.perf_counter()
        prepared = pipe.prepare(raw, pair)
        c = time.perf_counter()
        out = pipe.infer(prepared, setting)
        dev.sync()
        d = time.perf_counter()
        pnl, dcc, sources = pipe.measure(out, prepared, pair)
        return (a, b, c, d), pnl, dcc, sources

    def _load(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        dev, gpu = self.device, self.device.gpu
        needed = list(dict.fromkeys(p for p, _ in self.configurations.values()))
        # one after the other, in a fixed order (it decides what each model's GPU memory counts)
        order = list(dict.fromkeys([p for p in (CRAUNET, *self.cfg.pipelines) if p in needed] + needed))
        models: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        for p in order:
            dev.sync()
            m0, t0 = dev.memory_allocated(), time.perf_counter()
            try:
                info = self.pipelines[p].load()
                dev.sync()
                info.update(model_load_s=time.perf_counter() - t0,
                            model_gpu_MB=(dev.memory_allocated() - m0) / 2 ** 20 if gpu else None,
                            weights_fp32_MB=info["params_M"] * 4)  # parameters x 4 bytes, comparable across pipelines
                models[p] = info
                print(f"loaded {p}: {info['params_M']:.1f} M parameters, {info['model_load_s']:.1f} s", flush=True)
            except Exception:  # noqa: BLE001 - a pipeline that cannot load is reported, the others still run
                tb = traceback.format_exc()
                errors.update({c: "load: " + tb for c, (q, _) in self.configurations.items() if q == p})
                print(f"LOAD FAILED {p}\n{tb}", flush=True)
        dev.set_threads()
        return models, errors

    def run(self) -> Tuple[Dict[str, Any], int]:
        """Run the session, write ``<name>.csv`` / ``<name>.json``; returns (summary, exit code 0 or 3).

        Raises :class:`InvalidMeasurement` (after writing ``<name>_INVALID_<time>.json``) when the GPU guard fires.
        """
        import numpy as np
        import pandas as pd

        from breast_seg.benchmark import summary as S
        from breast_seg.benchmark.pipelines import decide, describe

        warnings.simplefilter("ignore")
        cfg, dev, gpu = self.cfg, self.device, self.device.gpu
        if self.guard is not None:
            try:
                self.guard.check_start()
            except InvalidMeasurement as exc:
                self._invalid(exc, None)
                raise
            print("card", self.guard.info, flush=True)
        for p in self.pairs:  # warm page cache: every configuration reads from memory
            for f in (p.mlo, p.cc):
                with open(f, "rb") as fh:
                    fh.read()
        print(len(self.pairs), "pairs; page cache warm", flush=True)
        models, errors = self._load()

        active = [c for c in self.configurations if c not in errors]
        base = protocol.base_order(active, cfg.seed)
        jobs = protocol.build_jobs(len(self.pairs), cfg.warmup, cfg.passes)
        rows: List[Dict[str, Any]] = []
        failures = {c: 0 for c in active}
        foreign: Optional[Dict[str, Any]] = None
        t_session = time.perf_counter()
        for j, pass_, pi in jobs:
            pair, warm = self.pairs[pi], pass_ == 0
            order = protocol.rotated(base, j)
            if self.guard is not None and j % cfg.card_check_every == 0:
                found = self.guard.check()
                if found:
                    foreign = {"pair_index": j, "processes": found}
                    break
            for f in (pair.mlo, pair.cc):  # keep this pair's files in the page cache for every configuration
                with open(f, "rb") as fh:
                    fh.read()
            load1 = protocol.load1()
            for pos, conf in enumerate(order):
                if conf in errors:
                    continue
                pname, setting = self.configurations[conf]
                if gpu:
                    dev.sync()
                    dev.reset_peak()
                m0 = dev.memory_allocated()
                try:
                    (a, b, c, d), pnl, dcc, sources = self._time(self.pipelines[pname], pair, setting)
                    delta = abs(pnl - dcc)
                    decision = decide(delta)
                    e = time.perf_counter()
                    failures[conf] = 0
                except Exception:  # noqa: BLE001 - three failures in a row (or one in warm-up) drop the configuration
                    failures[conf] += 1
                    tb = traceback.format_exc()
                    print(f"ERROR {conf} pair {j}\n{tb}", flush=True)
                    if failures[conf] >= ERRORS_BEFORE_DROP or warm:
                        errors[conf] = "run: " + tb
                    continue
                extra = (dev.max_memory_allocated() - m0) / 2 ** 20 if gpu else float("nan")
                rows.append({"index": j, "pass": pass_, "pair": pi, "warmup": warm, "position": pos,
                             "configuration": conf, "study": pair.study, "side": pair.side, "read_s": b - a,
                             "pre_s": c - b, "infer_s": d - c, "post_s": e - d, "total_s": e - a, "pnl_mm": pnl,
                             "dcc_mm": dcc, "delta_mm": delta, "decision": decision, "nipple_source_mlo": sources[0],
                             "nipple_source_cc": sources[1], "gpu_extra_MB": extra, "load1": load1})
            if j % PROGRESS_EVERY == 0:
                last = {x["configuration"]: round(x["total_s"], 3) for x in rows if x["index"] == j}
                print(f"{j:3d}/{len(jobs)} {'warm-up' if warm else f'pass {pass_}'} load {load1} total_s {last}",
                      flush=True)
        wall = time.perf_counter() - t_session
        end = protocol.machine_snapshot()
        if self.guard is not None:
            at_end = self.guard.finish()
            if foreign is None and at_end:
                foreign = {"pair_index": len(jobs), "processes": at_end}
            if foreign:
                exc = InvalidMeasurement("another compute process appeared on the card during the session", foreign)
                self._invalid(exc, rows)
                raise exc

        self.out_dir.mkdir(parents=True, exist_ok=True)
        csv_path, json_path = self.out_dir / f"{self.name}.csv", self.out_dir / f"{self.name}.json"
        _write_csv(csv_path, rows)
        R = pd.read_csv(csv_path, dtype={"study": str, "side": str}) if rows else pd.DataFrame(columns=_COLUMNS)
        T = S.measured(R)
        reference = pd.Series({(p.study, p.side): p.reference_delta_mm for p in self.pairs}, dtype=float)
        reference.index.names = ["study", "side"]
        configurations: Dict[str, Any] = {}
        for conf, (pname, setting) in self.configurations.items():
            if conf in errors or not (T["configuration"] == conf).any():
                continue
            t = T[T["configuration"] == conf]
            pipe = self.pipelines[pname]
            o = {"pipeline": pname, "family": pipe.family, "flip_option": bool(pipe.flip_option),
                 "preprocessing": "released", **describe(models[pname], pipe, setting)}
            if gpu:
                o["peak_extra_MB"] = float(R[R["configuration"] == conf]["gpu_extra_MB"].max())
                o["peak_gpu_MB"] = float(o["model_gpu_MB"] + o["peak_extra_MB"])
            o.update(S.timing_summary(t))
            o.update(S.output_summary(t, pipe.family, reference if reference.notna().any() else None,
                                      self.stored.get(pname)))
            configurations[conf] = o
        comparisons = {f"{c[:-len(UNBATCHED)]}_vs_unbatched": S.batching_comparison(T, c[:-len(UNBATCHED)], c, configurations)
                       for c in configurations if c.endswith(UNBATCHED) and c[:-len(UNBATCHED)] in configurations}
        torch_version, cudnn_benchmark = _torch_facts()
        summary = {
            "valid": True,
            "session": {
                "device": dev.name, "kind": "gpu" if gpu else "cpu", "threads": cfg.threads,
                "pinned_cpus": self.setup.get("pinned_cpus"), "core_choice": self.setup.get("core_choice"),
                "preloaded_before_pinning": self.setup.get("preloaded_before_pinning"),
                "threads_end": dev.threads_now(), "cudnn_benchmark": cudnn_benchmark,
                "start": self.setup.get("start"), "end": end, "wall_s": wall, "warmup_pairs": cfg.warmup,
                "n_pairs": int(T["pair"].nunique()) if len(T) else 0, "passes": cfg.passes,
                "load1_during": ({"median": float(T["load1"].median()), "min": float(T["load1"].min()),
                                  "max": float(T["load1"].max())} if len(T) else None),
                "load1_by_pass": S.load1_by_pass(T), "order_seed": cfg.seed,
                "order": "fixed random base order of the configurations, rotated by one per pair; warm-up pairs, then "
                         "every pair once per pass",
                "base_order": base, "batched": cfg.batched, "preprocessing": "released",
                "torch": torch_version, "numpy": np.__version__, "python": sys.version.split()[0],
                "command": self.command},
            "card": self.guard.info if self.guard is not None else None,
            "peak_gpu_method": ("per configuration: memory allocated by loading its model(s) (models are loaded one after "
                                "the other) plus the largest increase of torch.cuda.max_memory_allocated over the "
                                "allocation just before a pair, over all pairs incl. warm-up; memory of the other loaded "
                                "models is not counted") if gpu else None,
            "configurations": configurations,
            "comparisons": comparisons,
            "disturbed_passes": S.disturbed_passes(T, self.name),
            "disturbed_rule": S.DISTURBED_RULE,
            **S.session_flags(configurations),
            "skipped": self.skipped,
            "errors": errors,
        }
        json_path.write_text(json.dumps(summary, indent=1, default=str))
        return summary, 3 if errors else 0


def _torch_facts() -> Tuple[Optional[str], Optional[bool]]:
    try:
        import torch
    except ImportError:
        return None, None
    return torch.__version__, bool(torch.backends.cudnn.benchmark)


_COLUMNS = ["index", "pass", "pair", "warmup", "position", "configuration", "study", "side", "read_s", "pre_s",
            "infer_s", "post_s", "total_s", "pnl_mm", "dcc_mm", "delta_mm", "decision", "nipple_source_mlo",
            "nipple_source_cc", "gpu_extra_MB", "load1"]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(float(v)) if isinstance(v, float) or type(v).__name__.startswith("float") else v)
                        for k, v in r.items()})


# ── command line ──────────────────────────────────────────────────────────────────────────────────────────────

def _apply_flags(cfg: RuntimeConfig, args: Any) -> RuntimeConfig:
    changes: Dict[str, Any] = {}
    for flag, key in (("threads", "threads"), ("passes", "passes"), ("warmup", "warmup"), ("seed", "seed"),
                      ("card_check_every", "card_check_every"), ("core_sampling_s", "core_sampling_s")):
        if getattr(args, flag, None) is not None:
            changes[key] = getattr(args, flag)
    if getattr(args, "configurations", None):
        changes["configurations"] = [c.strip() for c in args.configurations.split(",") if c.strip()]
    if getattr(args, "unbatched", False):
        changes["batched"] = False
    if getattr(args, "no_flops", False):
        changes["flops"] = False
    if getattr(args, "allowed_pids", None):
        changes["allowed_pids"] = protocol.allowed_pid_list(args.allowed_pids)
    cra = cfg.craunet
    for flag, key in (("craunet_code", "code_dir"), ("craunet_weights", "weights_dir")):
        if getattr(args, flag, None):  # command-line paths are relative to the working directory
            cra = replace(cra, **{key: str(Path(getattr(args, flag)).expanduser().resolve())})
    if getattr(args, "craunet_fold", None) is not None:
        cra = replace(cra, fold=args.craunet_fold)
    changes["craunet"] = cra
    try:
        return replace(cfg, **changes)
    except ConfigError as exc:
        raise ConfigError(f"command line: {exc}") from exc


def build_pipelines(cfg: RuntimeConfig, config_dir: Path, paths: PathsConfig, needed: Sequence[str], device: Any,
                    weights: Mapping[str, str]) -> Dict[str, Any]:
    """The pipelines of ``needed`` (not loaded yet); weights from ``weights`` / the config entry / the training
    config's conventions. Raises FileNotFoundError listing every missing file."""
    from breast_seg.benchmark.pipelines import FAMILIES
    from breast_seg.training import build_trainer

    out: Dict[str, Any] = {}
    missing: List[str] = []
    for name in needed:
        if name == CRAUNET:
            from breast_seg.benchmark.craunet import CraunetPipeline

            c = cfg.craunet
            res = lambda v: _resolve(v, config_dir, paths) if v else None  # noqa: E731
            pipe = CraunetPipeline(res(c.code_dir), device, res(c.weights_dir), c.fold, res(c.mlo_weights),
                                   res(c.cc_weights), cfg.flops)
            missing += pipe.missing()
            out[name] = pipe
            continue
        entry = cfg.pipelines[name]
        trainer = build_trainer(train_config_path(entry, config_dir, paths), paths)
        family = trainer.config.family
        if family not in FAMILIES:
            raise ConfigError(f"pipelines.{name}: family {family!r} has no runtime pipeline ({', '.join(FAMILIES)})")
        raw = weights.get(name) or entry.weights
        if raw:
            path = _resolve(raw, Path.cwd() if weights.get(name) else config_dir, paths)
        else:
            path = trainer.model_dir if family == "nnunet" else trainer.best_weights()
        check = path / f"fold_{trainer.config.fold}" / trainer.config.checkpoint if family == "nnunet" else path
        if not check.exists():
            missing.append(f"{name}: {check}")
        out[name] = FAMILIES[family](name, trainer, path, device, cfg.flops)
    if missing:
        raise FileNotFoundError("weights / code not found: " + "; ".join(missing))
    return out


def run_cli(args: Any) -> int:
    """``breast-seg benchmark runtime``."""
    cfg_path = Path(args.config)
    cfg = _apply_flags(load_runtime_config(cfg_path, args.set), args)
    paths = PathsConfig.load(args.paths)
    configurations, skipped = resolve_configurations(cfg, families(cfg, cfg_path.resolve().parent, paths))
    for conf, why in skipped.items():
        print(f"skipped {conf}: {why}", flush=True)
    if not configurations:
        raise ConfigError("no configuration to run")
    try:
        weights = protocol.mapping_arg(args.weights, "--weights")
        stored_files = protocol.mapping_arg(args.stored, "--stored")
        gpu, _ = protocol.parse_device(args.device)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    in_use = {p for p, _ in configurations.values()}
    unknown = sorted((set(weights) | set(stored_files)) - in_use)
    if unknown:
        raise ConfigError(f"--weights / --stored name pipelines that do not run: {unknown} (running: {sorted(in_use)})")
    pairs = read_pairs(Path(args.pairs), Path(args.dicom_root) if args.dicom_root else None, args.first or 0)
    out_dir = Path(args.out) if args.out else paths.dir("results_dir") / "runtime"
    name = args.name or f"runtime_{'gpu' if gpu else 'cpu'}" + (f"_first{args.first}" if args.first else "")
    if (out_dir / f"{name}.json").exists() and not args.overwrite:
        raise ConfigError(f"{name}.json exists in {out_dir}; give another --name or --overwrite")
    cores = [int(c) for c in args.cores.split(",") if c.strip()] if args.cores else None
    command = "breast-seg " + " ".join(sys.argv[1:])

    setup = protocol.prepare_process(args.device, cfg.threads, cores, cfg.core_sampling_s)   # before numpy / torch
    from breast_seg.benchmark.pipelines import Device

    device = Device(setup["gpu"], cfg.threads)
    device.set_threads()
    guard = None
    if gpu:
        import torch

        try:
            torch_uuid: Optional[str] = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        except Exception:  # noqa: BLE001 - older torch has no uuid; the guard falls back to CUDA_VISIBLE_DEVICES
            torch_uuid = None
        guard = protocol.GpuGuard.for_visible_device(torch_uuid, os.environ.get("CUDA_VISIBLE_DEVICES"),
                                                     cfg.allowed_pids, name=device.name)
    pipelines = build_pipelines(cfg, cfg_path.resolve().parent, paths, list(in_use), device, weights)
    stored = {p: (load_stored(Path(f), cfg.craunet.fold if p == CRAUNET else None), Path(f).name)
              for p, f in stored_files.items()}
    session = Session(cfg, pairs, configurations, pipelines, device, out_dir, name, setup, guard, stored, skipped,
                      command, overwrite=args.overwrite)
    try:
        summary, code = session.run()
    except InvalidMeasurement:
        return 4
    short = {c: {"pairs_per_min": round(o["pairs_per_min"], 1),
                 "by_pass": [round(p["pairs_per_min"], 1) for p in o["passes"]],
                 "total_median_s": round(o["per_pair_s"]["total_s"]["median"], 3),
                 "flips_vs_stored": (o.get("check") or {}).get("decision_flips")}
             for c, o in summary["configurations"].items()}
    print(json.dumps(short, indent=1))
    print(f"disturbed passes: {[p['pass'] for p in summary['disturbed_passes'] if p['disturbed']]} | decisions differ "
          f"from stored: {summary['decisions_differ_from_stored']} | passes disagree (> 10 %): "
          f"{list(summary['passes_disagree_over_10pct'])} | errors: {list(summary['errors'])} | "
          f"written {out_dir / name}.json", flush=True)
    return code


def summary_cli(args: Any) -> int:
    """``breast-seg benchmark summary``: report of sessions, optionally composed from their undisturbed passes."""
    from breast_seg.benchmark import summary as S

    sessions = []
    for p in args.sessions:
        try:
            js, rows = S.load_session(Path(p))
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if "disturbed_passes" not in js:
            js["disturbed_passes"] = S.disturbed_passes(S.measured(rows), Path(p).stem)
        sessions.append((Path(p).stem, js, rows))
    out_dir = Path(args.out) if args.out else Path(args.sessions[0]).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    results = [(label, js) for label, js, _ in sessions]
    if args.compose:
        ordered = sorted(sessions, key=lambda s: str((s[1]["session"].get("start") or {}).get("time", "")))
        try:
            js, rows = S.compose(ordered)
        except ValueError as exc:
            raise ConfigError(f"--compose: {exc}") from exc
        (out_dir / f"{args.compose}.json").write_text(json.dumps(js, indent=1, default=str))
        rows.to_csv(out_dir / f"{args.compose}.csv", index=False)
        for p in js["session"]["composed"]["passes"]:
            print(f"{p['session']} pass {p['pass']}: {p['n_over']}/{p['n_measurements']} "
                  f"({100 * p['share_over']:.1f} %) {'DISTURBED' if p['disturbed'] else 'kept'}")
        results.append((args.compose, js))
    report = out_dir / f"{args.report}.md"
    report.write_text(S.report_markdown(results))
    print(f"written {report}")
    return 0
