"""Runtime benchmark (Table 11) with fakes: no weights, synthetic DICOMs only.

Covers the protocol (core choice, job list, rotating order, GPU guard), the summary math (pairs per minute,
spreads, the disturbed-pass rule, composition), the paper's nipple rule and measurement, a whole session with fake
pipelines (including the guard's abort path) and the command line.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

from breast_seg.benchmark import protocol
from breast_seg.benchmark import summary as S
from breast_seg.benchmark.pipelines import CanvasPipeline, Pipeline, canvas, decide, measure_pair, read_dicom_file
from breast_seg.benchmark.protocol import GpuGuard, InvalidMeasurement, Pair, Setting
from breast_seg.benchmark.runtime import (
    TABLE11,
    RuntimeConfig,
    Session,
    families,
    load_runtime_config,
    read_pairs,
    resolve_configurations,
)
from breast_seg.cli import main
from breast_seg.config import ConfigError, PathsConfig
from breast_seg.evaluation.landmarks import nipple_point
from breast_seg.imaging.phantom import phantom_dicom

REPO = Path(__file__).resolve().parents[1]
TABLE11_YAML = REPO / "configs" / "benchmark" / "runtime_table11.yaml"
PAIRS_LIST = REPO / "configs" / "benchmark" / "vindr_heldout_pairs.csv"


# ── nipple rule and measurement ───────────────────────────────────────────────────────────────────────────────

def test_nipple_point_follows_the_paper_rule():
    lab = np.zeros((64, 64), np.uint8)
    lab[10:12, 10:12] = 3            # 4 px speck
    lab[40:46, 30:36] = 3            # larger component wins, no size limit
    xy, src = nipple_point(lab)
    assert src == "argmax" and np.allclose(xy, [32.5, 42.5])
    speck = np.zeros((64, 64), np.uint8)
    speck[5, 7] = 3                  # a single pixel still counts
    assert nipple_point(speck)[1] == "argmax"

    empty = np.zeros((64, 64), np.uint8)
    assert nipple_point(empty) == (None, "none")
    prob = np.zeros((64, 64), np.float16)
    prob[20:24, 50:54] = 0.03        # above the 0.02 threshold
    xy, src = nipple_point(empty, prob)
    assert src == "low_confidence" and np.allclose(xy, [51.5, 21.5])
    low = np.full((64, 64), 0.001, np.float16)
    low[3, 60] = 0.01                # nothing reaches 0.02: the most probable pixel
    xy, src = nipple_point(empty, low)
    assert src == "max_probability" and np.allclose(xy, [60, 3])
    inst = np.zeros((64, 64), np.uint8)
    assert nipple_point(empty, inst) == (None, "none")
    inst[30:32, 30:32] = 1           # YOLO's low-confidence instance mask
    assert nipple_point(empty, inst)[1] == "low_confidence"


def _write_dicom(path: Path, view: str, side: str, **kw) -> Path:
    path.write_bytes(phantom_dicom(view, side, **kw))
    return path


def _painted(img: np.ndarray, mlo: bool, nipple: bool = True) -> np.ndarray:
    """A label map drawn from the canvas: tissue, a pectoral wedge at the top-left (MLO), a nipple disk."""
    lab = np.where(img > 0, 2, 0).astype(np.uint8)
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    if mlo:
        lab[(xs / 150.0 + ys / 300.0 <= 1.0) & (lab > 0)] = 1
    if nipple:
        cols = np.nonzero((lab == 2).any(0))[0]
        x0, y0 = int(cols.max()) - 6, int(np.nonzero(lab[:, cols.max()])[0].mean())
        lab[(xs - x0) ** 2 + (ys - y0) ** 2 <= 16] = 3
    return lab


def test_measure_pair_on_a_phantom_canvas(tmp_path):
    pm, dm = read_dicom_file(_write_dicom(tmp_path / "m.dcm", "MLO", "L"))
    pc, dc = read_dicom_file(_write_dicom(tmp_path / "c.dcm", "CC", "L"))
    im, tm = canvas(pm, dm)
    ic, tc = canvas(pc, dc)
    calls: List[int] = []

    def fallback(i):
        calls.append(i)
        p = np.zeros(ic.shape, np.float32)
        p[320, 400] = 0.5
        return p

    pnl, dcc, src = measure_pair(_painted(im, True), tm, _painted(ic, False, nipple=False), tc, "L", fallback)
    assert src == ("argmax", "low_confidence") and calls == [1]
    assert np.isfinite(pnl) and pnl > 0
    x_orig = (400 - tc.pad_left) / tc.scale + tc.crop_x1
    assert dcc == pytest.approx(x_orig * 0.1)
    assert decide(abs(pnl - dcc)) in ("adequate", "inadequate") and decide(float("nan")) == "no measurement"


# ── protocol ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_select_cores_takes_the_least_loaded_physical_cores():
    # 4 physical cores with 2 SMT siblings each: (0, 4), (1, 5), (2, 6), (3, 7)
    busy = {0: .9, 4: .0, 1: .1, 5: .1, 2: .0, 6: .3, 3: .05, 7: .0}
    snaps = iter([{c: (0, 0) for c in busy}, {c: (int(100 * b), 100) for c, b in busy.items()}])
    chosen, info = protocol.select_cores(2, 5.0, allowed=range(8), stat=lambda: next(snaps),
                                         siblings=lambda c: (c % 4, c % 4 + 4), sleep=lambda s: None)
    # core loads: (3,7) 0.05 < (1,5) 0.2 < (2,6) 0.3 < (0,4) 0.9; least-busy sibling of each
    assert chosen == [1, 7]
    assert info["busy_fraction_chosen"] == {"1": 0.1, "7": 0.0} and info["busy_fraction_sibling"]["7"] == 0.05
    snaps = iter([{c: (0, 0) for c in busy}, {c: (0, 100) for c in busy}])
    chosen, info = protocol.select_cores(6, 0.0, allowed=range(8), stat=lambda: next(snaps),
                                         siblings=lambda c: (c % 4, c % 4 + 4), sleep=lambda s: None)
    assert len(chosen) == 4 and "only 4 physical cores" in info["note"]
    assert protocol.parse_cpu_list("0-2,8") == (0, 1, 2, 8)


def test_jobs_and_rotating_order_match_the_research_protocol():
    jobs = protocol.build_jobs(n_pairs=3, warmup=5, passes=2)
    assert [(t, c) for _, t, c in jobs] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    assert [j for j, _, _ in jobs] == list(range(9))
    # the base order of the paper's sessions (seed 20261120, the Table 11 list)
    base = protocol.base_order(TABLE11, 20261120)
    assert base == ["yolo26m", "segformer_b3_no_tta", "nnunet_tta", "craunet", "unet_no_tta", "nnunet_no_tta",
                    "unet_tta", "segformer_b3_tta"]
    # over len(base) consecutive pairs every configuration takes every position once
    positions: Dict[str, set] = {c: set() for c in base}
    for j in range(3, 3 + len(base)):
        for pos, c in enumerate(protocol.rotated(base, j)):
            positions[c].add(pos)
    assert all(p == set(range(len(base))) for p in positions.values())
    assert protocol.rotated(base, 1) == base[1:] + base[:1]


def test_parse_device_and_mapping_args():
    assert protocol.parse_device("cpu") == (False, None)
    assert protocol.parse_device("cuda") == (True, None)
    assert protocol.parse_device("cuda:1") == (True, "1") and protocol.parse_device("0") == (True, "0")
    with pytest.raises(ValueError):
        protocol.parse_device("tpu")
    assert protocol.mapping_arg(["a=x", "b = y"], "--w") == {"a": "x", "b": "y"}
    with pytest.raises(ValueError):
        protocol.mapping_arg(["a"], "--w")


class FakeSmi:
    """nvidia-smi answers: two cards, a list of compute processes and a utilisation per card."""

    def __init__(self, apps: List[Tuple[str, int]], util: float = 0.0):
        self.apps, self.util = apps, util

    def __call__(self, *args):
        q = args[0]
        if q == "--query-gpu=index,uuid":
            return ["0, GPU-aaa", "1, GPU-bbb"]
        if q.startswith("--query-compute-apps"):
            return [f"{u}, {pid}, 100 MiB, proc" for u, pid in self.apps]
        if q.startswith("--query-gpu=uuid,utilization.gpu"):
            return [f"GPU-aaa, {self.util} %, 1 MiB", "GPU-bbb, 99 %, 1 MiB"]
        return []


def test_gpu_guard_identifies_the_card_and_finds_foreign_processes():
    smi = FakeSmi([("GPU-bbb", 7), ("GPU-bbb", 11), ("GPU-bbb", 12), ("GPU-aaa", 99)])
    g = GpuGuard.for_visible_device(None, "1", allowed_pids=[11], smi=smi, own_pid=7)
    assert g.uuid == "GPU-bbb" and g.info["uuid_from"] == "CUDA_VISIBLE_DEVICES"
    assert [p["pid"] for p in g.foreign()] == [12]          # not itself (7), not the allowed 11, not the other card
    assert GpuGuard.for_visible_device("GPU-aaa", "1", smi=smi).info["uuid_from"] == "torch"
    with pytest.raises(InvalidMeasurement, match="at start"):
        g.check_start(sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        GpuGuard.for_visible_device(None, "5", smi=smi)


def test_gpu_guard_aborts_on_a_busy_card_and_passes_an_idle_one():
    busy = GpuGuard("GPU-aaa", smi=FakeSmi([], util=55.0), own_pid=1)
    waits: List[float] = []
    with pytest.raises(InvalidMeasurement, match="busy"):
        busy.check_start(polls=3, poll_s=2.0, sleep=waits.append)
    assert waits == [2.0, 2.0, 2.0]
    idle = GpuGuard("GPU-aaa", smi=FakeSmi([("GPU-aaa", 1)], util=3.0), own_pid=1)
    idle.check_start(sleep=lambda s: None)
    assert idle.info["usage_start"]["utilization_pct"] == 3.0 and idle.check() == [] and idle.info["checks"] == 1


# ── summary math ──────────────────────────────────────────────────────────────────────────────────────────────

def _rows(times: Dict[Tuple[str, int], List[float]], warmup: int = 1) -> pd.DataFrame:
    """Measurement rows: (configuration, pass) -> total seconds per pair (pass 0 = warm-up)."""
    out = []
    for (conf, p), ts in times.items():
        for i, t in enumerate(ts):
            out.append({"configuration": conf, "pass": p, "pair": i, "warmup": p == 0, "study": f"s{i // 2}",
                        "side": "LR"[i % 2], "read_s": t / 4, "pre_s": t / 4, "infer_s": t / 4, "post_s": t / 4,
                        "total_s": t, "delta_mm": 5.0 + i, "decision": "adequate", "load1": 1.0 + p,
                        "nipple_source_mlo": "argmax", "nipple_source_cc": "argmax"})
    return pd.DataFrame(out)


def test_pairs_per_minute_median_and_spread():
    t = _rows({("a", 1): [1.0, 2.0, 3.0, 4.0], ("a", 2): [2.0, 2.0, 2.0, 2.0]})
    s = S.timing_summary(t)
    assert s["pairs_per_min"] == pytest.approx(60 * 8 / 18)
    assert s["per_pair_s"]["total_s"]["median"] == 2.0 and s["n_measurements"] == 8
    by_pass = [p["pairs_per_min"] for p in s["passes"]]
    assert by_pass == [pytest.approx(24.0), pytest.approx(30.0)]
    assert s["pass_spread_pct"]["pairs_per_min"] == pytest.approx(100 * 6 / 27)
    assert s["passes_disagree"] is True                       # 22 % > 10 %
    assert S.spread_pct([10.0]) is None and S.spread_pct([10.0, 10.5]) == pytest.approx(100 * 0.5 / 10.25)


def test_disturbed_pass_rule():
    # median of configuration "a" over the session is 1.0; slow = > 1.5 s
    fast = [1.0] * 40
    p1 = fast[:38] + [1.6, 1.6]                  # 2 / 40 = 5 % slow: not more than 5 %, undisturbed
    p2 = fast[:37] + [1.6, 1.6, 1.6]             # 3 / 40 = 7.5 % slow: disturbed
    t = _rows({("a", 1): p1, ("a", 2): p2, ("b", 1): [2.0] * 40, ("b", 2): [3.1] + [2.0] * 39})
    d = {p["pass"]: p for p in S.disturbed_passes(S.measured(t), "sess")}
    assert d[1]["n_over"] == 2 and d[1]["share_over"] == pytest.approx(2 / 80) and not d[1]["disturbed"]
    assert d[2]["n_over"] == 4 and d[2]["share_over"] == pytest.approx(4 / 80) and not d[2]["disturbed"]
    t2 = _rows({("a", 1): fast, ("a", 2): [1.0] * 36 + [1.6] * 4, ("b", 1): [2.0] * 40, ("b", 2): [2.0] * 39 + [3.1]})
    d2 = {p["pass"]: p for p in S.disturbed_passes(S.measured(t2), "sess")}
    assert d2[2]["n_over"] == 5 and d2[2]["disturbed"] and not d2[1]["disturbed"]   # 5 / 80 = 6.25 %
    assert S.disturbed_passes(S.measured(t2).iloc[:0]) == []


def _session_json(kind: str = "cpu", start: str = "2026-01-01 00:00:00") -> dict:
    conf = {"pipeline": "p", "family": "unet", "flip_option": True, "tta": True, "batched": True,
            "check": {"decision_flips": 0, "measurement_only_on_one_side": 0}}
    return {"valid": True, "session": {"kind": kind, "threads": 4, "batched": True, "preprocessing": "released",
                                       "start": {"time": start}, "end": {"time": start}, "pinned_cpus": [0],
                                       "device": kind},
            "configurations": {"a": dict(conf), "b": dict(conf)}, "decisions_differ_from_stored": {}}


def test_compose_keeps_the_undisturbed_passes_in_time_order():
    slow = [1.0] * 35 + [2.0] * 5                # 5 / 80 = 6.25 % of a pass slow: disturbed
    one = _rows({("a", 0): [1.0], ("a", 1): slow, ("a", 2): [1.0] * 40, ("b", 1): [2.0] * 40, ("b", 2): [2.0] * 40})
    two = _rows({("a", 1): [1.2] * 40, ("a", 2): slow, ("b", 1): [2.0] * 40, ("b", 2): [2.0] * 40})
    js, rows = S.compose([("first", _session_json(), one), ("second", _session_json(start="2026-01-02 00:00:00"), two)])
    kept = [(p["session"], p["pass"]) for p in js["session"]["composed"]["passes"] if not p["disturbed"]]
    assert kept == [("first", 2), ("second", 1)]
    assert sorted(rows["pass"].unique()) == [1, 2] and set(rows["source"]) == {"first pass 2", "second pass 1"}
    a = js["configurations"]["a"]
    assert [p["source"] for p in a["passes"]] == ["first pass 2", "second pass 1"]
    assert a["pairs_per_min"] == pytest.approx(60 * 80 / (40 * 1.0 + 40 * 1.2))
    assert js["session"]["composed"]["outputs_checked_in"] == "second"   # first session's pass 1 was disturbed
    assert js["session"]["passes"] == 2 and js["session"]["start"]["time"] == "2026-01-01 00:00:00"
    other = _session_json()
    other["session"]["threads"] = 8
    with pytest.raises(ValueError, match="threads"):
        S.compose([("first", _session_json(), one), ("x", other, two)])


def test_accuracy_and_stored_check():
    idx = pd.MultiIndex.from_tuples([("s", "L"), ("s", "R"), ("t", "L")], names=["study", "side"])
    delta = pd.Series([5.0, 12.0, np.nan], index=idx)
    ref = pd.Series([4.0, 8.0, 20.0], index=idx)          # good, good, bad
    acc = S.accuracy(delta, ref)
    assert acc["10 mm"]["TP"] == 1 and acc["10 mm"]["FN"] == 1 and acc["10 mm"]["n_fail"] == 1
    assert acc["8/12 mm"]["TP"] == 2                       # 12 mm on a reference-good pair is inside the +-2 mm band
    stored = pd.Series([5.5, 9.0], index=idx[:2])
    ch = S.stored_check(delta, stored, "x.csv", 1)
    assert ch["n_compared"] == 2 and ch["decision_flips"] == 1 and ch["max_abs_delta_diff_mm"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="more than one"):
        S.stored_check(delta, pd.Series([1.0, 2.0], index=pd.MultiIndex.from_tuples([("s", "L")] * 2)), "x.csv", 0)


# ── a whole session with fakes ────────────────────────────────────────────────────────────────────────────────

class FakeDevice:
    gpu, name, threads = False, "fake CPU", 1

    def sync(self): pass
    def memory_allocated(self): return 0
    def reset_peak(self): pass
    def max_memory_allocated(self): return 0
    def set_threads(self): pass
    def threads_now(self): return {"torch": 1, "cv2": 1}


class TablePipeline(Pipeline):
    """Reads the real DICOMs, returns fixed PNL / D_CC per pair."""

    family, flip_option = "unet", True

    def __init__(self, name: str, values: Dict[int, Tuple[float, float]], fail_load: bool = False):
        self.name, self.values, self.fail_load, self.calls = name, values, fail_load, 0

    def load(self):
        if self.fail_load:
            raise RuntimeError("no weights")
        return {"params_M": 2.0, "gflops_per_forward": 3.0, "models_per_pair": 1}

    def prepare(self, raw, pair):
        return raw[0][0].shape

    def infer(self, prepared, setting):
        self.calls += 1
        return setting.tta

    def measure(self, output, prepared, pair):
        pnl, dcc = self.values[pair.index]
        return pnl, dcc, ("argmax", "argmax")


class PaintedPipeline(CanvasPipeline):
    """The real canvas and measurement with a fake network that paints the label maps."""

    family, flip_option = "unet", True

    def __init__(self):
        super().__init__("painted", None, Path("."), FakeDevice())

    def load(self):
        return {"params_M": 0.1, "models_per_pair": 1}

    def model_input(self, im, ic):
        return im, ic

    def run_model(self, x, setting):
        return _painted(x[0], True), _painted(x[1], False), lambda i: None


def _pairs(tmp_path: Path, n: int = 3) -> List[Pair]:
    pairs = []
    for i in range(n):
        side = "LR"[i % 2]
        m = _write_dicom(tmp_path / f"m{i}.dcm", "MLO", side, width=300, height=400, seed=i)
        c = _write_dicom(tmp_path / f"c{i}.dcm", "CC", side, width=300, height=400, seed=i)
        pairs.append(Pair(i, f"study{i // 2}", side, str(m), str(c), reference_delta_mm=[4.0, 15.0, 9.0][i % 3]))
    return pairs


def _session(tmp_path, pairs, pipelines, configurations, guard=None, **kw):
    cfg = RuntimeConfig(passes=2, warmup=2, seed=7, threads=1, card_check_every=2)
    return Session(cfg, pairs, configurations, pipelines, FakeDevice(), tmp_path / "out", "run", {"pinned_cpus": [0]},
                   guard=guard, **kw)


def test_session_runs_every_pair_and_configuration(tmp_path):
    pairs = _pairs(tmp_path)
    values = {0: (50.0, 45.0), 1: (50.0, 30.0), 2: (float("nan"), 40.0)}   # adequate, inadequate, no measurement
    pipe = TablePipeline("m", values)
    confs = {"m_tta": ("m", Setting(True, True)), "m_no_tta": ("m", Setting(False, True)),
             "painted_tta": ("painted", Setting(True, True)), "broken": ("x", Setting())}
    stored = pd.Series([5.0, 5.0], index=pd.MultiIndex.from_tuples([("study0", "L"), ("study0", "R")],
                                                                     names=["study", "side"]))
    s = _session(tmp_path, pairs, {"m": pipe, "painted": PaintedPipeline(), "x": TablePipeline("x", {}, fail_load=True)},
                 confs, stored={"m": (stored, "stored.csv")}, skipped={"craunet": "not configured"})
    summary, code = s.run()
    assert code == 3 and list(summary["errors"]) == ["broken"] and summary["skipped"] == {"craunet": "not configured"}
    rows = pd.read_csv(tmp_path / "out" / "run.csv")
    assert len(rows) == (2 + 2 * 3) * 3                               # (warm-up + passes x pairs) x running configurations
    assert set(rows.groupby("index")["position"].apply(lambda p: tuple(sorted(p)))) == {(0, 1, 2)}
    assert rows.groupby("configuration")["position"].nunique().eq(3).all()   # the order rotates
    m = rows[(rows.configuration == "m_tta") & (rows["pass"] == 1)].set_index("pair")
    assert list(m.decision) == ["adequate", "inadequate", "no measurement"]
    o = summary["configurations"]["m_tta"]
    assert o["n_measurements"] == 6 and len(o["passes"]) == 2 and o["forwards_per_pair"] == 4
    assert o["gflops_per_pair"] == 12.0 and o["weights_fp32_MB"] == 8.0
    assert o["check"]["n_compared"] == 2 and o["check"]["decision_flips"] == 1 and o["check"]["no_measurement"] == 1
    assert o["accuracy"]["10 mm"]["n"] == 3 and o["accuracy_stored"]["10 mm"]["n"] == 3
    assert summary["decisions_differ_from_stored"] == {"m_tta": 1, "m_no_tta": 1}
    p = summary["configurations"]["painted_tta"]
    assert p["nipple_source"]["MLO"] == {"argmax": 3} and p["check"] is None
    assert [d["pass"] for d in summary["disturbed_passes"]] == [1, 2]
    assert summary["session"]["base_order"] == protocol.base_order(["m_tta", "m_no_tta", "painted_tta"], 7)
    assert json.loads((tmp_path / "out" / "run.json").read_text())["valid"] is True
    with pytest.raises(FileExistsError):
        _session(tmp_path, pairs, {"m": pipe}, {"m_tta": ("m", Setting(True, True))})


class FakeGuard:
    """Finds a foreign process on its ``n``-th periodic check."""

    def __init__(self, n: int):
        self.n, self.info = n, {"uuid": "GPU-x", "checks": 0}

    def check_start(self):
        pass

    def check(self):
        self.info["checks"] += 1
        return [{"pid": 4242, "used": "1 MiB", "name": "other"}] if self.info["checks"] >= self.n else []

    def finish(self):
        return []


def test_session_aborts_when_a_foreign_process_appears(tmp_path):
    pairs = _pairs(tmp_path)
    pipe = TablePipeline("m", {i: (50.0, 45.0) for i in range(3)})
    s = _session(tmp_path, pairs, {"m": pipe}, {"m_tta": ("m", Setting(True, True))}, guard=FakeGuard(2))
    with pytest.raises(InvalidMeasurement, match="appeared"):
        s.run()
    out = tmp_path / "out"
    assert not (out / "run.json").exists()
    invalid = list(out.glob("run_INVALID_*.json"))
    assert len(invalid) == 1 and json.loads(invalid[0].read_text())["valid"] is False
    partial = pd.read_csv(invalid[0].with_suffix(".csv"))
    assert len(partial) == 2 and pipe.calls == 2                   # stopped at job 2 (check every 2 pairs)


def test_session_aborts_at_start_on_a_busy_card(tmp_path):
    pipe = TablePipeline("m", {})
    guard = GpuGuard("GPU-aaa", smi=FakeSmi([("GPU-aaa", 31)]), own_pid=1)
    s = _session(tmp_path, _pairs(tmp_path, 1), {"m": pipe}, {"m_tta": ("m", Setting(True, True))}, guard=guard)
    with pytest.raises(InvalidMeasurement, match="at start"):
        s.run()
    assert pipe.calls == 0 and len(list((tmp_path / "out").glob("run_INVALID_*.json"))) == 1


# ── configuration, pairs and command line ─────────────────────────────────────────────────────────────────────

def test_table11_config_resolves_and_skips_craunet_without_its_code():
    cfg = load_runtime_config(TABLE11_YAML)
    fams = families(cfg, TABLE11_YAML.parent, PathsConfig(root=str(REPO)))
    assert fams == {"segformer_b3": "segformer", "nnunet": "nnunet", "yolo26m": "yolo_seg", "unet": "unet"}
    confs, skipped = resolve_configurations(cfg, fams)
    assert list(confs) == TABLE11[1:] and list(skipped) == ["craunet"]
    assert confs["nnunet_tta"] == ("nnunet", Setting(True, True)) and confs["yolo26m"] == ("yolo26m", Setting(False, True))
    cfg.craunet.code_dir = "somewhere"
    cfg.configurations = ["craunet", "segformer_b3_tta_unbatched"]
    confs, skipped = resolve_configurations(cfg, fams)
    assert confs == {"craunet": ("craunet", Setting(False, False)),
                     "segformer_b3_tta_unbatched": ("segformer_b3", Setting(True, False))} and not skipped
    cfg.configurations = ["yolo26m_tta"]
    with pytest.raises(ConfigError, match="unknown configuration"):
        resolve_configurations(cfg, fams)
    with pytest.raises(ConfigError):
        RuntimeConfig(passes=0)


def test_read_pairs_by_path_and_by_sop(tmp_path):
    root = tmp_path / "dicom" / "study1"
    root.mkdir(parents=True)
    for sop in ("m1", "c1"):
        (root / f"{sop}.dicom").write_bytes(b"x")
    by_sop = tmp_path / "sop.csv"
    by_sop.write_text("study,side,mlo_sop,cc_sop\nstudy1,l,m1,c1\n")
    (p,) = read_pairs(by_sop, tmp_path / "dicom")
    assert p.side == "L" and p.mlo.endswith("m1.dicom") and np.isnan(p.reference_delta_mm)
    by_path = tmp_path / "path.csv"
    by_path.write_text("study,side,mlo_dicom,cc_dicom,reference_delta_mm\nstudy1,R,dicom/study1/m1.dicom,"
                       "dicom/study1/c1.dicom,7.5\n")
    (q,) = read_pairs(by_path)
    assert q.reference_delta_mm == 7.5 and Path(q.cc).is_file()
    by_sop.write_text("study,side,mlo_sop,cc_sop\nstudy1,L,m1,missing\n")
    with pytest.raises(FileNotFoundError):
        read_pairs(by_sop, tmp_path / "dicom")
    with pytest.raises(ConfigError, match="dicom-root"):
        read_pairs(by_sop)


def test_heldout_pair_list_holds_identifiers_only():
    with open(PAIRS_LIST, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 146 and list(rows[0]) == ["study", "side", "mlo_sop", "cc_sop"]
    assert len({(r["study"], r["side"]) for r in rows}) == 146 and {r["side"] for r in rows} == {"L", "R"}
    assert all(len(r[k]) == 32 and int(r[k], 16) >= 0 for r in rows for k in ("study", "mlo_sop", "cc_sop"))


def test_cli_help_and_summary_report(tmp_path, capsys):
    for cmd in (["benchmark", "runtime", "--help"], ["benchmark", "summary", "--help"]):
        with pytest.raises(SystemExit) as e:
            main(cmd)
        assert e.value.code == 0
    pairs = _pairs(tmp_path)
    pipe = TablePipeline("m", {i: (50.0, 45.0) for i in range(3)})
    for name in ("one", "two"):
        cfg = RuntimeConfig(passes=2, warmup=1, seed=7, threads=1)
        Session(cfg, pairs, {"m_tta": ("m", Setting(True, True)), "m_no_tta": ("m", Setting(False, True))},
                {"m": pipe}, FakeDevice(), tmp_path / "out", name, {"pinned_cpus": [0]}).run()
    out = tmp_path / "out"
    assert main(["benchmark", "summary", str(out / "one.json"), str(out / "two.json"), "--compose", "both"]) == 0
    report = (out / "runtime_summary.md").read_text()
    assert "## one" in report and "## both" in report and "Table 11 layout" in report
    assert json.loads((out / "both.json").read_text())["session"]["composed"]["sources"] == ["one", "two"]
    assert main(["benchmark", "summary", str(tmp_path / "nothing.json")]) == 2
    capsys.readouterr()


def test_craunet_released_preprocessing_geometry(tmp_path):
    pytest.importorskip("skimage")
    from breast_seg.benchmark.craunet import perpendicular_distance, released_preprocess

    px, ds = read_dicom_file(_write_dicom(tmp_path / "m.dcm", "MLO", "L", spacing=0.1))
    img, geom = released_preprocess(px, ds, "L")
    assert img.shape == (512, 512) and img.dtype == np.float32 and img.min() >= 0 and img.max() <= 1
    assert geom["spacing"] == 0.1 and geom["padded_w"] >= geom["L"]
    assert float(img[:, -5:].max()) == 0.0                       # left breast: padding on the right
    p = np.array([0.0, 0.0], np.float32), np.array([0.0, 10.0], np.float32), np.array([3.0, 4.0], np.float32)
    assert perpendicular_distance(*p) == pytest.approx(3.0)
