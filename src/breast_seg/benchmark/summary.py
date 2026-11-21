"""Summary of runtime sessions: per-configuration statistics, passes, the disturbed-pass rule, composition, report.

Definitions (one measurement = one pair through one configuration, warm-up excluded):

* **seconds per pair**: median [q1, q3] of ``total_s`` over all measurements (pairs x passes); stages likewise.
* **pairs per minute**: ``60 x measurements / summed total_s``.
* **spread between passes**: ``(max - min) / mean`` of the per-pass values, per cent; passes *disagree* above 10 %.
* **disturbed pass**: more than 5 % of its measurements take longer than 1.5 x the median of the same
  configuration in the same session (:func:`disturbed_passes`). Something else on the machine slowed the pass down
  (pinning keeps this process on its cores, not other processes off them).
* **composed result** (:func:`compose`): the undisturbed passes of several sessions with the same settings (device,
  threads, batching, preprocessing, warm-up and pair set), in time order, renumbered 1..n. Every timing number is
  recomputed from the kept passes with the definitions above (seconds per pair, pairs per minute, passes, spread,
  peak GPU memory, batching speed-ups, differences between passes); model load times are dropped. Outputs (ΔD,
  decisions) do not depend on timing, so the checks against stored predictions and the accuracy come from the first
  session whose pass 1 is kept.

Rows of a configuration that was dropped after an error stay in the per-pair CSV but enter no summary (they are
counted under ``rows_ignored``).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from breast_seg.evaluation.rule import THRESHOLD_MM, PNLRule

STAGES = ("read_s", "pre_s", "infer_s", "post_s", "total_s")
#: Disturbed-pass rule: a measurement is slow above SLOW_FACTOR x its configuration's session median; a pass is
#: disturbed when the share of slow measurements exceeds SLOW_SHARE.
SLOW_FACTOR = 1.5
SLOW_SHARE = 0.05
#: Passes disagree when their pairs per minute or median time differ by more than this (per cent of the mean).
DISAGREE_PCT = 10.0
#: Decision scorings of the accuracy check: flat 10 mm, +-1 mm and +-2 mm tolerance bands.
RULES = {"10 mm": PNLRule(THRESHOLD_MM, 0.0), "9/11 mm": PNLRule(THRESHOLD_MM, 1.0), "8/12 mm": PNLRule(THRESHOLD_MM, 2.0)}
DISTURBED_RULE = (f"a pass is disturbed when more than {SLOW_SHARE:.0%} of its measurements take longer than "
                  f"{SLOW_FACTOR} x the median of the same configuration in the same session")


def measured(rows: pd.DataFrame) -> pd.DataFrame:
    """Measurements without the warm-up pairs."""
    return rows[~rows["warmup"].astype(str).str.lower().eq("true")]


def stage_stats(t: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {c: {"median": float(t[c].median()), "q1": float(t[c].quantile(.25)), "q3": float(t[c].quantile(.75)),
                "mean": float(t[c].mean()), "sd": float(t[c].std())} for c in STAGES}


def pairs_per_minute(t: pd.DataFrame) -> Optional[float]:
    return float(60 * len(t) / t["total_s"].sum()) if len(t) else None


def spread_pct(values: Sequence[Optional[float]]) -> Optional[float]:
    """``(max - min) / mean`` in per cent; None with fewer than two values."""
    v = [x for x in values if x is not None]
    return float(100 * (max(v) - min(v)) / (sum(v) / len(v))) if len(v) > 1 else None


def disturbed_passes(t: pd.DataFrame, session: str = "") -> List[Dict[str, Any]]:
    """The disturbed-pass rule on the measurements ``t`` of ONE session (warm-up excluded)."""
    if t.empty:
        return []
    ratio = t["total_s"] / t.groupby("configuration")["total_s"].transform("median")
    out = []
    for p, g in t.assign(_ratio=ratio).groupby("pass"):
        slow = g["_ratio"] > SLOW_FACTOR
        share = float(slow.mean())
        out.append({"session": session, "pass": int(p), "n_measurements": int(len(g)), "n_over": int(slow.sum()),
                    "share_over": share, "disturbed": bool(share > SLOW_SHARE),
                    "load1_median": float(g["load1"].median()), "load1_max": float(g["load1"].max())})
    return out


def pass_summaries(t: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for p, g in t.groupby("pass"):
        d = {"pass": int(p), "n_measurements": int(len(g)), "pairs_per_min": pairs_per_minute(g),
             "total_s": stage_stats(g)["total_s"], "load1_start": float(g["load1"].iloc[0]),
             "load1_end": float(g["load1"].iloc[-1]), "load1_median": float(g["load1"].median()),
             "load1_max": float(g["load1"].max())}
        if "source" in g:
            d["source"] = str(g["source"].iloc[0])
        out.append(d)
    return out


def timing_summary(t: pd.DataFrame) -> Dict[str, Any]:
    """Every timing field of one configuration (``t``: its measurements)."""
    passes = pass_summaries(t)
    sp_ppm = spread_pct([x["pairs_per_min"] for x in passes])
    sp_med = spread_pct([x["total_s"]["median"] for x in passes])
    return {"n_measurements": int(len(t)), "per_pair_s": stage_stats(t), "total_s_all_pairs": float(t["total_s"].sum()),
            "pairs_per_min": pairs_per_minute(t), "passes": passes,
            "pass_spread_pct": {"pairs_per_min": sp_ppm, "total_median": sp_med},
            "passes_disagree": bool(max(sp_ppm or 0, sp_med or 0) > DISAGREE_PCT)}


def _delta_by_pair(t: pd.DataFrame) -> pd.Series:
    return t.set_index(["study", "side"])["delta_mm"].astype(float)


def accuracy(delta: pd.Series, reference_delta: pd.Series) -> Dict[str, Dict[str, float]]:
    """Decision metrics of ``delta`` against the reference ΔD (reference adequate = ``<= 10 mm``), per scoring."""
    ref = reference_delta.reindex(delta.index)
    has = ref.notna()
    good = ref[has] <= THRESHOLD_MM
    keep = ["n", "n_fail", "TP", "FP", "FN", "TN", "accuracy", "precision", "recall", "f1", "specificity"]
    return {name: {k: v for k, v in rule.matrix(delta[has].values, good.values).items() if k in keep}
            for name, rule in RULES.items()}


def stored_check(delta: pd.Series, stored: pd.Series, source: str, n_no_measurement: int) -> Dict[str, Any]:
    """Agreement of the first pass with stored per-pair results of the same model (pairs in both)."""
    if stored.index.duplicated().any():
        raise ValueError(f"{source}: more than one stored row for a (study, side)")
    common = delta.index.intersection(stored.index)
    a, b = delta.loc[common], stored.loc[common]
    diff = (a - b).abs().dropna()
    flips = ((a <= THRESHOLD_MM) != (b <= THRESHOLD_MM)) & a.notna() & b.notna()
    return {"reference": source, "n_compared": int(len(diff)),
            "median_abs_delta_diff_mm": float(diff.median()) if len(diff) else None,
            "max_abs_delta_diff_mm": float(diff.max()) if len(diff) else None,
            "decision_flips": int(flips.sum()), "measurement_only_on_one_side": int((a.isna() != b.isna()).sum()),
            "no_measurement": int(n_no_measurement)}


def output_summary(t: pd.DataFrame, family: str, reference: Optional[pd.Series] = None,
                   stored: Optional[Tuple[pd.Series, str]] = None) -> Dict[str, Any]:
    """Checks of one configuration's outputs on its first measured pass."""
    first = t[t["pass"] == t["pass"].min()]
    k = _delta_by_pair(first)
    out: Dict[str, Any] = {"n_pairs": int(first["pair"].nunique()), "check": None, "accuracy": None,
                           "accuracy_stored": None, "nipple_source": None}
    stored_ok = False
    if stored is not None:
        try:
            out["check"] = stored_check(k, stored[0], stored[1], int((first["decision"] == "no measurement").sum()))
            stored_ok = True
        except Exception as exc:  # noqa: BLE001 - a broken reference must not cost the timing session
            out["check"] = {"reference": stored[1], "failed": True, "error": f"{type(exc).__name__}: {exc}"}
    if reference is not None and reference.notna().any():
        out["accuracy"] = accuracy(k, reference)
        if stored_ok and family != "craunet":
            out["accuracy_stored"] = accuracy(stored[0].reindex(k.index), reference)
    if family != "craunet":
        out["nipple_source"] = {v: {str(a): int(b) for a, b in first[c].fillna("").value_counts().items()}
                                for v, c in (("MLO", "nipple_source_mlo"), ("CC", "nipple_source_cc"))}
    out["pass_max_abs_delta_diff_mm"] = pass_delta_diff(t)
    return out


def pass_delta_diff(t: pd.DataFrame) -> Optional[float]:
    """Largest |ΔD| difference of a later pass from the first one (None with one pass)."""
    first = t[t["pass"] == t["pass"].min()]
    k = _delta_by_pair(first)
    deltas = [(_delta_by_pair(g) - k).abs().dropna() for p, g in t.groupby("pass") if p != first["pass"].iloc[0]]
    return max((float(d.max()) for d in deltas if len(d)), default=0.0) if deltas else None


def batching_comparison(t: pd.DataFrame, batched: str, unbatched: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Batched against unbatched forward of the same model in one session (first pass)."""
    first = t[t["pass"] == t["pass"].min()]
    x = _delta_by_pair(first[first["configuration"] == batched])
    y = _delta_by_pair(first[first["configuration"] == unbatched]).reindex(x.index)
    f = (x - y).abs().dropna()
    return {"n": int(len(f)), "median_abs_delta_diff_mm": float(f.median()) if len(f) else None,
            "max_abs_delta_diff_mm": float(f.max()) if len(f) else None,
            "decision_differences": int(((x <= THRESHOLD_MM) != (y <= THRESHOLD_MM)).sum()),
            "speedup_total": float(summary[unbatched]["per_pair_s"]["total_s"]["median"]
                                   / summary[batched]["per_pair_s"]["total_s"]["median"])}


def session_flags(configurations: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    differ, failed = {}, []
    for name, o in configurations.items():
        ch = o.get("check")
        if ch and ch.get("failed"):
            failed.append(name)
        elif ch and ch["decision_flips"] + ch["measurement_only_on_one_side"] > 0:
            differ[name] = ch["decision_flips"] + ch["measurement_only_on_one_side"]
    return {"decisions_differ_from_stored": differ, "stored_checks_failed": failed,
            "passes_disagree_over_10pct": {a: o["pass_spread_pct"] for a, o in configurations.items()
                                           if o.get("passes_disagree")}}


def pair_set_hash(t: pd.DataFrame) -> str:
    """SHA-256 of the session's pair keys (sorted ``study,side`` lines)."""
    keys = sorted({f"{a},{b}" for a, b in zip(t["study"].astype(str), t["side"].astype(str))})
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def load1_by_pass(t: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {str(int(p)): {"start": float(g["load1"].iloc[0]), "end": float(g["load1"].iloc[-1]),
                          "median": float(g["load1"].median()), "max": float(g["load1"].max())}
            for p, g in t.groupby("pass")}


# ── several sessions ──────────────────────────────────────────────────────────────────────────────────────────

def load_session(json_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """A session's JSON and its per-measurement CSV (same name, ``.csv``)."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    if not data.get("valid", False):
        raise ValueError(f"{json_path.name}: an invalid session (it was aborted)")
    return data, pd.read_csv(json_path.with_suffix(".csv"), dtype={"study": str, "side": str})


_SAME = ("kind", "device", "threads", "batched", "preprocessing", "warmup_pairs", "n_pairs")


def compose(sessions: Sequence[Tuple[str, Dict[str, Any], pd.DataFrame]]) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """One result from the undisturbed passes of several sessions (``(label, json, rows)``, in time order; labels
    unique). The sessions must agree in device, threads, batching, preprocessing, warm-up, configurations and pairs."""
    if not sessions:
        raise ValueError("no session to compose")
    if len({label for label, _, _ in sessions}) != len(sessions):
        raise ValueError("session labels are not unique")
    first = sessions[0][1]
    for label, s, _ in sessions[1:]:
        for k in _SAME:
            if s["session"].get(k) != first["session"].get(k):
                raise ValueError(f"{label}: {k} = {s['session'].get(k)!r} differs from {first['session'].get(k)!r} "
                                 f"in {sessions[0][0]}")
        if sorted(s["configurations"]) != sorted(first["configurations"]):
            raise ValueError(f"{label}: other configurations than {sessions[0][0]}")
    passes, kept, ignored, hashes = [], [], {}, {}
    for label, js, rows in sessions:
        known = rows["configuration"].isin(list(js["configurations"]))
        if (~known).any():  # a configuration dropped after an error: its rows (warm-up incl.) are in the CSV only
            ignored[label] = {str(c): int(n) for c, n in rows.loc[~known, "configuration"].value_counts().items()}
        t = measured(rows[known]).copy()
        hashes[label] = pair_set_hash(t)
        for p in disturbed_passes(t, label):
            passes.append(p)
            if not p["disturbed"]:
                kept.append(t[t["pass"] == p["pass"]].assign(source=f"{label} pass {p['pass']}",
                                                             source_session=label, source_pass=p["pass"]))
    if len(set(hashes.values())) > 1:
        raise ValueError(f"the sessions measured different pairs (pair-set SHA-256 {hashes})")
    if not kept:
        raise ValueError("every pass of every session is disturbed: nothing to compose")
    T = pd.concat(kept, ignore_index=True)
    order = {(s, p): i + 1 for i, (s, p) in enumerate(dict.fromkeys(zip(T.source_session, T.source_pass)))}
    T["pass"] = [order[(s, p)] for s, p in zip(T.source_session, T.source_pass)]
    checked = next((p["session"] for p in passes if not p["disturbed"] and p["pass"] == 1), sessions[0][0])
    base = next(s for label, s, _ in sessions if label == checked)
    configurations = {}
    for name in base["configurations"]:  # the session's order
        t = T[T["configuration"] == name]
        if t.empty:
            continue
        o = dict(base["configurations"][name])
        o.pop("model_load_s", None)  # one session's load time, not a property of the composed passes
        o.update(timing_summary(t))
        o["pass_max_abs_delta_diff_mm"] = pass_delta_diff(t)
        if o.get("peak_extra_MB") is not None:  # from the kept measurements (no warm-up pairs)
            o["peak_extra_MB"] = float(t["gpu_extra_MB"].max())
            o["peak_gpu_MB"] = float((o.get("model_gpu_MB") or 0.0) + o["peak_extra_MB"])
        configurations[name] = o
    comparisons = {f"{c[:-len('_unbatched')]}_vs_unbatched": batching_comparison(T, c[:-len("_unbatched")], c,
                                                                                 configurations)
                   for c in configurations if c.endswith("_unbatched") and c[:-len("_unbatched")] in configurations}
    s0, s1 = sessions[0][1]["session"], sessions[-1][1]["session"]
    se = dict(s1)
    se.update(start=s0["start"], end=s1["end"], wall_s=float(T["total_s"].sum()), passes=len(order),
              pinned_cpus={label: s["session"].get("pinned_cpus") for label, s, _ in sessions},
              load1_during={"median": float(T["load1"].median()), "min": float(T["load1"].min()),
                            "max": float(T["load1"].max())},
              load1_by_pass=load1_by_pass(T),
              pairs_sha256=next(iter(hashes.values())),
              composed={"rule": DISTURBED_RULE + "; the undisturbed passes are kept, in time order",
                        "sources": [label for label, _, _ in sessions], "passes": passes, "outputs_checked_in": checked,
                        "rows_ignored": ignored,
                        "recomputed": "every timing number from the kept passes (incl. peak GPU memory without the "
                                      "warm-up pairs and the batching speed-ups); model load times dropped; output "
                                      "checks from outputs_checked_in"})
    out = dict(sessions[-1][1])
    out.update(session=se, configurations=configurations, disturbed_passes=passes, card=None, comparisons=comparisons,
               **session_flags(configurations))
    out["errors"], out["rows_ignored"] = {}, {}
    return out, T.drop(columns=["source_session", "source_pass"])


# ── report ────────────────────────────────────────────────────────────────────────────────────────────────────

def _f(x: Optional[float], nd: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{nd}f}"


def report_markdown(results: Sequence[Tuple[str, Dict[str, Any]]]) -> str:
    """English Markdown report of sessions (and composed results), plus the Table 11 layout of the last result of
    each device kind."""
    L = ["# Runtime benchmark", "",
         "One MLO-CC pair end to end, from the two DICOM files to the adequacy decision. Seconds per pair: median "
         "[q1-q3] over all measurements (pairs x passes); pairs per minute = 60 x measurements / summed time; spread "
         f"= (max - min) / mean of the passes. Disturbed pass: {DISTURBED_RULE}.", ""]
    last: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for label, s in results:
        se, K = s["session"], s["configurations"]
        last[se.get("kind", "?")] = (label, s)
        L += [f"## {label} ({se.get('device')})", "",
              f"- {(se.get('start') or {}).get('time')} to {(se.get('end') or {}).get('time')}; {se.get('n_pairs')} pairs x "
              f"{se.get('passes')} pass(es) + {se.get('warmup_pairs')} warm-up pairs; {se.get('threads')} threads pinned "
              f"to {se.get('pinned_cpus')}; preprocessing {se.get('preprocessing')}; batched {se.get('batched')}."]
        dist = s.get("disturbed_passes") or []
        if dist:
            L.append("- Passes: " + "; ".join(
                f"{(p['session'] + ' ') if p.get('session') else ''}pass {p['pass']} {p['n_over']}/{p['n_measurements']} "
                f"slow ({100 * p['share_over']:.1f} %) -> {'DISTURBED' if p['disturbed'] else 'undisturbed'}"
                for p in dist) + ".")
        if se.get("composed"):
            L.append(f"- Composed from the undisturbed passes of {se['composed']['sources']}; outputs checked in "
                     f"{se['composed']['outputs_checked_in']}.")
        if s.get("skipped"):
            L.append(f"- Skipped: {s['skipped']}.")
        if s.get("errors"):
            L.append(f"- **Errors**: {list(s['errors'])}.")
        if s.get("stored_checks_failed"):
            L.append(f"- **Stored check failed** for {s['stored_checks_failed']} (see `check.error` in the JSON).")
        if s.get("rows_ignored"):
            L.append(f"- Rows of configurations dropped after an error, not summarised: {s['rows_ignored']}.")
        if s.get("decisions_differ_from_stored"):
            L.append(f"- Decisions differing from the stored predictions (pairs): {s['decisions_differ_from_stored']}.")
        if s.get("passes_disagree_over_10pct"):
            L.append(f"- **Passes disagree by more than 10 %**: {list(s['passes_disagree_over_10pct'])}.")
        L += ["", "| Configuration | s per pair, median [IQR] | pairs/min | pairs/min by pass | spread, % "
                  "| read / pre / infer / post, ms | measurements |", "|---|---|---|---|---|---|---|"]
        for name, o in K.items():
            t, pp = o["per_pair_s"]["total_s"], o["per_pair_s"]
            stages = " / ".join(_f(1000 * pp[k]["median"], 0) for k in ("read_s", "pre_s", "infer_s", "post_s"))
            L.append(f"| `{name}` | {_f(t['median'], 3)} [{_f(t['q1'], 3)}-{_f(t['q3'], 3)}] | {_f(o['pairs_per_min'])} | "
                     f"{' / '.join(_f(p['pairs_per_min']) for p in o.get('passes', []))} | "
                     f"{_f(o['pass_spread_pct']['pairs_per_min'])}{' (> 10)' if o.get('passes_disagree') else ''} | "
                     f"{stages} | {o['n_measurements']} |")
        L.append("")
    if last:
        kinds = [k for k in ("gpu", "cpu") if k in last] + [k for k in last if k not in ("gpu", "cpu")]
        L += ["## Table 11 layout", "",
              "Source per device: " + "; ".join(f"{k.upper()} = {last[k][0]}" for k in kinds) + ".", "",
              "| Pipeline | Flip | " + " | ".join(f"{k.upper()}: s per pair (pairs/min)" for k in kinds) + " |",
              "|---|---|" + "---|" * len(kinds)]
        rows: Dict[Tuple[str, str], Dict[str, str]] = {}
        for k in kinds:
            for name, o in last[k][1]["configurations"].items():
                if name.endswith("_unbatched"):
                    continue
                flip = "on" if o.get("tta") else ("off" if o.get("flip_option") else "single setting")
                rows.setdefault((o.get("pipeline", name), flip), {})[k] = (
                    f"{_f(o['per_pair_s']['total_s']['median'], 3)} ({_f(o['pairs_per_min'])})")
        for (pipe, flip), cells in rows.items():
            L.append(f"| {pipe} | {flip} | " + " | ".join(cells.get(k, "not measured") for k in kinds) + " |")
        L.append("")
    return "\n".join(L)
