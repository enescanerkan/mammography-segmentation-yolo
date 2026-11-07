"""Paired comparison and resampling helpers (no SciPy dependency)."""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd


def binomial_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test of ``k`` successes in ``n`` trials against ``p = 0.5``."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


@dataclass(frozen=True)
class McNemarResult:
    """Discordant pairs of two classifiers scored on the same cases and the exact McNemar p-value."""

    only_a: int
    only_b: int
    p_value: float

    @property
    def n_discordant(self) -> int:
        return self.only_a + self.only_b


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Exact McNemar test (binomial on the discordant pairs) for two per-case correctness vectors."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    if a.shape != b.shape:
        raise ValueError("both vectors must score the same cases")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    return McNemarResult(only_a, only_b, binomial_two_sided_p(only_a, only_a + only_b))


def paired_mcnemar(pairs_a: pd.DataFrame, pairs_b: pd.DataFrame, correct: Callable[[pd.DataFrame], pd.Series],
                   key: Sequence[str] = ("group", "side")) -> McNemarResult:
    """McNemar on two per-pair tables joined on ``key``; the joined reference labels must agree.

    ``correct(df)`` returns the per-pair correctness (e.g. ``lambda d: TOLERANCE_RULE.correct(d.delta_mm,
    d.truth == "good")``).
    """
    key = list(key)
    j = pairs_a.merge(pairs_b, on=key, suffixes=("_a", "_b"))
    if len(j) != len(pairs_a) or len(j) != len(pairs_b):
        raise ValueError("the two tables do not cover the same pairs")
    if "truth_a" in j and not (j.truth_a.str.lower() == j.truth_b.str.lower()).all():
        raise ValueError("the reference labels differ between the two tables")
    cols = [c for c in pairs_a.columns if c not in key]
    ca = correct(j.rename(columns={f"{c}_a": c for c in cols}))
    cb = correct(j.rename(columns={f"{c}_b": c for c in cols}))
    return mcnemar_exact(ca.to_numpy(), cb.to_numpy())


def group_bootstrap(df: pd.DataFrame, group_column: str, statistic: Callable[[pd.DataFrame], float],
                    n_resamples: int = 2000, seed: int = 42, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile CI of ``statistic`` resampling whole groups (e.g. both breasts of a study together)."""
    rng = np.random.default_rng(seed)
    groups: Dict[object, np.ndarray] = {g: idx.to_numpy() for g, idx in df.groupby(group_column).groups.items()}
    keys = list(groups)
    values = []
    for _ in range(n_resamples):
        pick = rng.choice(len(keys), len(keys), replace=True)
        idx = np.concatenate([groups[keys[i]] for i in pick])
        values.append(statistic(df.loc[idx]))
    return float(np.percentile(values, 100 * alpha / 2)), float(np.percentile(values, 100 * (1 - alpha / 2)))
