"""The PNL positioning rule and its decision metrics.

For an MLO/CC pair of one breast, ``delta = |PNL_MLO - D_CC|``. The pair is *good* when ``delta <= 10 mm`` and *bad*
otherwise (:data:`THRESHOLD_MM`).

Two scorings of a model's ``delta`` against the reference label are reported:

* **Flat 10 mm** (``lo = hi = 10``): the deployable rule itself.
* **+-1 mm tolerance** (``lo = 9, hi = 11``): the threshold is chosen *by the reference class*; a reference-good
  pair counts as correct when ``delta <= 11``, a reference-bad pair when ``delta > 9``. Measurements falling in
  ``(9, 11]`` therefore agree with either label. Because it looks at the reference it is an evaluation tolerance,
  not a decision rule.

Counting convention (both scorings): a pair without a measurement is never correct and never leaves a denominator::

    TP = answered & predicted good & ref good      FN = ref good & not TP   (unanswered ref-good included)
    FP = answered & predicted good & ref bad       TN = answered & predicted bad & ref bad
    U_bad = unanswered & ref bad                   (outside the matrix, inside the specificity denominator)

    Recall = TP / (TP + FN)    Specificity = TN / (TN + FP + U_bad)    Accuracy = (TP + TN) / n
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

THRESHOLD_MM = 10.0
TOLERANCE_MM = 1.0


@dataclass(frozen=True)
class PNLRule:
    """``delta <= threshold_mm`` -> good. ``tolerance_mm > 0`` scores with the reference-dependent band."""

    threshold_mm: float = THRESHOLD_MM
    tolerance_mm: float = 0.0

    @property
    def lo(self) -> float:
        return self.threshold_mm - self.tolerance_mm

    @property
    def hi(self) -> float:
        return self.threshold_mm + self.tolerance_mm

    def predict(self, delta: Any) -> pd.Series:
        """``"good" / "bad"`` with the flat threshold; ``""`` where there is no measurement."""
        d = pd.Series(delta, dtype=float)
        return pd.Series(np.where(d.isna(), "", np.where(d <= self.threshold_mm, "good", "bad")), index=d.index)

    def correct(self, delta: Any, ref_good: Any) -> pd.Series:
        """Per-pair agreement with the reference (``False`` without a measurement)."""
        d = pd.Series(delta, dtype=float).reset_index(drop=True)
        good = pd.Series(ref_good).reset_index(drop=True).astype(bool)
        return d.notna() & pd.Series(np.where(good, d <= self.hi, d > self.lo))

    def matrix(self, delta: Any, ref_good: Any, n_total: Optional[int] = None) -> Dict[str, float]:
        """Confusion counts and rates (percent) under the counting convention of this module.

        ``n_total`` larger than the number of pairs adds unmatched pairs to the denominator (as failures).
        """
        d = pd.Series(delta, dtype=float).reset_index(drop=True)
        good = pd.Series(ref_good).reset_index(drop=True).astype(bool)
        if len(d) != len(good):
            raise ValueError(f"{len(d)} measurements but {len(good)} reference labels")
        answered = d.notna()
        pred_good = answered & pd.Series(np.where(good, d <= self.hi, d <= self.lo))
        tp = int((pred_good & good).sum())
        fn = int((~pred_good & good).sum())
        fp = int((pred_good & ~good).sum())
        tn = int((answered & ~pred_good & ~good).sum())
        u_bad = int((~answered & ~good).sum())
        n = len(good) if n_total is None else int(n_total)
        if n < len(good):
            raise ValueError(f"n_total {n} < number of pairs {len(good)}")

        def pct(a: int, b: int) -> float:
            return float("nan") if b == 0 else 100.0 * a / b

        pre, rec = pct(tp, tp + fp), pct(tp, tp + fn)
        f1 = float("nan") if np.isnan(pre) or np.isnan(rec) or pre + rec == 0 else 2 * pre * rec / (pre + rec)
        return {"n": n, "n_fail": int((~answered).sum()) + n - len(good), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "U_bad": u_bad, "accuracy": pct(tp + tn, n), "precision": pre, "recall": rec,
                "specificity": pct(tn, tn + fp + u_bad), "f1": f1}


FLAT_RULE = PNLRule()
TOLERANCE_RULE = PNLRule(tolerance_mm=TOLERANCE_MM)


def ref_is_good(labels: Iterable[Any]) -> pd.Series:
    """Reference labels (``Good`` / ``good`` / ``Bad`` ...) -> boolean series."""
    return pd.Series([str(x).strip().lower() == "good" for x in labels])


def decision_summary(delta: Any, ref_good: Any, n_total: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """Both scorings: ``{"flat_10mm": matrix, "tolerance_1mm": matrix}``."""
    return {"flat_10mm": FLAT_RULE.matrix(delta, ref_good, n_total),
            "tolerance_1mm": TOLERANCE_RULE.matrix(delta, ref_good, n_total)}
