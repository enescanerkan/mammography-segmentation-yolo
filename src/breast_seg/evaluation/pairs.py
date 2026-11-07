"""MLO/CC pairs: build the pair list from the label tables and score ``delta = |PNL_MLO - D_CC|`` per pair."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from breast_seg.evaluation.rule import FLAT_RULE

PAIR_COLUMNS = ["group", "side", "mlo_stem", "cc_stem", "truth"]


def pairs_from_labels(mlo_labels: pd.DataFrame, cc_labels: pd.DataFrame, split: Optional[str] = None,
                      cc_prefix: str = "cc_") -> pd.DataFrame:
    """One row per breast: the study's MLO and CC image of the same side (first letter of ``SeriesDescription``).

    ``split`` keeps only rows whose ``Split`` column equals it. ``truth`` is the CC table's ``qualitativeLabel``
    (lower case).
    """
    cols = ["StudyInstanceUID", "SOPInstanceUID", "SeriesDescription"]
    if split is not None:
        mlo_labels = mlo_labels[mlo_labels["Split"] == split]
        cc_labels = cc_labels[cc_labels["Split"] == split]
    cc = cc_labels[cols + ["qualitativeLabel"]].drop_duplicates(subset=cols).copy()
    mlo = mlo_labels[cols].drop_duplicates().copy()
    cc["side"] = cc.SeriesDescription.astype(str).str[0]
    mlo["side"] = mlo.SeriesDescription.astype(str).str[0]
    j = cc.merge(mlo, on=["StudyInstanceUID", "side"], suffixes=("_cc", "_mlo"))
    out = pd.DataFrame({"group": j.StudyInstanceUID.astype(str), "side": j.side,
                        "mlo_stem": j.SOPInstanceUID_mlo.astype(str), "cc_stem": cc_prefix + j.SOPInstanceUID_cc.astype(str),
                        "truth": j.qualitativeLabel.astype(str).str.strip().str.lower()})
    return out.reset_index(drop=True)


def pairs_from_fold_table(df: pd.DataFrame, group: str = "StudyInstanceUID", side: str = "Side",
                          mlo: str = "SOPInstanceUID_mlo", cc: str = "SOPInstanceUID_cc", truth: str = "qualitativeLabel",
                          fold: Optional[str] = "fold", cc_prefix: str = "cc_") -> pd.DataFrame:
    """Pair-level table (e.g. a fold-assignment CSV) -> the pair format of this module (plus ``fold``)."""
    out = pd.DataFrame({"group": df[group].astype(str), "side": df[side].astype(str), "mlo_stem": df[mlo].astype(str),
                        "cc_stem": cc_prefix + df[cc].astype(str), "truth": df[truth].astype(str).str.strip().str.lower()})
    if fold and fold in df:
        out["fold"] = df[fold].astype(int).to_numpy()
    return out


def score_pairs(views: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Join per-image measurements (``views`` indexed by stem, with ``pnl_mm``, ``gt_pnl_mm``, ``cc_mm``,
    ``gt_cc_mm`` where available) onto ``pairs``; adds ``delta_mm``, ``gt_delta_mm`` and the flat-rule ``pred``."""
    v = views.copy()
    for col in ("pnl_mm", "gt_pnl_mm", "cc_mm", "gt_cc_mm"):
        if col not in v:
            v[col] = np.nan
    out = pairs.copy()
    out["pnl_mlo_mm"] = out.mlo_stem.map(v.pnl_mm)
    out["d_cc_mm"] = out.cc_stem.map(v.cc_mm)
    out["gt_pnl_mm"] = out.mlo_stem.map(v.gt_pnl_mm)
    out["gt_d_cc_mm"] = out.cc_stem.map(v.gt_cc_mm)
    out["delta_mm"] = (out.pnl_mlo_mm - out.d_cc_mm).abs()
    out["gt_delta_mm"] = (out.gt_pnl_mm - out.gt_d_cc_mm).abs()
    out["pred"] = FLAT_RULE.predict(out.delta_mm).to_numpy()
    return out
