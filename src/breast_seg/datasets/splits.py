"""Group-level (study-level) splits, fold assignments and leakage checks.

Every split in this package is made on a *group* (a study: the MLO and CC images of one woman's examination share
it), never on single images, so that no study is in two splits. :func:`find_leaks` is the audit used by the dataset
builder and the cross-validation runner.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

SPLITS = ("train", "val", "test")


class LeakageError(ValueError):
    """A group occurs in more than one split / fold."""


def split_groups(groups: Iterable[str], ratios: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 42,
                 names: Sequence[str] = SPLITS) -> Dict[str, str]:
    """Shuffle the distinct groups with ``seed`` and cut them by ``ratios`` -> ``{group: split}``.

    The held-out splits (all but the first) get ``max(1, int(ratio * n))`` groups each, taken from the front of the
    shuffled list (the last name first, as the original builders did: test, then val); the rest is the first
    split. Deterministic for a given set of groups and seed.
    """
    if len(ratios) != len(names):
        raise ValueError("ratios and names differ in length")
    items = sorted(set(groups))
    random.Random(seed).shuffle(items)
    n = len(items)
    out: Dict[str, str] = {}
    start = 0
    for name, ratio in zip(reversed(names[1:]), reversed(ratios[1:])):
        k = max(1, int(ratio * n)) if n else 0
        for g in items[start:start + k]:
            out[g] = name
        start += k
    for g in items[start:]:
        out[g] = names[0]
    return out


def find_leaks(split_of: Mapping[str, str], groups_of: Mapping[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """``{group: {splits}}`` for every group that occurs in more than one split.

    ``split_of`` maps item -> split, ``groups_of`` item -> its group keys (an item may carry several, e.g. a study
    and a series key); items without groups are ignored.
    """
    seen: Dict[str, Set[str]] = {}
    for item, split in split_of.items():
        for g in groups_of.get(item, ()):
            if g:
                seen.setdefault(g, set()).add(split)
    return {g: s for g, s in seen.items() if len(s) > 1}


def assert_no_leaks(split_of: Mapping[str, str], groups_of: Mapping[str, Iterable[str]]) -> None:
    """Raise :class:`LeakageError` if :func:`find_leaks` finds anything."""
    leaks = find_leaks(split_of, groups_of)
    if leaks:
        sample = ", ".join(f"{sorted(v)}" for v in list(leaks.values())[:3])
        raise LeakageError(f"{len(leaks)} group(s) occur in more than one split (e.g. {sample})")


@dataclass(frozen=True)
class FoldSplit:
    """The train / val / test stems of one cross-validation fold."""

    fold: int
    train: Tuple[str, ...]
    val: Tuple[str, ...]
    test: Tuple[str, ...]
    unlabelled: Tuple[str, ...] = ()

    def split_of(self) -> Dict[str, str]:
        out = {s: "train" for s in self.train}
        out.update({s: "val" for s in self.val})
        out.update({s: "test" for s in self.test})
        return out

    def check(self) -> None:
        """Disjointness and "unlabelled images only in test"."""
        tr, va, te = set(self.train), set(self.val), set(self.test)
        if tr & va or tr & te or va & te:
            raise LeakageError(f"fold {self.fold}: train / val / test overlap")
        if set(self.unlabelled) & (tr | va):
            raise LeakageError(f"fold {self.fold}: an unlabelled image is in train or val")


@dataclass
class FoldAssignment:
    """Image-level fold assignment (one row per image) read from a CSV.

    Protocol of fold ``i``: test = fold ``i``, validation = fold ``(i + val_offset) % k``, train = the remaining
    folds. Unlabelled images never enter train / val; they are only predicted in their own test fold.
    """

    fold_of: Dict[str, int]
    group_of: Dict[str, str] = field(default_factory=dict)
    labelled: Dict[str, bool] = field(default_factory=dict)
    n_folds: int = 10
    val_offset: int = 1

    @classmethod
    def from_csv(cls, path: Union[str, Path], stem_column: str = "stem", fold_column: str = "fold",
                 group_column: Optional[str] = None, labelled_column: Optional[str] = None,
                 n_folds: Optional[int] = None, val_offset: int = 1) -> FoldAssignment:
        df = pd.read_csv(path, dtype={stem_column: str})
        return cls.from_frame(df, stem_column, fold_column, group_column, labelled_column, n_folds, val_offset)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, stem_column: str = "stem", fold_column: str = "fold",
                   group_column: Optional[str] = None, labelled_column: Optional[str] = None,
                   n_folds: Optional[int] = None, val_offset: int = 1) -> FoldAssignment:
        missing = [c for c in (stem_column, fold_column, group_column, labelled_column) if c and c not in df.columns]
        if missing:
            raise ValueError(f"fold table lacks column(s) {missing}")
        if df[stem_column].duplicated().any():
            raise ValueError("fold table lists an image twice")
        fold_of = {str(s): int(f) for s, f in zip(df[stem_column], df[fold_column])}
        group_of = ({str(s): str(g) for s, g in zip(df[stem_column], df[group_column]) if pd.notna(g) and str(g)}
                    if group_column else {})
        labelled = ({str(s): bool(int(v)) if not isinstance(v, bool) else v for s, v in zip(df[stem_column], df[labelled_column])}
                    if labelled_column else {})
        k = n_folds if n_folds is not None else (max(fold_of.values()) + 1 if fold_of else 0)
        out = cls(fold_of, group_of, labelled, k, val_offset)
        out.check()
        return out

    def check(self) -> None:
        bad = sorted({f for f in self.fold_of.values() if not 0 <= f < self.n_folds})
        if bad:
            raise ValueError(f"fold ids {bad} outside 0..{self.n_folds - 1}")
        by_group: Dict[str, Set[int]] = {}
        for stem, g in self.group_of.items():
            by_group.setdefault(g, set()).add(self.fold_of[stem])
        split = [g for g, f in by_group.items() if len(f) > 1]
        if split:
            raise LeakageError(f"{len(split)} group(s) are spread over several folds")

    def is_labelled(self, stem: str) -> bool:
        return self.labelled.get(stem, True)

    def extend_by_group(self, stems: Iterable[str], group_of: Mapping[str, str]) -> List[str]:
        """Assign images that are missing from the table to the fold of their group (no leakage through a study
        that is already assigned). Returns the stems that stayed unassigned."""
        fold_of_group = {g: self.fold_of[s] for s, g in self.group_of.items()}
        for s, g in group_of.items():
            if s in self.fold_of and g:
                fold_of_group.setdefault(g, self.fold_of[s])
        left = []
        for stem in stems:
            if stem in self.fold_of:
                continue
            g = group_of.get(stem)
            if g and g in fold_of_group:
                self.fold_of[stem] = fold_of_group[g]
                self.group_of[stem] = g
            else:
                left.append(stem)
        self.check()
        return left

    def split(self, fold: int) -> FoldSplit:
        """Train / val / test of ``fold`` under the protocol above."""
        if not 0 <= fold < self.n_folds:
            raise ValueError(f"fold {fold} outside 0..{self.n_folds - 1}")
        val_fold = (fold + self.val_offset) % self.n_folds
        train, val, test, unl = [], [], [], []
        for stem in sorted(self.fold_of):
            f = self.fold_of[stem]
            if f == fold:
                test.append(stem)
                if not self.is_labelled(stem):
                    unl.append(stem)
            elif not self.is_labelled(stem):
                continue
            elif f == val_fold:
                val.append(stem)
            else:
                train.append(stem)
        out = FoldSplit(fold, tuple(train), tuple(val), tuple(test), tuple(unl))
        out.check()
        if self.group_of:
            assert_no_leaks(out.split_of(), {s: [g] for s, g in self.group_of.items()})
        return out

    def to_frame(self) -> pd.DataFrame:
        rows = [{"stem": s, "fold": f, "group": self.group_of.get(s, ""), "labelled": int(self.is_labelled(s))}
                for s, f in sorted(self.fold_of.items())]
        return pd.DataFrame(rows, columns=["stem", "fold", "group", "labelled"])


def folds_from_pairs(df: pd.DataFrame, mlo_column: str, cc_column: str, fold_column: str = "fold",
                     group_column: Optional[str] = None, mlo_prefix: str = "", cc_prefix: str = "cc_") -> pd.DataFrame:
    """Pair-level fold table (one row per MLO/CC pair) -> image-level table with ``stem, fold[, group]``.

    Stems follow the dataset naming: ``mlo_prefix + id`` and ``cc_prefix + id``.
    """
    rows = []
    for d in df.to_dict("records"):
        g = d.get(group_column, "") if group_column else ""
        rows.append({"stem": f"{mlo_prefix}{d[mlo_column]}", "fold": int(d[fold_column]), "group": g})
        rows.append({"stem": f"{cc_prefix}{d[cc_column]}", "fold": int(d[fold_column]), "group": g})
    return pd.DataFrame(rows, columns=["stem", "fold", "group"])
