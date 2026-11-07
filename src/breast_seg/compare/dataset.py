"""The compare dataset: label tables, canvas metadata and MLO/CC pairing of one split.

Layout of ``compare_root``::

    labels/mlo_labels.csv, labels/cc_labels.csv     reference annotations (see breast_seg.evaluation.gt)
    MLO/metadata.csv, CC/metadata.csv               canvas transform + pixel spacing per image (sop_uid)
    MLO/images/<split>/<sop>.png, CC/images/...     640 canvases
    test_dicom/<study>/<sop>.dicom                  optional originals for the visualisations
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from breast_seg.evaluation.gt import cc_nipples, mlo_landmarks, quality_labels


@dataclass(frozen=True)
class TestPair:
    study_uid: str
    laterality: str
    cc_sop: str
    mlo_sop: str


def _side(desc: str, view: str) -> Optional[str]:
    s = str(desc)
    for lat in ("L", "R"):
        if f"{lat}-{view}" in s:
            return lat
    return None


class MammographyDataset:
    """Loads the tables once; pairs the CC and MLO image of the same study and side."""

    def __init__(self, labels_dir: Path, cc_dir: Path, mlo_dir: Path, split: str = "Test"):
        self.labels_dir, self.cc_dir, self.mlo_dir, self.split = Path(labels_dir), Path(cc_dir), Path(mlo_dir), split
        self.cc_labels = pd.read_csv(self.labels_dir / "cc_labels.csv", dtype={"SOPInstanceUID": str, "StudyInstanceUID": str})
        self.mlo_labels = pd.read_csv(self.labels_dir / "mlo_labels.csv", dtype={"SOPInstanceUID": str, "StudyInstanceUID": str})
        self.cc_meta = pd.read_csv(self.cc_dir / "metadata.csv", dtype={"sop_uid": str}).set_index("sop_uid")
        self.mlo_meta = pd.read_csv(self.mlo_dir / "metadata.csv", dtype={"sop_uid": str}).set_index("sop_uid")
        self._mlo_gt = mlo_landmarks(self.mlo_labels)
        self._cc_gt = cc_nipples(self.cc_labels)
        self._cc_quality = quality_labels(self.cc_labels[self.cc_labels["labelName"] == "Nipple"])
        self._mlo_quality = quality_labels(self.mlo_labels)

    @classmethod
    def from_root(cls, compare_root: Path, split: str = "Test") -> MammographyDataset:
        root = Path(compare_root)
        return cls(root / "labels", root / "CC", root / "MLO", split)

    def get_test_pairs(self, limit: int = 0) -> List[TestPair]:
        cc = self.cc_labels[self.cc_labels["Split"] == self.split].copy()
        mlo = self.mlo_labels[self.mlo_labels["Split"] == self.split].copy()
        cc["_lat"] = cc["SeriesDescription"].map(lambda d: _side(d, "CC"))
        mlo["_lat"] = mlo["SeriesDescription"].map(lambda d: _side(d, "MLO"))
        pairs: List[TestPair] = []
        for _, r in cc.iterrows():
            if r["_lat"] is None:
                continue
            m = mlo[(mlo["StudyInstanceUID"] == r["StudyInstanceUID"]) & (mlo["_lat"] == r["_lat"])
                    & (mlo["labelName"] == "Nipple")]
            if len(m):
                pairs.append(TestPair(str(r["StudyInstanceUID"]), r["_lat"], str(r["SOPInstanceUID"]),
                                      str(m.iloc[0]["SOPInstanceUID"])))
        return pairs[:limit] if limit > 0 else pairs

    @staticmethod
    def _row(meta: pd.DataFrame, sop: str) -> Optional[pd.Series]:
        if sop not in meta.index:
            return None
        row = meta.loc[sop]
        row = (row.iloc[0] if isinstance(row, pd.DataFrame) else row).copy()
        row["sop_uid"] = sop
        return row

    def get_cc_metadata(self, sop: str) -> Optional[pd.Series]:
        return self._row(self.cc_meta, str(sop))

    def get_mlo_metadata(self, sop: str) -> Optional[pd.Series]:
        return self._row(self.mlo_meta, str(sop))

    def get_mlo_gt_keypoints(self, sop: str) -> Optional[Dict[str, Tuple[float, float]]]:
        """``{"nipple", "pectoral_top", "pectoral_bottom"}`` in original coordinates, or ``None``."""
        d = self._mlo_gt.get(str(sop))
        if d is None:
            return None
        return {"nipple": tuple(map(float, d["nipple"])), "pectoral_top": tuple(map(float, d["pec_top"])),
                "pectoral_bottom": tuple(map(float, d["pec_bot"]))}

    def get_cc_gt_nipple(self, sop: str) -> Optional[Tuple[float, float]]:
        p = self._cc_gt.get(str(sop))
        return None if p is None else (float(p[0]), float(p[1]))

    def get_qualitative_label(self, sop: str, view: str) -> Optional[str]:
        return (self._cc_quality if view.upper() == "CC" else self._mlo_quality).get(str(sop))
