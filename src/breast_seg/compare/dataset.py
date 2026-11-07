"""
SOLID OOP: Dataset Management and Data Loaders
"""
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class TestPair:
    study_uid: str
    laterality: str
    cc_sop: str
    mlo_sop: str


class MammographyDataset:
    """Manages loading and pairing MLO and CC test splits."""
    def __init__(self, labels_dir: Path, cc_dir: Path, mlo_dir: Path):
        self.labels_dir = labels_dir
        self.cc_dir = cc_dir
        self.mlo_dir = mlo_dir
        
        self.cc_labels = pd.read_csv(self.labels_dir / "cc_labels.csv")
        self.mlo_labels = pd.read_csv(self.labels_dir / "mlo_labels.csv")
        self.cc_meta = pd.read_csv(self.cc_dir / "metadata.csv")
        self.mlo_meta = pd.read_csv(self.mlo_dir / "metadata.csv")

    def get_test_pairs(self, limit: int = 0) -> List[TestPair]:
        cc_t = self.cc_labels[self.cc_labels["Split"] == "Test"].copy()
        mlo_t = self.mlo_labels[self.mlo_labels["Split"] == "Test"].copy()
        
        def _lat_cc(desc: str) -> Optional[str]:
            s = str(desc)
            if "L-CC" in s or s.strip() == "L-CC": return "L"
            if "R-CC" in s or s.strip() == "R-CC": return "R"
            return None
            
        def _lat_mlo(desc: str) -> Optional[str]:
            s = str(desc)
            if "L-MLO" in s: return "L"
            if "R-MLO" in s: return "R"
            return None

        cc_t["_lat"] = cc_t["SeriesDescription"].map(_lat_cc)
        mlo_t["_lat"] = mlo_t["SeriesDescription"].map(_lat_mlo)

        pairs: List[TestPair] = []
        for _, r in cc_t.iterrows():
            st = str(r["StudyInstanceUID"])
            lat = r["_lat"]
            if lat is None:
                continue
                
            m = mlo_t[(mlo_t["StudyInstanceUID"] == st) & 
                      (mlo_t["_lat"] == lat) & 
                      (mlo_t["labelName"] == "Nipple")]
            if len(m) == 0:
                continue
                
            pairs.append(
                TestPair(
                    study_uid=st,
                    laterality=lat,
                    cc_sop=str(r["SOPInstanceUID"]),
                    mlo_sop=str(m.iloc[0]["SOPInstanceUID"]),
                )
            )
            
        if limit > 0:
            pairs = pairs[:limit]
        return pairs

    def get_cc_metadata(self, sop_uid: str) -> Optional[pd.Series]:
        m = self.cc_meta[self.cc_meta["sop_uid"].astype(str) == str(sop_uid)]
        return m.iloc[0] if not m.empty else None

    def get_mlo_metadata(self, sop_uid: str) -> Optional[pd.Series]:
        m = self.mlo_meta[self.mlo_meta["sop_uid"].astype(str) == str(sop_uid)]
        return m.iloc[0] if not m.empty else None

    def get_mlo_gt_keypoints(self, sop_uid: str) -> Optional[Dict[str, Tuple[float, float]]]:
        rows = self.mlo_labels[self.mlo_labels["SOPInstanceUID"].astype(str) == str(sop_uid)]
        if rows.empty:
            return None
            
        keypoints: Dict[str, Tuple[float, float]] = {}
        for _, row in rows.iterrows():
            label = str(row.get("labelName", "")).strip()
            raw = row.get("data", "")
            if pd.isna(raw): continue
            
            try:
                data = ast.literal_eval(str(raw))
            except (ValueError, SyntaxError):
                continue
                
            if label == "Nipple":
                keypoints["nipple"] = (float(data["x"] + data["width"] / 2), float(data["y"] + data["height"] / 2))
            elif label == "Pectoralis":
                raw_pts = data.get("vertices", [])
                if not raw_pts: continue
                try:
                    pts = [(float(p[0]), float(p[1])) for p in raw_pts]
                except (IndexError, TypeError, ValueError):
                    continue
                if len(pts) < 2: continue
                by_y = sorted(pts, key=lambda p: p[1])
                keypoints["pectoral_top"] = (by_y[0][0], by_y[0][1])
                keypoints["pectoral_bottom"] = (by_y[-1][0], by_y[-1][1])
                
        if not {"nipple", "pectoral_top", "pectoral_bottom"}.issubset(keypoints):
            return None
        return keypoints

    def get_cc_gt_nipple(self, sop_uid: str) -> Optional[Tuple[float, float]]:
        rows = self.cc_labels[(self.cc_labels["SOPInstanceUID"].astype(str) == str(sop_uid)) & 
                              (self.cc_labels["labelName"] == "Nipple")]
        if rows.empty: return None
        raw = rows.iloc[0].get("data", "")
        if pd.isna(raw): return None
        try:
            data = ast.literal_eval(str(raw))
            return (float(data["x"] + data["width"] / 2), float(data["y"] + data["height"] / 2))
        except (ValueError, SyntaxError):
            return None

    def get_qualitative_label(self, sop_uid: str, view: str) -> Optional[str]:
        if view.upper() == "CC":
            rows = self.cc_labels[(self.cc_labels["SOPInstanceUID"].astype(str) == str(sop_uid)) & 
                                  (self.cc_labels["labelName"] == "Nipple")]
            if not rows.empty and not pd.isna(rows.iloc[0].get("qualitativeLabel", "")):
                return str(rows.iloc[0].get("qualitativeLabel", "")).strip()
        else:
            rows = self.mlo_labels[self.mlo_labels["SOPInstanceUID"].astype(str) == str(sop_uid)]
            if not rows.empty:
                for col in ("qualitativeLabel", "quality", "labelQuality"):
                    if col in rows.columns:
                        for v in rows[col].values:
                            if pd.notna(v) and str(v).strip() and str(v).strip().lower() not in ("nan", "none"):
                                return str(v).strip()
        return None
