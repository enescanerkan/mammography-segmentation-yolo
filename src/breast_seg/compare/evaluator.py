"""Inference of both routes on one image pair, and the summary metrics of the compare run."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from breast_seg.compare.geometry import GeometryUtils
from breast_seg.config import Config
from breast_seg.evaluation.rule import FLAT_RULE, THRESHOLD_MM, ref_is_good


def _chest_wall(nipple_dicom, meta_row: pd.Series):
    lat = str(meta_row.get("laterality", "L")).upper()
    wall = (0.0 if lat == "L" else float(meta_row["original_width"]) - 1.0, nipple_dicom[1])
    return wall, GeometryUtils.dicom_to_640(wall[0], wall[1], meta_row)


class InferenceEngine:
    """Runs the pose and segmentation models and maps their landmarks to original-image coordinates."""

    def __init__(self, pose_mlo: Any, pose_cc: Any, seg_model: Any, seg_config: Optional[Config] = None, analyzer: Any = None):
        self.pose_mlo, self.pose_cc, self.seg_model = pose_mlo, pose_cc, seg_model
        self.seg_config = seg_config or Config()
        if analyzer is None and seg_model is not None:
            from breast_seg.analyzer import MLOAnalyzer

            analyzer = MLOAnalyzer(self.seg_config)
        self.analyzer = analyzer

    def run_mlo_pose(self, image_path: Path, meta: pd.Series) -> Optional[Dict[str, Any]]:
        k = self.pose_mlo.predict(image_path) if self.pose_mlo is not None else None
        if k is None or len(k) < 3:
            return None
        nd, p1d, p2d = (GeometryUtils.inverse_transform_640(p[0], p[1], meta) for p in k[:3])
        fd = GeometryUtils.foot_on_infinite_line(nd, p1d, p2d)
        return {"kpts_640": [(float(p[0]), float(p[1])) for p in k[:3]], "kpts_dicom": [nd, p1d, p2d],
                "foot_dicom": fd, "foot_640": GeometryUtils.dicom_to_640(fd[0], fd[1], meta)}

    def run_cc_pose(self, image_path: Path, meta: pd.Series) -> Optional[Dict[str, Any]]:
        k = self.pose_cc.predict(image_path) if self.pose_cc is not None else None
        if k is None or len(k) < 1:
            return None
        nd = GeometryUtils.inverse_transform_640(k[0][0], k[0][1], meta)
        wall, wall640 = _chest_wall(nd, meta)
        return {"nipple_640": (float(k[0][0]), float(k[0][1])), "nipple_dicom": nd, "wall_dicom": wall,
                "wall_point_640": wall640}

    def _segment(self, image_path: Path):
        res = self.seg_model.predict(image_path, self.seg_config)
        if res is None:
            return None, None
        out = self.analyzer.analyze_single(res)
        if not out.has_nipple or out.nipple_mask is None:
            return res, None
        from breast_seg.geometry import compute_nipple_centroid

        return res, (out, compute_nipple_centroid(out.nipple_mask))

    @staticmethod
    def _masks(res: Any) -> Dict[str, Any]:
        return {"yolo_masks": res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None,
                "yolo_cls": res.boxes.cls.cpu().numpy() if getattr(res, "boxes", None) is not None else None}

    def run_mlo_seg(self, image_path: Path, meta: pd.Series) -> Optional[Dict[str, Any]]:
        if self.seg_model is None:
            return None
        res, found = self._segment(image_path)
        if found is None or found[1] is None or found[0].pectoral_line is None:
            return None
        out, nip = found
        pl = out.pectoral_line
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta)
        p1d = GeometryUtils.inverse_transform_640(pl.point_a[0], pl.point_a[1], meta)
        p2d = GeometryUtils.inverse_transform_640(pl.point_b[0], pl.point_b[1], meta)
        fd = GeometryUtils.foot_on_infinite_line(nd, p1d, p2d)
        return {"nipple_dicom": nd, "pec_a_dicom": p1d, "pec_b_dicom": p2d, "foot_dicom": fd,
                "nip_640": (float(nip[0]), float(nip[1])), "pec_a_640": tuple(map(float, pl.point_a)),
                "pec_b_640": tuple(map(float, pl.point_b)), "foot_640": GeometryUtils.dicom_to_640(fd[0], fd[1], meta),
                **self._masks(res)}

    def run_cc_seg(self, image_path: Path, meta: pd.Series) -> Optional[Dict[str, Any]]:
        if self.seg_model is None:
            return None
        res, found = self._segment(image_path)
        if found is None or found[1] is None:
            return None
        nip = found[1]
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta)
        wall, wall640 = _chest_wall(nd, meta)
        return {"nipple_dicom": nd, "wall_dicom": wall, "nip_640": (float(nip[0]), float(nip[1])),
                "wall_point_640": wall640, **self._masks(res)}


ROUTES = {"pose": "Pose", "seg": "Seg"}
LANDMARK_GROUPS = {
    "MLO": {"Nipple": "nip", "Pec Top": "pectop", "Pec Bottom": "pecbot", "PNL Distance": "pnl"},
    "CC": {"Nipple": "nip", "CW Distance": "cw"},
}


class MetricsCalculator:
    """Applies the flat 10 mm rule per route and writes the classification and landmark summaries."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_10mm_rules(self, rule_mm: float = THRESHOLD_MM) -> None:
        for route in ("gt", *ROUTES):
            err = (self.df[f"{route}_pnl_mm"] - self.df[f"{route}_chest_mm"]).abs()
            self.df[f"{route}_abs_err"] = err
            self.df[f"{route}_clinical_pass"] = err <= rule_mm

    def classification_table(self) -> pd.DataFrame:
        """Decision metrics of each route against the reference label (unanswered pairs count as failures)."""
        df = self.df[self.df["cc_qualitativeLabel"].astype(str).str.lower().isin(["good", "bad"])]
        rows = []
        for route, name in ROUTES.items():
            m = FLAT_RULE.matrix(df[f"{route}_abs_err"], ref_is_good(df["cc_qualitativeLabel"]))
            rows.append({"Model": name, **m})
        return pd.DataFrame(rows)

    def build_classification_metrics(self, output_dir: Path) -> pd.DataFrame:
        table = self.classification_table()
        table.to_csv(Path(output_dir) / "classification_metrics.csv", index=False)
        return table

    def landmark_table(self) -> pd.DataFrame:
        def fmt(v: pd.Series) -> str:
            v = v.dropna()
            return "N/A" if v.empty else f"{v.mean():.2f} ± {v.std():.2f} (median {v.median():.2f})"

        rows = []
        for view, metrics in LANDMARK_GROUPS.items():
            for label, key in metrics.items():
                cols = {name: f"{view.lower()}_{route}_{key}_err" for route, name in (("pose", "Pose"), ("seg", "Seg"))}
                rows.append({"View": view, "Metric": label,
                             **{name: fmt(self.df[c]) if c in self.df else "N/A" for name, c in cols.items()}})
        return pd.DataFrame(rows)

    def summarize_landmark_errors(self, output_dir: Path) -> pd.DataFrame:
        table = self.landmark_table()
        table.to_csv(Path(output_dir) / "landmark_summary.csv", index=False, encoding="utf-8")
        return table
