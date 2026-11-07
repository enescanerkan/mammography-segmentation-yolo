"""
SOLID OOP: Inference Execution and Metrics Evaluation
"""
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from ultralytics.engine.results import Results

from breast_seg.config import Config
from breast_seg.analyzer import MLOAnalyzer
from breast_seg.geometry import compute_nipple_centroid
from pipeline.geometry import GeometryUtils


class InferenceEngine:
    """Executes models and processes outputs using Geometry mapping."""
    def __init__(self, pose_mlo, pose_cc, seg_model, seg_config: Config):
        self.pose_mlo = pose_mlo
        self.pose_cc = pose_cc
        self.seg_model = seg_model
        self.seg_config = seg_config
        self.mlo_analyzer = MLOAnalyzer(seg_config)

    def run_mlo_pose(self, image_path: Path, meta_row: pd.Series) -> Optional[Dict[str, Any]]:
        k = self.pose_mlo.predict(image_path)
        if k is None:
            return None
        nip, p1, p2 = k[0], k[1], k[2]
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta_row)
        p1d = GeometryUtils.inverse_transform_640(p1[0], p1[1], meta_row)
        p2d = GeometryUtils.inverse_transform_640(p2[0], p2[1], meta_row)
        fd = GeometryUtils.foot_on_infinite_line(nd, p1d, p2d)
        f640 = GeometryUtils.dicom_to_640(fd[0], fd[1], meta_row)
        return {
            "kpts_640": [(float(x[0]), float(x[1])) for x in [nip, p1, p2]],
            "kpts_dicom": [nd, p1d, p2d],
            "foot_dicom": fd,
            "foot_640": f640
        }

    def run_cc_pose(self, image_path: Path, meta_row: pd.Series) -> Optional[Dict[str, Any]]:
        k = self.pose_cc.predict(image_path)
        if k is None:
            return None
        nip = k[0]
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta_row)
        lat = str(meta_row.get("laterality", "L")).upper()
        w = float(meta_row["original_width"])
        wall_x = 0.0 if lat == "L" else w - 1.0
        wall_d = (wall_x, nd[1])
        w640 = GeometryUtils.dicom_to_640(wall_d[0], wall_d[1], meta_row)
        return {
            "nipple_640": (float(nip[0]), float(nip[1])),
            "nipple_dicom": nd,
            "wall_dicom": wall_d,
            "wall_point_640": w640
        }

    def run_mlo_seg(self, image_path: Path, meta_row: pd.Series) -> Optional[Dict[str, Any]]:
        res: Results = self.seg_model.predict(image_path, self.seg_config)
        if res is None:
            return None
        out = self.mlo_analyzer.analyze_single(res)
        if not out.has_nipple or out.nipple_mask is None or out.pectoral_line is None:
            return None
            
        nip = compute_nipple_centroid(out.nipple_mask)
        if nip is None:
            return None
        pl = out.pectoral_line
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta_row)
        p1d = GeometryUtils.inverse_transform_640(pl.point_a[0], pl.point_a[1], meta_row)
        p2d = GeometryUtils.inverse_transform_640(pl.point_b[0], pl.point_b[1], meta_row)
        fd = GeometryUtils.foot_on_infinite_line(nd, p1d, p2d)
        f640 = GeometryUtils.dicom_to_640(fd[0], fd[1], meta_row)
        
        return {
            "nipple_dicom": nd,
            "pec_a_dicom": p1d,
            "pec_b_dicom": p2d,
            "foot_dicom": fd,
            "nip_640": (float(nip[0]), float(nip[1])),
            "pec_a_640": (float(pl.point_a[0]), float(pl.point_a[1])),
            "pec_b_640": (float(pl.point_b[0]), float(pl.point_b[1])),
            "foot_640": f640,
            "yolo_masks": res.masks.data.cpu().numpy() if res.masks and res.masks.data is not None else None,
            "yolo_cls": res.boxes.cls.cpu().numpy() if res.boxes else None,
        }

    def run_cc_seg(self, image_path: Path, meta_row: pd.Series) -> Optional[Dict[str, Any]]:
        res: Results = self.seg_model.predict(image_path, self.seg_config)
        if res is None:
            return None
        out = self.mlo_analyzer.analyze_single(res)
        if not out.has_nipple or out.nipple_mask is None:
            return None
        nip = compute_nipple_centroid(out.nipple_mask)
        if nip is None:
            return None
        nd = GeometryUtils.inverse_transform_640(nip[0], nip[1], meta_row)
        lat = str(meta_row.get("laterality", "L")).upper()
        w = float(meta_row["original_width"])
        wall_x = 0.0 if lat == "L" else w - 1.0
        wall_d = (wall_x, nd[1])
        w640 = GeometryUtils.dicom_to_640(wall_d[0], wall_d[1], meta_row)
        
        return {
            "nipple_dicom": nd,
            "wall_dicom": wall_d,
            "nip_640": (float(nip[0]), float(nip[1])),
            "wall_point_640": w640,
            "yolo_masks": res.masks.data.cpu().numpy() if res.masks and res.masks.data is not None else None,
            "yolo_cls": res.boxes.cls.cpu().numpy() if res.boxes else None,
        }


class MetricsCalculator:
    """Calculates Errors, Accuracy, Precision, Recall, and Confusion Matrices."""
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_10mm_rules(self, rule_mm: float = 10.0):
        # GT Rule
        self.df["gt_abs_err"] = (self.df["gt_pnl_mm"] - self.df["gt_chest_mm"]).abs()
        self.df["gt_clinical_pass"] = self.df["gt_abs_err"] <= rule_mm
        
        # Pose Rule 
        self.df["pose_abs_err"] = (self.df["pose_pnl_mm"] - self.df["pose_chest_mm"]).abs()
        self.df["pose_clinical_pass"] = self.df["pose_abs_err"] <= rule_mm
        
        # Seg Rule
        self.df["seg_abs_err"] = (self.df["seg_pnl_mm"] - self.df["seg_chest_mm"]).abs()
        self.df["seg_clinical_pass"] = self.df["seg_abs_err"] <= rule_mm

    def build_classification_metrics(self, output_dir: Path):
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
            try:
                import seaborn as sns
                has_sns = True
            except ImportError:
                has_sns = False
        except ImportError:
            print("[WARN] matplotlib/sklearn not installed. Skipping classification metrics.")
            return

        df_clf = self.df[self.df["cc_qualitativeLabel"].astype(str).str.lower().isin(["good", "bad"])].copy()
        if df_clf.empty: return

        y_true = df_clf["cc_qualitativeLabel"].astype(str).str.lower()
        metrics_rows = []
        labels_order = ["bad", "good"]

        for model_name, col_name in [("Pose", "pose_clinical_pass"), ("Seg", "seg_clinical_pass")]:
            if col_name in df_clf.columns and df_clf[col_name].notna().any():
                v = df_clf[col_name].dropna()
                idx = v.index
                yt = y_true.loc[idx]
                yp = v.map({True: "good", False: "bad", 1: "good", 0: "bad"})

                acc = accuracy_score(yt, yp)
                cm = confusion_matrix(yt, yp, labels=labels_order)
                prec, rec, f1, _ = precision_recall_fscore_support(yt, yp, labels=labels_order, zero_division=0)

                metrics_rows.append({
                    "Model": model_name,
                    "Accuracy": f"{acc:.4f}",
                    "Precision_Good": f"{prec[1]:.4f}",
                    "Recall_Good": f"{rec[1]:.4f}",
                    "F1_Good": f"{f1[1]:.4f}",
                    "Precision_Bad": f"{prec[0]:.4f}",
                    "Recall_Bad": f"{rec[0]:.4f}",
                    "F1_Bad": f"{f1[0]:.4f}",
                    "Total_Pairs": len(yt)
                })

                if has_sns:
                    plt.figure(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
                    plt.title(f"{model_name} Confusion Matrix (vs Qualitative Label)")
                    plt.xlabel("Predicted (Pass=Good, Fail=Bad)")
                    plt.ylabel("Ground Truth")
                    plt.savefig(output_dir / f"confusion_matrix_{model_name.lower()}.png", dpi=150, bbox_inches="tight")
                    plt.close()

        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(output_dir / "classification_metrics.csv", index=False)

    def summarize_landmark_errors(self, output_dir: Path):
        # Report MLO and CC landmark errors separately in a concise table.
        groups = {
            "MLO": {
                "Nipple": ["mlo_pose_nip_err", "mlo_seg_nip_err"],
                "Pec Top": ["mlo_pose_pectop_err", "mlo_seg_pectop_err"],
                "Pec Bottom": ["mlo_pose_pecbot_err", "mlo_seg_pecbot_err"],
                "PNL Distance": ["mlo_pose_pnl_err", "mlo_seg_pnl_err"]
            },
            "CC": {
                "Nipple": ["cc_pose_nip_err", "cc_seg_nip_err"],
                "CW Distance": ["cc_pose_cw_err", "cc_seg_cw_err"]
            }
        }
        
        models = ["Rule-Based (Pose)", "Seg-Model"]
        report_data = []

        def fmt_v(v):
            if v.empty: return "N/A"
            return f"{v.mean():.2f} ± {v.std():.2f} (med: {v.median():.2f})"

        for view, metrics in groups.items():
            for label, cols in metrics.items():
                row = {"View": view, "Metric": label}
                row[models[0]] = fmt_v(self.df[cols[0]].dropna()) if len(cols) > 0 else "N/A"
                row[models[1]] = fmt_v(self.df[cols[1]].dropna()) if len(cols) > 1 else "N/A"
                report_data.append(row)

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(output_dir / "scientific_summary.csv", index=False, encoding="utf-8")
        
        # Markdown Tablo olarak da kaydet (Manual formatting to avoid tabulate dependency)
        md_content = "# Scientific Metrics Summary\n\n"
        md_content += "| View | Metric | Rule-Based (Pose) | Seg-Model |\n"
        md_content += "| :--- | :--- | :--- | :--: |\n"
        for _, r in df_report.iterrows():
            md_content += f"| {r['View']} | {r['Metric']} | {r['Rule-Based (Pose)']} | {r['Seg-Model']} |\n"

        with open(output_dir / "scientific_summary.md", "w", encoding="utf-8") as f:
            f.write(md_content)
