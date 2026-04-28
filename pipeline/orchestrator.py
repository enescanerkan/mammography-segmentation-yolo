"""
SOLID OOP: Pipeline Orchestrator linking models, datasets, inference, evaluator, and visualizer.
"""

from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd

from pipeline.models import ModelFactory
from pipeline.dataset import MammographyDataset
from pipeline.evaluator import InferenceEngine, MetricsCalculator
from pipeline.visualizer import ResultVisualizer
from pipeline.geometry import GeometryUtils
from pipeline.dicom_utils import DicomUtils
from breast_seg.config import Config


class PipelineOrchestrator:
    def __init__(self, root_dir: Path, output_dir: Path, limit: int = 0, no_viz: bool = False, no_dicom_viz: bool = False):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.limit = limit
        self.no_viz = no_viz
        self.no_dicom_viz = no_dicom_viz
        
        self.viz_dir = self.output_dir / "viz"
        self.dicom_viz_dir = self.output_dir / "viz_dicom"
        
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.dicom_viz_dir.mkdir(parents=True, exist_ok=True)

        print("[Orchestrator] Initializing DatasetManager...")
        self.dataset = MammographyDataset(
            labels_dir=self.root_dir / "compare-dataset" / "labels",
            cc_dir=self.root_dir / "compare-dataset" / "CC",
            mlo_dir=self.root_dir / "compare-dataset" / "MLO"
        )
        
        print("[Orchestrator] Initializing Models & Inference Engine...")
        model_fac = ModelFactory(self.root_dir / "pose_weights")
        pose_mlo = model_fac.get_pose_model("MLO")
        pose_cc = model_fac.get_pose_model("CC")
        
        # Look for seg weights
        seg_weights_dir = self.root_dir / "runs" / "breast_seg_yolo26m" / "weights"
        seg_path = seg_weights_dir / "best.pt"
        if not seg_path.is_file() and (seg_weights_dir / "last.pt").is_file():
            seg_path = seg_weights_dir / "last.pt"
            
        seg_cfg = Config()
        seg_model = model_fac.get_segmentation_model(seg_path) if seg_path.is_file() else None

        self.engine = InferenceEngine(pose_mlo, pose_cc, seg_model, seg_cfg)
        self.visualizer = ResultVisualizer()

    def run_compare(self):
        pairs = self.dataset.get_test_pairs(limit=self.limit)
        print(f"[Orchestrator] {len(pairs)} MLO+CC test çifti bulundu.")
        
        rows = []
        c_p = 0
        
        for p in tqdm(pairs, desc="Processing Test Pairs", dynamic_ncols=True):
            mlo_png = self.dataset.mlo_dir / "images" / "test" / f"{p.mlo_sop}.png"
            cc_png = self.dataset.cc_dir / "images" / "test" / f"{p.cc_sop}.png"
            dicom_base = self.root_dir / "compare-dataset" / "test_dicom" / p.study_uid
            mlo_dcm = dicom_base / f"{p.mlo_sop}.dicom"
            cc_dcm = dicom_base / f"{p.cc_sop}.dicom"
            
            if not mlo_png.is_file() or not cc_png.is_file():
                continue
                
            m_meta = self.dataset.get_mlo_metadata(p.mlo_sop)
            c_meta = self.dataset.get_cc_metadata(p.cc_sop)
            if m_meta is None or c_meta is None: continue
            
            g_mlo = self.dataset.get_mlo_gt_keypoints(p.mlo_sop)
            g_nip_cc = self.dataset.get_cc_gt_nipple(p.cc_sop)
            
            row = {
                "pair_mlo_sop": p.mlo_sop,
                "pair_cc_sop": p.cc_sop,
                "study_uid": p.study_uid,
                "laterality": p.laterality,
                "cc_qualitativeLabel": self.dataset.get_qualitative_label(p.cc_sop, "CC"),
                "mlo_qualitativeLabel": self.dataset.get_qualitative_label(p.mlo_sop, "MLO")
            }
            c_p += 1
            
            m_bgr, m_sp, _ = DicomUtils.load_dicom_bgr(mlo_dcm)
            c_bgr, c_sp, _ = DicomUtils.load_dicom_bgr(cc_dcm)
            
            # Fallback: DICOM yoksa PNG'den oku
            if m_bgr is None:
                m_bgr = cv2.imread(str(mlo_png))
                m_sp = float(m_meta.get("pixel_spacing", 0.085) or 0.085)
            if c_bgr is None:
                c_bgr = cv2.imread(str(cc_png))
                c_sp = float(c_meta.get("pixel_spacing", 0.085) or 0.085)
            
            row["pixel_spacing_mm"] = m_sp
            
            gt_pnl, gt_chest = None, None
            if g_mlo:
                gt_pnl = GeometryUtils.pnl_infinite_line_mm(g_mlo["nipple"], g_mlo["pectoral_top"], g_mlo["pectoral_bottom"], m_sp)
            if g_nip_cc:
                w_orig = float(c_meta["original_width"])
                gt_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(g_nip_cc, p.laterality, w_orig, c_sp)
                
            row["gt_pnl_mm"] = gt_pnl
            row["gt_chest_mm"] = gt_chest
            
            pm = self.engine.run_mlo_pose(mlo_png, m_meta)
            pc = self.engine.run_cc_pose(cc_png, c_meta)
            sm = self.engine.run_mlo_seg(mlo_png, m_meta) if self.engine.seg_model else None
            sc = self.engine.run_cc_seg(cc_png, c_meta) if self.engine.seg_model else None
            
            pose_pnl, pose_chest = None, None
            if pm: pose_pnl = GeometryUtils.pnl_infinite_line_mm(*pm["kpts_dicom"], m_sp)
            if pc: pose_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(pc["nipple_dicom"], p.laterality, float(c_meta["original_width"]), c_sp)
                
            seg_pnl, seg_chest = None, None
            if sm: seg_pnl = GeometryUtils.pnl_infinite_line_mm(sm["nipple_dicom"], sm["pec_a_dicom"], sm["pec_b_dicom"], m_sp)
            if sc: seg_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(sc["nipple_dicom"], p.laterality, float(c_meta["original_width"]), c_sp)
                
            row["pose_pnl_mm"], row["pose_chest_mm"] = pose_pnl, pose_chest
            row["seg_pnl_mm"], row["seg_chest_mm"] = seg_pnl, seg_chest
            
            def e_diff(a, b): return abs(a - b) if a is not None and b is not None else None
            def l_eval(e): return "Unk" if e is None else ("Good" if e <= 10.0 else "Bad")
            
            row["gt_abs_err"] = e_diff(gt_pnl, gt_chest)
            row["pose_abs_err"] = e_diff(pose_pnl, pose_chest)
            row["seg_abs_err"] = e_diff(seg_pnl, seg_chest)
            
            # --- Distance Errors (Abs diff in mm) ---
            def abs_diff(v1, v2):
                if v1 is None or v2 is None: return None
                return abs(v1 - v2)

            row["mlo_pose_pnl_err"] = abs_diff(pose_pnl, gt_pnl)
            row["mlo_seg_pnl_err"] = abs_diff(seg_pnl, gt_pnl)
            row["cc_pose_cw_err"] = abs_diff(pose_chest, gt_chest)
            row["cc_seg_cw_err"] = abs_diff(seg_chest, gt_chest)

            # --- Landmark Euclidean Errors (mm) ---
            def dist_mm(p1, p2, sp):
                if p1 is None or p2 is None: return None
                return GeometryUtils.euclidean_mm_dicom(p1, p2, sp)

            # Pose Errors
            row["mlo_pose_nip_err"] = dist_mm(g_mlo.get("nipple") if g_mlo else None, pm["kpts_dicom"][0] if pm else None, m_sp)
            row["mlo_pose_pectop_err"] = dist_mm(g_mlo.get("pectoral_top") if g_mlo else None, pm["kpts_dicom"][1] if pm else None, m_sp)
            row["mlo_pose_pecbot_err"] = dist_mm(g_mlo.get("pectoral_bottom") if g_mlo else None, pm["kpts_dicom"][2] if pm else None, m_sp)
            row["cc_pose_nip_err"] = dist_mm(g_nip_cc, pc["nipple_dicom"] if pc else None, c_sp)
            
            # Seg Errors
            row["mlo_seg_nip_err"] = dist_mm(g_mlo.get("nipple") if g_mlo else None, sm["nipple_dicom"] if sm else None, m_sp)
            row["mlo_seg_pectop_err"] = dist_mm(g_mlo.get("pectoral_top") if g_mlo else None, sm["pec_a_dicom"] if sm else None, m_sp)
            row["mlo_seg_pecbot_err"] = dist_mm(g_mlo.get("pectoral_bottom") if g_mlo else None, sm["pec_b_dicom"] if sm else None, m_sp)
            row["cc_seg_nip_err"] = dist_mm(g_nip_cc, sc["nipple_dicom"] if sc else None, c_sp)

            rows.append(row)
            
            # --- Visualisation ---
            if not self.no_dicom_viz and m_bgr is not None and c_bgr is not None:
                gt_qual = str(row.get("cc_qualitativeLabel") or "N/A").capitalize()
                
                def fmt_cap(mode, err, pnl_mm=None, depth_mm=None, nip_err=None):
                    es = f"{err:.1f}mm" if err is not None else "N/A"
                    pnl_s = f"{pnl_mm:.1f}" if pnl_mm is not None else "?"
                    dep_s = f"{depth_mm:.1f}" if depth_mm is not None else "?"
                    ne_s = f"{nip_err:.1f}mm" if nip_err is not None else "N/A"
                    verdict = "N/A" if err is None else ("Good" if err <= 10.0 else "Bad")
                    if mode == "gt":
                        return f"GT: {gt_qual} | PNL:{pnl_s} Depth:{dep_s} | Err:{es}"
                    else:
                        tag = "Rule-Based" if mode == "pose" else "Seg-Model"
                        return f"{tag}: {verdict} | PNL:{pnl_s} Depth:{dep_s} | Nipple-Err:{ne_s} | Err:{es}"

                mlo_res = {
                    "gt": {"label": fmt_cap("gt", row["gt_abs_err"], gt_pnl, gt_chest), "nipple": g_mlo["nipple"], "p1": g_mlo["pectoral_top"], "p2": g_mlo["pectoral_bottom"], "foot": GeometryUtils.foot_on_infinite_line(g_mlo["nipple"], g_mlo["pectoral_top"], g_mlo["pectoral_bottom"])} if g_mlo else {},
                    "pose": {"label": fmt_cap("pose", row["pose_abs_err"], pose_pnl, pose_chest, row["mlo_pose_nip_err"]), "nipple": pm["kpts_dicom"][0], "p1": pm["kpts_dicom"][1], "p2": pm["kpts_dicom"][2], "foot": pm["foot_dicom"]} if pm else {},
                    "seg": {"label": fmt_cap("seg", row["seg_abs_err"], seg_pnl, seg_chest, row["mlo_seg_nip_err"]), "nipple": sm["nipple_dicom"], "p1": sm["pec_a_dicom"], "p2": sm["pec_b_dicom"], "foot": sm["foot_dicom"], "masks": sm.get("yolo_masks"), "cls": sm.get("yolo_cls")} if sm else {},
                }
                cc_res = {
                    "gt": {"label": fmt_cap("gt", row["gt_abs_err"], gt_pnl, gt_chest), "nipple": g_nip_cc, "wall": (0.0 if p.laterality.upper()=="L" else float(c_meta["original_width"])-1, g_nip_cc[1]) if g_nip_cc else None} if g_nip_cc else {},
                    "pose": {"label": fmt_cap("pose", row["pose_abs_err"], pose_pnl, pose_chest, row["cc_pose_nip_err"]), "nipple": pc["nipple_dicom"], "wall": pc["wall_dicom"]} if pc else {},
                    "seg": {"label": fmt_cap("seg", row["seg_abs_err"], seg_pnl, seg_chest, row["cc_seg_nip_err"]), "nipple": sc["nipple_dicom"], "wall": sc["wall_dicom"], "masks": sc.get("yolo_masks"), "cls": sc.get("yolo_cls")} if sc else {}
                }
                dicom_out = self.dicom_viz_dir / f"{p.study_uid}_{p.laterality}.png"
                self.visualizer.draw_dicom_grid(dicom_out, m_bgr, c_bgr, m_meta, c_meta, mlo_res, cc_res, pd.Series(row))

        if not rows:
            print("[Orchestrator] Hiç veri işlenemedi!")
            return
            
        df = pd.DataFrame(rows)
        calc = MetricsCalculator(df)
        calc.compute_10mm_rules()
        calc.build_classification_metrics(self.output_dir)
        calc.summarize_landmark_errors(self.output_dir)
        
        df.to_csv(self.output_dir / "metrics_clinical.csv", index=False)
        print(f"[Orchestrator] Bitti. İşlenen vaka: {c_p}. Sonuçlar '{self.output_dir}' dizininde.")
