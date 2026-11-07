"""The pose-vs-segmentation comparison on the paired test split.

For every MLO/CC pair both routes measure PNL (MLO) and the nipple-to-chest-wall distance D_CC (CC) in original-image
millimetres; the pair is classified with the flat 10 mm rule and compared with the reference label and the
reference landmarks. Outputs: ``metrics_clinical.csv`` (one row per pair), ``classification_metrics.csv``,
``landmark_summary.csv`` and, optionally, one DICOM panel per pair.

    run_compare(CompareConfig(compare_root="${compare_root}", seg_weights="${weights_dir}/breast_seg_v4_yolo26m.pt",
                              pose_weights_dir="${weights_dir}"))
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from breast_seg.compare.dataset import MammographyDataset, TestPair
from breast_seg.compare.dicom_utils import DEFAULT_SPACING_MM, DicomUtils
from breast_seg.compare.evaluator import InferenceEngine, MetricsCalculator
from breast_seg.compare.geometry import GeometryUtils
from breast_seg.compare.models import ModelFactory
from breast_seg.compare.visualizer import ResultVisualizer
from breast_seg.config import Config, PathsConfig
from breast_seg.evaluation.rule import THRESHOLD_MM


@dataclass
class CompareConfig:
    """Inputs of a compare run (paths may use ``${placeholders}``)."""

    compare_root: str = ""
    output_dir: str = "${results_dir}/compare"
    pose_weights_dir: Optional[str] = None
    pose_mlo_weights: Optional[str] = None
    pose_cc_weights: Optional[str] = None
    seg_weights: Optional[str] = None
    split: str = "Test"
    image_dir: str = "test"
    limit: int = 0
    dicom_viz: bool = True
    wavelet: bool = True


def _abs_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if a is None or b is None else abs(a - b)


def _dist(a: Any, b: Any, spacing: float) -> Optional[float]:
    return None if a is None or b is None else GeometryUtils.euclidean_mm_dicom(a, b, spacing)


class PipelineOrchestrator:
    """Runs both routes on every pair of ``dataset`` and writes the tables (and panels) to ``output_dir``."""

    def __init__(self, dataset: MammographyDataset, engine: InferenceEngine, output_dir: Path,
                 visualizer: Optional[ResultVisualizer] = None, limit: int = 0, image_dir: str = "test",
                 dicom_root: Optional[Path] = None, dicom_loader: Callable[[Path], Any] = DicomUtils.load_dicom_bgr):
        self.dataset, self.engine, self.visualizer = dataset, engine, visualizer
        self.output_dir, self.limit, self.image_dir = Path(output_dir), limit, image_dir
        self.dicom_root, self.dicom_loader = dicom_root, dicom_loader

    @classmethod
    def from_config(cls, cfg: CompareConfig, paths: Optional[PathsConfig] = None) -> PipelineOrchestrator:
        paths = paths or PathsConfig.load()
        root = paths.resolve(cfg.compare_root)

        def opt(v: Optional[str]) -> Optional[Path]:
            return paths.resolve(v) if v else None

        factory = ModelFactory(opt(cfg.pose_weights_dir), wavelet=cfg.wavelet)
        seg_path = opt(cfg.seg_weights)
        seg = factory.get_segmentation_model(seg_path) if seg_path is not None and seg_path.is_file() else None
        engine = InferenceEngine(factory.get_pose_model("MLO", opt(cfg.pose_mlo_weights)),
                                 factory.get_pose_model("CC", opt(cfg.pose_cc_weights)), seg, Config())
        return cls(MammographyDataset.from_root(root, cfg.split), engine, paths.resolve(cfg.output_dir),
                   ResultVisualizer() if cfg.dicom_viz else None, cfg.limit, cfg.image_dir, root / "test_dicom")

    def _image(self, view_dir: Path, sop: str) -> Path:
        return view_dir / "images" / self.image_dir / f"{sop}.png"

    def _load(self, dcm: Optional[Path], meta: pd.Series):
        """Original image for the panels and its pixel spacing; without a DICOM the metadata spacing is used and no
        panel is drawn (the landmarks are in original coordinates, not on the canvas)."""
        bgr, spacing, _ = self.dicom_loader(dcm) if dcm is not None else (None, DEFAULT_SPACING_MM, None)
        if bgr is None:
            return None, float(meta.get("pixel_spacing", DEFAULT_SPACING_MM) or DEFAULT_SPACING_MM)
        return bgr, spacing

    def process_pair(self, p: TestPair) -> Optional[Dict[str, Any]]:
        """Measure one pair; ``None`` when an image or its metadata is missing."""
        ds = self.dataset
        mlo_png, cc_png = self._image(ds.mlo_dir, p.mlo_sop), self._image(ds.cc_dir, p.cc_sop)
        m_meta, c_meta = ds.get_mlo_metadata(p.mlo_sop), ds.get_cc_metadata(p.cc_sop)
        if not mlo_png.is_file() or not cc_png.is_file() or m_meta is None or c_meta is None:
            return None
        g_mlo, g_cc = ds.get_mlo_gt_keypoints(p.mlo_sop), ds.get_cc_gt_nipple(p.cc_sop)
        base = self.dicom_root / p.study_uid if self.dicom_root is not None else None
        m_bgr, m_sp = self._load(base / f"{p.mlo_sop}.dicom" if base else None, m_meta)
        c_bgr, c_sp = self._load(base / f"{p.cc_sop}.dicom" if base else None, c_meta)
        width = float(c_meta["original_width"])

        gt_pnl = (GeometryUtils.pnl_infinite_line_mm(g_mlo["nipple"], g_mlo["pectoral_top"], g_mlo["pectoral_bottom"], m_sp)
                  if g_mlo else None)
        gt_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(g_cc, p.laterality, width, c_sp) if g_cc else None
        pm, pc = self.engine.run_mlo_pose(mlo_png, m_meta), self.engine.run_cc_pose(cc_png, c_meta)
        sm, sc = self.engine.run_mlo_seg(mlo_png, m_meta), self.engine.run_cc_seg(cc_png, c_meta)
        pose_pnl = GeometryUtils.pnl_infinite_line_mm(*pm["kpts_dicom"], m_sp) if pm else None
        pose_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(pc["nipple_dicom"], p.laterality, width, c_sp) if pc else None
        seg_pnl = (GeometryUtils.pnl_infinite_line_mm(sm["nipple_dicom"], sm["pec_a_dicom"], sm["pec_b_dicom"], m_sp)
                   if sm else None)
        seg_chest = GeometryUtils.cc_chest_mm_from_nipple_dicom(sc["nipple_dicom"], p.laterality, width, c_sp) if sc else None
        gm = g_mlo or {}
        row: Dict[str, Any] = {
            "pair_mlo_sop": p.mlo_sop, "pair_cc_sop": p.cc_sop, "study_uid": p.study_uid, "laterality": p.laterality,
            "cc_qualitativeLabel": ds.get_qualitative_label(p.cc_sop, "CC"),
            "mlo_qualitativeLabel": ds.get_qualitative_label(p.mlo_sop, "MLO"), "pixel_spacing_mm": m_sp,
            "gt_pnl_mm": gt_pnl, "gt_chest_mm": gt_chest, "pose_pnl_mm": pose_pnl, "pose_chest_mm": pose_chest,
            "seg_pnl_mm": seg_pnl, "seg_chest_mm": seg_chest,
            "gt_abs_err": _abs_diff(gt_pnl, gt_chest), "pose_abs_err": _abs_diff(pose_pnl, pose_chest),
            "seg_abs_err": _abs_diff(seg_pnl, seg_chest),
            "mlo_pose_pnl_err": _abs_diff(pose_pnl, gt_pnl), "mlo_seg_pnl_err": _abs_diff(seg_pnl, gt_pnl),
            "cc_pose_cw_err": _abs_diff(pose_chest, gt_chest), "cc_seg_cw_err": _abs_diff(seg_chest, gt_chest),
            "mlo_pose_nip_err": _dist(gm.get("nipple"), pm["kpts_dicom"][0] if pm else None, m_sp),
            "mlo_pose_pectop_err": _dist(gm.get("pectoral_top"), pm["kpts_dicom"][1] if pm else None, m_sp),
            "mlo_pose_pecbot_err": _dist(gm.get("pectoral_bottom"), pm["kpts_dicom"][2] if pm else None, m_sp),
            "cc_pose_nip_err": _dist(g_cc, pc["nipple_dicom"] if pc else None, c_sp),
            "mlo_seg_nip_err": _dist(gm.get("nipple"), sm["nipple_dicom"] if sm else None, m_sp),
            "mlo_seg_pectop_err": _dist(gm.get("pectoral_top"), sm["pec_a_dicom"] if sm else None, m_sp),
            "mlo_seg_pecbot_err": _dist(gm.get("pectoral_bottom"), sm["pec_b_dicom"] if sm else None, m_sp),
            "cc_seg_nip_err": _dist(g_cc, sc["nipple_dicom"] if sc else None, c_sp),
        }
        if self.visualizer is not None and m_bgr is not None and c_bgr is not None:
            self._draw(p, row, m_bgr, c_bgr, m_meta, c_meta, g_mlo, g_cc, pm, pc, sm, sc)
        return row

    def _draw(self, p, row, m_bgr, c_bgr, m_meta, c_meta, g_mlo, g_cc, pm, pc, sm, sc) -> None:
        quality = str(row.get("cc_qualitativeLabel") or "N/A").capitalize()

        def caption(mode: str, err: Optional[float], nip_err: Optional[float] = None) -> str:
            es = "N/A" if err is None else f"{err:.1f} mm"
            pnl = row[f"{mode}_pnl_mm"]
            depth = row[f"{mode}_chest_mm"]
            vals = f"PNL {'?' if pnl is None else f'{pnl:.1f}'}  D_CC {'?' if depth is None else f'{depth:.1f}'}"
            if mode == "gt":
                return f"GT: {quality} | {vals} | delta {es}"
            verdict = "N/A" if err is None else ("Good" if err <= THRESHOLD_MM else "Bad")
            ne = "N/A" if nip_err is None else f"{nip_err:.1f} mm"
            return f"{'Pose' if mode == 'pose' else 'Seg'}: {verdict} | {vals} | nipple err {ne} | delta {es}"

        width = float(c_meta["original_width"])
        wall_x = 0.0 if p.laterality.upper() == "L" else width - 1
        mlo = {
            "gt": {"label": caption("gt", row["gt_abs_err"]), "nipple": g_mlo["nipple"], "p1": g_mlo["pectoral_top"],
                   "p2": g_mlo["pectoral_bottom"],
                   "foot": GeometryUtils.foot_on_infinite_line(g_mlo["nipple"], g_mlo["pectoral_top"], g_mlo["pectoral_bottom"])}
            if g_mlo else {},
            "pose": {"label": caption("pose", row["pose_abs_err"], row["mlo_pose_nip_err"]), "nipple": pm["kpts_dicom"][0],
                     "p1": pm["kpts_dicom"][1], "p2": pm["kpts_dicom"][2], "foot": pm["foot_dicom"]} if pm else {},
            "seg": {"label": caption("seg", row["seg_abs_err"], row["mlo_seg_nip_err"]), "nipple": sm["nipple_dicom"],
                    "p1": sm["pec_a_dicom"], "p2": sm["pec_b_dicom"], "foot": sm["foot_dicom"],
                    "masks": sm.get("yolo_masks"), "cls": sm.get("yolo_cls")} if sm else {},
        }
        cc = {
            "gt": {"label": caption("gt", row["gt_abs_err"]), "nipple": g_cc, "wall": (wall_x, g_cc[1])} if g_cc else {},
            "pose": {"label": caption("pose", row["pose_abs_err"], row["cc_pose_nip_err"]), "nipple": pc["nipple_dicom"],
                     "wall": pc["wall_dicom"]} if pc else {},
            "seg": {"label": caption("seg", row["seg_abs_err"], row["cc_seg_nip_err"]), "nipple": sc["nipple_dicom"],
                    "wall": sc["wall_dicom"], "masks": sc.get("yolo_masks"), "cls": sc.get("yolo_cls")} if sc else {},
        }
        out = self.output_dir / "viz_dicom" / f"{p.study_uid}_{p.laterality}.png"
        self.visualizer.draw_dicom_grid(out, m_bgr, c_bgr, m_meta, c_meta, mlo, cc, row)

    def run_compare(self) -> pd.DataFrame:
        """Process every pair and write the output tables; returns the per-pair table."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pairs = self.dataset.get_test_pairs(limit=self.limit)
        rows = [r for r in (self.process_pair(p) for p in pairs) if r is not None]
        df = pd.DataFrame(rows)
        if df.empty:
            print("[compare] no pair could be processed")
            return df
        calc = MetricsCalculator(df)
        calc.compute_10mm_rules()
        calc.build_classification_metrics(self.output_dir)
        calc.summarize_landmark_errors(self.output_dir)
        df.to_csv(self.output_dir / "metrics_clinical.csv", index=False)
        print(f"[compare] {len(df)} of {len(pairs)} pair(s) processed -> {self.output_dir}")
        return df


def run_compare(config: CompareConfig, paths: Optional[PathsConfig] = None) -> pd.DataFrame:
    """Build the pipeline from ``config`` and run it."""
    return PipelineOrchestrator.from_config(config, paths).run_compare()


__all__ = ["CompareConfig", "PipelineOrchestrator", "run_compare"]
