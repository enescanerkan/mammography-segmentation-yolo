"""
SOLID OOP: Visualizer for DICOM and PNG outputs
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


class ResultVisualizer:
    """Consolidates complex plotting logic for MLO and CC views."""

    @staticmethod
    def _apply_seg_mask(bgr: np.ndarray, masks: Optional[np.ndarray], cls_arr: Optional[np.ndarray], meta: pd.Series) -> np.ndarray:
        if masks is None or cls_arr is None or len(masks) == 0:
            return bgr
            
        out = bgr.copy()
        dh, dw = out.shape[:2]
        layer = np.zeros_like(out, dtype=np.float32)
        weight = np.zeros((dh, dw), dtype=np.float32)
        
        scale = float(meta["scale"])
        pt, pb = int(float(meta["pad_top"])), int(float(meta["pad_bottom"]))
        pl, pr = int(float(meta["pad_left"])), int(float(meta["pad_right"]))
        cx1, cx2 = int(float(meta["crop_x1"])), int(float(meta["crop_x2"]))
        cy1, cy2 = int(float(meta["crop_y1"])), int(float(meta["crop_y2"]))
        
        colors = {0: (90, 90, 200), 1: (50, 200, 80), 2: (50, 60, 255)}
        
        for i in range(len(masks)):
            c = int(cls_arr[i])
            if c not in colors: continue
            
            m640 = masks[i]
            mh, mw = m640.shape
            m_unpad = m640[pt:mh-pb, pl:mw-pr] if pt < mh-pb and pl < mw-pr else m640
            
            orig_crop_w, orig_crop_h = max(1, cx2 - cx1), max(1, cy2 - cy1)
            m_resized = cv2.resize(m_unpad, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_NEAREST)
            bool_m = m_resized > 0.5
            
            mask_d = np.zeros((dh, dw), dtype=bool)
            dh_crop = min(bool_m.shape[0], dh - cy1)
            dw_crop = min(bool_m.shape[1], dw - cx1)
            mask_d[cy1:cy1+dh_crop, cx1:cx1+dw_crop] = bool_m[:dh_crop, :dw_crop]
            
            layer[mask_d] = np.array(colors[c], dtype=np.float32)
            weight[mask_d] = 1.0

        a = 0.38
        w3 = weight[..., None]
        blend = out.astype(np.float32) * (1.0 - a * w3) + layer * (a * w3)
        return np.clip(blend, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_points_mlo(out: np.ndarray, pts: dict, color: tuple, w: int, r: int):
        nip, p1, p2, fd = pts.get("nipple"), pts.get("p1"), pts.get("p2"), pts.get("foot")
        if all(x is not None for x in [nip, p1, p2, fd]):
            cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, w)
            cv2.circle(out, (int(nip[0]), int(nip[1])), r, color, -1)
            cv2.line(out, (int(nip[0]), int(nip[1])), (int(fd[0]), int(fd[1])), color, w)

    @staticmethod
    def _draw_points_cc(out: np.ndarray, pts: dict, color: tuple, w: int, r: int):
        nip, wall = pts.get("nipple"), pts.get("wall")
        if nip is not None and wall is not None:
            cv2.circle(out, (int(nip[0]), int(nip[1])), r, color, -1)
            cv2.line(out, (int(nip[0]), int(nip[1])), (int(wall[0]), int(wall[1])), color, w)

    def draw_dicom_grid(self, out_path: Path, m_bgr: np.ndarray, c_bgr: np.ndarray, 
                        m_meta: pd.Series, c_meta: pd.Series,
                        mlo_res: dict, cc_res: dict, row: pd.Series) -> bool:
        
        mh, mw = m_bgr.shape[:2]
        if c_bgr.shape[0] != mh:
            c_bgr = cv2.resize(c_bgr, (int(c_bgr.shape[1] * mh / c_bgr.shape[0]), mh))
        if c_bgr.shape[1] != mw:
            c_bgr = cv2.resize(c_bgr, (mw, mh))

        panels_mlo, panels_cc = [], []
        colors = {"gt": (0, 255, 0), "pose": (0, 165, 255), "seg": (255, 0, 255)}
        w, r = max(1, mh // 500), max(2, mh // 500)
        
        def cap(l1, l2):
            h = max(80, int(mh * 0.065))
            b = np.zeros((h, mw, 3), dtype=np.uint8)
            b[:] = (35, 35, 45)
            fs1, fs2, th = max(0.8, mw * 0.0015), max(0.65, mw * 0.0011), max(2, int(mw * 0.003))
            cv2.putText(b, l1[:80], (16, int(h * 0.45)), cv2.FONT_HERSHEY_SIMPLEX, fs1, (220, 220, 200), th, cv2.LINE_AA)
            if l2:
                cv2.putText(b, l2[:120], (16, int(h * 0.88)), cv2.FONT_HERSHEY_SIMPLEX, fs2, (170, 200, 255), th, cv2.LINE_AA)
            return b

        for mode, color in colors.items():
            pm = mlo_res.get(mode, {})
            pc = cc_res.get(mode, {})
            
            # MLO
            m_img = m_bgr.copy()
            if mode == "seg" and "masks" in pm:
                m_img = self._apply_seg_mask(m_img, pm["masks"], pm["cls"], m_meta)
            self._draw_points_mlo(m_img, pm, color, w, r)
            panels_mlo.append(np.vstack([cap(f"MLO {mode.upper()}", pm.get("label", "")), m_img]))
            
            # CC
            c_img = c_bgr.copy()
            if mode == "seg" and "masks" in pc:
                c_img = self._apply_seg_mask(c_img, pc["masks"], pc["cls"], c_meta)
            self._draw_points_cc(c_img, pc, color, w, r)
            panels_cc.append(np.vstack([cap(f"CC {mode.upper()}", pc.get("label", "")), c_img]))

        mlo_row = np.hstack(panels_mlo)
        cc_row = np.hstack(panels_cc)
        
        # Title
        th = max(70, int(mh * 0.05))
        top = np.zeros((th, mlo_row.shape[1], 3), dtype=np.uint8)
        top[:] = (20, 30, 40)
        fs, tfw = max(0.9, mw * 0.0018), max(2, int(mw * 0.003))
        
        def ef(e): return f"{e:.1f}mm" if e is not None else "N/A"
        
        title_lines = f"Kural PNL 10mm  |  GT: {ef(row.get('gt_abs_err'))}  |  Pose: {ef(row.get('pose_abs_err'))}  |  Seg: {ef(row.get('seg_abs_err'))}"
        cv2.putText(top, title_lines, (16, int(th * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, fs, (200, 220, 255), tfw, cv2.LINE_AA)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), np.vstack([top, mlo_row, cc_row]))
        return True
