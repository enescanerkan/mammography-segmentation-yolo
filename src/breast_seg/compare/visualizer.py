"""Side-by-side DICOM panels of the compare run: reference, pose route and segmentation route, MLO over CC."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

COLORS = {"gt": (0, 255, 0), "pose": (0, 165, 255), "seg": (255, 0, 255)}
MASK_COLORS = {0: (90, 90, 200), 1: (50, 200, 80), 2: (50, 60, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pt(p: Any) -> tuple:
    return int(p[0]), int(p[1])


class ResultVisualizer:
    """Draws one PNG per breast pair."""

    @staticmethod
    def _apply_seg_mask(bgr: np.ndarray, masks: Optional[np.ndarray], classes: Optional[np.ndarray], meta: Any) -> np.ndarray:
        """Paste 640-canvas instance masks back onto the original image (undo padding, resize to the crop)."""
        if masks is None or classes is None or len(masks) == 0:
            return bgr
        out = bgr.copy()
        dh, dw = out.shape[:2]
        layer = np.zeros_like(out, dtype=np.float32)
        weight = np.zeros((dh, dw), dtype=np.float32)
        pt, pb = int(float(meta["pad_top"])), int(float(meta["pad_bottom"]))
        pl, pr = int(float(meta["pad_left"])), int(float(meta["pad_right"]))
        cx1, cx2 = int(float(meta["crop_x1"])), int(float(meta["crop_x2"]))
        cy1, cy2 = int(float(meta["crop_y1"])), int(float(meta["crop_y2"]))
        for m640, c in zip(masks, classes):
            c = int(c)
            if c not in MASK_COLORS:
                continue
            mh, mw = m640.shape
            unpadded = m640[pt:mh - pb, pl:mw - pr] if pt < mh - pb and pl < mw - pr else m640
            resized = cv2.resize(unpadded, (max(1, cx2 - cx1), max(1, cy2 - cy1)), interpolation=cv2.INTER_NEAREST) > 0.5
            oy, ox = max(0, min(cy1, dh)), max(0, min(cx1, dw))
            h, w = max(0, min(resized.shape[0], dh - oy)), max(0, min(resized.shape[1], dw - ox))
            if h == 0 or w == 0:
                continue
            full = np.zeros((dh, dw), dtype=bool)
            full[oy:oy + h, ox:ox + w] = resized[:h, :w]
            layer[full] = np.array(MASK_COLORS[c], dtype=np.float32)
            weight[full] = 1.0
        alpha = 0.38 * weight[..., None]
        return np.clip(out.astype(np.float32) * (1.0 - alpha) + layer * alpha, 0, 255).astype(np.uint8)

    @staticmethod
    def _draw_mlo(img: np.ndarray, pts: Dict[str, Any], color: tuple, w: int, r: int) -> None:
        nip, p1, p2, foot = pts.get("nipple"), pts.get("p1"), pts.get("p2"), pts.get("foot")
        if all(x is not None for x in (nip, p1, p2, foot)):
            cv2.line(img, _pt(p1), _pt(p2), color, w)
            cv2.circle(img, _pt(nip), r, color, -1)
            cv2.line(img, _pt(nip), _pt(foot), color, w)

    @staticmethod
    def _draw_cc(img: np.ndarray, pts: Dict[str, Any], color: tuple, w: int, r: int) -> None:
        nip, wall = pts.get("nipple"), pts.get("wall")
        if nip is not None and wall is not None:
            cv2.circle(img, _pt(nip), r, color, -1)
            cv2.line(img, _pt(nip), _pt(wall), color, w)

    @staticmethod
    def _caption(width: int, height: int, title: str, label: str) -> np.ndarray:
        h = max(80, int(height * 0.065))
        bar = np.full((h, width, 3), (35, 35, 45), dtype=np.uint8)
        th = max(2, int(width * 0.003))
        cv2.putText(bar, title[:80], (16, int(h * 0.45)), FONT, max(0.8, width * 0.0015), (220, 220, 200), th, cv2.LINE_AA)
        if label:
            cv2.putText(bar, label[:120], (16, int(h * 0.88)), FONT, max(0.65, width * 0.0011), (170, 200, 255), th, cv2.LINE_AA)
        return bar

    def draw_dicom_grid(self, out_path: Path, m_bgr: np.ndarray, c_bgr: np.ndarray, m_meta: Any, c_meta: Any,
                        mlo_res: Dict[str, Dict[str, Any]], cc_res: Dict[str, Dict[str, Any]], row: Any) -> bool:
        mh, mw = m_bgr.shape[:2]
        if c_bgr.shape[:2] != (mh, mw):
            c_bgr = cv2.resize(c_bgr, (mw, mh))
        w, r = max(1, mh // 500), max(2, mh // 500)
        mlo_panels, cc_panels = [], []
        for mode, color in COLORS.items():
            pm, pc = mlo_res.get(mode, {}), cc_res.get(mode, {})
            m_img, c_img = m_bgr.copy(), c_bgr.copy()
            if mode == "seg" and "masks" in pm:
                m_img = self._apply_seg_mask(m_img, pm["masks"], pm["cls"], m_meta)
            if mode == "seg" and "masks" in pc:
                c_img = self._apply_seg_mask(c_img, pc["masks"], pc["cls"], c_meta)
            self._draw_mlo(m_img, pm, color, w, r)
            self._draw_cc(c_img, pc, color, w, r)
            mlo_panels.append(np.vstack([self._caption(mw, mh, f"MLO {mode.upper()}", pm.get("label", "")), m_img]))
            cc_panels.append(np.vstack([self._caption(mw, mh, f"CC {mode.upper()}", pc.get("label", "")), c_img]))
        mlo_row, cc_row = np.hstack(mlo_panels), np.hstack(cc_panels)
        th = max(70, int(mh * 0.05))
        top = np.full((th, mlo_row.shape[1], 3), (20, 30, 40), dtype=np.uint8)

        def fmt(e: Any) -> str:
            return "N/A" if e is None or (isinstance(e, float) and np.isnan(e)) else f"{e:.1f} mm"

        title = (f"PNL rule 10 mm  |  GT: {fmt(row.get('gt_abs_err'))}  |  Pose: {fmt(row.get('pose_abs_err'))}  |  "
                 f"Seg: {fmt(row.get('seg_abs_err'))}")
        cv2.putText(top, title, (16, int(th * 0.65)), FONT, max(0.9, mw * 0.0018), (200, 220, 255), max(2, int(mw * 0.003)),
                    cv2.LINE_AA)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(out_path), np.vstack([top, mlo_row, cc_row])))
