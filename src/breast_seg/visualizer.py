"""Visualization of analysis results on mammography images."""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from breast_seg.analyzer import AnalysisResult
from breast_seg.config import Config


class ResultVisualizer:
    """Draws geometric analysis overlays on mammography images.

    Supports both MLO view (pectoral line + PNL) and CC view (depth line).
    """

    # Color palette (BGR)
    COLOR_PECTORAL_LINE = (0, 255, 0)     # Green
    COLOR_PNL_LINE = (0, 0, 255)          # Red
    COLOR_CC_DEPTH_LINE = (0, 165, 255)   # Orange
    COLOR_NIPPLE_CENTER = (255, 0, 255)   # Magenta
    COLOR_FOOT_POINT = (255, 255, 0)      # Cyan
    COLOR_EDGE_POINTS = (0, 255, 255)     # Yellow
    COLOR_TEXT_BG = (0, 0, 0)             # Black

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    LINE_THICKNESS = 2

    def __init__(self, config: Config):
        self._config = config

    def draw_analysis(self, result: AnalysisResult) -> np.ndarray:
        """Draw all analysis overlays on the image."""
        image = result.image.copy()

        # Draw segmentation mask overlays
        image = self._draw_mask_overlay(image, result.pectoral_mask, (0, 200, 0), 0.2)
        image = self._draw_mask_overlay(image, result.breast_mask, (200, 150, 0), 0.15)
        image = self._draw_mask_overlay(image, result.nipple_mask, (200, 0, 200), 0.3)

        if result.is_mlo:
            image = self._draw_mlo_analysis(image, result)
        else:
            image = self._draw_cc_analysis(image, result)

        self._draw_status(image, result)
        return image

    # ── MLO Drawing ────────────────────────────────────────

    def _draw_mlo_analysis(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Draw pectoral line + PNL for MLO views."""

        if result.pectoral_line is not None:
            # Boundary edge sample points (yellow dots)
            for pt in result.pectoral_line.boundary_points:
                cv2.circle(image, pt, 3, self.COLOR_EDGE_POINTS, -1, cv2.LINE_AA)

            # Fitted pectoral line (green)
            cv2.line(image, result.pectoral_line.point_a, result.pectoral_line.point_b,
                     self.COLOR_PECTORAL_LINE, self.LINE_THICKNESS, cv2.LINE_AA)
            self._put_label(image, "Pectoral Line",
                            result.pectoral_line.point_a, self.COLOR_PECTORAL_LINE, -10)

        if result.pnl is not None:
            # PNL line: nipple → foot on pectoral line (red, perpendicular)
            cv2.line(image, result.pnl.nipple_center, result.pnl.foot_point,
                     self.COLOR_PNL_LINE, self.LINE_THICKNESS, cv2.LINE_AA)

            # Markers
            cv2.circle(image, result.pnl.nipple_center, 6,
                       self.COLOR_NIPPLE_CENTER, -1, cv2.LINE_AA)
            cv2.circle(image, result.pnl.foot_point, 6,
                       self.COLOR_FOOT_POINT, -1, cv2.LINE_AA)

            # Right angle indicator
            self._draw_right_angle(image, result.pnl, result.pectoral_line)

            # Distance label at midpoint
            mid_x = (result.pnl.nipple_center[0] + result.pnl.foot_point[0]) // 2
            mid_y = (result.pnl.nipple_center[1] + result.pnl.foot_point[1]) // 2
            self._put_label(image, f"PNL: {result.pnl.distance_px:.1f} px",
                            (mid_x, mid_y), self.COLOR_PNL_LINE, -10)

        return image

    # ── CC Drawing ─────────────────────────────────────────

    def _draw_cc_analysis(self, image: np.ndarray, result: AnalysisResult) -> np.ndarray:
        """Draw CC depth line (nipple to breast edge) for CC views."""

        if result.cc_depth is not None:
            # Horizontal depth line: nipple → breast edge (orange)
            cv2.line(image, result.cc_depth.nipple_center, result.cc_depth.edge_point,
                     self.COLOR_CC_DEPTH_LINE, self.LINE_THICKNESS, cv2.LINE_AA)

            # Markers
            cv2.circle(image, result.cc_depth.nipple_center, 6,
                       self.COLOR_NIPPLE_CENTER, -1, cv2.LINE_AA)
            cv2.circle(image, result.cc_depth.edge_point, 6,
                       self.COLOR_FOOT_POINT, -1, cv2.LINE_AA)

            # Distance label at midpoint
            mid_x = (result.cc_depth.nipple_center[0] + result.cc_depth.edge_point[0]) // 2
            mid_y = result.cc_depth.nipple_center[1]
            side = result.cc_depth.breast_side.upper()
            self._put_label(
                image,
                f"CC Depth ({side}): {result.cc_depth.distance_px:.1f} px",
                (mid_x, mid_y), self.COLOR_CC_DEPTH_LINE, -15,
            )

        return image

    # ── Save ───────────────────────────────────────────────

    def save(self, image: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)

    def draw_and_save(self, result: AnalysisResult, output_dir: Path) -> Path:
        image = self.draw_analysis(result)
        filename = Path(result.image_path).stem + "_analysis.png"
        output_path = output_dir / filename
        self.save(image, output_path)
        return output_path

    # ── Private Helpers ────────────────────────────────────

    @staticmethod
    def _draw_mask_overlay(image, mask, color, alpha=0.3):
        if mask is None:
            return image
        overlay = image.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    def _put_label(self, image, text, position, color, offset_y=0):
        x, y = position[0], position[1] + offset_y
        (tw, th), _ = cv2.getTextSize(text, self.FONT, self.FONT_SCALE, 1)
        cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(image, text, (x + 2, y), self.FONT, self.FONT_SCALE, color, 1, cv2.LINE_AA)

    def _draw_right_angle(self, image, pnl, pectoral_line):
        if pectoral_line is None:
            return
        size = 12
        direction = pectoral_line.direction
        perp = np.array([-direction[1], direction[0]])

        foot = np.array(pnl.foot_point, dtype=float)
        nipple = np.array(pnl.nipple_center, dtype=float)
        if np.dot(nipple - foot, perp) < 0:
            perp = -perp

        p1 = foot + direction * size
        p2 = p1 + perp * size
        p3 = foot + perp * size
        pts = np.array([[int(p1[0]), int(p1[1])],
                         [int(p2[0]), int(p2[1])],
                         [int(p3[0]), int(p3[1])]], dtype=np.int32)
        cv2.polylines(image, [pts], False, self.COLOR_PNL_LINE, 1, cv2.LINE_AA)

    def _draw_status(self, image, result):
        lines = []
        if result.is_mlo:
            lines.append("View: MLO")
            lines.append("Pectoral Muscle: DETECTED")
        else:
            lines.append("View: CC")
            lines.append("Pectoral Muscle: NOT FOUND")

        if result.has_nipple:
            lines.append("Nipple: DETECTED")
        else:
            lines.append("Nipple: NOT FOUND")

        if result.pnl is not None:
            lines.append(f"PNL: {result.pnl.distance_px:.1f} px")
        if result.cc_depth is not None:
            lines.append(f"CC Depth: {result.cc_depth.distance_px:.1f} px")

        y_offset = 20
        for line in lines:
            color = self.COLOR_PECTORAL_LINE if "DETECTED" in line or "MLO" in line else (0, 128, 255)
            self._put_label(image, line, (10, y_offset), color)
            y_offset += 25
