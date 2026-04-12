"""MLO/CC analysis pipeline — orchestrates mask extraction and geometric computations."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from breast_seg.config import Config
from breast_seg.geometry import (
    CCDepthResult,
    PectoralLine,
    PNLResult,
    compute_cc_depth,
    compute_nipple_centroid,
    compute_pnl,
    fit_pectoral_line,
)


@dataclass
class AnalysisResult:
    """Complete analysis output for a single mammography image.

    Works for both MLO (pectoral line + PNL) and CC (depth) views.
    """
    image_path: str
    image: np.ndarray
    pectoral_line: Optional[PectoralLine]
    pnl: Optional[PNLResult]
    cc_depth: Optional[CCDepthResult]
    has_pectoral: bool
    has_nipple: bool
    is_mlo: bool  # True = MLO (has pectoral), False = CC view
    pectoral_mask: Optional[np.ndarray] = None
    breast_mask: Optional[np.ndarray] = None
    nipple_mask: Optional[np.ndarray] = None


class MLOAnalyzer:
    """Analyzes YOLO segmentation results for mammography images.

    Handles both MLO views (pectoral line + PNL) and CC views (nipple depth).
    """

    def __init__(self, config: Config):
        self._config = config

    def analyze_single(self, yolo_result) -> AnalysisResult:
        """Analyze a single YOLO result for one image.

        - If pectoral muscle is found → MLO view → compute pectoral line + PNL
        - If no pectoral muscle → CC view → compute CC depth (nipple to breast edge)
        """
        image = yolo_result.orig_img.copy()
        h, w = image.shape[:2]
        image_path = yolo_result.path

        # Extract masks per class
        nipple_mask = self._extract_class_mask(
            yolo_result, self._config.NIPPLE_CLASS_ID, (h, w)
        )
        breast_mask = self._extract_class_mask(
            yolo_result, self._config.BREAST_TISSUE_CLASS_ID, (h, w)
        )
        pectoral_mask = self._extract_class_mask(
            yolo_result, self._config.PECTORAL_MUSCLE_CLASS_ID, (h, w)
        )

        has_pectoral = pectoral_mask is not None
        has_nipple = nipple_mask is not None
        is_mlo = has_pectoral

        pectoral_line = None
        pnl = None
        cc_depth = None

        if is_mlo:
            # ── MLO View: Pectoral Line + PNL ──
            if breast_mask is not None:
                pectoral_line = fit_pectoral_line(pectoral_mask, breast_mask, (h, w))

            if has_nipple and pectoral_line is not None:
                centroid = compute_nipple_centroid(nipple_mask)
                if centroid is not None:
                    pnl = compute_pnl(centroid, pectoral_line)
        else:
            # ── CC View: Nipple to Breast Edge ──
            if has_nipple and breast_mask is not None:
                centroid = compute_nipple_centroid(nipple_mask)
                if centroid is not None:
                    cc_depth = compute_cc_depth(centroid, breast_mask, (h, w))

        return AnalysisResult(
            image_path=image_path,
            image=image,
            pectoral_line=pectoral_line,
            pnl=pnl,
            cc_depth=cc_depth,
            has_pectoral=has_pectoral,
            has_nipple=has_nipple,
            is_mlo=is_mlo,
            pectoral_mask=pectoral_mask,
            breast_mask=breast_mask,
            nipple_mask=nipple_mask,
        )

    def analyze_predictions(self, yolo_results: List) -> List[AnalysisResult]:
        """Analyze a batch of YOLO prediction results."""
        return [self.analyze_single(r) for r in yolo_results]

    @staticmethod
    def _extract_class_mask(
        yolo_result, class_id: int, shape: tuple
    ) -> Optional[np.ndarray]:
        """Extract and merge all masks for a given class ID."""
        if yolo_result.masks is None:
            return None

        masks_data = yolo_result.masks.data.cpu().numpy()
        classes = yolo_result.boxes.cls.cpu().numpy().astype(int)

        indices = [i for i, c in enumerate(classes) if c == class_id]
        if not indices:
            return None

        h, w = shape
        combined = np.zeros((h, w), dtype=np.uint8)

        for idx in indices:
            mask = masks_data[idx]
            if mask.shape != (h, w):
                import cv2
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            combined = np.maximum(combined, (mask > 0.5).astype(np.uint8) * 255)

        return combined if combined.any() else None
