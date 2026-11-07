"""Models of the compare pipeline: the landmark (pose) route and the segmentation route.

Ultralytics is imported when a model is first loaded, so the pipeline can be built with fakes in tests.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

#: Confidence ladder of the pose route: the first level that returns a detection wins.
POSE_CONFIDENCES: Sequence[float] = (0.25, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)


def wavelet_enhance(image_bgr: np.ndarray) -> np.ndarray:
    """NLM denoise -> Daubechies-4 (level 2) detail boost -> CLAHE, the enhancement the landmark models were
    trained on (feeding raw canvases is out of distribution for them)."""
    import pywt

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=5, searchWindowSize=11)
    coeffs = pywt.wavedec2(denoised.astype(np.float64), "db4", level=2)
    boosted = [coeffs[0]] + [tuple(c * (1.3 if lvl == 1 else 1.6) for c in det) for lvl, det in enumerate(coeffs[1:], 1)]
    rec = pywt.waverec2(boosted, "db4")[: gray.shape[0], : gray.shape[1]]
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(np.clip(rec, 0, 255).astype(np.uint8))
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class BaseModel(ABC):
    """Lazily loaded Ultralytics model."""

    def __init__(self, weights_path: Optional[Path]):
        self.weights_path = Path(weights_path) if weights_path else None
        self._model: Any = None

    @property
    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.is_file()

    @property
    def model(self) -> Any:
        if self._model is None:
            if not self.available:
                raise FileNotFoundError("model weights not found (see docs/MODELS.md for the expected file names)")
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path))
        return self._model

    @abstractmethod
    def predict(self, image_path: Path, *args: Any, **kwargs: Any) -> Any:
        ...


class PoseModel(BaseModel):
    """Landmark model: returns the keypoints ``(K, 2)`` of the most confident detection (canvas pixels)."""

    def __init__(self, weights_path: Optional[Path], wavelet: bool = True, confidences: Sequence[float] = POSE_CONFIDENCES):
        super().__init__(weights_path)
        self.wavelet, self.confidences = wavelet, tuple(confidences)
        self._warned = False

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if not self.wavelet:
            return image
        try:
            return wavelet_enhance(image)
        except ImportError as exc:
            if not self._warned:
                self._warned = True
                print(f"[WARN] wavelet preprocessing unavailable ({exc}); the landmark models are wavelet-trained, so "
                      "their output degrades. Install PyWavelets.", file=sys.stderr)
            return image

    def predict(self, image_path: Path, min_keypoints: int = 1) -> Optional[np.ndarray]:
        if not self.available:
            return None
        im = cv2.imread(str(image_path))
        if im is None:
            return None
        im = self._prepare(im)
        for c in self.confidences:
            res = self.model.predict(source=im, conf=float(c), iou=0.45, imgsz=640, verbose=False)[0]
            if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
                continue
            best = int(res.boxes.conf.argmax().item()) if res.boxes is not None and len(res.boxes) > 0 else 0
            k = res.keypoints.xy[best].cpu().numpy()
            if k.shape[0] >= 15 or k.shape[0] < min_keypoints:  # >= 15 keypoints = a generic COCO pose model
                continue
            return k
        return None


class SegmentationRunner(BaseModel):
    """YOLO segmentation model; returns the Ultralytics result of one image."""

    def predict(self, image_path: Path, config: Any) -> Any:
        if not self.available:
            return None
        import torch

        device = config.device if torch.cuda.is_available() else "cpu"
        return self.model.predict(source=str(image_path), imgsz=config.image_size, device=device,
                                  conf=config.confidence_threshold, iou=config.iou_threshold, verbose=False,
                                  retina_masks=True)[0]


class ModelFactory:
    """Builds the compare models from explicit weight paths (pose weights default to the released file names)."""

    POSE_FILES = {"MLO": "mlo-yolo26-pose-advanced.pt", "CC": "cc-yolo26-pose-advanced.pt"}

    def __init__(self, weights_dir: Optional[Path] = None, wavelet: bool = True):
        self.weights_dir = Path(weights_dir) if weights_dir else None
        self.wavelet = wavelet

    def get_pose_model(self, view: str, weights: Optional[Path] = None) -> PoseModel:
        if weights is None and self.weights_dir is not None:
            weights = self.weights_dir / self.POSE_FILES[view.upper()]
        return PoseModel(weights, wavelet=self.wavelet)

    def get_segmentation_model(self, weights_path: Optional[Path]) -> SegmentationRunner:
        return SegmentationRunner(weights_path)
