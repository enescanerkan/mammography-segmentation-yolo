"""
SOLID OOP: Model Management and Downloader
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Any
from abc import ABC, abstractmethod

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class ModelDownloader:
    """Handles downloading models from Google Drive or GitHub if they are missing."""
    DRIVE_LINKS = {
        "MLO": "https://drive.google.com/drive/folders/1V9j-Hm4j64lh2doTpoj4u07-F-vKNrUJ",
        "CC": "https://drive.google.com/drive/folders/11p_uYnbdJmnIjHbNgKMgkEe7mtsdyVQE"
    }

    @staticmethod
    def ensure_model(view: str, target_path: Path) -> Path:
        if target_path.is_file():
            return target_path
            
        print(f"[ModelDownloader] Weights not found: {target_path}")
        try:
            import gdown
        except ImportError:
            print("[WARN] gdown is not installed. Run `pip install gdown` for automatic downloads.")
            print(f"[WARN] Or download weights from {ModelDownloader.DRIVE_LINKS.get(view, '')} and place at: {target_path}")
            return target_path

        # gdown cannot download an entire folder reliably easily without specific format,
        # but if we had the file IDs it would be direct. Since the user gave folder links,
        # we will attempt folder download if gdown supports it.
        try:
            print(f"[ModelDownloader] Downloading {view} model from Google Drive...")
            url = ModelDownloader.DRIVE_LINKS.get(view)
            if url:
                gdown.download_folder(url=url, output=str(target_path.parent), quiet=False)
                # Note: gdown outputs the folder contents to the directory, it might be nested.
        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
        return target_path


class BaseModel(ABC):
    def __init__(self, weights_path: str):
        self.weights_path = Path(weights_path)
        self.model: Optional[YOLO] = None
        self._load()

    def _load(self):
        if self.weights_path.is_file():
            self.model = YOLO(str(self.weights_path))
        else:
            print(f"[WARN] YOLO weights not found: {self.weights_path}", file=sys.stderr)
            # Default fallback behaviour can be added.
            self.model = YOLO("yolo26l-pose.pt") if "pose" in str(self.weights_path).lower() else None

    @abstractmethod
    def predict(self, image_path: Path, **kwargs) -> Any:
        pass


_WAVELET_WARNED = False


def _wavelet_enhance(im_bgr: np.ndarray) -> np.ndarray:
    """NLM denoise -> Daubechies-4 (level 2) edge boost -> CLAHE.

    The advanced pose models (`*-yolo26-pose-advanced.pt`) were TRAINED on
    images passed through this exact enhancement (NLM+Wavelet+CLAHE). Feeding
    raw images at inference is out-of-distribution and degrades keypoints, so
    the compare pipeline must apply the same preprocessing here. Mirrors
    eklenecek/.../src/preprocessing/strategies.py::AdvancedWaveletStrategy.
    """
    import pywt
    gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) if im_bgr.ndim == 3 else im_bgr
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=5, searchWindowSize=11)
    coeffs = pywt.wavedec2(denoised.astype(np.float64), "db4", level=2)
    boosted = [coeffs[0]]
    for lvl, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        f = 1.3 if lvl == 1 else 1.6
        boosted.append((cH * f, cV * f, cD * f))
    rec = pywt.waverec2(boosted, "db4")[: gray.shape[0], : gray.shape[1]]
    enhanced = np.clip(rec, 0, 255).astype(np.uint8)
    enhanced = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(enhanced)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


class PoseModel(BaseModel):
    def predict(self, image_path: Path, min_keypoints: int = 1) -> Optional[np.ndarray]:
        """Runs multi-confidence inference for mammography YOLO Pose."""
        if self.model is None:
            return None

        im = cv2.imread(str(image_path))
        if im is None:
            return None

        # Pose models are wavelet-trained -> apply matching enhancement.
        # Toggle off with POSE_WAVELET=0 to compare raw-input behaviour.
        if os.environ.get("POSE_WAVELET", "1") != "0":
            try:
                im = _wavelet_enhance(im)
            except Exception as exc:
                # Do NOT fail silently: without the enhancement the pose models
                # run out-of-distribution and accuracy drops (0.815 -> 0.753)
                # with nothing in the output to explain why. Warn once.
                global _WAVELET_WARNED
                if not _WAVELET_WARNED:
                    _WAVELET_WARNED = True
                    print(
                        f"[WARN] Wavelet preprocessing unavailable ({exc}). Pose "
                        f"models are wavelet-trained, so results will be degraded. "
                        f"Install PyWavelets, or set POSE_WAVELET=0 to silence this.",
                        file=sys.stderr,
                    )

        confs = (0.25, 0.15, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001)
        for c in confs:
            res = self.model.predict(source=im, conf=float(c), iou=0.45, imgsz=640, verbose=False)[0]
            if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
                continue
            xy = res.keypoints.xy
            if xy.numel() == 0:
                continue
            
            n_det = int(xy.shape[0])
            if n_det == 0:
                continue
                
            bi = int(res.boxes.conf.argmax().item()) if res.boxes is not None and len(res.boxes) > 0 else 0
            k = xy[bi].cpu().numpy()
            nk = int(k.shape[0])
            
            # Prevent using COCO fallback
            if nk >= 15:
                continue
            if nk < min_keypoints:
                continue
            return k
        return None


class SegmentationModel(BaseModel):
    def predict(self, image_path: Path, config: Any) -> Any:
        if self.model is None:
            return None
        import torch
        device = config.device if torch.cuda.is_available() else "cpu"
        res = self.model.predict(
            source=str(image_path),
            imgsz=config.image_size,
            device=device,
            conf=config.confidence_threshold,
            iou=config.iou_threshold,
            verbose=False,
            retina_masks=True,
        )[0]
        return res


class ModelFactory:
    """Factory to instantiate and ensure models."""
    def __init__(self, weights_dir: Path):
        self.weights_dir = weights_dir

    def get_pose_model(self, view: str) -> PoseModel:
        target_name = f"{view.lower()}-yolo26-pose-advanced.pt"
        path = self.weights_dir / target_name
        path = ModelDownloader.ensure_model(view, path)
        return PoseModel(str(path))

    def get_segmentation_model(self, weights_path: Path) -> SegmentationModel:
        return SegmentationModel(str(weights_path))
