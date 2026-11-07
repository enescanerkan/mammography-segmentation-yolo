"""
Model manager for loading and managing MLO and CC YOLO pose models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

from ..utils.paths import weight_paths


class IModelManager(ABC):
    """Interface for model managers."""

    @abstractmethod
    def load_models(self) -> None:
        pass

    @abstractmethod
    def predict_landmarks(self, image: np.ndarray, model_type: str) -> Optional[np.ndarray]:
        pass


class ModelManager(IModelManager):
    """Manager for loading and using MLO and CC YOLO pose landmark detection models."""

    def __init__(self):
        import torch  # lazy: keeps importing this module cheap

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}

    def load_models(self) -> None:
        """Load both pose models from ``positioning.mlo_weights`` / ``positioning.cc_weights``."""
        from ultralytics import YOLO

        for view, path in weight_paths().items():
            if not path.is_file():
                raise FileNotFoundError(f"{view.upper()} pose model not found: {path}")
            self.models[view] = YOLO(str(path))

    def predict_landmarks(self, image: np.ndarray, model_type: str,
                         original_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """Predict landmarks using YOLO pose model.

        Args:
            image: Input image array (640x640 preprocessed, float32 0-1)
            model_type: 'mlo' or 'cc'
            original_shape: Not used, kept for compatibility

        Returns:
            Predicted landmarks as (N, 2) numpy array in 640x640 pixel space.
            MLO: [[nipple_x, nipple_y], [pec_top_x, pec_top_y], [pec_bottom_x, pec_bottom_y]]
            CC:  [[nipple_x, nipple_y]]
        """
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not available")

        try:
            model = self.models[model_type]

            if len(image.shape) == 3:
                image = image[0]

            img = image.copy()
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            results = model(img, verbose=False)[0]

            if results.keypoints is None or len(results.keypoints.xy) == 0:
                return None

            kp = results.keypoints.xy[0].cpu().numpy()
            return kp if len(kp) > 0 else None

        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def is_model_loaded(self, model_type: str) -> bool:
        return model_type in self.models

    def get_available_models(self) -> list:
        return list(self.models.keys())
