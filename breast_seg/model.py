"""YOLO segmentation model wrapper following Single Responsibility Principle."""

from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

from breast_seg.config import Config


class SegmentationModel:
    """Wrapper around YOLO for breast segmentation tasks.

    Responsibilities: model loading, training, and inference only.
    """

    def __init__(self, config: Config, weights: Optional[Path] = None):
        """Load a YOLO segmentation model.

        Args:
            config: Project configuration.
            weights: Path to model weights. If None, uses config.weights_path
                     for inference or config.model_name for training.
        """
        self._config = config
        self._weights = weights
        self._model: Optional[YOLO] = None

    def _load(self, weights_path: Path) -> YOLO:
        """Lazy load the YOLO model."""
        if self._model is None or self._weights != weights_path:
            self._weights = weights_path
            self._model = YOLO(str(weights_path))
        return self._model

    def train(self) -> None:
        """Train the segmentation model using settings from config."""
        model = self._load(Path(self._config.model_name))
        cfg = self._config

        model.train(
            data=str(cfg.data_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.image_size,
            batch=cfg.batch_size,
            patience=cfg.patience,
            device=cfg.device,
            workers=cfg.workers,
            project=str(cfg.runs_dir),
            name="breast_seg_yolo11",
            exist_ok=True,
            # Augmentation
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            # Optimizer
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            # Segmentation
            overlap_mask=True,
            mask_ratio=4,
            # Output
            verbose=True,
            save=True,
            save_period=10,
            plots=True,
        )
        print(f"\n[INFO] Training complete. Results saved to: {cfg.runs_dir}")

    def predict(self, source: Path, **overrides) -> List:
        """Run segmentation inference.

        Args:
            source: Path to images directory or single image.
            **overrides: Additional YOLO predict kwargs.

        Returns:
            List of ultralytics Results objects.
        """
        weights = self._weights or self._config.weights_path
        model = self._load(weights)
        cfg = self._config

        default_kwargs = dict(
            source=str(source),
            imgsz=cfg.image_size,
            conf=cfg.confidence_threshold,
            iou=cfg.iou_threshold,
            device=cfg.device,
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(cfg.base_dir),
            name="predictions",
            exist_ok=True,
            retina_masks=True,
            show_boxes=False,
            show_labels=True,
            show_conf=True,
            line_width=2,
        )
        default_kwargs.update(overrides)
        return model.predict(**default_kwargs)
