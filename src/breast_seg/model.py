"""YOLO segmentation model wrapper for inference (training goes through :mod:`breast_seg.training`)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from breast_seg.config import Config


class SegmentationModel:
    """Lazy ``ultralytics.YOLO`` wrapper: load on first use, predict with the :class:`Config` defaults."""

    def __init__(self, config: Optional[Config] = None, weights: Optional[Union[str, Path]] = None):
        self._config = config or Config()
        self._weights = Path(weights) if weights else self._config.weights_path
        self._model: Any = None

    @property
    def weights(self) -> Optional[Path]:
        return self._weights

    def _load(self) -> Any:
        if self._model is None:
            if self._weights is None:
                raise ValueError("no segmentation weights given (argument, Config.weights_path or BREAST_SEG_SEG_WEIGHTS)")
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights))
        return self._model

    @property
    def model(self) -> Any:
        """The underlying ``ultralytics.YOLO`` object."""
        return self._load()

    def predict(self, source: Union[str, Path, Any], **overrides: Any) -> List[Any]:
        """Ultralytics ``predict`` with the configured image size, thresholds and device; ``overrides`` win."""
        cfg = self._config
        kwargs = dict(imgsz=cfg.image_size, conf=cfg.confidence_threshold, iou=cfg.iou_threshold, device=cfg.device,
                      retina_masks=True, verbose=False)
        kwargs.update(overrides)
        return self._load().predict(source=str(source) if isinstance(source, Path) else source, **kwargs)

    @staticmethod
    def train(config_path: Union[str, Path], overrides: Optional[List[str]] = None) -> Any:
        """Train with a ``configs/train/*.yaml`` recipe (see :func:`breast_seg.training.build_trainer`)."""
        from breast_seg.training import build_trainer

        return build_trainer(config_path, overrides=overrides).fit()
