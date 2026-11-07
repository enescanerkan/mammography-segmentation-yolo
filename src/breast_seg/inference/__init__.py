"""Model inference: the landmark / segmentation / lesion engine behind the positioning review.

Ultralytics is imported only when a model is actually loaded.
"""
from .engine import (
    MIN_MASK_AREA_PX,
    POLYGON_EPS,
    REQUIRED_ROLES,
    ROLES,
    EngineError,
    LandmarkEngine,
    landmark_record,
    largest_polygon,
)
from .loader import ModelLoader, UnsupportedModel, ultralytics_loader

__all__ = [
    "MIN_MASK_AREA_PX",
    "POLYGON_EPS",
    "REQUIRED_ROLES",
    "ROLES",
    "EngineError",
    "LandmarkEngine",
    "ModelLoader",
    "UnsupportedModel",
    "landmark_record",
    "largest_polygon",
    "ultralytics_loader",
]
