"""Structural types of the compare pipeline: concrete classes satisfy these protocols, and tests can substitute
fakes with the same shape."""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Protocol, runtime_checkable

from pandas import Series


@runtime_checkable
class PosePredictor(Protocol):
    def predict(self, image_path: Path, min_keypoints: int = 1) -> Any:
        """Keypoints ``(K, 2)`` on the canvas, or ``None`` when nothing is detected."""
        ...


@runtime_checkable
class SegmentationRunner(Protocol):
    def predict(self, image_path: Path, config: Any) -> Any:
        """An Ultralytics ``Results`` object (or equivalent)."""
        ...


@runtime_checkable
class ClinicalDataset(Protocol):
    def get_test_pairs(self, limit: int = 0) -> List[Any]:
        ...

    def get_mlo_metadata(self, sop: str) -> Optional[Series]:
        ...

    def get_cc_metadata(self, sop: str) -> Optional[Series]:
        ...
