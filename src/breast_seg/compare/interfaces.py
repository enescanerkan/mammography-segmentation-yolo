"""
Structural typing for the comparison pipeline (SOLID: dependency inversion,
interface segregation). Concrete classes in `models.py` satisfy these
protocols; callers can be tested or extended against these shapes only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from pandas import Series


@runtime_checkable
class PosePredictor(Protocol):
    def predict(self, image_path: Path, min_keypoints: int = 1) -> Any:
        """Return keypoints array or None if detection fails."""
        ...


@runtime_checkable
class SegmentationRunner(Protocol):
    def predict(self, image_path: Path, config: Any) -> Any:
        """Return an ultralytics Results object (or equivalent)."""
        ...


@runtime_checkable
class ClinicalDataset(Protocol):
    def get_test_pairs(self, limit: int = 0) -> list:
        ...

    def get_mlo_metadata(self, sop: str) -> Optional[Series]:
        ...

    def get_cc_metadata(self, sop: str) -> Optional[Series]:
        ...
