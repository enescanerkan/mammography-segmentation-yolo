"""Centralized configuration for the breast segmentation project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Config:
    """Immutable project configuration.

    Single source of truth for all paths, model parameters,
    and class definitions used across the project.
    """

    # ── Paths ──────────────────────────────────────────────
    base_dir: Path = Path(r"C:\Users\Monster\Desktop\segment_breast")

    @property
    def data_yaml(self) -> Path:
        return self.base_dir / "data.yaml"

    @property
    def weights_path(self) -> Path:
        return self.base_dir / "runs" / "breast_seg_yolo11" / "weights" / "best.pt"

    @property
    def test_images_dir(self) -> Path:
        return self.base_dir / "images" / "test"

    @property
    def predictions_dir(self) -> Path:
        return self.base_dir / "predictions"

    @property
    def analysis_output_dir(self) -> Path:
        return self.base_dir / "analysis_output"

    @property
    def runs_dir(self) -> Path:
        return self.base_dir / "runs"

    # ── Class Mapping ──────────────────────────────────────
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
        0: "Nipple",
        1: "Breast Tissue",
        2: "Pectoral Muscle",
    })

    NIPPLE_CLASS_ID: int = 0
    BREAST_TISSUE_CLASS_ID: int = 1
    PECTORAL_MUSCLE_CLASS_ID: int = 2

    # ── Model Defaults ─────────────────────────────────────
    model_name: str = "yolo11n-seg.pt"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = "0"

    # ── Training Defaults ──────────────────────────────────
    epochs: int = 100
    batch_size: int = 8
    patience: int = 20
    workers: int = 4

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
