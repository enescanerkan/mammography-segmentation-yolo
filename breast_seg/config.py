"""Centralized configuration for the breast segmentation project."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


def _default_train_device() -> str:
    """First CUDA GPU (`0`) by default. Override with env `BREAST_SEG_DEVICE` (e.g. `cpu`, `1`)."""
    return os.environ.get("BREAST_SEG_DEVICE", "0")


def _default_workers() -> int:
    """Windows'ta çok iş parçacığı DataLoader ilk epoch öncesi takılma/sessiz çıkışa yol açabiliyor.

    Geçersiz kıl: `BREAST_SEG_WORKERS` (örn. 4).
    """
    raw = os.environ.get("BREAST_SEG_WORKERS")
    if raw is not None:
        return max(0, int(raw))
    return 0 if sys.platform == "win32" else 4


@dataclass
class Config:
    """Immutable project configuration.

    Single source of truth for all paths, model parameters,
    and class definitions used across the project.
    """

    # ── Paths ──────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parents[1]
    base_dir: Path = Path(__file__).resolve().parents[1] / "seg-dataset"

    @property
    def data_yaml(self) -> Path:
        return self.base_dir / "data.yaml"

    @property
    def weights_path(self) -> Path:
        return self.project_root / "runs" / "breast_seg_yolo26m" / "weights" / "best.pt"

    @property
    def test_images_dir(self) -> Path:
        return self.base_dir / "images" / "test"

    @property
    def predictions_dir(self) -> Path:
        return self.project_root / "predictions"

    @property
    def analysis_output_dir(self) -> Path:
        return self.project_root / "analysis_output"

    @property
    def runs_dir(self) -> Path:
        return self.project_root / "runs"

    # ── Class Mapping (seg-dataset/data.yaml ile uyumlu) ──
    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: {
        0: "pectoral",
        1: "breast-tissue",
        2: "nipple",
    })

    PECTORAL_MUSCLE_CLASS_ID: int = 0
    BREAST_TISSUE_CLASS_ID: int = 1
    NIPPLE_CLASS_ID: int = 2

    # ── Model Defaults ─────────────────────────────────────
    model_name: str = "yolo26m-seg.pt"
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = field(default_factory=_default_train_device)

    # ── Training Defaults ──────────────────────────────────
    epochs: int = 100
    batch_size: int = 8
    patience: int = 20
    workers: int = field(default_factory=_default_workers)
    run_name: str = "breast_seg_yolo26m"
    # AMP (FP16): Windows + bazı laptop GPU / cuDNN sürümlerinde
    # CUDNN_STATUS_EXECUTION_FAILED_CUDART üretebiliyor. Varsayılan kapalı.
    # Hız için denemek istersen: BREAST_SEG_AMP=1 ortam değişkeni veya True yap.
    use_amp: bool = field(
        default_factory=lambda: os.environ.get("BREAST_SEG_AMP", "").lower() in ("1", "true", "yes")
    )

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
