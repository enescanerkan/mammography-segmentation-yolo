"""Centralized configuration for the breast segmentation project."""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


def _default_train_device() -> str:
    """CUDA GPU `0` if available; otherwise `cpu`. Override with `BREAST_SEG_DEVICE`."""
    raw = os.environ.get("BREAST_SEG_DEVICE")
    if raw is not None and str(raw).strip() != "":
        return raw
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _env_int(name: str, default: int) -> int:
    """Read positive int from env, or use default."""
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return max(1, int(raw))


def _default_workers() -> int:
    """On Windows, many DataLoader workers can hang or exit silently before epoch 1.

    Override with `BREAST_SEG_WORKERS` (e.g. 4).
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

    # Dataset generation. Every version is a full re-export, not an increment:
    #   v1 ""   ->   169 images (hand-labelled seed set)
    #   v2      ->  1857 images (+ MedSAM2-assisted labels)
    #   v3      ->  2924 images (+ CC pseudo-labels, reviewed)
    #   v4      ->  2924 images, split from the compare CSV so the Test cases
    #               are held out -> the compare pipeline stays leak-free.
    # Override with BREAST_SEG_DATASET (e.g. "_v3", or "" for the seed set).
    dataset_version: str = field(
        default_factory=lambda: os.environ.get("BREAST_SEG_DATASET", "_v4")
    )

    @property
    def base_dir(self) -> Path:
        return self.project_root / f"seg-dataset{self.dataset_version}"

    @property
    def data_yaml(self) -> Path:
        return self.base_dir / "data.yaml"

    @property
    def run_name(self) -> str:
        """`breast_seg_v4_yolo26m` for v4, `breast_seg_yolo26m` for the seed set."""
        suffix = self.dataset_version.lstrip("_")
        return f"breast_seg_{suffix}_yolo26m" if suffix else "breast_seg_yolo26m"

    @property
    def weights_path(self) -> Path:
        """Best checkpoint of the run matching `dataset_version`.

        Kept in sync with `pipeline/orchestrator.py`, which loads the same
        weights for the compare pipeline. Before, this property still pointed
        at the v1 run while the orchestrator had moved to v4.
        """
        return self.runs_dir / self.run_name / "weights" / "best.pt"

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

    # ── Class mapping (aligned with seg-dataset/data.yaml) ──
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
    # Override from terminal: BREAST_SEG_EPOCHS, BREAST_SEG_BATCH, BREAST_SEG_PATIENCE
    epochs: int = field(default_factory=lambda: _env_int("BREAST_SEG_EPOCHS", 100))
    batch_size: int = field(default_factory=lambda: _env_int("BREAST_SEG_BATCH", 8))
    patience: int = field(default_factory=lambda: _env_int("BREAST_SEG_PATIENCE", 15))
    workers: int = field(default_factory=_default_workers)
    # AMP (FP16): some Windows laptops / cuDNN builds raise CUDNN_STATUS_EXECUTION_FAILED_CUDART.
    # Disabled by default. To try AMP: set env BREAST_SEG_AMP=1 or force True below.
    use_amp: bool = field(
        default_factory=lambda: os.environ.get("BREAST_SEG_AMP", "").lower() in ("1", "true", "yes")
    )

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)
