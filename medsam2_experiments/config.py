"""
Paths and label semantics for MedSAM2 experiments in this repo.

Edit MASK_CLASS_LABELS to match your multi-class PNG mask pixel values.
"""

from __future__ import annotations

import os
from pathlib import Path

# This folder (medsam2_experiments/)
EXP_ROOT = Path(__file__).resolve().parent
# mammography-segmentation-yolo/
PROJECT_ROOT = EXP_ROOT.parent

# Clone MedSAM here or set env MEDSAM2_REPO to the repo root (MedSAM2 branch).
MEDSAM2_REPO = Path(os.environ.get("MEDSAM2_REPO", str(EXP_ROOT / "vendor" / "MedSAM")))

# SAM2 config file name (lives inside MedSAM2 repo, e.g. sam2_hiera_t.yaml)
SAM2_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "sam2_hiera_t.yaml")

# Checkpoints (download into medsam2_experiments/checkpoints/)
CHECKPOINT_DIR = EXP_ROOT / "checkpoints"
MEDSAM2_WEIGHTS = CHECKPOINT_DIR / "MedSAM2_latest.pt"
SAM2_BASE_WEIGHTS = CHECKPOINT_DIR / "sam2_hiera_tiny.pt"

# Raw data layout (you provide this — not the YOLO txt layout):
#   data/raw_mammo/images/{case}.png
#   data/raw_mammo/masks/{case}.png   # single-channel, integer class ids
RAW_MAMMO_ROOT = EXP_ROOT / "data" / "raw_mammo"

# Multi-class mask PNG: 0 = background, 1 = YOLO class 0, 2 = YOLO class 1, 3 = YOLO class 2
# (matches seg-dataset/data.yaml after `export_seg_dataset_to_raw_mammo.py`)
MASK_CLASS_LABELS: dict[int, str] = {
    1: "pectoral",
    2: "breast-tissue",
    3: "nipple",
}

# Minimum foreground pixels to run a class (skip tiny / missing regions, e.g. CC pectoral)
MIN_FG_PIXELS = int(os.environ.get("MEDSAM2_MIN_FG", "50"))

# Prepared .npy layout for bowang-lab MedSAM2 finetune_sam2_img.py:
# imgs: (1024, 1024, 3) uint8
# gts:  (256, 256) uint8 with values {0,1} per file (one binary structure per npy pair)
NPY_OUT_ROOT = EXP_ROOT / "data" / "medsam2_npy"

# Bbox padding when deriving box from GT (zero-shot evaluation)
BBOX_PAD_PX = int(os.environ.get("MEDSAM2_BBOX_PAD", "10"))
