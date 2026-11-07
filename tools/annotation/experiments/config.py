"""Paths and label semantics for the MedSAM2 research scripts in this folder.

Model locations (MedSAM2 code, SAM2 base weights, fine-tuned checkpoint) come from the
toolkit configuration (``tools/annotation/toolkit_config.py``) so the scripts and the
desktop app always agree. Data / output folders below are local to ``experiments/``
and git-ignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# experiments/ -> tools/annotation/ -> tools/ -> repository root
EXP_ROOT = Path(__file__).resolve().parent
TOOLKIT_ROOT = EXP_ROOT.parent
PROJECT_ROOT = EXP_ROOT.parents[2]

if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

from toolkit_config import get_config  # noqa: E402  (needs the sys.path entry above)

_cfg = get_config()

MEDSAM2_REPO = _cfg.medsam2_repo
SAM2_MODEL_CFG = _cfg.sam2_model_cfg
SAM2_BASE_WEIGHTS = _cfg.sam2_base_weights
MEDSAM2_CHECKPOINT = _cfg.medsam2_checkpoint

# zero_shot_test.py loads this into build_sam2(SAM2_MODEL_CFG, ckpt, ...). SAM2 Hiera-Tiny
# by default; the Hugging Face MedSAM2_latest.pt may not match sam2_hiera_t.yaml on every
# MedSAM2 revision, so it is opt-in via MEDSAM2_ZERO_SHOT_CKPT.
ZERO_SHOT_CKPT = Path(os.environ.get("MEDSAM2_ZERO_SHOT_CKPT", str(SAM2_BASE_WEIGHTS)))

# YOLO-seg dataset (images/{split}/*.png + labels/{split}/*.txt) used as the GT source.
SEG_DATASET_ROOT = Path(os.environ.get("SEG_DATASET_ROOT", str(PROJECT_ROOT / "seg-dataset")))

# Raster layout produced by export_seg_dataset_to_raw_mammo.py:
#   data/raw_mammo/images/{case}.png
#   data/raw_mammo/masks/{case}.png   # single-channel, integer class ids
RAW_MAMMO_ROOT = EXP_ROOT / "data" / "raw_mammo"

# Multi-class mask PNG: 0 = background, 1 = YOLO class 0, 2 = YOLO class 1, 3 = YOLO class 2
MASK_CLASS_LABELS: dict[int, str] = {
    1: "pectoral",
    2: "breast-tissue",
    3: "nipple",
}

# Minimum foreground pixels to use a class (skips tiny / missing regions, e.g. CC pectoral).
MIN_FG_PIXELS = int(os.environ.get("MEDSAM2_MIN_FG", "50"))

# Prepared .npy layout for bowang-lab MedSAM2 finetune_sam2_img.py:
#   imgs: (1024, 1024, 3) uint8
#   gts:  (256, 256) uint8 with values {0, 1} (one binary structure per npy pair)
NPY_OUT_ROOT = EXP_ROOT / "data" / "medsam2_npy"

# Fine-tune outputs (finetune_sam2_img.py writes <work_dir>/<task_name>-<timestamp>/).
WORK_DIR = EXP_ROOT / "work_dir"

# Box padding when deriving the prompt box from the GT mask (zero-shot evaluation).
BBOX_PAD_PX = int(os.environ.get("MEDSAM2_BBOX_PAD", "10"))
