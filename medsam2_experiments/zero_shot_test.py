"""
MedSAM 2 zero-shot: derive a box from the GT mask, prompt the model, report Dice per class.

Run from an environment where MedSAM2 is installed (`pip install -e .` inside the MedSAM repo).
This script temporarily `chdir`s to `MEDSAM2_REPO` so Hydra can resolve `sam2_hiera_t.yaml`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _ensure_medsam_imports(medsam_repo: Path) -> None:
    if not (medsam_repo / "sam2").is_dir():
        raise SystemExit(
            f"MedSAM2 repo not found at {medsam_repo}. "
            "Clone with: git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git\n"
            "Then: set MEDSAM2_REPO to that path or place it at medsam2_experiments/vendor/MedSAM"
        )
    sys.path.insert(0, str(medsam_repo))


def mask_to_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def clip_pad_bbox(bbox: np.ndarray, w: int, h: int, pad: int) -> np.ndarray:
    out = bbox + np.array([-pad, -pad, pad, pad], dtype=np.float32)
    out[0] = np.clip(out[0], 0, w - 1)
    out[1] = np.clip(out[1], 0, h - 1)
    out[2] = np.clip(out[2], 0, w - 1)
    out[3] = np.clip(out[3], 0, h - 1)
    return out


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum() + 1e-8
    return float(2.0 * inter / denom)


def load_mask_classes(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None, help="Override RAW_MAMMO_ROOT")
    parser.add_argument("--ckpt", type=Path, default=None, help="Override MedSAM2_latest.pt path")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    from config import (
        BBOX_PAD_PX,
        MASK_CLASS_LABELS,
        MEDSAM2_REPO,
        MEDSAM2_WEIGHTS,
        MIN_FG_PIXELS,
        RAW_MAMMO_ROOT,
        SAM2_MODEL_CFG,
    )

    data_root = args.data_root or RAW_MAMMO_ROOT
    ckpt = args.ckpt or MEDSAM2_WEIGHTS
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    img_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected {img_dir} and {mask_dir}. See medsam2_experiments/README.md.")

    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint missing: {ckpt}. Download per README.")

    _ensure_medsam_imports(Path(MEDSAM2_REPO))
    prev = os.getcwd()
    os.chdir(MEDSAM2_REPO)
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2 = build_sam2(SAM2_MODEL_CFG, str(ckpt), device=device, apply_postprocessing=True)
        predictor = SAM2ImagePredictor(sam2)
    finally:
        os.chdir(prev)

    results: dict[int, list[tuple[str, float]]] = {k: [] for k in MASK_CLASS_LABELS}

    for img_path in sorted(img_dir.glob("*.png")):
        case_id = img_path.stem
        mask_path = mask_dir / f"{case_id}.png"
        if not mask_path.is_file():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        img_rgb = np.stack([img, img, img], axis=-1)
        gt_mask = load_mask_classes(mask_path)

        predictor.set_image(img_rgb)

        for cls_id, cls_name in MASK_CLASS_LABELS.items():
            gt_binary = gt_mask == int(cls_id)
            if gt_binary.sum() < MIN_FG_PIXELS:
                continue

            bbox = mask_to_bbox_xyxy(gt_binary)
            if bbox is None:
                continue
            bbox = clip_pad_bbox(bbox, w, h, BBOX_PAD_PX)

            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox[None, :],
                multimask_output=False,
            )
            pred = masks[0].astype(bool)
            if pred.shape != gt_binary.shape:
                pred = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            d = dice(pred, gt_binary)
            results[int(cls_id)].append((case_id, d))
            print(f"{case_id} {cls_name}: Dice={d:.3f} (score={float(scores[0]):.3f})")

    print("\n=== Zero-shot summary ===")
    for cls_id, cls_name in MASK_CLASS_LABELS.items():
        rows = results.get(int(cls_id), [])
        if not rows:
            continue
        dices = [d for _, d in rows]
        print(f"{cls_name:20s}: mean Dice = {np.mean(dices):.3f}  (n={len(dices)})")


if __name__ == "__main__":
    main()
