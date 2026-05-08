"""
Box-prompt inference with a fine-tuned MedSAM2 checkpoint (`medsam_model_best.pth`).

Requires MedSAM2 installed and `chdir` to `MEDSAM2_REPO` for `build_sam2`.
Base SAM2 tiny weights must match the architecture used during fine-tune.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def mask_to_bbox_xyxy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--gt-mask", type=Path, default=None, help="Optional PNG; if set, bbox is derived from class id")
    parser.add_argument("--cls", type=int, default=1, help="Class id in mask PNG for bbox derivation")
    parser.add_argument("--bbox", type=str, default=None, help="Override: 'x0,y0,x1,y1' in pixel coords")
    parser.add_argument("--medsam2-ckpt", type=Path, required=True, help="medsam_model_best.pth from finetune")
    parser.add_argument("--sam2-base", type=Path, default=None, help="sam2_hiera_tiny.pt (SAM2 base)")
    parser.add_argument("--out", type=Path, default=None, help="Save binary mask PNG here")
    args = parser.parse_args()

    from config import MEDSAM2_REPO, SAM2_BASE_WEIGHTS, SAM2_MODEL_CFG

    repo = Path(MEDSAM2_REPO)
    if not (repo / "sam2").is_dir():
        raise SystemExit("Set MEDSAM2_REPO to a MedSAM2 checkout.")

    sam2_base = args.sam2_base or SAM2_BASE_WEIGHTS
    sys.path.insert(0, str(repo))

    prev = os.getcwd()
    os.chdir(repo)
    try:
        from sam2.build_sam import build_sam2
        from sam2.utils.transforms import SAM2Transforms
    finally:
        os.chdir(prev)

    if not sam2_base.is_file():
        raise SystemExit(f"Missing SAM2 base weights: {sam2_base}")
    if not args.medsam2_ckpt.is_file():
        raise SystemExit(f"Missing fine-tuned checkpoint: {args.medsam2_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(repo)
    try:
        sam2_model = build_sam2(SAM2_MODEL_CFG, str(sam2_base), device=str(device), mode="eval", apply_postprocessing=True)
    finally:
        os.chdir(prev)

    from wrapped_model import MedSAM2, medsam2_segment_from_box

    ckpt = torch.load(args.medsam2_ckpt, map_location=str(device))
    medsam = MedSAM2(model=sam2_model)
    medsam.load_state_dict(ckpt["model"], strict=True)
    medsam.eval()
    sam2_transforms = SAM2Transforms(resolution=1024, mask_threshold=0)

    img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    h, w = img.shape
    img_rgb = np.stack([img, img, img], axis=-1)

    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            raise SystemExit("--bbox must be x0,y0,x1,y1")
        box = np.array(parts, dtype=np.float32)
    elif args.gt_mask and args.gt_mask.is_file():
        m = cv2.imread(str(args.gt_mask), cv2.IMREAD_UNCHANGED)
        if m is not None and m.ndim == 3:
            m = m[:, :, 0]
        if m is None:
            raise SystemExit("Could not read --gt-mask")
        binm = m == int(args.cls)
        box = mask_to_bbox_xyxy(binm)
        if box is None:
            raise SystemExit("Empty mask for requested class; cannot build bbox.")
    else:
        raise SystemExit("Provide --bbox or (--gt-mask and --cls).")

    seg, prob = medsam2_segment_from_box(medsam, img_rgb, box, device, sam2_transforms)
    out_path = args.out or (args.image.parent / f"{args.image.stem}_medsam2_pred.png")
    cv2.imwrite(str(out_path), (seg * 255).astype(np.uint8))
    print(f"Wrote {out_path} (fg pixels={int(seg.sum())})")


if __name__ == "__main__":
    main()
