"""
Convert PNG mammography images + multi-class PNG masks to MedSAM2 `finetune_sam2_img.py` layout:

  <out>/<split>/imgs/<sample_id>.npy   # uint8 (1024, 1024, 3)
  <out>/<split>/gts/<sample_id>.npy    # uint8 (256, 256) values {0, 1}

One training line per (case, class) where the class region is large enough.

If `--use-project-splits` is set, case stems are taken from `../seg-dataset/images/{train,val,test}/`
(relative to project root). Otherwise every case under `--data-root` goes to `--default-split`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 1024
GT_SIZE = 256


def load_split_stems(project_root: Path) -> dict[str, set[str]] | None:
    seg = project_root / "seg-dataset" / "images"
    if not seg.is_dir():
        return None
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split in out:
        d = seg / split
        if not d.is_dir():
            continue
        for p in d.glob("*.png"):
            out[split].add(p.stem)
    if not any(out.values()):
        return None
    return out


def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m[:, :, 0]
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None, help="images/ + masks/ with matching stems")
    parser.add_argument("--out", type=Path, default=None, help="Output root (train/val/test subdirs)")
    parser.add_argument("--use-project-splits", action="store_true")
    parser.add_argument("--default-split", type=str, default="train", choices=("train", "val", "test"))
    args = parser.parse_args()

    from config import MASK_CLASS_LABELS, MIN_FG_PIXELS, NPY_OUT_ROOT, PROJECT_ROOT, RAW_MAMMO_ROOT

    raw = args.data_root or RAW_MAMMO_ROOT
    out_root = args.out or NPY_OUT_ROOT

    img_dir = raw / "images"
    mask_dir = raw / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Need {img_dir} and {mask_dir}.")

    splits = None
    if args.use_project_splits:
        splits = load_split_stems(PROJECT_ROOT)
        if splits is None:
            print("[WARN] seg-dataset/images not found; falling back to --default-split for all cases.")

    for sp in ("train", "val", "test"):
        (out_root / sp / "imgs").mkdir(parents=True, exist_ok=True)
        (out_root / sp / "gts").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "test": 0}

    for img_path in sorted(img_dir.glob("*.png")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.is_file():
            continue

        if splits and stem in splits["train"]:
            split = "train"
        elif splits and stem in splits["val"]:
            split = "val"
        elif splits and stem in splits["test"]:
            split = "test"
        else:
            split = args.default_split if not splits else "train"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        img_rgb = np.stack([img, img, img], axis=-1)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

        mask_full = read_mask(mask_path)

        for cls_id in MASK_CLASS_LABELS:
            binary = (mask_full == int(cls_id)).astype(np.uint8)
            if int(binary.sum()) < MIN_FG_PIXELS:
                continue
            binary_256 = cv2.resize(binary, (GT_SIZE, GT_SIZE), interpolation=cv2.INTER_NEAREST)
            sample_id = f"{stem}_cls{int(cls_id)}"
            np.save(out_root / split / "imgs" / f"{sample_id}.npy", img_resized.astype(np.uint8))
            np.save(out_root / split / "gts" / f"{sample_id}.npy", binary_256.astype(np.uint8))
            counts[split] += 1

    for sp in ("train", "val", "test"):
        ids = sorted(p.stem for p in (out_root / sp / "gts").glob("*.npy"))
        (out_root / f"{sp}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    print("Samples written per split:", counts)
    print("Output root:", out_root.resolve())
    print(
        "\nFine-tune (run inside MedSAM2 repo, pointing -i at train folder only):\n"
        f"  python finetune_sam2_img.py -i {out_root / 'train'} ...\n"
        "Val/test folders are for your own evaluation scripts."
    )


if __name__ == "__main__":
    main()
