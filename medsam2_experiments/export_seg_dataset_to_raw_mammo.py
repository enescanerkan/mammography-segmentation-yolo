"""
Export YOLO-seg `seg-dataset/` (PNG images + polygon `.txt` labels) to the layout
expected by `zero_shot_test.py` / `prepare_medsam2_data.py`:

  medsam2_experiments/data/raw_mammo/images/{stem}.png
  medsam2_experiments/data/raw_mammo/masks/{stem}.png   # uint8, 0=bg, 1=YOLO0, 2=YOLO1, 3=YOLO2

YOLO class ids follow `seg-dataset/data.yaml`:
  0 -> mask pixel 1 (pectoral)
  1 -> mask pixel 2 (breast-tissue)
  2 -> mask pixel 3 (nipple)

Run from repo root or from `medsam2_experiments/`:

  python export_seg_dataset_to_raw_mammo.py
  python export_seg_dataset_to_raw_mammo.py --seg-root ../seg-dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_yolo_seg_lines(text: str) -> list[tuple[int, np.ndarray]]:
    """Return list of (class_id, Nx2 int32 polygon in pixel coords)."""
    out: list[tuple[int, np.ndarray]] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        cls = int(parts[0])
        coords = np.array([float(x) for x in parts[1:]], dtype=np.float64)
        if coords.size < 6 or coords.size % 2 != 0:
            continue
        poly = coords.reshape(-1, 2)
        out.append((cls, poly))
    return out


def fill_mask_from_yolo(
    h: int,
    w: int,
    instances: list[tuple[int, np.ndarray]],
    yolo_class_to_pixel: dict[int, int],
) -> np.ndarray:
    """Paint polygons; later classes in sort order overwrite overlaps."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Draw lower YOLO ids first, then higher (nipple on top of breast if overlap)
    instances_sorted = sorted(instances, key=lambda t: t[0])
    for cls, poly_norm in instances_sorted:
        if cls not in yolo_class_to_pixel:
            continue
        pix = int(yolo_class_to_pixel[cls])
        pts = np.empty_like(poly_norm, dtype=np.int32)
        pts[:, 0] = np.clip((poly_norm[:, 0] * w).round().astype(np.int32), 0, w - 1)
        pts[:, 1] = np.clip((poly_norm[:, 1] * h).round().astype(np.int32), 0, h - 1)
        cv2.fillPoly(mask, [pts], pix)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seg-root",
        type=Path,
        default=None,
        help="Path to seg-dataset (default: ../seg-dataset from this script)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output raw_mammo root (default: medsam2_experiments/data/raw_mammo)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names under images/ and labels/",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    project = here.parent
    seg = args.seg_root or (project / "seg-dataset")
    out_root = args.out or (here / "data" / "raw_mammo")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # YOLO 0,1,2 -> PNG mask pixel 1,2,3 (0 = background)
    yolo_to_pixel = {0: 1, 1: 2, 2: 3}

    img_out = out_root / "images"
    msk_out = out_root / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    n_img, n_skip, n_warn = 0, 0, 0
    seen_stems: set[str] = set()

    for split in splits:
        img_dir = seg / "images" / split
        lbl_dir = seg / "labels" / split
        if not img_dir.is_dir():
            print(f"[WARN] Missing {img_dir}, skip split '{split}'")
            continue
        if not lbl_dir.is_dir():
            print(f"[WARN] Missing {lbl_dir}, skip split '{split}'")
            continue

        for img_path in sorted(img_dir.glob("*.png")):
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.is_file():
                n_skip += 1
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                n_skip += 1
                continue
            if img.ndim == 2:
                h, w = img.shape
            else:
                h, w = img.shape[:2]

            text = lbl_path.read_text(encoding="utf-8")
            instances = parse_yolo_seg_lines(text)
            if not instances:
                n_skip += 1
                continue

            mask = fill_mask_from_yolo(h, w, instances, yolo_to_pixel)

            out_stem = stem
            if out_stem in seen_stems:
                out_stem = f"{split}__{stem}"
                n_warn += 1
            seen_stems.add(out_stem)

            # Keep grayscale mammo as single-channel or BGR copy for consistency
            if img.ndim == 2:
                to_save = img
            else:
                to_save = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(str(img_out / f"{out_stem}.png"), to_save)
            cv2.imwrite(str(msk_out / f"{out_stem}.png"), mask)
            n_img += 1

    print(f"Done. Wrote {n_img} image/mask pairs to {out_root.resolve()}")
    print(f"  skipped (no label / empty / unreadable): {n_skip}")
    if n_warn:
        print(f"  renamed duplicates with split prefix: {n_warn}")
    print("Next: cd medsam2_experiments && python zero_shot_test.py")


if __name__ == "__main__":
    main()
