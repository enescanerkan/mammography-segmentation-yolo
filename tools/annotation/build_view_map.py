"""
Build data/raw_mammo/view_map.csv mapping image stem -> CC or MLO.

Scans project compare-dataset:
  compare-dataset/CC/images/{train,val,test}/*.png  -> CC
  compare-dataset/MLO/images/{train,val,test}/*.png -> MLO

If the same stem appears in both (unusual), the last scan wins — re-run after cleaning duplicates.

Run from repo root:
  python medsam2_experiments/build_view_map.py

Or:
  python build_view_map.py   # from medsam2_experiments/
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--compare-root",
        type=Path,
        default=None,
        help="compare-dataset root (default: ../compare-dataset from this script)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: medsam2_experiments/data/raw_mammo/view_map.csv)",
    )
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    project = here.parent
    compare = args.compare_root or (project / "compare-dataset")
    out = args.out or (here / "data" / "raw_mammo" / "view_map.csv")

    stem_to_view: dict[str, str] = {}
    splits = ("train", "val", "test")
    for view in ("CC", "MLO"):
        for sp in splits:
            d = compare / view / "images" / sp
            if not d.is_dir():
                continue
            for img in d.glob("*.png"):
                stem_to_view[img.stem] = view

    if not stem_to_view:
        raise SystemExit(
            f"No images found under {compare / 'CC' / 'images'} or {compare / 'MLO' / 'images'}. "
            "compare-dataset may be missing or paths differ. Create view_map.csv manually: stem,view"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "view"])
        for stem in sorted(stem_to_view):
            w.writerow([stem, stem_to_view[stem]])

    print(f"Wrote {len(stem_to_view)} rows to {out.resolve()}")


if __name__ == "__main__":
    main()
