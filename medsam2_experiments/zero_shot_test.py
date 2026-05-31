"""
Zero-shot style segmentation check using Meta **SAM2** (loaded via bowang-lab **MedSAM** repo code: `build_sam2`, `SAM2ImagePredictor`).

**Weights (important)**  
- Default checkpoint is **`sam2_hiera_tiny.pt`** = **SAM2 Hiera Tiny base** (Meta), *not* the separate Hugging Face `MedSAM2_latest.pt` medical bundle unless you pass a compatible `--ckpt`.  
- So we say **SAM2** for the model family; **MedSAM** here is only the **code/vendor** path (Hydra configs + Python), not automatically “MedSAM2 medical weights”.

**Where the box (prompt) comes from — “oracle” box**  
For each class we load your **GT mask** from the PNG (`gt_mask == class_id`). We take all `True` pixels and compute the **axis-aligned bounding box** (min/max x,y), then add a small padding (`BBOX_PAD_PX`).  
That rectangle is passed to SAM as the **only geometric prompt** (no text). So the model is *not* told “breast” in words; it only sees **image + box coordinates**.

**Is that a problem?**  
It is a **deliberate evaluation setup**: “If a user (or detector) drew a box this good around the organ, how good is the mask?” It **does not** test finding the organ without hints. For deployment you would replace the box with a **human box** or **automatic detector**. Dice then measures mask quality given that hint.

**MLO / CC labels on figures**  
If `data/raw_mammo/view_map.csv` exists (`stem,view` with `CC` or `MLO`), run `build_view_map.py` from `compare-dataset` paths — overlays show `view=...`. Augmented copies named `{stem}_flipped` inherit the same view as `{stem}`. Otherwise `view=?`.

**Where MLO + full masks come from**  
Full **pectoral / breast / nipple** polygons live in **`seg-dataset`** → `export_seg_dataset_to_raw_mammo.py`. `compare-dataset/CC|/MLO/labels/*.txt` are often **pose-style** (small pectoral triangle only), not a substitute for full YOLO-seg labels — do not use them for MedSAM GT masks.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
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


def _norm_bgr(gray: np.ndarray) -> np.ndarray:
    n = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)


def _blend_mask(base_bgr: np.ndarray, mask_bool: np.ndarray, bgr: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.astype(np.float32)
    m = mask_bool.astype(np.float32)[..., None]
    col = np.array(bgr, dtype=np.float32).reshape(1, 1, 3)
    blended = out * (1.0 - 0.42 * m) + col * (0.42 * m)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _model_display_name(ckpt: Path) -> str:
    n = ckpt.name.lower()
    if "medsam2" in n or "med_sam2" in n:
        return "MedSAM2-style ckpt (if compatible)"
    if "hiera_t" in n or "tiny" in n:
        return "SAM2 Hiera Tiny (Meta base)"
    return ckpt.name


def load_view_map(csv_path: Path) -> dict[str, str]:
    if not csv_path.is_file():
        return {}
    out: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = (row.get("stem") or "").strip()
            view = (row.get("view") or "").strip().upper()
            if stem and view:
                out[stem] = view
    return out


def resolve_view(case_id: str, stem_to_view: dict[str, str]) -> str:
    """Map `case_id` to CC/MLO using view_map; horizontal-flip aug copies inherit base stem."""
    v = stem_to_view.get(case_id)
    if v:
        return v
    if case_id.endswith("_flipped"):
        return stem_to_view.get(case_id[: -len("_flipped")], "?")
    return "?"


def _caption_strip(width: int, text: str, font_scale: float = 0.42) -> np.ndarray:
    h = 34
    strip = np.full((h, width, 3), 235, dtype=np.uint8)
    cv2.putText(
        strip,
        text[:118],
        (6, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (18, 18, 18),
        1,
        cv2.LINE_AA,
    )
    return strip


def render_labeled_triptych(
    img_gray: np.ndarray,
    gt_bool: np.ndarray,
    pred_bool: np.ndarray,
    *,
    case_id: str,
    cls_name: str,
    dice: float,
    view: str,
    model_name: str,
    bbox_xyxy: np.ndarray,
) -> np.ndarray:
    """Three columns: Original | GT | Pred; captions + top meta bar (bbox, view, model)."""
    base = _norm_bgr(img_gray)
    p_gt = _blend_mask(base, gt_bool, (0, 255, 0))
    p_pr = _blend_mask(base, pred_bool, (255, 0, 255))
    panels = [
        ("Original (norm)", base),
        ("GT mask (PNG label)", p_gt),
        (f"Prediction ({model_name})", p_pr),
    ]
    capped = []
    for cap, panel in panels:
        capped.append(np.vstack([_caption_strip(panel.shape[1], cap), panel]))
    trip = np.hstack(capped)
    bx = ",".join(str(int(round(x))) for x in bbox_xyxy.tolist())
    line1 = f"{case_id}  |  view={view}  |  class={cls_name}  |  Dice={dice:.3f}"
    line2 = f"BBox prompt from GT (oracle): [{bx}]  |  code=MedSAM repo  |  weights={model_name}"
    meta = np.vstack(
        [
            _caption_strip(trip.shape[1], line1, font_scale=0.38),
            _caption_strip(trip.shape[1], line2, font_scale=0.34),
        ]
    )
    return np.vstack([meta, trip])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None, help="Override RAW_MAMMO_ROOT")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Weights for build_sam2 (default: ZERO_SHOT_CKPT in config, usually sam2_hiera_tiny.pt)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max number of images to process (0 = all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write metrics.csv + summary.txt here (default: medsam2_experiments/data/zero_shot_output)",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save per (case, class) triptych PNGs under <out-dir>/pred_vis/",
    )
    parser.add_argument(
        "--view-map",
        type=Path,
        default=None,
        help="CSV with columns stem,view (CC|MLO). Default: <data-root>/view_map.csv",
    )
    args = parser.parse_args()

    from config import (
        BBOX_PAD_PX,
        EXP_ROOT,
        MASK_CLASS_LABELS,
        MEDSAM2_REPO,
        MIN_FG_PIXELS,
        RAW_MAMMO_ROOT,
        SAM2_MODEL_CFG,
        ZERO_SHOT_CKPT,
    )

    data_root = args.data_root or RAW_MAMMO_ROOT
    view_map_path = args.view_map or (data_root / "view_map.csv")
    view_by_stem = load_view_map(view_map_path)
    if not view_map_path.is_file():
        print(
            f"[zero_shot] view_map missing: {view_map_path}; view=? on figures/CC+MLO labels. "
            "Run: python build_view_map.py (see README)."
        )
    elif not view_by_stem:
        print(f"[zero_shot] view_map empty (no rows): {view_map_path}")
    else:
        print(f"[zero_shot] Loaded view_map: {view_map_path} ({len(view_by_stem)} stems)")

    ckpt = args.ckpt or ZERO_SHOT_CKPT
    model_label = _model_display_name(ckpt)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or (EXP_ROOT / "data" / "zero_shot_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "pred_vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    img_dir = data_root / "images"
    mask_dir = data_root / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise SystemExit(f"Expected {img_dir} and {mask_dir}. See medsam2_experiments/README.md.")

    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint missing: {ckpt}. Download sam2_hiera_tiny.pt into medsam2_experiments/checkpoints/ (see README).")

    print(f"[zero_shot] MEDSAM2_REPO={MEDSAM2_REPO}")
    print(f"[zero_shot] ckpt={ckpt}  ({model_label})  device={device}")

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
    dice_by_cls_view: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    per_row: list[dict[str, object]] = []

    img_paths = sorted(img_dir.glob("*.png"))
    if args.limit > 0:
        img_paths = img_paths[: args.limit]

    for img_path in img_paths:
        case_id = img_path.stem
        view = resolve_view(case_id, view_by_stem)
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
            sc = float(scores[0])
            results[int(cls_id)].append((case_id, d))
            dice_by_cls_view[(int(cls_id), view)].append(d)
            bbox_str = ",".join(str(int(round(x))) for x in bbox.tolist())
            per_row.append(
                {
                    "case_id": case_id,
                    "view": view,
                    "class_id": int(cls_id),
                    "class_name": cls_name,
                    "dice": round(d, 6),
                    "sam_score": round(sc, 6),
                    "bbox_xyxy": bbox_str,
                    "model_label": model_label,
                    "ckpt": str(ckpt.resolve()),
                }
            )
            print(f"{case_id} [{view}] {cls_name}: Dice={d:.3f} (score={sc:.3f})")
            if args.save_vis:
                trip = render_labeled_triptych(
                    img,
                    gt_binary,
                    pred,
                    case_id=case_id,
                    cls_name=cls_name,
                    dice=d,
                    view=view,
                    model_name=model_label,
                    bbox_xyxy=bbox,
                )
                cv2.imwrite(str(vis_dir / f"{case_id}_cls{int(cls_id)}_{cls_name}.png"), trip)

    print("\n=== Zero-shot summary ===")
    summary_lines = []
    for cls_id, cls_name in MASK_CLASS_LABELS.items():
        rows = results.get(int(cls_id), [])
        if not rows:
            continue
        dices = [d for _, d in rows]
        mean_d = float(np.mean(dices))
        line = f"{cls_name:20s}: mean Dice = {mean_d:.3f}  (n={len(dices)})"
        print(line)
        summary_lines.append(f"class_id={cls_id}\tclass_name={cls_name}\tmean_dice={mean_d:.6f}\tn={len(dices)}\n")

    print("\n=== By view (CC / MLO / ?) — mean Dice ===")
    summary_lines.append("\n# per view (same eval rows as above)\n")
    for cls_id, cls_name in MASK_CLASS_LABELS.items():
        for view_tag in ("CC", "MLO", "?"):
            dices = dice_by_cls_view.get((int(cls_id), view_tag), [])
            if not dices:
                continue
            mean_v = float(np.mean(dices))
            line = f"{cls_name:16s}  {view_tag:3s}: mean Dice = {mean_v:.3f}  (n={len(dices)})"
            print(line)
            summary_lines.append(
                f"class_id={cls_id}\tclass_name={cls_name}\tview={view_tag}\tmean_dice={mean_v:.6f}\tn={len(dices)}\n"
            )

    metrics_csv = out_dir / "metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "view",
                "class_id",
                "class_name",
                "dice",
                "sam_score",
                "bbox_xyxy",
                "model_label",
                "ckpt",
            ],
        )
        w.writeheader()
        w.writerows(per_row)

    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"checkpoint\t{ckpt.resolve()}\n")
        f.write(f"data_root\t{data_root.resolve()}\n")
        f.write(f"n_images_processed\t{len(img_paths)}\n")
        f.write(f"n_eval_rows\t{len(per_row)}\n\n")
        f.writelines(summary_lines)

    print(f"\nWrote: {metrics_csv.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    if args.save_vis:
        n_vis = len(list(vis_dir.glob("*.png")))
        print(
            f"Wrote: {vis_dir.resolve()}  ({n_vis} PNGs: Original | GT | Pred + captions; bbox=oracle from GT)"
        )


if __name__ == "__main__":
    main()
