"""
Build the final YOLO-seg dataset:

    seg-dataset/
        data.yaml
        images/
            train/   <- orijinal + yatay flip augmentation
            val/     <- orijinal + yatay flip augmentation
            test/    <- sadece orijinal
        labels/
            train/
            val/
            test/

Split: 80/10/10 (orijinal görüntü bazında).
Augmentation: train+val içindeki her orijinal için _flipped eşleniği.
  * Annotator tarafından üretilmiş bir _flipped varsa (aynı stem)
    onu koru (poligon sayısı farklı olabilir, ayrı manuel anotasyon).
  * Yoksa PIL.ImageOps.mirror + etiket x=1-x ile üret.

Sınıflar (seg-dataset/labels/data.yaml ile uyumlu):
    0: pectoral
    1: breast-tissue
    2: nipple
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
SEG = ROOT / "seg-dataset"

SRC_IMAGES_ROOT = SEG / "images"
SRC_LABELS_TRAIN = SEG / "labels" / "labels" / "train"

DST_IMG = {split: SEG / "images" / split for split in ("train", "val", "test")}
DST_LBL = {split: SEG / "labels" / split for split in ("train", "val", "test")}

CLASS_NAMES = {0: "pectoral", 1: "breast-tissue", 2: "nipple"}
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def ensure_dirs() -> None:
    for d in list(DST_IMG.values()) + list(DST_LBL.values()):
        d.mkdir(parents=True, exist_ok=True)


def collect_originals() -> tuple[list[str], dict[str, bool]]:
    """Return list of original stems and whether a pre-existing flipped label exists."""
    stems: list[str] = []
    has_flipped: dict[str, bool] = {}
    for lbl in sorted(SRC_LABELS_TRAIN.glob("*.txt")):
        if lbl.stem.endswith("_flipped"):
            continue
        stems.append(lbl.stem)
        has_flipped[lbl.stem] = (SRC_LABELS_TRAIN / f"{lbl.stem}_flipped.txt").exists()
    return stems, has_flipped


def split_stems(stems: list[str]) -> dict[str, list[str]]:
    rng = random.Random(SEED)
    shuffled = stems[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * SPLITS["train"]))
    n_val = int(round(n * SPLITS["val"]))
    # Remaining => test (to guarantee all are used).
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def flip_label_file(src: Path, dst: Path) -> None:
    """Write horizontally flipped YOLO-seg label file (x -> 1 - x)."""
    out_lines: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        cls = parts[0]
        coords = [float(x) for x in parts[1:]]
        for i in range(0, len(coords), 2):
            coords[i] = 1.0 - coords[i]
        out_lines.append(" ".join([cls] + [f"{c:.6f}" for c in coords]))
    dst.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def move_original(stem: str, split: str) -> None:
    src_img = SRC_IMAGES_ROOT / f"{stem}.png"
    src_lbl = SRC_LABELS_TRAIN / f"{stem}.txt"
    shutil.move(str(src_img), DST_IMG[split] / src_img.name)
    shutil.move(str(src_lbl), DST_LBL[split] / src_lbl.name)


def emit_flipped(stem: str, split: str, has_existing: bool) -> None:
    """Place the flipped pair in the target split dirs."""
    dst_img = DST_IMG[split] / f"{stem}_flipped.png"
    dst_lbl = DST_LBL[split] / f"{stem}_flipped.txt"

    if has_existing:
        src_img = SRC_IMAGES_ROOT / f"{stem}_flipped.png"
        src_lbl = SRC_LABELS_TRAIN / f"{stem}_flipped.txt"
        shutil.move(str(src_img), dst_img)
        shutil.move(str(src_lbl), dst_lbl)
        return

    orig_img = DST_IMG[split] / f"{stem}.png"
    orig_lbl = DST_LBL[split] / f"{stem}.txt"
    with Image.open(orig_img) as im:
        ImageOps.mirror(im).save(dst_img)
    flip_label_file(orig_lbl, dst_lbl)


def delete_flipped_in_test_if_any(stem: str) -> None:
    """Test split'e giden orijinalin flipped'ini dataset'e dahil etme."""
    for p in [
        SRC_IMAGES_ROOT / f"{stem}_flipped.png",
        SRC_LABELS_TRAIN / f"{stem}_flipped.txt",
    ]:
        if p.exists():
            p.unlink()


def write_data_yaml() -> None:
    content = [
        f"path: {SEG.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for i, n in CLASS_NAMES.items():
        content.append(f"  {i}: {n}")
    (SEG / "data.yaml").write_text("\n".join(content) + "\n", encoding="utf-8")


def cleanup_legacy() -> None:
    legacy = [
        SEG / "labels" / "labels",
        SEG / "labels" / "eksik",
    ]
    for p in legacy:
        if p.exists():
            shutil.rmtree(p)
            print(f"  [RM] {p.relative_to(ROOT)}")
    for f in [SEG / "labels" / "data.yaml", SEG / "labels" / "train.txt"]:
        if f.exists():
            f.unlink()
            print(f"  [RM] {f.relative_to(ROOT)}")


def purge_flat_images() -> None:
    """Kökteki seg-dataset/images/*.png dosyalarından artakalan varsa sil."""
    for p in SRC_IMAGES_ROOT.glob("*.png"):
        if p.is_file():
            p.unlink()


def summary() -> None:
    print("\n=== ÖZET ===")
    for split in ("train", "val", "test"):
        n_img = len(list(DST_IMG[split].glob("*.png")))
        n_lbl = len(list(DST_LBL[split].glob("*.txt")))
        print(f"  {split:5} | images: {n_img:3d}  labels: {n_lbl:3d}")


def main() -> None:
    print("=== Dataset Build Başlıyor ===\n")
    assert SRC_IMAGES_ROOT.exists() and SRC_LABELS_TRAIN.exists(), "Kaynak dizinler yok!"

    ensure_dirs()

    originals, has_flipped = collect_originals()
    splits = split_stems(originals)
    for k, v in splits.items():
        print(f"[INFO] {k}: {len(v)} orijinal")

    print("\n[1/4] Orijinalleri split klasörlerine taşı")
    for split, stems in splits.items():
        for stem in stems:
            move_original(stem, split)

    print("\n[2/4] Test split: flipped'leri dahil etme")
    for stem in splits["test"]:
        delete_flipped_in_test_if_any(stem)

    print("\n[3/4] Train & Val için yatay flip augmentation")
    created_existing = 0
    created_generated = 0
    for split in ("train", "val"):
        for stem in splits[split]:
            existing = has_flipped.get(stem, False) and (SRC_IMAGES_ROOT / f"{stem}_flipped.png").exists()
            emit_flipped(stem, split, existing)
            if existing:
                created_existing += 1
            else:
                created_generated += 1
    print(f"  Annotator-flipped taşındı: {created_existing}")
    print(f"  PIL ile üretildi         : {created_generated}")

    print("\n[4/4] Eski yapıyı temizle")
    purge_flat_images()
    cleanup_legacy()

    write_data_yaml()
    print(f"\n[INFO] data.yaml yazıldı: {(SEG / 'data.yaml').relative_to(ROOT)}")

    summary()
    print("\n[OK] Dataset hazır.")


if __name__ == "__main__":
    main()
