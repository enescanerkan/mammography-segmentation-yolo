"""
YOLO Segmentation Dataset Setup Script
- Splits images/ into images/train/ and images/val/ (80/20 ratio)
- Splits labels/train/ into labels/train/ and labels/val/
- Moves images belonging to the test/ folder into images/test/
- Generates/Updates the data.yaml configuration file
"""

import os
import shutil
import random
import glob

BASE_DIR = r"C:\Users\Monster\Desktop\segment_breast"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
TEST_DIR   = os.path.join(BASE_DIR, "test")

# Target Directories
IMAGES_TRAIN = os.path.join(IMAGES_DIR, "train")
IMAGES_VAL   = os.path.join(IMAGES_DIR, "val")
IMAGES_TEST  = os.path.join(IMAGES_DIR, "test")
LABELS_TRAIN = os.path.join(LABELS_DIR, "train")
LABELS_VAL   = os.path.join(LABELS_DIR, "val")
LABELS_TEST  = os.path.join(LABELS_DIR, "test")

def ensure_dirs():
    """Ensure target data directories exist."""
    for d in [IMAGES_TRAIN, IMAGES_VAL, IMAGES_TEST,
              LABELS_TRAIN, LABELS_VAL, LABELS_TEST]:
        os.makedirs(d, exist_ok=True)

def move_images_to_train_val():
    """Distribute PNGs from images/ root into train and validation sets."""
    all_images = [f for f in os.listdir(IMAGES_DIR)
                  if f.lower().endswith(".png") and
                  os.path.isfile(os.path.join(IMAGES_DIR, f))]

    if not all_images:
        print("[INFO] No root PNG files found in images/. Skipping distribution.")
        return

    # Filter strictly for images containing corresponding label data
    valid = []
    for img in all_images:
        stem = os.path.splitext(img)[0]
        lbl_src = os.path.join(LABELS_TRAIN, stem + ".txt")
        if os.path.isfile(lbl_src):
            valid.append(img)
        else:
            print(f"[WARN] Skipped image with missing label: {img}")

    random.seed(42)
    random.shuffle(valid)
    split_idx = int(len(valid) * 0.8)
    train_imgs = valid[:split_idx]
    val_imgs   = valid[split_idx:]

    print(f"\n[INFO] Total valid images found: {len(valid)}")
    print(f"[INFO] Dataset split -> Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    for img in train_imgs:
        stem = os.path.splitext(img)[0]
        shutil.move(os.path.join(IMAGES_DIR, img),
                    os.path.join(IMAGES_TRAIN, img))
        print(f"  [TRAIN] {img}")
        # Labels are already located in labels/train/ by default.

    for img in val_imgs:
        stem = os.path.splitext(img)[0]
        shutil.move(os.path.join(IMAGES_DIR, img),
                    os.path.join(IMAGES_VAL, img))
        print(f"  [VAL]   {img}")
        
        # Transfer the corresponding label to validation directory
        lbl_src = os.path.join(LABELS_TRAIN, stem + ".txt")
        lbl_dst = os.path.join(LABELS_VAL,   stem + ".txt")
        if os.path.isfile(lbl_src):
            shutil.move(lbl_src, lbl_dst)
            print(f"  [VAL-LBL] {stem}.txt")

def move_test_images():
    """Transfer files from the root test/ directory into images/test/."""
    if not os.path.isdir(TEST_DIR):
        print("\n[INFO] Root test/ directory not found. Skipping test migration.")
        return

    test_imgs = [f for f in os.listdir(TEST_DIR)
                 if f.lower().endswith(".png") and
                 os.path.isfile(os.path.join(TEST_DIR, f))]

    print(f"\n[INFO] Test images identified: {len(test_imgs)}")
    for img in test_imgs:
        shutil.move(os.path.join(TEST_DIR, img),
                    os.path.join(IMAGES_TEST, img))
        print(f"  [TEST] {img}")

def write_data_yaml():
    """Generates the data.yaml mapping required by Ultralytics YOLO framework."""
    yaml_path = os.path.join(BASE_DIR, "data.yaml")
    content = f"""path: {BASE_DIR.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test

nc: 3
names:
  0: Nipple
  1: Breast Tissue
  2: Pectoral Muscle
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[INFO] data.yaml successfully written to: {yaml_path}")

def count_files():
    """Log file counts across the dataset directory structure."""
    print("\n[INFO] === Dataset File Summary ===")
    for name, path in [("images/train", IMAGES_TRAIN),
                        ("images/val",   IMAGES_VAL),
                        ("images/test",  IMAGES_TEST),
                        ("labels/train", LABELS_TRAIN),
                        ("labels/val",   LABELS_VAL),
                        ("labels/test",  LABELS_TEST)]:
        n = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]) if os.path.isdir(path) else 0
        print(f"  {name}: {n} files")

if __name__ == "__main__":
    print("=== Commencing Dataset Structure Setup ===")
    ensure_dirs()
    move_images_to_train_val()
    move_test_images()
    write_data_yaml()
    count_files()
    print("\n[SUCCESS] Dataset setup complete and ready for YOLO training.")
