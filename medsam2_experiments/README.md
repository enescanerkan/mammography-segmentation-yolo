# MedSAM 2 experiments (mammography)

Self-contained folder to try **MedSAM 2** on mammography PNGs with **multi-class PNG masks** (integer pixel labels), before committing to a full fine-tune.

This repo’s YOLO path (`seg-dataset/`) uses **polygon `.txt` labels**, not PNG masks. MedSAM2 scripts need **raster masks** (`masks/*.png` with integer class pixels). Use the exporter:

```bash
cd medsam2_experiments
python export_seg_dataset_to_raw_mammo.py
# optional: --seg-root D:\path\to\seg-dataset --out data\raw_mammo
```

That reads `seg-dataset/images/{train,val,test}/*.png` + matching `labels/{split}/*.txt`, draws polygons, and writes `data/raw_mammo/images/` + `masks/` with pixel values **0=bg, 1=pectoral (YOLO 0), 2=breast-tissue (YOLO 1), 3=nipple (YOLO 2)** — aligned with `config.py` → then run `zero_shot_test.py`.

Alternatively, supply your own PNG masks and set `MASK_CLASS_LABELS` accordingly.

## Why MedSAM 2 here

- **Zero-shot first**: pre-trained MedSAM 2 + a **box prompt** (here: tight box from GT) shows whether fine-tuning is worth it.  
- **Medical pre-training**: broad medical image statistics; breast / soft-tissue structures are often reasonable zero-shot.  
- **Fine-tune later**: after zero-shot, use ~hundreds of **(image, box, mask)** instances from your cases.

## Layout

| Path | Role |
|------|------|
| `config.py` | Paths, `MASK_CLASS_LABELS` (matches exporter: 1/2/3 = pectoral / breast / nipple), thresholds. |
| `export_seg_dataset_to_raw_mammo.py` | **YOLO `seg-dataset` → `data/raw_mammo`** (PNG + raster mask from polygons). |
| `zero_shot_test.py` | GT-derived bbox → MedSAM2 mask → **Dice** per class (run **before** fine-tune). |
| `prepare_medsam2_data.py` | PNG → `.npy` for `finetune_sam2_img.py` (**img 1024²×3**, **gt 256²** binary per sample). |
| `inference_finetuned.py` | Load `medsam_model_best.pth` + SAM2 base, run one box on one image. |
| `wrapped_model.py` | `MedSAM2` wrapper aligned with upstream `finetune_sam2_img.py` / `infer_medsam2_flare22.py`. |
| `data/raw_mammo/` | Put `images/*.png` and `masks/*.png` here (gitignored). |

**Important:** `bowang-lab/MedSAM` `finetune_sam2_img.py` expects each `gts/*.npy` to be **256×256** and each `imgs/*.npy` to be **1024×1024×3** (uint8). `prepare_medsam2_data.py` follows that.

## 1) Install MedSAM 2 (separate clone)

```bash
conda create -n medsam2 python=3.10 -y
conda activate medsam2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git
cd MedSAM
pip install -e .
```

Point this project at the clone (default: `medsam2_experiments/vendor/MedSAM`):

```bash
set MEDSAM2_REPO=C:\path\to\MedSAM
```

## 2) Checkpoints

Download into `medsam2_experiments/checkpoints/` (gitignored):

- [MedSAM2_latest.pt](https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt)  
- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)

Fine-tune uses **MedSAM2_latest.pt** as `-pretrain_model_path`. Inference after fine-tune loads **SAM2 base** + your **`medsam_model_best.pth`** (see `inference_finetuned.py`).

## 2.5) From your YOLO `seg-dataset` (export masks)

If you only have `seg-dataset/images/` + `seg-dataset/labels/` (polygon `.txt`), run once:

```bash
cd medsam2_experiments
python export_seg_dataset_to_raw_mammo.py
```

This fills `data/raw_mammo/images/` and `data/raw_mammo/masks/` so `zero_shot_test.py` can run.

## 3) Zero-shot (Step 1)

From repo root, with `medsam2` env and `MEDSAM2_REPO` set:

```bash
cd medsam2_experiments
python zero_shot_test.py
# optional:
python zero_shot_test.py --data-root D:\mammo\raw_mammo --ckpt checkpoints\MedSAM2_latest.pt
```

Interpret mean Dice per class; if e.g. breast-tissue is already very high, you can limit fine-tune to harder classes.

## 4) Prepare `.npy` for fine-tune (Step 2)

```bash
cd medsam2_experiments
python prepare_medsam2_data.py --use-project-splits
# writes data/medsam2_npy/{train,val,test}/imgs|gts + *.txt lists
```

If `../seg-dataset/images/{train,val,test}/` is missing, everything goes to `--default-split train`.

## 5) Fine-tune (Step 3 — run inside MedSAM clone)

```bash
cd %MEDSAM2_REPO%
python finetune_sam2_img.py ^
  -i C:\path\to\mammography-segmentation-yolo\medsam2_experiments\data\medsam2_npy\train ^
  -task_name MammoBNP_v1 ^
  -work_dir C:\path\to\medsam2_experiments\work_dir ^
  -batch_size 8 ^
  -num_epochs 100 ^
  -lr 1e-4 ^
  -pretrain_model_path C:\path\to\medsam2_experiments\checkpoints\MedSAM2_latest.pt ^
  -model_cfg sam2_hiera_t.yaml
```

Best weights: `work_dir/MammoBNP_v1-*/medsam_model_best.pth` (upstream naming).

## 6) Inference with fine-tuned weights (Step 4)

```bash
cd medsam2_experiments
python inference_finetuned.py --image path\to\case.png --gt-mask path\to\mask.png --cls 1 ^
  --medsam2-ckpt path\to\medsam_model_best.pth
```

## 7) Active labeling (Step 5)

Use the **3D Slicer MedSAM2 extension** from the MedSAM2 distribution: box prompts per structure, quick brush fixes, export to your training pool. Rough workflow is described upstream; point the plugin at your fine-tuned checkpoint when available.

## Checklist

- [ ] MedSAM2 repo installed (`pip install -e .`) and `MEDSAM2_REPO` set  
- [ ] Both checkpoints in `medsam2_experiments/checkpoints/`  
- [ ] `data/raw_mammo/images` + `masks` with consistent stems and `MASK_CLASS_LABELS`  
- [ ] `python zero_shot_test.py` — review Dice  
- [ ] `python prepare_medsam2_data.py --use-project-splits`  
- [ ] `finetune_sam2_img.py` on `.../medsam2_npy/train`  
- [ ] Compare val/test Dice zero-shot vs fine-tuned  
- [ ] (Optional) Slicer plugin + iterative retrain  

## Expectations (rough)

- **Breast tissue**: often strong already with a good box.  
- **Pectoral (MLO)**: moderate; geometry helps.  
- **Nipple**: small FOV; zero-shot may be weakest — fine-tune often helps most here.

Adjust `MIN_FG_PIXELS` and `BBOX_PAD_PX` in `config.py` if you skip tiny regions or need tighter boxes.
