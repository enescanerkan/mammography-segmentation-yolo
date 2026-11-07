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

- **Zero-shot first**: **SAM2** image weights + a **box prompt** (in our script: a tight **axis-aligned box from the GT mask**, padded slightly — an *oracle* box for evaluation) shows whether fine-tuning is worth it. This does **not** test “find the organ without a hint”; it tests “given a good box, how good is the mask?” For deployment, replace the box with a **human** or **detector** box.  
- **Code path**: we import `sam2` from the bowang-lab **MedSAM** repository (MedSAM2 branch) for Hydra configs and `build_sam2`. Default checkpoint is **`sam2_hiera_tiny.pt`** (Meta **SAM2**), not necessarily Hugging Face `MedSAM2_latest.pt` unless you pass a compatible `--ckpt`.  
- **Fine-tune later**: after zero-shot, use ~hundreds of **(image, box, mask)** instances from your cases.

## Layout

| Path | Role |
|------|------|
| `config.py` | Paths, `MASK_CLASS_LABELS` (matches exporter: 1/2/3 = pectoral / breast / nipple), thresholds. |
| `export_seg_dataset_to_raw_mammo.py` | **YOLO `seg-dataset` → `data/raw_mammo`** (PNG + raster mask from polygons). |
| `zero_shot_test.py` | **SAM2** (`build_sam2` + `SAM2ImagePredictor` from bowang-lab **MedSAM** repo): GT-derived **oracle bbox** → mask → **Dice** per class. |
| `build_view_map.py` | Writes `data/raw_mammo/view_map.csv` (`stem,view` = CC or MLO) from `compare-dataset/` so zero-shot figures and `metrics.csv` can show **view**. |
| `prepare_medsam2_data.py` | PNG → `.npy` for `finetune_sam2_img.py` (**img 1024²×3**, **gt 256²** binary per sample). |
| `inference_finetuned.py` | Load `medsam_model_best.pth` + SAM2 base, run one box on one image. |
| `wrapped_model.py` | `MedSAM2` wrapper aligned with upstream `finetune_sam2_img.py` / `infer_medsam2_flare22.py`. |
| `data/raw_mammo/` | Put `images/*.png` and `masks/*.png` here (gitignored). |

**Important:** `bowang-lab/MedSAM` `finetune_sam2_img.py` expects each `gts/*.npy` to be **256×256** and each `imgs/*.npy` to be **1024×1024×3** (uint8). `prepare_medsam2_data.py` follows that.

## 1) Install MedSAM 2 (separate clone)

**Important Python & Conda Requirement for Windows:**
You **must** use Python 3.10 and a Conda environment. Windows Store Python (e.g. 3.13 or 3.14) lacks pre-compiled PyTorch CUDA support, and compiling MedSAM's `setup.py` C++ extensions (`sam2._C`) will fail without the NVCC compiler.

If you don't have Conda, you can install Miniconda silently via PowerShell:
```powershell
cd $env:USERPROFILE\Desktop
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda_installer.exe"
Start-Process -FilePath ".\miniconda_installer.exe" -ArgumentList "/S /RegisterPython=0 /D=$env:USERPROFILE\Miniconda3" -Wait
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" init powershell
```
**CRITICAL:** After running the init command, you *must* completely close and reopen your VS Code / PowerShell windows for `conda` to be recognized!

Once your terminal is restarted and you see `(base)` at the start of your prompt:
```bash
conda create -n medsam2 python=3.10 -y
conda activate medsam2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Assuming you cloned or will clone the repo:
# git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git
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

Fine-tune (per upstream README) uses **`sam2_hiera_tiny.pt`** as `-pretrain_model_path` for the image script. The Hugging Face `MedSAM2_latest.pt` file may **not** load into `sam2_hiera_t.yaml` on every checkout; if `build_sam2` fails on that file, keep using **tiny** as the backbone. Inference after your own fine-tune still uses SAM2 tiny + `medsam_model_best.pth`.

## 2.5) From your YOLO `seg-dataset` (export masks)

If you only have `seg-dataset/images/` + `seg-dataset/labels/` (polygon `.txt`), run once:

```bash
cd medsam2_experiments
python export_seg_dataset_to_raw_mammo.py
```

This fills `data/raw_mammo/images/` and `data/raw_mammo/masks/` so `zero_shot_test.py` can run.

### CC / MLO labels on outputs (optional)

If you have `compare-dataset/CC/...` and `compare-dataset/MLO/...` in the project (same stems as exported `raw_mammo` images), build a lookup table once:

```bash
cd medsam2_experiments
python build_view_map.py
# writes data/raw_mammo/view_map.csv
```

`zero_shot_test.py` loads `<data-root>/view_map.csv` by default (`--view-map` to override). Rows in `metrics.csv` get a **`view`** column; `--save-vis` PNGs show **view=CC|MLO|?** in the top banner. Files named `{stem}_flipped` reuse the view of `{stem}`.

**Important:** Full **pectoral / breast / nipple** masks must come from **`seg-dataset`** via `export_seg_dataset_to_raw_mammo.py`. The files under `compare-dataset/.../labels/*.txt` are often **not** full multi-class polygons (e.g. small pectoral triangle for pose); they are **only** used by `build_view_map.py` to tag **CC vs MLO** by image stem, not to build GT masks.

To include **more MLO cases** in zero-shot, add MLO images + full YOLO-seg `.txt` into your `seg-dataset` source pool, run `build_dataset.py`, re-export `raw_mammo`, then `build_view_map.py` again.

## 3) Zero-shot (Step 1)

You already need:

1. **MedSAM2 repo** on disk (e.g. `medsam2_experiments/vendor/MedSAM` from `git clone -b MedSAM2 ...`).
2. **`PYTHONPATH`** pointing at that repo root (or `pip install -e .` inside a **conda Python 3.10 + CUDA** env — Store Python 3.13/3.14 often fails the CUDA extension build).
3. **`hydra-core`** for the same interpreter: `python -m pip install hydra-core iopath` (use the same `python` you run the script with).
4. Checkpoints under `medsam2_experiments/checkpoints/` — at minimum **`sam2_hiera_tiny.pt`**.

**Do not** `cd medsam2_experiments` twice: if your prompt is already `...\medsam2_experiments>`, run `python zero_shot_test.py` only.

Default weights for `zero_shot_test.py` are **SAM2 Hiera Tiny** (`ZERO_SHOT_CKPT` in `config.py`), not `MedSAM2_latest.pt`, so `build_sam2` matches `sam2_hiera_t.yaml`. Quick test:

```powershell
cd path\to\mammography-segmentation-yolo\medsam2_experiments
$env:PYTHONPATH = "$PWD\vendor\MedSAM"
$env:MEDSAM2_REPO = "$PWD\vendor\MedSAM"
python zero_shot_test.py --limit 5 --device cpu
```

Optional overrides:

```bash
python zero_shot_test.py --ckpt checkpoints\MedSAM2_latest.pt   # only if compatible with your MedSAM commit
set MEDSAM2_ZERO_SHOT_CKPT=C:\path\to\custom.pt
```

**Outputs** (each run overwrites the same folder by default):

- `medsam2_experiments/data/zero_shot_output/metrics.csv` — one row per (case, class): `view`, Dice, SAM score, **`bbox_xyxy`** (oracle box fed to the model), **`model_label`** (e.g. SAM2 Hiera Tiny), checkpoint path  
- `medsam2_experiments/data/zero_shot_output/summary.txt` — mean Dice per class, **plus mean Dice per class × view (CC / MLO / ?)** when views are known, and run metadata  
- Optional **images**: `--save-vis` → PNGs in `.../zero_shot_output/pred_vis/`  
  Each file: **Original (norm) | GT (green) | Prediction (magenta)** with column captions plus a **top meta bar** (case id, **view**, class, Dice, bbox-from-GT note, weights label).

Override directory: `python zero_shot_test.py --out-dir D:\runs\zs1`

Example with visuals:

```powershell
python zero_shot_test.py --limit 10 --save-vis --device cpu
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

If `finetune_sam2_img.py` fails on imports, install **once** in the same `(medsam2)` env (SAM2 `sam2/__init__.py` needs Hydra):

```bash
pip install hydra-core iopath scikit-image monai
```

**GPU / CUDA:** `-device cuda:0` only works if PyTorch is the **CUDA** build (`torch.cuda.is_available()` is `True`). A `+cpu` install causes `Torch not compiled with CUDA enabled`. Reinstall in the same env (choose a CUDA wheel that matches your GPU driver; [pytorch.org](https://pytorch.org/get-started/locally/) has the matrix), for example:

```powershell
python -m pip uninstall torch torchvision -y
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then: `python -c "import torch; print(torch.cuda.is_available())"` → `True`. **CPU** (`-device cpu`) is only for tiny sanity runs, not real fine-tune.

```bash
cd %MEDSAM2_REPO%
python finetune_sam2_img.py ^
  -i C:\path\to\mammography-segmentation-yolo\medsam2_experiments\data\medsam2_npy\train ^
  -task_name MammoBNP_v1 ^
  -work_dir C:\path\to\medsam2_experiments\work_dir ^
  -batch_size 8 ^
  -num_epochs 100 ^
  -lr 1e-4 ^
  -pretrain_model_path C:\path\to\medsam2_experiments\checkpoints\sam2_hiera_tiny.pt ^
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
- [ ] `python build_view_map.py` after export (so `view_map.csv` exists — CC/MLO on figures and per-view summary)  
- [ ] `python prepare_medsam2_data.py --use-project-splits`  
- [ ] `finetune_sam2_img.py` on `.../medsam2_npy/train`  
- [ ] Compare val/test Dice zero-shot vs fine-tuned  
- [ ] (Optional) Slicer plugin + iterative retrain  

## Expectations (rough)

- **Breast tissue**: often strong already with a good box.  
- **Pectoral (MLO)**: moderate; geometry helps.  
- **Nipple**: small FOV; zero-shot may be weakest — fine-tune often helps most here.

Adjust `MIN_FG_PIXELS` and `BBOX_PAD_PX` in `config.py` if you skip tiny regions or need tighter boxes.
