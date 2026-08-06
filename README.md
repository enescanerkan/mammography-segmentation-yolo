# Mammography segmentation & clinical metrics (YOLO)

This project trains a **YOLO segmentation** model for mammography (pectoral muscle, breast tissue, nipple) and runs a **comparison pipeline** that scores **ground truth**, **rule-based YOLO pose**, and **segmentation-derived** geometry. It computes **Posterior Nipple Line (PNL)** on **MLO** views and **chest-wall depth** on **CC** views, then evaluates the **10 mm clinical rule** (|PNL − Depth|).

It also ships a **PyQt5 annotation toolkit** (`medsam2_experiments/qt_app/`) that grew the label set from 169 to 2 924 images using a MedSAM2 model fine-tuned on this data.

## Example visualization (DICOM grid)

The figure below shows one study *(example: `01dd49731329cbe81a9876b583b92f30`, right breast)* — **MLO** top row, **CC** bottom row: **GT**, **rule-based pose**, and **segmentation** with PNL / depth overlays and whether the sample passes the **10 mm** rule.

![DICOM comparison: ground truth vs rule-based pose vs segmentation](docs/images/dicom_grid_01dd4973_R.png)

## Models

| Model | Purpose | Location |
|-------|---------|----------|
| **YOLO26m-seg `v4`** | Segmentation: `pectoral` / `breast-tissue` / `nipple`. Current model, used by the compare pipeline. | `runs/breast_seg_v4_yolo26m/weights/best.pt` |
| **YOLO26-pose MLO / CC** | Rule-based landmark baseline. Wavelet-trained (see below). | `pose_weights/{mlo,cc}-yolo26-pose-advanced.pt` |
| **MedSAM2 fine-tune `v4`** | Box-prompted interactive segmentation *inside the annotation toolkit only* — not part of the compare pipeline. | `medsam2_experiments/work_dir_v4/MammoV4-*/medsam_model_best.pth` |

Weights are gitignored (size). Train them locally or copy them in.

### Dataset & model generations

Each generation is a **full re-export**, not an increment. `breast_seg/config.py` selects one via `dataset_version`; override with `BREAST_SEG_DATASET`.

Mask metrics are the best epoch of each run (the epoch `best.pt` was saved at).

| Gen | Images | Epochs | What changed | Mask mAP50 | mAP50-95 |
|-----|-------:|-------:|--------------|-----------:|---------:|
| v1 (`seg-dataset`) | 169 | 100 | Hand-labelled seed set | — | — |
| v2 | 1 857 | 121 | + MedSAM2-assisted labels | 0.943 | 0.811 |
| v3 | 2 924 | 109 | + reviewed CC pseudo-labels | 0.949 | 0.820 |
| **v4** | 2 924 | 150 | Split driven by the compare CSV so **Test cases are held out** → leak-free compare | **0.967** | **0.822** |

v4 has the same images as v3; the gain comes from the corrected split, and it is the only generation whose compare numbers are trustworthy.

## Compare pipeline results

146 MLO+CC test pairs, seg model = v4, qualitative clinical labels as ground truth:

| Model | Accuracy | F1 (Good) | F1 (Bad) |
|-------|---------:|----------:|---------:|
| Rule-based pose | 0.815 | 0.884 | 0.542 |
| **Segmentation (v4)** | **0.849** | **0.907** | **0.607** |

Landmark error, mm (mean ± sd):

| View | Metric | Rule-based pose | Segmentation |
|------|--------|-----------------|--------------|
| MLO | Nipple | 1.47 ± 0.89 | 1.75 ± 1.16 |
| MLO | Pectoral top | 14.13 ± 10.89 | 37.21 ± 21.13 |
| MLO | Pectoral bottom | 7.05 ± 8.81 | 5.36 ± 6.56 |
| MLO | PNL distance | 2.58 ± 2.53 | **2.31 ± 2.36** |
| CC | Nipple | 1.36 ± 0.86 | 2.11 ± 1.23 |
| CC | Chest-wall distance | **0.94 ± 0.80** | 1.68 ± 1.24 |

> **Wavelet preprocessing matters.** The `*-yolo26-pose-advanced.pt` models were *trained* on NLM-denoised → Daubechies-4 → CLAHE enhanced images. Feeding raw images at inference is out-of-distribution: pose accuracy drops from **0.815 to 0.753**. `pipeline/models.py` now applies the same enhancement automatically; set `POSE_WAVELET=0` to reproduce the raw-input behaviour.

## Architecture (SOLID-oriented)

| Principle | How it is reflected |
|-----------|---------------------|
| **SRP** | `PipelineOrchestrator` wires the run only; `InferenceEngine` runs models; `MammographyDataset` loads CSVs and pairs; `MetricsCalculator` builds tables; `ResultVisualizer` draws outputs. |
| **OCP** | New pose/seg backends can subclass `BaseModel` or swap implementations via `ModelFactory` without rewriting the orchestrator. |
| **DIP** | `pipeline/interfaces.py` documents expected capabilities (`Protocol`s) so higher-level modules depend on behaviors, not concrete classes. |

## Repository layout

```
mammography-segmentation-yolo/
├── start.py                  # CLI entry (compare | train)
├── run_train.py              # Segmentation training wrapper
├── build_dataset.py          # Build seg-dataset train/val/test + flips
├── setup_dataset.py          # Optional dataset prep
├── requirements.txt
├── .gitattributes            # LF normalization (repo started on Windows)
├── .gitignore
├── README.md
├── docs/images/              # README assets (versioned)
│
├── pipeline/                 # Compare pipeline
│   ├── orchestrator.py       # Coordinator; reads seg weights from Config
│   ├── interfaces.py         # Protocols (DIP)
│   ├── models.py             # Models, wavelet preprocessing, factory
│   ├── dataset.py            # CSV + MLO/CC test pairing
│   ├── evaluator.py          # Inference engine + metrics
│   ├── geometry.py           # DICOM ↔ model space, PNL / depth
│   ├── dicom_utils.py        # DICOM load, LUT, BGR
│   └── visualizer.py         # DICOM grid + mask overlay
│
├── breast_seg/               # Segmentation library
│   ├── analyzer.py           # Mask → landmarks (MLO)
│   ├── config.py             # Single source of truth: dataset gen, paths, HPs
│   ├── geometry.py           # Pectoral line fit (tangent rule)
│   ├── model.py              # YOLO seg wrapper (train/infer)
│   └── visualizer.py
│
├── medsam2_experiments/      # MedSAM2 fine-tune + annotation toolkit
│   ├── qt_app/               # PyQt5: SAM Label, Manual, Positioning, PNL overlay
│   ├── interactive/          # Box-prompt MedSAM2 wrapper
│   ├── prepare_medsam2_data.py
│   └── train_medsam2.py
│
├── compare-dataset/          # Gitignored: private test data
├── seg-dataset*/             # Gitignored: v1..v4 YOLO-seg datasets
├── label_pool/               # Gitignored: unlabelled MLO/CC pool for annotation
├── pose_weights/             # Gitignored: MLO/CC pose weights
├── runs/                     # Gitignored: training runs (best.pt)
└── compare_output*/          # Gitignored: metrics + viz_dicom PNGs
```

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` includes `pylibjpeg` + `pylibjpeg-libjpeg` + `pylibjpeg-openjpeg`. Mammography DICOMs are usually **JPEG Lossless (Process 14)** or **JPEG 2000**; without those plugins pydicom fails with *"Unable to decompress … all plugins are missing dependencies"*. `PyQt5` is only needed for the annotation toolkit.

Then place the weights: pose weights under `pose_weights/` (links below), and a trained segmentation checkpoint under `runs/breast_seg_v4_yolo26m/weights/best.pt`.

## Usage

### Compare pipeline (metrics + optional DICOM figures)

```bash
python start.py compare

# Quick run on first N pairs
python start.py compare --limit 5

# Metrics only (skip DICOM visualization)
python start.py compare --no-dicom-viz

# Custom output directory
python start.py compare --out compare_output

# Reproduce the pre-wavelet pose numbers
POSE_WAVELET=0 python start.py compare
```

### Train segmentation

```bash
python start.py train
# or
python run_train.py

# Train against an older generation
BREAST_SEG_DATASET=_v3 python run_train.py
```

Other overrides: `BREAST_SEG_EPOCHS`, `BREAST_SEG_BATCH`, `BREAST_SEG_PATIENCE`, `BREAST_SEG_WORKERS`, `BREAST_SEG_DEVICE`, `BREAST_SEG_AMP`.

### Build YOLO-seg dataset layout

After placing source images and labels under `seg-dataset*/` as expected by `build_dataset.py`:

```bash
python build_dataset.py
```

### Annotation toolkit

```bash
MEDSAM2_INTERACTIVE_CKPT=/path/to/medsam_model_best.pth \
  python medsam2_experiments/medsam2_qt_demo.py
```

Three modes: **SAM Label** (box-prompt MedSAM2), **Manual** (polygon editing), **Positioning** (MLO/CC pose analysis). The canvas draws a live **PNL / CC-depth overlay** with a HUD; `←`/`→` navigate images and `Ctrl+S` saves. See `medsam2_experiments/README.md`.

## Clinical metrics

| Term | Meaning |
|------|---------|
| **PNL** | On MLO: perpendicular distance (mm) from nipple to the pectoral line. |
| **Depth** | On CC: distance (mm) from nipple to the medial chest wall. |
| **10 mm rule** | Clinical pass if \|PNL − Depth\| ≤ 10 mm (**Good**), else **Bad**. |

The pectoral line is fitted with a **tangent rule**: the bottom anchor stays fixed at the true bottom edge of the muscle, and the line is drawn through that point and the outermost pixel of the middle band (the apex of the muscle's convex bulge), so it runs tangent to the bulge. `pec_top` follows from that geometry rather than from shifting the bottom.

## Outputs (`compare_output/` by default)

- `metrics_clinical.csv` — per-case PNL, depth, errors, landmark metrics
- `classification_metrics.csv` — accuracy / precision / recall / F1 vs qualitative labels
- `scientific_summary.{csv,md}` — landmark error tables
- `confusion_matrix_pose.png`, `confusion_matrix_seg.png`
- `viz_dicom/*.png` — multi-panel GT / pose / segmentation comparisons

## Pose model weights

Place under `pose_weights/`:

- **MLO**: [Google Drive folder](https://drive.google.com/drive/folders/1V9j-Hm4j64lh2doTpoj4u07-F-vKNrUJ)
- **CC**: [Google Drive folder](https://drive.google.com/drive/folders/11p_uYnbdJmnIjHbNgKMgkEe7mtsdyVQE)

If `gdown` is installed, missing files may be fetched automatically; otherwise download manually and copy the `.pt` files into `pose_weights/`.
