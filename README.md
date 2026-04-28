# Mammography segmentation & clinical metrics (YOLO)

This project trains a **YOLO segmentation** model for mammography (pectoral muscle, breast tissue, nipple) and runs a **comparison pipeline** that scores **ground truth**, **rule-based YOLO pose**, and **segmentation-derived** geometry. It computes **Posterior Nipple Line (PNL)** on **MLO** views and **chest-wall depth** on **CC** views, then evaluates the **10 mm clinical rule** (|PNL − Depth|).

## Example visualization (DICOM grid)

The figure below shows one study *(example: `0f8c73f15309c38d4867102295eb068c`, right breast)* — **MLO** top row, **CC** bottom row: **GT**, **rule-based pose**, and **segmentation** with PNL / depth overlays and whether the sample passes the **10 mm** rule.

![DICOM comparison: ground truth vs rule-based pose vs segmentation](docs/images/dicom_grid_0f8c73f_R.png)

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
├── .gitignore
├── README.md
├── docs/images/              # README assets (versioned)
│
├── pipeline/
│   ├── orchestrator.py       # Compare pipeline coordinator
│   ├── interfaces.py         # Protocols (DIP)
│   ├── models.py             # Models, downloader, factory
│   ├── dataset.py            # CSV + MLO/CC test pairing
│   ├── evaluator.py          # Inference engine + metrics
│   ├── geometry.py           # DICOM ↔ model space, PNL / depth
│   ├── dicom_utils.py       # DICOM load, LUT, BGR
│   └── visualizer.py         # DICOM grid + mask overlay
│
├── breast_seg/               # Segmentation library
│   ├── analyzer.py           # Mask → landmarks (MLO)
│   ├── config.py             # Paths & hyperparameters
│   ├── geometry.py
│   ├── model.py              # YOLO seg wrapper (train/infer)
│   └── visualizer.py
│
├── compare-dataset/          # Usually gitignored: private test data
├── pose_weights/             # Usually gitignored: MLO/CC pose weights
├── runs/                     # Usually gitignored: training runs (best.pt)
└── compare_output/           # Usually gitignored: metrics + viz_dicom PNGs
```

## Setup

```bash
pip install ultralytics opencv-python pandas pydicom scikit-learn seaborn matplotlib gdown tqdm
```

Download **pose** weights into `pose_weights/` (see links below). Train or copy **segmentation** weights under `runs/breast_seg_yolo26m/weights/best.pt` for the compare pipeline.

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
```

### Train segmentation

```bash
python start.py train
# or
python run_train.py
```

### Build YOLO-seg dataset layout

After placing source images and labels under `seg-dataset/` as expected by `build_dataset.py`:

```bash
python build_dataset.py
```

## Clinical metrics

| Term | Meaning |
|------|---------|
| **PNL** | On MLO: perpendicular distance (mm) from nipple to the pectoral line. |
| **Depth** | On CC: distance (mm) from nipple to the medial chest wall. |
| **10 mm rule** | Clinical pass if \|PNL − Depth\| ≤ 10 mm (**Good**), else **Bad**. |

## Outputs (`compare_output/` by default)

- `metrics_clinical.csv` — per-case PNL, depth, errors, landmark metrics  
- `classification_metrics.csv` — accuracy / precision / recall / F1 vs qualitative labels  
- `confusion_matrix_pose.png`, `confusion_matrix_seg.png`  
- `viz_dicom/*.png` — multi-panel GT / pose / segmentation comparisons  

## Pose model weights

Place under `pose_weights/`:

- **MLO**: [Google Drive folder](https://drive.google.com/drive/folders/1V9j-Hm4j64lh2doTpoj4u07-F-vKNrUJ)  
- **CC**: [Google Drive folder](https://drive.google.com/drive/folders/11p_uYnbdJmnIjHbNgKMgkEe7mtsdyVQE)  

If `gdown` is installed, missing files may be fetched automatically; otherwise download manually and copy the `.pt` files into `pose_weights/`.
