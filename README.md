<p align="center"><img src="assets/logo.png" width="96" alt="Mammography Positioning Review"></p>

<h1 align="center">Mammography Segmentation &amp; Positioning</h1>

<p align="center">
  Breast tissue, pectoral muscle and nipple segmentation · MLO / CC landmark models · positioning assessment with the
  posterior nipple line (PNL) rule · a review web application · a configurable training framework · an annotation
  toolkit with MedSAM2
</p>

<p align="center">
  <a href="https://github.com/enescanerkan/mammography-segmentation-yolo/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/enescanerkan/mammography-segmentation-yolo/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-14213D">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-0E7C7B">
  <img alt="Docker" src="https://img.shields.io/badge/docker-CPU-14213D">
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/positioning_review_dark.png">
  <img src="docs/figures/positioning_review.png" alt="Positioning review: segmentation and landmarks on an MLO / CC pair, PNL and chest-wall distance, criteria panel">
</picture>

<sub>Public CMMD images. Both measurement routes (landmark and segmentation) shown with their PNL / chest-wall
distances and the positioning criteria.</sub>

## What is in the repository

| Part | What it does | Where |
|---|---|---|
| **Positioning review app** | Upload a study (DICOM folder or files); the app segments each view, places the landmarks, measures the MLO posterior nipple line and the CC nipple-to-chest-wall distance, applies the ΔD rule, grades the positioning criteria and writes an EN / TR report. Labels can be corrected in the browser. | `breast_seg.apps.positioning` · [docs/APP.md](docs/APP.md) |
| **Inference engine** | One registry of models (`configs/models.yaml`), the preprocessing recipes they were trained with, and a landmark engine usable from Python. | `breast_seg.inference`, `breast_seg.registry`, `breast_seg.imaging` |
| **Positioning geometry** | PNL, chest-wall distance, criteria and scoring, independent of any model. | `breast_seg.geometry`, `breast_seg.positioning` |
| **Training** | One `Trainer` interface over YOLO-seg, YOLO-pose, YOLO-detect, SegFormer, nnU-Net, U-Net and AttUNet; YAML configs with inheritance and `--set` overrides; k-fold cross-validation. | `breast_seg.training` · [docs/TRAINING.md](docs/TRAINING.md) |
| **Datasets** | Builders that export one labelled pool to YOLO, semantic-mask and nnU-Net layouts with study-consistent, leak-checked splits. | `breast_seg.datasets` |
| **Evaluation** | Dice / IoU, landmark and distance errors in mm, the PNL decision rule, paired tests and bootstrap intervals, out-of-fold merging. | `breast_seg.evaluation` · [docs/EVALUATION.md](docs/EVALUATION.md) |
| **Annotation toolkit** | PyQt5 desktop app: MedSAM2 box / point prompting, manual polygon labelling with an optional detector, and a landmark-based positioning check. | `tools/annotation` · [README](tools/annotation/README.md) |

Every model, its training recipe and its expected weight file are listed in [docs/MODELS.md](docs/MODELS.md).

## The PNL rule in one paragraph

On the MLO view the posterior nipple line runs from the nipple perpendicular to the pectoral muscle edge; on the CC
view the matching distance runs from the nipple to the chest wall. For one breast,
`ΔD = |PNL(MLO) − D(CC)|`; a pair with `ΔD ≤ 10 mm` counts as adequately positioned (the threshold is a setting).
The app measures ΔD twice, from the pose landmarks and from the segmentation masks, and says when the two disagree.

## Quick start

### 1. Weights

Weights are not part of the repository. Put them in a `weights/` folder (git-ignored) with this layout, or point
`--weights` / `$BREAST_SEG_WEIGHTS` at a folder that has it:

```
weights/
  positioning/mlo-yolo26-pose-advanced.pt     MLO landmarks        (required)
  positioning/cc-yolo26-pose-advanced.pt      CC landmarks         (required)
  segmentation/breast_seg_v4_yolo26m.pt       segmentation         (required)
  detector/manual_detector.pt                 lesion detector      (optional)
  medsam2/medsam_model_best.pth               MedSAM2 fine-tune    (annotation toolkit)
  medsam2/sam2_hiera_tiny.pt                  SAM2 base            (annotation toolkit)
```

The annotation toolkit reads the same folder, so one copy serves both.

### 2a. Docker (CPU, no Python setup)

```bash
docker compose up --build            # http://localhost:8000
WEIGHTS_DIR=/path/to/weights docker compose up
```

The port is bound to this machine only; `POSITIONING_HOST=0.0.0.0 docker compose up` opens it to the network, which needs an
authenticated proxy in front (the app has no login). Studies are kept in the `positioning-data` volume. Uploaded DICOM files are never written to disk: only the
display renders and the label JSON are stored, and they expire after 30 days by default.

### 2b. Python

```bash
pip install -c constraints.txt -e ".[app,models,dicom]"   # CPU is enough; a CUDA build of torch is used when present
breast-seg models                             # every model with "weights found" / MISSING
breast-seg serve --device cpu                 # http://127.0.0.1:8000
breast-seg serve --host 0.0.0.0 --port 8077 --device 0 --weights /path/to/weights
```

`constraints.txt` pins the library versions the measurements were validated with (OpenCV, NumPy and Ultralytics
releases move ΔD by a few tenths of a millimetre); it applies to Python 3.9–3.11, the versions the Docker image and
CI use. Without it the package still installs, with the newest libraries.

Settings (threshold, retention, upload limits, criteria bands, interface language) live in an app config; copy
[`configs/app.example.yaml`](configs/app.example.yaml) and pass it with `--config`, or override single keys with
`--set rules.threshold_mm=12`.

### Using the models from Python

```python
from breast_seg.imaging import read_dicom
from breast_seg.inference import LandmarkEngine
from breast_seg.registry import Registry

engine = LandmarkEngine(Registry.load("configs/models.yaml"), device="cpu").load()
pixels, meta = read_dicom(open("L-MLO.dcm", "rb").read(), "L-MLO.dcm")
result = engine.analyse_landmarks(pixels, meta)   # nipple, pectoral line, masks as polygons, in original pixels
```

## Training and evaluation

```bash
pip install -e ".[models,train]"                       # + ".[segformer]" / ".[nnunet]" for those families
cp configs/paths.example.yaml configs/paths.yaml       # machine paths live only here (git-ignored)

breast-seg build-dataset configs/datasets/seg_v5.yaml
breast-seg train configs/train/yolo26m_seg_v5.yaml --set device=0
breast-seg export configs/train/yolo26m_seg_v5.yaml          # copy the selected checkpoint to its release name
breast-seg cv configs/cv/cv10_yolo26m.yaml             # train, predict out-of-fold, merge, evaluate
breast-seg evaluate --config configs/cv/cv10_yolo26m.yaml --preds runs/preds --out results --name yolo26m
```

Training configs exist for every model family in the catalogue (`configs/train/`); cross-validation configs for
YOLO26m-seg, SegFormer-B3 and nnU-Net (`configs/cv/`). See [docs/TRAINING.md](docs/TRAINING.md).

## Annotation toolkit

```bash
pip install -e ".[annotation,models,dicom]"
python tools/annotation/app.py
```

Three modes behind one launcher: **SAM Label** (MedSAM2 box / point prompts, refine as polygons), **Manual Label**
(polygons and boxes, optional one-click detector) and **Breast Positioning** (landmarks on an MLO + CC pair with a
live PNL / chest-wall overlay). Labels are saved in YOLO segmentation format. Details in
[tools/annotation/README.md](tools/annotation/README.md).

## Repository layout

```
src/breast_seg/
  apps/positioning/     review web application (FastAPI + build-free client)
  inference/            landmark engine, lazy model loading
  imaging/              DICOM reading, preprocessing recipes, display renders
  positioning/          measures, criteria, scoring, reports
  training/             trainers per model family, k-fold runner
  datasets/             dataset builders and exporters, split checks
  evaluation/           metrics, PNL rule, statistics
  compare/              ground truth vs landmarks vs segmentation on a labelled set
  geometry.py           PNL and chest-wall geometry
  registry.py           model registry
  cli.py                the breast-seg command
configs/                models.yaml, app example, train / datasets / cv configs, paths example
tools/annotation/       PyQt5 annotation toolkit (MedSAM2, manual, positioning)
docs/                   APP, MODELS, TRAINING, EVALUATION
tests/                  unit and API tests (CPU, no weights needed)
```

## Development

```bash
pip install -c constraints.txt -e ".[app,models,dicom,train,dev]"
pytest                                              # package tests
QT_QPA_PLATFORM=offscreen pytest tools/annotation/tests
ruff check src tests tools
```

Tests that need the real weights are marked `weights` and skip themselves when the files are absent.

## Notes

Research software, not a medical device: every measurement must be checked by a reader. The interface is English
by default with a Turkish switch.

## License

MIT, see [LICENSE](LICENSE). The vendored [Tabler](https://tabler.io) stylesheet is MIT-licensed as well.
