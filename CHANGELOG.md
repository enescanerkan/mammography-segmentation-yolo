# Changelog

## 1.0.0 — unreleased

The repository is restructured into one installable package, `breast_seg` (distribution `mammography-segmentation`),
with a command line, a review web application, a training framework and the annotation toolkit.

### Added
- **Positioning review application** (`breast-seg serve`, `breast_seg.apps.positioning`): DICOM upload, segmentation
  and landmarks per view, PNL / chest-wall distance from both routes, ΔD rule, positioning criteria and score,
  EN / TR reports, a label editor, light and dark themes, full screen and pan / zoom. Uploaded DICOM files are parsed
  in memory and never stored. Previously a separate application.
- **Model registry** (`configs/models.yaml`, `breast_seg.registry`): every weight file with its preprocessing recipe
  and role; one shared `weights/` layout for the app and the annotation toolkit.
- **Inference engine** (`breast_seg.inference.LandmarkEngine`) and **imaging** (`breast_seg.imaging`: DICOM reading,
  the `wavelet640` / `baseline640` recipes, display renders).
- **Positioning** (`breast_seg.positioning`): measures, criteria grading, scoring, reports, label edits.
- **Training framework** (`breast_seg.training`, `breast-seg train` / `cv`): one `Trainer` interface for YOLO-seg,
  YOLO-pose, YOLO-detect, SegFormer, nnU-Net, U-Net and AttUNet; typed YAML configs with inheritance and `--set`
  overrides; k-fold cross-validation with out-of-fold merging.
- **Dataset builders** (`breast_seg.datasets`, `breast-seg build-dataset`): YOLO, semantic and nnU-Net exports with
  study-consistent, leak-checked splits.
- **Evaluation library** (`breast_seg.evaluation`, `breast-seg evaluate`): mask metrics, landmark and distance errors,
  the PNL rule with its counting conventions, paired tests and bootstrap intervals.
- `breast-seg` command line, Docker image (CPU) and compose file, CI (lint, tests, image smoke test).
- **Windows desktop app** (`packaging/windows`, `breast_seg.apps.desktop`): the review app in a native window, CPU
  only, weights included; built, self-tested and published by GitHub Actions on every change (`dev-build`) and for
  every `v*` tag. The review app's weights are published in the `weights-v1` release (`packaging/weights.py`).
- Documentation: `docs/APP.md`, `docs/MODELS.md`, `docs/TRAINING.md`, `docs/EVALUATION.md`.
- Architecture figures of every model (light and dark SVG, `docs/figures/architectures`, drawn by
  `tools/figures/architectures.py`) and a model overview in the README.

### Changed
- src layout; heavy dependencies are optional extras (`app`, `models`, `dicom`, `train`, `segformer`, `nnunet`,
  `annotation`, `dev`).
- The annotation toolkit moved to `tools/annotation` (launcher `app.py`), reads a YAML / environment config, and its
  positioning mode uses the package geometry. Research scripts are under `tools/annotation/experiments`.
- The compare pipeline is `breast_seg.compare` (`breast-seg compare`); pairs without an answer now count as failures.

### Removed
- `start.py`, `run_train.py`, `build_dataset.py`, `requirements.txt` (replaced by the command line and the extras).
- Hard-coded download links and machine paths.
