# Project TODOs & Future Roadmap

## 1. Algorithm Enhancements
- [ ] Evaluate polynomial curve fitting (degree > 1) rather than strict linear constraints for pectoral boundaries if needed for severe anatomical aberrations.
- [ ] Implement iterative refinement on the "middle bulge" (pt_max) calculation with contour smoothing algorithms to bypass noise.

## 2. Advanced Modeling & Configuration
- [ ] Integrate YOLO26 implementation options directly into the pipeline (`yolo26n.pt`, `yolo26m.pt`, etc.).
- [ ] Implement hyperparameter sweeps inside `config.py` using Optuna for optimal confidence/IOU thresholds.
- [ ] Allow dynamic CLI arguments to overwrite `Config` paths (e.g. `--model yolo26-seg.pt`).

## 3. Data Augmentation Pipelines
- [ ] Integrate Albumentations framework manually into the Ultralytics loader context for advanced morphological operations (CLAHE, Elastic Transform).
- [ ] Introduce conditional augmentations specifically targeting CC view vs. MLO view.

## 4. Ground Truth Metrics & Statistical Evaluation
- [ ] Add coordinate-parsing logic for Doctor Annotations (JSON/CSV bounding boxes/polygons).
- [ ] Calculate Hausdorff Distance and Dice Similarity Coefficient between the predicted segmentation mask and the Ground Truth.
- [ ] Evaluate measurement error of the PNL & CC Depth lengths (predicted vs baseline reference lengths).
