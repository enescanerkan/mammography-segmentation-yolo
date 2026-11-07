# Model catalogue

Every model of the project: what it is, what it is fed, what it predicts, how it was trained, which dataset
generation it was trained on, the config that reproduces the recipe and the weight file name the tools expect.
Weights are not part of the repository; they are distributed separately and placed under the weights root
(`weights_dir` of `configs/paths.yaml`, or the root given to the model registry `configs/models.yaml`).

This page describes recipes only. It deliberately contains no results.

## Conventions

* **Canvas.** All models except MedSAM2 work on a 640 x 640 canvas made from the DICOM: VOI LUT, MONOCHROME1
  inversion, breast bounding box, crop, min-max normalisation of the crop to 8 bit, aspect-preserving resize to a
  longest side of 640 and black padding on the side away from the chest wall. Per-image metadata (crop offset,
  scale, padding, pixel spacing, laterality) maps canvas points back to the original image.
  Two input recipes exist: `baseline640` (the canvas as is) and `wavelet640` (the canvas enhanced by
  non-local-means denoising, a Daubechies-4 level-2 detail boost and CLAHE, applied before padding).
* **Segmentation classes.** YOLO class ids `0 pectoral`, `1 breast-tissue`, `2 nipple`. Semantic label maps use
  `0 background`, `1 pectoral`, `2 breast-tissue`, `3 nipple`; instance masks are flattened in the order breast
  tissue, pectoral over it, nipple on top (see `breast_seg.config.PAINT_ORDER`).
* **Landmarks.** MLO: `nipple`, `pectoral_top`, `pectoral_bottom`; CC: `nipple`.
* **Anatomy-safe augmentation.** Horizontal flips are allowed (left and right breasts both occur), vertical flips
  never (the pectoral muscle is always at the top of an MLO view).
* **Configs.** `configs/train/<name>.yaml`, loaded by `breast_seg.training.load_train_config`; files starting with
  `_` are shared fragments inherited through `base:`.

## Summary

| Model | Family | Config | Expected weight file |
|---|---|---|---|
| Landmarks, MLO | `yolo_pose` | `pose_mlo_yolo26l.yaml` | `mlo-yolo26-pose-advanced.pt` |
| Landmarks, CC | `yolo_pose` | `pose_cc_yolo26l.yaml` | `cc-yolo26-pose-advanced.pt` |
| YOLO26m-seg v1 | `yolo_seg` | `yolo26m_seg_v1.yaml` | `breast_seg_yolo26m.pt` |
| YOLO26m-seg v2 | `yolo_seg` | `yolo26m_seg_v2.yaml` | `breast_seg_v2_yolo26m.pt` |
| YOLO26m-seg v3 | `yolo_seg` | `yolo26m_seg_v3.yaml` | `breast_seg_v3_yolo26m.pt` |
| YOLO26m-seg v4 (released segmentation model) | `yolo_seg` | `yolo26m_seg_v4.yaml` | `breast_seg_v4_yolo26m.pt` |
| YOLO26m-seg, 300-epoch schedule | `yolo_seg` | `yolo26m_seg_300ep.yaml` | `yolo26m_seg_300ep.pt` |
| YOLO26m-seg v5 | `yolo_seg` | `yolo26m_seg_v5.yaml` | `yolo26m_seg_v5.pt` |
| YOLO26l-seg | `yolo_seg` | `yolo26l_seg.yaml` | `yolo26l_seg_300ep.pt` |
| YOLOv8m-seg | `yolo_seg` | `yolov8m_seg.yaml` | `yolov8m_seg_300ep.pt` |
| YOLOv8l-seg | `yolo_seg` | `yolov8l_seg.yaml` | `yolov8l_seg_300ep.pt` |
| SegFormer-B3 | `segformer` | `segformer_b3.yaml` | `segformer_b3/` (Hugging Face folder) |
| nnU-Net ResEnc-M | `nnunet` | `nnunet_resenc_m.yaml` (+ `_1000ep`, `_v5`) | `nnunet_resenc_m_fold0.zip` (nnU-Net model export) |
| U-Net | `unet` | `unet_plain.yaml` | `unet_plain.pt` |
| AttUNet, landmark arm | `attunet` | `attunet_landmark.yaml` | `attunet_landmark.pt` |
| AttUNet, multitask arm | `attunet` | `attunet_multitask.yaml` | `attunet_multitask.pt` |
| MedSAM2 fine-tune | (annotation toolkit) | `tools/annotation/experiments/train_medsam2.py` | `medsam_model_best.pth` |
| Lesion detector | `yolo_detect` | `lesion_yolov8m_detect.yaml` | `manual_detector.pt` |

`Trainer.export()` copies the selected checkpoint of a run to `<weights_dir>/<expected file>`. Inside a run
directory the checkpoints are `weights/best.pt` (YOLO), `best.pt` / `last.pt` (U-Net, AttUNet), `best/` / `last/`
(SegFormer) and `<nnunet_results>/Dataset<ID>_<name>/<trainer>__<plans>__2d/fold_<k>/checkpoint_final.pth` (nnU-Net).
The run names in the configs are the historical ones, so a registry entry may equally point at a run directory
(for example `breast_seg_v4_yolo26m/weights/best.pt`) instead of an exported file.

## Landmark models (MLO / CC)

* **Architecture.** Ultralytics YOLO26l-pose, one class (`Breast`) whose box carries the keypoints;
  `kpt_shape` `[3, 3]` for MLO, `[1, 3]` for CC. Initialised from the COCO pose weights `yolo26l-pose.pt`.
* **Input.** `wavelet640` canvases (training and inference must both use the enhancement; feeding raw canvases is
  out of distribution).
* **Outputs.** MLO: nipple, pectoral_top, pectoral_bottom; CC: nipple. The posterior nipple line is the
  perpendicular from the nipple to the line through the two pectoral points.
* **Training recipe.** 300 epochs, patience 50, batch 26, 640 input, optimizer `auto`, `lr0` 0.01, `lrf` 0.01,
  warm-up 3 epochs, AMP, mosaic 1.0 (off for the last 10 epochs), translate 0.1, scale 0.5, HSV 0.015 / 0.7 / 0.4,
  no rotation, no flips (Ultralytics disables horizontal flipping for keypoints without a `flip_idx`), seed 0.
  Values as stored in the released checkpoints.
* **Dataset.** One canvas per image with reference landmarks, split by the label tables' `Split` column; the crop
  box is widened so that every landmark stays on the canvas (80 px margin in the original image). Label line per
  image: the keypoint-extent box padded by 100 px and the keypoints with visibility 2
  (`breast_seg.datasets.pose_label_line`).

## YOLO26m-seg, dataset generations v1-v5

* **Architecture.** Ultralytics YOLO26m-seg (NMS-free detection head with mask prototypes and per-instance mask
  coefficients), initialised from the COCO weights `yolo26m-seg.pt`.
* **Input.** `baseline640` canvases, MLO and CC in one model (CC images carry the `cc_` file prefix).
* **Outputs.** Instances of pectoral, breast tissue and nipple. For evaluation they are flattened into a label map
  (confidence 0.25; if a class has no instance above it, its best instance is taken back when above 0.01, except
  the pectoral class on CC views).
* **Shared recipe (v2-v5, `_yolo_seg_base.yaml`).** AdamW, `lr0` 0.001, batch 16, 640 input, AMP, seed 42;
  rotation 8 degrees, translate 0.05, scale 0.2, horizontal flip 0.5, no vertical flip, brightness jitter only
  (HSV 0 / 0 / 0.3), mosaic 0.5 closed for the last 15 epochs, no mixup / copy-paste; only best / last
  checkpoints. Pixel-level Albumentations replace Ultralytics' default list (CLAHE, brightness / contrast, gamma,
  Gaussian / ISO / multiplicative / salt-and-pepper noise, blur, median blur, sharpen).
* **Schedules.** v2, v3, v4: 150 epochs, patience 30. The 300-epoch variant (benchmark baseline) and v5: 300
  epochs, patience 50; nothing else changes.
* **v1 (legacy).** Seed set; 100 epochs, patience 15, batch 8, AdamW `lr0` 0.001 / `lrf` 0.01, warm-up 3, no AMP,
  rotation 5, translate 0.1, scale 0.3, mosaic 0.5, default HSV jitter, Ultralytics flips off because the dataset
  already contains flipped copies, checkpoint every 10 epochs, seed 0.
* **Datasets.** See [Dataset generations](#dataset-generations). v4 is the released model.

## Benchmark backbones: YOLO26l-seg, YOLOv8m-seg, YOLOv8l-seg

Ultralytics YOLO26l-seg / YOLOv8m-seg / YOLOv8l-seg (COCO weights), trained with the 300-epoch YOLO26m-seg recipe
on seg-dataset_v4; only the backbone differs.

## SegFormer-B3

* **Architecture.** SegFormer with the MiT-B3 hierarchical transformer encoder (`nvidia/mit-b3`, ImageNet
  pre-trained) and the all-MLP decoder, four output classes; logits at 1/4 resolution are upsampled bilinearly to
  the canvas.
* **Input.** `baseline640` grey canvas repeated on three channels, ImageNet mean / std normalisation.
* **Training recipe.** Cross-entropy + soft Dice (foreground classes), AdamW `lr` 6e-5, weight decay 0.01,
  polynomial decay (power 1) per iteration, batch 16, 300 epochs, early stopping after 50 epochs without
  improvement of the validation mean foreground Dice, bf16 autocast, seed 42. Augmentation: horizontal flip,
  affine (rotation 8, scale 0.8-1.2, translation 5 %), brightness / contrast, gamma, CLAHE, Gaussian noise,
  blur / median blur, sharpen. Prediction averages the softmax with the horizontally flipped pass.
* **Dataset.** Label maps of seg-dataset_v4 (semantic export).

## nnU-Net ResEnc-M

* **Architecture.** nnU-Net v2 residual-encoder U-Net, ResEnc-M preset (`nnUNetPlannerResEncM`), 2D
  configuration, deep supervision.
* **Deviations from nnU-Net's defaults** (`nnUNetTrainerMammo`, shipped in `breast_seg/training/nnunet_variants.py`
  and installed into nnU-Net by `prepare()`): the patch is the full 640 x 640 canvas with batch size 12 (plans
  `nnUNetResEncUNetMPlans_fullM`, same network as the preset; the planner's smaller patch would crop wide
  breasts), and mirroring is horizontal only, in training and at inference.
* **Training recipe.** 300 epochs of nnU-Net's fixed 250 iterations (`nnunet_resenc_m_1000ep.yaml`: 1000), no
  early stopping, prediction with `checkpoint_final.pth`. Everything else is nnU-Net's own pipeline.
* **Dataset.** Label maps of seg-dataset_v4 as `Dataset601_BreastSeg`: train + val in `imagesTr`, test in
  `imagesTs`; `splits_final.json` fold 0 = the dataset's own train / val, folds 1-5 = study-level 5-fold CV over
  train + val. v5: `Dataset602_BreastSegV5`, fold 0 only.

## U-Net

* **Architecture.** Plain U-Net: four levels of double 3x3 convolution + BatchNorm + ReLU (widths 64-512,
  bottleneck 1024), max-pool down, transposed convolution up, concatenated skips, 1x1 head; random
  initialisation. The reference baseline without attention or residual blocks.
* **Input.** Single-channel canvas scaled to [0, 1] (no ImageNet statistics for a network trained from scratch).
* **Training recipe.** As SegFormer (loss, optimizer, schedule, augmentation, early stopping, flip TTA) except
  `lr` 1e-3 and batch 8.

## AttUNet (landmark and multitask arms)

* **Architecture.** Attention U-Net (attention gates on the skip connections) with CoordConv input (grey + x + y
  channels), the U-Net widths above and two 1x1 heads: four-class segmentation and one heatmap per landmark, read
  out with a spatial soft-argmax.
* **Arms.** Landmark (`w_seg = 0`): landmark loss only. Multitask (`w_seg = 1`): the segmentation loss
  (cross-entropy + soft Dice) is added through the shared network. The arms differ only in that loss, so their
  comparison isolates what the segmentation signal contributes to landmark placement.
* **Training recipe.** Landmark loss: masked L1 of normalised coordinates, weight 20 (missing landmarks and points
  pushed off the canvas by augmentation are masked); AdamW `lr` 1e-3, weight decay 0.01, polynomial decay, batch
  8, 300 epochs, early stopping on the validation landmark error (both arms), bf16 autocast, the SegFormer
  augmentation applied to image, mask and keypoints. Prediction: flip-averaged heatmap probabilities; the peak
  probability is reported as a confidence.
* **Dataset.** seg-dataset_v4 label maps plus the reference landmarks on the canvas
  (`breast_seg.evaluation.canvas_landmark_table`).

## MedSAM2 fine-tune (prompted segmentation)

SAM2 (Hiera-tiny image encoder) fine-tuned with box prompts through the MedSAM2 branch of the MedSAM code base,
used by the annotation toolkit to turn a box into a mask. Input: the raw image at 1024 x 1024 (`raw1024`), one
binary 256 x 256 target per structure and image. Training defaults of the launcher: 100 epochs, batch 2,
`lr` 1e-4, weight decay 0.01, box jitter 10 px. Data preparation, training and inference live in
`tools/annotation` (see its README); the expected checkpoint is `medsam_model_best.pth`. As a prompted model it
is not compared with the automatic models.

## Lesion detector

Ultralytics YOLOv8m detect (COCO weights) with two classes, `Mass` and `Calcification`, used by the annotation
toolkit to propose boxes. Recipe as stored in the released checkpoint: 100 epochs, patience 15, batch 8, 1024
input, optimizer `auto`, `lr0` 0.01, horizontal flip 0.5, mosaic 1.0, scale 0.5, translate 0.1, default HSV
jitter, seed 0. Its training dataset is not part of this repository; the config expects a YOLO detection dataset
under `${data_root}/lesions`.

## Dataset generations

Every generation is a full re-export, built from annotation pools that are never modified.

| Generation | Content | Split | Builder |
|---|---|---|---|
| v1 | Hand-labelled seed set, horizontal flips pre-built for train / val | random, by image | not ported (legacy) |
| v2 | Cleaned MLO annotations + the CC images of the seed set; CC oversampled in train | random by image, MLO and CC separately | not ported |
| v3 | v2 MLO without broken annotations + CC pseudo-labels of the v2 model, some corrected by hand; CC val / test only from human-verified images | random by image | not ported |
| v4 | Hand-annotated MLO + CC pseudo-labels of the v2 model (a subset corrected by hand), cleaned | label tables' `Split` column (study-consistent; test held out) | `configs/datasets/seg_v4.yaml` |
| v5 | v4 + completed public / external images, all in train | v4 kept exactly | `configs/datasets/seg_v5.yaml` |

v1-v3 split single images at random, so the two views of one study could fall into different splits; they are
documented for provenance only. From v4 on every split is study-level and audited (`DatasetBuilder.audit`).

* **Clean-up (v2 onward).** MLO: the pectoral mask is reduced to its largest component, holes filled and closed
  (9 px ellipse); the seam between breast tissue and pectoral muscle (41 px ellipse) is added to the breast tissue,
  never over the muscle. CC: breast and nipple holes filled, no pectoral class. Masks are re-polygonised
  (contours of at least 20 px, `approxPolyDP` with 0.2 % of the perimeter).
* **v5 sources.** Human-annotated and rule-checked model-completed images of public / external sets; every new
  image must contain all classes of its view. Images of the external test set and every other image of the same
  patients are excluded; no new image may share a study or a series with v4 validation / test.
* **Exports.** Label maps for SegFormer / U-Net / AttUNet (`semantic_v4`, `semantic_v5`) and the nnU-Net raw
  datasets 601 / 602.
