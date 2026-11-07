# Building datasets, training and cross-validation

Everything here is configuration-driven: datasets from `configs/datasets/`, models from `configs/train/`,
cross-validation from `configs/cv/`. The model catalogue with every recipe is in [MODELS.md](MODELS.md); metric
definitions are in [EVALUATION.md](EVALUATION.md).

## Setup

```bash
pip install -e ".[models,train]"        # YOLO families, U-Net, AttUNet
pip install -e ".[segformer]"           # SegFormer (transformers, timm)
pip install -e ".[nnunet]"              # nnU-Net v2
cp configs/paths.example.yaml configs/paths.yaml   # then edit (git-ignored)
```

`configs/paths.yaml` is the only place where machine paths are written. Committed configs refer to its entries
as `${name}`: the fixed locations (`data_root`, `runs_dir`, `weights_dir`, `predictions_dir`, `results_dir`,
`nnunet_raw`, `nnunet_preprocessed`, `nnunet_results`) and the named inputs under `extra:` (label tables, canvas
metadata, annotation pools, the fold table...). Relative values are resolved against `root`, which is relative to
the paths file. Another file can be selected with `BREAST_SEG_PATHS`; without any file every location defaults to
a folder under the working directory.

## Config files

* **Inheritance.** `base: other.yaml` (or a list) merges the base first; nested mappings merge key by key, lists
  are replaced. Files starting with `_` are fragments meant only as bases.
* **Overrides.** Dotted `key=value` strings, the value parsed as YAML: `hyp.epochs=300`, `device=1`,
  `augment=[]`, `data.dataset_dir=${data_root}/seg-dataset_v5`.
* **Validation.** Configs become typed dataclasses; unknown keys, wrong types and missing required values raise a
  `ConfigError` naming the dotted key.

```python
from breast_seg.training import load_train_config
cfg = load_train_config("configs/train/yolo26m_seg_v4.yaml", ["device=1"])
```

## Building a dataset generation

```python
from breast_seg.config import PathsConfig
from breast_seg.datasets import DatasetBuilder, load_dataset_config

report = DatasetBuilder(load_dataset_config("configs/datasets/seg_v4.yaml"), PathsConfig.load()).build()
```

The builder collects its sources, assigns splits, cleans the labels, writes
`<output_dir>/{images,labels}/<split>` and `index.csv`, audits the result and only then writes `data.yaml` and
the configured exports. It never modifies a source and refuses an output folder that contains one (the output
folder is rebuilt from scratch). Build v4 before v5: v5 reads the v4 dataset.

**Sources.** `yolo_dir` (flat images + YOLO polygon labels, stem = `prefix` + file name), `yolo_layout` (an
existing YOLO dataset whose split is kept with `split: inherit`) and `manifest` (a CSV naming each image and label
through templates, filtered with `include` and `exclude` rules; an exclusion can also drop every image sharing a
group with an excluded one, e.g. the other images of an external-test patient).

**Splits.** `map` (the split column of one or more tables, keyed by image id), `ratio` (one joint group-level
random split over every ratio source, so a study never straddles two splits), `inherit`, or a fixed split.
`oversample` replicates train images of a source.

**Groups and leakage.** `groups` tables map image ids to study / series groups (namespaced, e.g. `study:` and
`series:`); manifest rows bring their own group. The audit fails if any group occurs in more than one split.

**Audit.** Image size and dtype, identical pixels, label syntax, breast tissue present, every class of the view
present (sources with `strict_classes`), no pectoral polygon on CC views, files and index in agreement, leakage.
On failure `DatasetAuditError` lists every problem (`build(strict=False)` returns them instead) and no
`data.yaml` is written.

**Exports.** `semantic`: grey images, label maps and an index (input of SegFormer, U-Net, AttUNet and nnU-Net).
`nnunet`: the raw nnU-Net dataset with `splits_final.json` (fold 0 = the dataset's own train / val, plus optional
group-level CV folds).

**Landmark (pose) and AttUNet inputs.** The landmark models train on `wavelet640` canvases with one YOLO-pose label
per image (`breast_seg.datasets.pose_label_line`); AttUNet reads a canvas landmark table
(`breast_seg.evaluation.canvas_landmark_table`, columns `stem, role, x, y`) next to the semantic export.

## Training

```python
from breast_seg.training import build_trainer

trainer = build_trainer("configs/train/yolo26m_seg_v4.yaml", overrides=["device=0"])
trainer.prepare()          # writes <runs_dir>/<name>/data.yaml and train_config.json; trains nothing
result = trainer.fit()     # result.weights = the selected checkpoint
trainer.evaluate("val")    # family metrics on a split (see EVALUATION.md)
trainer.predict(images, out_dir)   # common label-map format (landmarks.csv / detections.csv for pose / detect)
trainer.export()           # copy the checkpoint to <weights_dir>/<export.filename>
```

The run directory is `<runs_dir>/<name>`. Family notes:

* **YOLO (`yolo_seg`, `yolo_pose`, `yolo_detect`).** Base weights are looked up under `weights_dir` and otherwise
  downloaded by Ultralytics. Only non-null `hyp` values are passed to `YOLO.train`, everything else keeps the
  Ultralytics default. The Albumentations patch (`albumentations.enabled`) swaps Ultralytics' default
  pixel-level list for the configured one while training runs; it works for single-device training in the
  current process (a multi-GPU DDP launch starts new interpreters that do not see it).
* **SegFormer, U-Net, AttUNet.** One PyTorch loop: AdamW, polynomial decay per iteration, autocast
  (`optim.amp_dtype`), validation every epoch, early stopping on the monitored metric (mean foreground Dice;
  landmark error for AttUNet), `best` and `last` checkpoints, `history.csv`. Albumentations is needed when
  `augment` is not empty.
* **nnU-Net.** `prepare()` converts the semantic dataset (unless the raw dataset exists and `data.reuse_raw`),
  runs `nnUNetv2_plan_and_preprocess`, `nnUNetv2_plan_experiment -pl nnUNetPlannerResEncM`, derives the
  full-canvas plans (`patch_size`, `batch_size`), preprocesses with them and installs `nnUNetTrainerMammo` into the
  nnU-Net package. `fit()` runs `nnUNetv2_train` (continuing from `checkpoint_latest.pth` when a run was
  interrupted). The three nnU-Net locations come from the paths file and are exported to every command.

Training a recipe exactly as the config states reproduces the training *procedure*; bit-identical weights also
depend on library versions, hardware and non-deterministic kernels.

## Cross-validation

```python
from breast_seg.training import CrossValidationRunner

runner = CrossValidationRunner.from_yaml("configs/cv/cv10_yolo26m.yaml")
runner.prepare()           # materialise every fold, no training
report = runner.run()      # train + predict each fold, merge out-of-fold predictions, evaluate
```

* **Folds are read, never redrawn**, from the fold table (`folds.csv`): one row per image (`format: images`) or one
  row per MLO/CC pair (`format: pairs`, converted with the dataset naming). Images of the cohort that the table
  misses are placed in the fold of their study (`data.group_table`); the table is checked for studies spread over
  several folds.
* **Protocol** of fold `i`: test = fold `i`, validation = fold `(i + val_offset) % k`, train = the remaining folds.
  Images without labels never enter train or validation; they are predicted in their own test fold only
  (`data.unlabelled_images`).
* **Per family.** YOLO folds are trees of links (`images/` and `labels/`), the PyTorch families get a fold index
  with absolute paths, nnU-Net gets one shared dataset with one `splits_final.json` entry per fold and
  `imagesTest_fold<i>` folders.
* **Outputs.** Fold runs are `<runs_dir>/<cv name>_fold<i>`; each fold model predicts only its test fold into
  `<predictions_dir>/<cv name>_fold<i>`; `merge()` copies them into `<predictions_dir>/<cv name>` and fails if an
  image has no prediction from its own fold; `evaluate()` scores the merged directory and adds a per-fold decision
  table. Finished folds are skipped (`skip_finished`), so a run can be restarted.
* **Configs.** `cv10_yolo26m`, `cv10_segformer_b3`, `cv10_nnunet` reuse the main recipes; YOLO and SegFormer run
  the full schedule in every fold (patience equal to the epoch count, the checkpoint still selected on the fold's
  validation fold), nnU-Net transfers the full-canvas plans of the main dataset to the CV dataset.
