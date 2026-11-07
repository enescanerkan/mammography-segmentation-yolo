# MedSAM2 research scripts

**These are research scripts, not part of the toolkit.** They were used to prepare data
for, fine-tune and evaluate the MedSAM2 model that SAM Label mode loads. They run from the
command line, write only into `experiments/data/` and `experiments/work_dir/` (both
git-ignored) and take the MedSAM2 code / weight locations from the toolkit config
(`toolkit_config.py`, see the [toolkit README](../README.md#configuration)).

| Script | Purpose |
|--------|---------|
| `config.py` | Data / output folders, mask class ids, thresholds; model paths from the toolkit config |
| `export_seg_dataset_to_raw_mammo.py` | YOLO-seg dataset (PNG + polygon `.txt`) → raster `data/raw_mammo/{images,masks}/` (0 = bg, 1 pectoral, 2 breast, 3 nipple) |
| `build_view_map.py` | `data/raw_mammo/view_map.csv` (`stem,view`) from a CC / MLO-split dataset, to tag results by view |
| `zero_shot_test.py` | SAM2 base model with an oracle box from the GT mask → per-class Dice (`metrics.csv`, `summary.txt`, optional figures) |
| `prepare_medsam2_data.py` | Raster masks → `.npy` pairs for `finetune_sam2_img.py` (image 1024² × 3, GT 256² binary, one pair per case × class) |
| `train_medsam2.py` | Launches the upstream `finetune_sam2_img.py` with these defaults |
| `inference_finetuned.py` | One image + one box through a fine-tuned checkpoint → mask PNG |
| `gradio_demo.py`, `gradio_ui.py` | Browser prototype of SAM mode (needs `pip install gradio`) |

## Setup

Same environment as the toolkit plus the MedSAM2 code (see the toolkit README,
*Installation*). Fine-tuning needs a CUDA build of PyTorch and `monai` / `scikit-image`
for the upstream script:

```bash
pip install monai scikit-image
```

Data defaults: the YOLO-seg dataset is `$SEG_DATASET_ROOT` (default `<repo>/seg-dataset`);
every script also takes explicit paths (`--help`).

## Typical sequence

```bash
E=tools/annotation/experiments
python $E/export_seg_dataset_to_raw_mammo.py               # seg-dataset -> data/raw_mammo
python $E/build_view_map.py --compare-root /path/to/compare-dataset   # optional CC/MLO tags
python $E/zero_shot_test.py --limit 5 --device cpu         # sanity check of SAM2 base + box prompts
python $E/prepare_medsam2_data.py --use-project-splits     # data/medsam2_npy/{train,val,test}
python $E/train_medsam2.py --dry-run                       # print the fine-tune command
python $E/train_medsam2.py --device cuda:0                 # fine-tune (long-running, GPU)
```

The fine-tune writes `work_dir/<task-name>-<timestamp>/medsam_model_best.pth`. The toolkit
does **not** pick runs automatically: set `medsam2.checkpoint` (or `$MEDSAM2_CHECKPOINT`)
to the checkpoint you want SAM mode to use, then check it with
`python tools/annotation/app.py --print-config`.

## Notes

- `zero_shot_test.py` uses an *oracle* box derived from the GT mask: it measures mask
  quality given a good box, not detection. The default weights are the SAM2 Hiera-Tiny
  base; `$MEDSAM2_ZERO_SHOT_CKPT` or `--ckpt` selects another compatible checkpoint (the
  Hugging Face `MedSAM2_latest.pt` may not match `sam2_hiera_t.yaml` on every MedSAM2
  revision).
- `finetune_sam2_img.py` expects `gts/*.npy` of 256 × 256 and `imgs/*.npy` of
  1024 × 1024 × 3 (uint8); `prepare_medsam2_data.py` produces exactly that.
- Use full segmentation polygons as GT. Pose-style labels (a small pectoral triangle for
  landmark models) are not a substitute; `build_view_map.py` only reads file names from
  such datasets.
