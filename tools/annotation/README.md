# Mammography Annotation Toolkit

A PyQt5 desktop application for labelling mammograms and checking positioning. It
produces YOLO-segmentation labels (pectoral muscle, breast tissue, nipple, or any classes
you define) and shows the posterior nipple line (PNL) / CC depth computed from those labels
with the same geometry as the `breast_seg` package.

One launcher opens three tools; **File → Back to Main Menu** (`Ctrl+M`) switches between
them without restarting.

| Mode | What it does | Needs |
|------|--------------|-------|
| **SAM Label** | Draw a box or click points; a fine-tuned MedSAM2 model returns the mask. Refine it as a polygon. | MedSAM2 code + SAM2 base weights + fine-tuned checkpoint |
| **Manual Label** | Polygon / box labelling with your own classes. Optional one-click YOLO box detector (for example a lesion detector) whose boxes become editable polygons. | Nothing (detector weights only for the auto-detect button) |
| **Breast Positioning** | YOLO-pose landmarks on an MLO + CC DICOM pair; compares the MLO nipple-to-pectoral and CC nipple-to-chest-wall distances against a mm threshold. | Two YOLO-pose models |

When a model file is missing, the mode says which file it expected and which config key
sets it, then returns to the launcher (Manual mode itself needs no weights).

## Modes

### SAM Label (MedSAM2 prompting)

1. Open an image or a folder (PNG / JPEG / TIFF / DICOM).
2. Pick the class: **Pectoral Muscle** (`1`), **Breast Tissue** (`2`), **Nipple** (`3`).
3. **Bounding Box** mode (`B`, default): click two opposite corners; the mask appears.
   **Point** mode (`P`): left-click adds positive points. In both modes a right-click adds
   an *ignore* point.
4. Drag box corners or points to update the prediction; `Alt`+click deletes a point.
5. **Edit Mask as Polygon** turns the mask into a draggable polygon (drag vertices, click
   an edge to insert, `Alt`+click to delete). `Ctrl+Enter` commits it.
6. `Ctrl+S` saves every annotated image as YOLO labels.

Settings in the side panel:

- **Ignore radius**: an ignore point removes a disk of this radius from the mask, whatever
  the model predicts (and is also sent to the model as a negative point unless
  **soft negative** is off).
- **Implicit box**: with points only, a padded box around the positive points is sent too.
  The fine-tuned decoder was trained with box prompts, so this keeps point prompts in
  distribution.
- **Class competition**: when you draw breast tissue after the pectoral muscle, the drawn
  pectoral pixels are subtracted from the breast mask (the usual MLO leak).

The model is class-agnostic: the selected class only sets the color, the YOLO id and the
class-competition rule. Where you click decides what is segmented. Details:
[docs/prompting.md](docs/prompting.md) and [docs/sam_label.md](docs/sam_label.md).

### Manual Label (with the box detector)

Manual mode starts with an empty class list: **＋ Add Custom Class** creates a class (the
first one is YOLO id 0) with a unique color. Draw with **Bounding Box** (two clicks give a
4-vertex polygon) or **Polygon** (click vertices, double-click / `Enter` closes). Several
instances per class are kept; **◀ Prev / Next ▶** picks the instance to edit, right-click
on a box renames or deletes it.

**Run Box Detector** runs the YOLO detector configured as `detector.weights` on the current
image and adds every detected box as a polygon of the *active* class (the detector's own
class ids are ignored), ready for corner-drag refinement. See
[docs/manual_label.md](docs/manual_label.md).

### Breast Positioning

Select an MLO + CC DICOM pair (or a folder; files are paired when swapping `mlo` / `cc` in
the file name gives an existing file), run **MLO Analysis**, **CC Analysis**, then
**Compare**. Landmarks are draggable and the distances update live. The threshold slider
sets the mm tolerance used by Compare and Save. **Save Results** writes TXT / CSV / JSON and
**Save Images** writes PNG figures to `positioning.results_dir`. See
[docs/breast_positioning.md](docs/breast_positioning.md).

### PNL / CC-depth overlay (SAM and Manual modes)

Press **Measure** (`C`) after labelling the nipple plus the pectoral muscle and breast
(MLO) or the nipple plus the breast (CC). The overlay draws the fitted pectoral line, the
PNL perpendicular (or the CC depth line) and a HUD with the length in pixels. The geometry
is `breast_seg.geometry` (pectoral line by the tangent rule, nipple centroid, foot point
clamped to the pectoral segment), so the toolkit and the package always agree.

## Installation

Python 3.9+. From the repository root:

```bash
pip install -e ".[annotation,models,dicom]"      # package + PyQt5, torch, ultralytics, DICOM codecs
# or, without installing the package:
pip install -r tools/annotation/requirements.txt
```

The toolkit imports `breast_seg.geometry`; if the package is not installed it falls back
to `<repo>/src` automatically.

SAM mode additionally needs the MedSAM2 code (MedSAM2 branch of bowang-lab/MedSAM), which
provides `sam2` and its Hydra configs:

```bash
git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git tools/annotation/vendor/MedSAM
pip install -e tools/annotation/vendor/MedSAM
```

`tools/annotation/vendor/` is git-ignored; another location works through `medsam2.repo`.
The toolkit adds that folder to `sys.path` itself, but `hydra-core` and `iopath` (listed
in `requirements.txt`) must be importable.

## Weights

Model weights are not stored in git; they are distributed separately from the code (see
`docs/MODELS.md` at the repository root). By default every file is looked up under
`tools/annotation/weights/` (git-ignored):

```
tools/annotation/weights/
├── medsam2/
│   ├── sam2_hiera_tiny.pt          # SAM2 Hiera-Tiny base (Meta), architecture of the fine-tune
│   └── medsam_model_best.pth       # fine-tuned MedSAM2 checkpoint (experiments/train_medsam2.py)
├── detector/
│   └── manual_detector.pt          # YOLO detection model for "Run Box Detector"
└── positioning/
    ├── mlo-yolo26-pose-advanced.pt # YOLO-pose: nipple + 2 pectoral-line points
    └── cc-yolo26-pose-advanced.pt  # YOLO-pose: nipple
```

The SAM2 base checkpoint is Meta's public `sam2_hiera_tiny.pt`
(`https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt`).

**The MedSAM2 checkpoint is explicit.** The app loads exactly the file set as
`medsam2.checkpoint` (default `weights/medsam2/medsam_model_best.pth`, or
`$MEDSAM2_CHECKPOINT`). It does not scan training folders for the newest run; after a new
fine-tune, point the key at the run you want.

## Configuration

Settings are read from, in increasing priority: built-in defaults → a YAML file → one
environment variable per key. The YAML file is `--config PATH`, else `$ANNOTATION_CONFIG`,
else `tools/annotation/annotation.yaml` if present (git-ignored; start from
[annotation.example.yaml](annotation.example.yaml)).

| Key | Default | Environment variable |
|-----|---------|----------------------|
| `weights_root` | `weights` | `ANNOTATION_WEIGHTS_ROOT` |
| `medsam2.repo` | `vendor/MedSAM` | `MEDSAM2_REPO` |
| `medsam2.model_cfg` | `sam2_hiera_t.yaml` | `SAM2_MODEL_CFG` |
| `medsam2.sam2_base` | `medsam2/sam2_hiera_tiny.pt` | `MEDSAM2_SAM2_BASE` |
| `medsam2.checkpoint` | `medsam2/medsam_model_best.pth` | `MEDSAM2_CHECKPOINT` (legacy alias `MEDSAM2_INTERACTIVE_CKPT`) |
| `medsam2.device` | `auto` (CUDA if available) | `MEDSAM2_DEVICE` |
| `detector.weights` | `detector/manual_detector.pt` | `ANNOTATION_DETECTOR_WEIGHTS` |
| `positioning.mlo_weights` | `positioning/mlo-yolo26-pose-advanced.pt` | `POSITIONING_MLO_WEIGHTS` |
| `positioning.cc_weights` | `positioning/cc-yolo26-pose-advanced.pt` | `POSITIONING_CC_WEIGHTS` |
| `positioning.results_dir` | `results/positioning` | `POSITIONING_RESULTS_DIR` |
| `logs_dir` | `logs` | `ANNOTATION_LOGS_DIR` |

Relative paths: `weights_root`, `medsam2.repo`, `positioning.results_dir` and `logs_dir`
resolve against the YAML file's folder (or `tools/annotation/` without a YAML); weight files
resolve against `weights_root`; relative paths in environment variables resolve against the
current directory. Unknown YAML keys are reported in the log.

## Running

```bash
python tools/annotation/app.py                   # launcher with the three tools
python tools/annotation/app.py --config my.yaml
python tools/annotation/app.py --print-config    # show the resolved paths and exit
```

The startup screen asks for an image or a folder for SAM / Manual mode. Opening a folder
scans it recursively and resumes after the last image that already has a label file.

## Output

**Save All to YOLO** writes `labels/<stem>.txt` next to each image, one line per polygon
instance:

```
<class_id> x1/W y1/H x2/W y2/H ...
```

Built-in classes map to YOLO ids `0` pectoral, `1` breast tissue, `2` nipple (the
repository's segmentation datasets use the same ids). An edited polygon is written as
drawn; otherwise the mask contours are exported.

## Keyboard and mouse

| Input | Action |
|-------|--------|
| `Ctrl+S` | Save all labels to YOLO |
| `Ctrl+Z` | Undo the most recent polygon instance |
| `Ctrl+M` | Back to the main menu (asks about unsaved work) |
| `←` / `→` | Previous / next image in the folder |
| `C` | Measure PNL (MLO) / CC depth |
| `B` / `P` | Box / Point mode (SAM) |
| `1` / `2` / `3` | Pectoral / Breast / Nipple (SAM) |
| `Ctrl+Enter` | Commit the polygon edit |
| `Del` | Delete the polygon being edited |
| `Esc` | Cancel an unfinished box / polygon |
| `Ctrl+0`, `Ctrl++`, `Ctrl+-`, `Ctrl`+wheel | Fit, zoom in, zoom out, zoom at cursor |
| Middle-drag | Pan |
| Right-drag | Window / level (brightness / contrast) |
| Double-click background | Fit view and reset window / level |

## Layout

```
tools/annotation/
├── app.py                  # entry point: startup launcher + mode controller
├── toolkit_config.py       # config: defaults, YAML, environment variables
├── annotation.example.yaml
├── requirements.txt
├── interactive/            # MedSAM2 inference stack (no Qt): model, prompts, post-processing, pipeline
├── qt_app/
│   ├── main_window.py      # StartupDialog + SAM / Manual MainWindow
│   ├── canvas.py           # zoomable QGraphicsView: boxes, points, polygons, W/L, overlay
│   ├── session.py          # in-memory annotation store + YOLO export
│   ├── worker.py           # QThread for SAM inference (latest request wins)
│   ├── detector.py         # YOLO box detector for Manual mode
│   ├── pnl_analysis.py     # masks -> breast_seg.geometry -> overlay data
│   ├── image_io.py         # PNG / JPEG / DICOM loading (VOI LUT, MONOCHROME1)
│   ├── theme.py, logging_setup.py
│   ├── positioning_launcher.py
│   └── positioning/        # Breast Positioning tool (pose models, analysis, GUI)
├── docs/                   # per-mode guides, prompting details, roadmap
├── experiments/            # MedSAM2 research scripts (fine-tune, zero-shot), see its README
└── tests/                  # offscreen tests, no weights needed
```

## Tests and lint

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tools/annotation/tests
ruff check tools/annotation
```

The tests hide CUDA, use an empty temporary weights folder and run Qt offscreen: they
check the configuration rules, that the PNL overlay equals `breast_seg.geometry`, and that
every window builds and reports missing weights.

## Logs

A rotating log (`qt_app.log`, 2 MB × 4) is written to `logs_dir` (default
`tools/annotation/logs/`): resolved configuration, image loads, detector runs, inference
errors and save reports.

## Research scripts

[experiments/](experiments/README.md) holds the MedSAM2 data preparation, fine-tuning,
zero-shot evaluation and a Gradio browser prototype of SAM mode. They are not needed to
run the toolkit.
