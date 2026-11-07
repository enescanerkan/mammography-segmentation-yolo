# Mammography Toolkit

A PyQt5 desktop toolkit for mammography image annotation and positioning
analysis. Three tools share one window family, one theme, and one save
format.

```
medsam2_experiments/
├── medsam2_qt_demo.py        # entry point — launches the toolkit
├── interactive/              # SAM2 inference stack (reused by SAM Label)
├── qt_app/                   # PyQt5 GUI + supporting modules
│   ├── main_window.py        # SAM & Manual annotation MainWindow + StartupDialog
│   ├── canvas.py             # zoomable QGraphicsView (W/L, polygon edit, box)
│   ├── session.py            # in-memory annotation store + YOLO export
│   ├── worker.py             # QThread runner for SAM inference
│   ├── detector.py           # YOLO box detector (Manual auto-label)
│   ├── image_io.py           # PNG/JPEG/DICOM loader + W/L apply
│   ├── theme.py              # medical dark palette + QSS
│   ├── logging_setup.py      # rolling file + console logger
│   ├── positioning_launcher.py # in-process launcher for Positioning tool
│   ├── positioning/          # Breast Positioning sub-app (ported)
│   └── weights/manual_detector.pt   # YOLO box detector weights
└── docs/                     # tool-specific READMEs
    ├── README.md             # ← you are here
    ├── sam_label.md
    ├── manual_label.md
    └── breast_positioning.md
```

## Three tools, one launcher

The toolkit opens with a maximized startup screen showing three large
launcher cards. Pick one — the others stay one menu away (File → Back to
Main Menu) so you can switch between tools without restarting the app.

| Tool | What it does | See |
|------|--------------|-----|
| **SAM Label** | Box + point prompts → MedSAM2 mask. Polygon-edit to refine. | [sam_label.md](sam_label.md) |
| **Manual Label** | Polygon / box labeling with a bundled YOLO auto-detector. User-defined classes. | [manual_label.md](manual_label.md) |
| **Breast Positioning** | YOLO-pose landmark detection on MLO + CC views, with comparison. | [breast_positioning.md](breast_positioning.md) |

## Architecture at a glance

The codebase follows a layered design with single-responsibility modules
per concern:

- **inference stack** ([interactive/](../interactive)) — SAM2 service, prompt
  state, post-processing, pipeline. Reusable; no Qt imports.
- **canvas** ([qt_app/canvas.py](../qt_app/canvas.py)) — owns *visual* state
  for one image: pixmap, masks, points, boxes, polygons, W/L. Emits
  signals when the user mutates state.
- **session** ([qt_app/session.py](../qt_app/session.py)) — owns *persistent*
  state per image: masks, prompts, polygon instances. YOLO export lives
  here.
- **worker** ([qt_app/worker.py](../qt_app/worker.py)) — single-threaded
  inference runner with latest-only request coalescing.
- **main window** ([qt_app/main_window.py](../qt_app/main_window.py)) —
  orchestrates the above. Knows nothing about how SAM works or how a
  polygon is rasterized; delegates to the focused modules.

This separation means each piece is independently testable
(`_qt_smoke_test.py` exercises canvas + session + YOLO export without
ever loading SAM weights).

## Common controls (all tools)

- **Ctrl + S** save all labels to YOLO format
- **Ctrl + Z** undo most-recent polygon instance
- **Ctrl + M** back to main menu (with unsaved-changes prompt)
- **Ctrl + 0 / + / −** fit / zoom in / zoom out
- **Left / Right arrow** previous / next image in folder
- **Right-drag** window / level (brightness / contrast)
- **Middle-drag** pan
- **Double-click image** fit view + reset W/L
- **Ctrl + Enter** commit current polygon edit
- **Esc** cancel in-progress draw

The full list is also accessible via the **?** button on the startup
screen.

## YOLO output format

For every annotated image, a sibling `labels/<stem>.txt` file is created
on Save. Each detected/drawn instance becomes one line:

```
<class_id> x1/W y1/H x2/W y2/H x3/W y3/H ...
```

Polygon vertices are normalized to the image size. Polygon edits take
precedence over the rasterized mask, so the saved coordinates match what
the user sees on canvas.

## Installation

Use the conda environment described in the project root README. Required
packages: `PyQt5`, `ultralytics` (for Manual auto-detector and
Positioning), `pydicom` (for `.dcm` support), `opencv-python`, `numpy`,
`torch`.

```powershell
conda activate medsam2
cd medsam2_experiments
python medsam2_qt_demo.py
```

## Logs

Rolling logs land under `medsam2_experiments/logs/qt_app.log` (2 MB ×
4 backups). Each tool writes detailed events: image loads, detector
runs, SAM inference outcomes, save reports.
