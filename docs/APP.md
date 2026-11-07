# Mammography Positioning Review — the web app

Upload a study, get ΔD for each breast computed two independent ways, and review, correct and report it in the
browser.

- **Landmark route** — YOLO26-pose: nipple + pectoral top/bottom on MLO, nipple on CC. The PNL is the perpendicular
  from the nipple to the line through the two pectoral points; on CC the nipple-to-chest-wall distance.
- **Segmentation route** — YOLO26m-seg: pectoral / breast tissue / nipple masks. The pectoral line is fitted to the
  mask boundary, the nipple is the mask centroid.

Both routes work in the same 640 px model frame at the same millimetre scale, so their outputs are directly
comparable — **their disagreement is the signal** that sends a study to a human. Beyond ΔD the app grades the
criteria a radiographer checks (pectoral angle and depth, inframammary fold, nipple in profile, CC centring,
left/right symmetry, focus, exposure, and an *estimated* compression pressure that never counts towards the score),
every contour / point / lesion box is editable, and a written report is generated from the numbers in English or
Turkish.

> Research software. Not a medical device; every result must be checked by a reader.

## Install and run

```bash
pip install -e ".[models,dicom,app]"          # torch + ultralytics, DICOM codecs, FastAPI + uvicorn
cp configs/app.example.yaml configs/app.yaml  # configs/app.yaml is gitignored
```

From Python:

```python
from breast_seg.apps.positioning import load_app_config, serve

config = load_app_config("configs/app.yaml", overrides=["models.weights_root=/path/to/weights", "models.device=cpu"])
serve(config)                       # host / port from config.server, or serve(config, host=..., port=...)
```

With an ASGI server (the factory reads `$BREAST_SEG_APP_CONFIG`; defaults when unset):

```bash
BREAST_SEG_APP_CONFIG=configs/app.yaml \
  uvicorn --factory breast_seg.apps.positioning.api:app_from_env --host 127.0.0.1 --port 8000
```

`create_app(config, engine=None, loader=None)` returns the FastAPI application itself, for embedding or tests.

Models load in a background thread at start-up: the page shows a splash screen that ticks off the landmark,
segmentation and lesion models as they arrive, and `/api/health` answers immediately (`"loading": true` until
done). A model that fails to load is reported there, never raised — the service still starts.

The server binds to `127.0.0.1` by default. Expose it only behind an authenticated tunnel or reverse proxy; real
patient data should never travel over a public URL.

## Configuration (`configs/app.example.yaml`)

Relative paths are resolved against the directory of the config file. Every key is optional.

| Key | Default | Meaning |
|---|---|---|
| `models.registry` | `models.yaml` | model registry (see below) |
| `models.weights_root` | `null` | weights folder; `null` = `$BREAST_SEG_WEIGHTS`, else the registry's own `weights_root` |
| `models.device` | `cpu` | Ultralytics device: `cpu`, `0`, `cuda:1`, … |
| `storage.data_dir` | `../data` | where study renders and label JSON are kept |
| `storage.retention_days` | `30` | studies older than this are deleted automatically; `0` keeps them |
| `storage.purge_every_s` | `3600` | how often requests may trigger the retention purge |
| `upload.max_file_mb` | `220` | larger files are skipped with a note |
| `upload.max_files` | `40` | per upload (a study folder carries more than four files) |
| `display.preview_size` | `1280` | size of the stored display render; measurements always use the 640 frame |
| `rules.threshold_mm` | `10` | ΔD at or below this is adequate (editable per study in the UI) |
| `rules.severe_mm` | `15` | beyond this the pair is clearly inadequate |
| `criteria.*` | see file | `{good, acceptable}` bands per criterion; omitted keys keep their default |
| `server.host` / `server.port` | `127.0.0.1` / `8000` | used by `serve()` |
| `title` | `Mammography Positioning Review` | page title |
| `language` | `en` | default interface and report language: `en` or `tr` |

`load_app_config(path, overrides)` accepts `dotted.key=value` overrides (values parsed as YAML), e.g.
`storage.retention_days=7` or `criteria.sharpness={good: 900, acceptable: 400}`.

The focus and exposure bands were calibrated on one screening cohort; re-measure them for a different vendor mix.

## Model registry (`configs/models.yaml`)

The registry is the only place a weight path, its preprocessing recipe and its output type are written; nothing
else in the code names a checkpoint.

```yaml
weights_root: ..                     # optional, relative to this file
models:
  - id: pose_mlo
    task: landmark                   # landmark | segmentation | detection | prompt_segmentation
    view: MLO                        # MLO | CC | both
    recipe: wavelet640               # wavelet640 | baseline640 | raw1024
    weights: pose_weights/mlo-yolo26-pose-advanced.pt
    keypoints: [nipple, pectoral_top, pectoral_bottom]
  - id: seg_v4
    task: segmentation
    recipe: baseline640
    weights: runs/breast_seg_v4_yolo26m/weights/best.pt
    classes: [pectoral, breast-tissue, nipple]
    conf: 0.25
    iou: 0.5
    nipple_rescue: {enabled: true, conf: 0.02, max_mm: 15.0}
```

The engine fills four roles with the **first enabled** entry of each kind: a landmark model for MLO, one for CC, one
segmentation model and an optional lesion detector. To serve `seg_v5` instead of `seg_v4`, flip their `enabled`
flags. `medsam2` (box-prompted segmentation) is declared but disabled — it is not served by this app yet.

*Nipple rescue*: the nipple is the class the segmentation model under-scores. When the normal pass finds none, a
second pass at `nipple_rescue.conf` is accepted only if its centroid lies within `max_mm` of the pose model's nipple.
Rescued contours carry `"rescued": true` and a badge in the UI.

## Weights folder

Weights are not in the repository. Every `weights:` entry is relative to the *weights root*, resolved in this order:

1. `models.weights_root` of the app config (or a `models.weights_root=…` override),
2. the `BREAST_SEG_WEIGHTS` environment variable,
3. `weights_root` in the registry (`../weights` = the git-ignored `weights/` folder of the repository),

so a committed file never contains a machine path. With the shipped registry the root must look like this. The
annotation toolkit (`tools/annotation`) reads the same folder with the same sub-paths, so one copy serves both:

```
<weights root>/
  positioning/
    mlo-yolo26-pose-advanced.pt          pose_mlo     (required)
    cc-yolo26-pose-advanced.pt           pose_cc      (required)
  segmentation/
    breast_seg_v4_yolo26m.pt             seg_v4       (required)
    yolo26m_seg_v5.pt                    seg_v5       (optional, disabled by default)
  detector/
    manual_detector.pt                   lesion_det   (optional)
  medsam2/
    medsam_model_best.pth                medsam2      (annotation toolkit; disabled in the app)
```

`breast-seg train` + `export` write the segmentation weights under these names, and `breast-seg models` lists every
entry with whether its file was found.

Any other layout works by editing the `weights:` entries. `/api/health` lists each model with its relative path and
whether the file was found.

## What is stored

Uploaded DICOMs are parsed in memory and **never written to disk** — the multipart body is read straight into
memory (no temporary spool files). Only the following is kept:

```
<data_dir>/studies/<study id>/
  R-MLO.png  L-MLO.png  R-CC.png  L-CC.png    display renders (preview_size px)
  study.json                                   labels, measurements, criteria, verdicts
```

No patient name, ID, accession, UID or date is read from the header — only view, laterality, pixel spacing,
vendor and acquisition physics (compression force, thickness, angles, dose). The view and side are resolved from
`ViewPosition` / `ImageLaterality`, then `ViewCodeSequence`, then descriptive text fields, then the folder and file
name of the upload (`case/RMLO/IM0001.dcm` works); a view taken from a name is flagged in the UI.

## HTTP API

| | |
|---|---|
| `GET /api/health` | model status and roles, rules, storage usage, client defaults |
| `GET /api/studies` | recent studies |
| `POST /api/studies` | multipart `files` (DICOMs or a study folder) → study record |
| `DELETE /api/studies` | delete every stored study |
| `GET /api/studies/{id}` | full study record |
| `GET /api/studies/{id}/labels` | coordinates only, in the 640 frame |
| `PUT /api/studies/{id}/labels` | replace edited labels, recompute every measurement |
| `PUT /api/studies/{id}/rules` | move the ΔD threshold, re-decide every verdict |
| `GET /api/studies/{id}/report?lang=en\|tr` | written report (sections + plain text) |
| `GET /api/studies/{id}/image/{view}.png` | display render |
| `DELETE /api/studies/{id}` | delete one study |

All labels (`landmarks`, `polygons`, `lesions`, measurement endpoints) are in the 640 × 640 model frame; each image
entry carries `mm_per_px` for that frame.

## The interface

A build-free single-page client (ES modules, no framework, no npm) on Tabler with the *Clinical Light* theme:
navy bar, paper background, teal accent, near-black image viewports. English by default, Turkish with the EN/TR
switch (the choice is remembered per browser; the report has its own language switch).

- **Landing** — drop files or a study folder; staged progress: upload (bytes), server analysis, view loading.
- **Review** — the four views in the reading hanging protocol (right breast on the reader's left, MLO above CC)
  with each breast's verdicts on its own side, layer toggles, 1×1 / 2×1 / 1×2 / 2×2 layouts paired by view or by
  breast, synchronised zoom and pan, window / contrast / invert, the criteria drawer, upload notes and the report.
- **Editor** — every contour, landmark and lesion box is editable (vertex drag / insert / delete, polygon and box
  drawing, point placement, undo, copy / paste of boxes). Saving recomputes both routes from the edited shapes.
- **Viewer** — any tile opens full screen with zoom and pan.

A study id in the URL fragment (`/#<id>`) opens that study.

## Using the engine without the app

```python
from breast_seg.imaging import read_dicom
from breast_seg.inference import LandmarkEngine
from breast_seg.registry import Registry

engine = LandmarkEngine(Registry.load("configs/models.yaml"), device="cpu").load()
pixels, meta = read_dicom(open("L-MLO.dcm", "rb").read(), "L-MLO.dcm")
analysis = engine.analyse(pixels, meta)            # renders, landmarks, masks, polygons, lesions — 640 frame
record = engine.analyse_landmarks(pixels, meta)    # landmark record in ORIGINAL pixel coordinates
```

`analyse_landmarks` returns `{"points": {name: [x, y]}, "conf", "polygons": {class: {"p": [[x, y], …], "conf",
"rescued"}}, "chest_side", "lat", "view", "W", "H"}` — the per-image record a cross-view matcher stores.
`breast_seg.positioning` holds the measurement, criteria and report logic (`PositioningAssessor`,
`CriteriaGrader`, `build_report`) and works on any engine's output.

## Development

- `breast_seg.imaging.phantom` builds synthetic mammogram phantoms and in-memory DICOMs (no patient data);
  `breast_seg.inference.testing.fake_loader()` provides weight-free stand-ins for the models. Together they run the
  whole app on a laptop: `create_app(config, loader=fake_loader())` after `placeholder_weights(registry, root)`.
- `pytest` runs everything on CPU without weights. Tests marked `weights` need `BREAST_SEG_WEIGHTS` and are skipped
  otherwise. Setting `BREAST_SEG_REFERENCE_PREPROCESS` to the folder of the training-time reference preprocessing
  (`positioning_infer.py`) enables the byte-for-byte parity tests of the 640 recipes.
