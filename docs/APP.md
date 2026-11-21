# Mammography Positioning Review — the web app

Upload a study, get ΔD for each breast computed two independent ways, and review, correct and report it in the
browser.

- **Landmark route** — YOLO26-pose: nipple + pectoral top/bottom on MLO, nipple on CC. The PNL is the perpendicular
  from the nipple to the line through the two pectoral points; on CC the nipple-to-chest-wall distance.
- **Segmentation route** — YOLO26m-seg: pectoral / breast tissue / nipple masks. The pectoral line is fitted to the
  mask boundary, the nipple is the mask centroid.

Both routes work in the same 640 px model frame at the same millimetre scale, so their outputs are directly
comparable — **their disagreement is the signal** that sends a study to a human. Beyond ΔD the app grades one
item on the MLO alone, the **MLO quality** (the criterion of the MLO-only RAUNet paper): the foot of the
perpendicular from the nipple — the mask centroid the PNL is measured from, the pose point only when there is no
nipple mask — to the fitted pectoral line
taken as an infinite line must fall inside the original image widened by 0.25% of its longest side. Inside is
good (green), outside poor (red), n/a when the pectoral line or the nipple is missing. It is shown in the criteria
drawer, under each breast's verdicts and in the report, and it never changes the ΔD verdicts; the app states the
quality and does not advise a repeat. The line and the perpendicular it was judged from are drawn with the
segmentation measurement layer, also when that route measured no distance (no nipple mask). Three values are shown as information only, with no grade or colour: the
pectoral angle (MLO), how far the nipple sits from the middle of the breast's height on the CC (|nipple row − middle
of the first and last breast-tissue rows| in mm, and as a share of the half-height), and the gantry angle from the
header. Compression, dose, focus, exposure, nipple profile, pectoral on CC, pectoral reaching the nipple level,
inframammary fold, left/right symmetry, the 0–100 criterion score and the mass / calcification boxes were
removed on 2026-11-20. Every contour and point is editable, and a written report is generated from the numbers in
English or Turkish.

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

## Windows desktop app

`packaging/windows` turns the app into a Windows program that needs no Python, no server and no internet. GitHub
Actions (`.github/workflows/windows-app.yml`) builds it on every push to `main` that touches the app and publishes
the zip as the pre-release [`dev-build`](https://github.com/enescanerkan/mammography-segmentation-yolo/releases/tag/dev-build)
(same link every time); a `v*` tag publishes a release of that version. Each build starts the finished executable
with `--self-test` (models loaded, DICOM decoders present, a synthetic study decoded plain and as JPEG 2000 and
analysed) before anything is published.

* **Running it.** Unzip `MammographyPositioning-windows-x64.zip` anywhere and start `MammographyPositioning.exe`.
  The app runs on `127.0.0.1` (port 8765, or a free one) inside the process and is shown in a native window
  (Microsoft Edge WebView2, part of Windows 10 / 11). Without WebView2 it opens the default browser and shows a
  small window with a *Quit* button. `MammographyPositioning.exe --browser` always uses the browser.
* **What it contains.** Python, the libraries in `constraints.txt` (CPU build of PyTorch), the client, the model
  registry and the three weights of the `weights-v1` release. Models whose weights are not bundled (lesion
  detector, MedSAM2) are disabled in the bundled registry.
* **User files** in `%LOCALAPPDATA%\MammographyPositioning`:
  * `app.yaml` (optional): any key of `configs/app.example.yaml`, e.g. `language: tr` or
    `rules: {threshold_mm: 12}`; it overrides the desktop defaults key by key;
  * `data\`: study renders and label JSON (DICOM files are never written), deleted after `storage.retention_days`;
  * `logs\app.log`: the server log, the first place to look when something fails.
* **Building it yourself** (on Windows, Python 3.11):

  ```bat
  pip install -c constraints.txt ".[app,models,dicom,desktop]" "pyinstaller>=6.6,<7"
  python packaging\weights.py build\desktop\weights
  python -m PyInstaller packaging\windows\positioning.spec --distpath dist --workpath build\pyinstaller --noconfirm
  dist\MammographyPositioning\MammographyPositioning.exe --self-test
  ```

  The spec also builds on Linux (without the splash screen when Tk is missing), which is how the bundle is tested
  outside Windows.

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
| `nipple.brighten` | `true` | the brightened nipple re-search, for this app's engine only (see "Model registry") |
| `nipple.window_mm` | `50` | side of the window it re-windows around the anterior tip |
| `nipple.max_gap_mm` | `null` | optional adjacency check of a brightened nipple (off unless a distance is given) |
| `nipple.contour` | `true` | no nipple found at all: the segmentation route is estimated at the breast contour |
| `criteria` | – | not a setting any more (no criterion has a band); a section with the retired keys (e.g. `pectoral_below_nipple_mm`) still loads and is ignored |
| `server.host` / `server.port` | `127.0.0.1` / `8000` | used by `serve()` |
| `title` | `Mammography Positioning Review` | page title |
| `language` | `en` | default interface and report language: `en` or `tr` |

`load_app_config(path, overrides)` accepts `dotted.key=value` overrides (values parsed as YAML), e.g.
`storage.retention_days=7` or `rules.threshold_mm=12`.

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

The two steps below are the review app's own, not the models': they are switched in the app config (`nipple:`), and
the engine runs the brightened re-search only when its caller asks for it (`LandmarkEngine(...,
brighten=BrightenSearch(window_mm, max_gap_mm))`, which the app does with `nipple.brighten`). An engine built from
the registry alone, as a cross-view matcher builds it, finds exactly the nipples the two model passes find.

*Brightened re-search* (`nipple.brighten`): when both passes miss, typically on a dark image, the engine takes
the anterior tip of the segmented breast (MLO: the contour point farthest from the fitted pectoral line; CC: farthest
from the chest-wall edge), cuts a `window_mm` square around it from the original pixels, re-windows it linearly from
the background level to the 99.5th percentile of the breast pixels, pastes it back into the 640 canvas at the same
place and scale, and searches it again at `nipple_rescue.conf`. Only the segmentation model looks at the brightened
canvas; the pose model is not run on it and its nipple is not used as a check (it is often what is wrong on these
images). The most confident nipple whose centroid lies in the window is taken. The pass adds that contour and
nothing else: the breast-tissue and pectoral masks stay those of the normal pass. Such contours carry
`"brightened": true`, a "nipple found after brightening" badge on the view and a line in the report.

The variant was chosen on the VinDr validation split, before any test set was looked at, out of six (50 mm window
pasted back or 100 mm window enlarged to the whole input, each with a linear, gamma 0.5 or CLAHE window). On the
held-out sets, among images where the normal pass and the rescue both missed the nipple:

| Set | Images | Missed | Found after brightening | Within 10 mm of the reference | Farther |
|---|---|---|---|---|---|
| VinDr validation | 262 | 3 | 2 | 1 | 1 (12.0 mm) |
| VinDr test | 292 | 1 | 0 | – | – |
| CMMD (97 pairs) | 194 | 4 | 4 | 3 | 1 (20.8 mm) |
| EMBED (375 pairs) | 750 | 102 | 28 | 23 | 5 (14.9–29.0 mm) |

Run on every image where the nipple *was* found, the candidates the same search produced lay within 10 mm of the
app's nipple on 547 / 547 VinDr, 185 / 190 CMMD and 604 / 609 EMBED images. On half of the EMBED misses the reference
nipple coincides with the pose model's own point, so the EMBED distances partly measure agreement with that model.
A wrong brightened nipple usually makes the two ΔD routes disagree, which flags the study; the badge asks the reader
to check it either way.

Against the reference nipple *masks* (VinDr and EMBED have them, CMMD does not), which are areas rather than single
points:

| Set | Found after brightening | Centroid on the reference mask | Within 5 mm of it | Farther | Centroid to mask centroid, median (max) |
|---|---|---|---|---|---|
| VinDr validation | 2 | 0 | 2 (1.0 and 2.9 mm) | 0 | 6.2 mm (7.0) |
| EMBED | 28 | 15 | 20 | 8 (5.1–22.5 mm) | 4.0 mm (28.2) |
| – reference point is the pose model's own | 18 | 10 | 12 | 6 | 4.1 mm (28.2) |
| – reference point placed independently | 10 | 5 | 8 | 2 | 2.5 mm (11.0) |

This is the same reference seen as an area, not a second opinion: on 100 of the 102 missed EMBED images the
reference mask contains the reference point (and on most EMBED images where the app finds the nipple the reference
mask is the model's own contour, accepted by the annotator; the VinDr masks are drawn by hand).

*No nipple found: the contour estimate* (`nipple.contour`, default `true`). When the normal pass, the rescue
and the brightened re-search all miss, the segmentation route is not left unmeasured: it measures at the
breast-tissue contour point that reaches farthest — on an MLO the point farthest from the fitted pectoral line
taken as an infinite line, on a CC the point farthest from the chest wall (the frame edge on the chest side) — and
measures it exactly as it would measure a nipple there (`breast_seg.geometry.anterior_point`, the rule the
evaluation's `fallback` used; the brightened re-search finds its anterior tip with the same function). The result carries `"estimated": true, "nipple": "contour", "missing":
<reason>` and enters the ΔD verdict like any other; the breast's segmentation verdict lists the
views it was estimated on (`"estimated": ["MLO"]`). It is shown everywhere as an estimate: a "nipple estimated
(contour)" badge on the view, *est.* after the value under the image and on the card, a sentence under the card and
in Notes, a dashed distance from an open ring in the overlay, "(estimated)" after the number in the report and a
line naming the view. A nipple contour, once found or drawn by a reader, is always measured from instead; the landmark
route and the MLO quality are not changed (the MLO quality keeps the nipple mask's centroid, else the pose point).
`contour: false` leaves such a view unmeasured, as before. The setting is read when the app starts; stored studies
keep their measurements until a label edit or a threshold change recomputes them (the criteria are unchanged, so the
criteria version is not raised).

Why the contour and why last: on the 16 November benchmark (YOLO26m-seg v4, the app's model) the contour estimate
lifted the segmentation route's accuracy on the CMMD 97 pairs from 81.4 % (rescue alone) to 85.6 % (rescue +
contour) and on EMBED 203 from 81.3 % to 83.7 %. In this app's chain it comes after every search, so on the 96
public CMMD studies of the test folder (192 images) it is never reached: every nipple is found (22 of them by the
rescue or the brightened re-search). Run on those 22 images with the searches switched off, the contour point lay a
median 3.5 mm (at most 14.0 mm) from the measurement the found nipple gives, and 15 of the 19 breasts kept the same
segmentation verdict.

*Adjacency check, off by default* (`nipple.max_gap_mm`, not set): with a distance given, a brightened nipple
counts only if its mask lies within that distance of the breast-tissue mask, taking the normal pass and the
brightened pass together. It was evaluated at 3 mm, a value fixed on the VinDr validation split alone (with the
search forced on every image there, every candidate within 10 mm of the reference lay within 2.1 mm of the breast
mask), and it is off because it does not do what it was meant for: all 34 finds above pass it, the wrong finds
included, since they sit on the skin line at the wrong place (0–1.3 mm from the breast mask) and not out in the
background; and forced on every image of the four sets it would turn down 6 of 1 380 candidates, all on EMBED and all
on the reference nipple (the breast mask ends short of the skin line there). Measured against the normal-pass breast
mask alone it would also turn down one correct CMMD find, which lies 11.9 mm beyond that mask on a dark image.

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
  study.key                                    dataset studies only: the study key (a salted hash, see "The interface")
  study.unopened                               empty; there until the study is opened for review
  study.opened                                 empty; its time is when the study was last opened for review
  study.incomplete                             empty; a file failed in a way another attempt can change
<data_dir>/study_key_salt                      the installation's random salt for study keys (not a secret, see there)
```

`study.json` records the version of the criteria set it was graded with. A study saved under an older set (for
example with the retired pectoral-depth item) is regraded when it is opened: the criteria are rebuilt from the
stored measurements, the ΔD verdicts are left as they are and nothing is written back. The regraded record is kept
in memory until the stored file changes, so a study is regraded once, not on every request.

*One rounding.* The record holds every millimetre unrounded, in the per-view measurements and in the per-breast
summary alike, and a number is rounded once, where it is written: `formatNumber` in the page (image footer, cards,
overlay labels, drawer) and `positioning/numbers.py` on the server (report, English notes, criterion values), both
half away from zero on the exact value. A summary stored by an older version with two decimals gets its unrounded
values back from the per-view measurements when the study is regraded — it used to show 47.949 mm as 47.9 mm under
the image and 48.0 mm on the card.

No patient name, ID, accession, UID or date is read from the header — only view, laterality, pixel spacing,
vendor and acquisition physics (compression force, thickness, angles, dose). The view and side are resolved from
`ViewPosition` / `ImageLaterality`, then `ViewCodeSequence`, then descriptive text fields, then the folder and file
name of the upload (`case/RMLO/IM0001.dcm` works); a view taken from a name is flagged in the UI.

## HTTP API

| | |
|---|---|
| `GET /api/health` | model status and roles, rules, storage usage, client defaults, the study-key salt |
| `GET /api/studies?limit=N` | the `N` most recently written studies somebody opened (default 40, at most 200), the `total` stored and how many are `unopened`; `limit=0` only counts |
| `POST /api/studies` | multipart `files` (DICOMs or a study folder) → study record. Optional headers of a dataset upload: `X-Study-Key` (64 hex, stored with it), `X-Study-Unopened: 1` (analysed ahead: not listed until opened), `X-Study-Left-Out: n` (files not sent) |
| `POST /api/studies/lookup` | `{"keys": [...]}` → `{"found": {key: study id}}` for the study keys a stored study carries (incomplete, unedited studies are left out) |
| `POST /api/studies/{id}/opened` | the study is on screen for review: list it among the recent studies |
| `DELETE /api/studies?expected=N` | delete every stored study; refused (409) when `N` is not the number stored |
| `GET /api/studies/{id}` | full study record |
| `GET /api/studies/{id}/labels` | coordinates only, in the 640 frame |
| `PUT /api/studies/{id}/labels` | replace edited labels, recompute every measurement |
| `PUT /api/studies/{id}/rules` | move the ΔD threshold, re-decide every verdict |
| `POST /api/studies/{id}/clean` | `{"view_key", "polygons"}` → `{"view_key", "polygons", "changed"}`: the contours cleaned by the label-cleaning rules (the editor's "Smooth / clean labels") and the classes whose contour was replaced or removed (empty when the label maps are pixel-identical, or the contours are still those a clean returned); nothing is stored |
| `GET /api/studies/{id}/report?lang=en\|tr` | written report (sections + plain text) |
| `GET /api/studies/{id}/image/{view}.png` | display render |
| `DELETE /api/studies/{id}` | delete one study |

All labels (`landmarks`, `polygons`, `lesions`, measurement endpoints) are in the 640 × 640 model frame; each image
entry carries `mm_per_px` for that frame.

A refused request answers `{"detail": "<English sentence>"}`. The refusals a reader can meet also carry a `code`
and its `params`, which the page translates by (`service.ERRORS`, `ERRORS_TR` in i18n.js; an unknown code, or
English, shows the sentence): `study_not_found`, `image_not_found`, `models_loading`, `models_missing`, `no_files`,
`too_many_files` (`limit`), `clear_changed` (`stored`, `expected`), `only_dicom` (`names`: every file sent was a
PNG / JPG or another picture, which has no pixel spacing and no view or side, so no millimetre can be measured) and
`nothing_processed`, whose `params.files` lists every file of the upload with its own `text`, `code` and `params`
(`not_an_image`, `raster_image`, `not_dicom`, `undecodable`, `not_single_frame`, `view_unknown`, `side_unknown`,
`view_unsupported`, `file_too_large`, `duplicate_view`, `analysis_refused`, `inference_failed`). A stored study keeps
its upload notes as English sentences (`problems`) and, next to them, the same notes with their codes and parameters
(`problem_codes`), so the Notes panel and the report write them in the reader's language; a record stored before
the codes were kept shows its sentences as they are.

## The interface

A build-free single-page client (ES modules, no framework, no npm) on Tabler with the *Clinical Light* theme:
navy bar, teal accent, near-black image viewports. It opens in the dark theme; the theme button switches to light,
and that choice is remembered by the browser. Every start is in the configured `language` (English unless set); the
EN / TR switch, and the report's own language switch, hold for the browser session (a reload keeps them, a new
window or a new start of the desktop app does not).

- **Landing** — drop files or a study folder; staged progress: upload (bytes), server analysis, view loading.
  The page says what it could not send, by name: PNG / JPG pictures (this app reads DICOM only: a picture has no pixel
  spacing and no view, so no millimetre can be measured), empty files (0 bytes; the study still goes through without
  them, and a notice names them), and a file the browser can no longer read (moved, changed or unreadable after it
  was picked: the request then fails before it leaves the browser, which would otherwise look like "Could not reach
  the server"; the page reads one byte of each file to tell the two apart).
  **Open a dataset** picks a folder of many studies: the browser groups its DICOMs by the top-level
  StudyInstanceUID, read by walking each file's header elements (first 16 KB, up to 256 KB; sequences, including a
  UID nested in one, are skipped). The UID is never shown or sent; a file whose UID cannot be read (a header longer
  than that, a deflated data set) is grouped by folder. DICOM files that are not views of a study are left out and
  counted as skipped: a DICOMDIR, a data set with no study UID, a file with no pixel data. Each study is named after
  its folder; studies that share a folder also carry the name of their first file. Studies open one at a time in
  folder order. A bar under the top bar shows `Dataset <folder> [‹ Previous] [i / N · name] [Next ›] ✕`; the
  buttons, PageDown / Shift+→ and PageUp / Shift+← move between studies (the keys are ignored while typing or while
  the report, the editor, the viewer or a popover is open; the criteria drawer does not stop them), the list jumps
  anywhere; what the marks of the list mean is written in the bar (✓ opened, • analysed but not opened yet, ✕ could
  not be analysed, – removed). The next study is analysed in the background (• in the list until it is opened, ✓
  afterwards). A study
  that fails is marked ✕; Next and Previous step over it, and the dataset can be finished when the last one failed.
  A study with more files than `upload.max_files` sends the files whose path names a view first, and a notice says
  how many were not sent. Opening another folder replaces the open dataset only once the new one is known to hold a
  study. Deleting the shown study keeps the dataset and moves on; ✕, New study or uploading a single study leaves
  it. A study opened from the recent list or by its address while a dataset is open moves the bar to it when the
  dataset holds it, and leaves the dataset otherwise, so the bar never stays over another study. On the last study the Next button becomes Finish ✓, which reports how many studies were opened, analysed but
  not opened, failed, removed or not reached.

  *"Recent studies" lists what somebody opened.* A study analysed ahead is stored as not opened yet and is left out
  of the list until a page opens it for review, by any route (the dataset, the list, a study id in the address) or
  edits it; the server keeps that mark, not the page. The list is ordered by the later of when a study was last
  written and when it was last opened, so a study analysed ahead long ago and opened now comes first. The page never deletes a record on its own account: leaving
  the dataset or closing the page changes nothing on the server, an analysis still running simply finishes, and a
  record nobody opens goes with the retention window like any other. "Clear" counts and deletes these too, and says
  how many of them there are.

  *The same folder opened again finds its studies.* With each dataset study the browser sends a study key:
  HMAC-SHA-256, under a random salt this installation created for itself (`<data_dir>/study_key_salt`, handed to
  every page by `/api/health`), of the StudyInstanceUID with the number and total size of the study's files. When a
  folder is opened the page asks which of its keys the server holds and opens those studies from their stored
  records, corrections included, instead of uploading them again (• until opened in this session). A study whose
  files changed has another key and is analysed afresh; studies stored before the keys existed are simply not
  matched. A study stored from an incomplete upload — a file failed in a way another attempt can change (pixel data
  that could not be decoded, the models refusing or failing for the moment) — is not offered again unless a reviewer
  edited it: it is analysed afresh and the new record replaces the old one. A failure that gives the same result on
  every attempt (not DICOM, a picture, an unsupported view, a multi-frame file, a file over the size limit), the
  files the upload limit left out and the second file of a view do not make a study incomplete: it is found again by
  its key. Of two stored studies with the same key, the one created last (by its record's creation time) is opened.

  What the salt is for: the UID itself is not sent or stored, and the keys of two installations cannot be compared.
  What it is not: a secret from the clients of this server. Every page receives it, so anyone who can call the API
  and already knows a UID can compute its key and ask whether that study is stored; the API has no authentication
  and lists every stored study in any case. Put the app behind an authenticated tunnel if that matters.
- **Review** — the four views in the reading hanging protocol (right breast on the reader's left, MLO above CC)
  with each breast's verdicts on its own side, layer toggles, 1×1 / 2×1 / 1×2 / 2×2 layouts paired by view or by
  breast, window / contrast / invert, the criteria drawer, upload notes and the report. Only the views the study
  holds take a pane: a study of one breast (common in CMMD) fills the grid with its MLO and CC side by side, a single
  view fills it alone, and a page that would hold none is left out; a four-view grid missing one view keeps its
  hanging order and marks the missing one "not uploaded". Each view zooms (wheel) and pans (drag) on its own;
  "Synchronise zoom & pan" in the layout menu ties them together. The recent list names the views in this grid order.
- **Editor** — every contour and landmark is editable (vertex drag / insert / delete, polygon drawing, point
  placement, undo). The left button edits; the right or the middle button drags the image (a right click opens no
  menu). **Hide contours** (H) switches the breast, pectoral and nipple contours off and back on to show the image
  under them: it switches the review's own layer toggles, changes no label and marks nothing edited; while a contour
  is hidden it cannot be edited (its handles would be invisible), and the landmark points stay editable. **Smooth /
  clean labels** applies the label-cleaning rules (`breast_seg.positioning.label_cleanup`) to the view on screen:
  pectoral muscle and breast tissue keep their largest 8-connected region with holes filled; the nipple keeps the
  region touching (or nearest to) the breast's front edge; discarded pixels inside the foreground go to the nearest
  kept structure, outside it to the background; on an MLO the unlabelled tissue strip between pectoral and breast
  (within 25 px of both, image level above 15 / 255) goes to the nearest of the two; the edges are smoothed (Gaussian,
  sigma 3 px, nipple 2 px; arg-max) and the region rules run once more; on a CC pectoral pixels join the breast. The
  result is an unsaved edit (Undo takes it back). Whether anything changed is decided on label maps: when the
  cleaned contours paint the same pixels as the current ones, the view is left exactly as it was and not marked
  edited. The smoothing is not idempotent (run on its own result it takes off a few more edge pixels, 10–100 per
  view on the public CMMD test images), so the contours a clean returns carry a fingerprint of their points
  (`"cleaned"`, kept when saved): a second click on them changes nothing, before or after saving, and an edit of any
  contour makes the next click clean the view for real. The answer goes to the view it was asked for, also when the
  reader has moved to another view meanwhile (the hint names it); it is not applied over an edit made to that view
  while it was computed, and the hint says so. Saving recomputes both routes from the edited shapes.
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
"rescued"}}, "chest_side", "lat", "view", "W", "H"}` — the per-image record a cross-view matcher stores. The
brightened re-search is off unless asked for (`LandmarkEngine(registry, brighten=BrightenSearch())`); a nipple it
found carries `"brightened": true`.
`breast_seg.positioning` holds the measurement, criteria and report logic (`PositioningAssessor`,
`CriteriaGrader`, `build_report`) and works on any engine's output.

## Development

- `breast_seg.imaging.phantom` builds synthetic mammogram phantoms and in-memory DICOMs (no patient data);
  `breast_seg.inference.testing.fake_loader()` provides weight-free stand-ins for the models. Together they run the
  whole app on a laptop: `create_app(config, loader=fake_loader())` after `placeholder_weights(registry, root)`.
- `pytest` runs everything on CPU without weights. Tests marked `weights` need `BREAST_SEG_WEIGHTS` and are skipped
  otherwise. Setting `BREAST_SEG_REFERENCE_PREPROCESS` to the folder of the training-time reference preprocessing
  (`positioning_infer.py`) enables the byte-for-byte parity tests of the 640 recipes.
