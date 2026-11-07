# SAM mode: prompts, pipeline and design notes

How SAM Label turns clicks into a mask: what each prompt means, the forward path through
the model and the post-processing applied afterwards. The code lives in
[`interactive/`](../interactive); the Qt side only collects prompts and displays results.
The Gradio prototype in [`experiments/`](../experiments) uses the same stack.

## Overview

```
                ┌─────────────────────────────────────────────────────────┐
   image (RGB) ►│  MedSAM2Service                                          │
                │   ├── SAM2 backbone (Hiera-Tiny, frozen prompt encoder)  │
                │   └── fine-tuned mask decoder (medsam2.checkpoint)       │
                └────────────┬────────────────────────────────────────────┘
                             │  set_image() → cached image embeddings
   prompts:                  │
   ├─ box (xyxy) ────────────┤
   ├─ positive points (1) ───┤   predict() → raw mask (H×W uint8) + score
   └─ ignore points (0) ─────┤
                             ▼
                ┌─────────────────────────────────────────────────────────┐
                │  InferencePipeline.run(state, tissue, settings)          │
                │   1. resolve box (explicit | implicit from points)       │
                │   2. service.predict(box=..., points=...)                │
                │   3. fit_to_image (nearest-neighbour resize)             │
                │   4. ClassCompetition (breast tissue only)               │
                │   5. HardIgnoreMask (disk of radius R per ignore point)  │
                └────────────┬────────────────────────────────────────────┘
                             ▼
                       PredictionResult (mask + score + notes)
```

In the Qt app the pipeline runs on `qt_app/worker.py`'s thread; when several prompt
changes arrive while a prediction is running, only the latest one is computed next.

## The model is class-agnostic

Selecting *Nipple* does not make the model look for a nipple. There is one fine-tuned mask
decoder and it receives no class input; its only inputs are `(image, box, points)`. The
selected class affects only:

1. the overlay color and the YOLO class id of the result, and
2. the class-competition rule (which previously drawn mask is subtracted, see below).

The prompt *location* decides what is segmented: a box around the nipple gives the nipple,
a box around the breast gives the breast. A jointly trained multi-class decoder would
remove this dependency (see [roadmap.md](roadmap.md)).

## Prompt types

### Positive point

- Meaning: "the object under this pixel is the one I want".
- Qt: left-click in **Point** mode. Sent to the prompt encoder as `(x, y, label=1)`.
- A point is a single pixel; its effect spreads through the model's attention, so there is
  no fixed "area" per point. Several positive points are read as parts of the same object.
- The fine-tuned checkpoint was trained with **box prompts only**
  (`interactive/wrapped_model.py`), so point-only prompts are out of distribution. With the
  **Implicit box** setting on (default), a box is derived from the points (below).

### Ignore point (hard)

- Meaning: "nothing around this point belongs to the mask".
- Qt: right-click in Box or Point mode. Drawn as a red disk of radius R.
- Applied twice:
  1. **Soft prior**: sent to SAM2 as a negative point `(x, y, label=0)` when
     **Soft negative** is on. On its own this does not guarantee exclusion.
  2. **Hard subtraction**: after the model returns, a disk of radius R around each ignore
     point is set to 0 (`mask[disk(x, y, R)] = 0`), whatever the model predicted.
- Area removed per point: `round(π·R²)` pixels, e.g. R = 10 → 314 px, R = 20 → 1,257 px,
  R = 50 → 7,854 px.

### Bounding box

- Meaning: "the object is inside this rectangle".
- Qt: **Box** mode, two clicks on opposite corners (a third click starts a new box);
  corners stay draggable.
- Sent as two corner tokens `[(x0, y0, label=2), (x1, y1, label=3)]`, the format the
  fine-tune was trained with. Only axis-aligned boxes are supported.

### Combinations

| Prompts | Behaviour |
|---------|-----------|
| Box | Most stable; matches the fine-tune distribution |
| Positive point(s) | Implicit box derived, then box + points |
| Box + positive points | Box sets the geometry, points refine |
| Box + ignore points | Box prediction, then disks subtracted |
| Ignore points only | No positive prompt, so the model is given only negatives; add a box or a positive point |

## Forward path

1. **Image → embeddings (once per image).** `MedSAM2Service.set_image` runs the Hiera-Tiny
   image encoder and caches the embeddings; later prompts on the same image reuse them.
   The cache is keyed on the image array itself (a held reference, not `id()`), so a new
   image can never reuse stale embeddings.
2. **Prompt encoding.** Box → two corner tokens; positive / negative points → tokens with
   labels 1 / 0. The prompt encoder was frozen during the fine-tune.
3. **Mask decoder.** Image + prompt embeddings → low-resolution mask logits. This decoder
   is the part fine-tuned on mammograms.
4. **Mask post-processing (model side).** Sigmoid, upsampling, threshold 0.5; if the
   output size differs from the image it is resized with nearest neighbour
   (`MedSAM2Service.fit_to_image`). With exactly one positive point, no ignore point and no
   box (i.e. Implicit box off) the model returns several candidate masks and the
   highest-scoring one is used.
5. **Pipeline post-processing.** Class competition (breast only, when enabled), then the
   hard ignore disks.
6. **Display.** The canvas draws each class's mask in its color with an adjustable
   opacity, and the prompts on top.

## Implicit box

Problem: a decoder trained only on `(image, box) → mask` behaves inconsistently with pure
point prompts, increasingly so as points are added.

Fix: when no box is drawn but positive points exist, `InferencePipeline` sends

```python
implicit_box = bbox(positive_points) padded by 64 px   # implicit_box_pad_px
```

together with the points. The decoder therefore always sees a box. Turning **Implicit
box** off restores pure point prompting, which is useful for comparison.

## Class competition (breast vs. pectoral on MLO)

Observation: the fine-tune is per-class binary (one `(image, box, mask)` sample per class,
see `experiments/prepare_medsam2_data.py`). On CC views the pectoral muscle is small or
absent, so breast predictions are clean; on MLO views the breast mask tends to leak into
the pectoral muscle because the decoder does not know the pectoral is there.

Rule: draw the **pectoral muscle first**. When you then segment **breast tissue** with
class competition on, the pectoral pixels you drew are subtracted:
`breast_final = breast_pred AND NOT pectoral_drawn`.

- No extra forward pass: it is a set difference with your own mask.
- The subtraction uses a mask you have checked yourself. (An earlier version predicted the
  pectoral automatically; with a class-agnostic decoder that sometimes segmented the breast
  itself and emptied the result.)
- Without a pectoral mask (typically CC) it is a no-op. A badly drawn pectoral gives a bad
  subtraction; clear that class and redraw it.

## Per-class state

Each image keeps, per class, the last prompt state, the mask and any polygon instances
(`qt_app/session.py`). Switching the class keeps every stored mask on screen and restores
that class's prompts; **Clear Tissue** resets only the active class, **Clear All** the
whole image. A new prediction for a class replaces a user-edited polygon of that class.

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `interactive/model.py` | Model loading (explicit checkpoint) + raw predict |
| `interactive/wrapped_model.py` | MedSAM2 wrapper matching the upstream fine-tune forward |
| `interactive/prompts.py` | Prompt state, tissue presets, drawing helpers |
| `interactive/postprocess.py` | `HardIgnoreMask`, `ClassCompetition` |
| `interactive/inference.py` | Composition: model + post-processing |
| `qt_app/worker.py` | Background thread, latest-request coalescing |
| `qt_app/main_window.py`, `qt_app/canvas.py` | Collect prompts, show masks, polygon edit |

New post-processing steps (e.g. morphological smoothing) belong in `postprocess.py` and are
composed in `InferencePipeline`; the UI does not need to change.

## Performance notes

- The image encoder runs once per image; later prompts on that image are cheap.
- CUDA is used when available (`medsam2.device: auto`); on CPU a prediction takes seconds.

## Manual acceptance check

1. Start the toolkit and open **SAM Label** on a CC image.
2. Class = Breast Tissue, Point mode, one click in the middle of the breast: the mask
   should cover the breast (implicit box).
3. Right-click outside the breast: a hole of the disk size appears where the mask touched
   it; increase **Ignore radius** and the hole grows.
4. On an MLO image draw the pectoral muscle first, then Breast Tissue with class
   competition on: the breast mask stops at the pectoral edge. Turn class competition off
   to see the leak.
5. Box mode around the breast gives the most stable mask.

If these fail, check `--print-config` (is `medsam2.checkpoint` the intended run?) and the
log for MedSAM2 import errors.
