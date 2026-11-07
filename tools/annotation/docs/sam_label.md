# SAM Label

MedSAM2-assisted annotation. You draw a bounding box (or click prompt
points), the fine-tuned SAM2 model returns a binary mask, and you can
refine the result with polygon-edit corner handles.

Pick this tool from the toolkit's startup screen. Loading the SAM
weights takes a few seconds — a loading screen runs in place of the
cards until the model is ready.

## When to use it

- You have a small set of unlabeled mammograms and want a quick mask
  per tissue type with one click + one drag.
- You want the polygon to come from a deep-learning model, not your
  cursor.

If you already have weights for a region-specific YOLO detector,
Manual Label's auto-detector may be faster — try both.

## Workflow

1. **Open**: pick Single Image or Folder.
2. **Pick a tissue class**: Pectoral Muscle, Breast Tissue, or Nipple.
   The active class drives mask color and class-competition.
3. **Pick a drawing mode**: Bounding Box (default — best for the fine-
   tuned model) or Point only.
4. **Draw**: in Box mode click two corners. Live rubber-band preview
   between clicks; corner handles after release. The mask appears
   instantly under the box.
5. **Refine**: drag corners to adjust the box (re-runs SAM). Add
   positive points (Left-click) or ignore points (Right-click) inside
   the existing box for fine-grained edits.
6. **Polygon edit**: when SAM produces a noisy boundary, press
   **Edit Mask as Polygon**. The contour is converted to a draggable
   polygon — pull each vertex to refine, click on an edge to insert a
   new vertex, Alt+click to delete one. **Commit** locks the polygon
   as the saved label.
7. **Move to next image** with the Next button or `→`.
8. **Save**: 💾 Save All to YOLO writes one `<stem>.txt` per image
   under `<folder>/labels/`.

## Class IDs

The SAM Label class IDs match `seg-dataset/data.yaml`:

| Tissue | Mask pixel id | YOLO class id |
|--------|--------------|---------------|
| Pectoral Muscle | 1 | 0 |
| Breast Tissue | 2 | 1 |
| Nipple | 3 | 2 |

## Settings panel

- **Ignore radius** — radius of the disk that ignore points carve out
  of the mask (mechanical subtract on top of the SAM negative).
- **Implicit box** — when you only have positive points, derives a
  bounding box from their hull and feeds it to SAM so the prompt
  graph matches the fine-tune distribution.
- **Class competition** — when drawing Breast Tissue *after* you've
  already drawn Pectoral, subtracts the pectoral mask from the breast
  prediction (the most common false-positive pattern on MLO views).
- **Soft negative** — also feed ignore points to SAM as negative
  prompts (belt-and-suspenders with the hard subtract).

## Tips

- Right-drag the canvas to window/level the image (DICOM-style W/L).
  Double-click the background to reset zoom + W/L.
- `B` / `P` toggles between Bounding Box and Point modes. `1` `2` `3`
  jump to Pectoral / Breast / Nipple respectively.
- `Ctrl + Z` removes your most recently-committed polygon. Re-running
  SAM with a new prompt automatically discards any user-edited polygon
  (the new mask is now the source of truth).
