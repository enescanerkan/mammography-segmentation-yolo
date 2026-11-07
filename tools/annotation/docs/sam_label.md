# SAM Label

MedSAM2-assisted annotation. You draw a bounding box (or click prompt points), the
fine-tuned SAM2 model returns a binary mask, and you refine the result with polygon-edit
handles.

Pick this tool on the startup screen. A loading message replaces the cards while the model
is built. If a required file is missing (MedSAM2 code, SAM2 base weights, fine-tuned
checkpoint), a dialog lists each missing path with the config key that sets it; see the
[configuration table](../README.md#configuration).

## When to use it

- Quick masks for pectoral muscle, breast tissue and nipple with one box per structure.
- Boundaries that should follow the image rather than your cursor.

For other label sets, or when a YOLO detector for the target already exists, Manual Label
is usually faster.

## Workflow

1. **Open** a single image or a folder.
2. **Pick a class**: Pectoral Muscle, Breast Tissue or Nipple. The class sets the overlay
   color, the YOLO id and the class-competition rule.
3. **Pick a mode**: Bounding Box (default, best for the box-trained model) or Point only.
4. **Draw**: in Box mode click two corners. A live rubber band follows the cursor between
   the clicks; the mask appears when the box is finished and the corners stay draggable.
5. **Refine**: drag the corners (the model re-runs), add positive points (left-click in
   Point mode) or ignore points (right-click in either mode).
6. **Polygon edit**: **Edit Mask as Polygon** converts the mask contour to a draggable
   polygon: drag a vertex, click an edge to insert one, `Alt`+click to delete one.
   **Commit** (`Ctrl+Enter`) locks it as the saved label.
7. **Next image** with the Next button or `→`.
8. **Save**: **Save All to YOLO** writes one `<stem>.txt` per image under `<folder>/labels/`.

## Class ids

| Tissue | Mask pixel id | YOLO class id |
|--------|---------------|---------------|
| Pectoral Muscle | 1 | 0 |
| Breast Tissue | 2 | 1 |
| Nipple | 3 | 2 |

## Settings panel

- **Ignore radius**: radius of the disk an ignore point removes from the mask (a hard
  subtraction on top of the model's negative point).
- **Implicit box**: with positive points only, a padded box around them is sent as well,
  matching the box-only fine-tune distribution.
- **Class competition**: when drawing Breast Tissue after Pectoral, the drawn pectoral
  mask is subtracted from the breast prediction (the common MLO leak).
- **Soft negative**: also send ignore points to the model as negative prompts.

How each prompt reaches the model is described in [prompting.md](prompting.md).

## Tips

- Right-drag the image for window / level; double-click the background to reset zoom and
  window / level.
- `B` / `P` switch Box / Point mode; `1` `2` `3` select Pectoral / Breast / Nipple.
- `Ctrl+Z` removes the most recent polygon. Running the model again with a new prompt
  discards a user-edited polygon for that class (the new mask becomes the label).
- `C` draws the PNL / CC-depth overlay from the current labels.
