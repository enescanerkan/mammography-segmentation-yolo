# Manual Label

Polygon and bounding-box labelling with an optional YOLO box detector. Unlike SAM Label,
Manual mode starts with an empty class list: **you** define the taxonomy, and the first
class you add becomes YOLO id 0.

## When to use it

- The label set is not `pectoral / breast / nipple` (for example `mass`, `calcification`,
  `lesion`).
- A YOLO detector for the target exists and you want one-click proposals to refine.
- Pure manual polygons without loading any model.

## Workflow

1. **Add a class**: **＋ Add Custom Class**, type a name, pick a color (Cancel picks one
   automatically). The new class becomes active and gets the next free YOLO id.
2. **Pick a tool** in the Drawing Tool card:
   - **Bounding Box**: two clicks give a 4-vertex polygon ready for corner refinement.
   - **Polygon**: click each vertex; double-click or `Enter` closes it.
   - Neither is selected at start, so stray clicks do nothing until you pick one.
3. **Or auto-detect**: **⚙ Run Box Detector** runs the detector on the current image and
   appends every detected box as a polygon of the active class. The latest detection opens
   in polygon edit immediately.
4. **Refine**: drag a vertex, click an edge to insert, `Alt`+click to delete. **◀ Prev /
   Next ▶** (or a click on another outline) selects the instance to edit. Right-click a box
   to rename it to another class or delete it; `Del` deletes the one being edited.
5. **Navigate** with `←` / `→` or the Files list.
6. **Save**: **Save All to YOLO** writes one `<stem>.txt` per image.

## Class registry rules

- Classes added in a window belong to that window: closing it or going back to the menu
  removes them, and the next Manual window starts clean.
- Colors must be unique; a color already in use is refused.
- The built-in classes (`pectoral`, `breast`, `nipple`) are hidden in Manual mode so the
  first user class is YOLO id 0.
- The `✕` next to a class removes it together with its masks and polygons on every image.

## The box detector

The detector is any ultralytics YOLO **detection** model, loaded lazily from
`detector.weights` (default `weights/detector/manual_detector.pt`, environment variable
`ANNOTATION_DETECTOR_WEIGHTS`). Its own class labels are ignored: every box goes to the
**active class**, on the assumption that you label one class at a time. Overlapping boxes
are merged class-agnostically (IoU 0.4) so near-duplicates do not appear twice.

Detection runs on a worker thread (`◐ detecting…` in the status bar). If the weights file
is missing, a dialog shows the expected path and how to set it.

## Tips

- The **Display** card has a mask-opacity slider (0 % shows the bare image).
- DICOM files are shown with their VOI LUT / window applied and MONOCHROME1 inverted.
- Right-drag for window / level; double-click the background to reset zoom and window /
  level.
- `Ctrl+Z` drops the most recently added polygon (detected or drawn); `Ctrl+Enter`
  commits the polygon being edited.
- `C` draws the PNL / CC-depth overlay when the classes are named `nipple`, `pectoral` and
  `breast` (the built-in keys); custom names are not measured.
