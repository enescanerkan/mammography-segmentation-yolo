# Manual Label

Polygon and bounding-box labeling with a bundled YOLO auto-detector.
Unlike SAM Label, Manual mode starts with an empty class registry —
**you** decide the taxonomy. The first class you add becomes YOLO id 0.

## When to use it

- The label set you need isn't `pectoral / breast / nipple`.
- You already have a YOLO detector (`qt_app/weights/manual_detector.pt`)
  and want one-click auto-labels that you can then refine by hand.
- You want pure manual polygon labeling without any model load.

## Workflow

1. **Add a class**: hit **＋ Add Custom Class**, type a name (e.g.
   `mass`, `calcification`, `lesion`), pick a color. The new class
   becomes the active one and is assigned the next available YOLO
   class id (0 if it's the first).
2. **Pick a tool** in the Drawing Tool card:
   - **Bounding Box** — two clicks → automatic 4-vertex polygon
     ready for corner-drag refinement.
   - **Polygon** — click each vertex; double-click or Enter to close.
   - (Neither selected by default — clicks on the canvas do nothing
     until you pick.)
3. **Or auto-detect**: press **⚙ Run Box Detector**. The bundled YOLO
   model runs on the current image, and every detected rectangle is
   appended as a polygon under the currently active class. The latest
   detection opens in polygon-edit mode immediately so you can adjust
   it without further clicks.
4. **Refine**: drag any vertex; click an edge to insert; Alt+click to
   delete. Use **◀ Prev / Next ▶** to cycle which polygon instance is
   active — handy when the detector returned multiple boxes.
5. **Iterate over images** with `→` / `←` or the Files panel.
6. **Save**: 💾 Save All to YOLO writes one `<stem>.txt` per image.

## Class registry rules

- Classes added in this window are **session-scoped**. Closing the
  window or clicking Back to Main Menu wipes them, so the next time
  Manual Label opens you start clean.
- Color uniqueness is enforced — you cannot give two classes the same
  RGB color. Use the color dialog or let auto-pick choose for you.
- Built-in classes (`pectoral`, `breast`, `nipple`) are **hidden** in
  Manual mode so the first user class lands at YOLO id 0.
- Remove a custom class with the small `✕` button next to its radio.
  Every mask/polygon stored for that class on every image is cleared.

## The auto-detector

The detector is a YOLO model loaded from
`qt_app/weights/manual_detector.pt`. Its own class labels are
ignored — every detection is attributed to the **active class** in the
side panel, on the assumption that the user is doing one class at a
time. To label multiple classes, switch the active class between runs.

The model runs on a `QThread` so the UI stays responsive. Status:
`◐ detecting…` shows in the status bar while it works.

If the weights file is missing, a dialog points at the expected path.

## Tips

- Mask opacity has its own slider (Display card). Drop it to 0% to
  inspect the underlying image without a redraw, push it past 50% to
  pre-check label coverage.
- DICOM files are loaded with their VOI LUT applied — mammography
  studies show in proper grayscale tone (no more washed-out white).
- Right-drag the image for window/level on top of that. Double-click
  the background to reset zoom and W/L together.
- `Ctrl + Z` drops the most recently added polygon (auto-detected or
  manually drawn).
- `Ctrl + Enter` commits the polygon currently being edited (locks
  vertex positions).
