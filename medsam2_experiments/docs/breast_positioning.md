# Breast Positioning

YOLO-pose landmark detection for MLO and CC mammography views, with
distance comparison against a user-configurable threshold (default
10 mm).

This tool was ported from a standalone PyQt5 app into the toolkit. It
shares the gradient header and the dark palette with the other tools.

## When to use it

- You have a paired MLO + CC DICOM study and want a quantitative
  positioning quality report.
- You want to QA an existing dataset by walking through detected
  MLO/CC pairs in a folder.

## Workflow

1. **Pick input**:
   - **Select DICOM Pair** — choose MLO then CC manually.
   - **Select Folder** — recursively scans for `.dcm` / `.dicom`,
     auto-pairs MLO ↔ CC by filename (`mlo` ↔ `cc` swap), and surfaces
     the count: e.g. `12 pair(s) detected • 3 unpaired`.
2. **Navigate** detected pairs with **◀ Prev Pair / Next Pair ▶**.
3. **Run analysis**:
   - **MLO Analysis** runs the MLO landmark model.
   - **CC Analysis** runs the CC landmark model.
   - **Compare** computes pectoral-line ↔ nipple distance for each
     view and labels the positioning as `GOOD` or `POOR` based on the
     current threshold.
4. **Adjust the threshold** with the slider (0.0 — 20.0 mm in 0.5 mm
   steps). The slider does *not* auto-recompute — press **Compare**
   again to apply the new value.
5. **Save**: **Save Results** writes timestamped TXT/CSV/JSON files;
   **Save Images** writes the rendered visualization PNGs.

The current threshold is included in the saved report header so older
files remain interpretable.

## File layout the tool expects

- DICOMs anywhere under the chosen folder. Nested folders are fine.
- Filenames containing `mlo` and `cc` are paired automatically when
  swapping one substring for the other yields an existing file. For
  example `study42_mlo.dcm` + `study42_cc.dcm` becomes one pair.
- Files without an `mlo` / `cc` token (or with no matching partner)
  are surfaced as `unpaired` and ignored by the pair navigator. Use
  the manual `Select DICOM Pair` button to pick them.

## Model weights

The MLO and CC YOLO-pose models live under
`qt_app/positioning/weights/`:

- `mlo-yolo26-pose-advanced.pt`
- `cc-yolo26-pose-advanced.pt`

If either file is missing, the launcher dialog reports the missing
name and refuses to start. Replace the file and retry.

## Tips

- Each canvas is matplotlib-backed: scroll to zoom, right-click to
  reset, and drag landmarks to nudge them. The pectoral-line / nipple
  distance recalculates live as you drag.
- The Analysis Log shows analysis steps + comparison outcomes. Use
  `Clear Log` between studies for a clean per-study record.
- The window has its own **← Back to Toolkit Menu** entry in the File
  menu (`Ctrl + M`). Closing the X button quits the toolkit.
