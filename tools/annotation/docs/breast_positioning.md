# Breast Positioning

YOLO-pose landmark detection on an MLO + CC mammography pair, with a distance comparison
against a user-set threshold (default 10 mm). It shares the toolkit's header and dark theme.

## When to use it

- A paired MLO + CC DICOM study needs a quantitative positioning check.
- A folder of studies should be reviewed pair by pair.

## Workflow

1. **Pick input**:
   - **Select DICOM Pair**: choose the MLO file, then the CC file.
   - **Select Folder**: scans recursively for `.dcm` / `.dicom`, pairs MLO and CC by file
     name (swapping `mlo` and `cc` must give an existing file) and reports how many pairs
     and unpaired files were found.
2. **Navigate** pairs with **◀ Prev Pair / Next Pair ▶**. **↔** swaps the two views if they
   were loaded the wrong way round.
3. **Run the analysis**:
   - **MLO Analysis**: nipple + two pectoral-line points; distance = perpendicular from the
     nipple to the line through the pectoral points.
   - **CC Analysis**: nipple; distance = horizontal distance to the chest-wall image edge
     (the side holding the breast is detected from the image).
   - **Compare**: difference of the two distances; within the threshold is *correct*, up to
     1.5 × the threshold *questionable*, above that *incorrect*.
4. **Adjust the threshold** with the slider (0–20 mm in 0.5 mm steps). It does not
   re-compare by itself; press **Compare** again.
5. **Save**: **Save Results** writes time-stamped TXT / CSV / JSON files (including the
   threshold used); **Save Images** writes the two figures as PNG. Both go to
   `positioning.results_dir` (default `tools/annotation/results/positioning/`, git-ignored).

Distances use the DICOM `ImagerPixelSpacing` (or `PixelSpacing`) scaled to the model's
640 × 640 input; 0.085 mm/pixel is assumed when the header has neither.

## File layout the tool expects

- DICOMs anywhere under the chosen folder; nested folders are fine.
- Names containing `mlo` and `cc` are paired automatically, e.g. `case01_mlo.dcm` +
  `case01_cc.dcm`.
- Files without a partner are listed as *unpaired*; open them with **Select DICOM Pair**.

## Model weights

Two ultralytics YOLO-pose models, configured as `positioning.mlo_weights` and
`positioning.cc_weights` (environment variables `POSITIONING_MLO_WEIGHTS` /
`POSITIONING_CC_WEIGHTS`), by default:

- `weights/positioning/mlo-yolo26-pose-advanced.pt` (keypoints: nipple, pectoral top, pectoral bottom)
- `weights/positioning/cc-yolo26-pose-advanced.pt` (keypoint: nipple)

If either file is missing (or smaller than 1 MB, i.e. a broken download), the launcher
names the missing path and returns to the menu.

This tool measures from **pose landmarks**. The PNL overlay of SAM / Manual mode measures
from **segmentation masks** with `breast_seg.geometry`; the two are independent methods.

## Tips

- Each canvas is matplotlib-based: scroll to zoom, right-click to reset, drag a landmark
  to move it; the distance is recomputed live.
- The Analysis Log records each step and comparison; **Clear Log** between studies.
- **File → ← Back to Toolkit Menu** (`Ctrl+M`) returns to the launcher; closing the
  window with ✕ quits the toolkit.
