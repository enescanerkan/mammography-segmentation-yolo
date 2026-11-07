# Roadmap and known limitations

Open items for SAM mode and the MedSAM2 fine-tune. Each one states what to measure before
it counts as done; nothing here is needed to use the toolkit.

## Known limitations

- **Box-only fine-tune.** The decoder was trained with box prompts only; the implicit box
  hides this for point prompts, but the real fix is a fine-tune with mixed box + point
  prompts (sample points in `experiments/prepare_medsam2_data.py`).
- **Per-class binary training.** Class competition compensates for breast masks leaking
  into the pectoral muscle on MLO views. The lasting fix is one decoder with a multi-class
  output (background, pectoral, breast, nipple); the exported masks are already
  multi-class, only the training is binary.
- **Axis-aligned boxes only.** The pectoral muscle is oblique on MLO views; oriented boxes
  would need rotation handling around SAM2.
- **One ignore radius.** All ignore points share the same radius.
- **PNL overlay in pixels.** The SAM / Manual overlay reports pixel lengths; converting to
  mm needs the DICOM pixel spacing, which the annotation session does not carry yet.

## Inference / UX (no retraining)

1. **Iterative refinement through `mask_input`.** Feed the previous low-resolution logits
   back into `SAM2ImagePredictor.predict(mask_input=...)` on the next click. The fine-tuned
   decoder never saw `mask_input`, so add it as a toggle (default off) and A/B it on
   multi-click (3+) refinements; keep it only if Dice improves.
2. **Per-class mask threshold.** Optional `mask_threshold` per `TissuePreset` (for example a
   lower one for the small nipple). Keep only if it raises nipple Dice without changing
   breast / pectoral.
3. **Connected-component cleanup.** After ignore disks, optionally keep the largest
   component for breast and pectoral (not for nipple). Accept when an ignore point in the
   middle of the breast leaves no fragments.
4. **View-aware classes.** When the view is known to be CC, disable the Pectoral class.
   No-op when the view is unknown.

## Training (requires a new fine-tune)

5. **Validation tracking.** Evaluate saved checkpoints on the prepared `val` split
   (per-class mean Dice) so `medsam2.checkpoint` can be chosen on validation rather than
   training loss.
6. **Smaller box jitter for small structures.** `--bbox-shift` (in 256-pixel space) is
   large relative to the nipple; try a smaller value in the next fine-tune and compare
   nipple Dice and loss stability.
7. **Minimum foreground size.** Re-evaluate `MEDSAM2_MIN_FG` (default 50 px): count how
   many samples a higher threshold would drop before changing it.
