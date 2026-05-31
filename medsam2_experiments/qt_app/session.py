"""In-memory annotation store + YOLO segmentation export.

Holds ``{image_path: ImageAnnotation}`` for the whole labeling session so the
user can navigate between images without losing work. ``save_all`` writes
every dirty image to ``<images_dir>/labels/<stem>.txt`` (YOLO seg format).

YOLO class id derivation
------------------------
``TissuePreset.class_id`` from :mod:`interactive.prompts` is the *raster
mask pixel* id (1=pectoral, 2=breast, 3=nipple, matching
``config.MASK_CLASS_LABELS``). The corresponding YOLO class id is
``class_id - 1`` (0=pectoral, 1=breast, 2=nipple) which matches
``seg-dataset/data.yaml``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from interactive.prompts import TISSUE_PRESETS, PromptState


def tissue_to_yolo_id(tissue_key: str) -> int:
    preset = TISSUE_PRESETS[tissue_key]
    return preset.class_id - 1


@dataclass
class ImageAnnotation:
    """All persistent state for a single image.

    ``masks``    : binary uint8 ``H×W`` mask per tissue (UNION of all that
                   tissue's polygon instances + any SAM mask).
    ``prompts``  : last prompt state per tissue (serialized) so the user can
                   re-enter the image and continue editing.
    ``polygons`` : list of user-drawn polygon instances per tissue (so the
                   same class can appear multiple times on one image, e.g.
                   two separate breast tissue regions). YOLO export writes
                   one line per polygon.
    """

    image_hw: tuple[int, int]
    masks: dict[str, np.ndarray] = field(default_factory=dict)
    prompts: dict[str, dict] = field(default_factory=dict)
    polygons: dict[str, list[list[tuple[float, float]]]] = field(default_factory=dict)

    def has_any_mask(self) -> bool:
        return any(m is not None and m.any() for m in self.masks.values())

    def tissues_with_mask(self) -> list[str]:
        return [k for k, m in self.masks.items() if m is not None and m.any()]


@dataclass
class SaveReport:
    written: list[Path] = field(default_factory=list)
    skipped_empty: list[Path] = field(default_factory=list)
    errors: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.written) + len(self.skipped_empty) + len(self.errors)


class AnnotationSession:
    """In-memory store with deferred YOLO save.

    The session does NOT own the active image; the main window passes the
    image path on every mutation. This keeps the store independent of the
    UI's notion of "currently displayed image".
    """

    def __init__(
        self,
        output_dir: Path | None = None,
        polygon_epsilon: float = 1.5,
        min_polygon_pts: int = 3,
        min_contour_area_px: int = 16,
    ) -> None:
        self._store: dict[Path, ImageAnnotation] = {}
        self.output_dir = output_dir
        self.polygon_epsilon = polygon_epsilon
        self.min_polygon_pts = min_polygon_pts
        self.min_contour_area_px = min_contour_area_px

    # --- store access ---------------------------------------------------

    def get(self, image_path: Path) -> ImageAnnotation | None:
        return self._store.get(Path(image_path))

    def get_or_create(self, image_path: Path, image_hw: tuple[int, int]) -> ImageAnnotation:
        p = Path(image_path)
        ann = self._store.get(p)
        if ann is None:
            ann = ImageAnnotation(image_hw=image_hw)
            self._store[p] = ann
        return ann

    def set_mask(self, image_path: Path, tissue_key: str, mask: np.ndarray | None) -> None:
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        if mask is None:
            ann.masks.pop(tissue_key, None)
        else:
            ann.masks[tissue_key] = mask.astype(np.uint8)

    def set_prompts(self, image_path: Path, tissue_key: str, state: PromptState) -> None:
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        ann.prompts[tissue_key] = _prompts_to_dict(state)

    def get_prompts(self, image_path: Path, tissue_key: str) -> PromptState:
        ann = self._store.get(Path(image_path))
        if ann is None or tissue_key not in ann.prompts:
            return PromptState()
        return _prompts_from_dict(ann.prompts[tissue_key])

    def set_polygon(
        self,
        image_path: Path,
        tissue_key: str,
        polygon: list[tuple[float, float]] | None,
    ) -> None:
        """Replace the entire polygon list for ``tissue_key`` with a single
        polygon (or drop the list when ``polygon`` is None). Used when SAM
        is the source of truth and we only have one polygon per tissue."""
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        if polygon is None or len(polygon) < 3:
            ann.polygons.pop(tissue_key, None)
        else:
            ann.polygons[tissue_key] = [[(float(x), float(y)) for x, y in polygon]]

    def append_polygon(
        self,
        image_path: Path,
        tissue_key: str,
        polygon: list[tuple[float, float]],
    ) -> None:
        """Add a new polygon instance for ``tissue_key`` (manual mode multi-
        instance). The mask should be updated separately to the union of
        all polygons."""
        if not polygon or len(polygon) < 3:
            return
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        ann.polygons.setdefault(tissue_key, []).append(
            [(float(x), float(y)) for x, y in polygon]
        )

    def replace_last_polygon(
        self,
        image_path: Path,
        tissue_key: str,
        polygon: list[tuple[float, float]],
    ) -> None:
        """Replace the most-recently-added polygon (used while the user is
        editing the latest polygon's vertices)."""
        if not polygon or len(polygon) < 3:
            return
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        polys = ann.polygons.get(tissue_key)
        if not polys:
            ann.polygons[tissue_key] = [[(float(x), float(y)) for x, y in polygon]]
            return
        polys[-1] = [(float(x), float(y)) for x, y in polygon]

    def replace_polygon_at(
        self,
        image_path: Path,
        tissue_key: str,
        idx: int,
        polygon: list[tuple[float, float]],
    ) -> None:
        """Replace the polygon at index ``idx`` in the per-tissue list."""
        if not polygon or len(polygon) < 3:
            return
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        polys = ann.polygons.get(tissue_key)
        if polys is None or not (0 <= idx < len(polys)):
            return
        polys[idx] = [(float(x), float(y)) for x, y in polygon]

    def remove_polygon_at(
        self,
        image_path: Path,
        tissue_key: str,
        idx: int,
    ) -> None:
        """Drop the polygon at index ``idx`` (e.g. user pressed Delete on
        a selected instance). No-op if out of range."""
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        polys = ann.polygons.get(tissue_key)
        if polys is None or not (0 <= idx < len(polys)):
            return
        polys.pop(idx)
        if not polys:
            ann.polygons.pop(tissue_key, None)

    def get_polygons(
        self, image_path: Path, tissue_key: str,
    ) -> list[list[tuple[float, float]]]:
        ann = self._store.get(Path(image_path))
        if ann is None:
            return []
        return list(ann.polygons.get(tissue_key, []))

    def get_polygon(
        self, image_path: Path, tissue_key: str,
    ) -> list[tuple[float, float]] | None:
        """Return the LAST polygon (kept for backward compatibility)."""
        polys = self.get_polygons(image_path, tissue_key)
        return polys[-1] if polys else None

    def clear_tissue(self, image_path: Path, tissue_key: str) -> None:
        ann = self._store.get(Path(image_path))
        if ann is None:
            return
        ann.masks.pop(tissue_key, None)
        ann.prompts.pop(tissue_key, None)
        ann.polygons.pop(tissue_key, None)

    def clear_image(self, image_path: Path) -> None:
        self._store.pop(Path(image_path), None)

    def annotated_image_count(self) -> int:
        return sum(1 for ann in self._store.values() if ann.has_any_mask())

    def all_paths(self) -> list[Path]:
        return list(self._store.keys())

    # --- YOLO save ------------------------------------------------------

    def _resolve_output_dir(self, image_path: Path) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        return Path(image_path).parent / "labels"

    def save_one(self, image_path: Path) -> Path | None:
        """Write one image's labels. Returns the txt path, or None if nothing
        to save."""
        ann = self._store.get(Path(image_path))
        if ann is None or not ann.has_any_mask():
            return None
        out_dir = self._resolve_output_dir(image_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / (Path(image_path).stem + ".txt")
        lines = self._masks_to_yolo_lines(ann)
        txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return txt_path

    def save_all(self) -> SaveReport:
        report = SaveReport()
        for path, ann in self._store.items():
            try:
                if not ann.has_any_mask():
                    report.skipped_empty.append(path)
                    continue
                written = self.save_one(path)
                if written is not None:
                    report.written.append(written)
            except Exception as exc:  # noqa: BLE001 — surface to UI
                report.errors.append((path, str(exc)))
        return report

    def _masks_to_yolo_lines(self, ann: ImageAnnotation) -> list[str]:
        H, W = ann.image_hw
        lines: list[str] = []
        # Iterate over the union of mask + polygon keys so a tissue with
        # only a polygon (no rasterized mask yet) still gets exported.
        tissue_keys = set(ann.masks.keys()) | set(ann.polygons.keys())
        for tissue_key in tissue_keys:
            cls_id = tissue_to_yolo_id(tissue_key)
            polys: list[np.ndarray] = []

            user_polys = ann.polygons.get(tissue_key, [])
            if user_polys:
                # One YOLO line per polygon instance (multi-instance per class).
                for p in user_polys:
                    if p and len(p) >= 3:
                        polys.append(np.array(p, dtype=np.float32))
            else:
                mask = ann.masks.get(tissue_key)
                if mask is None or not mask.any():
                    continue
                polys.extend(_extract_contours(
                    mask, self.polygon_epsilon, self.min_polygon_pts, self.min_contour_area_px,
                ))

            for poly in polys:
                xs = poly[:, 0] / float(W)
                ys = poly[:, 1] / float(H)
                xs = np.clip(xs, 0.0, 1.0)
                ys = np.clip(ys, 0.0, 1.0)
                parts = [f"{cls_id}"]
                for x, y in zip(xs, ys):
                    parts.append(f"{x:.6f}")
                    parts.append(f"{y:.6f}")
                lines.append(" ".join(parts))
        return lines


def _extract_contours(
    mask: np.ndarray,
    epsilon: float,
    min_pts: int,
    min_area: int,
) -> list[np.ndarray]:
    """Return a list of ``(N, 2)`` polygon arrays in image (x, y) pixel space.

    Uses ``RETR_EXTERNAL`` so internal holes are dropped (consistent with
    the seg-dataset polygon convention). Each disjoint blob becomes its own
    polygon — and thus its own YOLO line.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, epsilon=max(epsilon, 0.1), closed=True)
        poly = approx.reshape(-1, 2)
        if poly.shape[0] < min_pts:
            continue
        out.append(poly.astype(np.float32))
    return out


# --- PromptState <-> dict bridging (kept local to avoid coupling) -------

def _prompts_to_dict(s: PromptState) -> dict:
    return {
        "positive": [list(p) for p in s.positive],
        "ignore": [list(p) for p in s.ignore],
        "box": {
            "x0": s.box.x0, "y0": s.box.y0,
            "x1": s.box.x1, "y1": s.box.y1,
            "finished": bool(s.box.finished),
        },
    }


def _prompts_from_dict(d: dict) -> PromptState:
    from interactive.prompts import BoxPrompt
    s = PromptState()
    s.positive = [tuple(p) for p in d.get("positive", [])]
    s.ignore = [tuple(p) for p in d.get("ignore", [])]
    b = d.get("box") or {}
    s.box = BoxPrompt(
        x0=b.get("x0"), y0=b.get("y0"),
        x1=b.get("x1"), y1=b.get("y1"),
        finished=bool(b.get("finished", False)),
    )
    return s


SUPPORTED_IMAGE_EXTS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".dcm", ".dicom",
)


def list_images(
    folder: Path,
    exts: Iterable[str] = SUPPORTED_IMAGE_EXTS,
    recursive: bool = True,
) -> list[Path]:
    """Return sorted image paths under ``folder``.

    By default searches recursively because labeling datasets usually live
    in nested ``images/train``-style folders. Pass ``recursive=False`` for
    a flat scan.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    exts_lower = tuple(e.lower() for e in exts)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in exts_lower
    )
