"""Prompt state container + drawing helpers (SRP: hold prompt + render it).

Why a dedicated module?
    The Gradio UI is event-driven (each click mutates state) and the model
    expects raw numpy/list inputs. Keeping the prompt schema + drawing in
    one place means the UI module only orchestrates events; nothing in
    ``ui.py`` knows OpenCV drawing primitives.

Tissue presets
    A tissue class maps to (display_name, BGR overlay color, target class
    id used for class-competition). The dictionary is the single source of
    truth; UI radio buttons and inference pipeline both read from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


# --- Tissue presets ----------------------------------------------------------

@dataclass(frozen=True)
class TissuePreset:
    key: str
    label: str
    overlay_bgr: tuple[int, int, int]
    class_id: int  # matches MASK_CLASS_LABELS in config.py


TISSUE_PRESETS: dict[str, TissuePreset] = {
    "nipple": TissuePreset("nipple", "Nipple", (32, 32, 255), 3),
    "pectoral": TissuePreset("pectoral", "Pectoral Muscle", (255, 96, 32), 1),
    "breast": TissuePreset("breast", "Breast Tissue", (32, 220, 96), 2),
}


def tissue_by_label(label: str) -> TissuePreset:
    for p in TISSUE_PRESETS.values():
        if p.label == label:
            return p
    return TISSUE_PRESETS["breast"]


# --- Prompt state ------------------------------------------------------------

@dataclass
class BoxPrompt:
    """Two-click axis-aligned box. ``finished`` flips True on the second click."""

    x0: float | None = None
    y0: float | None = None
    x1: float | None = None
    y1: float | None = None
    finished: bool = False

    def to_xyxy(self) -> np.ndarray | None:
        if not self.finished or None in (self.x0, self.y0, self.x1, self.y1):
            return None
        x0, x1 = sorted([float(self.x0), float(self.x1)])  # type: ignore[arg-type]
        y0, y1 = sorted([float(self.y0), float(self.y1)])  # type: ignore[arg-type]
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def reset(self) -> None:
        self.x0 = self.y0 = self.x1 = self.y1 = None
        self.finished = False

    def first_corner_only(self) -> bool:
        return self.x0 is not None and not self.finished


@dataclass
class PromptState:
    """All prompt inputs for ONE prediction.

    ``positive`` and ``ignore`` are pixel-space ``(x, y)`` lists. ``box`` is
    optional. Resetting the image clears everything.
    """

    positive: list[tuple[float, float]] = field(default_factory=list)
    ignore: list[tuple[float, float]] = field(default_factory=list)
    box: BoxPrompt = field(default_factory=BoxPrompt)

    def is_empty(self) -> bool:
        return not self.positive and not self.ignore and self.box.to_xyxy() is None

    def clear(self) -> None:
        self.positive.clear()
        self.ignore.clear()
        self.box.reset()

    def pop_last(self, kind: str) -> None:
        if kind == "positive" and self.positive:
            self.positive.pop()
        elif kind == "ignore" and self.ignore:
            self.ignore.pop()
        elif kind == "box":
            self.box.reset()


# --- Drawing -----------------------------------------------------------------

# Color palette used for prompt markers (BGR for OpenCV).
_COLOR_POS_OUTER = (0, 255, 80)
_COLOR_POS_INNER = (0, 100, 0)
_COLOR_IGNORE_OUTER = (40, 40, 230)
_COLOR_IGNORE_INNER = (10, 10, 90)
_COLOR_IGNORE_DISK = (40, 40, 230)
_COLOR_BOX = (50, 220, 230)
_COLOR_BOX_CORNER = (10, 180, 200)
_COLOR_FIRST_CORNER = (220, 220, 50)


def ensure_rgb(img: np.ndarray | None) -> np.ndarray | None:
    """Promote 2D or 4-channel images to a 3-channel RGB copy."""
    if img is None:
        return None
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] >= 3:
        return img[:, :, :3].copy()
    return img


def _bgr_to_rgb(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(c[2]), int(c[1]), int(c[0]))


def _draw_marker_positive(canvas: np.ndarray, x: int, y: int) -> None:
    cv2.circle(canvas, (x, y), 11, _bgr_to_rgb(_COLOR_POS_OUTER), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 4, _bgr_to_rgb(_COLOR_POS_OUTER), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 4, _bgr_to_rgb(_COLOR_POS_INNER), 1, cv2.LINE_AA)


def _draw_marker_ignore(canvas: np.ndarray, x: int, y: int, radius: int) -> None:
    overlay = canvas.copy()
    cv2.circle(overlay, (x, y), max(radius, 1), _bgr_to_rgb(_COLOR_IGNORE_DISK), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

    cv2.circle(canvas, (x, y), max(radius, 1), _bgr_to_rgb(_COLOR_IGNORE_OUTER), 1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 11, _bgr_to_rgb(_COLOR_IGNORE_OUTER), 2, cv2.LINE_AA)
    cv2.line(canvas, (x - 6, y), (x + 6, y), _bgr_to_rgb(_COLOR_IGNORE_OUTER), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), 3, _bgr_to_rgb(_COLOR_IGNORE_INNER), -1, cv2.LINE_AA)


def _dashed_rect(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int], thick: int = 2, dash: int = 8) -> None:
    x0, y0 = p1
    x1, y1 = p2
    for x_start in range(x0, x1, dash * 2):
        x_end = min(x_start + dash, x1)
        cv2.line(canvas, (x_start, y0), (x_end, y0), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (x_start, y1), (x_end, y1), color, thick, cv2.LINE_AA)
    for y_start in range(y0, y1, dash * 2):
        y_end = min(y_start + dash, y1)
        cv2.line(canvas, (x0, y_start), (x0, y_end), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (x1, y_start), (x1, y_end), color, thick, cv2.LINE_AA)


def _draw_box(canvas: np.ndarray, box: BoxPrompt) -> None:
    if box.first_corner_only():
        x, y = int(box.x0), int(box.y0)  # type: ignore[arg-type]
        color = _bgr_to_rgb(_COLOR_FIRST_CORNER)
        h, w = canvas.shape[:2]
        cv2.line(canvas, (0, y), (w - 1, y), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (x, 0), (x, h - 1), color, 1, cv2.LINE_AA)
        cv2.drawMarker(
            canvas, (x, y), color,
            markerType=cv2.MARKER_CROSS, markerSize=36, thickness=3, line_type=cv2.LINE_AA,
        )
        cv2.circle(canvas, (x, y), 18, color, 3, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas, "1. kose - simdi karsi kosesine tikla",
            (x + 22, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        return

    xyxy = box.to_xyxy()
    if xyxy is None:
        return
    x0, y0, x1, y1 = (int(v) for v in xyxy.tolist())
    _dashed_rect(canvas, (x0, y0), (x1, y1), _bgr_to_rgb(_COLOR_BOX))
    for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        cv2.circle(canvas, (cx, cy), 5, _bgr_to_rgb(_COLOR_BOX_CORNER), -1, cv2.LINE_AA)


def _draw_mask_overlay(
    canvas: np.ndarray,
    mask: np.ndarray | None,
    overlay_bgr: tuple[int, int, int],
    *,
    alpha: float = 0.45,
) -> None:
    if mask is None:
        return
    m = mask.astype(bool)
    if not m.any():
        return
    color_rgb = np.array(_bgr_to_rgb(overlay_bgr), dtype=np.float32)
    region = canvas[m].astype(np.float32)
    canvas[m] = (region * (1.0 - alpha) + color_rgb * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, contours, -1, _bgr_to_rgb(overlay_bgr), 2, cv2.LINE_AA)


def render_overlay(
    rgb: np.ndarray,
    state: PromptState,
    masks_by_tissue: dict[str, np.ndarray] | None,
    *,
    current_tissue_key: str | None = None,
    ignore_radius: int,
    show_ignore_disks: bool = True,
) -> np.ndarray:
    """Compose the final visual: image + ALL stored tissue masks + prompt markers.

    ``masks_by_tissue`` is rendered first (each in its own color, looked up
    from :data:`TISSUE_PRESETS`). The current tissue is rendered last so
    its contour sits on top. Then prompts (box + markers) go above.
    """
    canvas = rgb.copy()

    if masks_by_tissue:
        for key, mask in masks_by_tissue.items():
            if key == current_tissue_key:
                continue
            preset = TISSUE_PRESETS.get(key)
            if preset is None:
                continue
            _draw_mask_overlay(canvas, mask, preset.overlay_bgr, alpha=0.35)

        if current_tissue_key is not None:
            cur_mask = masks_by_tissue.get(current_tissue_key)
            cur_preset = TISSUE_PRESETS.get(current_tissue_key)
            if cur_mask is not None and cur_preset is not None:
                _draw_mask_overlay(canvas, cur_mask, cur_preset.overlay_bgr, alpha=0.50)

    _draw_box(canvas, state.box)

    if show_ignore_disks:
        for x, y in state.ignore:
            _draw_marker_ignore(canvas, int(x), int(y), ignore_radius)
    for x, y in state.positive:
        _draw_marker_positive(canvas, int(x), int(y))

    return canvas


def implicit_box_from_points(
    points: Iterable[tuple[float, float]],
    image_hw: tuple[int, int],
    pad_px: int = 64,
) -> np.ndarray | None:
    """Bounding box of positive points, padded.

    Used in the inference pipeline to align point-only prompts with the
    fine-tune distribution (which only saw box prompts). Returns None if
    fewer than 1 points are given. With one point we synthesize a small
    square around it (radius = pad).
    """
    pts = list(points)
    if not pts:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    h, w = image_hw
    x0 = max(0.0, x0 - pad_px)
    y0 = max(0.0, y0 - pad_px)
    x1 = min(float(w - 1), x1 + pad_px)
    y1 = min(float(h - 1), y1 + pad_px)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return np.array([x0, y0, x1, y1], dtype=np.float32)
