"""YOLO label files: polygons <-> per-class binary masks <-> semantic label maps, and YOLO pose label lines.

Coordinates in YOLO files are normalised to ``[0, 1]``. Every function here is pure numpy / OpenCV.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from breast_seg.config import PAINT_ORDER, YOLO_CLASSES

Polygons = Dict[int, List[np.ndarray]]
PointOrNone = Optional[Tuple[float, float]]


def parse_polygons(text: str) -> Polygons:
    """YOLO-seg text -> ``{class: [normalised (N, 2) polygon, ...]}``; lines with fewer than 3 points are skipped."""
    polys: Polygons = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        pts = np.asarray(parts[1:], dtype=np.float32)
        pts = pts[: len(pts) // 2 * 2].reshape(-1, 2)
        polys.setdefault(int(parts[0]), []).append(pts)
    return polys


def read_polygons(path: Union[str, Path]) -> Polygons:
    """Read a YOLO-seg ``.txt`` file (see :func:`parse_polygons`)."""
    return parse_polygons(Path(path).read_text())


def polygons_to_masks(polys: Polygons, height: int, width: int,
                      classes: Iterable[int] = tuple(YOLO_CLASSES)) -> Dict[int, np.ndarray]:
    """Per-class binary ``uint8`` masks (instances of a class are unioned). Points are truncated to pixels, as the
    dataset builders always did."""
    out = {c: np.zeros((height, width), np.uint8) for c in classes}
    for cls, items in polys.items():
        if cls not in out:
            continue
        for p in items:
            pts = (p.astype(np.float64) * [width, height]).astype(np.int32)
            cv2.fillPoly(out[cls], [pts], 1)
    return out


def masks_to_lines(masks: Dict[int, np.ndarray], min_area: float = 20.0, epsilon: float = 0.002) -> List[str]:
    """Binary masks -> YOLO-seg lines: external contours with area >= ``min_area`` pixels, simplified with
    ``approxPolyDP(epsilon * perimeter)``, normalised and clipped to ``[0, 1]`` (5 decimals)."""
    lines: List[str] = []
    for cls in sorted(masks):
        m = masks[cls]
        if not m.any():
            continue
        h, w = m.shape[:2]
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            ap = cv2.approxPolyDP(cnt, epsilon * cv2.arcLength(cnt, True), True).reshape(-1, 2).astype(float)
            if len(ap) < 3:
                continue
            ap[:, 0] /= w
            ap[:, 1] /= h
            ap = np.clip(ap, 0, 1)
            lines.append(f"{cls} " + " ".join(f"{x:.5f} {y:.5f}" for x, y in ap))
    return lines


def write_lines(path: Union[str, Path], lines: Sequence[str]) -> None:
    """Write label lines with a trailing newline (an empty file for no lines)."""
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def rasterise(polys: Polygons, height: int, width: int,
              paint_order: Sequence[Tuple[int, int]] = PAINT_ORDER) -> Tuple[np.ndarray, int]:
    """Polygons -> semantic label map (``uint8``) painted in ``paint_order``, plus the number of already labelled
    pixels that later classes overwrote. Coordinates are rounded to the nearest pixel."""
    label = np.zeros((height, width), np.uint8)
    overwritten = 0
    for cls, value in paint_order:
        if cls not in polys:
            continue
        layer = np.zeros((height, width), np.uint8)
        for p in polys[cls]:
            q = np.empty_like(p)
            q[:, 0] = p[:, 0] * width
            q[:, 1] = p[:, 1] * height
            cv2.fillPoly(layer, [q.round().astype(np.int32)], 1)
        overwritten += int(((layer > 0) & (label > 0)).sum())
        label[layer > 0] = value
    return label, overwritten


def check_label_text(text: str, allowed: Iterable[int] = tuple(YOLO_CLASSES)) -> Tuple[Counter, List[str]]:
    """Validate a YOLO-seg label: returns ``({class: polygon count}, [error, ...])``.

    Errors: unknown class, fewer than 3 points or an odd coordinate count, non-finite or out-of-range values.
    """
    allowed = {str(c) for c in allowed}
    counts: Counter = Counter()
    errors: List[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if parts[0] not in allowed:
            errors.append(f"line {n}: class {parts[0]}")
            continue
        coords = parts[1:]
        if len(coords) < 6 or len(coords) % 2:
            errors.append(f"line {n}: {len(coords)} coordinates")
            continue
        try:
            v = np.asarray(coords, float)
        except ValueError:
            errors.append(f"line {n}: non-numeric coordinate")
            continue
        if not np.isfinite(v).all() or v.min() < 0 or v.max() > 1:
            errors.append(f"line {n}: coordinate outside [0, 1]")
            continue
        counts[int(parts[0])] += 1
    return counts, errors


def pose_label_line(keypoints: Sequence[PointOrNone], width: int = 640, height: int = 640,
                    class_id: int = 0, bbox_padding_px: float = 100.0) -> str:
    """YOLO-pose line for one breast: ``cls cx cy w h  x1 y1 v1 ...`` with keypoints in canvas pixels.

    A keypoint outside the canvas (or ``None``) is written as ``0 0 0`` (not visible); visible keypoints get
    visibility 2. The box is the keypoints' extent padded by ``bbox_padding_px`` (normalised by the width on both
    axes, as the landmark models were trained). Returns ``""`` when no keypoint is visible.
    """
    parts: List[str] = []
    xs: List[float] = []
    ys: List[float] = []
    for kp in keypoints:
        if kp is None or not (0 <= kp[0] <= width and 0 <= kp[1] <= height):
            parts.append("0.000000 0.000000 0")
            continue
        x, y = min(max(kp[0] / width, 0.0), 1.0), min(max(kp[1] / height, 0.0), 1.0)
        parts.append(f"{x:.6f} {y:.6f} 2")
        xs.append(x)
        ys.append(y)
    if not xs:
        return ""
    pad = bbox_padding_px / width
    x0, x1 = max(0.0, min(xs) - pad), min(1.0, max(xs) + pad)
    y0, y1 = max(0.0, min(ys) - pad), min(1.0, max(ys) + pad)
    box = f"{(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
    return f"{class_id} {box} " + " ".join(parts)
