"""YOLO box detector wrapper used by Manual Label mode.

Wraps ultralytics' ``YOLO`` model with lazy loading and a small
``QThread`` runner so inference doesn't block the UI. The detector
returns a list of (xyxy, class_id, confidence) tuples — class labels
from the underlying model are ignored; the Manual workflow assigns
boxes to the user's *active* annotation class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

log = logging.getLogger(__name__)


# Default location next to this file. Falls back to env override.
_HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _HERE / "weights" / "manual_detector.pt"


@dataclass
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float
    model_class_id: int  # class id from the YOLO model (informational only)

    def as_polygon(self) -> list[tuple[float, float]]:
        return [
            (self.x0, self.y0),
            (self.x1, self.y0),
            (self.x1, self.y1),
            (self.x0, self.y1),
        ]


class BoxDetector:
    """Thin wrapper around an ultralytics YOLO model. Lazily imports
    ``ultralytics`` so that startup cost is paid only when Manual mode
    actually uses the detector."""

    def __init__(self, weights_path: Optional[Path] = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
        self._model = None  # populated on first predict

    @property
    def weights_exists(self) -> bool:
        return self.weights_path.is_file()

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.weights_exists:
            raise FileNotFoundError(f"Detector weights not found: {self.weights_path}")
        log.info("Loading box detector from %s", self.weights_path)
        from ultralytics import YOLO  # type: ignore
        self._model = YOLO(str(self.weights_path))
        log.info("Box detector loaded")

    def predict(
        self,
        rgb: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 50,
        dedupe_iou: float = 0.4,
    ) -> list[Detection]:
        """Run detection on an RGB uint8 image.

        ``dedupe_iou`` collapses boxes that overlap heavily into the
        single highest-confidence box. Ultralytics' own NMS uses ``iou``
        as the suppression threshold, but for our labeling pipeline we
        also want to fuse boxes that the model returned with different
        class ids — the user is going to attribute them all to the
        active annotation class anyway, so two near-identical boxes look
        like a bug to them.
        """
        self.ensure_loaded()
        assert self._model is not None
        results = self._model.predict(
            source=rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.zeros(len(xyxy))
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        raw: list[Detection] = []
        h, w = rgb.shape[:2]
        for (x0, y0, x1, y1), c, k in zip(xyxy, confs, clss):
            x0 = max(0.0, min(float(x0), float(w - 1)))
            y0 = max(0.0, min(float(y0), float(h - 1)))
            x1 = max(0.0, min(float(x1), float(w - 1)))
            y1 = max(0.0, min(float(y1), float(h - 1)))
            if (x1 - x0) < 2 or (y1 - y0) < 2:
                continue
            raw.append(Detection(
                x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1),
                confidence=float(c), model_class_id=int(k),
            ))
        # Class-agnostic NMS pass on top of ultralytics' own suppression.
        return _dedupe_overlapping(raw, iou_thresh=dedupe_iou)


class DetectorWorker(QObject):
    """Runs :meth:`BoxDetector.predict` on a worker thread and emits the
    result back to the GUI thread."""

    finished = pyqtSignal(list)   # list[Detection]
    failed = pyqtSignal(str)

    def __init__(
        self,
        detector: BoxDetector,
        rgb: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        super().__init__()
        self._detector = detector
        self._rgb = rgb
        self._conf = conf
        self._iou = iou

    def run(self) -> None:
        try:
            dets = self._detector.predict(self._rgb, conf=self._conf, iou=self._iou)
            self.finished.emit(dets)
        except Exception as exc:  # noqa: BLE001 — surface to UI
            log.exception("Detector predict failed")
            self.failed.emit(str(exc))


def _iou_xyxy(a: Detection, b: Detection) -> float:
    ix0 = max(a.x0, b.x0)
    iy0 = max(a.y0, b.y0)
    ix1 = min(a.x1, b.x1)
    iy1 = min(a.y1, b.y1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a.x1 - a.x0) * (a.y1 - a.y0)
    area_b = (b.x1 - b.x0) * (b.y1 - b.y0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _dedupe_overlapping(dets: list[Detection], iou_thresh: float) -> list[Detection]:
    """Class-agnostic NMS. Keeps the highest-confidence box from each
    overlapping cluster — the user wouldn't want two near-identical
    rectangles for the same region."""
    if len(dets) <= 1:
        return dets
    sorted_dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for cand in sorted_dets:
        if all(_iou_xyxy(cand, k) < iou_thresh for k in kept):
            kept.append(cand)
    return kept


def run_detector_async(
    detector: BoxDetector,
    rgb: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    on_finished=None,
    on_failed=None,
    parent: Optional[QObject] = None,
) -> tuple[QThread, DetectorWorker]:
    """Convenience: start ``DetectorWorker`` on a fresh ``QThread`` and
    wire its signals. The caller is responsible for keeping the returned
    references alive (e.g. as attributes on the calling widget) until the
    thread finishes."""
    thread = QThread(parent)
    worker = DetectorWorker(detector, rgb, conf=conf, iou=iou)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_finished is not None:
        worker.finished.connect(on_finished)
    if on_failed is not None:
        worker.failed.connect(on_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread, worker
