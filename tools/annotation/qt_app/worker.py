"""Background inference worker (QThread).

Rapid user clicks would otherwise serialize on the main thread and freeze
the UI while SAM2 runs. This worker pulls inference off the GUI thread and
coalesces pending requests so only the *latest* prompt configuration is
predicted (older requests are dropped — there's no point computing a mask
the user already replaced with a newer click).

Wire-up
-------
    worker = InferenceWorker(pipeline)
    worker.finished.connect(on_finished)
    worker.start()
    worker.submit(req)        # called from UI thread, non-blocking

When ``submit`` is called while the worker is busy, the in-flight job
finishes uninterrupted; the latest queued request runs immediately after.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from PyQt5.QtCore import QMutex, QMutexLocker, QObject, QThread, pyqtSignal

from interactive.inference import InferencePipeline, InferenceSettings, PredictionResult
from interactive.prompts import PromptState, TissuePreset


@dataclass
class InferenceRequest:
    """One prediction request. Identified by ``(image_path, tissue_key)`` so
    the receiver can tell which one came back."""

    image_path: str
    tissue_key: str
    rgb: np.ndarray
    state: PromptState
    tissue: TissuePreset
    settings: InferenceSettings
    masks_by_tissue: Mapping[str, np.ndarray]
    request_id: int = 0


class InferenceWorker(QThread):
    """Single-threaded inference runner with latest-only request coalescing.

    Signals
    -------
    started_request(request_id)  — emitted when a request starts running.
    finished(request_id, image_path, tissue_key, result)  — successful run.
    failed(request_id, image_path, tissue_key, message)   — exception path.
    """

    started_request = pyqtSignal(int)
    finished = pyqtSignal(int, str, str, object)
    failed = pyqtSignal(int, str, str, str)

    def __init__(self, pipeline: InferencePipeline, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._mutex = QMutex()
        self._pending: Optional[InferenceRequest] = None
        self._stop = False
        self._next_id = 0

    # called from UI thread
    def submit(self, req: InferenceRequest) -> int:
        """Queue a request; replaces any not-yet-started one. Returns the
        assigned request id."""
        with QMutexLocker(self._mutex):
            self._next_id += 1
            req.request_id = self._next_id
            self._pending = req
            rid = req.request_id
        # Wake the worker loop. We use a short non-blocking wait, so just
        # tickle the event by calling exit on the QWaitCondition... here we
        # poll with a small sleep instead — keeps the threading model dead
        # simple and CPU usage negligible.
        return rid

    def stop(self) -> None:
        with QMutexLocker(self._mutex):
            self._stop = True
            self._pending = None
        self.wait(2000)

    def run(self) -> None:  # QThread entry
        while True:
            with QMutexLocker(self._mutex):
                if self._stop:
                    return
                req = self._pending
                self._pending = None

            if req is None:
                self.msleep(15)
                continue

            self.started_request.emit(req.request_id)
            try:
                result = self._pipeline.run(
                    rgb=req.rgb,
                    state=req.state,
                    tissue=req.tissue,
                    settings=req.settings,
                    masks_by_tissue=req.masks_by_tissue,
                )
                self.finished.emit(
                    req.request_id, req.image_path, req.tissue_key, result,
                )
            except Exception as exc:  # noqa: BLE001 — propagate to UI
                self.failed.emit(
                    req.request_id, req.image_path, req.tissue_key, str(exc),
                )
