"""The review application's behaviour, independent of HTTP.

:class:`PositioningService` wires the engine (inference), the assessor (measurements, criteria, verdicts) and the
store (renders + label JSON). The FastAPI layer in :mod:`.api` only translates requests into these calls, so the
whole flow can be exercised — and tested — without a web server.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np

from ... import __version__
from ...imaging import DicomError, read_dicom
from ...inference import EngineError, LandmarkEngine, ModelLoader
from ...positioning import CriteriaGrader, LabelError, PositioningAssessor, apply_labels, build_report, clean_labels
from ...positioning import clean_rules as _clean_rules
from ...positioning.study import labels_of
from ...registry import Registry
from .config import AppConfig
from .storage import StudyStore
from .uploads import UploadedFile

log = logging.getLogger("breast_seg.positioning_app")


class Engine(Protocol):
    """What the service needs from an inference engine (see :class:`~breast_seg.inference.LandmarkEngine`)."""

    @property
    def ready(self) -> bool: ...

    def load(self) -> Any: ...

    def status(self) -> Dict[str, Any]: ...

    def loaded_weights(self) -> Dict[str, str]: ...

    def analyse(self, pixels: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]: ...


class ServiceError(Exception):
    """A request the service refuses; ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class PositioningService:
    """Upload, analyse, store, edit and report positioning studies."""

    def __init__(self, config: AppConfig, engine: Engine, store: StudyStore,
                 assessor: Optional[PositioningAssessor] = None) -> None:
        self.config = config
        self.engine = engine
        self.store = store
        self.assessor = assessor or PositioningAssessor(config.rules.to_rules(), CriteriaGrader(config.criteria))
        self._loader_thread: Optional[threading.Thread] = None
        self.started = time.time()

    @classmethod
    def from_config(cls, config: AppConfig, engine: Optional[Engine] = None,
                    loader: Optional[ModelLoader] = None) -> PositioningService:
        """Build the default engine from the registry named in ``config`` unless one is injected."""
        if engine is None:
            registry = Registry.load(config.registry_path, weights_root=config.weights_root)
            engine = LandmarkEngine(registry, device=config.models.device,
                                    preview_size=config.display.preview_size, loader=loader)
        store = StudyStore(config.data_dir, config.storage.retention_days, config.storage.purge_every_s)
        return cls(config, engine, store)

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self, background: bool = True) -> None:
        """Purge expired studies and load the models (in a thread, so the splash screen can poll health)."""
        removed = self.store.purge_if_due()
        if removed:
            log.info("deleted %d studies past the retention window", removed)
        if background:
            self._loader_thread = threading.Thread(target=self._load_models, name="model-loader", daemon=True)
            self._loader_thread.start()
        else:
            self._load_models()

    def _load_models(self) -> None:
        started = time.time()
        log.info("loading models on %s ...", self.config.models.device)
        self.engine.load()
        status = self.engine.status()
        if self.engine.ready:
            log.info("models ready (%.1f s): %s", time.time() - started, ", ".join(status.get("loaded", [])))
        else:
            log.error("models missing: %s", status.get("errors"))

    @property
    def loading(self) -> bool:
        return self._loader_thread is not None and self._loader_thread.is_alive()

    def wait_until_loaded(self, timeout: Optional[float] = None) -> bool:
        if self._loader_thread is not None:
            self._loader_thread.join(timeout)
        return not self.loading

    # ── queries ──────────────────────────────────────────────────────────
    def health(self) -> Dict[str, Any]:
        status = self.engine.status()
        status.update({
            "loading": self.loading,
            "storage": self.store.usage(),
            "rules": self.assessor.rules.as_dict(),
            "title": self.config.title,
            "language": self.config.language,
            "version": __version__,
            "preview_size": self.config.display.preview_size,
            "upload": {"max_files": self.config.upload.max_files, "max_file_mb": self.config.upload.max_file_mb},
        })
        return status

    def list_studies(self) -> Dict[str, Any]:
        self.store.purge_if_due()
        return {"studies": self.store.recent()}

    def study(self, study_id: str) -> Dict[str, Any]:
        record = self._load(study_id)
        # Studies saved before a measurement existed carry none; fill them in on the way out rather than leaving
        # the reviewer with a panel that is silently empty.
        if "criteria" not in record:
            self.assessor.recompute(record)
            self.store.save(study_id, record)
        return record

    def labels(self, study_id: str) -> Dict[str, Any]:
        return labels_of(self._load(study_id))

    def report(self, study_id: str, lang: Optional[str] = None) -> Dict[str, Any]:
        return build_report(self.study(study_id), lang or self.config.language)

    def image_path(self, study_id: str, image_id: str):
        path = self.store.image_file(study_id, image_id)
        if path is None:
            raise ServiceError(404, "Image not found.")
        return path

    # ── commands ─────────────────────────────────────────────────────────
    def require_ready(self) -> None:
        if not self.engine.ready:
            message = ("Models are still loading — try again in a moment." if self.loading
                       else "Models are not loaded — see /api/health for details.")
            raise ServiceError(503, message)

    def ingest(self, files: Iterable[UploadedFile]) -> Dict[str, Any]:
        """Read, analyse and store one study; per-file failures become the record's ``problems``."""
        self.require_ready()
        files = list(files)
        if not files:
            raise ServiceError(400, "No files were sent.")
        if len(files) > self.config.upload.max_files:
            raise ServiceError(400, f"At most {self.config.upload.max_files} files can be uploaded at once.")
        self.store.purge_if_due()

        images: List[Dict[str, Any]] = []
        renders: Dict[str, np.ndarray] = {}
        problems: List[str] = []
        seen: Dict[str, str] = {}
        max_bytes = self.config.upload.max_file_mb * 1024 * 1024

        for upload in files:
            name = upload.filename or "file"
            if upload.too_large or len(upload.data) > max_bytes:
                problems.append(f"{name}: file is too large")
                continue
            try:
                pixels, meta = read_dicom(upload.data, name)
            except DicomError as exc:
                problems.append(str(exc))
                continue
            finally:
                upload.data = b""  # the bytes are not needed past this point

            view_key = meta["view_key"]
            if view_key in seen:
                problems.append(f"{meta['filename']}: {view_key} was already uploaded as {seen[view_key]} — skipped")
                continue
            seen[view_key] = meta["filename"]

            try:
                analysis = self.engine.analyse(pixels, meta)
            except EngineError as exc:
                problems.append(f"{meta['filename']}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - reported per file, the rest of the study still goes through
                log.error("inference failed for %s: %s", meta["filename"], traceback.format_exc())
                problems.append(f"{meta['filename']}: inference failed ({exc})")
                continue

            images.append(self.assessor.build_image(view_key, meta, analysis))
            renders[view_key] = analysis["display"]

        if not images:
            raise ServiceError(422, "No image could be processed.\n" + "\n".join(problems))

        study_id = self.store.new_id()
        record = self.assessor.new_record(study_id, images, problems, self.engine.loaded_weights(),
                                          self.config.display.preview_size)
        self.store.save(study_id, record, renders)
        return record

    def put_labels(self, study_id: str, payload: Any) -> Dict[str, Any]:
        """Replace the labels a reviewer edited, then recompute every measurement."""
        record = self._load(study_id)
        try:
            cleaned = clean_labels(payload)
        except LabelError as exc:
            raise ServiceError(400, str(exc)) from None
        if not apply_labels(record, cleaned):
            raise ServiceError(400, "None of the view keys in the request match this study.")
        self.assessor.recompute(record)
        self.store.save(study_id, record)
        return record

    def put_rules(self, study_id: str, payload: Any) -> Dict[str, Any]:
        """Move the delta-D threshold for this study and re-decide every verdict."""
        record = self._load(study_id)
        try:
            record["rules"] = _clean_rules(self.assessor.rules_of(record), payload)
        except LabelError as exc:
            raise ServiceError(400, str(exc)) from None
        self.assessor.recompute(record)
        self.store.save(study_id, record)
        return record

    def delete(self, study_id: str) -> Dict[str, Any]:
        if not self.store.delete(study_id):
            raise ServiceError(404, "Study not found.")
        return {"deleted": study_id}

    def delete_all(self) -> Dict[str, Any]:
        """Empty the archive. The renders and labels go; nothing else was ever kept."""
        removed = self.store.delete_all()
        log.info("cleared %d studies", removed)
        return {"deleted": removed}

    def _load(self, study_id: str) -> Dict[str, Any]:
        record = self.store.load(study_id)
        if record is None:
            raise ServiceError(404, "Study not found.")
        return record
