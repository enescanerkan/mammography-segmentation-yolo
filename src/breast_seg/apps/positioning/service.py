"""The review application's behaviour, independent of HTTP.

:class:`PositioningService` wires the engine (inference), the assessor (measurements, criteria, verdicts) and the
store (renders + label JSON). The FastAPI layer in :mod:`.api` only translates requests into these calls, so the
whole flow can be exercised — and tested — without a web server.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

from ... import __version__
from ...imaging import DicomError, read_dicom
from ...imaging.dicom import ERROR_CODES as DICOM_ERROR_CODES
from ...inference import BrightenSearch, EngineError, LandmarkEngine, ModelLoader
from ...positioning import (
    CRITERIA_VERSION,
    CriteriaGrader,
    LabelError,
    PositioningAssessor,
    apply_labels,
    build_report,
    clean_labels,
)
from ...positioning import clean_rules as _clean_rules
from ...positioning.study import labels_of
from ...registry import Registry
from .config import AppConfig
from .storage import CountChanged, StudyStore, is_study_key
from .uploads import UploadedFile

log = logging.getLogger("breast_seg.positioning_app")

REGRADED_CACHE_SIZE = 64  # regraded old-version records kept in memory (see PositioningService.study)


class Engine(Protocol):
    """What the service needs from an inference engine (see :class:`~breast_seg.inference.LandmarkEngine`)."""

    @property
    def ready(self) -> bool: ...

    def load(self) -> Any: ...

    def status(self) -> Dict[str, Any]: ...

    def loaded_weights(self) -> Dict[str, str]: ...

    def analyse(self, pixels: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]: ...


# Every refusal a reader can meet, by code. The answer carries the code and its parameters next to the English
# sentence, so the interface translates by code (i18n.js ``ERRORS_TR``) instead of matching the wording; English, or
# a code the page does not know, shows the sentence as it is. A test checks that every code has a translation.
ERRORS: Dict[str, str] = {
    "study_not_found": "Study not found.",
    "image_not_found": "Image not found.",
    "models_loading": "Models are still loading — try again in a moment.",
    "models_missing": "Models are not loaded — see /api/health for details.",
    "no_files": "No files were sent.",
    "too_many_files": "At most {limit} files can be uploaded at once.",
    "nothing_processed": "No image could be processed.",
    "only_dicom": "This application reads DICOM (.dcm) only; PNG/JPG files have no pixel spacing or view "
                  "information, so mm cannot be measured. Not used: {names}.",
    "clear_changed": "{stored} studies are stored now, not {expected}; nothing was deleted. "
                     "Refresh the list and try again.",
}

# Why one file of an upload was not used, by code: the reader's reasons (dicom.py) and the service's own. When no
# file of an upload could be used, the refusal lists them with their codes (``params["files"]``).
PROBLEMS: Tuple[str, ...] = (*DICOM_ERROR_CODES, "file_too_large", "duplicate_view", "analysis_refused",
                             "inference_failed")

# The failures another attempt with the same files can change: a codec installed since, the models refusing or failing
# for the moment (busy, out of memory). Every other reason is in the bytes or the settings and gives the same result
# every time (a picture, an unsupported view, a multi-frame file, a file over the size limit), and so does the
# client's file limit: a study with only such failures is complete as stored, and found again by its key.
RETRYABLE: Tuple[str, ...] = ("undecodable", "analysis_refused", "inference_failed")


class ServiceError(Exception):
    """A request the service refuses; ``status`` is the HTTP status to answer with, ``code`` (a key of
    :data:`ERRORS`, empty for a refusal only a programmer meets) and ``params`` what the interface translates by."""

    def __init__(self, status: int, message: str, code: str = "", **params: Any) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.params = params


def refusal(status: int, code: str, **params: Any) -> ServiceError:
    """The refusal ``code`` of :data:`ERRORS`, its sentence filled in with ``params``."""
    return ServiceError(status, ERRORS[code].format(**params), code, **params)


def _problem(text: str, code: str, **params: Any) -> Dict[str, Any]:
    """One file's reason as the refusal lists it (and the record keeps it, ``problem_codes``): the English line, and
    its code and parameters."""
    return {"text": text, "code": code, "params": params}


class PositioningService:
    """Upload, analyse, store, edit and report positioning studies."""

    def __init__(self, config: AppConfig, engine: Engine, store: StudyStore,
                 assessor: Optional[PositioningAssessor] = None) -> None:
        self.config = config
        self.engine = engine
        self.store = store
        self.assessor = assessor or PositioningAssessor(config.rules.to_rules(), CriteriaGrader(),
                                                        contour_fallback=config.nipple.contour)
        self._loader_thread: Optional[threading.Thread] = None
        # study id -> ((criteria version, stored file stamp), regraded record); see :meth:`study`
        self._regraded: OrderedDict[str, Tuple[Tuple[int, Optional[Tuple[int, int]]], Dict[str, Any]]] = OrderedDict()
        self._regraded_lock = threading.Lock()
        self.started = time.time()

    @classmethod
    def from_config(cls, config: AppConfig, engine: Optional[Engine] = None,
                    loader: Optional[ModelLoader] = None) -> PositioningService:
        """Build the default engine from the registry named in ``config`` unless one is injected; the app's own
        nipple steps (``config.nipple``) are switched on here, for this engine only."""
        if engine is None:
            registry = Registry.load(config.registry_path, weights_root=config.weights_root)
            nipple = config.nipple
            brighten = BrightenSearch(nipple.window_mm, nipple.max_gap_mm) if nipple.brighten else None
            engine = LandmarkEngine(registry, device=config.models.device,
                                    preview_size=config.display.preview_size, loader=loader, brighten=brighten)
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
            # The browser keys a dataset's studies with it (HMAC of the study UID, so the UID itself is neither
            # sent nor stored). Not a secret: every client gets it, and with it the key of any UID it knows.
            "dataset": {"key_salt": self.store.salt()},
        })
        return status

    def list_studies(self, limit: int = 40) -> Dict[str, Any]:
        """The ``limit`` most recent studies somebody opened, the number stored in all (``total``) and
        how many of those were analysed ahead and not opened yet (``unopened``); ``limit=0`` only counts."""
        self.store.purge_if_due()
        rows, total, unopened = self.store.listing(min(max(int(limit), 0), 200))
        return {"studies": rows, "total": total, "unopened": unopened}

    def mark_opened(self, study_id: str) -> Dict[str, Any]:
        """A client has opened the study for review: it is listed among the recent studies from now on."""
        if not self.store.mark_opened(study_id):
            raise refusal(404, "study_not_found")
        return {"opened": study_id}

    def lookup(self, payload: Any) -> Dict[str, Any]:
        """``{"keys": [...]}`` -> ``{"found": {key: study id}}`` for the study keys a stored study carries (a study
        built from an incomplete upload is left out unless it was edited; see :meth:`StudyStore.find_keys`)."""
        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list) or len(keys) > 20000:
            raise ServiceError(400, "Expected an object with a 'keys' list.")
        return {"found": self.store.find_keys(key for key in keys if isinstance(key, str))}

    def study(self, study_id: str) -> Dict[str, Any]:
        """The study record, regraded in memory when it was saved under an older criteria set (read-only result)."""
        stamp = (CRITERIA_VERSION, self.store.stamp(study_id))
        with self._regraded_lock:
            cached = self._regraded.get(study_id)
            if cached is not None and cached[0] == stamp and stamp[1] is not None:
                self._regraded.move_to_end(study_id)
                return cached[1]
        record = self._load(study_id)
        stored = (record.get("criteria") or {}).get("version")
        if not isinstance(stored, int) or stored < CRITERIA_VERSION:
            # Saved under an older criteria set, or before criteria existed: regrade without writing back, so a read
            # can never overwrite a concurrent edit, and the stored ΔD and verdicts stay as they were measured. The
            # next edit (labels or threshold) saves a fully recomputed record anyway. The result is kept in memory
            # for as long as the stored file does not change, so a study is regraded once, not on every request; a
            # failed regrade is kept too (the stored record), so it is logged once.
            try:
                upgraded = copy.deepcopy(record)
                if "summary" in upgraded:
                    self.assessor.regrade(upgraded)
                else:  # never measured at all: there is no stored ΔD or verdict to preserve
                    self.assessor.recompute(upgraded)
                record = upgraded
            except Exception:  # noqa: BLE001 - the stored record is still worth showing
                log.warning("could not regrade study %s, serving it as stored: %s", study_id, traceback.format_exc())
            with self._regraded_lock:
                self._regraded[study_id] = (stamp, record)
                self._regraded.move_to_end(study_id)
                while len(self._regraded) > REGRADED_CACHE_SIZE:
                    self._regraded.popitem(last=False)
        return record

    def labels(self, study_id: str) -> Dict[str, Any]:
        return labels_of(self._load(study_id))

    def report(self, study_id: str, lang: Optional[str] = None) -> Dict[str, Any]:
        return build_report(self.study(study_id), lang or self.config.language)

    def image_path(self, study_id: str, image_id: str):
        path = self.store.image_file(study_id, image_id)
        if path is None:
            raise refusal(404, "image_not_found")
        return path

    # ── commands ─────────────────────────────────────────────────────────
    def require_ready(self) -> None:
        if not self.engine.ready:
            raise refusal(503, "models_loading" if self.loading else "models_missing")

    def ingest(self, files: Iterable[UploadedFile], study_key: Optional[str] = None, unopened: bool = False,
               left_out: Any = None) -> Dict[str, Any]:
        """Read, analyse and store one study; per-file failures become the record's ``problems``.

        What a dataset upload adds, all kept next to the record and not in it:

        * ``study_key`` (see :meth:`StudyStore.salt`) lets the dataset find this study again instead of uploading
          it a second time.
        * ``unopened``: the study is analysed ahead of the reviewer; it is not listed among the recent studies until
          a client opens it (:meth:`mark_opened`).
        * ``left_out``: how many files of the study the client did not send (the per-upload limit); validated, and
          otherwise not used: the same folder always leaves out the same files.

        A study with a file that failed in a way a retry can change (:data:`RETRYABLE`) is *incomplete*: its key is
        not offered to the next lookup, so the folder is analysed afresh, and this upload replaces an older
        incomplete record of the same key that nobody edited. A failure that gives the same result every time (a
        picture, an unsupported view, a file over the size limit), the files left out by the limit and the second
        file of a view do not make a study incomplete: analysing it again could only give the same record.
        """
        self.require_ready()
        if study_key is not None and not is_study_key(study_key):
            raise ServiceError(400, "The study key must be 64 hexadecimal characters.")
        try:
            missing = int(left_out or 0)
        except (TypeError, ValueError):
            missing = -1
        if missing < 0:
            raise ServiceError(400, "The number of files left out must be a whole number.")
        files = list(files)
        if not files:
            raise refusal(400, "no_files")
        if len(files) > self.config.upload.max_files:
            raise refusal(400, "too_many_files", limit=self.config.upload.max_files)
        self.store.purge_if_due()

        images: List[Dict[str, Any]] = []
        renders: Dict[str, np.ndarray] = {}
        problems: List[str] = []
        coded: List[Dict[str, Any]] = []  # the same reasons with their codes, for the refusal when nothing is usable

        def note(text: str, code: str, **params: Any) -> None:
            problems.append(text)
            coded.append(_problem(text, code, **params))

        seen: Dict[str, str] = {}
        max_bytes = self.config.upload.max_file_mb * 1024 * 1024

        for upload in files:
            name = upload.filename or "file"
            if upload.too_large or len(upload.data) > max_bytes:
                note(f"{name}: file is too large", "file_too_large", file=name)
                continue
            try:
                pixels, meta = read_dicom(upload.data, name)
            except DicomError as exc:
                note(str(exc), exc.code, **exc.params)
                continue
            finally:
                upload.data = b""  # the bytes are not needed past this point

            view_key = meta["view_key"]
            if view_key in seen:
                note(f"{meta['filename']}: {view_key} was already uploaded as {seen[view_key]} — skipped",
                     "duplicate_view", file=meta["filename"], view=view_key, first=seen[view_key])
                continue
            seen[view_key] = meta["filename"]

            try:
                analysis = self.engine.analyse(pixels, meta)
            except EngineError as exc:
                note(f"{meta['filename']}: {exc}", "analysis_refused", file=meta["filename"], why=str(exc))
                continue
            except Exception as exc:  # noqa: BLE001 - reported per file, the rest of the study still goes through
                log.error("inference failed for %s: %s", meta["filename"], traceback.format_exc())
                note(f"{meta['filename']}: inference failed ({exc})", "inference_failed",
                     file=meta["filename"], why=str(exc))
                continue

            images.append(self.assessor.build_image(view_key, meta, analysis))
            renders[view_key] = analysis["display"]

        if not images:
            if coded and all(entry["code"] == "raster_image" for entry in coded):
                # Only pictures were sent: one sentence says why, instead of the same reason once per file.
                raise refusal(422, "only_dicom", names=", ".join(entry["params"]["file"] for entry in coded))
            raise ServiceError(422, "\n".join([ERRORS["nothing_processed"], *problems]), "nothing_processed",
                               files=coded)

        study_id = self.store.new_id()
        record = self.assessor.new_record(study_id, images, problems, self.engine.loaded_weights(),
                                          self.config.display.preview_size)
        # The notes again with their codes, so the page and the report write them in the reader's language; a
        # record stored before this has only the English sentences, and they are shown as they are.
        record["problem_codes"] = coded
        retry = any(entry["code"] in RETRYABLE for entry in coded)
        # Written complete with its tags, or not at all: the record never exists unkeyed or looking opened.
        self.store.save(study_id, record, renders, key=study_key, unopened=bool(unopened),
                        incomplete=bool(study_key) and retry)
        if study_key:
            for stale in self.store.delete_replaceable(study_key, keep=study_id):
                self._forget_regraded(stale)
        return record

    def put_labels(self, study_id: str, payload: Any) -> Dict[str, Any]:
        """Replace the labels a reviewer edited, then recompute every measurement."""
        try:
            cleaned = clean_labels(payload)
        except LabelError as exc:
            raise ServiceError(400, str(exc)) from None

        def change(record: Dict[str, Any]) -> None:
            if not apply_labels(record, cleaned):
                raise ServiceError(400, "None of the view keys in the request match this study.")
            self.assessor.recompute(record)

        # Read, changed and written in one step under the store's write lock (StudyStore.update).
        record = self.store.update(study_id, change)
        if record is None:
            raise refusal(404, "study_not_found")
        self.store.mark_opened(study_id)  # an edit is a review, whatever route it came by
        self._forget_regraded(study_id)
        return record

    def clean_view(self, study_id: str, payload: Any) -> Dict[str, Any]:
        """The editor's "Smooth / clean labels": ``{"view_key", "polygons"}`` (the contours the editor holds, edited
        or not) -> ``{"view_key", "polygons", "changed"}`` by :func:`breast_seg.positioning.label_cleanup.clean_view`.

        Nothing is stored: the editor shows the result as an unsaved edit, which Save then sends like any other.
        ``changed`` (the classes whose contour was replaced or removed) is decided on label maps: when the cleaned
        contours paint the same pixels as the ones sent, it is empty and the contours come back as they were sent;
        so it is for a view that is still exactly what a clean returned.
        """
        from ...positioning.label_cleanup import clean_view
        from ...positioning.labels import EDITABLE_CLASSES, clean_polygon

        record = self._load(study_id)
        view_key = payload.get("view_key") if isinstance(payload, dict) else None
        image = next((entry for entry in record.get("images", []) if entry.get("view_key") == view_key), None)
        if image is None:
            raise refusal(404, "image_not_found")
        raw = payload.get("polygons")
        if raw is not None and not isinstance(raw, dict):
            raise ServiceError(400, "'polygons' has the wrong type.")
        polygons = {name: polygon for name, polygon in ((name, clean_polygon(shape)) for name, shape in (raw or {}).items()
                                                        if name in EDITABLE_CLASSES) if polygon}
        render = None
        path = self.store.image_file(study_id, view_key)
        if path is not None:
            import cv2

            render = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # the frame the contours are drawn on
        cleaned, changed = clean_view(polygons, image.get("view", "MLO"), render)
        return {"view_key": view_key, "polygons": cleaned, "changed": changed}

    def put_rules(self, study_id: str, payload: Any) -> Dict[str, Any]:
        """Move the delta-D threshold for this study and re-decide every verdict."""
        def change(record: Dict[str, Any]) -> None:
            try:
                record["rules"] = _clean_rules(self.assessor.rules_of(record), payload)
            except LabelError as exc:
                raise ServiceError(400, str(exc)) from None
            self.assessor.recompute(record)

        record = self.store.update(study_id, change)
        if record is None:
            raise refusal(404, "study_not_found")
        self.store.mark_opened(study_id)
        self._forget_regraded(study_id)
        return record

    def delete(self, study_id: str) -> Dict[str, Any]:
        self._forget_regraded(study_id)
        if not self.store.delete(study_id):
            raise refusal(404, "study_not_found")
        return {"deleted": study_id}

    def delete_all(self, expected: Optional[int] = None) -> Dict[str, Any]:
        """Empty the archive. The renders and labels go; nothing else was ever kept.

        ``expected`` is the count the user confirmed; when the store holds a different number (a study arrived or
        went since), nothing is deleted. The store counts and deletes in one step, so a study stored meanwhile is
        neither deleted uncounted nor cut off while it is being written.
        """
        try:
            removed = self.store.delete_all(expected)
        except CountChanged as exc:
            raise refusal(409, "clear_changed", stored=exc.stored, expected=expected) from None
        self._forget_regraded()
        log.info("cleared %d studies", removed)
        return {"deleted": removed}

    def _forget_regraded(self, study_id: Optional[str] = None) -> None:
        """Drop the cached regrade of one study (all when None): its stored record was rewritten or removed."""
        with self._regraded_lock:
            if study_id is None:
                self._regraded.clear()
            else:
                self._regraded.pop(study_id, None)

    def _load(self, study_id: str) -> Dict[str, Any]:
        record = self.store.load(study_id)
        if record is None:
            raise refusal(404, "study_not_found")
        return record
