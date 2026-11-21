"""What the server keeps: one display PNG per view and one JSON of labels per study.

The uploaded DICOM is never written. A study directory therefore holds nothing that identifies a patient — no
name, ID, accession, UID or date — only the rendered pixels and the coordinates the models produced::

    <data_dir>/studies/<study id>/
        R-MLO.png  L-MLO.png  R-CC.png  L-CC.png    display renders
        study.json                                   labels, measurements, criteria, verdicts
        study.key                                    only for a study uploaded from a dataset: its study key
        study.unopened                               empty; there until the study is opened for review
        study.opened                                 empty; its time is when the study was last opened
        study.incomplete                             empty; a file of the study failed in a way a retry can change
    <data_dir>/study_key_salt                        this installation's random salt for study keys

A study key is a salted hash the browser makes (see :meth:`StudyStore.salt`); it lets the same dataset folder find
its studies again without the StudyInstanceUID being sent or stored.

Every change to the set of studies (a new study, an edit, a deletion) happens under one write lock. A new study is
written into a hidden folder first, its small files and its record included, and renamed into place: no reader ever
sees a study without its tags, a "clear" never counts or deletes one that is still being written, and a crash leaves
only a hidden folder that the retention purge removes.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import cv2
import numpy as np

from ...positioning.study import view_rank

RECORD_FILE = "study.json"
KEY_FILE = "study.key"  # the study key of a dataset study (see StudyStore.salt), absent otherwise
UNOPENED_FILE = "study.unopened"  # empty: analysed ahead by a dataset, nobody has opened it for review yet
INCOMPLETE_FILE = "study.incomplete"  # empty: a file of the study failed in a way another attempt can change
OPENED_FILE = "study.opened"  # empty: its modification time is when a client last opened the study
SALT_FILE = "study_key_salt"
PENDING_PREFIX = ".pending-"  # a new study being written, renamed into place once complete
PENDING_MAX_AGE_S = 3600  # a pending folder older than this was left by a crash and is removed by the purge


@dataclasses.dataclass
class _Index:
    """What the small files next to the records say, read once and kept in step by the store's own writes."""

    key_of: Dict[str, str] = dataclasses.field(default_factory=dict)  # study id -> study key
    ids_of: Dict[str, List[str]] = dataclasses.field(default_factory=dict)  # study key -> its ids, oldest first
    created_of: Dict[str, float] = dataclasses.field(default_factory=dict)  # keyed study id -> its creation time
    unopened: Set[str] = dataclasses.field(default_factory=set)
    incomplete: Set[str] = dataclasses.field(default_factory=set)

    def add_key(self, study_id: str, key: str, created: float) -> None:
        """File ``study_id`` under ``key``, in creation order (the record's own time, not the order it was read in,
        so a restart never changes which study of a key is the newest)."""
        if study_id in self.key_of:
            return
        self.key_of[study_id] = key
        self.created_of[study_id] = float(created)
        ids = self.ids_of.setdefault(key, [])
        ids.append(study_id)
        ids.sort(key=lambda other: (self.created_of.get(other, 0.0), other))

    def drop(self, study_id: str) -> None:
        self.created_of.pop(study_id, None)
        key = self.key_of.pop(study_id, None)
        if key is not None:
            remaining = [other for other in self.ids_of.get(key, []) if other != study_id]
            if remaining:
                self.ids_of[key] = remaining
            else:
                self.ids_of.pop(key, None)
        self.unopened.discard(study_id)
        self.incomplete.discard(study_id)


def is_study_key(value: Any) -> bool:
    """A study key is the 64 lower-case hex characters of an HMAC-SHA-256."""
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def attention(summary: Dict[str, Any]) -> Optional[str]:
    """Why a study needs a look, most serious first: ``"disagree"`` (the two methods reach different verdicts),
    ``"wide"`` (same verdict, measurements far apart), ``"partial"`` (only one method produced a result), or None.
    A breast with no measurable pair at all (a lone view) is not marked: the row's dot already shows no verdict.
    """
    comparisons = [side.get("comparison") or {} for side in summary.get("sides") or []]
    if any(c.get("state") == "disagree" for c in comparisons):
        return "disagree"
    if any(c.get("wide_gap") for c in comparisons):
        return "wide"
    if any(c.get("state") == "partial" for c in comparisons):
        return "partial"
    return None


class StudyStore:
    """Study directories under ``root / "studies"``, with a retention window.

    Args:
        root: the data directory.
        retention_days: studies older than this are purged; ``0`` disables purging.
        purge_every_s: minimum interval between two purges triggered by :meth:`purge_if_due`.
    """

    def __init__(self, root: Path, retention_days: int = 30, purge_every_s: int = 3600) -> None:
        self.root = Path(root)
        self.study_dir = self.root / "studies"
        self.retention_days = int(retention_days)
        self.purge_every_s = int(purge_every_s)
        self._last_purge = 0.0
        self._salt = ""
        self._index: Optional[_Index] = None
        self._lock = threading.Lock()  # the index and the salt
        # Every change to the set of studies or to a record: creating, editing, deleting. Taken before ``_lock``,
        # never while holding it.
        self._write = threading.RLock()
        self.study_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex[:12]

    # ── paths ────────────────────────────────────────────────────────────
    def path(self, study_id: str) -> Optional[Path]:
        """The study directory, refusing anything that is not a plain alphanumeric id."""
        if not study_id or not study_id.isalnum() or len(study_id) > 32:
            return None
        path = self.study_dir / study_id
        return path if path.is_dir() else None

    def image_file(self, study_id: str, image_id: str) -> Optional[Path]:
        directory = self.path(study_id)
        if directory is None or not image_id or not image_id.replace("-", "").isalnum():
            return None
        path = directory / f"{image_id}.png"
        return path if path.is_file() else None

    # ── read / write ─────────────────────────────────────────────────────
    def save(self, study_id: str, record: Mapping[str, Any], renders: Optional[Mapping[str, np.ndarray]] = None,
             key: Optional[str] = None, unopened: bool = False, incomplete: bool = False) -> None:
        """Write a study: a new one complete with its tags (``key``, ``unopened``, ``incomplete``, see :meth:`tag`)
        or not at all, an existing one by replacing its renders (if given) and its record.

        A new study is written into a hidden folder, its record last, and renamed into place under the write lock,
        so nobody sees it half written or without its tags. Its creation time is the record's ``created``.
        """
        if not study_id.isalnum():
            raise ValueError("study ids are alphanumeric")
        directory = self.study_dir / study_id
        with self._write:
            if directory.is_dir():
                self._write_into(directory, record, renders)
                if key is not None or unopened or incomplete:
                    self.tag(study_id, key=key, unopened=unopened, incomplete=incomplete)
                return
        pending = self.study_dir / f"{PENDING_PREFIX}{study_id}-{uuid.uuid4().hex[:6]}"
        pending.mkdir(parents=True)
        try:
            keyed = key is not None and is_study_key(key)
            if keyed:
                (pending / KEY_FILE).write_text(str(key), encoding="ascii")
            if unopened:
                (pending / UNOPENED_FILE).touch()
            if incomplete:
                (pending / INCOMPLETE_FILE).touch()
            self._write_into(pending, record, renders)  # the record last
            with self._write:
                if directory.exists():
                    raise FileExistsError(f"study {study_id} exists already")
                pending.rename(directory)
                with self._lock:
                    if self._index is not None:
                        if keyed:
                            self._index.add_key(study_id, str(key), _created(record, directory))
                        if unopened:
                            self._index.unopened.add(study_id)
                        if incomplete:
                            self._index.incomplete.add(study_id)
        finally:
            if pending.exists():
                shutil.rmtree(pending, ignore_errors=True)

    @staticmethod
    def _write_into(directory: Path, record: Mapping[str, Any], renders: Optional[Mapping[str, np.ndarray]]) -> None:
        for image_id, image in (renders or {}).items():
            cv2.imwrite(str(directory / f"{image_id}.png"), image)
        tmp = directory / (RECORD_FILE + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(directory / RECORD_FILE)

    def update(self, study_id: str, change: Callable[[Dict[str, Any]], Any]) -> Optional[Dict[str, Any]]:
        """Read the record, ``change`` it in place and write it back, all under the write lock: an edit can neither
        be lost to a deletion that checked the record before it was written, nor bring a deleted study back as a
        folder with a record and no renders. None (nothing written) when the study is gone."""
        with self._write:
            record = self.load(study_id)
            if record is None:
                return None
            change(record)
            self._write_into(self.study_dir / study_id, record, None)
            return record

    def load(self, study_id: str) -> Optional[Dict[str, Any]]:
        directory = self.path(study_id)
        if directory is None:
            return None
        try:
            return json.loads((directory / RECORD_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def stamp(self, study_id: str) -> Optional[Tuple[int, int]]:
        """Changes whenever the stored record is rewritten (modification time and size); None if there is none."""
        directory = self.path(study_id)
        if directory is None:
            return None
        try:
            stat = (directory / RECORD_FILE).stat()
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def delete(self, study_id: str) -> bool:
        with self._write:
            directory = self.path(study_id)
            if directory is None:
                return False
            self._remove(directory)
        return True

    def _remove(self, directory: Path) -> None:
        """Delete one study folder and forget it (call with the write lock held)."""
        shutil.rmtree(directory, ignore_errors=True)
        with self._lock:
            if self._index is not None:
                self._index.drop(directory.name)

    def delete_all(self, expected: Optional[int] = None) -> int:
        """Delete every stored study and return how many. With ``expected``, only if exactly that many are stored:
        otherwise nothing is deleted and :class:`CountChanged` says how many there are. Counting and deleting are
        one step under the write lock, and a study still being written is neither counted nor deleted."""
        with self._write:
            directories = self._directories()
            if expected is not None and len(directories) != expected:
                raise CountChanged(len(directories))
            for directory in directories:
                shutil.rmtree(directory, ignore_errors=True)
            with self._lock:
                self._index = None  # read again (from an empty folder) when it is next needed
        return len(directories)

    def count(self) -> int:
        """How many studies are stored (what :meth:`delete_all` would remove); no record is read."""
        return len(self._directories())

    def recent(self, limit: int = 40) -> List[Dict[str, Any]]:
        """The ``limit`` most recent studies that were opened, most recent first; see :meth:`listing`."""
        return self.listing(limit)[0]

    def listing(self, limit: int = 40) -> Tuple[List[Dict[str, Any]], int, int]:
        """``(rows, total, unopened)``: just enough of the ``limit`` most recent studies to draw a list row each,
        most recent first; how many studies are stored in all; and how many of those nobody has opened yet.

        A study a dataset analysed ahead is not listed until it is opened for review (:meth:`mark_opened`): the
        list is of studies somebody looked at. One pass over the directory; only the ``limit`` most recent listed
        records are read and parsed, so a long archive does not slow the landing page down, and ``limit=0`` only
        counts. Recent is the later of when the record was written and when the study was last opened, so a study
        analysed ahead long ago and opened now is among them.
        """
        with self._lock:
            unopened = set(self._indexed().unopened)
        stamped = [(_recency(directory), directory) for directory in self._directories()]
        total = len(stamped)
        hidden = sum(1 for _, directory in stamped if directory.name in unopened)
        stamped.sort(key=lambda entry: entry[0], reverse=True)
        rows: List[Dict[str, Any]] = []
        for _, directory in stamped:
            if len(rows) >= max(int(limit), 0):
                break
            if directory.name in unopened:
                continue
            record = self.load(directory.name)
            if not record:
                continue
            summary = record.get("summary") or {}
            rows.append({
                "id": record.get("id", directory.name),
                "created": record.get("created", 0),
                "images": len(record.get("images", [])),
                "overall": summary.get("overall"),
                "flagged": bool(summary.get("flagged")),
                "flag": attention(summary),
                "views": sorted((image.get("view_key") for image in record.get("images", [])), key=view_rank),
            })
        return rows, total, hidden

    # ── study keys ───────────────────────────────────────────────────────
    # A dataset is a folder of studies the browser groups by StudyInstanceUID. To find a study it has stored before
    # without the UID being sent or stored, the browser sends an HMAC-SHA-256 under this installation's salt of the
    # UID (with the number and size of the study's files), and the server only ever compares keys for equality.
    #
    # What the salt does: it keeps the UID itself out of storage, and it makes the keys of two installations
    # incomparable. What it does not do: hide anything from a client of this API. The salt is handed to every page
    # (/api/health), so anyone who can call the API can compute the key of a UID they already know and ask whether
    # it is stored; the API has no authentication and lists every study anyway.
    def salt(self) -> str:
        """This installation's random study-key salt (64 hex characters), created on first use."""
        if self._salt:
            return self._salt
        path = self.root / SALT_FILE
        with self._lock:
            try:
                salt = path.read_text(encoding="ascii").strip()
            except OSError:
                salt = ""
            if not is_study_key(salt):
                salt = secrets.token_hex(32)
                self.root.mkdir(parents=True, exist_ok=True)
                path.write_text(salt, encoding="ascii")
                with contextlib.suppress(OSError):  # a file system without permissions
                    path.chmod(0o600)
            self._salt = salt
        return salt

    def _indexed(self) -> _Index:
        """The index of the small files, read from disk the first time (call with the lock held).

        One walk over the study folders for the lifetime of the store: after it, a lookup or a listing costs no
        directory walk of its own, and :meth:`tag`, :meth:`mark_opened`, :meth:`delete` and the retention purge
        keep it in step.
        """
        if self._index is None:
            index = _Index()
            for directory in self._directories():
                study_id = directory.name
                try:
                    key = (directory / KEY_FILE).read_text(encoding="ascii").strip()
                except OSError:
                    key = ""
                if is_study_key(key):  # only a keyed study's record is read, for its creation time
                    index.add_key(study_id, key, _created(self.load(study_id), directory))
                if (directory / UNOPENED_FILE).exists():
                    index.unopened.add(study_id)
                if (directory / INCOMPLETE_FILE).exists():
                    index.incomplete.add(study_id)
            self._index = index
        return self._index

    def tag(self, study_id: str, key: Optional[str] = None, unopened: bool = False, incomplete: bool = False) -> None:
        """What a dataset upload adds to a freshly stored study: its study key, that nobody has opened it yet, and
        that some of its files could not be used (small files next to the record, never inside it)."""
        with self._write:
            directory = self.path(study_id)
            if directory is None:
                return
            keyed = key is not None and is_study_key(key)
            if keyed:
                (directory / KEY_FILE).write_text(str(key), encoding="ascii")
            if unopened:
                (directory / UNOPENED_FILE).touch()
            if incomplete:
                (directory / INCOMPLETE_FILE).touch()
            created = _created(self.load(study_id), directory) if keyed else 0.0
            with self._lock:
                index = self._indexed()  # when read just now, it has the files written above already
                if keyed:
                    index.add_key(study_id, str(key), created)
                if unopened:
                    index.unopened.add(study_id)
                if incomplete:
                    index.incomplete.add(study_id)

    def mark_opened(self, study_id: str) -> bool:
        """A client has put the study on screen for review (or edited it): it belongs in the recent list from now
        on. False when there is no such study."""
        with self._write:  # never a file written into a folder that is being deleted
            directory = self.path(study_id)
            if directory is None:
                return False
            with contextlib.suppress(OSError):
                (directory / OPENED_FILE).touch()  # when it was opened: the recent list counts it from now
            with contextlib.suppress(OSError):
                (directory / UNOPENED_FILE).unlink()
            with self._lock:
                if self._index is not None:
                    self._index.unopened.discard(study_id)
        return True

    def _edited(self, study_id: str) -> bool:
        record = self.load(study_id)
        return bool(record and record.get("edited"))

    def find_keys(self, keys: Iterable[str]) -> Dict[str, str]:
        """``{key: study id}`` for those of ``keys`` a stored study carries; the newest such study wins.

        A study built from an incomplete upload is not offered again (the folder is analysed afresh instead),
        unless a reviewer has edited it: corrections are never passed over. No directory walk: the index answers.
        """
        wanted = {key for key in keys if is_study_key(key)}
        with self._lock:
            index = self._indexed()
            candidates = {key: list(reversed(index.ids_of[key])) for key in wanted if key in index.ids_of}
            incomplete = set(index.incomplete)
        found: Dict[str, str] = {}
        for key, study_ids in candidates.items():
            for study_id in study_ids:
                if study_id not in incomplete or self._edited(study_id):
                    found[key] = study_id
                    break
        return found

    def delete_replaceable(self, key: str, keep: str) -> List[str]:
        """Delete the other studies under ``key`` a new upload makes pointless: built from an incomplete upload and
        never edited (so reopening a folder does not pile up one more copy every time); returns their ids.

        Checked and deleted under the write lock, the "edited" mark read at the moment of deleting: an edit saved
        meanwhile keeps its study (:meth:`update` takes the same lock).
        """
        removed = []
        with self._write:
            with self._lock:
                index = self._indexed()
                others = [study_id for study_id in index.ids_of.get(key, [])
                          if study_id != keep and study_id in index.incomplete]
            for study_id in others:
                directory = self.path(study_id)
                if directory is not None and not self._edited(study_id):
                    self._remove(directory)
                    removed.append(study_id)
        return removed

    # ── retention ────────────────────────────────────────────────────────
    def purge_expired(self, now: Optional[float] = None) -> int:
        """Delete studies past the retention window. Zero days disables this."""
        if self.retention_days <= 0:
            return 0
        now = time.time() if now is None else now
        cutoff = now - self.retention_days * 86400
        removed = 0
        for directory in self._directories():
            with self._write:
                if not directory.is_dir():
                    continue
                if _created(self.load(directory.name), directory) < cutoff:
                    self._remove(directory)
                    removed += 1
        for pending in self.study_dir.glob(f"{PENDING_PREFIX}*"):  # left by a crash while a study was written
            with contextlib.suppress(OSError):
                if now - pending.stat().st_mtime > PENDING_MAX_AGE_S:
                    shutil.rmtree(pending, ignore_errors=True)
        return removed

    def purge_if_due(self) -> int:
        """:meth:`purge_expired` at most once per ``purge_every_s``, so any request can afford to call it.

        Purging only at start-up would let a server that stays up for weeks keep studies far past the retention
        window it advertises.
        """
        now = time.time()
        with self._lock:
            if now - self._last_purge < self.purge_every_s:
                return 0
            self._last_purge = now
        return self.purge_expired(now)

    def usage(self) -> Dict[str, Any]:
        total = 0
        count = 0
        for directory in self._directories():
            count += 1
            for entry in directory.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        return {"studies": count, "bytes": total, "retention_days": self.retention_days}

    def _directories(self) -> List[Path]:
        """The stored studies' folders (not the hidden ones of studies still being written)."""
        if not self.study_dir.is_dir():
            return []
        return [d for d in self.study_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]


class CountChanged(Exception):
    """:meth:`StudyStore.delete_all` was told another number of studies than are stored; nothing was deleted."""

    def __init__(self, stored: int) -> None:
        super().__init__(f"{stored} studies are stored")
        self.stored = stored


def _created(record: Optional[Mapping[str, Any]], directory: Path) -> float:
    """A study's creation time: the record's ``created``, else (an older or unreadable record) its file's time."""
    created = (record or {}).get("created")
    if isinstance(created, (int, float)) and not isinstance(created, bool):
        return float(created)
    for path in (directory / RECORD_FILE, directory):
        with contextlib.suppress(OSError):
            return path.stat().st_mtime
    return 0.0


def _recency(directory: Path) -> int:
    """When a study was last written or opened, whichever is later (nanoseconds)."""
    latest = 0
    for name in (RECORD_FILE, OPENED_FILE):
        with contextlib.suppress(OSError):
            latest = max(latest, (directory / name).stat().st_mtime_ns)
    return latest
