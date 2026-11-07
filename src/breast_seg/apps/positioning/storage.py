"""What the server keeps: one display PNG per view and one JSON of labels per study.

The uploaded DICOM is never written. A study directory therefore holds nothing that identifies a patient — no
name, ID, accession, UID or date — only the rendered pixels and the coordinates the models produced::

    <data_dir>/studies/<study id>/
        R-MLO.png  L-MLO.png  R-CC.png  L-CC.png    display renders
        study.json                                   labels, measurements, criteria, verdicts
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import cv2
import numpy as np

RECORD_FILE = "study.json"


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
        self._lock = threading.Lock()
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
    def save(self, study_id: str, record: Mapping[str, Any], renders: Optional[Mapping[str, np.ndarray]] = None) -> None:
        if not study_id.isalnum():
            raise ValueError("study ids are alphanumeric")
        directory = self.study_dir / study_id
        directory.mkdir(parents=True, exist_ok=True)
        for image_id, image in (renders or {}).items():
            cv2.imwrite(str(directory / f"{image_id}.png"), image)
        tmp = directory / (RECORD_FILE + ".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(directory / RECORD_FILE)

    def load(self, study_id: str) -> Optional[Dict[str, Any]]:
        directory = self.path(study_id)
        if directory is None:
            return None
        try:
            return json.loads((directory / RECORD_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def delete(self, study_id: str) -> bool:
        directory = self.path(study_id)
        if directory is None:
            return False
        shutil.rmtree(directory, ignore_errors=True)
        return True

    def delete_all(self) -> int:
        removed = 0
        for directory in self._directories():
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
        return removed

    def recent(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Newest first, with just enough of each study to draw a list row."""
        rows = []
        for directory in self._directories():
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
                "views": [image.get("view_key") for image in record.get("images", [])],
            })
        rows.sort(key=lambda row: row["created"], reverse=True)
        return rows[:limit]

    # ── retention ────────────────────────────────────────────────────────
    def purge_expired(self, now: Optional[float] = None) -> int:
        """Delete studies past the retention window. Zero days disables this."""
        if self.retention_days <= 0:
            return 0
        cutoff = (time.time() if now is None else now) - self.retention_days * 86400
        removed = 0
        for directory in self._directories():
            record = self.load(directory.name)
            created = (record or {}).get("created")
            if created is None:
                created = directory.stat().st_mtime
            if created < cutoff:
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
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
        if not self.study_dir.is_dir():
            return []
        return [d for d in self.study_dir.iterdir() if d.is_dir()]
