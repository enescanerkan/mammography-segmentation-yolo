"""Multipart upload parsing that keeps every byte in memory.

Starlette's form parser spools any file larger than 1 MB to a temporary file on disk, which would break the
promise that an uploaded DICOM is never written anywhere. This reader feeds the request stream straight into
python-multipart and keeps each file part in a bytearray, enforcing the per-file and file-count limits while the
upload is still arriving.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Dict, List, Optional

try:  # python-multipart >= 0.0.13 ships the module as ``python_multipart``
    from python_multipart.multipart import MultipartParser, parse_options_header
except ImportError:  # pragma: no cover - older releases
    from multipart.multipart import MultipartParser, parse_options_header


class UploadError(ValueError):
    """The request is not an acceptable multipart upload; the message is user-facing."""


@dataclass
class UploadedFile:
    """One file part: its client-side name (folder uploads keep the relative path) and its bytes."""

    filename: str
    data: bytes
    too_large: bool = False


def _decode(value: bytes, charset: str) -> str:
    try:
        return value.decode(charset)
    except (LookupError, UnicodeDecodeError):
        return value.decode("latin-1")


class InMemoryMultipartReader:
    """Collects the file parts named ``field`` of one multipart body.

    Args:
        content_type: the request's Content-Type header (must carry a boundary).
        field: form field name of the files.
        max_files: more file parts than this aborts the upload.
        max_file_bytes: a part larger than this is kept as an empty, ``too_large`` entry.
    """

    def __init__(self, content_type: str, field: str = "files", max_files: int = 40,
                 max_file_bytes: int = 220 * 1024 * 1024) -> None:
        kind, params = parse_options_header(content_type or "")
        if kind != b"multipart/form-data" or b"boundary" not in params:
            raise UploadError("Expected a multipart/form-data upload with a 'files' field.")
        charset = params.get(b"charset", b"utf-8")
        self._charset = charset.decode("latin-1") if isinstance(charset, bytes) else str(charset)
        self.field = field
        self.max_files = int(max_files)
        self.max_file_bytes = int(max_file_bytes)
        self.files: List[UploadedFile] = []
        self._headers: Dict[bytes, bytes] = {}
        self._header_name = b""
        self._header_value = b""
        self._buffer: Optional[bytearray] = None
        self._name = ""
        self._too_large = False
        self._parser = MultipartParser(params[b"boundary"], {
            "on_part_begin": self._on_part_begin,
            "on_part_data": self._on_part_data,
            "on_part_end": self._on_part_end,
            "on_header_field": self._on_header_field,
            "on_header_value": self._on_header_value,
            "on_header_end": self._on_header_end,
            "on_headers_finished": self._on_headers_finished,
        })

    # ── python-multipart callbacks ───────────────────────────────────────
    def _on_part_begin(self) -> None:
        self._headers = {}
        self._buffer = None
        self._too_large = False

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def _on_header_end(self) -> None:
        self._headers[self._header_name.lower()] = self._header_value
        self._header_name = b""
        self._header_value = b""

    def _on_headers_finished(self) -> None:
        _, options = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = _decode(options.get(b"name", b""), self._charset)
        if name != self.field or b"filename" not in options:
            return  # other form fields are ignored
        if len(self.files) >= self.max_files:
            raise UploadError(f"At most {self.max_files} files can be uploaded at once.")
        self._name = _decode(options[b"filename"], self._charset)
        self._buffer = bytearray()

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._buffer is None or self._too_large:
            return
        self._buffer += data[start:end]
        if len(self._buffer) > self.max_file_bytes:
            self._too_large = True
            self._buffer = bytearray()

    def _on_part_end(self) -> None:
        if self._buffer is not None:
            self.files.append(UploadedFile(self._name or "file", bytes(self._buffer), self._too_large))
        self._buffer = None

    # ── feeding ─────────────────────────────────────────────────────────
    def feed(self, chunk: bytes) -> None:
        self._parser.write(chunk)

    def finish(self) -> List[UploadedFile]:
        self._parser.finalize()
        return self.files

    async def read(self, stream: AsyncIterator[bytes]) -> List[UploadedFile]:
        async for chunk in stream:
            if chunk:
                self.feed(chunk)
        return self.finish()
