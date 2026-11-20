"""The browser-side DICOM header walker (static/js/dataset.js), run as JavaScript on headers written by pydicom.

The walker groups a dataset folder into studies by the top-level StudyInstanceUID, so it is tested where it can go
wrong: a Study Instance UID nested in a sequence that comes first, sequences of undefined length, implicit VR, big
endian, a UN element holding an implicit-VR sequence, a long private element that pushes the UID past the first
read, and files that are not DICOM at all.

The JavaScript runs under Node when it is installed, otherwise in headless Chromium (``$BREAST_SEG_CHROMIUM`` or a
Playwright download); without either the test is skipped.
"""
from __future__ import annotations

import base64
import functools
import glob
import io
import json
import os
import shutil
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.filebase import DicomBytesIO  # noqa: E402
from pydicom.filewriter import write_dataset  # noqa: E402
from pydicom.sequence import Sequence  # noqa: E402
from pydicom.uid import ExplicitVRBigEndian, ExplicitVRLittleEndian, ImplicitVRLittleEndian  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "src" / "breast_seg" / "apps" / "positioning" / "static"
DATASET_JS = STATIC / "js" / "dataset.js"
STUDY = "1.2.826.0.1.3680043.8.498.1"
NESTED = "9.9.9.9"


def _dataset(*, nested=True, undefined=True, study=True, private_ob=0):
    ds = Dataset()
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"
    ds.SOPInstanceUID = STUDY + ".5"
    ds.Modality = "MG"
    if nested:  # (0008,1250) Related Series Sequence, before the study's own UID, carrying another study's UID
        item = Dataset()
        item.StudyInstanceUID = NESTED
        item.SeriesInstanceUID = NESTED + ".1"
        if undefined:
            item.is_undefined_length_sequence_item = True
        ds.RelatedSeriesSequence = Sequence([item])
        ds["RelatedSeriesSequence"].is_undefined_length = undefined
    if private_ob:
        ds.add_new(0x00090010, "LO", "TEST")
        ds.add_new(0x00091001, "OB", b"\0" * private_ob)
    if study:
        ds.StudyInstanceUID = STUDY
    ds.SeriesInstanceUID = STUDY + ".2"
    return ds


def _file(ds, syntax=ExplicitVRLittleEndian):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = ds.SOPClassUID
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = syntax
    ds.file_meta = meta
    ds.is_little_endian = syntax != ExplicitVRBigEndian
    ds.is_implicit_VR = syntax == ImplicitVRLittleEndian
    buffer = io.BytesIO()
    ds.save_as(buffer, write_like_original=False)
    return buffer.getvalue()


def _bare(ds):
    """A data set with no preamble and no file meta (implicit VR little endian), as some exporters write."""
    fp = DicomBytesIO()
    fp.is_little_endian, fp.is_implicit_VR = True, True
    write_dataset(fp, ds)
    return fp.getvalue()


def _un_sequence_file():
    """Explicit VR LE, where (0008,1250) arrives as UN of undefined length holding an implicit-VR sequence."""
    head = _file(_dataset(nested=False, study=False))
    head = head[:head.index(b"\x20\x00\x0e\x00")]  # cut before SeriesInstanceUID; rebuilt below in tag order
    uid = NESTED.encode() + (b"\0" if len(NESTED) % 2 else b"")
    item = struct.pack("<HHI", 0x0020, 0x000D, len(uid)) + uid                  # implicit element inside
    un = (struct.pack("<HH", 0x0008, 0x1250) + b"UN\0\0" + struct.pack("<I", 0xFFFFFFFF)
          + struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF) + item + struct.pack("<HHI", 0xFFFE, 0xE00D, 0)
          + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0))
    study = STUDY.encode() + (b"\0" if len(STUDY) % 2 else b"")
    own = struct.pack("<HH", 0x0020, 0x000D) + b"UI" + struct.pack("<H", len(study)) + study
    return head + un + own


def _cases():
    long_header = _file(_dataset(private_ob=40_000))
    return {
        "explicit_le_undefined_sequence": (_file(_dataset()), None, {"dicom": True, "uid": STUDY, "more": False}),
        "explicit_le_defined_sequence": (_file(_dataset(undefined=False)), None,
                                         {"dicom": True, "uid": STUDY, "more": False}),
        "implicit_le": (_file(_dataset(), ImplicitVRLittleEndian), None, {"dicom": True, "uid": STUDY, "more": False}),
        "explicit_be": (_file(_dataset(), ExplicitVRBigEndian), None, {"dicom": True, "uid": STUDY, "more": False}),
        "un_holding_a_sequence": (_un_sequence_file(), None, {"dicom": True, "uid": STUDY, "more": False}),
        "bare_dataset": (_bare(_dataset()), None, {"dicom": True, "uid": STUDY, "more": False}),
        "only_a_nested_uid": (_file(_dataset(study=False)), None, {"dicom": True, "uid": "", "more": False}),
        "long_header_first_16k": (long_header, 16 * 1024, {"dicom": True, "uid": "", "more": True}),
        "long_header_whole": (long_header, None, {"dicom": True, "uid": STUDY, "more": False}),
        "text_file": (b"hello world\n" * 200, None, {"dicom": False, "uid": "", "more": False}),
        "zeros": (b"\0" * 300, None, {"dicom": False, "uid": "", "more": False}),
    }


# ── JavaScript runners ─────────────────────────────────────────────────────
_JS_CASES = """
const out = {};
for (const c of cases) {
  const raw = atob(c.b64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  out[c.name] = parseStudyUid(c.cut ? bytes.subarray(0, c.cut) : bytes);
}
"""


def _run_node(node, cases):
    script = (f"import {{ parseStudyUid }} from {json.dumps(DATASET_JS.as_uri())};\n"
              f"const cases = {json.dumps(cases)};\n{_JS_CASES}\nconsole.log(JSON.stringify(out));")
    done = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _chromium():
    candidates = [os.environ.get("BREAST_SEG_CHROMIUM", "")]
    candidates += sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")))
    return next((path for path in candidates if path and os.access(path, os.X_OK)), None)


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - keep the test output clean
        pass


def _run_chromium(chrome, cases, tmp_path):
    websocket = pytest.importorskip("websocket")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_QuietHandler, directory=str(STATIC)))
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = _free_port()
    browser = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", "--remote-allow-origins=*",
                                "--no-sandbox", "--disable-gpu", f"--user-data-dir={tmp_path / 'chrome'}",
                                f"{origin}/js/dataset.js"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        page = None
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2))
                page = next(tab for tab in tabs if tab["type"] == "page" and "dataset.js" in tab["url"])
                break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(0.1)
        assert page, "headless Chromium did not start"
        ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
        expression = (f"(async () => {{ const {{ parseStudyUid }} = await import('{origin}/js/dataset.js');\n"
                      f"const cases = {json.dumps(cases)};\n{_JS_CASES}\nreturn out; }})()")
        message = {}
        for attempt in range(1, 30):  # the page may still be loading: its first context can be replaced under us
            ws.send(json.dumps({"id": attempt, "method": "Runtime.evaluate",
                                "params": {"expression": expression, "awaitPromise": True, "returnByValue": True}}))
            while True:
                message = json.loads(ws.recv())
                if message.get("id") == attempt:
                    break
            if "result" in message:
                break
            time.sleep(0.2)
        ws.close()
        assert "result" in message, message.get("error")
        result = message["result"]
        assert "exceptionDetails" not in result, result["exceptionDetails"]
        return result["result"]["value"]
    finally:
        browser.terminate()
        browser.wait(timeout=10)
        server.shutdown()


def test_the_walker_finds_the_top_level_study_uid(tmp_path):
    cases = _cases()
    payload = [{"name": name, "b64": base64.b64encode(data).decode(), "cut": cut}
               for name, (data, cut, _) in cases.items()]
    node = shutil.which("node")
    chrome = None if node else _chromium()
    if node:
        results = _run_node(node, payload)
    elif chrome:
        results = _run_chromium(chrome, payload, tmp_path)
    else:
        pytest.skip("needs Node or a headless Chromium to run the JavaScript")
    assert {name: results[name] for name in cases} == {name: expected for name, (_, _, expected) in cases.items()}
