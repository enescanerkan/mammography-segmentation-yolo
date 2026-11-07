"""Desktop launcher for the positioning review app (the Windows build in ``packaging/windows``).

The app runs on a loopback port inside the process and is shown in a native window (WebView2 through pywebview).
Without a web view it opens the default browser and keeps a small window that stops the app. Models run on the
CPU unless the user config says otherwise.

User files live in ``%LOCALAPPDATA%\\MammographyPositioning`` (``~/.local/share/MammographyPositioning`` elsewhere):

* ``app.yaml`` (optional): any key of the app config (``configs/app.example.yaml``) overrides the desktop defaults;
* ``data/``: study renders and label JSON (never the DICOM files), deleted after ``storage.retention_days``;
* ``logs/app.log``: the server log.

``--self-test`` starts the server without a window, waits for the models, decodes a synthetic DICOM (plain and
JPEG 2000), runs one full analysis and exits 0 on success: the build workflow runs it on the frozen executable.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import logging.handlers
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

APP_NAME = "MammographyPositioning"
PREFERRED_PORT = 8765  # fixed when free, so the web view keeps its stored theme and language between runs
log = logging.getLogger("breast_seg.desktop")


# ── locations ────────────────────────────────────────────────────────────────
def bundle_dir() -> Path:
    """Folder that holds ``configs/`` and ``weights/``: the PyInstaller bundle, else the repository root."""
    frozen = getattr(sys, "_MEIPASS", None)
    return Path(frozen) if frozen else Path(__file__).resolve().parents[3]


def user_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    path = Path(base) / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def desktop_config(bundle: Path, home: Path):
    """The app config of the desktop build: bundled registry and weights, CPU, data in the user folder."""
    from .positioning.config import app_config_from_dict

    data: Dict[str, Any] = {
        "models": {"registry": str(bundle / "configs" / "models.yaml"), "weights_root": str(bundle / "weights"),
                   "device": "cpu"},
        "storage": {"data_dir": str(home / "data")},
        "server": {"host": "127.0.0.1", "port": PREFERRED_PORT},
    }
    user_file = home / "app.yaml"
    if user_file.is_file():
        data = _merge(data, yaml.safe_load(user_file.read_text(encoding="utf-8")) or {})
    return app_config_from_dict(data, base_dir=home)


def free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return s.getsockname()[1]
    raise OSError("no free local port")


def setup_logging(home: Path) -> Path:
    """Log to ``logs/app.log``; a windowed build has no console, so stdout / stderr go there as well."""
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / "app.log"
    if sys.stdout is None or sys.stderr is None:  # PyInstaller --windowed
        stream = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - lives as long as the process
        sys.stdout = sys.stdout or stream
        sys.stderr = sys.stderr or stream
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[handler, logging.StreamHandler(sys.stderr)], force=True)
    return path


# ── server ───────────────────────────────────────────────────────────────────
class LocalServer:
    """uvicorn in a background thread; the main thread keeps the window."""

    def __init__(self, app, port: int) -> None:
        import uvicorn

        self.app, self.port = app, port
        self.url = f"http://127.0.0.1:{port}"
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None,
                                                    log_level="info", access_log=False))
        self.thread = threading.Thread(target=self.server.run, name="uvicorn", daemon=True)

    def start(self, timeout: float = 60.0) -> LocalServer:
        self.thread.start()
        deadline = time.time() + timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.time() > deadline:
                raise RuntimeError(f"the local server did not start on port {self.port}")
            time.sleep(0.05)
        return self

    def get(self, path: str, timeout: float = 5.0) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as response:
            return response.read()

    def health(self) -> Dict[str, Any]:
        return json.loads(self.get("/api/health"))

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


def _close_splash() -> None:
    try:
        import pyi_splash  # only inside a PyInstaller build with a splash screen

        pyi_splash.close()
    except Exception:  # noqa: BLE001 - no splash outside the frozen build
        pass


# ── windows ──────────────────────────────────────────────────────────────────
def show_window(title: str, url: str, home: Path) -> bool:
    """Native window with the app; False when no web view is available."""
    try:
        import webview
    except Exception as exc:  # noqa: BLE001
        log.warning("no web view (%s); opening the browser instead", exc)
        return False
    try:
        webview.settings["ALLOW_DOWNLOADS"] = True  # reports and label exports
        webview.create_window(title, url, width=1440, height=900, min_size=(1024, 680), maximized=True,
                              text_select=True)
        gui = "edgechromium" if sys.platform == "win32" else None  # never the legacy MSHTML engine
        _close_splash()
        webview.start(gui=gui, private_mode=False, storage_path=str(home / "webview"))
        return True
    except Exception as exc:  # noqa: BLE001 - e.g. no WebView2 runtime
        log.warning("the web view failed (%s); opening the browser instead", exc)
        return False


def browser_fallback(title: str, url: str) -> None:
    """Default browser plus a small window whose Quit button stops the app."""
    import webbrowser

    _close_splash()
    webbrowser.open(url)
    try:
        import tkinter as tk
    except Exception:  # noqa: BLE001 - no Tk: run until the process is ended
        threading.Event().wait()
        return
    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)
    tk.Label(root, text=f"{title} is running at\n{url}", padx=24, pady=16, justify="center").pack()
    row = tk.Frame(root, pady=12)
    row.pack()
    tk.Button(row, text="Open in browser", width=16, command=lambda: webbrowser.open(url)).pack(side="left", padx=6)
    tk.Button(row, text="Quit", width=10, command=root.destroy).pack(side="left", padx=6)
    root.mainloop()


# ── self-test ────────────────────────────────────────────────────────────────
def synthetic_dicom(view: str = "MLO", side: str = "L", jpeg2000: bool = False) -> bytes:
    """A phantom mammogram as DICOM bytes (generated UIDs, no patient data)."""
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, JPEG2000Lossless, generate_uid

    rows, cols = 2400, 1900
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    breast = ((xx / (0.78 * cols)) ** 2 + ((yy - rows / 2) / (0.45 * rows)) ** 2) <= 1.0
    image = np.where(breast, 1400 + 900 * np.exp(-xx / cols), 60).astype(np.float32)
    if view == "MLO":
        image[(xx + yy * 0.55) < 0.42 * cols] = 2600  # pectoral wedge in the upper corner
    image += np.random.default_rng(0).normal(0, 25, image.shape).astype(np.float32)
    image = np.clip(image, 0, 4095).astype(np.uint16)
    if side == "R":
        image = image[:, ::-1].copy()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # digital mammography, for presentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.ViewPosition, ds.ImageLaterality = "MG", view, side
    ds.PresentationIntentType = "FOR PRESENTATION"
    ds.Rows, ds.Columns = rows, cols
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelSpacing = ds.ImagerPixelSpacing = [0.1, 0.1]
    ds.WindowCenter, ds.WindowWidth = 2048, 4096
    if jpeg2000:
        import openjpeg
        from pydicom.encaps import encapsulate

        encode = getattr(openjpeg, "encode_array", None) or openjpeg.encode  # renamed in pylibjpeg-openjpeg 2.4
        ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
        ds.PixelData = encapsulate([encode(image, bits_stored=12, photometric_interpretation=2, use_mct=False)])
        ds["PixelData"].is_undefined_length = True
        ds.is_little_endian, ds.is_implicit_VR = True, False
    else:
        ds.PixelData = image.tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, write_like_original=False)
    return buffer.getvalue()


def dicom_decoders() -> List[str]:
    """Transfer syntaxes the pylibjpeg plug-ins registered (found through package metadata entry points)."""
    from pylibjpeg.utils import get_pixel_data_decoders

    return sorted(get_pixel_data_decoders())


def self_test(server: LocalServer, timeout: float) -> Dict[str, Any]:
    from ..imaging import read_dicom

    report: Dict[str, Any] = {}
    deadline = time.time() + timeout
    while True:
        health = server.health()
        if not health.get("loading", True):
            break
        if time.time() > deadline:
            raise RuntimeError("the models did not finish loading in time")
        time.sleep(1)
    report["ready"], report["errors"], report["loaded"] = health.get("ready"), health.get("errors"), health.get("loaded")
    if not health.get("ready"):
        raise RuntimeError(f"models not ready: {health.get('errors')}")
    page = server.get("/")
    if b"<html" not in page.lower():
        raise RuntimeError("the client page did not load")

    decoders = dicom_decoders()
    report["decoders"] = decoders
    for uid in ("1.2.840.10008.1.2.4.70", "1.2.840.10008.1.2.4.90"):  # JPEG lossless SV1, JPEG 2000 lossless
        if uid not in decoders:
            raise RuntimeError(f"no DICOM decoder for {uid}")

    engine = server.app.state.service.engine
    started = time.time()
    plain, meta = read_dicom(synthetic_dicom("MLO", "L"), "L-MLO.dcm")
    compressed, _ = read_dicom(synthetic_dicom("MLO", "L", jpeg2000=True), "L-MLO-j2k.dcm")
    if plain.shape != compressed.shape or float(abs(plain - compressed).max()) > 0:
        raise RuntimeError("JPEG 2000 decoding differs from the plain pixels")
    result = engine.analyse(plain, meta)
    missing = {"landmarks", "masks", "polygons", "display", "quality"} - set(result)
    if missing:
        raise RuntimeError(f"analysis result lacks {sorted(missing)}")
    report["analysis_s"] = round(time.time() - started, 2)
    report["landmarks"] = sorted((result.get("landmarks") or {}).keys())
    return report


# ── entry point ──────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog=APP_NAME, description="Mammography positioning review (desktop)")
    parser.add_argument("--browser", action="store_true", help="open the default browser instead of a window")
    parser.add_argument("--self-test", action="store_true", help="start, analyse a synthetic study, exit")
    parser.add_argument("--timeout", type=float, default=600.0, help="self-test: seconds to wait for the models")
    args = parser.parse_args(argv)

    home = user_dir()
    log_path = setup_logging(home)
    os.environ.setdefault("YOLO_OFFLINE", "true")  # no update checks or downloads from Ultralytics
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # a frozen app cannot pip-install (it would start itself)
    (home / "ultralytics").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "ultralytics"))
    try:
        import torch  # noqa: F401 - load torch's DLLs before any GUI toolkit brings its own

        from .positioning.api import create_app

        config = desktop_config(bundle_dir(), home)
        port = free_port(config.server.port)
        server = LocalServer(create_app(config), port).start()
    except Exception:
        log.exception("the app could not start (log: %s)", log_path)
        _close_splash()
        return 1
    log.info("%s %s at %s (data: %s)", APP_NAME, _version(), server.url, config.data_dir)

    if args.self_test:
        try:
            report = self_test(server, args.timeout)
        except Exception:
            log.exception("self-test failed")
            return 1
        finally:
            server.stop()
        log.info("self-test passed: %s", json.dumps(report))
        print(json.dumps({"self_test": "passed", **report}))
        return 0

    try:
        if args.browser or not show_window(config.title, server.url, home):
            browser_fallback(config.title, server.url)
    finally:
        server.stop()
    return 0


def _version() -> str:
    from .. import __version__

    return __version__


if __name__ == "__main__":
    sys.exit(main())
