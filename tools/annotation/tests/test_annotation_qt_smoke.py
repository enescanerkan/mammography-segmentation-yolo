"""Offscreen smoke tests of the Qt toolkit without any model weights."""

from __future__ import annotations

import importlib.util
import sys

import cv2
import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QMessageBox  # noqa: E402

from interactive.prompts import TISSUE_PRESETS  # noqa: E402
from toolkit_config import TOOLKIT_ROOT  # noqa: E402


@pytest.fixture
def no_dialogs(monkeypatch):
    """Record modal message boxes instead of blocking on them."""
    shown = []

    def _record(kind, answer):
        def _fn(*args, **kwargs):
            shown.append((kind, args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else ""))
            return answer
        return staticmethod(_fn)

    monkeypatch.setattr(QMessageBox, "critical", _record("critical", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "warning", _record("warning", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "information", _record("information", QMessageBox.Ok))
    monkeypatch.setattr(QMessageBox, "question", _record("question", QMessageBox.Discard))
    return shown


def _load_app_module():
    spec = importlib.util.spec_from_file_location("annotation_app", TOOLKIT_ROOT / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_print_config(capsys):
    app = _load_app_module()
    assert app.main(["--print-config"]) == 0
    out = capsys.readouterr().out
    assert "medsam2.checkpoint" in out and "positioning.mlo_weights" in out


def test_startup_dialog(qapp):
    from qt_app.main_window import StartupDialog

    dlg = StartupDialog()
    assert dlg.windowTitle()
    dlg.show_loading_message("Loading…")
    dlg.show_cards()
    dlg.close()


def test_sam_mode_reports_missing_files_without_loading_torch_models():
    app = _load_app_module()
    with pytest.raises(FileNotFoundError) as exc:
        app._build_pipeline()
    assert "medsam2.checkpoint" in str(exc.value)
    assert "sam2" not in sys.modules  # failed before importing the MedSAM2 code


def test_manual_mode_window(qapp, no_dialogs):
    from qt_app.main_window import MainWindow

    builtin = dict(TISSUE_PRESETS)
    win = MainWindow(pipeline=None, manual_mode=True)
    try:
        assert not TISSUE_PRESETS  # Manual mode starts with an empty class registry
        assert not win._c_detector.isHidden()
        win._on_run_detector()  # no image yet -> status message only
        assert not no_dialogs
    finally:
        win._exit_prompt_skip = True
        win.close()
    assert builtin == TISSUE_PRESETS  # built-ins restored on close


def test_detector_weights_missing_dialog(qapp, no_dialogs, tmp_path):
    from qt_app.main_window import MainWindow, register_custom_tissue

    img = tmp_path / "synthetic.png"
    cv2.imwrite(str(img), np.full((64, 48), 128, np.uint8))
    win = MainWindow(pipeline=None, manual_mode=True)
    try:
        win._open_image_path(img)
        key = register_custom_tissue("lesion")
        win._custom_class_keys.append(key)
        win._append_tissue_row(key)
        win._select_tissue(key)
        win._on_run_detector()
        assert no_dialogs and no_dialogs[-1][1] == "Detector weights missing"
        assert "manual_detector.pt" in no_dialogs[-1][2]
    finally:
        win._exit_prompt_skip = True
        win.close()


def test_sam_window_measure_and_save(qapp, no_dialogs, tmp_path):
    """SAM workspace without a model: label tissues as polygons, measure the PNL, save YOLO."""
    from qt_app.main_window import MainWindow

    h, w = 400, 300
    img = tmp_path / "synthetic_mlo.png"
    cv2.imwrite(str(img), np.full((h, w), 90, np.uint8))

    win = MainWindow(pipeline=None, manual_mode=False)
    try:
        win._open_image_path(img)
        assert win._current_image_rgb.shape == (h, w, 3)
        polys = {
            "pectoral": [(0, 0), (110, 0), (95, 120), (60, 220), (0, 260)],
            "breast": [(110, 0), (260, 0), (280, 200), (240, 380), (0, 390), (0, 260), (60, 220), (95, 120)],
            "nipple": [(244, 204), (256, 204), (256, 216), (244, 216)],
        }
        for key, poly in polys.items():
            win._session.append_polygon(img, key, poly)
            win._session.set_mask(img, key, win._rebuild_tissue_mask(key))

        win._calculate_geometry()
        assert not win._canvas._hud.isHidden()
        assert "PNL" in win._status.currentMessage()

        report = win._session.save_all()
        assert len(report.written) == 1 and not report.errors
        lines = (tmp_path / "labels" / "synthetic_mlo.txt").read_text().splitlines()
        assert sorted(int(line.split()[0]) for line in lines) == [0, 1, 2]  # pectoral, breast, nipple
    finally:
        win._exit_prompt_skip = True
        win.close()


def test_positioning_launcher_without_weights(qapp, no_dialogs):
    from qt_app.positioning_launcher import launch_positioning_window

    window = launch_positioning_window(on_back_to_menu=lambda: None)
    assert window is None
    kind, title, text = no_dialogs[-1]
    assert kind == "critical" and title == "Missing model weights"
    assert "mlo-yolo26-pose-advanced.pt" in text and "cc-yolo26-pose-advanced.pt" in text


def test_positioning_window_builds_without_weights(qapp, no_dialogs):
    pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from qt_app.positioning.gui.main_window import MainApplicationWindow

    window = MainApplicationWindow()
    try:
        assert window.sys_label.text() == "Error"  # models could not load
        assert no_dialogs and "pose model not found" in no_dialogs[-1][2]
    finally:
        window.close()
