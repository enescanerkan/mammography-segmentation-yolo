"""Entry point for the PyQt5 Mammography Toolkit.

Run:
    python medsam2_qt_demo.py
Optional:
    set MEDSAM2_INTERACTIVE_CKPT=C:\\path\\to\\medsam_model_best.pth
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

from qt_app.logging_setup import setup_logging
from qt_app.main_window import MainWindow, StartupDialog
from qt_app.theme import QSS as THEME_QSS

log = logging.getLogger(__name__)


def _build_pipeline():
    """Lazy-build the SAM pipeline. Imports torch inside so manual mode
    doesn't pay the torch import cost."""
    import torch

    from config import MEDSAM2_REPO, SAM2_BASE_WEIGHTS, SAM2_MODEL_CFG
    from interactive import InferencePipeline, MedSAM2Service
    from interactive.model import ModelConfig, resolve_finetuned_ckpt

    finetune_ckpt = resolve_finetuned_ckpt(_HERE)
    log.info("SAM2 fine-tune checkpoint: %s", finetune_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("SAM2 device=%s repo=%s", device, MEDSAM2_REPO)
    config = ModelConfig(
        medsam2_repo=Path(MEDSAM2_REPO),
        sam2_model_cfg=SAM2_MODEL_CFG,
        sam2_base_weights=Path(SAM2_BASE_WEIGHTS),
        finetune_ckpt=Path(finetune_ckpt),
        device=__import__("torch").device(device),
    )
    service = MedSAM2Service(config)
    return InferencePipeline(service)


class _ToolkitController:
    """Owns the startup dialog and the active main window. Reopens the
    startup screen when the main window asks to go back."""

    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.dlg: Optional[StartupDialog] = None
        self.window: Optional[MainWindow] = None

    def run(self) -> int:
        self._open_startup()
        return self.app.exec_()

    # ── startup ─────────────────────────────────────────────────────

    def _open_startup(self) -> None:
        self.dlg = StartupDialog()
        self.dlg.choiceMade.connect(self._on_choice)
        self.dlg.rejected.connect(self.app.quit)
        self.dlg.showMaximized()
        log.info("Startup dialog opened")

    def _on_choice(self, choice: str, path) -> None:
        assert self.dlg is not None
        log.info("Startup choice: %s path=%s", choice, path)

        if choice == StartupDialog.CHOICE_POSITIONING:
            self.dlg.show_loading_message("Loading Breast Positioning models…")
            self.app.processEvents()
            self._launch_positioning_inprocess()
            return

        if choice == StartupDialog.CHOICE_MANUAL:
            self.dlg.show_loading_message("Opening Manual Label workspace…")
            self.app.processEvents()
            # Build the workspace FIRST so a real window is on-screen
            # before the dialog goes away — avoids any zero-window flicker.
            self._spawn_window(pipeline=None, manual=True, path=path)
            self.app.processEvents()
            self.dlg.finish_with(choice, path)
            return

        if choice == StartupDialog.CHOICE_SAM:
            self.dlg.show_loading_message("Loading SAM model weights…")
            self.app.processEvents()
            try:
                pipeline = _build_pipeline()
            except Exception:  # noqa: BLE001
                log.exception("Failed to build SAM pipeline")
                QMessageBox.critical(
                    self.dlg, "SAM load failed",
                    "Could not load MedSAM2 weights. See logs/qt_app.log.",
                )
                self.dlg.show_cards()
                return
            self._spawn_window(pipeline=pipeline, manual=False, path=path)
            self.app.processEvents()
            self.dlg.finish_with(choice, path)
            return

    # ── main window lifecycle ───────────────────────────────────────

    def _spawn_window(self, pipeline, manual: bool, path: Optional[Path]) -> None:
        if self.window is not None:
            self.window.close()
            self.window = None
        self.window = MainWindow(pipeline=pipeline, manual_mode=manual)
        self.window.backToMenuRequested.connect(self._on_back_to_menu)
        self.window.showMaximized()
        if path is not None:
            if path.is_dir():
                self.window._open_folder_path(path)
            else:
                self.window._open_image_path(path)

    def _on_back_to_menu(self) -> None:
        log.info("Returning to startup menu")
        # The main window emits this then closes itself.
        self.window = None
        self._open_startup()

    # ── positioning launcher (in-process) ───────────────────────────

    def _launch_positioning_inprocess(self) -> None:
        from qt_app.positioning_launcher import launch_positioning_window
        from qt_app.theme import QSS

        try:
            window = launch_positioning_window(
                on_back_to_menu=self._on_back_to_menu,
                theme_qss=QSS,
            )
        except Exception:
            log.exception("Positioning launch raised")
            QMessageBox.critical(
                self.dlg, "Positioning failed",
                "Breast Positioning failed to load. See logs/qt_app.log.",
            )
            if self.dlg is not None:
                self.dlg.show_cards()
            return

        if window is None:
            if self.dlg is not None:
                self.dlg.show_cards()
            return

        # Show the new window FIRST. Closing the dialog before any other
        # window is visible would empty the window list, and that would
        # otherwise let Qt fire its app-shutdown cascade.
        if self.window is not None:
            self.window.close()
        self.window = window
        window.showMaximized()
        self.app.processEvents()
        if self.dlg is not None:
            self.dlg.finish_with(StartupDialog.CHOICE_POSITIONING, None)


def main() -> int:
    setup_logging()
    log.info("─── Mammography Toolkit starting ───")
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("Mammography Toolkit")
    # Apply the toolkit theme at the application level so EVERY widget —
    # including stand-alone QInputDialog / QColorDialog / QMessageBox
    # popups — inherits the dark palette without needing per-call wiring.
    app.setStyleSheet(THEME_QSS)
    # We manage app lifetime explicitly (back-to-menu transitions
    # temporarily have zero visible windows; auto-quit on last-window-
    # closed would terminate the toolkit during those transitions).
    app.setQuitOnLastWindowClosed(False)
    controller = _ToolkitController(app)
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
