"""Entry point for the PyQt5 Mammography Annotation Toolkit.

Run (from anywhere):
    python tools/annotation/app.py                     # uses annotation.yaml if present
    python tools/annotation/app.py --config my.yaml
    python tools/annotation/app.py --print-config      # show resolved paths and exit

Model files are configured in ``annotation.yaml`` / environment variables, see
``toolkit_config.py`` and ``tools/annotation/README.md``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

import toolkit_config  # noqa: E402  (the imports below need the sys.path entry above)
from qt_app.logging_setup import setup_logging  # noqa: E402
from qt_app.main_window import MainWindow, StartupDialog  # noqa: E402
from qt_app.theme import QSS as THEME_QSS  # noqa: E402

log = logging.getLogger(__name__)


def _build_pipeline():
    """Build the MedSAM2 pipeline (imports torch lazily, so Manual mode does not pay for it)."""
    from interactive.model import build_pipeline

    cfg = toolkit_config.get_config()
    log.info("MedSAM2 checkpoint: %s", cfg.medsam2_checkpoint)
    log.info("SAM2 base: %s  repo: %s  device: %s", cfg.sam2_base_weights, cfg.medsam2_repo, cfg.device)
    return build_pipeline(cfg)


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
            except FileNotFoundError as exc:
                log.error("SAM mode unavailable: %s", exc)
                QMessageBox.critical(
                    self.dlg, "MedSAM2 files missing",
                    f"{exc}\n\nConfigure the medsam2.* keys in annotation.yaml (or the matching "
                    "environment variables). See tools/annotation/README.md.",
                )
                self.dlg.show_cards()
                return
            except Exception as exc:  # noqa: BLE001 - surface any build error to the user
                log.exception("Failed to build SAM pipeline")
                QMessageBox.critical(
                    self.dlg, "SAM load failed",
                    f"Could not load MedSAM2:\n{exc}\n\nDetails are in the toolkit log (logs_dir).",
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
                "Breast Positioning failed to load. Details are in the toolkit log (logs_dir).",
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


def _parse_args(argv: Optional[list] = None) -> tuple[argparse.Namespace, list]:
    """Toolkit options; anything unrecognised is passed through to Qt (e.g. ``-platform``)."""
    p = argparse.ArgumentParser(description="Mammography annotation toolkit (SAM / Manual / Positioning).")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: $ANNOTATION_CONFIG, else tools/annotation/annotation.yaml if present)")
    p.add_argument("--print-config", action="store_true", help="Print the resolved configuration and exit.")
    return p.parse_known_args(argv)


def main(argv: Optional[list] = None) -> int:
    args, qt_args = _parse_args(argv)
    toolkit_config.set_config_path(args.config)
    cfg = toolkit_config.get_config()
    if args.print_config:
        print(cfg.describe())
        return 0

    setup_logging(cfg.logs_dir)
    log.info("─── Mammography Toolkit starting ───")
    for line in cfg.describe().splitlines():
        log.info("config: %s", line)
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv[:1] + qt_args)
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
