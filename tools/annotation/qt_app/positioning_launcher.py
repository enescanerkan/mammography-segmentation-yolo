"""In-process launcher for the Breast Positioning tool (``qt_app.positioning``).

Checks that the two pose models configured under ``positioning.*`` exist, then
builds and returns the positioning ``QMainWindow`` so the toolkit controller can
manage it like any other window.

Lifecycle
---------
- Adds a ``← Back to Toolkit Menu`` action to the positioning window's
  menubar; clicking it invokes ``on_back_to_menu`` BEFORE closing the
  window, so :class:`QApplication`'s "quit on last window closed" doesn't
  terminate the toolkit.
- Applies the toolkit's QSS over the positioning's own dark theme.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QObject, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QMenu, QMessageBox

log = logging.getLogger(__name__)


def _check_weights() -> tuple[bool, list[str]]:
    """Return (ok, missing_paths) for the configured MLO / CC pose models."""
    from .positioning.utils.paths import missing_weights

    missing = [str(p) for p in missing_weights()]
    return not missing, missing


class _PositioningCloseFilter(QObject):
    """Event filter that distinguishes a 'Back to Menu' close from a true
    X-close. The launcher attaches one of these per window.

    Back-to-Menu sets ``going_back = True`` before calling ``window.close()``;
    on the resulting ``QEvent.Close`` we leave the application alive (the
    startup dialog has already been shown). For any other close (the user
    clicking X), we schedule a single ``QApplication.quit()`` after the
    close completes — required because the toolkit disables
    ``quitOnLastWindowClosed``.
    """

    def __init__(self, parent: QMainWindow) -> None:
        super().__init__(parent)
        self.going_back = False

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QEvent.Close and not self.going_back:
            QTimer.singleShot(0, lambda: QApplication.instance().quit())
        return False  # don't block — let the window's own closeEvent run


def _add_back_to_menu_action(
    window: QMainWindow,
    callback: Callable[[], None],
    close_filter: _PositioningCloseFilter,
) -> None:
    """Append a 'Back to Toolkit Menu' action to the window's File menu
    (creating the menu if it doesn't exist yet)."""
    mb = window.menuBar()
    file_menu: Optional[QMenu] = None
    for act in mb.actions():
        m = act.menu()
        if m is None:
            continue
        title = m.title().replace("&", "").strip().lower()
        if title == "file":
            file_menu = m
            break
    if file_menu is None:
        file_menu = mb.addMenu("&File")
    file_menu.addSeparator()

    back = QAction("← Back to Toolkit Menu", window)
    back.setShortcut(QKeySequence("Ctrl+M"))

    def _go_back() -> None:
        # Mark the upcoming Close as menu-bound so the filter doesn't quit.
        close_filter.going_back = True
        try:
            callback()
        finally:
            window.close()

    back.triggered.connect(_go_back)
    file_menu.addAction(back)


def launch_positioning_window(
    on_back_to_menu: Callable[[], None],
    theme_qss: Optional[str] = None,
) -> Optional[QMainWindow]:
    """Build and return the positioning MainApplicationWindow.

    Returns ``None`` if weights are missing or the import fails; in those
    cases an error dialog is shown to the user.
    """
    ok, missing = _check_weights()
    if not ok:
        log.error("Positioning weights missing: %s", missing)
        QMessageBox.critical(
            None,
            "Missing model weights",
            "Breast Positioning needs two YOLO-pose models. Not found:\n• "
            + "\n• ".join(missing)
            + "\n\nSet positioning.mlo_weights / positioning.cc_weights in annotation.yaml "
            "(or $POSITIONING_MLO_WEIGHTS / $POSITIONING_CC_WEIGHTS). See tools/annotation/README.md.",
        )
        return None

    try:
        from .positioning.gui.main_window import MainApplicationWindow
    except Exception:
        log.exception("Failed to import positioning MainApplicationWindow")
        QMessageBox.critical(
            None,
            "Positioning failed to load",
            "Could not import the Breast Positioning module. See logs/qt_app.log.",
        )
        return None

    try:
        window = MainApplicationWindow()
    except Exception:
        log.exception("MainApplicationWindow constructor raised")
        QMessageBox.critical(
            None,
            "Positioning failed to start",
            "Could not initialize Breast Positioning. See logs/qt_app.log.",
        )
        return None

    # Install the close-filter (must outlive the window — it's parented to it).
    close_filter = _PositioningCloseFilter(window)
    window.installEventFilter(close_filter)

    # Stick a "Back to Toolkit Menu" action into its menubar.
    try:
        _add_back_to_menu_action(window, on_back_to_menu, close_filter)
    except Exception:
        log.exception("Failed to add Back-to-Menu action")

    # Apply toolkit QSS over the positioning's built-in theme so the look
    # matches the other windows. The positioning still sets its own
    # stylesheet inside its constructor — applying ours AFTER overrides it.
    if theme_qss:
        try:
            window.setStyleSheet(theme_qss)
        except Exception:
            log.exception("Failed to apply toolkit QSS to positioning window")

    log.info("Positioning window built — weights OK, ready to show.")
    return window
