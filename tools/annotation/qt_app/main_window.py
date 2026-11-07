"""Main application window for the PyQt5 MedSAM2 annotation tool."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


_IMAGE_DIALOG_FILTER = (
    "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.dicom);;"
    "DICOM (*.dcm *.dicom);;"
    "PNG/JPEG (*.png *.jpg *.jpeg);;"
    "All files (*.*)"
)


# Palette used to auto-pick colors for user-added custom classes. Cycles
# through hues distinct from the three built-in tissue colors.
_CUSTOM_TISSUE_PALETTE_RGB: list[tuple[int, int, int]] = [
    (245, 158, 11),   # amber
    (217, 70, 239),   # fuchsia
    (236, 72, 153),   # pink
    (139, 92, 246),   # violet
    (14, 165, 233),   # sky
    (132, 204, 22),   # lime
    (251, 113, 133),  # rose
    (8, 145, 178),    # cyan
]


def _slugify_class_key(label: str) -> str:
    import re
    key = re.sub(r"\W+", "_", label.lower()).strip("_")
    return key or "custom"


class _CustomClassError(Exception):
    """Raised by :func:`register_custom_tissue` on validation failure."""


def _existing_rgb_colors() -> set[tuple[int, int, int]]:
    """All currently-registered tissue colors as RGB tuples."""
    out: set[tuple[int, int, int]] = set()
    for k in TISSUE_PRESETS:
        if k in TISSUE_RGB:
            out.add(tuple(TISSUE_RGB[k]))  # type: ignore[arg-type]
        else:
            b, g, r = TISSUE_PRESETS[k].overlay_bgr
            out.add((int(r), int(g), int(b)))
    return out


def _pick_auto_color() -> tuple[int, int, int]:
    used = _existing_rgb_colors()
    for c in _CUSTOM_TISSUE_PALETTE_RGB:
        if c not in used:
            return c
    # Palette exhausted — fall back to a random hue that isn't already used.
    import random
    for _ in range(64):
        c = (random.randint(60, 240), random.randint(60, 240), random.randint(60, 240))
        if c not in used:
            return c
    return _CUSTOM_TISSUE_PALETTE_RGB[0]


def register_custom_tissue(
    label: str,
    color_rgb: Optional[tuple[int, int, int]] = None,
) -> str:
    """Add a user-defined class to ``TISSUE_PRESETS`` (in-memory, session-only).

    Returns the new ``tissue_key``. The class id is auto-assigned to one
    past the current max, so YOLO ids continue contiguously (3, 4, ...).

    Raises ``_CustomClassError`` if the supplied color is already used by
    another class (preventing visual ambiguity).
    """
    from interactive.prompts import TISSUE_PRESETS, TissuePreset  # local: avoid cycles

    base_key = _slugify_class_key(label)
    key = base_key
    i = 1
    while key in TISSUE_PRESETS:
        i += 1
        key = f"{base_key}_{i}"

    used = _existing_rgb_colors()
    if color_rgb is not None:
        c = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))
        if c in used:
            raise _CustomClassError(
                "This color is already used by another class. Pick a different one."
            )
        color_rgb = c
    else:
        color_rgb = _pick_auto_color()

    # Default 0 handles the empty-registry case (Manual mode starts blank →
    # first user-added class gets class_id=1 → YOLO id 0).
    next_id = max((p.class_id for p in TISSUE_PRESETS.values()), default=0) + 1
    # TissuePreset stores BGR. Convert from RGB.
    bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    TISSUE_PRESETS[key] = TissuePreset(key, label, bgr, next_id)
    TISSUE_RGB[key] = color_rgb
    return key


def unregister_custom_tissue(key: str) -> None:
    """Remove a previously-registered class. Used when the user deletes a
    custom class or when the window closes (back-to-menu cleanup)."""
    TISSUE_PRESETS.pop(key, None)
    TISSUE_RGB.pop(key, None)
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from interactive.inference import InferencePipeline, InferenceSettings
from interactive.prompts import TISSUE_PRESETS, PromptState, TissuePreset

from .canvas import (
    MODE_BOX,
    MODE_EDIT_POLYGON,
    MODE_IDLE,
    MODE_POINT,
    MammoCanvas,
    mask_to_polygon,
    polygon_to_mask,
)
from .detector import BoxDetector, Detection, run_detector_async
from .image_io import load_image_rgb
from .pnl_analysis import analyze_annotation
from .session import AnnotationSession, list_images, tissue_to_yolo_id
from .theme import (
    ACCENT,
    ALERT,
    APP_BG,
    QSS,
    SUCCESS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TISSUE_RGB,
)
from .worker import InferenceRequest, InferenceWorker


# ─── Startup dialog ──────────────────────────────────────────────────────


class _StartupCard(QFrame):
    """A hoverable launcher tile used in :class:`StartupDialog`."""

    clicked = pyqtSignal()

    def __init__(self, icon: str, title: str, desc: str, variant: str = "default") -> None:
        super().__init__()
        self.setObjectName("startupCard")
        self.setProperty("variant", variant)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(QSize(230, 240))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 22, 18, 22)
        v.setSpacing(10)
        v.setAlignment(Qt.AlignTop)

        ic = QLabel(icon)
        ic.setObjectName("startupCardIcon")
        ic.setAlignment(Qt.AlignCenter)
        v.addWidget(ic)

        t = QLabel(title)
        t.setObjectName("startupCardTitle")
        t.setAlignment(Qt.AlignCenter)
        v.addWidget(t)

        d = QLabel(desc)
        d.setObjectName("startupCardDesc")
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        v.addWidget(d)
        v.addStretch(1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StartupDialog(QDialog):
    """Opening screen with 3 large launcher cards.

    Caller flow:
        dlg = StartupDialog()
        dlg.choiceMade.connect(handler)   # handler may run async work
        dlg.show()    (or showMaximized)
        # ... handler eventually calls dlg.finish_with(choice, path)

    ``choiceMade`` fires BEFORE the dialog accepts; this lets the caller
    show ``show_loading_message(...)`` while it builds heavy resources
    (SAM weights). Call ``finish_with`` to dismiss.

    The dialog never auto-closes on card click — that's the caller's job.
    """

    CHOICE_SAM = "sam"
    CHOICE_MANUAL = "manual"
    CHOICE_POSITIONING = "positioning"

    choiceMade = pyqtSignal(str, object)  # choice, Optional[Path]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("startupDialog")
        self.setWindowTitle("Mammography Toolkit — Choose Tool")
        self.setMinimumSize(QSize(960, 600))
        self.setStyleSheet(QSS)
        self.choice: Optional[str] = None
        self.path: Optional[Path] = None
        self._build()

    def showMaximized(self) -> None:  # type: ignore[override]
        super().showMaximized()

    # ── public API for the caller ───────────────────────────────────

    def show_loading_message(self, message: str) -> None:
        """Replace the cards with a centered loading message."""
        self._cards_widget.setVisible(False)
        self._loading_label.setText(message)
        self._loading_widget.setVisible(True)
        # Force a repaint so the user sees the change immediately.
        self.repaint()

    def show_cards(self) -> None:
        self._loading_widget.setVisible(False)
        self._cards_widget.setVisible(True)

    def finish_with(self, choice: str, path: Optional[Path]) -> None:
        self.choice = choice
        self.path = path
        self.accept()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 28, 48, 36)
        root.setSpacing(22)

        # Header row: spacer + title (centered) + help button (right).
        header_row = QHBoxLayout()
        header_row.setSpacing(0)
        spacer = QWidget()
        spacer.setFixedSize(40, 40)  # mirror the help button so title stays centered
        header_row.addWidget(spacer)
        header_row.addStretch(1)
        title = QLabel("Mammography Toolkit")
        title.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {TEXT_PRIMARY};")
        header_row.addWidget(title, alignment=Qt.AlignCenter)
        header_row.addStretch(1)
        self._btn_help = QPushButton("?")
        self._btn_help.setObjectName("primary")
        self._btn_help.setToolTip("Shortcuts & settings")
        self._btn_help.setFixedSize(40, 40)
        self._btn_help.setStyleSheet("border-radius: 20px; font-weight: 700; font-size: 16px;")
        self._btn_help.clicked.connect(self._show_help)
        header_row.addWidget(self._btn_help)
        root.addLayout(header_row)

        sub = QLabel("Pick a tool — you can come back to this menu later via File → Back to Main Menu.")
        sub.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        sub.setAlignment(Qt.AlignHCenter)
        root.addWidget(sub)
        root.addSpacing(20)

        # Cards container
        self._cards_widget = QWidget()
        cards_row = QHBoxLayout(self._cards_widget)
        cards_row.setContentsMargins(0, 0, 0, 0)
        cards_row.setSpacing(20)
        root.addWidget(self._cards_widget, stretch=1)

        # Loading container (shown after a choice is made; hidden initially)
        self._loading_widget = QWidget()
        lw = QVBoxLayout(self._loading_widget)
        lw.setAlignment(Qt.AlignCenter)
        self._loading_label = QLabel("Loading…")
        self._loading_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 18px; font-weight: 600;"
        )
        self._loading_label.setAlignment(Qt.AlignCenter)
        lw.addWidget(self._loading_label)
        sub2 = QLabel("Please wait — this may take a moment.")
        sub2.setStyleSheet(f"color: {TEXT_SECONDARY};")
        sub2.setAlignment(Qt.AlignCenter)
        lw.addWidget(sub2)
        self._loading_widget.setVisible(False)
        root.addWidget(self._loading_widget, stretch=1)

        self._card_sam = _StartupCard(
            icon="◎",
            title="SAM Label",
            desc=("Box + point prompts → MedSAM2 produces a mask.\n"
                  "Refine with polygon edit after.\n"
                  "Loads model weights at startup."),
            variant="sam",
        )
        self._card_sam.clicked.connect(self._pick_sam)
        cards_row.addWidget(self._card_sam)

        self._card_manual = _StartupCard(
            icon="✎",
            title="Manual Label",
            desc=("Polygon + Box labeling with bundled YOLO auto-detector.\n"
                  "Add your own classes — first class becomes YOLO id 0.\n"
                  "Per-class multi-instance, polygon-edit refinement."),
            variant="manual",
        )
        self._card_manual.clicked.connect(self._pick_manual)
        cards_row.addWidget(self._card_manual)

        self._card_pos = _StartupCard(
            icon="◴",
            title="Breast Positioning",
            desc=("YOLO landmark-based MLO/CC positioning analysis.\n"
                  "Launches the standalone positioning app\n"
                  "(uses its own model weights)."),
            variant="positioning",
        )
        self._card_pos.clicked.connect(self._pick_positioning)
        cards_row.addWidget(self._card_pos)

        root.addLayout(cards_row, stretch=1)

        hint = QLabel(
            "Tip: Manual & SAM modes both support folder navigation, multi-class "
            "labeling and 'Save All' to YOLO format."
        )
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        hint.setWordWrap(True)
        root.addWidget(hint)

    # ── card handlers ───────────────────────────────────────────────

    def _pick_sam(self) -> None:
        path = self._ask_image_or_folder()
        if path is None:
            return
        self.choiceMade.emit(self.CHOICE_SAM, path)

    def _pick_manual(self) -> None:
        path = self._ask_image_or_folder()
        if path is None:
            return
        self.choiceMade.emit(self.CHOICE_MANUAL, path)

    def _pick_positioning(self) -> None:
        self.choiceMade.emit(self.CHOICE_POSITIONING, None)

    def _show_help(self) -> None:
        from PyQt5.QtWidgets import QMessageBox as _MB
        body = (
            "<h3 style='color:#FFFFFF'>Keyboard shortcuts</h3>"
            "<table cellpadding='4' style='color:#FFFFFF; font-family: monospace;'>"
            "<tr><td><b>Ctrl + S</b></td><td>Save all labels to YOLO</td></tr>"
            "<tr><td><b>Ctrl + Z</b></td><td>Undo last label (drops most recent polygon)</td></tr>"
            "<tr><td><b>Ctrl + M</b></td><td>Back to main menu</td></tr>"
            "<tr><td><b>Ctrl + 0</b></td><td>Fit image to view</td></tr>"
            "<tr><td><b>Ctrl + +/−</b></td><td>Zoom in / out</td></tr>"
            "<tr><td><b>Ctrl + Enter</b></td><td>Commit current polygon edit</td></tr>"
            "<tr><td><b>← / →</b></td><td>Previous / next image in folder</td></tr>"
            "<tr><td><b>B / P</b></td><td>Box mode / Point mode (SAM)</td></tr>"
            "<tr><td><b>1 / 2 / 3</b></td><td>Quick-pick tissue class (SAM)</td></tr>"
            "<tr><td><b>Esc</b></td><td>Cancel in-progress draw</td></tr>"
            "<tr><td><b>Ctrl + Wheel</b></td><td>Zoom at cursor</td></tr>"
            "<tr><td><b>Right-drag</b></td><td>Window / Level (brightness / contrast)</td></tr>"
            "<tr><td><b>Middle-drag</b></td><td>Pan</td></tr>"
            "<tr><td><b>Double-click background</b></td><td>Fit view + reset W/L</td></tr>"
            "<tr><td><b>Alt + click vertex</b></td><td>Delete a polygon vertex</td></tr>"
            "</table>"
            "<p style='color:#A0AEC0'>For full documentation, see the README files under <code>medsam2_experiments/docs/</code>.</p>"
        )
        box = _MB(self)
        box.setIcon(_MB.NoIcon)
        box.setWindowTitle("Help & Shortcuts")
        box.setTextFormat(Qt.RichText)
        box.setText(body)
        box.exec_()

    def _ask_image_or_folder(self) -> Optional[Path]:
        from PyQt5.QtWidgets import QMessageBox as MB
        box = MB(self)
        box.setIcon(MB.Question)
        box.setWindowTitle("Open")
        box.setText("Open a single image or a folder of images?")
        b_img = box.addButton("Single Image…", MB.AcceptRole)
        b_folder = box.addButton("Folder…", MB.AcceptRole)
        box.addButton("Cancel", MB.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_img:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open image", "",
                _IMAGE_DIALOG_FILTER,
            )
            return Path(path) if path else None
        if clicked is b_folder:
            path = QFileDialog.getExistingDirectory(self, "Open folder of images")
            return Path(path) if path else None
        return None


# ─── Main window ─────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Annotation main window. Owns canvas, session, and inference worker."""

    backToMenuRequested = pyqtSignal()

    def __init__(
        self,
        pipeline: Optional[InferencePipeline] = None,
        manual_mode: bool = False,
    ) -> None:
        super().__init__()
        title_suffix = " — Annotation Studio" if manual_mode else " — SAM Workspace"
        self.setWindowTitle(f"Mammography Toolkit{title_suffix}")
        self.resize(1500, 950)

        self._pipeline = pipeline
        self._manual_mode = bool(manual_mode)
        self._session = AnnotationSession()

        # Manual mode starts with an EMPTY class registry — user adds
        # their own (first one gets YOLO id 0). The toolkit-wide built-in
        # classes (pectoral / breast / nipple) are snapshotted here and
        # restored on close so other windows are unaffected.
        self._builtin_snapshot: dict = {}
        self._builtin_rgb_snapshot: dict = {}
        if self._manual_mode:
            self._builtin_snapshot = dict(TISSUE_PRESETS)
            self._builtin_rgb_snapshot = dict(TISSUE_RGB)
            TISSUE_PRESETS.clear()
            TISSUE_RGB.clear()
        # No SAM in manual mode → no background worker, no inference.
        self._worker: Optional[InferenceWorker] = None
        if pipeline is not None and not self._manual_mode:
            self._worker = InferenceWorker(pipeline, self)
            self._worker.finished.connect(self._on_inference_finished)
            self._worker.failed.connect(self._on_inference_failed)
            self._worker.started_request.connect(self._on_inference_started)
            self._worker.start()

        # State
        self._folder: Optional[Path] = None
        self._image_paths: list[Path] = []
        self._current_idx: int = -1
        self._current_image_rgb: Optional[np.ndarray] = None
        # Manual mode starts with no classes → no active tissue. User must
        # add a class before any annotation operation works.
        self._current_tissue_key: Optional[str] = None if manual_mode else "breast"
        self._latest_request_id: int = 0
        self._pending_first_image: Optional[Path] = None
        self._custom_class_keys: list[str] = []  # custom classes added in THIS window
        self._detector: Optional[BoxDetector] = None  # lazy-loaded on first use
        self._detector_thread = None  # type: ignore[var-annotated]
        self._detector_worker = None  # type: ignore[var-annotated]

        # Debounce timer for inference triggers — coalesces several mutations
        # that arrive in the same Qt event loop tick.
        self._predict_debounce = QTimer(self)
        self._predict_debounce.setSingleShot(True)
        self._predict_debounce.setInterval(40)
        self._predict_debounce.timeout.connect(self._submit_inference)

        self._build_ui()
        self._build_menu_and_toolbar()
        self._build_shortcuts()

        if self._manual_mode:
            device_label = "CUDA" if torch.cuda.is_available() else "CPU"
            self._lbl_device.setText(f"Annotation Studio • {device_label}")
        else:
            device_label = "CUDA" if torch.cuda.is_available() else "CPU"
            self._lbl_device.setText(f"SAM Workspace • {device_label}")

    # ─── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_v = QVBoxLayout(central)
        root_v.setContentsMargins(0, 0, 0, 0)
        root_v.setSpacing(0)

        # Colored header (gradient title bar). ``QWidget`` doesn't paint
        # its QSS background by default — the styled-background attr is
        # required for the linear-gradient rule to actually render.
        header = QWidget()
        header.setObjectName("appHeader")
        header.setAttribute(Qt.WA_StyledBackground, True)
        header.setFixedHeight(46)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(0)
        if self._manual_mode:
            title_text = "MAMMOGRAPHY TOOLKIT — Annotation Studio"
        else:
            title_text = "MAMMOGRAPHY TOOLKIT — SAM Workspace"
        title = QLabel(title_text)
        title.setObjectName("appTitle")
        hl.addWidget(title)
        hl.addStretch(1)
        self._hdr_subtitle = QLabel("Ready")
        self._hdr_subtitle.setObjectName("appSubtitle")
        hl.addWidget(self._hdr_subtitle)
        root_v.addWidget(header)

        body = QWidget()
        h = QHBoxLayout(body)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        root_v.addWidget(body, stretch=1)

        # ── Side panel (scrollable cards + fixed footer) ──────────────
        side = QWidget()
        side.setObjectName("sidePanel")
        side.setMinimumWidth(340)
        side.setMaximumWidth(380)
        side_v = QVBoxLayout(side)
        side_v.setContentsMargins(0, 0, 0, 0)
        side_v.setSpacing(0)

        # Scroll area wraps the stacked cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_inner = QWidget()
        cards_v = QVBoxLayout(scroll_inner)
        cards_v.setContentsMargins(14, 14, 14, 14)
        cards_v.setSpacing(12)

        # ── Card: Tissue ─────────────────────────────────────────────
        c_tissue, l_tissue = self._card("CLASS")
        self._l_tissue = l_tissue
        self._tissue_row_widgets: list[QWidget] = []
        self._tissue_group = QButtonGroup(self)
        self._tissue_group.buttonClicked.connect(self._on_tissue_changed)
        self._tissue_radios: dict[str, QRadioButton] = {}
        for key in list(TISSUE_PRESETS.keys()):
            self._append_tissue_row(key)
        if self._current_tissue_key and self._current_tissue_key in self._tissue_radios:
            self._tissue_radios[self._current_tissue_key].setChecked(True)

        # Manual-mode-only: button to register a new custom class.
        self._btn_add_class = QPushButton("＋  Add Custom Class")
        self._btn_add_class.clicked.connect(self._on_add_custom_class)
        l_tissue.addWidget(self._btn_add_class)

        cards_v.addWidget(c_tissue)

        # ── Card: Mode (re-labeled per app mode) ─────────────────────
        self._c_mode, l_mode = self._card("DRAWING TOOL" if self._manual_mode else "INTERACTION MODE")
        c_mode = self._c_mode
        self._mode_group = QButtonGroup(self)
        if self._manual_mode:
            self._rb_mode_box = QRadioButton("Bounding Box  (2-click → polygon)")
            self._rb_mode_point = QRadioButton("Polygon  (click vertices)")
            # Don't auto-pick either — Manual mode starts neutral so a
            # casual click on the canvas doesn't accidentally start a
            # vertex chain.
        else:
            self._rb_mode_box = QRadioButton("Bounding Box  (2 clicks)")
            self._rb_mode_point = QRadioButton("Point only  (no box)")
            self._rb_mode_box.setChecked(True)
        self._mode_group.addButton(self._rb_mode_box)
        self._mode_group.addButton(self._rb_mode_point)
        # Use buttonClicked (not toggled) so we don't fire on the silent
        # un-toggle of the previously-selected radio when the user picks
        # the other one. Manual mode starts with neither checked → only
        # explicit user clicks should drive the canvas mode.
        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        l_mode.addWidget(self._rb_mode_box)
        l_mode.addWidget(self._rb_mode_point)
        hint = QLabel(
            "Left-click = positive  •  Right-click = ignore\n"
            "Drag to move  •  Alt+click to delete\n"
            "Ctrl+Wheel = zoom  •  Middle drag = pan\n"
            "Double-click background = fit zoom"
        )
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        l_mode.addWidget(hint)
        cards_v.addWidget(c_mode)

        # ── Card: Settings ───────────────────────────────────────────
        self._c_set, l_set = self._card("SAM SETTINGS")
        c_set = self._c_set
        self._lbl_ignore_r = QLabel("Ignore radius: 20 px")
        l_set.addWidget(self._lbl_ignore_r)
        self._sl_ignore_r = QSlider(Qt.Horizontal)
        self._sl_ignore_r.setMinimum(4)
        self._sl_ignore_r.setMaximum(80)
        self._sl_ignore_r.setValue(20)
        self._sl_ignore_r.valueChanged.connect(self._on_ignore_radius_changed)
        l_set.addWidget(self._sl_ignore_r)

        self._cb_implicit_box = QCheckBox("Implicit box from positive points")
        self._cb_implicit_box.setChecked(True)
        self._cb_implicit_box.toggled.connect(self._trigger_predict)
        l_set.addWidget(self._cb_implicit_box)

        self._cb_class_comp = QCheckBox("Class competition (breast − pectoral)")
        self._cb_class_comp.setChecked(True)
        self._cb_class_comp.toggled.connect(self._trigger_predict)
        l_set.addWidget(self._cb_class_comp)

        self._cb_soft_neg = QCheckBox("Ignore as soft negative to model")
        self._cb_soft_neg.setChecked(True)
        self._cb_soft_neg.toggled.connect(self._trigger_predict)
        l_set.addWidget(self._cb_soft_neg)
        cards_v.addWidget(c_set)

        # ── Card: Actions ────────────────────────────────────────────
        c_act, l_act = self._card("ACTIONS")
        row1 = QHBoxLayout(); row1.setSpacing(6)
        self._b_undo_pos = QPushButton("↶ Pos")
        self._b_undo_pos.setToolTip("Undo last positive point (SAM)")
        self._b_undo_pos.clicked.connect(lambda: self._undo("positive"))
        self._b_undo_ign = QPushButton("↶ Ign")
        self._b_undo_ign.setToolTip("Undo last ignore point (SAM)")
        self._b_undo_ign.clicked.connect(lambda: self._undo("ignore"))
        self._b_reset_box = QPushButton("□ Box")
        self._b_reset_box.setToolTip("Reset bounding box (SAM)")
        self._b_reset_box.clicked.connect(lambda: self._undo("box"))
        # Manual-mode-only: undo last drawn polygon vertex.
        self._b_undo_vertex = QPushButton("↶ Vertex")
        self._b_undo_vertex.setToolTip("Undo last polygon vertex (manual)")
        self._b_undo_vertex.clicked.connect(lambda: self._canvas.undo_last_draw_vertex())
        for b in (self._b_undo_pos, self._b_undo_ign, self._b_reset_box, self._b_undo_vertex):
            row1.addWidget(b)
        row1_w = QWidget(); row1_w.setLayout(row1)
        l_act.addWidget(row1_w)

        row2 = QHBoxLayout(); row2.setSpacing(6)
        b_clear_tissue = QPushButton("Clear Tissue")
        b_clear_tissue.setObjectName("danger")
        b_clear_tissue.clicked.connect(self._clear_active_tissue)
        b_clear_all = QPushButton("Clear All")
        b_clear_all.setObjectName("danger")
        b_clear_all.clicked.connect(self._clear_image)
        row2.addWidget(b_clear_tissue); row2.addWidget(b_clear_all)
        row2_w = QWidget(); row2_w.setLayout(row2)
        l_act.addWidget(row2_w)

        self._btn_rerun = QPushButton("▶  Re-run Prediction")
        self._btn_rerun.setObjectName("primary")
        self._btn_rerun.setMinimumHeight(34)
        self._btn_rerun.clicked.connect(self._submit_inference_now)
        l_act.addWidget(self._btn_rerun)
        cards_v.addWidget(c_act)

        # ── Card: Polygon ────────────────────────────────────────────
        poly_title = "POLYGON" if self._manual_mode else "POLYGON EDIT"
        c_poly, l_poly = self._card(poly_title)

        # Manual mode gets a primary "Draw New Polygon" button.
        self._btn_draw_polygon = QPushButton("✎  Draw New Polygon")
        self._btn_draw_polygon.setObjectName("primary")
        self._btn_draw_polygon.setMinimumHeight(36)
        self._btn_draw_polygon.clicked.connect(self._start_polygon_draw)
        l_poly.addWidget(self._btn_draw_polygon)

        self._btn_edit_polygon = QPushButton("✎  Edit Mask as Polygon")
        self._btn_edit_polygon.setObjectName("success")
        self._btn_edit_polygon.setMinimumHeight(34)
        self._btn_edit_polygon.setCheckable(True)
        self._btn_edit_polygon.toggled.connect(self._toggle_polygon_edit)
        l_poly.addWidget(self._btn_edit_polygon)

        if self._manual_mode:
            poly_hint_text = (
                "Click to add vertices  •  Right-click pops last\n"
                "Double-click or Enter to close  •  Esc cancels\n"
                "After closing, drag vertices to refine."
            )
        else:
            poly_hint_text = (
                "Drag vertex = reshape  •  Click edge = insert\n"
                "Alt+click vertex = delete  (min 3)"
            )
        poly_hint = QLabel(poly_hint_text)
        poly_hint.setObjectName("hint")
        poly_hint.setWordWrap(True)
        l_poly.addWidget(poly_hint)

        # Instance picker (visible in manual mode where multiple polygons
        # per class are common — e.g. after Run Box Detector).
        inst_row = QHBoxLayout()
        inst_row.setSpacing(6)
        self._btn_prev_instance = QPushButton("◀  Prev")
        self._btn_prev_instance.setToolTip("Edit the previous polygon instance")
        self._btn_prev_instance.clicked.connect(lambda: self._step_instance(-1))
        self._btn_next_instance = QPushButton("Next  ▶")
        self._btn_next_instance.setToolTip("Edit the next polygon instance")
        self._btn_next_instance.clicked.connect(lambda: self._step_instance(+1))
        self._lbl_instance_pos = QLabel("—")
        self._lbl_instance_pos.setObjectName("hint")
        inst_row.addWidget(self._btn_prev_instance)
        inst_row.addWidget(self._lbl_instance_pos, 1, Qt.AlignCenter)
        inst_row.addWidget(self._btn_next_instance)
        inst_w = QWidget(); inst_w.setLayout(inst_row)
        l_poly.addWidget(inst_w)
        # We currently edit the LAST instance; this index counts from the
        # end (0 = last, 1 = second to last, ...).
        self._edit_instance_back_idx: int = 0

        cards_v.addWidget(c_poly)

        # ── Card: Display (mask opacity slider) ──────────────────────
        self._c_display, l_disp = self._card("DISPLAY")
        self._lbl_mask_alpha = QLabel("Mask opacity: 15%")
        l_disp.addWidget(self._lbl_mask_alpha)
        self._sl_mask_alpha = QSlider(Qt.Horizontal)
        self._sl_mask_alpha.setMinimum(0)
        self._sl_mask_alpha.setMaximum(255)
        # Default 15 % — keeps the overlay subtle so the underlying
        # mammogram detail stays legible.
        self._sl_mask_alpha.setValue(38)
        self._sl_mask_alpha.setToolTip(
            "Translucency of segmentation overlays. 0% = fully transparent "
            "(image only), 100% = solid color."
        )
        self._sl_mask_alpha.valueChanged.connect(self._on_mask_alpha_changed)
        l_disp.addWidget(self._sl_mask_alpha)
        cards_v.addWidget(self._c_display)

        # ── Card: Auto detector (manual-only) ────────────────────────
        self._c_detector, l_det = self._card("AUTO DETECTOR")
        self._btn_run_detector = QPushButton("⚙  Run Box Detector")
        self._btn_run_detector.setObjectName("primary")
        self._btn_run_detector.setMinimumHeight(36)
        self._btn_run_detector.setToolTip(
            "Runs the bundled YOLO box detector and adds every detected "
            "rectangle as a polygon under the currently-active class. "
            "Edit the corners afterward with the Polygon tools."
        )
        self._btn_run_detector.clicked.connect(self._on_run_detector)
        l_det.addWidget(self._btn_run_detector)
        det_hint = QLabel(
            "Adds one polygon per detected box to the active class.\n"
            "Use Polygon Edit and the corner handles to refine them."
        )
        det_hint.setObjectName("hint")
        det_hint.setWordWrap(True)
        l_det.addWidget(det_hint)
        cards_v.addWidget(self._c_detector)

        # ── Card: Files ──────────────────────────────────────────────
        c_files, l_files = self._card("FILES")
        nav_row = QHBoxLayout(); nav_row.setSpacing(6)
        self._btn_prev = QPushButton("◀  Prev")
        self._btn_prev.clicked.connect(self._goto_prev)
        self._btn_next = QPushButton("Next  ▶")
        self._btn_next.clicked.connect(self._goto_next)
        nav_row.addWidget(self._btn_prev); nav_row.addWidget(self._btn_next)
        nav_w = QWidget(); nav_w.setLayout(nav_row)
        l_files.addWidget(nav_w)

        self._image_list = QListWidget()
        self._image_list.setMinimumHeight(120)
        self._image_list.setMaximumHeight(200)
        self._image_list.itemActivated.connect(self._on_list_activated)
        self._image_list.itemClicked.connect(self._on_list_activated)
        l_files.addWidget(self._image_list)
        cards_v.addWidget(c_files)

        cards_v.addStretch(1)
        scroll.setWidget(scroll_inner)
        side_v.addWidget(scroll, stretch=1)

        # ── Pinned footer: Save All ──────────────────────────────────
        footer = QWidget()
        footer.setObjectName("sideFooter")
        fl = QVBoxLayout(footer)
        fl.setContentsMargins(14, 12, 14, 14)
        self._btn_calc = QPushButton("📐  Hesapla (PNL / CC)")
        self._btn_calc.setMinimumHeight(40)
        self._btn_calc.setToolTip(
            "Nipple + Pectoral (MLO) etiketliyse PNL'i, sadece Nipple + Breast "
            "(CC) ise meme derinliğini çizer (C)"
        )
        self._btn_calc.clicked.connect(self._calculate_geometry)
        fl.addWidget(self._btn_calc)
        self._btn_save_all = QPushButton("💾  Save All to YOLO")
        self._btn_save_all.setObjectName("primary")
        self._btn_save_all.setMinimumHeight(40)
        self._btn_save_all.clicked.connect(self._save_all)
        fl.addWidget(self._btn_save_all)
        side_v.addWidget(footer)

        h.addWidget(side)

        # ── Canvas ───────────────────────────────────────────────────
        self._canvas = MammoCanvas()
        # Manual mode never wants the SAM-style "right-click drops an
        # ignore point" behavior — right-click is reserved for the
        # box context menu (Rename / Delete).
        self._canvas.set_allow_right_click_ignore(not self._manual_mode)
        self._canvas.promptsMutated.connect(self._on_canvas_mutated)
        self._canvas.polygonMutated.connect(self._on_polygon_mutated)
        self._canvas.polygonDrawCompleted.connect(self._on_polygon_draw_completed)
        self._canvas.activePolygonRequested.connect(self._on_active_polygon_requested)
        self._canvas.backgroundClickedInEdit.connect(self._on_background_clicked_in_edit)
        self._canvas.polygonRenameRequested.connect(self._on_polygon_rename_requested)
        self._canvas.polygonContextMenuRequested.connect(self._on_polygon_context_menu_requested)
        self._canvas.zoomChanged.connect(self._on_zoom_changed)
        self._canvas.windowLevelChanged.connect(self._on_window_level_changed)
        h.addWidget(self._canvas, stretch=1)

        # Manual mode → hide SAM-only controls; show polygon-only undo.
        if self._manual_mode:
            # Drawing-tool card stays visible (relabeled above) — user can
            # pick between Bounding-Box and Polygon.
            self._c_set.setVisible(False)
            self._btn_rerun.setVisible(False)
            self._b_undo_pos.setVisible(False)
            self._b_undo_ign.setVisible(False)
            self._b_reset_box.setVisible(False)
        else:
            # SAM mode hides the explicit "Draw New Polygon" — polygon edit
            # is reached via the toggle once SAM has produced a mask.
            self._btn_draw_polygon.setVisible(False)
            # The vertex-undo button is manual-only.
            self._b_undo_vertex.setVisible(False)
            # Custom classes & auto detector are manual-only features.
            self._btn_add_class.setVisible(False)
            self._c_detector.setVisible(False)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._lbl_device = QLabel("Device: —")
        self._lbl_zoom = QLabel("Zoom: 100%")
        self._lbl_image_idx = QLabel("Image: —")
        self._lbl_dirty = QLabel("Pending: 0 images")
        self._lbl_busy = QLabel("")
        self._lbl_wl = QLabel("W/L: 0/0")
        for w in (self._lbl_image_idx, self._lbl_dirty, self._lbl_zoom, self._lbl_wl, self._lbl_busy, self._lbl_device):
            w.setObjectName("statusInfo")
            self._status.addPermanentWidget(w)

        # Theme
        self.setStyleSheet(QSS)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionHeader")
        return lbl

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        """Return a (card_frame, inner_layout) pair. Caller adds widgets to
        the layout; the frame is added to the side panel."""
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        if title:
            hdr = QLabel(title)
            hdr.setObjectName("sectionHeader")
            layout.addWidget(hdr)
        return frame, layout

    def _tissue_button_label(self, key: str, preset: TissuePreset) -> str:
        yolo_id = preset.class_id - 1
        return f"[{yolo_id}]  {preset.label}"

    # Built-in tissue keys (cannot be deleted by the user)
    _BUILTIN_TISSUE_KEYS = {"pectoral", "breast", "nipple"}

    def _append_tissue_row(self, key: str) -> None:
        """Append one tissue radio row to the Class card. Custom keys get a
        small ✕ delete button on the right; built-ins do not."""
        preset = TISSUE_PRESETS[key]
        row = QHBoxLayout()
        row.setSpacing(8)
        row.setContentsMargins(0, 0, 0, 0)
        dot = QLabel()
        dot.setObjectName("tissueDot")
        r, g, b = TISSUE_RGB.get(key, (200, 200, 200))
        dot.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border-radius: 6px;"
        )
        row.addWidget(dot)
        rb = QRadioButton(self._tissue_button_label(key, preset))
        row.addWidget(rb, stretch=1)
        self._tissue_radios[key] = rb
        self._tissue_group.addButton(rb)
        if key not in self._BUILTIN_TISSUE_KEYS:
            x_btn = QPushButton("✕")
            x_btn.setToolTip(f"Remove class '{preset.label}'")
            x_btn.setFixedSize(22, 22)
            x_btn.setObjectName("danger")
            x_btn.clicked.connect(lambda _=False, k=key: self._on_remove_custom_class(k))
            row.addWidget(x_btn)
        wrap = QWidget()
        wrap.setProperty("tissue_key", key)
        wrap.setLayout(row)
        # Place above the "+ Add Class" button if it already exists.
        insert_at = self._l_tissue.count()
        if hasattr(self, "_btn_add_class") and self._btn_add_class is not None:
            idx = self._l_tissue.indexOf(self._btn_add_class)
            if idx >= 0:
                insert_at = idx
        self._l_tissue.insertWidget(insert_at, wrap)
        self._tissue_row_widgets.append(wrap)

    def _on_add_custom_class(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            "Add Custom Class",
            "Class name (e.g. 'mass', 'calcification'):",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        # Color picker — required (otherwise auto-pick a unique color).
        chosen = QColorDialog.getColor(parent=self, title="Pick a color (Cancel = auto)")
        color_rgb: Optional[tuple[int, int, int]] = None
        if chosen.isValid():
            color_rgb = (chosen.red(), chosen.green(), chosen.blue())

        try:
            key = register_custom_tissue(name, color_rgb)
        except _CustomClassError as e:
            QMessageBox.warning(self, "Color in use", str(e))
            return
        self._custom_class_keys.append(key)
        self._append_tissue_row(key)
        log.info("Registered custom class: %s (id=%d)", name, TISSUE_PRESETS[key].class_id)
        self._status.showMessage(
            f"Added class '{name}' (YOLO id {TISSUE_PRESETS[key].class_id - 1}).",
            5000,
        )
        self._tissue_radios[key].setChecked(True)
        self._switch_active_tissue(key)

    def _on_remove_custom_class(self, key: str) -> None:
        if key in self._BUILTIN_TISSUE_KEYS:
            return  # safety
        label = TISSUE_PRESETS.get(key)
        label_text = label.label if label else key
        ans = QMessageBox.question(
            self, "Remove class",
            f"Remove class '{label_text}'?\n\n"
            "All masks/polygons stored for this class on every image will also be cleared.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return
        # Clean session state for this tissue on every image.
        for ann in self._session._store.values():  # noqa: SLF001 — internal access OK
            ann.masks.pop(key, None)
            ann.prompts.pop(key, None)
            ann.polygons.pop(key, None)
        # Clear canvas overlay if currently displayed.
        self._canvas.set_mask(key, None)
        # If the deleted tissue was active, switch back to whatever class
        # remains in the panel. Falling back to a built-in name (e.g.
        # "pectoral") would crash Manual mode where built-ins don't
        # exist — so we pick the first remaining radio, or None if the
        # user just removed the last class.
        if self._current_tissue_key == key:
            remaining_keys = [k for k in self._tissue_radios.keys() if k != key]
            if remaining_keys:
                fallback = remaining_keys[0]
                self._tissue_radios[fallback].setChecked(True)
                self._switch_active_tissue(fallback)
            else:
                self._current_tissue_key = None
                self._edit_instance_back_idx = 0
                self._refresh_instance_label()
        # Remove the row widget.
        for w in list(self._tissue_row_widgets):
            if w.property("tissue_key") == key:
                self._l_tissue.removeWidget(w)
                w.deleteLater()
                self._tissue_row_widgets.remove(w)
                break
        # Remove from registries.
        rb = self._tissue_radios.pop(key, None)
        if rb is not None:
            self._tissue_group.removeButton(rb)
        if key in self._custom_class_keys:
            self._custom_class_keys.remove(key)
        unregister_custom_tissue(key)
        log.info("Removed custom class: %s", key)
        self._refresh_status()

    def _build_menu_and_toolbar(self) -> None:
        menu = self.menuBar()

        m_file = menu.addMenu("&File")
        a_open_img = QAction("Open &Image…", self)
        a_open_img.setShortcut(QKeySequence.Open)
        a_open_img.triggered.connect(self._action_open_image)
        m_file.addAction(a_open_img)

        a_open_folder = QAction("Open &Folder…", self)
        a_open_folder.setShortcut("Ctrl+Shift+O")
        a_open_folder.triggered.connect(self._action_open_folder)
        m_file.addAction(a_open_folder)

        m_file.addSeparator()

        a_save = QAction("&Save All to YOLO", self)
        a_save.setShortcut(QKeySequence.Save)
        a_save.triggered.connect(self._save_all)
        m_file.addAction(a_save)

        m_file.addSeparator()

        a_back = QAction("← Back to Main Menu", self)
        a_back.setShortcut("Ctrl+M")
        a_back.triggered.connect(self._request_back_to_menu)
        m_file.addAction(a_back)

        a_exit = QAction("E&xit", self)
        a_exit.setShortcut("Ctrl+Q")
        a_exit.triggered.connect(self.close)
        m_file.addAction(a_exit)

        m_view = menu.addMenu("&View")
        a_zoom_in = QAction("Zoom &In", self); a_zoom_in.setShortcut("Ctrl++")
        a_zoom_in.triggered.connect(lambda: self._canvas.zoom_in())
        m_view.addAction(a_zoom_in)
        a_zoom_out = QAction("Zoom &Out", self); a_zoom_out.setShortcut("Ctrl+-")
        a_zoom_out.triggered.connect(lambda: self._canvas.zoom_out())
        m_view.addAction(a_zoom_out)
        a_fit = QAction("&Fit to View", self); a_fit.setShortcut("Ctrl+0")
        a_fit.triggered.connect(lambda: self._canvas.fit_to_view())
        m_view.addAction(a_fit)

        a_commit = QAction("✓ Commit Polygon", self)
        a_commit.setShortcut("Ctrl+Return")
        a_commit.setToolTip("Lock the current polygon edit and store it as the saved label (Ctrl+Enter)")
        a_commit.triggered.connect(self._commit_current_polygon)
        self._a_commit = a_commit

        a_reset_wl = QAction("Reset W/L", self)
        a_reset_wl.setToolTip("Reset brightness/contrast (right-drag = window/level)")
        a_reset_wl.triggered.connect(lambda: self._canvas.reset_window_level())

        a_calc = QAction("📐 Hesapla (PNL/CC)", self)
        a_calc.setShortcut("C")
        a_calc.setToolTip("Etiketlenen dokulardan PNL / CC derinliğini hesapla ve çiz (C)")
        a_calc.triggered.connect(self._calculate_geometry)

        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        # Relabel some actions to read better in the toolbar.
        a_open_img.setIconText("Open Image")
        a_open_folder.setIconText("Open Folder")
        a_save.setIconText("Save All to YOLO")
        a_zoom_in.setIconText("Zoom +")
        a_zoom_out.setIconText("Zoom −")
        a_fit.setIconText("Fit View")
        a_reset_wl.setIconText("Reset W/L")
        a_calc.setIconText("📐 Hesapla")
        a_commit.setIconText("✓ Commit")
        a_back.setIconText("← Menu")

        tb.addAction(a_open_img)
        tb.addAction(a_open_folder)
        tb.addSeparator()
        tb.addAction(a_save)
        tb.addSeparator()
        tb.addAction(a_zoom_in)
        tb.addAction(a_zoom_out)
        tb.addAction(a_fit)
        tb.addAction(a_reset_wl)
        tb.addSeparator()
        tb.addAction(a_calc)
        tb.addSeparator()
        tb.addAction(a_commit)
        tb.addSeparator()
        tb.addAction(a_back)

    def _build_shortcuts(self) -> None:
        # Left/Right navigation is handled by an application-level event
        # filter (see eventFilter) rather than a QShortcut. The canvas is a
        # QGraphicsView with StrongFocus; once you click into it, it consumes
        # the arrow keys for its own scrolling before a WindowShortcut can
        # fire — which is exactly why arrow navigation "stops working" after
        # the first click. Filtering at the QApplication level catches the
        # key press before any focused child widget sees it.
        QApplication.instance().installEventFilter(self)
        QShortcut(QKeySequence("B"), self, activated=lambda: self._rb_mode_box.setChecked(True))
        QShortcut(QKeySequence("P"), self, activated=lambda: self._rb_mode_point.setChecked(True))
        QShortcut(QKeySequence("1"), self, activated=lambda: self._select_tissue("pectoral"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._select_tissue("breast"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self._select_tissue("nipple"))
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo_last_label)
        QShortcut(QKeySequence(Qt.Key_Delete), self, activated=self._delete_active_polygon)

    # ─── File actions ──────────────────────────────────────────────────

    def _action_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", str(self._folder or Path.home()),
            _IMAGE_DIALOG_FILTER,
        )
        if path:
            self._open_image_path(Path(path))

    def _action_open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open folder", str(self._folder or Path.home()))
        if path:
            self._open_folder_path(Path(path))

    def _open_image_path(self, path: Path) -> None:
        # Single-image mode: folder = parent, but only show this image in list.
        self._folder = path.parent
        self._image_paths = [path]
        self._current_idx = -1
        self._refresh_image_list()
        self._load_image_by_index(0)

    def _resume_index(self, paths: list[Path]) -> int:
        """Where to reopen a partially labeled folder.

        A labeled image is one that already has ``<images_dir>/labels/<stem>.txt``
        (the path :mod:`qt_app.session` exports to). We resume right after the
        LAST labeled image rather than at the first unlabeled one, so images
        that were deliberately skipped mid-run do not drag the session back to
        the beginning. Returns 0 when nothing is labeled yet.
        """
        last_labeled = -1
        for idx, p in enumerate(paths):
            if (p.parent / "labels" / f"{p.stem}.txt").exists():
                last_labeled = idx
        return min(last_labeled + 1, len(paths) - 1)

    def _open_folder_path(self, folder: Path) -> None:
        paths = list_images(folder, recursive=True)
        if not paths:
            log.warning("No supported images found under %s", folder)
            QMessageBox.warning(
                self, "No images found",
                f"No images found under:\n{folder}\n\n"
                "Searched recursively for: .png .jpg .jpeg .bmp .tif .dcm",
            )
            return
        log.info("Opening folder %s — %d image(s) found", folder, len(paths))
        self._folder = folder
        self._image_paths = paths
        self._current_idx = -1
        self._refresh_image_list()
        start = self._resume_index(paths)
        if start:
            log.info("Resuming at %d/%d (%d already labeled)",
                     start + 1, len(paths), start)
        self._load_image_by_index(start)

    def _refresh_image_list(self) -> None:
        self._image_list.clear()
        for p in self._image_paths:
            item = QListWidgetItem(p.name)
            self._image_list.addItem(item)

    def _on_list_activated(self, item: QListWidgetItem) -> None:
        idx = self._image_list.row(item)
        if idx != self._current_idx:
            self._load_image_by_index(idx)

    # ─── Image loading / navigation ────────────────────────────────────

    def _load_image_by_index(self, idx: int) -> None:
        if not (0 <= idx < len(self._image_paths)):
            return

        # Decode FIRST. We only commit state changes (persisting prompts,
        # exiting polygon edit, advancing _current_idx, …) once the new
        # image has been validated — otherwise a bad file in the folder
        # would leave the session pointing at something nothing can
        # render.
        path = self._image_paths[idx]
        rgb = load_image_rgb(path)
        if rgb is None:
            log.error("Failed to load image: %s", path)
            QMessageBox.warning(self, "Load failed", f"Could not open: {path}")
            return
        if not (
            rgb.ndim == 3 and rgb.shape[2] in (3, 4)
            and rgb.shape[0] >= 1 and rgb.shape[1] >= 1
        ):
            log.error("Unsupported image shape for %s: %s", path, rgb.shape)
            QMessageBox.warning(
                self, "Cannot display image",
                f"This file decoded to an unsupported shape:\n"
                f"{path.name}  →  {rgb.shape}\n\n"
                "Skipping. Pick a different file. See logs/qt_app.log.",
            )
            return

        log.info("Loaded image %d/%d: %s (shape=%s)",
                 idx + 1, len(self._image_paths), path.name, rgb.shape)

        # NOW the new image is good; commit state changes.
        self._exit_polygon_edit_if_active()
        self._persist_current_prompts()

        self._current_idx = idx
        self._current_image_rgb = rgb
        self._image_list.setCurrentRow(idx)

        self._session.get_or_create(path, image_hw=(rgb.shape[0], rgb.shape[1]))

        try:
            self._canvas.load_image(rgb)
        except Exception:
            log.exception("Canvas refused to render %s", path)
            QMessageBox.warning(
                self, "Cannot display image",
                f"This file could not be rendered:\n{path.name}\n\n"
                "Skipping. See logs/qt_app.log for details.",
            )
            return
        self._canvas.set_active_tissue(self._current_tissue_key)
        self._canvas.set_ignore_radius(self._sl_ignore_r.value())
        if self._manual_mode:
            # Apply the currently-selected drawing tool (Box / Polygon).
            # If the user hasn't picked one yet, stay in IDLE so clicks
            # on the canvas don't accidentally place vertices.
            if self._rb_mode_box.isChecked():
                self._canvas.set_mode(MODE_BOX)
            elif self._rb_mode_point.isChecked():
                self._canvas.start_polygon_draw()
            else:
                self._canvas.set_mode(MODE_IDLE)

        # Restore any previously-drawn masks for this image
        ann = self._session.get(path)
        if ann is not None:
            for tk, mask in ann.masks.items():
                self._canvas.set_mask(tk, mask)
            # Restore active tissue's prompt state
            self._canvas.set_prompt_state(self._session.get_prompts(path, self._current_tissue_key))

        self._refresh_status()
        self._edit_instance_back_idx = 0
        self._refresh_instance_label()
        self._btn_prev.setEnabled(idx > 0)
        self._btn_next.setEnabled(idx < len(self._image_paths) - 1)

    def _goto_prev(self) -> None:
        if self._current_idx > 0:
            self._load_image_by_index(self._current_idx - 1)

    def _goto_next(self) -> None:
        if 0 <= self._current_idx < len(self._image_paths) - 1:
            self._load_image_by_index(self._current_idx + 1)

    # ─── Tissue / mode handlers ────────────────────────────────────────

    def _on_tissue_changed(self) -> None:
        for key, rb in self._tissue_radios.items():
            if rb.isChecked():
                self._switch_active_tissue(key)
                return

    def _select_tissue(self, key: str) -> None:
        # Programmatic setChecked does NOT fire buttonClicked, so drive the
        # state machine directly. Keep the radio in sync visually.
        if key not in self._tissue_radios:
            return
        self._tissue_radios[key].setChecked(True)
        self._switch_active_tissue(key)

    def _switch_active_tissue(self, new_key: Optional[str]) -> None:
        if new_key == self._current_tissue_key:
            return
        # Guard against switching to a tissue that doesn't exist in the
        # current registry — Manual mode wipes built-ins, so passing
        # "pectoral" or any other ghost id would crash downstream.
        if new_key and new_key not in TISSUE_PRESETS:
            log.warning("Refusing to switch to unregistered tissue: %r", new_key)
            return
        self._exit_polygon_edit_if_active()
        self._persist_current_prompts()
        self._current_tissue_key = new_key
        self._edit_instance_back_idx = 0
        if new_key:
            self._canvas.set_active_tissue(new_key)
            if self._current_image_path is not None:
                self._canvas.set_prompt_state(
                    self._session.get_prompts(self._current_image_path, new_key),
                )
        self._refresh_instance_label()

    def _on_mode_changed(self, *args) -> None:
        # ``*args`` swallows the QAbstractButton that QButtonGroup.buttonClicked
        # passes — we don't need it because we read the radio states directly.
        if self._manual_mode:
            if self._rb_mode_box.isChecked():
                self._canvas.set_mode(MODE_BOX)
            elif self._rb_mode_point.isChecked():
                self._canvas.start_polygon_draw()
            else:
                # No tool picked yet → keep canvas inert.
                self._canvas.set_mode(MODE_IDLE)
            return
        mode = MODE_BOX if self._rb_mode_box.isChecked() else MODE_POINT
        self._canvas.set_mode(mode)

    def _on_mask_alpha_changed(self, value: int) -> None:
        pct = int(round(value / 255.0 * 100))
        self._lbl_mask_alpha.setText(f"Mask opacity: {pct}%")
        self._canvas.set_mask_alpha(value)

    def _on_ignore_radius_changed(self, value: int) -> None:
        self._lbl_ignore_r.setText(f"Ignore radius: {value} px")
        self._canvas.set_ignore_radius(value)
        self._trigger_predict()

    # ─── Inference flow ────────────────────────────────────────────────

    def _on_canvas_mutated(self) -> None:
        # In manual mode, a finalized box is consumed immediately as a
        # 4-point polygon so the label persists across tissue/image
        # switches. Multiple boxes per class accumulate (each new box adds
        # a new polygon instance — multi-instance YOLO label).
        if self._manual_mode and self._current_image_path is not None \
                and self._current_image_rgb is not None:
            if not self._current_tissue_key:
                self._status.showMessage(
                    "Add a class first (＋ Add Custom Class) before drawing.", 5000,
                )
                return
            state = self._canvas.get_prompt_state()
            box = state.box.to_xyxy()
            if box is not None:
                x0, y0, x1, y1 = box.tolist()
                poly = [(float(x0), float(y0)), (float(x1), float(y0)),
                        (float(x1), float(y1)), (float(x0), float(y1))]
                state.box.reset()
                self._canvas.set_prompt_state(state)
                self._session.append_polygon(
                    self._current_image_path, self._current_tissue_key, poly,
                )
                self._edit_instance_back_idx = 0
                # Persist full mask (used by save / class competition).
                full_mask = self._rebuild_tissue_mask(self._current_tissue_key)
                self._session.set_mask(
                    self._current_image_path, self._current_tissue_key, full_mask,
                )
                # Display edit-time mask (excludes the polygon we're about
                # to drop into edit mode).
                self._apply_canvas_mask_for_edit(self._current_tissue_key)
                self._canvas.enter_polygon_edit(poly)
                self._btn_edit_polygon.blockSignals(True)
                self._btn_edit_polygon.setChecked(True)
                self._btn_edit_polygon.blockSignals(False)
                instance_count = len(
                    self._session.get_polygons(self._current_image_path, self._current_tissue_key)
                )
                log.info(
                    "Manual box -> polygon for '%s' (instances now: %d)",
                    self._current_tissue_key, instance_count,
                )
                self._refresh_status()
                return
            # Other prompt mutations (stray points) — manual has no SAM.
            return

        # SAM-mode behavior unchanged.
        self._exit_polygon_edit_if_active()
        if self._current_image_path is not None:
            self._session.set_polygon(self._current_image_path, self._current_tissue_key, None)
        self._trigger_predict()

    def _rebuild_tissue_mask(
        self, tissue_key: str, exclude_idx: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """OR-merge every stored polygon for ``tissue_key`` into a single
        binary mask. If ``exclude_idx`` is given, that polygon is left
        OUT of the union — used while it's being edited so the canvas
        doesn't display a stale "ghost" at the polygon's previous
        position underneath the moving outline."""
        if self._current_image_path is None or self._current_image_rgb is None:
            return None
        polys = self._session.get_polygons(self._current_image_path, tissue_key)
        if not polys:
            return None
        h, w = self._current_image_rgb.shape[:2]
        accum = np.zeros((h, w), dtype=np.uint8)
        for i, p in enumerate(polys):
            if i == exclude_idx:
                continue
            accum = np.maximum(accum, polygon_to_mask(p, (h, w)))
        return accum

    def _apply_canvas_mask_for_edit(self, tissue_key: str) -> None:
        """Display the union of every polygon EXCEPT the one currently
        being edited, and push the OTHER polygons to the canvas as
        dashed outlines so the user can see every instance and click any
        of them to switch the edit cursor."""
        if self._current_image_path is None:
            return
        polys = self._session.get_polygons(self._current_image_path, tissue_key)
        if not polys:
            self._canvas.set_mask(tissue_key, None)
            self._canvas.set_other_polygons([])
            return
        n = len(polys)
        exclude_idx = max(0, min(n - 1, n - 1 - self._edit_instance_back_idx))
        mask = self._rebuild_tissue_mask(tissue_key, exclude_idx=exclude_idx)
        self._canvas.set_mask(tissue_key, mask)
        # Build (back_idx, polygon) tuples for siblings (everything except
        # the active polygon). back_idx is what _step_instance uses, so
        # clicking on a sibling can directly translate to the right
        # picker offset.
        siblings: list[tuple[int, list[tuple[float, float]]]] = []
        for i, p in enumerate(polys):
            if i == exclude_idx:
                continue
            siblings.append((n - 1 - i, list(p)))
        self._canvas.set_other_polygons(siblings)

    def _apply_canvas_mask_full(self, tissue_key: str) -> None:
        """Display the full union of every polygon — used after exiting
        polygon edit so the canvas shows the committed truth, no
        sibling outlines."""
        if self._current_image_path is None:
            return
        mask = self._rebuild_tissue_mask(tissue_key)
        self._canvas.set_mask(tissue_key, mask)
        self._canvas.set_other_polygons([])

    def _on_background_clicked_in_edit(self, x: float, y: float) -> None:
        """User clicked on empty canvas while a polygon was being edited.

        Manual + Box tool: commit the current edit and treat the click
        as the FIRST corner of a brand-new box — so the user can chain
        boxes without manually toggling Edit Mask as Polygon off.
        """
        if not self._manual_mode:
            return
        if not self._current_tissue_key:
            self._status.showMessage("Add a class first.", 3000)
            return
        if not self._rb_mode_box.isChecked():
            return
        # Exit the current polygon edit (mask flips to full union).
        self._exit_polygon_edit_if_active()
        # Switch canvas to BOX tool and inject the first corner so the
        # user sees the rubber-band preview immediately.
        self._canvas.set_mode(MODE_BOX)
        state = self._canvas.get_prompt_state()
        state.box.reset()
        state.box.x0, state.box.y0 = float(x), float(y)
        self._canvas.set_prompt_state(state)
        self._canvas._render_first_corner(x, y)
        self._canvas._render_box_preview(x, y, x, y)

    def _on_polygon_rename_requested(self, back_idx: int) -> None:
        """User double-clicked on a polygon — open the class picker so
        they can re-label it to a different existing class."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        if not polys:
            return
        idx = max(0, min(len(polys) - 1, len(polys) - 1 - back_idx))
        self._open_rename_dialog_for_polygon(idx)

    def _on_polygon_context_menu_requested(self, back_idx: int, global_pos) -> None:
        """Right-click on a polygon: small menu — rename or delete."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        if not polys:
            return
        idx = max(0, min(len(polys) - 1, len(polys) - 1 - back_idx))
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        act_rename = menu.addAction("Rename class…")
        act_delete = menu.addAction("Delete box")
        chosen = menu.exec_(global_pos)
        if chosen is act_rename:
            self._open_rename_dialog_for_polygon(idx)
        elif chosen is act_delete:
            self._delete_polygon_at(idx)

    def _open_rename_dialog_for_polygon(self, idx: int) -> None:
        """Reassign a polygon to a different existing class via a small
        chooser dialog. New polygon is appended to the chosen tissue's
        list; old is removed from current tissue."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        from_tissue = self._current_tissue_key
        candidates = [k for k in TISSUE_PRESETS.keys() if k != from_tissue]
        if not candidates:
            QMessageBox.information(
                self, "No other classes",
                "There are no other classes to reassign to. Use ＋ Add Custom Class first.",
            )
            return
        labels = [TISSUE_PRESETS[k].label for k in candidates]
        chosen_label, ok = QInputDialog.getItem(
            self,
            "Reassign box to class",
            "Pick the new class for this box:",
            labels,
            0,
            False,
        )
        if not ok:
            return
        new_tissue = candidates[labels.index(chosen_label)]
        polys = self._session.get_polygons(self._current_image_path, from_tissue)
        if not (0 <= idx < len(polys)):
            return
        poly = polys[idx]
        self._session.remove_polygon_at(self._current_image_path, from_tissue, idx)
        self._session.append_polygon(self._current_image_path, new_tissue, poly)
        # Rebuild both masks.
        for tk in (from_tissue, new_tissue):
            m = self._rebuild_tissue_mask(tk)
            self._session.set_mask(self._current_image_path, tk, m)
            self._canvas.set_mask(tk, m)
        self._exit_polygon_edit_if_active()
        self._refresh_instance_label()
        log.info("Reassigned polygon %d from %s → %s", idx, from_tissue, new_tissue)
        self._status.showMessage(
            f"Box reassigned to class '{TISSUE_PRESETS[new_tissue].label}'.", 4000,
        )

    def _delete_active_polygon(self) -> None:
        """Delete the polygon currently being edited (Del key)."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        if not polys:
            return
        idx = max(0, min(len(polys) - 1, len(polys) - 1 - self._edit_instance_back_idx))
        self._delete_polygon_at(idx)
        self._status.showMessage("Box deleted.", 2500)

    def _delete_polygon_at(self, idx: int) -> None:
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        if not (0 <= idx < len(polys)):
            return
        self._exit_polygon_edit_if_active()
        self._session.remove_polygon_at(self._current_image_path, self._current_tissue_key, idx)
        mask = self._rebuild_tissue_mask(self._current_tissue_key)
        self._session.set_mask(self._current_image_path, self._current_tissue_key, mask)
        self._canvas.set_mask(self._current_tissue_key, mask)
        log.info("Deleted polygon %d from '%s'", idx, self._current_tissue_key)
        self._edit_instance_back_idx = 0
        self._refresh_instance_label()

    def _on_active_polygon_requested(self, back_idx: int) -> None:
        """Canvas click landed inside a sibling polygon — switch the
        edit cursor to it. The new index becomes the picker position so
        Prev/Next continues to make sense from there."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        n = len(polys)
        if n == 0:
            return
        new_back = max(0, min(n - 1, int(back_idx)))
        if new_back == self._edit_instance_back_idx:
            return
        self._edit_instance_back_idx = new_back
        target = polys[-1 - new_back]
        self._apply_canvas_mask_for_edit(self._current_tissue_key)
        self._canvas.enter_polygon_edit(target)
        self._btn_edit_polygon.blockSignals(True)
        self._btn_edit_polygon.setChecked(True)
        self._btn_edit_polygon.blockSignals(False)
        self._refresh_instance_label()
        self._status.showMessage(
            f"Now editing box {n - new_back} of {n}.", 3500,
        )

    def _trigger_predict(self) -> None:
        self._predict_debounce.start()

    def _submit_inference_now(self) -> None:
        """Bypass the debounce timer — invoked from the Re-run button."""
        if self._predict_debounce.isActive():
            self._predict_debounce.stop()
        self._submit_inference()

    def _submit_inference(self) -> None:
        if self._manual_mode or self._worker is None:
            return
        if self._current_image_rgb is None or self._current_image_path is None:
            return
        state = self._canvas.get_prompt_state()
        if state.is_empty():
            # No prompts → clear the active mask
            self._canvas.set_mask(self._current_tissue_key, None)
            self._session.set_mask(self._current_image_path, self._current_tissue_key, None)
            self._session.set_prompts(self._current_image_path, self._current_tissue_key, state)
            self._refresh_status()
            return

        tissue = TISSUE_PRESETS[self._current_tissue_key]
        settings = InferenceSettings(
            ignore_radius_px=self._sl_ignore_r.value(),
            use_implicit_box=self._cb_implicit_box.isChecked(),
            use_class_competition=self._cb_class_comp.isChecked(),
            use_soft_negative=self._cb_soft_neg.isChecked(),
        )
        ann = self._session.get(self._current_image_path)
        masks_by_tissue = dict(ann.masks) if ann is not None else {}
        # The active tissue's *previous* mask shouldn't drive class
        # competition against itself.
        masks_by_tissue.pop(self._current_tissue_key, None)

        req = InferenceRequest(
            image_path=str(self._current_image_path),
            tissue_key=self._current_tissue_key,
            rgb=self._current_image_rgb,
            state=state,
            tissue=tissue,
            settings=settings,
            masks_by_tissue=masks_by_tissue,
        )
        rid = self._worker.submit(req)
        self._latest_request_id = rid
        # Persist prompts immediately so navigation preserves them even if
        # the inference is still in flight.
        self._session.set_prompts(self._current_image_path, self._current_tissue_key, state)

    def _on_inference_started(self, rid: int) -> None:
        self._lbl_busy.setText("◐ predicting…")

    def _on_inference_finished(self, rid: int, image_path: str, tissue_key: str, result) -> None:
        if rid != self._latest_request_id:
            return  # stale — user moved on
        self._lbl_busy.setText("")
        path = Path(image_path)
        mask = None if result is None else result.mask
        # A new SAM mask invalidates any previously-edited polygon.
        self._session.set_polygon(path, tissue_key, None)
        if self._current_image_path == path and tissue_key == self._current_tissue_key:
            self._canvas.set_mask(tissue_key, mask)
        self._session.set_mask(path, tissue_key, mask)
        self._refresh_status()

    def _on_inference_failed(self, rid: int, image_path: str, tissue_key: str, msg: str) -> None:
        self._lbl_busy.setText("")
        self._status.showMessage(f"Inference error: {msg}", 6000)

    # ─── Polygon edit flow ─────────────────────────────────────────────

    def _toggle_polygon_edit(self, checked: bool) -> None:
        if checked:
            self._enter_polygon_edit()
        else:
            self._exit_polygon_edit()

    def _enter_polygon_edit(self) -> None:
        if self._current_image_path is None or self._current_image_rgb is None \
                or not self._current_tissue_key:
            self._btn_edit_polygon.setChecked(False)
            return
        ann = self._session.get(self._current_image_path)
        if ann is None:
            self._btn_edit_polygon.setChecked(False)
            return

        # Prefer one of the user-stored polygon instances; else derive
        # from the current mask (SAM workflow).
        polys = ann.polygons.get(self._current_tissue_key, [])
        if polys:
            # Edit the instance the picker is currently pointing at
            # (defaults to most-recent).
            self._edit_instance_back_idx = max(
                0, min(len(polys) - 1, self._edit_instance_back_idx),
            )
            poly = polys[-1 - self._edit_instance_back_idx]
        else:
            mask = ann.masks.get(self._current_tissue_key)
            if mask is None or not mask.any():
                self._status.showMessage(
                    "No mask to edit — run SAM / detector or draw a polygon first.", 4500,
                )
                self._btn_edit_polygon.setChecked(False)
                return
            poly = mask_to_polygon(mask, epsilon=2.0)
            if len(poly) < 3:
                self._status.showMessage("Mask too small for polygon edit.", 4500)
                self._btn_edit_polygon.setChecked(False)
                return
            self._session.set_polygon(self._current_image_path, self._current_tissue_key, poly)
            self._edit_instance_back_idx = 0

        # Display the edit-time mask so the active polygon's old footprint
        # vanishes while its outline/fill follow the vertices.
        self._apply_canvas_mask_for_edit(self._current_tissue_key)
        self._canvas.enter_polygon_edit(poly)
        self._refresh_instance_label()
        self._status.showMessage(
            "Polygon edit: drag vertices, click an edge to insert, Alt+click to delete.", 6000,
        )

    def _exit_polygon_edit(self) -> None:
        self._canvas.exit_polygon_edit()
        # Restore the full mask now that no polygon is being edited.
        if self._current_image_path is not None and self._current_tissue_key:
            self._apply_canvas_mask_full(self._current_tissue_key)
        # Restore tissue-radio-driven mode (Box / Point)
        self._on_mode_changed()

    def _exit_polygon_edit_if_active(self) -> None:
        if self._canvas.get_mode() == MODE_EDIT_POLYGON or self._btn_edit_polygon.isChecked():
            self._btn_edit_polygon.blockSignals(True)
            self._btn_edit_polygon.setChecked(False)
            self._btn_edit_polygon.blockSignals(False)
            self._exit_polygon_edit()

    # ─── Auto-detector flow ────────────────────────────────────────────

    def _on_run_detector(self) -> None:
        if self._current_image_rgb is None or self._current_image_path is None:
            self._status.showMessage("Open an image first.", 3500)
            return
        if not self._current_tissue_key:
            self._status.showMessage(
                "Add a class first — detected boxes need an owner.", 4500,
            )
            return
        # Lazy-load the detector. Raises if the weights file is missing.
        if self._detector is None:
            self._detector = BoxDetector()
            if not self._detector.weights_exists:
                QMessageBox.critical(
                    self, "Detector weights missing",
                    f"Expected weights file not found:\n{self._detector.weights_path}\n\n"
                    "Place the YOLO model there and try again.",
                )
                return
        # Prevent re-entrant runs. The thread may already have been
        # scheduled for deletion (Qt's deleteLater) after a previous run —
        # touching it raises RuntimeError. Treat that as "not running".
        if self._detector_thread is not None:
            try:
                still_running = self._detector_thread.isRunning()
            except RuntimeError:
                still_running = False
                self._detector_thread = None
                self._detector_worker = None
            if still_running:
                self._status.showMessage("Detector is already running…", 2000)
                return
        self._btn_run_detector.setEnabled(False)
        self._btn_run_detector.setText("⚙  Running…")
        self._lbl_busy.setText("◐ detecting…")
        log.info("Running box detector on %s", self._current_image_path.name)
        self._detector_thread, self._detector_worker = run_detector_async(
            self._detector, self._current_image_rgb,
            on_finished=self._on_detector_finished,
            on_failed=self._on_detector_failed,
            parent=self,
        )

    def _on_detector_finished(self, detections: list) -> None:
        # Drop refs to the worker thread NOW so a quick re-run sees a
        # clean slot (deleteLater is queued and would race us otherwise).
        self._detector_thread = None
        self._detector_worker = None
        self._lbl_busy.setText("")
        self._btn_run_detector.setEnabled(True)
        self._btn_run_detector.setText("⚙  Run Box Detector")
        if not detections:
            self._status.showMessage("Detector found no boxes.", 4000)
            log.info("Detector returned 0 boxes")
            return
        if self._current_image_path is None or self._current_image_rgb is None:
            return
        tissue = self._current_tissue_key
        if not tissue:
            return
        for det in detections:
            self._session.append_polygon(
                self._current_image_path, tissue, det.as_polygon(),
            )
        # Newest detection is the active editing target.
        self._edit_instance_back_idx = 0
        # Persist the full union mask (YOLO save uses this).
        full_mask = self._rebuild_tissue_mask(tissue)
        self._session.set_mask(self._current_image_path, tissue, full_mask)

        # Display the EDIT-time mask (excludes the active polygon) so
        # the latest box doesn't double up with its own polygon-edit fill.
        self._apply_canvas_mask_for_edit(tissue)

        # Auto-enter polygon edit on the latest detection so corner
        # vertices show up immediately — the user can grab them right
        # away to resize/move. Earlier instances are reachable with the
        # Prev/Next picker.
        latest_poly = detections[-1].as_polygon()
        self._btn_edit_polygon.blockSignals(True)
        self._btn_edit_polygon.setChecked(True)
        self._btn_edit_polygon.blockSignals(False)
        self._canvas.enter_polygon_edit(latest_poly)
        self._refresh_instance_label()

        log.info("Detector added %d box(es) to class '%s'", len(detections), tissue)
        tissue_label = TISSUE_PRESETS[tissue].label if tissue in TISSUE_PRESETS else tissue
        self._status.showMessage(
            f"Added {len(detections)} box(es) to class '{tissue_label}'. "
            "Drag corner handles to resize; use Prev/Next to edit other boxes.",
            7000,
        )
        self._refresh_status()

    def _on_detector_failed(self, msg: str) -> None:
        self._detector_thread = None
        self._detector_worker = None
        self._lbl_busy.setText("")
        self._btn_run_detector.setEnabled(True)
        self._btn_run_detector.setText("⚙  Run Box Detector")
        QMessageBox.warning(self, "Detector failed", msg)
        log.error("Detector failed: %s", msg)

    def _start_polygon_draw(self) -> None:
        if self._current_image_path is None or self._current_image_rgb is None:
            self._status.showMessage("Open an image first.", 3000)
            return
        if self._manual_mode and not self._current_tissue_key:
            self._status.showMessage(
                "Add a class first (＋ Add Custom Class) before drawing.", 4000,
            )
            return
        # Exit polygon edit (if any) so the canvas isn't carrying old vertices.
        self._exit_polygon_edit_if_active()
        # Reflect the action in the Drawing-Tool radio group so the side
        # panel doesn't lie about which tool is active. ``buttonClicked``
        # only fires on user clicks, so ``setChecked`` here is silent.
        if self._manual_mode and hasattr(self, "_rb_mode_point"):
            self._rb_mode_point.setChecked(True)
        self._canvas.start_polygon_draw()
        self._status.showMessage(
            "Drawing: click vertices, right-click pops last, double-click / Enter closes.",
            6000,
        )

    def _on_polygon_draw_completed(self, tissue_key: str) -> None:
        # Mirror the polygon edit toggle state — we are now in edit mode.
        self._btn_edit_polygon.blockSignals(True)
        self._btn_edit_polygon.setChecked(True)
        self._btn_edit_polygon.blockSignals(False)

    def _undo_last_label(self) -> None:
        """Ctrl+Z — drop the most recently added polygon instance for the
        active tissue and rebuild the mask."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        if not polys:
            self._status.showMessage("Nothing to undo.", 2500)
            return
        last_idx = len(polys) - 1
        self._exit_polygon_edit_if_active()
        self._session.remove_polygon_at(
            self._current_image_path, self._current_tissue_key, last_idx,
        )
        mask = self._rebuild_tissue_mask(self._current_tissue_key)
        self._session.set_mask(self._current_image_path, self._current_tissue_key, mask)
        self._canvas.set_mask(self._current_tissue_key, mask)
        log.info("Undo: removed last polygon for tissue '%s'", self._current_tissue_key)
        self._status.showMessage("Undone last polygon.", 2500)
        self._edit_instance_back_idx = 0
        self._refresh_instance_label()
        self._refresh_status()

    def _commit_current_polygon(self) -> None:
        """Lock the current polygon edit. Equivalent to toggling Edit off."""
        if self._btn_edit_polygon.isChecked():
            self._btn_edit_polygon.setChecked(False)
            self._status.showMessage("Polygon committed as label.", 3000)
            log.info("Polygon committed (tissue=%s, image=%s)",
                     self._current_tissue_key,
                     self._current_image_path)
        else:
            self._status.showMessage("Nothing to commit (no active polygon edit).", 3000)

    def _request_back_to_menu(self) -> None:
        """User asked to return to the startup screen."""
        n = self._session.annotated_image_count()
        if n > 0:
            ans = QMessageBox.question(
                self,
                "Unsaved annotations",
                f"{n} image(s) have unsaved annotations. Save before returning?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if ans == QMessageBox.Cancel:
                return
            if ans == QMessageBox.Save:
                self._save_all()
        # Make sure the closeEvent below doesn't show the prompt a second time.
        self._exit_prompt_skip = True
        log.info("User requested back-to-menu (manual=%s)", self._manual_mode)
        self.backToMenuRequested.emit()
        self.close()

    def _on_polygon_mutated(self, tissue_key: str) -> None:
        """Polygon vertices changed → replace the polygon currently being
        edited (selected via the Prev/Next instance picker, defaults to
        the most-recent) and re-rasterize.

        While editing, the canvas shows the union of every OTHER polygon
        plus the active polygon's own fill — keeping the active polygon's
        old position from "ghosting" through the mask after each drag.
        The committed truth (full union) is rebuilt on edit exit.
        """
        if self._current_image_path is None or self._current_image_rgb is None:
            return
        polygon = self._canvas.get_polygon()
        if len(polygon) < 3:
            return
        polys = self._session.get_polygons(self._current_image_path, tissue_key)
        target_idx: Optional[int] = None
        if polys:
            target_idx = max(0, min(len(polys) - 1, len(polys) - 1 - self._edit_instance_back_idx))
            self._session.replace_polygon_at(
                self._current_image_path, tissue_key, target_idx, polygon,
            )
        else:
            # First polygon for this tissue (SAM mask → polygon convert).
            self._session.replace_last_polygon(
                self._current_image_path, tissue_key, polygon,
            )

        # Persist the COMMITTED mask (full union) so YOLO save & class
        # competition stay correct.
        committed_mask = self._rebuild_tissue_mask(tissue_key)
        if committed_mask is None:
            committed_mask = polygon_to_mask(polygon, self._current_image_rgb.shape[:2])
        self._session.set_mask(self._current_image_path, tissue_key, committed_mask)

        # ... but DISPLAY the edit-time mask (excludes the active polygon)
        # so its old footprint vanishes the instant the user lets go of
        # a vertex.
        if target_idx is not None:
            display_mask = self._rebuild_tissue_mask(tissue_key, exclude_idx=target_idx)
            self._canvas.set_mask(tissue_key, display_mask)
        else:
            self._canvas.set_mask(tissue_key, committed_mask)
        self._refresh_status()

    def _step_instance(self, delta: int) -> None:
        """Move the polygon-edit cursor to the previous / next instance
        for the active tissue."""
        if self._current_image_path is None or not self._current_tissue_key:
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        n = len(polys)
        if n <= 1:
            return
        new_idx = max(0, min(n - 1, self._edit_instance_back_idx + delta))
        if new_idx == self._edit_instance_back_idx:
            return
        self._edit_instance_back_idx = new_idx
        target = polys[-1 - new_idx]
        # Recompute the displayed mask so the newly-active instance is
        # excluded (and the previously-active one is included again).
        self._apply_canvas_mask_for_edit(self._current_tissue_key)
        self._canvas.enter_polygon_edit(target)
        self._btn_edit_polygon.blockSignals(True)
        self._btn_edit_polygon.setChecked(True)
        self._btn_edit_polygon.blockSignals(False)
        self._refresh_instance_label()

    def _refresh_instance_label(self) -> None:
        if not hasattr(self, "_lbl_instance_pos"):
            return
        if self._current_image_path is None or not self._current_tissue_key:
            self._lbl_instance_pos.setText("—")
            self._btn_prev_instance.setEnabled(False)
            self._btn_next_instance.setEnabled(False)
            return
        polys = self._session.get_polygons(self._current_image_path, self._current_tissue_key)
        n = len(polys)
        if n == 0:
            self._lbl_instance_pos.setText("—")
            self._btn_prev_instance.setEnabled(False)
            self._btn_next_instance.setEnabled(False)
            return
        pos = n - self._edit_instance_back_idx
        self._lbl_instance_pos.setText(f"{pos} / {n}")
        self._btn_prev_instance.setEnabled(self._edit_instance_back_idx < n - 1)
        self._btn_next_instance.setEnabled(self._edit_instance_back_idx > 0)

    # ─── Edit actions ──────────────────────────────────────────────────

    def _undo(self, kind: str) -> None:
        self._canvas.undo_last_point(kind)

    def _clear_active_tissue(self) -> None:
        if self._current_image_path is None or not self._current_tissue_key:
            return
        self._exit_polygon_edit_if_active()
        self._session.clear_tissue(self._current_image_path, self._current_tissue_key)
        self._canvas.set_mask(self._current_tissue_key, None)
        self._canvas.clear_active_prompts()
        if self._manual_mode:
            self._canvas.start_polygon_draw()
        self._edit_instance_back_idx = 0
        self._refresh_instance_label()
        log.info("Cleared tissue '%s' on %s",
                 self._current_tissue_key, self._current_image_path)
        self._refresh_status()

    def _clear_image(self) -> None:
        if self._current_image_path is None:
            return
        self._exit_polygon_edit_if_active()
        ann = self._session.get(self._current_image_path)
        if ann is not None:
            ann.masks.clear()
            ann.prompts.clear()
            ann.polygons.clear()
        self._canvas.clear_all_visuals()
        self._edit_instance_back_idx = 0
        self._refresh_instance_label()
        if self._manual_mode and self._current_tissue_key:
            self._canvas.start_polygon_draw()
        log.info("Cleared ALL tissues on %s", self._current_image_path)
        self._refresh_status()

    def _save_all(self) -> None:
        if not self._session.all_paths():
            self._status.showMessage("Nothing to save.", 4000)
            return
        # Persist the in-flight canvas state first so it isn't lost.
        self._persist_current_prompts()
        report = self._session.save_all()
        n_ok = len(report.written)
        n_skip = len(report.skipped_empty)
        n_err = len(report.errors)
        msg = f"Saved {n_ok} YOLO file(s); {n_skip} empty skipped"
        if n_err:
            msg += f"; {n_err} error(s)"
        self._status.showMessage(msg, 6000)
        if n_err:
            details = "\n".join(f"{p.name}: {e}" for p, e in report.errors[:8])
            QMessageBox.warning(self, "Save errors", details)

    def _calculate_geometry(self) -> None:
        """Compute & draw the PNL (MLO) or CC-depth (CC) measurement from the
        currently-labelled tissues on this image."""
        path = self._current_image_path
        if path is None:
            self._status.showMessage("Önce bir görüntü açın.", 4000)
            return
        # Flush the in-flight canvas edit so the freshest masks are analysed.
        self._persist_current_prompts()
        ann = self._session.get(path)
        if ann is None or not ann.has_any_mask():
            self._canvas.clear_analysis_overlay()
            self._status.showMessage(
                "Hesaplama için en az Nipple + (Pectoral veya Breast) etiketleyin.", 6000
            )
            return

        result = analyze_annotation(ann)
        if result.ok:
            self._canvas.show_analysis_overlay(result)
            if result.pnl is not None:
                self._status.showMessage(
                    f"MLO • PNL = {result.pnl.distance_px:.1f} px", 8000
                )
            elif result.cc_depth is not None:
                self._status.showMessage(
                    f"CC • Derinlik = {result.cc_depth.distance_px:.1f} px", 8000
                )
        else:
            self._canvas.clear_analysis_overlay()
            msg = " ".join(result.messages) or "Ölçüm hesaplanamadı."
            self._status.showMessage(msg, 6000)

    # ─── Helpers ───────────────────────────────────────────────────────

    @property
    def _current_image_path(self) -> Optional[Path]:
        if 0 <= self._current_idx < len(self._image_paths):
            return self._image_paths[self._current_idx]
        return None

    def _persist_current_prompts(self) -> None:
        if self._current_image_path is None or not self._current_tissue_key:
            return
        state = self._canvas.get_prompt_state()
        self._session.set_prompts(self._current_image_path, self._current_tissue_key, state)

    def _on_zoom_changed(self, scale: float) -> None:
        self._lbl_zoom.setText(f"Zoom: {int(round(scale * 100))}%")

    def _on_window_level_changed(self, window: float, level: float) -> None:
        self._lbl_wl.setText(f"W/L: {window:+.0f}/{level:+.0f}")

    def _refresh_status(self) -> None:
        if self._current_image_path is None:
            self._lbl_image_idx.setText("Image: —")
            if hasattr(self, "_hdr_subtitle"):
                self._hdr_subtitle.setText("No image loaded")
        else:
            self._lbl_image_idx.setText(
                f"Image: {self._current_idx + 1}/{len(self._image_paths)}  "
                f"({self._current_image_path.name})"
            )
            if hasattr(self, "_hdr_subtitle"):
                if self._current_tissue_key and self._current_tissue_key in TISSUE_PRESETS:
                    tissue_label = TISSUE_PRESETS[self._current_tissue_key].label
                    self._hdr_subtitle.setText(f"Active: {tissue_label}")
                else:
                    self._hdr_subtitle.setText("No class — add one to start")
        self._lbl_dirty.setText(f"Pending: {self._session.annotated_image_count()} images")

    def eventFilter(self, obj, event):
        # Left/Right = previous/next image, no matter which child widget has
        # focus. Installed on the QApplication so it runs before the canvas
        # (a StrongFocus QGraphicsView) can eat the arrows for its own
        # scrolling. Skipped while a text field is focused so arrow keys still
        # move the cursor when typing (e.g. a class-name dialog).
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Left, Qt.Key_Right):
            fw = QApplication.focusWidget()
            if isinstance(fw, (QLineEdit, QComboBox, QAbstractSpinBox)):
                return False
            if event.key() == Qt.Key_Left:
                self._goto_prev()
            else:
                self._goto_next()
            return True  # consume: canvas must not also scroll
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:
        # If we already asked (e.g. via Back-to-Menu), skip the second prompt.
        already_asked = getattr(self, "_exit_prompt_skip", False)
        n = self._session.annotated_image_count()
        if n > 0 and not already_asked:
            ans = QMessageBox.question(
                self,
                "Unsaved annotations",
                f"{n} image(s) have unsaved annotations. Save before exit?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if ans == QMessageBox.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.Save:
                self._save_all()
        if self._worker is not None:
            self._worker.stop()
        # Custom classes are this-window-scoped — drop them so the next
        # SAM/Manual window (or this one re-opened) starts clean.
        for k in list(self._custom_class_keys):
            unregister_custom_tissue(k)
        self._custom_class_keys.clear()
        # Restore the built-in class registry that Manual mode wiped out.
        if self._manual_mode and self._builtin_snapshot:
            for k, v in self._builtin_snapshot.items():
                TISSUE_PRESETS[k] = v
            for k, v in self._builtin_rgb_snapshot.items():
                TISSUE_RGB[k] = v
            self._builtin_snapshot.clear()
            self._builtin_rgb_snapshot.clear()
        super().closeEvent(event)
        # The app explicitly manages its lifetime (quitOnLastWindowClosed
        # is disabled). When the user closes via X (NOT via Back-to-Menu),
        # there's no startup dialog to keep us alive — quit.
        if event.isAccepted() and not getattr(self, "_exit_prompt_skip", False):
            from PyQt5.QtWidgets import QApplication
            QApplication.instance().quit()


# ─── helpers ─────────────────────────────────────────────────────────────


