"""
Main application window -  Medical Imaging UI
Layout: Compact top bar + Large visualization area
Features: Zoom/Pan, Draggable landmarks, Live recalculation
"""

import sys
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMessageBox, QApplication, QStatusBar,
    QGroupBox, QPushButton, QLabel, QTextEdit, QFileDialog,
    QSizePolicy, QFrame, QSlider
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QTextCursor

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from gui.interactive_canvas import InteractiveCanvas
from analysis.analysis_controller import AnalysisController
from analysis.mlo_analyzer import MLOAnalyzer
from analysis.cc_analyzer import CCAnalyzer
from models.model_manager import ModelManager
from data.data_manager import DataManager
from utils.paths import app_logo_path, gui_bundle_root
from utils.qt_icons import q_icon_from_png
from version import DISPLAY_NAME, __version__

# Local stylesheet retained as a safety net when this module is run
# standalone (i.e. NOT inside the Mammography Toolkit). Inside the
# toolkit, the parent app's QSS already covers everything and overrides
# this on a per-window basis.
STYLESHEET = ""

# Match matplotlib figure colors to the toolkit palette.
matplotlib.rcParams.update({
    'figure.facecolor': '#1E2330',  # SURFACE
    'axes.facecolor': '#121620',    # CANVAS_BG
    'text.color': '#FFFFFF',
    'axes.labelcolor': '#FFFFFF',
    'xtick.color': '#A0AEC0',       # TEXT_SECONDARY
    'ytick.color': '#A0AEC0',
    'axes.edgecolor': '#2A3242',    # DIVIDER
})


class MainApplicationWindow(QMainWindow):
    """Professional Medical Imaging Application Window."""

    def __init__(self):
        super().__init__()
        self._mlo_results = None
        self._cc_results = None
        self._setup_managers()
        self._setup_window()
        self._setup_ui()
        self._initialize_system()

    def _setup_managers(self):
        self.data_manager = DataManager()
        self.model_manager = ModelManager()
        self.analysis_controller = AnalysisController(
            self.data_manager, self.model_manager
        )
        self.analysis_controller.on_mlo_analysis_complete = self._on_mlo_complete
        self.analysis_controller.on_cc_analysis_complete = self._on_cc_complete
        self.analysis_controller.on_comparison_complete = self._on_comparison_complete

    def _setup_window(self):
        # Title is set by the toolkit; fall back to the legacy name when
        # this module is run standalone.
        self.setWindowTitle("Mammography Toolkit — Breast Positioning")
        _logo = app_logo_path()
        if _logo is not None:
            self.setWindowIcon(q_icon_from_png(_logo))
        self.setMinimumSize(1400, 900)
        # Only apply the local stylesheet when not embedded in the toolkit
        # (toolkit applies its own QSS over us).
        if STYLESHEET:
            self.setStyleSheet(STYLESHEET)
        self._set_dark_titlebar()

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _set_dark_titlebar(self):
        try:
            import ctypes
            hwnd = int(self.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value), ctypes.sizeof(value)
            )
        except Exception:
            pass

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === HEADER (gradient title bar, same as the rest of the toolkit) ===
        header = QWidget()
        header.setObjectName("appHeader")
        header.setAttribute(Qt.WA_StyledBackground, True)
        header.setFixedHeight(46)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("MAMMOGRAPHY TOOLKIT — Breast Positioning")
        title.setObjectName("appTitle")
        hl.addWidget(title)
        hl.addStretch(1)
        self._hdr_subtitle = QLabel("Loading models…")
        self._hdr_subtitle.setObjectName("appSubtitle")
        hl.addWidget(self._hdr_subtitle)
        main_layout.addWidget(header)

        # === BODY container with consistent padding ===
        body = QWidget()
        body_v = QVBoxLayout(body)
        body_v.setContentsMargins(14, 14, 14, 14)
        body_v.setSpacing(12)
        main_layout.addWidget(body, stretch=1)

        # === TOP BAR ===
        top_bar = QFrame()
        top_bar.setObjectName("posContainer")
        top_bar.setFixedHeight(72)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(12)

        def _mkbtn(text: str, cb) -> QPushButton:
            b = QPushButton(text)
            b.setProperty("class", "secondary")
            b.clicked.connect(cb)
            return b

        self.select_btn = _mkbtn("Select DICOM Pair", self._select_files)
        self.select_btn.setMinimumWidth(150)

        self.clear_btn = _mkbtn("Clear", self._clear_files)

        self.file_label = QLabel("No files selected")
        self.file_label.setObjectName("posFileLabel")

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFixedWidth(1)
        sep1.setStyleSheet("background-color: #2A3242;")

        self.mlo_btn = _mkbtn("MLO Analysis", self._analyze_mlo)
        self.cc_btn = _mkbtn("CC Analysis", self._analyze_cc)
        self.compare_btn = _mkbtn("Compare", self._compare)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFixedWidth(1)
        sep2.setStyleSheet("background-color: #2A3242;")

        self.save_btn = _mkbtn("Save Results", self._save_results)
        self.save_img_btn = _mkbtn("Save Images", self._save_images)

        self.sys_label = QLabel("Loading…")
        self.sys_label.setObjectName("posStatusLoading")

        top_layout.addWidget(self.select_btn)
        top_layout.addWidget(self.clear_btn)
        top_layout.addWidget(self.file_label, 1)
        top_layout.addWidget(sep1)
        top_layout.addWidget(self.mlo_btn)
        top_layout.addWidget(self.cc_btn)
        top_layout.addWidget(self.compare_btn)
        top_layout.addWidget(sep2)
        top_layout.addWidget(self.save_btn)
        top_layout.addWidget(self.save_img_btn)
        top_layout.addWidget(self.sys_label)

        body_v.addWidget(top_bar)

        # === SECOND BAR: folder browse + pair nav + threshold slider ===
        nav_bar = QFrame()
        nav_bar.setObjectName("posContainer")
        nav_bar.setFixedHeight(58)
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(16, 8, 16, 8)
        nav_layout.setSpacing(10)

        self.folder_btn = QPushButton("Select Folder")
        self.folder_btn.setProperty("class", "secondary")
        self.folder_btn.setMinimumWidth(130)
        self.folder_btn.setToolTip(
            "Pick any folder. The app recursively scans for .dcm / .dicom\n"
            "files and auto-pairs MLO ↔ CC by filename (mlo/cc swap).\n"
            "Use Prev/Next to walk through detected pairs."
        )
        self.folder_btn.clicked.connect(self._select_folder)

        self.prev_pair_btn = QPushButton("◀  Prev Pair")
        self.prev_pair_btn.setProperty("class", "secondary")
        self.prev_pair_btn.setEnabled(False)
        self.prev_pair_btn.clicked.connect(self._prev_pair)

        self.next_pair_btn = QPushButton("Next Pair  ▶")
        self.next_pair_btn.setProperty("class", "secondary")
        self.next_pair_btn.setEnabled(False)
        self.next_pair_btn.clicked.connect(self._next_pair)

        self.pair_info_label = QLabel("No folder loaded")
        self.pair_info_label.setObjectName("posFileLabel")

        # Vertical separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFixedWidth(1)
        sep3.setStyleSheet("background-color: #2A3242;")

        # Threshold slider: 0.0 .. 20.0 mm in 0.5 mm steps (internal *10)
        self.threshold_label = QLabel("Threshold: 10.0 mm")
        self.threshold_label.setObjectName("posFileLabel")
        self.threshold_label.setMinimumWidth(130)
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 200)        # 0.0 .. 20.0 (×10)
        self.threshold_slider.setSingleStep(5)        # 0.5 mm
        self.threshold_slider.setPageStep(10)         # 1.0 mm
        self.threshold_slider.setValue(100)           # default 10 mm
        self.threshold_slider.setMinimumWidth(200)
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)

        nav_layout.addWidget(self.folder_btn)
        nav_layout.addWidget(self.prev_pair_btn)
        nav_layout.addWidget(self.next_pair_btn)
        nav_layout.addWidget(self.pair_info_label, 1)
        nav_layout.addWidget(sep3)
        nav_layout.addWidget(self.threshold_label)
        nav_layout.addWidget(self.threshold_slider)

        body_v.addWidget(nav_bar)

        # === CONTENT AREA ===
        content = QHBoxLayout()
        content.setSpacing(12)

        # Left: Analysis log (card-style)
        log_widget = QFrame()
        log_widget.setObjectName("posContainer")
        log_widget.setFixedWidth(340)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(14, 12, 14, 14)
        log_layout.setSpacing(8)

        log_header = QLabel("ANALYSIS LOG")
        log_header.setObjectName("posSectionHeader")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Cascadia Code', 10))

        zoom_hint = QLabel(
            "Scroll: Zoom  \u2022  Right-click: Reset  \u2022  Drag landmarks to adjust"
        )
        zoom_hint.setObjectName("posHint")
        zoom_hint.setWordWrap(True)

        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.setProperty("class", "secondary")
        clear_log_btn.clicked.connect(lambda: self.log_text.clear())

        log_layout.addWidget(log_header)
        log_layout.addWidget(self.log_text, stretch=1)
        log_layout.addWidget(zoom_hint)
        log_layout.addWidget(clear_log_btn)

        # Right: Visualization with InteractiveCanvas
        viz_widget = QWidget()
        viz_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        viz_layout = QHBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(12)

        # MLO Canvas card
        mlo_container = QFrame()
        mlo_container.setObjectName("posContainer")
        mlo_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mlo_lay = QVBoxLayout(mlo_container)
        mlo_lay.setContentsMargins(10, 10, 10, 10)

        mlo_title = QLabel("MLO VIEW")
        mlo_title.setObjectName("posSectionHeader")
        mlo_title.setAlignment(Qt.AlignCenter)

        self.mlo_canvas = InteractiveCanvas(parent=mlo_container)
        self.mlo_canvas.on_landmarks_moved = self._on_mlo_landmarks_moved

        mlo_lay.addWidget(mlo_title)
        mlo_lay.addWidget(self.mlo_canvas, stretch=1)

        # CC Canvas card
        cc_container = QFrame()
        cc_container.setObjectName("posContainer")
        cc_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cc_lay = QVBoxLayout(cc_container)
        cc_lay.setContentsMargins(10, 10, 10, 10)

        cc_title = QLabel("CC VIEW")
        cc_title.setObjectName("posSectionHeader")
        cc_title.setAlignment(Qt.AlignCenter)

        self.cc_canvas = InteractiveCanvas(parent=cc_container)
        self.cc_canvas.on_landmarks_moved = self._on_cc_landmarks_moved

        cc_lay.addWidget(cc_title)
        cc_lay.addWidget(self.cc_canvas, stretch=1)

        # Swap button (centered between the two canvases)
        swap_container = QWidget()
        swap_container.setFixedWidth(56)
        swap_layout = QVBoxLayout(swap_container)
        swap_layout.setContentsMargins(0, 0, 0, 0)
        swap_layout.addStretch()

        self.swap_btn = QPushButton("\u2194")
        self.swap_btn.setFixedSize(48, 48)
        self.swap_btn.setToolTip("Swap MLO and CC views")
        self.swap_btn.setProperty("class", "secondary")
        self.swap_btn.clicked.connect(self._swap_views)

        swap_layout.addWidget(self.swap_btn)
        swap_layout.addStretch()

        viz_layout.addWidget(mlo_container)
        viz_layout.addWidget(swap_container)
        viz_layout.addWidget(cc_container)

        content.addWidget(log_widget)
        content.addWidget(viz_widget, 1)

        body_v.addLayout(content, 1)

        # Initialize empty plots
        self.mlo_canvas.show_empty("Select MLO DICOM")
        self.cc_canvas.show_empty("Select CC DICOM")

        # Pair-navigation state (populated by "Select Folder")
        self._folder_pairs: list[tuple[str, str]] = []
        self._current_pair_idx: int = -1
        self._folder_unpaired: list[str] = []

        self._log("System initialized. Select DICOM pair to begin.")

    # ── Folder navigation ──

    def _select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder of DICOM files")
        if not folder:
            return
        pairs, unpaired = self._detect_dicom_pairs(folder)
        self._folder_pairs = pairs
        self._folder_unpaired = unpaired
        self._current_pair_idx = -1
        # No popups — just surface the result in the panel. Whatever was
        # found, the user decides next.
        if pairs:
            msg = f"{len(pairs)} pair(s) detected"
            if unpaired:
                msg += f"  •  {len(unpaired)} unpaired"
            self.pair_info_label.setText(msg)
            self._goto_pair(0)
        elif unpaired:
            self.pair_info_label.setText(
                f"{len(unpaired)} DICOM(s) — no MLO/CC pair auto-detected"
            )
        else:
            self.pair_info_label.setText("No DICOM files found in this folder")
        self._update_pair_nav_enabled()

    def _detect_dicom_pairs(self, folder: str) -> tuple[list[tuple[str, str]], list[str]]:
        """Return ([(mlo_path, cc_path), ...], unpaired_paths).

        Heuristic: scan recursively, group files whose names share a stem
        once 'mlo'/'cc' substrings are swapped. Case-insensitive.
        """
        from pathlib import Path as _P
        root = _P(folder)
        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in (".dcm", ".dicom")
        )
        by_name = {p.stem.lower(): p for p in files}
        used: set[_P] = set()
        pairs: list[tuple[str, str]] = []
        for p in files:
            if p in used:
                continue
            stem = p.stem.lower()
            mate_stem = None
            if "mlo" in stem:
                mate_stem = stem.replace("mlo", "cc", 1)
            elif "cc" in stem:
                mate_stem = stem.replace("cc", "mlo", 1)
            if mate_stem and mate_stem in by_name and by_name[mate_stem] not in used:
                mate = by_name[mate_stem]
                mlo, cc = (p, mate) if "mlo" in stem else (mate, p)
                pairs.append((str(mlo), str(cc)))
                used.add(p); used.add(mate)
        unpaired = [str(p) for p in files if p not in used]
        return pairs, unpaired

    def _goto_pair(self, idx: int) -> None:
        if not self._folder_pairs:
            return
        idx = max(0, min(idx, len(self._folder_pairs) - 1))
        mlo, cc = self._folder_pairs[idx]
        self._current_pair_idx = idx
        try:
            self._load_pair_paths(mlo, cc)
            self.pair_info_label.setText(
                f"Pair {idx + 1}/{len(self._folder_pairs)}  •  "
                f"{os.path.basename(mlo)}  +  {os.path.basename(cc)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))
        self._update_pair_nav_enabled()

    def _prev_pair(self) -> None:
        if self._current_pair_idx > 0:
            self._goto_pair(self._current_pair_idx - 1)

    def _next_pair(self) -> None:
        if 0 <= self._current_pair_idx < len(self._folder_pairs) - 1:
            self._goto_pair(self._current_pair_idx + 1)

    def _update_pair_nav_enabled(self) -> None:
        n = len(self._folder_pairs)
        self.prev_pair_btn.setEnabled(self._current_pair_idx > 0)
        self.next_pair_btn.setEnabled(0 <= self._current_pair_idx < n - 1)

    def _load_pair_paths(self, mlo: str, cc: str) -> None:
        """Shared helper: load an MLO+CC pair (used by both file dialog and
        folder navigation)."""
        self.analysis_controller.clear_results()
        self._mlo_results = None
        self._cc_results = None
        self.data_manager.load_image_pair(mlo, cc)

        mlo_name = os.path.basename(mlo)
        cc_name = os.path.basename(cc)
        self.file_label.setText(f"MLO: {mlo_name}  •  CC: {cc_name}")
        self.file_label.setObjectName("posFileLabelLoaded")
        self.file_label.style().unpolish(self.file_label)
        self.file_label.style().polish(self.file_label)

        self._log(f"\n{'='*40}")
        self._log(f"Loaded: {mlo_name}, {cc_name}")
        self.mlo_canvas.display_image(self.data_manager.current_mlo_image, "MLO")
        self.cc_canvas.display_image(self.data_manager.current_cc_image, "CC")
        self._log("Ready for analysis")
        self.status_bar.showMessage(f"Loaded: {mlo_name}, {cc_name}")

    # ── Threshold slider ──

    def _on_threshold_changed(self, value: int) -> None:
        mm = value / 10.0
        self.threshold_label.setText(f"Threshold: {mm:.1f} mm")
        # Quiet update — only the next Compare/Save call uses the new
        # value. No live re-compare, no toast/popup.
        try:
            self.analysis_controller.comparison_engine.set_threshold(max(0.01, mm))
        except Exception:
            pass

    def _log(self, msg):
        # Subtle dim color for ASCII-separator divider lines, bright for
        # everything else. Both readable on the toolkit's SURFACE bg.
        color = "#FFFFFF"
        if "=" in msg and len(msg) > 20:
            color = "#6B7785"
        self.log_text.append(f'<span style="color:{color}">{msg}</span>')
        self.log_text.moveCursor(QTextCursor.End)

    def _initialize_system(self):
        try:
            self.model_manager.load_models()
            self.data_manager.load_pixel_spacing_data()

            self.sys_label.setText(f"Ready • Device: {self.model_manager.device}")
            self.sys_label.setObjectName("posStatusGood")
            self.sys_label.style().unpolish(self.sys_label)
            self.sys_label.style().polish(self.sys_label)
            if hasattr(self, "_hdr_subtitle"):
                self._hdr_subtitle.setText(f"Ready • {self.model_manager.device}")
            self._log("Models loaded successfully.")
            self.status_bar.showMessage("System ready")
        except Exception as e:
            self.sys_label.setText("Error")
            self.sys_label.setObjectName("posStatusError")
            self.sys_label.style().unpolish(self.sys_label)
            self.sys_label.style().polish(self.sys_label)
            self._log(f"Init error: {e}")
            QMessageBox.critical(self, "Error", str(e))

    # ── File Operations ──

    def _select_files(self):
        filt = "DICOM (*.dicom *.dcm);;All (*.*)"
        mlo, _ = QFileDialog.getOpenFileName(self, "Select MLO DICOM", "", filt)
        if not mlo:
            return
        cc, _ = QFileDialog.getOpenFileName(self, "Select CC DICOM", "", filt)
        if not cc:
            return
        try:
            self._load_pair_paths(mlo, cc)
            # Manual selection breaks folder-navigation context.
            self._folder_pairs = []
            self._folder_unpaired = []
            self._current_pair_idx = -1
            self._update_pair_nav_enabled()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _clear_files(self):
        self.data_manager.clear_images()
        self.analysis_controller.clear_results()
        self._mlo_results = None
        self._cc_results = None
        self.mlo_canvas.show_empty("Select MLO DICOM")
        self.cc_canvas.show_empty("Select CC DICOM")
        self.file_label.setText("No files selected")
        self.file_label.setObjectName("posFileLabel")
        self.file_label.style().unpolish(self.file_label)
        self.file_label.style().polish(self.file_label)
        self._log("Cleared")

    def _swap_views(self):
        if self.data_manager.current_mlo_image is None or self.data_manager.current_cc_image is None:
            return

        self.data_manager.current_mlo_image, self.data_manager.current_cc_image = \
            self.data_manager.current_cc_image, self.data_manager.current_mlo_image
        self.data_manager.current_mlo_filename, self.data_manager.current_cc_filename = \
            self.data_manager.current_cc_filename, self.data_manager.current_mlo_filename
        self.data_manager.current_mlo_original_pixel_spacing, self.data_manager.current_cc_original_pixel_spacing = \
            self.data_manager.current_cc_original_pixel_spacing, self.data_manager.current_mlo_original_pixel_spacing
        self.data_manager.current_mlo_original_shape, self.data_manager.current_cc_original_shape = \
            self.data_manager.current_cc_original_shape, self.data_manager.current_mlo_original_shape
        self.data_manager.current_mlo_transformation_info, self.data_manager.current_cc_transformation_info = \
            self.data_manager.current_cc_transformation_info, self.data_manager.current_mlo_transformation_info

        self.analysis_controller.clear_results()
        self._mlo_results = None
        self._cc_results = None

        self.mlo_canvas.display_image(self.data_manager.current_mlo_image, "MLO")
        self.cc_canvas.display_image(self.data_manager.current_cc_image, "CC")

        self._log("Views swapped")
        self.status_bar.showMessage("Views swapped")

    # ── Analysis ──

    def _analyze_mlo(self):
        if self.data_manager.current_mlo_image is None:
            QMessageBox.warning(self, "Warning", "Load MLO image first!")
            return
        self._log("\nMLO Analysis...")
        self.status_bar.showMessage("Analyzing MLO...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_mlo()
        if results:
            self._mlo_results = results
            self._log_mlo_results(results)

    def _analyze_cc(self):
        if self.data_manager.current_cc_image is None:
            QMessageBox.warning(self, "Warning", "Load CC image first!")
            return
        self._log("\nCC Analysis...")
        self.status_bar.showMessage("Analyzing CC...")
        QApplication.processEvents()

        results = self.analysis_controller.analyze_cc()
        if results:
            self._cc_results = results
            self._log_cc_results(results)

    def _compare(self):
        status = self.analysis_controller.has_results()
        if not status['mlo']:
            QMessageBox.warning(self, "Warning", "Run MLO analysis first!")
            return
        if not status['cc']:
            QMessageBox.warning(self, "Warning", "Run CC analysis first!")
            return

        comp = self.analysis_controller.compare_results()
        if comp:
            self._log(f"\n{'='*40}")
            self._log("COMPARISON")
            self._log(f"MLO: {comp['mlo_distance']:.2f} mm")
            self._log(f"CC: {comp['cc_distance']:.2f} mm")
            self._log(f"Diff: {comp['difference']:.2f} mm")
            self._log(f"{comp['quality_result']}")

    # ── Save ──

    def _save_results(self):
        status = self.analysis_controller.has_results()
        if not status['mlo'] and not status['cc']:
            QMessageBox.warning(self, "Warning", "No results to save!")
            return
        saved = self.analysis_controller.save_results()
        if saved:
            self._log(f"Saved: {', '.join(saved)}")
            QMessageBox.information(self, "Saved", "\n".join(saved))

    def _save_images(self):
        status = self.analysis_controller.has_results()
        if not status['mlo'] and not status['cc']:
            QMessageBox.warning(self, "Warning", "No images to save!")
            return

        import pandas as pd
        results_dir = str(gui_bundle_root() / "results")
        os.makedirs(results_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        saved = []

        if status['mlo']:
            f = os.path.join(results_dir, f"mlo_{ts}.png")
            self.mlo_canvas.fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)
        if status['cc']:
            f = os.path.join(results_dir, f"cc_{ts}.png")
            self.cc_canvas.fig.savefig(f, dpi=150, bbox_inches='tight', facecolor='#22262E')
            saved.append(f)

        self._log(f"Saved: {', '.join(saved)}")
        QMessageBox.information(self, "Saved", "\n".join(saved))

    # ── Logging ──

    def _log_mlo_results(self, r):
        self._log(f"\n{'='*40}")
        self._log("MLO RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Pec Top: [{lm[1][0]:.0f}, {lm[1][1]:.0f}]")
        self._log(f"Pec Bottom: [{lm[2][0]:.0f}, {lm[2][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm")
        self.status_bar.showMessage(f"MLO complete: {r['distance_mm']:.2f} mm")

    def _log_cc_results(self, r):
        self._log(f"\n{'='*40}")
        self._log("CC RESULTS")
        lm = r['landmarks']
        self._log(f"Nipple: [{lm[0][0]:.0f}, {lm[0][1]:.0f}]")
        self._log(f"Distance: {r['distance_mm']:.2f} mm ({r['direction']})")
        self.status_bar.showMessage(f"CC complete: {r['distance_mm']:.2f} mm")

    # ── Analysis Display Callbacks ──

    def _on_mlo_complete(self, r):
        self._display_mlo_analysis(r)

    def _on_cc_complete(self, r):
        self._display_cc_analysis(r)

    def _on_comparison_complete(self, c):
        QMessageBox.information(self, "Assessment", c['result_text'])

    # ── MLO Display + Overlay ──

    def _display_mlo_analysis(self, r):
        img = self.data_manager.current_mlo_image
        if img is None:
            return

        self.mlo_canvas.display_image(img, "MLO")

        lm = r['landmarks']
        self.mlo_canvas.set_landmarks(
            ['nipple', 'pec_top', 'pec_bottom'],
            [(lm[0][0], lm[0][1]), (lm[1][0], lm[1][1]), (lm[2][0], lm[2][1])]
        )

        self._draw_mlo_overlay(r)

    def _draw_mlo_overlay(self, r):
        self.mlo_canvas._clear_overlay()

        lm = r['landmarks']
        inter = r['intersection']
        dist = r['distance_mm']

        nip = lm[0]
        p1 = lm[1]
        p2 = lm[2]

        self.mlo_canvas.draw_line(p1[0], p1[1], p2[0], p2[1],
                                  color='#3B82F6', linewidth=2, linestyle='-')
        self.mlo_canvas.draw_line(nip[0], nip[1], inter[0], inter[1],
                                  color='#EF4444', linewidth=2, linestyle='--')

        mid_x = (nip[0] + inter[0]) / 2
        mid_y = (nip[1] + inter[1]) / 2
        self.mlo_canvas.draw_distance_label(mid_x, mid_y, f"{dist:.1f} mm")

        self.mlo_canvas.update_title(f"MLO - {dist:.2f} mm")
        self.mlo_canvas.draw_idle()

    # ── CC Display + Overlay ──

    def _display_cc_analysis(self, r):
        img = self.data_manager.current_cc_image
        if img is None:
            return

        self.cc_canvas.display_image(img, "CC")

        lm = r['landmarks']
        self.cc_canvas.set_landmarks(
            ['cc_nipple'],
            [(lm[0][0], lm[0][1])]
        )

        self._draw_cc_overlay(r)

    def _draw_cc_overlay(self, r):
        self.cc_canvas._clear_overlay()

        lm = r['landmarks']
        edge = r['edge_point']
        dist = r['distance_mm']

        nip = lm[0]

        self.cc_canvas.draw_line(nip[0], nip[1], edge[0], edge[1],
                                 color='#EF4444', linewidth=2, linestyle='--')

        mid_x = (nip[0] + edge[0]) / 2
        mid_y = (nip[1] + edge[1]) / 2
        self.cc_canvas.draw_distance_label(mid_x, mid_y, f"{dist:.1f} mm")

        self.cc_canvas.update_title(f"CC - {dist:.2f} mm")
        self.cc_canvas.draw_idle()

    # ── Landmark Drag Recalculation ──

    def _on_mlo_landmarks_moved(self):
        if self._mlo_results is None:
            return

        coords = self.mlo_canvas.get_landmark_coords()
        if 'nipple' not in coords or 'pec_top' not in coords or 'pec_bottom' not in coords:
            return

        nipple = np.array(coords['nipple'])
        pec_top = np.array(coords['pec_top'])
        pec_bottom = np.array(coords['pec_bottom'])

        perp_dist_px, intersection = MLOAnalyzer.perpendicular_distance(
            pec_top, pec_bottom, nipple
        )

        scaled_ps = self._mlo_results.get('scaled_pixel_spacing',
                                           self._mlo_results.get('pixel_spacing', 0.085))
        dist_mm = perp_dist_px * scaled_ps

        updated = dict(self._mlo_results)
        updated['landmarks'] = np.array([nipple, pec_top, pec_bottom])
        updated['intersection'] = intersection
        updated['distance_pixels'] = perp_dist_px
        updated['distance_mm'] = dist_mm
        self._mlo_results = updated

        self.analysis_controller.mlo_results = updated

        self._draw_mlo_overlay(updated)

        self._log(f"MLO adjusted: {dist_mm:.2f} mm")
        self.status_bar.showMessage(f"MLO adjusted: {dist_mm:.2f} mm")

    def _on_cc_landmarks_moved(self):
        if self._cc_results is None:
            return

        coords = self.cc_canvas.get_landmark_coords()
        if 'cc_nipple' not in coords:
            return

        nipple = np.array(coords['cc_nipple'])
        breast_side = self._cc_results.get('breast_side', 'LEFT')

        direction, dist_px, edge_point = CCAnalyzer.edge_distance(
            nipple, 640, breast_side
        )

        scaled_ps = self._cc_results.get('scaled_pixel_spacing',
                                          self._cc_results.get('pixel_spacing', 0.085))
        dist_mm = dist_px * scaled_ps

        updated = dict(self._cc_results)
        updated['landmarks'] = np.array([nipple])
        updated['edge_point'] = edge_point
        updated['direction'] = direction
        updated['distance_pixels'] = dist_px
        updated['distance_mm'] = dist_mm
        self._cc_results = updated

        self.analysis_controller.cc_results = updated

        self._draw_cc_overlay(updated)

        self._log(f"CC adjusted: {dist_mm:.2f} mm ({direction})")
        self.status_bar.showMessage(f"CC adjusted: {dist_mm:.2f} mm")

    # ── Cleanup ──

    def closeEvent(self, event):
        plt.close('all')
        event.accept()
