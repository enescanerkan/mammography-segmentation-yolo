"""Medical dark theme — inspired by Figma Community Health App palettes.

Three-tier palette:
    Base / Surface  (60%) — backgrounds, panels, cards
    Text / Icons    (30%) — primary white + muted slate
    Accents         (10%) — brand blue, success green, alert coral

Single source of truth: change a constant here, the whole UI follows.
"""

from __future__ import annotations


# ─── Base & surface (60%) ──────────────────────────────────────────────
CANVAS_BG = "#121620"   # deep navy-charcoal (image viewport)
APP_BG = "#141A24"      # main window background (slightly lighter than canvas)
SURFACE = "#1E2330"     # cards, panels, group boxes
SURFACE_ALT = "#1F2A33" # hovered/elevated surface
SURFACE_HI = "#2A3242"  # focused / selected row
DIVIDER = "#2A3242"     # borders, separators

# ─── Typography (30%) ──────────────────────────────────────────────────
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0AEC0"
TEXT_MUTED = "#6B7785"

# ─── Accents (10%) ─────────────────────────────────────────────────────
ACCENT = "#0474ED"        # primary CTA / active state
ACCENT_HOVER = "#3A7BD5"
ACCENT_PRESSED = "#0353A8"

SUCCESS = "#2AAE8A"       # confirmed mask / safe status
SUCCESS_SOFT = "#7FE3C6"

ALERT = "#E47474"         # destructive / warning
ALERT_HOVER = "#EF8A8A"

# ─── Header gradient (colored title bar) ───────────────────────────────
HEADER_FROM = "#0474ED"
HEADER_TO = "#2AAE8A"


# ─── Tissue overlay colors (RGB) ───────────────────────────────────────
# Chosen to be distinct from UI accents so they read on the dark canvas
# without colliding with status colors (alert red is the ignore-disk hue,
# so tissue colors avoid pure red).
TISSUE_RGB = {
    "pectoral": (4, 116, 237),    # accent blue
    "breast":   (42, 174, 138),   # success green
    "nipple":   (245, 158, 11),   # warm amber
}


QSS = f"""
* {{
    font-family: "Inter", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 12px;
    color: {TEXT_PRIMARY};
}}

QMainWindow, QDialog {{
    background-color: {APP_BG};
}}

QWidget#sidePanel {{
    background-color: {SURFACE};
    border-right: 1px solid {DIVIDER};
}}

QWidget#appHeader {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                stop:0 {HEADER_FROM}, stop:1 {HEADER_TO});
    border-bottom: 1px solid {DIVIDER};
}}

QLabel#appTitle {{
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding-left: 14px;
}}

QLabel#appSubtitle {{
    color: rgba(255, 255, 255, 0.85);
    font-size: 11px;
    padding-right: 14px;
}}

QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QLabel#sectionHeader {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 6px 0px 2px 0px;
    border-bottom: 1px solid {DIVIDER};
    margin-top: 6px;
}}

QLabel#statusInfo {{
    color: {TEXT_SECONDARY};
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
}}

QLabel#hint {{
    color: {TEXT_MUTED};
    font-size: 10px;
}}

QPushButton {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 6px;
    padding: 6px 12px;
    min-height: 24px;
}}
QPushButton:hover {{
    background-color: {SURFACE_HI};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {SURFACE};
}}
QPushButton:disabled {{
    color: {TEXT_MUTED};
    border-color: {DIVIDER};
    background-color: {SURFACE};
}}

QPushButton#primary {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
    border: 1px solid {ACCENT};
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton#primary:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton#success {{
    background-color: {SUCCESS};
    color: {APP_BG};
    border: 1px solid {SUCCESS};
    font-weight: 600;
}}
QPushButton#success:hover {{
    background-color: {SUCCESS_SOFT};
    color: {APP_BG};
}}

QPushButton#danger {{
    background-color: transparent;
    border-color: {ALERT};
    color: {ALERT};
}}
QPushButton#danger:hover {{
    background-color: {ALERT};
    color: {TEXT_PRIMARY};
}}

QRadioButton, QCheckBox {{
    color: {TEXT_PRIMARY};
    spacing: 8px;
    padding: 4px 0px;
}}
QRadioButton:hover, QCheckBox:hover {{
    color: {SUCCESS_SOFT};
}}
QRadioButton::indicator, QCheckBox::indicator {{
    width: 14px;
    height: 14px;
}}
QRadioButton::indicator:unchecked {{
    background-color: {SURFACE_ALT};
    border: 1px solid {TEXT_SECONDARY};
    border-radius: 7px;
}}
QRadioButton::indicator:checked {{
    background-color: {ACCENT};
    border: 2px solid {SURFACE};
    border-radius: 7px;
}}
QCheckBox::indicator:unchecked {{
    background-color: {SURFACE_ALT};
    border: 1px solid {TEXT_SECONDARY};
    border-radius: 3px;
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 3px;
}}

QSlider::groove:horizontal {{
    background: {SURFACE_ALT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {TEXT_PRIMARY};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
    border: 2px solid {ACCENT};
}}
QSlider::handle:horizontal:hover {{
    background: {ACCENT_HOVER};
    border-color: {TEXT_PRIMARY};
}}

QGraphicsView#canvas {{
    background-color: {CANVAS_BG};
    border: none;
}}

QStatusBar {{
    background-color: {SURFACE};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {DIVIDER};
}}
QStatusBar::item {{
    border: none;
}}

QToolBar {{
    background-color: {SURFACE};
    border: none;
    padding: 8px 12px;
    spacing: 10px;
}}
QToolBar::separator {{
    background-color: {DIVIDER};
    width: 1px;
    margin: 6px 8px;
}}
QToolBar QToolButton {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 6px;
    padding: 8px 16px;
    min-height: 28px;
    font-size: 12px;
    font-weight: 500;
}}
QToolBar QToolButton:hover {{
    background-color: {SURFACE_HI};
    border-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}
QToolBar QToolButton:checked, QToolBar QToolButton:pressed {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
    border-color: {ACCENT};
}}

QMenuBar {{
    background-color: {APP_BG};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {DIVIDER};
}}
QMenuBar::item:selected {{
    background-color: {SURFACE_HI};
}}
QMenu {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
}}
QMenu::item:selected {{
    background-color: {ACCENT};
}}

QScrollBar:vertical {{
    background: {APP_BG};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {SURFACE_HI};
    min-height: 20px;
    border-radius: 5px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QListWidget {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 4px;
    padding: 2px;
}}
QListWidget::item {{
    padding: 4px 6px;
    border-radius: 3px;
}}
QListWidget::item:hover {{
    background-color: {SURFACE_HI};
}}
QListWidget::item:selected {{
    background-color: {ACCENT};
    color: {TEXT_PRIMARY};
}}

QGroupBox {{
    border: 1px solid {DIVIDER};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_SECONDARY};
}}

QFrame#card {{
    background-color: {SURFACE_ALT};
    border: 1px solid {DIVIDER};
    border-radius: 8px;
}}

QWidget#sideFooter {{
    background-color: {SURFACE};
    border-top: 1px solid {DIVIDER};
}}

QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

QLabel#tissueDot {{
    border-radius: 6px;
    min-width: 12px;
    max-width: 12px;
    min-height: 12px;
    max-height: 12px;
}}

/* ── Startup dialog cards ──────────────────────────────────────── */
QDialog#startupDialog {{
    background-color: {APP_BG};
}}

QFrame#startupCard {{
    background-color: {SURFACE};
    border: 1px solid {DIVIDER};
    border-radius: 12px;
}}
QFrame#startupCard:hover {{
    background-color: {SURFACE_ALT};
    border: 2px solid {ACCENT};
}}
QFrame#startupCard[variant="manual"]:hover {{
    border-color: {SUCCESS};
}}
QFrame#startupCard[variant="positioning"]:hover {{
    border-color: {ALERT};
}}

QLabel#startupCardIcon {{
    font-size: 38px;
    color: {ACCENT};
}}
QLabel#startupCardTitle {{
    font-size: 16px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
}}
QLabel#startupCardDesc {{
    font-size: 11px;
    color: {TEXT_SECONDARY};
}}

QFrame#card:hover {{
    border: 1px solid {ACCENT};
}}

/* ── Text widgets / logs ─────────────────────────────────────── */
QTextEdit, QPlainTextEdit {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 6px;
    padding: 8px;
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRIMARY};
}}
QTextEdit[readOnly="true"], QPlainTextEdit[readOnly="true"] {{
    background-color: {SURFACE};
}}

/* Single-line input (QInputDialog uses this internally) */
QLineEdit {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRIMARY};
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled {{
    color: {TEXT_MUTED};
    background-color: {SURFACE};
}}

/* QInputDialog / QFileDialog backgrounds */
QInputDialog, QFileDialog {{
    background-color: {APP_BG};
    color: {TEXT_PRIMARY};
}}
QInputDialog QLabel, QFileDialog QLabel, QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
    background: transparent;
}}

/* Spin boxes (used in QInputDialog for numbers) */
QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}

/* Combobox (dropdown) */
QComboBox {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 4px;
    padding: 5px 10px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    selection-background-color: {ACCENT};
    selection-color: {TEXT_PRIMARY};
}}

/* ── Buttons styled via class="secondary" (positioning compat) ─ */
QPushButton[class="secondary"] {{
    background-color: {SURFACE_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {DIVIDER};
    border-radius: 6px;
    padding: 8px 14px;
    min-height: 26px;
    font-weight: 500;
}}
QPushButton[class="secondary"]:hover {{
    background-color: {SURFACE_HI};
    border-color: {ACCENT};
}}
QPushButton[class="secondary"]:pressed {{
    background-color: {SURFACE};
}}
QPushButton[class="secondary"]:disabled {{
    color: {TEXT_MUTED};
    border-color: {DIVIDER};
}}

/* ── Generic frames (positioning's containers) ───────────────── */
QFrame#posContainer {{
    background-color: {SURFACE};
    border: 1px solid {DIVIDER};
    border-radius: 8px;
}}

QLabel#posSectionHeader {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}}

QLabel#posHint {{
    color: {TEXT_MUTED};
    font-size: 10px;
}}

QLabel#posStatusGood {{
    background-color: {SURFACE_HI};
    color: {SUCCESS};
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 11px;
}}
QLabel#posStatusLoading {{
    background-color: {SURFACE_HI};
    color: {TEXT_SECONDARY};
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 11px;
}}
QLabel#posStatusError {{
    background-color: {SURFACE_HI};
    color: {ALERT};
    padding: 8px 12px;
    border-radius: 6px;
    font-weight: 500;
    font-size: 11px;
}}
QLabel#posFileLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
}}
QLabel#posFileLabelLoaded {{
    color: {SUCCESS};
    font-size: 12px;
    font-weight: 500;
}}
"""
