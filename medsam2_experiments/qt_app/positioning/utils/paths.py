"""
Application root paths for development and PyInstaller frozen builds.

When frozen, data lives next to the .exe (writable). Bundled code lives under sys._MEIPASS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Same threshold as GoogleDriveDownloader (avoid treating HTML/error stubs as weights).
_MIN_WEIGHT_BYTES = 1_000_000


def gui_bundle_root() -> Path:
    """
    Directory used for weights/, data/, results/ next to the user:

    - Frozen (PyInstaller): folder containing MammogramAnalysis_1.1.exe
    - Dev: gui/ project directory
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # Ported layout: qt_app/positioning/utils/paths.py -> bundle = positioning/
    return Path(__file__).resolve().parent.parent


def weights_install_dir() -> Path:
    """Writable `weights/` next to the .exe (frozen) or under gui/ (dev)."""
    return gui_bundle_root() / "weights"


def resolve_weight_path(filename: str) -> Optional[Path]:
    """
    Resolve a .pt file: user override beside the exe first, then PyInstaller bundle.

    Bundled weights live under sys._MEIPASS/weights/ at runtime (read-only).
    """
    install = weights_install_dir() / filename
    bundled: Optional[Path] = None
    if getattr(sys, "frozen", False):
        root = getattr(sys, "_MEIPASS", None)
        if root:
            bundled = Path(root) / "weights" / filename

    def usable(p: Optional[Path]) -> bool:
        return (
            p is not None
            and p.is_file()
            and p.stat().st_size >= _MIN_WEIGHT_BYTES
        )

    if usable(install):
        return install
    if usable(bundled):
        return bundled
    if install.is_file():
        return install
    if bundled is not None and bundled.is_file():
        return bundled
    return None


def app_logo_path() -> Optional[Path]:
    """
    Optional branding image for window / taskbar icon (PNG).

    Dev: gui/assets/logo.png, or repo-root Gemini PNG if present.
    Frozen: bundled under _MEIPASS/assets/logo.png.
    """
    bundle = gui_bundle_root()
    local = bundle / "assets" / "logo.png"
    if local.is_file():
        return local
    if not getattr(sys, "frozen", False):
        repo_root = bundle.parent
        fallback = repo_root / "Gemini_Generated_Image_ksrkc8ksrkc8ksrk.png"
        if fallback.is_file():
            return fallback
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        frozen_logo = Path(meipass) / "assets" / "logo.png"
        if frozen_logo.is_file():
            return frozen_logo
    return None
