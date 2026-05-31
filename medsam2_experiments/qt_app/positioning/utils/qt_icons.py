"""Qt icons from PNG paths (multi-resolution for crisp taskbar / title bar)."""

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap


def q_icon_from_png(path: Path) -> QIcon:
    pm = QPixmap(str(path))
    if pm.isNull():
        return QIcon()
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(
            pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
    return icon
