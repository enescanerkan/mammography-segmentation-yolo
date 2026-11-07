"""Shared fixtures for the annotation-toolkit tests.

The tests never need model weights or a GPU: CUDA is hidden, Qt runs offscreen and the
toolkit config is pointed at an empty temporary weights folder, so every model shows its
"weights missing" state.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLKIT_ROOT))

import toolkit_config  # noqa: E402  (needs the sys.path entry above)

toolkit_config.ensure_breast_seg_importable()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Empty YAML + empty weights root; no inherited toolkit environment variables."""
    for names in toolkit_config.ENV_VARS.values():
        for name in names:
            monkeypatch.delenv(name, raising=False)
    cfg_file = tmp_path / "annotation.yaml"
    cfg_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(toolkit_config.CONFIG_ENV_VAR, str(cfg_file))
    monkeypatch.setenv("ANNOTATION_WEIGHTS_ROOT", str(tmp_path / "weights"))
    monkeypatch.setenv("ANNOTATION_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("POSITIONING_RESULTS_DIR", str(tmp_path / "results"))
    toolkit_config.set_config_path(None)
    yield tmp_path
    toolkit_config.set_config_path(None)


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["annotation-tests"])
    app.setQuitOnLastWindowClosed(False)
    yield app
