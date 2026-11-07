"""Browser prototype of SAM mode (Gradio), kept for quick experiments.

The desktop app (``tools/annotation/app.py``, SAM Label card) is the supported tool;
this demo shares its inference stack (``interactive/``) and configuration.

Run (needs ``pip install gradio`` in addition to the toolkit requirements):
    python tools/annotation/experiments/gradio_demo.py
"""

from __future__ import annotations

import config  # noqa: F401  (puts tools/annotation on sys.path)
from gradio_ui import build_app

from interactive.model import build_pipeline
from toolkit_config import get_config


def main() -> None:
    cfg = get_config()
    print(f"[gradio-demo] MedSAM2 checkpoint: {cfg.medsam2_checkpoint}")
    print(f"[gradio-demo] device={cfg.device} repo={cfg.medsam2_repo}")
    app = build_app(build_pipeline(cfg))
    print("[gradio-demo] starting Gradio...")
    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
