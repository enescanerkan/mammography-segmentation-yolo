"""Entry point for the interactive MedSAM2 demo.

Composition root:
  - resolves paths/checkpoints,
  - builds :class:`MedSAM2Service`,
  - wires :class:`InferencePipeline`,
  - launches the Gradio UI.

Implementation lives under ``interactive/`` (SOLID-split modules). See
``docs/PROMPTING.md`` for behavioral details (ignore semantics, implicit
box, class competition).

Run:
    python interactive_demo.py
Optional:
    set MEDSAM2_INTERACTIVE_CKPT=C:\\path\\to\\medsam_model_best.pth
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import MEDSAM2_REPO, SAM2_BASE_WEIGHTS, SAM2_MODEL_CFG
from interactive import InferencePipeline, MedSAM2Service
from interactive.model import ModelConfig, resolve_finetuned_ckpt
from interactive.ui import build_app


def _make_pipeline() -> InferencePipeline:
    finetune_ckpt = resolve_finetuned_ckpt(_HERE)
    print(f"[interactive] Fine-tune ckpt: {finetune_ckpt}")
    print(f"  (run folder: {finetune_ckpt.parent.name}, file: {finetune_ckpt.name})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[interactive] device={device}, repo={MEDSAM2_REPO}")

    config = ModelConfig(
        medsam2_repo=Path(MEDSAM2_REPO),
        sam2_model_cfg=SAM2_MODEL_CFG,
        sam2_base_weights=Path(SAM2_BASE_WEIGHTS),
        finetune_ckpt=Path(finetune_ckpt),
        device=device,
    )
    service = MedSAM2Service(config)
    return InferencePipeline(service)


def main() -> None:
    pipeline = _make_pipeline()
    app = build_app(pipeline)
    print("[interactive] Gradio başlıyor…")
    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
