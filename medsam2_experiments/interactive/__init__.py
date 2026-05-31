"""Interactive MedSAM2 demo package.

Modules
-------
- ``model``        : Model loading + low-level predict surface (SRP).
- ``prompts``      : PromptState dataclass + drawing helpers.
- ``postprocess``  : Strategies that transform raw masks (hard ignore disks,
                     class competition).
- ``inference``    : Composition pipeline: prompt + image -> final mask.
- ``ui``           : Gradio UI builder.

The split follows SRP: each module owns ONE concern. The Gradio UI module
imports from the others; nothing imports from ``ui``.
"""

from .model import MedSAM2Service
from .prompts import BoxPrompt, PromptState, render_overlay
from .postprocess import ClassCompetition, HardIgnoreMask
from .inference import InferencePipeline, InferenceSettings, PredictionResult

__all__ = [
    "BoxPrompt",
    "ClassCompetition",
    "HardIgnoreMask",
    "InferencePipeline",
    "InferenceSettings",
    "MedSAM2Service",
    "PredictionResult",
    "PromptState",
    "render_overlay",
]
