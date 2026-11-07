"""MedSAM2 inference stack used by the toolkit's SAM mode (no Qt imports).

Modules
-------
- ``model``         : model loading + low-level predict surface.
- ``wrapped_model`` : MedSAM2 wrapper matching the upstream fine-tune forward pass.
- ``prompts``       : prompt state (box / positive / ignore points), tissue presets, drawing helpers.
- ``postprocess``   : hard ignore disks and class competition.
- ``inference``     : composition pipeline: prompts + image -> final mask.

Each module owns one concern; ``torch`` is imported lazily, only when a model is built.
"""

from .inference import InferencePipeline, InferenceSettings, PredictionResult
from .model import MedSAM2Service, ModelConfig
from .postprocess import ClassCompetition, HardIgnoreMask
from .prompts import BoxPrompt, PromptState, render_overlay

__all__ = [
    "BoxPrompt",
    "ClassCompetition",
    "HardIgnoreMask",
    "InferencePipeline",
    "InferenceSettings",
    "MedSAM2Service",
    "ModelConfig",
    "PredictionResult",
    "PromptState",
    "render_overlay",
]
