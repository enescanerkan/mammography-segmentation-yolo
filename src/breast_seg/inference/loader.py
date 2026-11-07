"""How a registry entry becomes a callable model.

The engine never imports a framework itself: it receives a *loader*, a callable ``(spec) -> model`` where the
model exposes the Ultralytics interface it uses (``predict(image, **kwargs) -> [result]`` and ``names``). The
default loader imports Ultralytics lazily, so ``import breast_seg.inference`` stays cheap and tests can inject a
fake without any weights.
"""
from __future__ import annotations

from typing import Any, Callable

from ..registry import ModelSpec

ModelLoader = Callable[[ModelSpec], Any]


class UnsupportedModel(RuntimeError):
    """The loader cannot build this kind of model (e.g. a prompt-segmentation checkpoint)."""


def ultralytics_loader(spec: ModelSpec) -> Any:
    """Load a YOLO checkpoint with Ultralytics (imported on first use)."""
    if spec.task == "prompt_segmentation":
        raise UnsupportedModel(f"{spec.id}: prompt segmentation is not served by the Ultralytics loader")
    from ultralytics import YOLO

    return YOLO(str(spec.weights_path))
