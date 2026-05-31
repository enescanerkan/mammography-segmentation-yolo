"""Inference pipeline: compose model + post-process strategies.

Single Responsibility:
    Take ``(image, prompt_state, tissue, settings, masks_by_tissue)`` and
    return a clean binary mask + diagnostic info. Nothing Gradio-specific
    lives here.

Box-anchor heuristic
--------------------
The fine-tuned MedSAM2 decoder was trained with box-only prompts (see
``wrapped_model.py``). Pure point prompts therefore live out-of-distribution
and degrade as the user adds more points. To bring them back into
distribution we synthesize an "implicit box" from the convex bbox of all
positive points (with padding). The model then sees ``box + points``
which is much closer to its training regime.

The implicit box can be disabled in :class:`InferenceSettings` for
debugging / comparison.

Class competition
-----------------
Operates on a dict of ``{tissue_key: mask}`` provided by the UI (one per
previously-segmented tissue). If the current tissue is breast and a
pectoral mask exists, those pectoral pixels are subtracted from the
breast prediction. No auto-prediction of the exclude class is done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from .model import MedSAM2Service
from .postprocess import ClassCompetition, HardIgnoreMask
from .prompts import PromptState, TissuePreset, implicit_box_from_points


@dataclass
class InferenceSettings:
    """User-facing knobs surfaced in the UI."""

    ignore_radius_px: int = 20
    use_implicit_box: bool = True
    implicit_box_pad_px: int = 64
    use_class_competition: bool = True
    use_soft_negative: bool = True


@dataclass
class PredictionResult:
    mask: np.ndarray
    score: float
    used_box: Optional[np.ndarray] = None
    used_implicit_box: bool = False
    excluded_pixels: int = 0
    excluded_by: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def pixel_count(self) -> int:
        return int(self.mask.sum())


class InferencePipeline:
    """Composes model service + ignore + class-competition into one call.

    Order of operations:
      1. Resolve effective box: explicit user box, else implicit-from-points.
      2. Run model with box + positive points + (optionally) soft negatives.
      3. Resize mask to image space.
      4. If class competition is on AND an excluding tissue mask exists in
         ``masks_by_tissue`` → subtract it.
      5. Apply hard ignore disks (mechanical subtraction).
    """

    def __init__(self, service: MedSAM2Service) -> None:
        self._service = service
        self._competition = ClassCompetition()

    def run(
        self,
        rgb: np.ndarray,
        state: PromptState,
        tissue: TissuePreset,
        settings: InferenceSettings,
        masks_by_tissue: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Optional[PredictionResult]:
        if state.is_empty():
            return None

        h, w = rgb.shape[:2]
        notes: list[str] = []

        explicit_box = state.box.to_xyxy()
        used_implicit = False
        box = explicit_box
        if box is None and settings.use_implicit_box and state.positive:
            box = implicit_box_from_points(
                state.positive, (h, w), pad_px=settings.implicit_box_pad_px,
            )
            if box is not None:
                used_implicit = True
                notes.append(
                    f"Implicit box from {len(state.positive)} positive point(s) "
                    f"(pad={settings.implicit_box_pad_px}px)"
                )

        soft_neg = state.ignore if settings.use_soft_negative else []

        self._service.set_image(rgb)
        raw_mask, score = self._service.predict(
            box_xyxy=box,
            positive_points=state.positive,
            negative_points=soft_neg,
        )
        mask = self._service.fit_to_image(raw_mask, h, w)

        excluded_pixels = 0
        excluded_by: list[str] = []
        if settings.use_class_competition and masks_by_tissue:
            exclude = self._competition.resolve_exclude(masks_by_tissue, tissue.key)
            if exclude is not None:
                mask, removed = self._competition.subtract(mask, exclude)
                excluded_pixels = removed
                if removed > 0:
                    excluded_by.append("pectoral" if tissue.key == "breast" else "?")
                    notes.append(
                        f"Class competition: removed {removed}px "
                        f"(önceden çizilen pectoral maskesi ile)"
                    )

        ignore_applier = HardIgnoreMask(radius_px=settings.ignore_radius_px)
        if state.ignore:
            before = int(mask.sum())
            mask = ignore_applier.apply(mask, state.ignore)
            removed = before - int(mask.sum())
            notes.append(
                f"Hard ignore: {len(state.ignore)} disk(s) of R={settings.ignore_radius_px}px "
                f"removed {removed}px"
            )

        return PredictionResult(
            mask=mask,
            score=score,
            used_box=box,
            used_implicit_box=used_implicit,
            excluded_pixels=excluded_pixels,
            excluded_by=excluded_by,
            notes=notes,
        )
