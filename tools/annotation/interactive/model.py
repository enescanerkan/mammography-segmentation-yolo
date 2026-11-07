"""Model loading + thin predict surface for SAM mode.

Single responsibility:
    Hold the SAM2 backbone + fine-tuned MedSAM2 weights and expose a clean
    ``predict(box=None, positive_points=None, negative_points=None)`` API. Nothing
    here knows about the GUI, ignore-disk post-processing or class competition;
    those concerns live in :mod:`postprocess` and :mod:`inference`.

Checkpoint selection
--------------------
The fine-tuned checkpoint is always the explicit file given in
:class:`ModelConfig` (``medsam2.checkpoint`` in the toolkit config). There is no
directory scanning / "newest run" heuristic.

Note about the fine-tuned checkpoint
------------------------------------
:class:`~interactive.wrapped_model.MedSAM2` was trained with **box prompts only**
(``box_labels = [2, 3]`` SAM corner-token semantics). The mask decoder therefore
performs best when the prompt it receives at inference time *contains a box*.
Pure-point inference still runs (``SAM2ImagePredictor`` accepts it) but is
out-of-distribution for the fine-tune; :class:`~interactive.inference.InferencePipeline`
synthesizes an implicit box from positive points to compensate.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    """Locations + device for the model build. Immutable on purpose."""

    medsam2_repo: Path
    sam2_model_cfg: str
    sam2_base_weights: Path
    finetune_ckpt: Path
    device: str = "cpu"

    @classmethod
    def from_toolkit(cls, cfg) -> ModelConfig:
        """Build from a :class:`toolkit_config.ToolkitConfig` (``device: auto`` resolved here)."""
        from toolkit_config import resolve_device

        return cls(
            medsam2_repo=Path(cfg.medsam2_repo),
            sam2_model_cfg=cfg.sam2_model_cfg,
            sam2_base_weights=Path(cfg.sam2_base_weights),
            finetune_ckpt=Path(cfg.medsam2_checkpoint),
            device=resolve_device(cfg.device),
        )

    def missing_files(self) -> list[str]:
        problems = []
        if not (self.medsam2_repo / "sam2").is_dir():
            problems.append(f"MedSAM2 code not found at {self.medsam2_repo}")
        if not self.sam2_base_weights.is_file():
            problems.append(f"SAM2 base weights not found: {self.sam2_base_weights}")
        if not self.finetune_ckpt.is_file():
            problems.append(f"Fine-tuned MedSAM2 checkpoint not found: {self.finetune_ckpt}")
        return problems


class MedSAM2Service:
    """Wraps SAM2 + fine-tuned MedSAM2 weights and exposes a stateless predict API.

    Lifecycle
    ---------
    Build once. :meth:`set_image` caches the image embeddings inside the
    underlying ``SAM2ImagePredictor`` so subsequent predicts (point or box) on
    the same image are cheap.

    Threading
    ---------
    Not thread-safe (the predictor holds image-embedding state). The Qt app runs
    all predictions on one worker thread.
    """

    def __init__(self, config: ModelConfig) -> None:
        problems = config.missing_files()
        if problems:
            raise FileNotFoundError("Cannot build MedSAM2:\n  " + "\n  ".join(problems))
        self.config = config
        self._predictor = self._build_predictor(config)
        # Keep a reference to the cached image (not just its id()): an id can be reused
        # by a new array after the old one is garbage-collected, which would silently
        # reuse stale embeddings.
        self._cached_image: np.ndarray | None = None

    @staticmethod
    def _build_predictor(config: ModelConfig):
        import torch

        repo = str(config.medsam2_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # build_sam2 resolves its Hydra config relative to the MedSAM2 checkout.
        prev_cwd = os.getcwd()
        os.chdir(repo)
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(
                config.sam2_model_cfg,
                str(config.sam2_base_weights),
                device=str(config.device),
                mode="eval",
                apply_postprocessing=True,
            )
        finally:
            os.chdir(prev_cwd)

        from .wrapped_model import MedSAM2

        medsam = MedSAM2(model=sam2_model)
        try:
            ckpt = torch.load(str(config.finetune_ckpt), map_location=str(config.device), weights_only=False)
        except TypeError:  # torch < 1.13 has no weights_only
            ckpt = torch.load(str(config.finetune_ckpt), map_location=str(config.device))
        medsam.load_state_dict(ckpt["model"], strict=True)
        medsam.eval()
        return SAM2ImagePredictor(medsam.sam2_model)

    def set_image(self, rgb: np.ndarray) -> None:
        """Cache image embeddings. Calling again with the *same* array object is a no-op."""
        if rgb is self._cached_image:
            return
        self._predictor.set_image(rgb)
        self._cached_image = rgb

    def predict(
        self,
        *,
        box_xyxy: np.ndarray | None = None,
        positive_points: Iterable[tuple[float, float]] | None = None,
        negative_points: Iterable[tuple[float, float]] | None = None,
        multimask_output: bool | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run a single forward pass.

        Returns
        -------
        mask : np.ndarray  (H, W) uint8 in {0, 1}
        score : float      SAM's reported quality score for the chosen mask.

        Notes
        -----
        - When both box and points are given, the box drives the geometry
          (matches the fine-tune distribution) and points refine it.
        - ``multimask_output``: if not specified, ``True`` only when there is
          exactly one positive point and no box (genuine ambiguity).
        """
        pos = list(positive_points or [])
        neg = list(negative_points or [])

        point_coords: np.ndarray | None = None
        point_labels: np.ndarray | None = None
        if pos or neg:
            point_coords = np.asarray(pos + neg, dtype=np.float32)
            point_labels = np.asarray([1] * len(pos) + [0] * len(neg), dtype=np.int32)

        box_arr: np.ndarray | None = None
        if box_xyxy is not None:
            box_arr = np.asarray(box_xyxy, dtype=np.float32).reshape(4)

        if multimask_output is None:
            multimask_output = box_arr is None and len(pos) == 1 and len(neg) == 0

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_arr,
            multimask_output=multimask_output,
        )
        chosen, score = self._pick_best(masks, scores)
        return self._to_uint8(chosen), score

    @staticmethod
    def _pick_best(masks: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
        if masks.ndim == 2:
            return masks, float(np.atleast_1d(scores)[0])
        if masks.shape[0] == 1:
            return masks[0], float(np.atleast_1d(scores)[0])
        k = int(np.argmax(scores))
        return masks[k], float(scores[k])

    @staticmethod
    def _to_uint8(mask: np.ndarray) -> np.ndarray:
        if mask.dtype == np.bool_:
            return mask.astype(np.uint8)
        if np.issubdtype(mask.dtype, np.floating):
            return (mask > 0.5).astype(np.uint8)
        return (mask > 0).astype(np.uint8)

    @staticmethod
    def fit_to_image(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        """Resize a binary mask to (height, width) without antialiasing."""
        if mask.shape == (height, width):
            return mask
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def build_pipeline(cfg):
    """Build an :class:`~interactive.inference.InferencePipeline` from the toolkit config.

    Raises ``FileNotFoundError`` listing every missing file before importing torch / sam2.
    """
    from .inference import InferencePipeline

    problems = cfg.missing_medsam2_files()
    if problems:
        raise FileNotFoundError("Cannot start SAM mode:\n  " + "\n  ".join(problems))
    return InferencePipeline(MedSAM2Service(ModelConfig.from_toolkit(cfg)))
