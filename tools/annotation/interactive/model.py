"""Model loading + thin predict surface for the interactive demo.

Single Responsibility:
    Hold the SAM2 backbone + fine-tuned MedSAM2 weights, expose a clean
    ``predict(image, box=None, points=None, neg_points=None)`` API. Nothing
    here knows about Gradio, ignore-disk post-processing, or class
    competition; those concerns live in :mod:`postprocess` and
    :mod:`inference`.

Important note about the fine-tuned checkpoint
----------------------------------------------
``wrapped_model.MedSAM2.forward`` was trained with **box prompts only**
(``box_labels = [2, 3]`` SAM corner-token semantics, see
``wrapped_model.py``). The mask decoder therefore performs best when the
prompt graph it receives at inference time *contains a box*. Pure-point
inference still runs (the underlying ``SAM2ImagePredictor`` accepts it)
but is out-of-distribution for the fine-tune; :class:`MedSAM2Service`
exposes both paths and lets the caller decide.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class ModelConfig:
    """Locations + device for model build. Immutable on purpose."""

    medsam2_repo: Path
    sam2_model_cfg: str
    sam2_base_weights: Path
    finetune_ckpt: Path
    device: torch.device


def resolve_finetuned_ckpt(
    project_dir: Path,
    env_var: str = "MEDSAM2_INTERACTIVE_CKPT",
) -> Path:
    """Pick the newest run dir that actually contains a weight file.

    Order:
      1. Explicit env override (if file exists).
      2. Newest ``work_dir/MammoBNP_v1-*/medsam_model_best.pth``.
      3. Newest ``...-latest.pth`` if no ``best``.
    """
    explicit = os.environ.get(env_var, "").strip()
    if explicit and os.path.isfile(explicit):
        return Path(explicit)

    work_dirs = sorted(glob.glob(str(project_dir / "work_dir" / "MammoBNP_v1-*")))
    if not work_dirs:
        raise SystemExit(
            f"No fine-tune output: {project_dir / 'work_dir'}/MammoBNP_v1-*\n"
            f"Run finetune_sam2_img.py or set {env_var}=...\\medsam_model_best.pth"
        )

    for d in reversed(work_dirs):
        for name in ("medsam_model_best.pth", "medsam_model_latest.pth"):
            cand = Path(d) / name
            if cand.is_file():
                return cand

    raise SystemExit(
        f"work_dir/MammoBNP_v1-* exists but no medsam_model_best.pth/latest.pth.\n"
        f"Use {env_var} to point at a finished checkpoint."
    )


class MedSAM2Service:
    """Wraps SAM2 + fine-tuned MedSAM2 weights and exposes a stateless predict API.

    Lifecycle
    ---------
    Build once at app startup. Calling :meth:`set_image` caches image
    embeddings in the underlying ``SAM2ImagePredictor`` so subsequent
    predicts (point or box) on the same image are cheap (~ms).

    Threading
    ---------
    Not thread-safe (predictor holds image-embedding state). Gradio's
    default queue is single-worker; if you bump workers, wrap calls in a
    lock.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._predictor = self._build_predictor(config)
        self._image_cached_id: int | None = None

    @staticmethod
    def _build_predictor(config: ModelConfig):
        repo = str(config.medsam2_repo)
        if not (Path(repo) / "sam2").is_dir():
            raise SystemExit(
                f"MedSAM2 repo not found at {repo}. "
                "git clone -b MedSAM2 https://github.com/bowang-lab/MedSAM.git"
            )
        sys.path.insert(0, repo)

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

        from wrapped_model import MedSAM2  # local import: avoids hard dep at import time

        medsam = MedSAM2(model=sam2_model)
        try:
            ckpt = torch.load(
                str(config.finetune_ckpt),
                map_location=str(config.device),
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(str(config.finetune_ckpt), map_location=str(config.device))
        medsam.load_state_dict(ckpt["model"], strict=True)
        medsam.eval()
        return SAM2ImagePredictor(medsam.sam2_model)

    def set_image(self, rgb: np.ndarray) -> None:
        """Cache image embeddings. Calling with the *same* numpy array (by id)
        is a no-op so consecutive predicts on the same image stay fast."""
        ident = id(rgb)
        if ident == self._image_cached_id:
            return
        self._predictor.set_image(rgb)
        self._image_cached_id = ident

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
          (matches fine-tune distribution) and points are refinement.
        - ``multimask_output``: if not specified, uses ``True`` only when
          there's exactly one positive point and no box (genuine ambiguity).
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
            multimask_output = (
                box_arr is None and len(pos) == 1 and len(neg) == 0
            )

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
