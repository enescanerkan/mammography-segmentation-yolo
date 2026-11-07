"""Typed training configurations, one dataclass per model family (``configs/train/*.yaml``).

Every config has the common fields of :class:`BaseTrainConfig` plus the sections of its family. The ``family`` key
selects the dataclass; unknown keys are rejected. Values of ``None`` in the YOLO hyper-parameters mean "use the
Ultralytics default" and are not passed on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from breast_seg.config import LABEL_VALUES, YOLO_CLASSES, ConfigError

TransformSpec = Dict[str, Any]


@dataclass
class ExportConfig:
    """Where :meth:`Trainer.export` copies the selected weights: ``<dir>/<filename>``."""

    filename: str = ""
    dir: str = "${weights_dir}"
    checkpoint: str = "best"


@dataclass
class BaseTrainConfig:
    """Fields shared by every family."""

    family: str = ""
    name: str = ""
    description: str = ""
    seed: int = 42
    device: str = "0"
    workers: int = 8
    runs_dir: str = "${runs_dir}"
    export: ExportConfig = field(default_factory=ExportConfig)

    def __post_init__(self) -> None:
        if not self.family:
            raise ConfigError("family is required")
        if not self.name:
            raise ConfigError("name is required")


# ── YOLO (seg / pose / detect) ────────────────────────────────────────────────────────────────────────────────


@dataclass
class YoloDataConfig:
    """A YOLO-layout dataset: ``<dataset_dir>/images/<split>``, ``labels/<split>``. ``data.yaml`` is (re)written by
    :meth:`prepare` into the run directory, so the dataset folder is never modified."""

    dataset_dir: str = ""
    names: List[str] = field(default_factory=lambda: list(YOLO_CLASSES.values()))
    kpt_shape: Optional[List[int]] = None
    flip_idx: Optional[List[int]] = None
    keypoint_names: List[str] = field(default_factory=list)
    train: str = "images/train"
    val: str = "images/val"
    test: Optional[str] = "images/test"


@dataclass
class YoloHyperParams:
    """``model.train()`` keyword arguments (``None`` = Ultralytics default). ``extra`` passes anything else."""

    epochs: Optional[int] = None
    patience: Optional[int] = None
    batch: Optional[int] = None
    imgsz: Optional[int] = None
    optimizer: Optional[str] = None
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    warmup_epochs: Optional[float] = None
    cos_lr: Optional[bool] = None
    amp: Optional[bool] = None
    deterministic: Optional[bool] = None
    close_mosaic: Optional[int] = None
    save_period: Optional[int] = None
    plots: Optional[bool] = None
    overlap_mask: Optional[bool] = None
    mask_ratio: Optional[int] = None
    degrees: Optional[float] = None
    translate: Optional[float] = None
    scale: Optional[float] = None
    shear: Optional[float] = None
    perspective: Optional[float] = None
    fliplr: Optional[float] = None
    flipud: Optional[float] = None
    hsv_h: Optional[float] = None
    hsv_s: Optional[float] = None
    hsv_v: Optional[float] = None
    mosaic: Optional[float] = None
    mixup: Optional[float] = None
    copy_paste: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlbumentationsConfig:
    """Pixel-level Albumentations transforms injected into Ultralytics' own ``Albumentations`` step.

    Each entry is ``{name: <albumentations class>, p: ..., <kwargs>}``; lists become tuples. Geometry stays with
    Ultralytics (``degrees``, ``translate``, ``scale``, ``fliplr``).
    """

    enabled: bool = False
    transforms: List[TransformSpec] = field(default_factory=list)


@dataclass
class YoloPredictConfig:
    """Prediction export to the common label-map format (segmentation only)."""

    conf: float = 0.25
    floor: float = 0.01
    iou: Optional[float] = None
    imgsz: int = 640
    batch: int = 16


@dataclass
class YoloTrainConfig(BaseTrainConfig):
    model: str = ""
    data: YoloDataConfig = field(default_factory=YoloDataConfig)
    hyp: YoloHyperParams = field(default_factory=YoloHyperParams)
    albumentations: AlbumentationsConfig = field(default_factory=AlbumentationsConfig)
    predict: YoloPredictConfig = field(default_factory=YoloPredictConfig)
    cudnn_benchmark: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.model:
            raise ConfigError("model (base weights) is required")
        if self.family == "yolo_pose" and not self.data.kpt_shape:
            raise ConfigError("data.kpt_shape is required for yolo_pose")


# ── Semantic segmentation in plain PyTorch (SegFormer, U-Net, AttUNet) ───────────────────────────────────────


@dataclass
class SemanticDataConfig:
    """A semantic dataset: ``index.csv`` (``stem, split[, view, image, mask]``) under ``dataset_dir``; without
    ``image`` / ``mask`` columns the files are ``images/<split>/<stem>.png`` and ``masks/<split>/<stem>.png``."""

    dataset_dir: str = ""
    index: str = "index.csv"
    classes: List[str] = field(default_factory=lambda: list(LABEL_VALUES.values()))
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"


@dataclass
class OptimConfig:
    """Shared PyTorch loop: AdamW, polynomial decay per iteration, early stopping on the validation metric."""

    epochs: int = 300
    patience: int = 50
    batch: int = 8
    val_batch: Optional[int] = None
    lr: float = 1e-3
    weight_decay: float = 0.01
    poly_power: float = 1.0
    amp_dtype: str = "bfloat16"
    val_workers: int = 4

    def __post_init__(self) -> None:
        if self.amp_dtype not in ("bfloat16", "float16", "none"):
            raise ConfigError("optim.amp_dtype must be bfloat16, float16 or none")


@dataclass
class TorchTrainConfig(BaseTrainConfig):
    """Common part of the PyTorch families."""

    data: SemanticDataConfig = field(default_factory=SemanticDataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    augment: List[TransformSpec] = field(default_factory=list)
    flip_tta: bool = True


@dataclass
class SegFormerTrainConfig(TorchTrainConfig):
    backbone: str = "nvidia/mit-b3"
    image_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    image_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class UNetTrainConfig(TorchTrainConfig):
    base_channels: int = 64
    in_channels: int = 1


@dataclass
class AttUNetTrainConfig(TorchTrainConfig):
    """Attention U-Net with CoordConv and a soft-argmax landmark head (``w_seg = 0``: landmark only;
    ``w_seg > 0``: multitask with the segmentation head). ``landmarks_csv`` is the canvas table
    ``stem, role, x, y`` (see :func:`breast_seg.evaluation.gt.canvas_landmark_table`)."""

    base_channels: int = 64
    keypoints: List[str] = field(default_factory=lambda: ["nipple", "pec_top", "pec_bot"])
    landmarks_csv: str = ""
    w_seg: float = 0.0
    w_lm: float = 20.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.landmarks_csv:
            raise ConfigError("landmarks_csv is required for attunet")


# ── nnU-Net v2 ────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class NnUNetDataConfig:
    """Source semantic dataset converted to the raw nnU-Net layout by :meth:`prepare` (skipped when the raw
    dataset already exists and ``reuse_raw`` is set)."""

    dataset_dir: str = ""
    index: str = "index.csv"
    cv_folds: int = 0
    cv_seed: int = 42
    reuse_raw: bool = True
    description: str = ""


@dataclass
class NnUNetTrainConfig(BaseTrainConfig):
    dataset_id: int = 0
    dataset_name: str = ""
    configuration: str = "2d"
    fold: Union[int, str] = 0
    planner: str = "nnUNetPlannerResEncM"
    base_plans: str = "nnUNetResEncUNetMPlans"
    plans: str = "nnUNetResEncUNetMPlans_fullM"
    patch_size: Optional[List[int]] = None
    batch_size: Optional[int] = None
    transfer_plans_from: Optional[int] = None
    trainer: str = "nnUNetTrainerMammo"
    install_trainer: bool = True
    num_processes: int = 8
    npz: bool = False
    verify_dataset_integrity: bool = True
    checkpoint: str = "checkpoint_final.pth"
    disable_tta: bool = False
    nnunet_raw: str = "${nnunet_raw}"
    nnunet_preprocessed: str = "${nnunet_preprocessed}"
    nnunet_results: str = "${nnunet_results}"
    data: NnUNetDataConfig = field(default_factory=NnUNetDataConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dataset_id <= 0 or not self.dataset_name:
            raise ConfigError("dataset_id and dataset_name are required for nnunet")
        if (self.patch_size is None) != (self.batch_size is None) and self.plans != self.base_plans:
            raise ConfigError("give both patch_size and batch_size to derive plans, or neither")
