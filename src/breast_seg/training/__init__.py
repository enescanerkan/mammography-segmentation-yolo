"""Training: one :class:`Trainer` interface (prepare / fit / predict / evaluate / export) over every model family,
typed YAML configs (``configs/train``) and k-fold cross-validation (``configs/cv``).

    from breast_seg.training import build_trainer
    trainer = build_trainer("configs/train/yolo26m_seg_v4.yaml", overrides=["device=1"])
    trainer.prepare()      # writes data.yaml into the run directory, trains nothing
    trainer.fit()

Families: ``yolo_seg``, ``yolo_pose``, ``yolo_detect``, ``segformer``, ``nnunet``, ``unet``, ``attunet``.
"""
from breast_seg.training.base import (
    SplitSpec,
    Trainer,
    TrainResult,
    available_families,
    build_trainer,
    get_trainer_class,
    load_train_config,
    register_trainer,
    train_config_from_dict,
)
from breast_seg.training.config import (
    AttUNetTrainConfig,
    BaseTrainConfig,
    NnUNetTrainConfig,
    SegFormerTrainConfig,
    TorchTrainConfig,
    UNetTrainConfig,
    YoloTrainConfig,
)
from breast_seg.training.cv import CrossValidationRunner, CVConfig, load_cv_config

__all__ = [
    "AttUNetTrainConfig",
    "BaseTrainConfig",
    "CVConfig",
    "CrossValidationRunner",
    "NnUNetTrainConfig",
    "SegFormerTrainConfig",
    "SplitSpec",
    "TorchTrainConfig",
    "TrainResult",
    "Trainer",
    "UNetTrainConfig",
    "YoloTrainConfig",
    "available_families",
    "build_trainer",
    "get_trainer_class",
    "load_cv_config",
    "load_train_config",
    "register_trainer",
    "train_config_from_dict",
]
