"""The trainer interface and the family registry.

A trainer turns one training config into a run directory ``<runs_dir>/<name>``:

* :meth:`Trainer.prepare` - write what the framework needs (``data.yaml``, an nnU-Net raw dataset and plans...);
  nothing is trained, so it is safe to call on any machine.
* :meth:`Trainer.fit` - train; returns a :class:`TrainResult` with the selected weights.
* :meth:`Trainer.predict` - write predictions for a list of images in the common label-map format.
* :meth:`Trainer.evaluate` - score a split of the configured dataset.
* :meth:`Trainer.export` - copy the selected weights to their expected file name.

Families register with :func:`register_trainer`; :func:`load_train_config` picks the config dataclass from the
``family`` key and :func:`build_trainer` instantiates the matching trainer. Heavy frameworks (torch, ultralytics,
transformers, nnU-Net) are imported only inside the methods that need them.
"""
from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

from breast_seg.config import ConfigError, PathsConfig, apply_overrides, from_dict, read_yaml, to_dict
from breast_seg.training.config import BaseTrainConfig


@dataclass
class TrainResult:
    """Outcome of :meth:`Trainer.fit`."""

    run_dir: Path
    weights: Optional[Path] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    """Explicit images of one run (cross-validation): ``stem -> split``, an image path per stem and, for labelled
    stems, the YOLO label and / or semantic mask path."""

    split_of: Mapping[str, str]
    images: Mapping[str, Path]
    labels: Mapping[str, Path] = field(default_factory=dict)
    masks: Mapping[str, Path] = field(default_factory=dict)
    fold: int = 0

    def stems(self, split: str) -> List[str]:
        return sorted(s for s, sp in self.split_of.items() if sp == split)


class Trainer(ABC):
    """Base class of every family. Subclasses set ``family`` and ``config_class``."""

    family: ClassVar[str] = ""
    config_class: ClassVar[Type[BaseTrainConfig]] = BaseTrainConfig
    #: False when fit() expects prepare() to have run (nnU-Net: cross-validation prepares the shared dataset once).
    prepares_in_fit: ClassVar[bool] = True

    def __init__(self, config: BaseTrainConfig, paths: Optional[PathsConfig] = None):
        if not isinstance(config, self.config_class):
            raise TypeError(f"{type(self).__name__} needs a {self.config_class.__name__}, got {type(config).__name__}")
        self.config = config
        self.paths = paths or PathsConfig.load()

    # ── locations ──────────────────────────────────────────────────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        return self.paths.resolve(self.config.runs_dir) / self.config.name

    def resolve(self, value: Union[str, Path]) -> Path:
        return self.paths.resolve(value)

    def save_config(self) -> Path:
        """Write the resolved config next to the run (``train_config.json``)."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "train_config.json"
        path.write_text(json.dumps(to_dict(self.config), indent=2))
        return path

    # ── interface ──────────────────────────────────────────────────────────────────────────────────────────

    @abstractmethod
    def prepare(self) -> Path:
        """Write the framework inputs; return the main one (data.yaml, dataset folder...)."""

    @abstractmethod
    def fit(self) -> TrainResult:
        """Train and return the selected weights."""

    @abstractmethod
    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None) -> List[Path]:
        """Write one prediction per image to ``out_dir`` (``<stem>.png`` label maps for segmentation)."""

    @abstractmethod
    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        """Score ``split`` of the configured dataset."""

    @abstractmethod
    def best_weights(self) -> Path:
        """Path of the selected checkpoint of this run (``export.checkpoint``)."""

    def use_split(self, spec: SplitSpec, work_dir: Path) -> Trainer:
        """A trainer for the same recipe on an explicit image list (one cross-validation fold).

        Families materialise ``spec`` under ``work_dir`` (links, an index or a data.yaml) and return a new trainer;
        the original is not modified.
        """
        raise NotImplementedError(f"{self.family} does not support explicit splits")

    def export(self, dest: Optional[Path] = None, weights: Optional[Path] = None) -> Path:
        """Copy the selected weights to ``dest`` (default ``<export.dir>/<export.filename>``)."""
        src = Path(weights) if weights else self.best_weights()
        if dest is None:
            if not self.config.export.filename:
                raise ConfigError(f"{self.config.name}: export.filename is not set")
            dest = self.resolve(self.config.export.dir) / self.config.export.filename
        dest = Path(dest)
        if not src.exists():
            raise FileNotFoundError(f"no weights to export at {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)
        return dest


# ── registry ─────────────────────────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[Trainer]] = {}


def register_trainer(family: str):
    """Class decorator: register a :class:`Trainer` subclass under ``family``."""

    def deco(cls: Type[Trainer]) -> Type[Trainer]:
        if family in _REGISTRY and _REGISTRY[family] is not cls:
            raise ValueError(f"family {family!r} is already registered by {_REGISTRY[family].__name__}")
        cls.family = family
        _REGISTRY[family] = cls
        return cls

    return deco


def _ensure_families() -> None:
    from breast_seg.training import families  # noqa: F401  (registers the built-in families)


def available_families() -> List[str]:
    _ensure_families()
    return sorted(_REGISTRY)


def get_trainer_class(family: str) -> Type[Trainer]:
    _ensure_families()
    try:
        return _REGISTRY[family]
    except KeyError:
        raise ConfigError(f"unknown family {family!r} (available: {', '.join(sorted(_REGISTRY))})") from None


def train_config_from_dict(data: Mapping[str, Any]) -> BaseTrainConfig:
    """Build the family's config dataclass from a plain mapping (strict)."""
    family = data.get("family")
    if not family:
        raise ConfigError("family is required")
    return from_dict(get_trainer_class(str(family)).config_class, data)


def load_train_config(path: Union[str, Path], overrides: Union[None, Mapping[str, Any], Iterable[str]] = None) -> BaseTrainConfig:
    """Read ``configs/train/<name>.yaml`` (``base:`` chain, dotted ``overrides``) into its family's dataclass."""
    data = apply_overrides(read_yaml(path), overrides)
    try:
        return train_config_from_dict(data)
    except ConfigError as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


def build_trainer(config: Union[BaseTrainConfig, str, Path], paths: Optional[PathsConfig] = None,
                  overrides: Union[None, Mapping[str, Any], Iterable[str]] = None) -> Trainer:
    """Trainer for a config object or a config file path."""
    if not isinstance(config, BaseTrainConfig):
        config = load_train_config(config, overrides)
    elif overrides:
        config = train_config_from_dict(apply_overrides(to_dict(config), overrides))
    return get_trainer_class(config.family)(config, paths)
