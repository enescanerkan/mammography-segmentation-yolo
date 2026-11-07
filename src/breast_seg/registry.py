"""Model registry: the one place a weight path, its preprocessing recipe and its output type are declared.

A registry is a YAML file (see ``configs/models.yaml``)::

    weights_root: ..                 # optional; relative to this file
    models:
      - id: pose_mlo
        task: landmark               # landmark | segmentation | detection | prompt_segmentation
        view: MLO                    # MLO | CC | both
        recipe: wavelet640           # wavelet640 | baseline640 | raw1024
        weights: positioning/mlo-yolo26-pose-advanced.pt   # relative to weights_root
        keypoints: [nipple, pectoral_top, pectoral_bottom]

Weight paths are always relative to a *weights root*, resolved in this order: the ``weights_root`` argument of
:meth:`Registry.load` (application config / command line), the ``BREAST_SEG_WEIGHTS`` environment variable, the
``weights_root`` key of the YAML (relative to the YAML file), and finally the YAML file's own directory. No
machine path therefore ever has to be written into a committed file.

The engine picks its models by role (:meth:`Registry.select`): the first *enabled* entry with the requested task
whose view matches. Swapping a model therefore means flipping ``enabled`` flags, never touching inference code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import yaml

WEIGHTS_ENV = "BREAST_SEG_WEIGHTS"

TASKS = ("landmark", "segmentation", "detection", "prompt_segmentation")
VIEWS = ("MLO", "CC", "both")
RECIPES = ("wavelet640", "baseline640", "raw1024")

_MODEL_KEYS = frozenset({"id", "label", "task", "view", "recipe", "weights", "enabled", "classes", "keypoints",
                         "conf", "iou", "nipple_rescue", "note"})
_RESCUE_KEYS = frozenset({"enabled", "conf", "max_mm"})
# Accepted at the top level: ``root`` is the legacy spelling of ``weights_root``; the other legacy keys belonged
# to the application settings of the original web app and are ignored here (they live in the app config now).
_TOP_KEYS = frozenset({"weights_root", "root", "models"})
_LEGACY_TOP_KEYS = frozenset({"rules", "preview", "criteria", "retention"})


class RegistryError(ValueError):
    """The registry YAML is malformed (unknown key, bad task / view / recipe, duplicate id)."""


@dataclass(frozen=True)
class NippleRescue:
    """Second, low-confidence nipple pass of the segmentation model, accepted only near the pose nipple.

    The nipple is the class the segmentation model under-scores: at the normal confidence it is missed on a
    noticeable share of images, yet a second pass at a much lower confidence usually puts a candidate within a few
    millimetres of the pose model's nipple. The candidate is taken only when the independent landmark model vouches
    for its location (centroid within ``max_mm``).
    """

    enabled: bool = False
    conf: float = 0.02
    max_mm: float = 15.0

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> NippleRescue:
        raw = dict(raw or {})
        unknown = set(raw) - _RESCUE_KEYS
        if unknown:
            raise RegistryError(f"nipple_rescue: unknown keys {sorted(unknown)}")
        return cls(enabled=bool(raw.get("enabled", False)), conf=float(raw.get("conf", 0.02)),
                   max_mm=float(raw.get("max_mm", 15.0)))


@dataclass(frozen=True)
class ModelSpec:
    """One entry of the registry."""

    id: str
    task: str
    recipe: str
    weights: str  # relative to the weights root, as written in the YAML
    weights_path: Path  # resolved
    label: str = ""
    view: str = "both"
    enabled: bool = True
    classes: Tuple[str, ...] = ()
    keypoints: Tuple[str, ...] = ()
    conf: float = 0.25
    iou: float = 0.5
    nipple_rescue: NippleRescue = field(default_factory=NippleRescue)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], weights_root: Path) -> ModelSpec:
        if not isinstance(raw, Mapping):
            raise RegistryError(f"model entry must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - _MODEL_KEYS
        if unknown:
            raise RegistryError(f"model {raw.get('id', '?')!r}: unknown keys {sorted(unknown)}")
        for key in ("id", "task", "recipe", "weights"):
            if not raw.get(key):
                raise RegistryError(f"model {raw.get('id', '?')!r}: missing {key!r}")
        task, view, recipe = str(raw["task"]), str(raw.get("view", "both")), str(raw["recipe"])
        if task not in TASKS:
            raise RegistryError(f"model {raw['id']!r}: task {task!r} not in {TASKS}")
        if view not in VIEWS:
            raise RegistryError(f"model {raw['id']!r}: view {view!r} not in {VIEWS}")
        if recipe not in RECIPES:
            raise RegistryError(f"model {raw['id']!r}: recipe {recipe!r} not in {RECIPES}")
        weights = str(raw["weights"])
        if Path(weights).is_absolute():
            raise RegistryError(f"model {raw['id']!r}: weights must be relative to the weights root")
        return cls(
            id=str(raw["id"]),
            label=str(raw.get("label") or raw["id"]),
            task=task,
            view=view,
            recipe=recipe,
            weights=weights,
            weights_path=weights_root / weights,
            enabled=bool(raw.get("enabled", True)),
            classes=tuple(str(c) for c in (raw.get("classes") or ())),
            keypoints=tuple(str(k) for k in (raw.get("keypoints") or ())),
            conf=float(raw.get("conf", 0.25)),
            iou=float(raw.get("iou", 0.5)),
            nipple_rescue=NippleRescue.from_dict(raw.get("nipple_rescue")),
        )

    @property
    def available(self) -> bool:
        """The weights file exists."""
        return self.weights_path.is_file()

    def matches(self, task: str, view: Optional[str] = None) -> bool:
        """Same task, and (when ``view`` is given) a view of ``view`` or ``both``."""
        return self.task == task and (view is None or self.view in (view, "both"))

    def to_dict(self) -> Dict[str, Any]:
        """Public summary (the weights path stays relative, so no machine path leaves the server)."""
        return {
            "id": self.id,
            "label": self.label,
            "task": self.task,
            "view": self.view,
            "recipe": self.recipe,
            "enabled": self.enabled,
            "available": self.available,
            "classes": list(self.classes),
            "keypoints": list(self.keypoints),
            "weights": self.weights,
        }


@dataclass(frozen=True)
class Registry:
    """The parsed registry: model specs plus the weights root they were resolved against."""

    models: Tuple[ModelSpec, ...]
    weights_root: Path
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Union[str, Path], weights_root: Optional[Union[str, Path]] = None) -> Registry:
        """Read a registry YAML; ``weights_root`` overrides the environment and the YAML key."""
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, base_dir=path.resolve().parent, weights_root=weights_root, path=path)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base_dir: Union[str, Path] = ".",
                  weights_root: Optional[Union[str, Path]] = None, path: Optional[Path] = None) -> Registry:
        if not isinstance(raw, Mapping):
            raise RegistryError("registry must be a mapping with a 'models' list")
        unknown = set(raw) - _TOP_KEYS - _LEGACY_TOP_KEYS
        if unknown:
            raise RegistryError(f"registry: unknown keys {sorted(unknown)}")
        root = resolve_weights_root(raw.get("weights_root", raw.get("root")), Path(base_dir), weights_root)
        models = tuple(ModelSpec.from_dict(entry, root) for entry in (raw.get("models") or ()))
        ids = [m.id for m in models]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise RegistryError(f"registry: duplicate model ids {duplicates}")
        return cls(models=models, weights_root=root, path=path)

    def get(self, model_id: str) -> Optional[ModelSpec]:
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def active(self) -> List[ModelSpec]:
        """Enabled entries, in file order."""
        return [m for m in self.models if m.enabled]

    def select(self, task: str, view: Optional[str] = None) -> Optional[ModelSpec]:
        """The first enabled entry for ``task`` (and ``view``), or None."""
        for model in self.models:
            if model.enabled and model.matches(task, view):
                return model
        return None


def resolve_weights_root(yaml_value: Optional[Union[str, Path]], base_dir: Path,
                         explicit: Optional[Union[str, Path]] = None) -> Path:
    """explicit > ``$BREAST_SEG_WEIGHTS`` > YAML ``weights_root`` (relative to ``base_dir``) > ``base_dir``."""
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get(WEIGHTS_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    if yaml_value:
        value = Path(str(yaml_value)).expanduser()
        return value if value.is_absolute() else (base_dir / value).resolve()
    return base_dir
