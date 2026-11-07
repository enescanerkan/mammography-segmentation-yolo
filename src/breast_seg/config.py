"""Configuration: the class map every module shares, the local paths file, and the YAML loader behind every config.

Three things live here:

* **Class definitions.** YOLO class ids (``0 pectoral, 1 breast-tissue, 2 nipple``), the semantic label-map values
  (``0 background, 1 pectoral, 2 breast-tissue, 3 nipple``) and the paint order that flattens overlapping instance
  masks into one label map.
* :class:`Config`, the inference defaults used by :mod:`breast_seg.analyzer` and the compare pipeline.
* The YAML machinery used by training, dataset and cross-validation configs: ``base:`` inheritance, dotted
  overrides, strict typed dataclass construction (unknown keys are rejected) and :class:`PathsConfig`, the one
  place machine paths are written. Committed configs only reference ``${placeholders}`` resolved against it.
"""
from __future__ import annotations

import dataclasses
import os
import re
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

import yaml

# ── Class definitions ─────────────────────────────────────────────────────────────────────────────────────────

#: YOLO class id -> name (order of ``names`` in every YOLO ``data.yaml``).
YOLO_CLASSES: Dict[int, str] = {0: "pectoral", 1: "breast-tissue", 2: "nipple"}
#: Semantic label-map value -> name (nnU-Net, SegFormer, U-Net and the common prediction format).
LABEL_VALUES: Dict[int, str] = {0: "background", 1: "pectoral", 2: "breast-tissue", 3: "nipple"}
#: (yolo class id, label value) in painting order: breast tissue first, pectoral over it (so the pectoral keeps its
#: outermost column at the tissue border, which the pectoral-line fit reads), nipple on top.
PAINT_ORDER: Tuple[Tuple[int, int], ...] = ((1, 2), (0, 1), (2, 3))
#: Label value -> YOLO class id.
VALUE_TO_YOLO: Dict[int, int] = {value: cls for cls, value in PAINT_ORDER}


def _default_device() -> str:
    """CUDA device ``0`` if available, otherwise ``cpu``. ``BREAST_SEG_DEVICE`` overrides."""
    raw = os.environ.get("BREAST_SEG_DEVICE", "").strip()
    if raw:
        return raw
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Config:
    """Inference defaults of the YOLO segmentation model and the class ids the analyzer reads.

    Training is configured by YAML files under ``configs/train`` (see :mod:`breast_seg.training`); this class only
    carries what inference and the geometric analysis need.
    """

    CLASS_NAMES: Dict[int, str] = field(default_factory=lambda: dict(YOLO_CLASSES))
    PECTORAL_MUSCLE_CLASS_ID: int = 0
    BREAST_TISSUE_CLASS_ID: int = 1
    NIPPLE_CLASS_ID: int = 2

    model_name: str = "yolo26m-seg.pt"
    weights_path: Optional[Path] = field(
        default_factory=lambda: Path(os.environ["BREAST_SEG_SEG_WEIGHTS"]) if os.environ.get("BREAST_SEG_SEG_WEIGHTS") else None
    )
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.5
    device: str = field(default_factory=_default_device)
    output_dir: Path = Path("outputs")

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def analysis_output_dir(self) -> Path:
        return self.output_dir / "analysis"

    def ensure_output_dirs(self) -> None:
        """Create the output directories if they do not exist."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_output_dir.mkdir(parents=True, exist_ok=True)


# ── YAML machinery ────────────────────────────────────────────────────────────────────────────────────────────


class ConfigError(ValueError):
    """A configuration file is malformed: unknown key, wrong type, missing value, bad placeholder or base cycle."""


T = TypeVar("T")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge: nested mappings merge key by key, every other value (lists included) is replaced."""
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def read_yaml(path: Union[str, Path], _seen: Tuple[Path, ...] = ()) -> Dict[str, Any]:
    """Read a YAML mapping and resolve its ``base:`` chain.

    ``base`` is a path (or a list of paths, merged left to right) relative to the file that names it. The child's
    keys override the base's; nested mappings merge, lists are replaced.
    """
    path = Path(path).resolve()
    if path in _seen:
        raise ConfigError(f"base cycle: {' -> '.join(p.name for p in (*_seen, path))}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path.name}: top level must be a mapping")
    data = dict(data)
    bases = data.pop("base", None)
    if bases is None:
        return data
    merged: Dict[str, Any] = {}
    for base in [bases] if isinstance(bases, str) else list(bases):
        merged = merge(merged, read_yaml(path.parent / str(base), (*_seen, path)))
    return merge(merged, data)


def parse_override(text: str) -> Tuple[str, Any]:
    """``"hyp.epochs=300"`` -> ``("hyp.epochs", 300)``; the value is parsed as YAML (``null``, lists, numbers...)."""
    if "=" not in text:
        raise ConfigError(f"override {text!r} is not of the form key.path=value")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"override {text!r} has an empty key")
    return key, yaml.safe_load(raw) if raw.strip() else ""


def set_dotted(data: Dict[str, Any], key: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``key == "a.b"``, creating intermediate mappings."""
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise ConfigError(f"override {key!r}: {part!r} is not a mapping")
        node = child
    node[parts[-1]] = value


def apply_overrides(data: Mapping[str, Any], overrides: Union[None, Mapping[str, Any], Iterable[str]]) -> Dict[str, Any]:
    """Return a copy of ``data`` with dotted overrides applied (a mapping or ``key=value`` strings)."""
    import copy

    out = copy.deepcopy(dict(data))
    if not overrides:
        return out
    items = overrides.items() if isinstance(overrides, Mapping) else (parse_override(o) for o in overrides)
    for key, value in items:
        set_dotted(out, key, value)
    return out


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp).replace("typing.", "")


def _convert(tp: Any, value: Any, where: str) -> Any:
    """Check / convert ``value`` against the annotation ``tp``; raises :class:`ConfigError` with the dotted path."""
    if tp is Any:
        return value
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is Union:
        if value is None and type(None) in args:
            return None
        errors = []
        for arm in args:
            if arm is type(None):
                continue
            try:
                return _convert(arm, value, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(errors[0] if len(errors) == 1 else f"{where}: {value!r} matches none of {_type_name(tp)}")
    if value is None:
        raise ConfigError(f"{where}: null is not allowed (expected {_type_name(tp)})")
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping for {tp.__name__}, got {type(value).__name__}")
        return from_dict(tp, value, where)
    if origin in (list, List, tuple, Tuple) or tp in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if origin in (tuple, Tuple) and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
            return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        item = args[0] if args else Any
        items = [_convert(item, v, f"{where}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin in (tuple, Tuple) else items
    if origin in (dict, Dict) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        kt, vt = args if args else (Any, Any)
        return {_convert(kt, k, f"{where}.<key>"): _convert(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return str(value)
    if tp is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{where}: expected a path, got {value!r}")
        return Path(value)
    return value


def from_dict(cls: Type[T], data: Mapping[str, Any], where: str = "") -> T:
    """Build the dataclass ``cls`` from ``data`` strictly: unknown keys and wrong types raise :class:`ConfigError`.

    Missing keys take the dataclass defaults; a field without a default is required. ``__post_init__`` runs as
    usual, so dataclasses validate their own values there.
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        prefix = f"{where}." if where else ""
        raise ConfigError(f"unknown key(s) {', '.join(prefix + str(k) for k in unknown)} "
                          f"(allowed: {', '.join(sorted(fields))})")
    kwargs = {}
    for name, f in fields.items():
        key = f"{where}.{name}" if where else name
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], key)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
            raise ConfigError(f"missing required key {key}")
    try:
        return cls(**kwargs)
    except ConfigError:
        raise
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where or cls.__name__}: {exc}") from exc


def to_dict(obj: Any) -> Any:
    """Plain YAML-serialisable view of a (nested) dataclass."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_config(cls: Type[T], path: Union[str, Path], overrides: Union[None, Mapping[str, Any], Iterable[str]] = None) -> T:
    """Read ``path`` (with ``base:``), apply ``overrides`` and build ``cls`` strictly."""
    data = apply_overrides(read_yaml(path), overrides)
    try:
        return from_dict(cls, data)
    except ConfigError as exc:
        raise ConfigError(f"{Path(path).name}: {exc}") from exc


# ── Local paths ───────────────────────────────────────────────────────────────────────────────────────────────

PATHS_ENV = "BREAST_SEG_PATHS"
DEFAULT_PATHS_FILE = Path("configs") / "paths.yaml"


@dataclass
class PathsConfig:
    """Machine-specific locations, read from a git-ignored ``configs/paths.yaml``.

    Committed configs write ``${name}`` placeholders; :meth:`resolve` substitutes the fields below and the entries
    of ``extra`` (named inputs such as label tables or a fold-assignment CSV). Relative values are resolved
    against ``root``, which itself is relative to the paths file.
    """

    root: str = "."
    data_root: str = "data"
    runs_dir: str = "runs"
    weights_dir: str = "weights"
    predictions_dir: str = "outputs/predictions"
    results_dir: str = "outputs/results"
    nnunet_raw: str = "data/nnunet/raw"
    nnunet_preprocessed: str = "data/nnunet/preprocessed"
    nnunet_results: str = "data/nnunet/results"
    extra: Dict[str, str] = field(default_factory=dict)
    source: Optional[Path] = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        clash = set(self.extra) & {f.name for f in dataclasses.fields(self)}
        if clash:
            raise ConfigError(f"paths.extra may not redefine {sorted(clash)}")

    @classmethod
    def load(cls, path: Union[None, str, Path] = None) -> PathsConfig:
        """Load the paths file: ``path``, else ``$BREAST_SEG_PATHS``, else ``configs/paths.yaml`` in the working
        directory. Without any file every location defaults to a folder under the working directory."""
        candidate = path or os.environ.get(PATHS_ENV) or (DEFAULT_PATHS_FILE if DEFAULT_PATHS_FILE.is_file() else None)
        if candidate is None:
            return cls(root=str(Path.cwd()))
        candidate = Path(candidate).resolve()
        cfg = load_config(cls, candidate)
        root = Path(cfg.root)
        cfg.root = str(root if root.is_absolute() else (candidate.parent / root).resolve())
        cfg.source = candidate
        return cfg

    def variables(self) -> Dict[str, str]:
        """Every placeholder name -> raw value."""
        names = [f.name for f in dataclasses.fields(self) if f.name not in ("extra", "source")]
        return {**{n: getattr(self, n) for n in names}, **self.extra}

    def expand(self, value: str, extra: Optional[Mapping[str, str]] = None) -> str:
        """Substitute ``${name}`` placeholders (recursively). ``extra`` adds call-local names."""
        table = {**self.variables(), **(extra or {})}

        def sub(text: str, depth: int) -> str:
            if depth > 20:
                raise ConfigError(f"placeholder recursion too deep in {value!r}")

            def one(m: re.Match) -> str:
                name = m.group(1)
                if name not in table:
                    raise ConfigError(f"unknown placeholder ${{{name}}} in {value!r} "
                                      f"(define it in the paths file under extra:)")
                return sub(str(table[name]), depth + 1)

            return _PLACEHOLDER.sub(one, text)

        return sub(str(value), 0)

    def resolve(self, value: Union[str, Path], extra: Optional[Mapping[str, str]] = None) -> Path:
        """Expand placeholders and make the result absolute against ``root``."""
        p = Path(self.expand(str(value), extra)).expanduser()
        return p if p.is_absolute() else (Path(self.root) / p).resolve()

    def dir(self, name: str) -> Path:
        """Resolved location of a named entry (``"runs_dir"``, ``"weights_dir"``, an ``extra`` key...)."""
        return self.resolve(f"${{{name}}}")
