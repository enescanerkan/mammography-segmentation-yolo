"""Typed configuration of the positioning review app (``configs/app.example.yaml``).

Relative paths are resolved against the directory of the YAML file they were read from (the working directory
when no file is given), so a config can sit next to its registry and data folder without naming a machine path.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

import yaml

from ...positioning.config import (
    DEFAULT_CRITERIA,
    DEFAULT_SEVERE_MM,
    DEFAULT_THRESHOLD_MM,
    SEVERE_BOUNDS_MM,
    THRESHOLD_BOUNDS_MM,
    Rules,
    default_criteria,
)
from ...positioning.report import LANGUAGES


class ConfigError(ValueError):
    """The application config is invalid."""


@dataclass
class ModelsSection:
    registry: str = "configs/models.yaml"  # model registry YAML
    weights_root: Optional[str] = None  # overrides $BREAST_SEG_WEIGHTS and the registry's own weights_root
    device: str = "cpu"  # "cpu", "0", "cuda:1", ...

    def __post_init__(self) -> None:
        self.device = str(self.device)  # a YAML "0" arrives as an int


@dataclass
class StorageSection:
    data_dir: str = "data"  # study renders + label JSON live in <data_dir>/studies/<id>/
    retention_days: int = 30  # studies older than this are deleted; 0 disables automatic deletion
    purge_every_s: int = 3600  # how often requests may trigger the retention purge


@dataclass
class UploadSection:
    max_file_mb: int = 220
    max_files: int = 40  # a folder upload carries more than the four views


@dataclass
class DisplaySection:
    # Pixel size of the PNG kept for display. The measurement frame stays 640 (what the checkpoints were trained
    # on); this only controls how much detail a reviewer can zoom into (640 ≈ 0.3 MB, 1280 ≈ 1.4 MB per study).
    preview_size: int = 1280


@dataclass
class RulesSection:
    threshold_mm: float = DEFAULT_THRESHOLD_MM  # |PNL_MLO - D_CC| at or below this is adequate
    severe_mm: float = DEFAULT_SEVERE_MM  # beyond this the pair is clearly inadequate

    def to_rules(self) -> Rules:
        return Rules(threshold_mm=float(self.threshold_mm), severe_mm=float(self.severe_mm))


@dataclass
class ServerSection:
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self) -> None:
        self.host, self.port = str(self.host), int(self.port)


@dataclass
class AppConfig:
    models: ModelsSection = field(default_factory=ModelsSection)
    storage: StorageSection = field(default_factory=StorageSection)
    upload: UploadSection = field(default_factory=UploadSection)
    display: DisplaySection = field(default_factory=DisplaySection)
    rules: RulesSection = field(default_factory=RulesSection)
    criteria: Dict[str, Dict[str, Any]] = field(default_factory=default_criteria)
    server: ServerSection = field(default_factory=ServerSection)
    title: str = "Mammography Positioning Review"
    language: str = "en"  # interface and report default: en | tr
    base_dir: Path = field(default_factory=Path.cwd)  # not read from YAML: where relative paths are anchored

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.language not in LANGUAGES:
            raise ConfigError(f"language must be one of {LANGUAGES}, got {self.language!r}")
        low, high = THRESHOLD_BOUNDS_MM
        if not low <= float(self.rules.threshold_mm) <= high:
            raise ConfigError(f"rules.threshold_mm must be within {THRESHOLD_BOUNDS_MM}")
        low, high = SEVERE_BOUNDS_MM
        if not low <= float(self.rules.severe_mm) <= high:
            raise ConfigError(f"rules.severe_mm must be within {SEVERE_BOUNDS_MM}")
        unknown = set(self.criteria) - set(DEFAULT_CRITERIA)
        if unknown:
            raise ConfigError(f"criteria: unknown keys {sorted(unknown)}")
        if self.display.preview_size < 64 or self.display.preview_size > 4096:
            raise ConfigError("display.preview_size must be within 64..4096")
        if self.upload.max_files < 1 or self.upload.max_file_mb < 1:
            raise ConfigError("upload limits must be positive")

    # ── resolved paths ───────────────────────────────────────────────────
    def resolve(self, value: Union[str, Path]) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.base_dir / path)

    @property
    def registry_path(self) -> Path:
        return self.resolve(self.models.registry)

    @property
    def weights_root(self) -> Optional[Path]:
        return self.resolve(self.models.weights_root) if self.models.weights_root else None

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.storage.data_dir)


def _section(cls: Any, data: Any) -> Any:
    """One flat config section from a mapping, rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(f"{cls.__name__}: expected a mapping")
    unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ConfigError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def _set(data: Dict[str, Any], dotted: str, value: str) -> None:
    *head, last = dotted.split(".")
    for part in head:
        data = data.setdefault(part, {})
    data[last] = yaml.safe_load(value)


def app_config_from_dict(data: Optional[Dict[str, Any]], base_dir: Optional[Union[str, Path]] = None) -> AppConfig:
    """Build an :class:`AppConfig`; ``criteria`` entries override the default bands key by key."""
    data = dict(data or {})
    criteria = default_criteria()
    for key, band in (data.pop("criteria", None) or {}).items():
        criteria[key] = dict(band)
    hints = get_type_hints(AppConfig)
    allowed = {f.name for f in dataclasses.fields(AppConfig)} - {"base_dir", "criteria"}
    sections: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in allowed:
            raise ConfigError(f"AppConfig: unknown key {key!r}")
        kind = hints[key]
        sections[key] = _section(kind, value) if dataclasses.is_dataclass(kind) else value
    try:
        return AppConfig(**sections, criteria=criteria, base_dir=Path(base_dir) if base_dir else Path.cwd())
    except TypeError as exc:
        raise ConfigError(str(exc)) from None


def load_app_config(path: Optional[Union[str, Path]] = None, overrides: Optional[List[str]] = None) -> AppConfig:
    """Read the YAML at ``path`` (defaults only when None) and apply ``dotted.key=value`` overrides."""
    data: Dict[str, Any] = {}
    base_dir: Optional[Path] = None
    if path:
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base_dir = path.resolve().parent
    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(f"override {override!r} is not key=value")
        key, value = override.split("=", 1)
        _set(data, key.strip(), value)
    return app_config_from_dict(data, base_dir)
