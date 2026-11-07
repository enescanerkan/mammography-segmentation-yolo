"""Typed configuration of a dataset generation (``configs/datasets/*.yaml``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Union

from breast_seg.config import YOLO_CLASSES, ConfigError, load_config

SOURCE_TYPES = ("yolo_dir", "yolo_layout", "manifest")
SPLIT_RULES = ("map", "ratio", "inherit", "train", "val", "test")
CORRECTIONS = ("mlo", "cc", "none", "auto")
VIEWS = ("MLO", "CC")


@dataclass
class GroupMapConfig:
    """A table mapping an item key (e.g. an image UID) to a group (e.g. its study)."""

    csv: List[str] = field(default_factory=list)
    key: str = ""
    group: str = ""
    namespace: str = "study"

    def __post_init__(self) -> None:
        if not self.csv or not self.key or not self.group:
            raise ConfigError("groups[]: csv, key and group are required")


@dataclass
class SplitMapConfig:
    """Split taken from a column of one or more tables (item key -> split value)."""

    csv: List[str] = field(default_factory=list)
    key: str = ""
    column: str = "Split"
    values: Dict[str, str] = field(default_factory=lambda: {"Train": "train", "Validation": "val", "Test": "test"})

    def __post_init__(self) -> None:
        if not self.csv or not self.key:
            raise ConfigError("split_map: csv and key are required")


@dataclass
class ExcludeConfig:
    """Drop items whose key is listed in a table column or a plain text file (one key per line)."""

    csv: Optional[str] = None
    column: str = "stem"
    list_file: Optional[str] = None
    where: Dict[str, List[str]] = field(default_factory=dict)
    drop_same_group: bool = False
    require_group: bool = False

    def __post_init__(self) -> None:
        if not self.csv and not self.list_file and not self.require_group:
            raise ConfigError("exclude[]: give csv, list_file or require_group")


@dataclass
class ManifestColumns:
    """Column names of a manifest source."""

    key: str = "id"
    view: str = "view"
    group: str = "group"
    source: str = "source"
    bucket: str = "bucket"


@dataclass
class SourceConfig:
    """One input of the builder.

    ``yolo_dir``: flat ``images`` / ``labels`` folders, stem = ``prefix + key``. ``yolo_layout``: an existing YOLO
    dataset (``images/<split>``), its split kept with ``split: inherit``. ``manifest``: a CSV whose rows name the
    image and label through ``image_template`` / ``label_template`` (``{root}``, ``{key}``, ``{source}``,
    ``{bucket}``, ``{view}`` and any column).
    """

    name: str = ""
    type: str = "yolo_dir"
    split: str = "map"
    view: str = "MLO"
    corrections: str = "none"
    strict_classes: bool = False
    oversample: int = 1
    exclude: List[ExcludeConfig] = field(default_factory=list)
    # yolo_dir
    images: str = ""
    labels: str = ""
    prefix: str = ""
    image_ext: str = ".png"
    # yolo_layout
    root: str = ""
    view_prefixes: Dict[str, str] = field(default_factory=lambda: {"cc_": "CC"})
    # manifest
    manifest: str = ""
    columns: ManifestColumns = field(default_factory=ManifestColumns)
    include: Dict[str, List[str]] = field(default_factory=dict)
    image_template: str = ""
    label_template: str = ""
    stem_template: str = "{source}_{key}"
    stem_by_source: Dict[str, str] = field(default_factory=dict)
    view_prefix: Dict[str, str] = field(default_factory=lambda: {"CC": "cc_"})
    group_namespace: str = "patient"
    group_namespace_by_source: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        where = f"sources[{self.name or '?'}]"
        if not self.name:
            raise ConfigError("sources[]: name is required")
        if self.type not in SOURCE_TYPES:
            raise ConfigError(f"{where}.type must be one of {SOURCE_TYPES}")
        if self.split not in SPLIT_RULES:
            raise ConfigError(f"{where}.split must be one of {SPLIT_RULES}")
        if self.corrections not in CORRECTIONS:
            raise ConfigError(f"{where}.corrections must be one of {CORRECTIONS}")
        if self.view not in VIEWS:
            raise ConfigError(f"{where}.view must be MLO or CC")
        if self.oversample < 1:
            raise ConfigError(f"{where}.oversample must be >= 1")
        need = {"yolo_dir": ("images", "labels"), "yolo_layout": ("root",),
                "manifest": ("manifest", "image_template", "label_template")}[self.type]
        missing = [k for k in need if not getattr(self, k)]
        if missing:
            raise ConfigError(f"{where}: {self.type} source needs {missing}")
        if self.split == "inherit" and self.type != "yolo_layout":
            raise ConfigError(f"{where}: split 'inherit' only works for a yolo_layout source")


@dataclass
class CorrectionConfig:
    """Clean-up and polygonisation parameters (see :mod:`breast_seg.datasets.corrections`)."""

    pectoral_close: int = 9
    gap_kernel: int = 41
    min_contour_area: float = 20.0
    approx_epsilon: float = 0.002


@dataclass
class AuditConfig:
    """Checks run after writing; any failure aborts before ``data.yaml`` is written."""

    image_shape: Optional[List[int]] = field(default_factory=lambda: [640, 640])
    unique_pixels: bool = True
    require_breast: bool = True
    forbid_cc_pectoral: bool = True
    required_classes: Dict[str, List[int]] = field(default_factory=lambda: {"MLO": [0, 1, 2], "CC": [1, 2]})
    leakage: bool = True


@dataclass
class SemanticExportConfig:
    enabled: bool = False
    output_dir: str = ""


@dataclass
class NnUNetExportConfig:
    enabled: bool = False
    dataset_id: int = 0
    dataset_name: str = ""
    raw_dir: str = "${nnunet_raw}"
    preprocessed_dir: str = "${nnunet_preprocessed}"
    cv_folds: int = 0
    cv_seed: int = 42
    description: str = ""

    def __post_init__(self) -> None:
        if self.enabled and (self.dataset_id <= 0 or not self.dataset_name):
            raise ConfigError("exports.nnunet: dataset_id and dataset_name are required when enabled")


@dataclass
class ExportsConfig:
    semantic: SemanticExportConfig = field(default_factory=SemanticExportConfig)
    nnunet: NnUNetExportConfig = field(default_factory=NnUNetExportConfig)

    def __post_init__(self) -> None:
        if self.nnunet.enabled and not self.semantic.enabled:
            raise ConfigError("exports.nnunet needs exports.semantic (the nnU-Net dataset is made from the label maps)")


@dataclass
class DatasetConfig:
    """A dataset generation: sources, split rule, clean-up, audit and the extra formats to export."""

    name: str = ""
    description: str = ""
    output_dir: str = ""
    class_names: List[str] = field(default_factory=lambda: list(YOLO_CLASSES.values()))
    sources: List[SourceConfig] = field(default_factory=list)
    split_map: Optional[SplitMapConfig] = None
    groups: List[GroupMapConfig] = field(default_factory=list)
    ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    ratio_seed: int = 42
    on_duplicate: str = "error"
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    exports: ExportsConfig = field(default_factory=ExportsConfig)

    def __post_init__(self) -> None:
        if not self.name or not self.output_dir:
            raise ConfigError("name and output_dir are required")
        if not self.sources:
            raise ConfigError("at least one source is required")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ConfigError("source names must be unique")
        if any(s.split == "map" for s in self.sources) and self.split_map is None:
            raise ConfigError("a source uses split 'map' but no split_map is configured")
        if self.on_duplicate not in ("error", "first"):
            raise ConfigError("on_duplicate must be 'error' or 'first'")
        if len(self.ratios) != 3 or abs(sum(self.ratios) - 1.0) > 1e-6:
            raise ConfigError("ratios must be three numbers summing to 1")


def load_dataset_config(path: Union[str, Path],
                        overrides: Union[None, Mapping[str, object], Iterable[str]] = None) -> DatasetConfig:
    """Load ``configs/datasets/<name>.yaml`` (with ``base:`` and dotted overrides)."""
    return load_config(DatasetConfig, path, overrides)
