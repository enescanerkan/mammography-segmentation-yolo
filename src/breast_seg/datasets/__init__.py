"""Dataset generations: label primitives, clean-up, study-level splits, the config-driven builder and exporters.

    from breast_seg.datasets import DatasetBuilder, load_dataset_config
    DatasetBuilder(load_dataset_config("configs/datasets/seg_v5.yaml")).build()
"""
from breast_seg.datasets.builder import BuildReport, DatasetAuditError, DatasetBuilder, build_dataset
from breast_seg.datasets.config import DatasetConfig, load_dataset_config
from breast_seg.datasets.formats import NnUNetExporter, SemanticExporter, write_yolo_data_yaml
from breast_seg.datasets.labels import (
    masks_to_lines,
    parse_polygons,
    polygons_to_masks,
    pose_label_line,
    rasterise,
    read_polygons,
)
from breast_seg.datasets.splits import FoldAssignment, FoldSplit, LeakageError, find_leaks, folds_from_pairs, split_groups

__all__ = [
    "BuildReport",
    "DatasetAuditError",
    "DatasetBuilder",
    "DatasetConfig",
    "FoldAssignment",
    "FoldSplit",
    "LeakageError",
    "NnUNetExporter",
    "SemanticExporter",
    "build_dataset",
    "find_leaks",
    "folds_from_pairs",
    "load_dataset_config",
    "masks_to_lines",
    "parse_polygons",
    "polygons_to_masks",
    "pose_label_line",
    "rasterise",
    "read_polygons",
    "split_groups",
    "write_yolo_data_yaml",
]
