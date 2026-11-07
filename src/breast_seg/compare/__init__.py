"""Pose-vs-segmentation comparison on the paired test split (see :mod:`breast_seg.compare.orchestrator`).

    from breast_seg.compare import CompareConfig, run_compare
"""
from breast_seg.compare.dataset import MammographyDataset, TestPair
from breast_seg.compare.evaluator import InferenceEngine, MetricsCalculator
from breast_seg.compare.geometry import GeometryUtils
from breast_seg.compare.models import ModelFactory, PoseModel, SegmentationRunner, wavelet_enhance
from breast_seg.compare.orchestrator import CompareConfig, PipelineOrchestrator, run_compare
from breast_seg.compare.visualizer import ResultVisualizer

__all__ = [
    "CompareConfig",
    "GeometryUtils",
    "InferenceEngine",
    "MammographyDataset",
    "MetricsCalculator",
    "ModelFactory",
    "PipelineOrchestrator",
    "PoseModel",
    "ResultVisualizer",
    "SegmentationRunner",
    "TestPair",
    "run_compare",
    "wavelet_enhance",
]
