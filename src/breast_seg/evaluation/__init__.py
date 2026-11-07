"""Evaluation: mask metrics, landmark distances, the PNL rule, paired comparison, out-of-fold aggregation and the
end-to-end scorer of a prediction directory. Pure numpy / pandas (no framework imports)."""
from breast_seg.evaluation.evaluator import EvaluationConfig, EvaluationResult, SegmentationEvaluator, load_evaluation_config
from breast_seg.evaluation.gt import GroundTruth, canvas_landmark_table
from breast_seg.evaluation.landmarks import (
    MIN_NIPPLE_PX,
    CanvasTransform,
    ImageGeometry,
    centroid,
    chest_wall_mm,
    distance_mm,
    pnl_mm,
    point_line_distance,
)
from breast_seg.evaluation.masks import PooledDice, dice_iou
from breast_seg.evaluation.measure import MeasureOptions, measure_cc, measure_mlo
from breast_seg.evaluation.oof import OOFReport, merge_oof_predictions
from breast_seg.evaluation.pairs import pairs_from_fold_table, pairs_from_labels, score_pairs
from breast_seg.evaluation.predictions import instances_to_label_map
from breast_seg.evaluation.rule import FLAT_RULE, THRESHOLD_MM, TOLERANCE_RULE, PNLRule, decision_summary
from breast_seg.evaluation.stats import McNemarResult, group_bootstrap, mcnemar_exact, paired_mcnemar

__all__ = [
    "FLAT_RULE",
    "MIN_NIPPLE_PX",
    "THRESHOLD_MM",
    "TOLERANCE_RULE",
    "CanvasTransform",
    "EvaluationConfig",
    "EvaluationResult",
    "GroundTruth",
    "ImageGeometry",
    "McNemarResult",
    "MeasureOptions",
    "OOFReport",
    "PNLRule",
    "PooledDice",
    "SegmentationEvaluator",
    "canvas_landmark_table",
    "centroid",
    "chest_wall_mm",
    "decision_summary",
    "dice_iou",
    "distance_mm",
    "group_bootstrap",
    "instances_to_label_map",
    "load_evaluation_config",
    "mcnemar_exact",
    "measure_cc",
    "measure_mlo",
    "merge_oof_predictions",
    "paired_mcnemar",
    "pairs_from_fold_table",
    "pairs_from_labels",
    "pnl_mm",
    "point_line_distance",
    "score_pairs",
]
