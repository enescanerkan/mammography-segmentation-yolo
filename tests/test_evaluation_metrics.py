"""Evaluation library on hand-made masks, landmarks and decisions."""
import math

import numpy as np
import pandas as pd
import pytest

from breast_seg.evaluation import (
    FLAT_RULE,
    TOLERANCE_RULE,
    CanvasTransform,
    ImageGeometry,
    MeasureOptions,
    PNLRule,
    PooledDice,
    centroid,
    chest_wall_mm,
    decision_summary,
    dice_iou,
    group_bootstrap,
    instances_to_label_map,
    mcnemar_exact,
    measure_cc,
    measure_mlo,
    merge_oof_predictions,
    paired_mcnemar,
    pnl_mm,
    point_line_distance,
)
from breast_seg.evaluation import measure as measure_mod
from breast_seg.evaluation.fallback import farthest_point, tissue_contour
from breast_seg.evaluation.predictions import write_label_map
from breast_seg.evaluation.stats import binomial_two_sided_p


def test_dice_iou_hand_made():
    gt = np.zeros((10, 10), np.uint8)
    pred = np.zeros_like(gt)
    gt[0:4, 0:5] = 2           # 20 px breast
    pred[0:4, 0:4] = 2         # 16 px, all inside
    gt[8, 8] = 3               # nipple only in the reference
    r = dice_iou(pred, gt)
    assert r["dice_breast-tissue"] == pytest.approx(2 * 16 / 36)
    assert r["iou_breast-tissue"] == pytest.approx(16 / 20)
    assert r["dice_nipple"] == 0.0
    assert math.isnan(r["dice_pectoral"])        # absent from both -> no evidence
    with pytest.raises(ValueError):
        dice_iou(pred[:5], gt)


def test_pooled_dice():
    acc = PooledDice(4)
    a = np.array([[0, 1], [2, 3]])
    acc.update(a, a)
    acc.update(np.array([[0, 0], [2, 3]]), a)
    r = acc.result()
    assert r["pectoral"] == pytest.approx(2 / 3)
    assert r["nipple"] == 1.0 and r["mean_fg"] == pytest.approx((2 / 3 + 1 + 1) / 3)


def test_canvas_transform_round_trip_and_series_rows():
    row = pd.Series({"scale": 0.25, "crop_x1": 100, "crop_y1": 40, "pad_left": 12, "pad_top": 0, "pixel_spacing": 0.1,
                     "laterality": "r", "original_width": 2000})
    t = CanvasTransform.from_row(row)
    p = t.to_original((112, 10))
    np.testing.assert_allclose(p, [500, 80])
    np.testing.assert_allclose(t.to_canvas(p), [112, 10])
    g = ImageGeometry.from_row(row)
    assert g.laterality == "R" and g.width == 2000 and g.spacing_mm == 0.1


def test_distances():
    assert point_line_distance((0, 5), (0, 0), (10, 0)) == pytest.approx(5)
    assert point_line_distance((20, 5), (0, 0), (10, 0)) == pytest.approx(5)   # infinite line, no clamping
    assert pnl_mm((3, 4), (0, 0), (0, 0), 0.5) == pytest.approx(2.5)           # degenerate line
    assert chest_wall_mm((300, 0), "L", 1000, 0.1) == pytest.approx(30)
    assert chest_wall_mm((300, 0), "R", 1000, 0.1) == pytest.approx(70)


def test_centroid_minimum_size():
    m = np.zeros((20, 20), bool)
    m[2:4, 2:4] = True
    assert centroid(m, 30) is None
    np.testing.assert_allclose(centroid(m, 1), [2.5, 2.5])


def test_rule_flat_and_tolerance_with_unanswered_pairs():
    delta = [5.0, 10.5, 9.5, 20.0, np.nan, np.nan]
    good = [True, True, False, False, True, False]
    flat = FLAT_RULE.matrix(delta, good)
    assert (flat["TP"], flat["FN"], flat["FP"], flat["TN"], flat["U_bad"]) == (1, 2, 1, 1, 1)
    assert flat["accuracy"] == pytest.approx(100 * 2 / 6)
    assert flat["specificity"] == pytest.approx(100 * 1 / 3)
    tol = TOLERANCE_RULE.matrix(delta, good)
    # 10.5 is within the band for a good reference, 9.5 within the band for a bad one
    assert (tol["TP"], tol["TN"], tol["n_fail"]) == (2, 2, 2)
    assert tol["accuracy"] == pytest.approx(100 * 4 / 6)
    assert list(TOLERANCE_RULE.correct(delta, good)) == [True, True, True, True, False, False]
    assert FLAT_RULE.matrix(delta, good, n_total=8)["n_fail"] == 4
    assert list(FLAT_RULE.predict([3.0, 12.0, np.nan])) == ["good", "bad", ""]
    assert set(decision_summary(delta, good)) == {"flat_10mm", "tolerance_1mm"}
    assert PNLRule(10, 1).lo == 9 and PNLRule(10, 1).hi == 11


def test_mcnemar_exact():
    a = np.array([True] * 6 + [False] * 1 + [True] * 3)
    b = np.array([False] * 6 + [True] * 1 + [True] * 3)
    r = mcnemar_exact(a, b)
    assert (r.only_a, r.only_b) == (6, 1)
    assert r.p_value == pytest.approx(binomial_two_sided_p(1, 7))
    assert binomial_two_sided_p(0, 0) == 1.0 and binomial_two_sided_p(3, 6) == 1.0
    stats = pytest.importorskip("scipy.stats")
    for k, n in ((1, 7), (2, 20), (10, 25)):
        assert binomial_two_sided_p(k, n) == pytest.approx(stats.binomtest(k, n, 0.5).pvalue)


def test_paired_mcnemar_and_bootstrap():
    base = pd.DataFrame({"group": ["a", "a", "b", "c"], "side": ["L", "R", "L", "L"], "truth": ["good"] * 4})
    pa = base.assign(delta_mm=[1.0, 2.0, 3.0, 30.0])
    pb = base.assign(delta_mm=[20.0, 2.0, 3.0, 30.0])
    r = paired_mcnemar(pa, pb, lambda d: FLAT_RULE.correct(d.delta_mm, d.truth == "good"))
    assert (r.only_a, r.only_b) == (1, 0)
    lo, hi = group_bootstrap(pa, "group", lambda d: float((d.delta_mm <= 10).mean()), n_resamples=200, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    assert (lo, hi) == group_bootstrap(pa, "group", lambda d: float((d.delta_mm <= 10).mean()), n_resamples=200, seed=1)


def test_instances_to_label_map_rescue_and_cc():
    h = w = 8
    m = np.zeros((3, h, w), np.float32)
    m[0, :, :4] = 1    # breast instance, confident
    m[1, :2, :2] = 1   # pectoral, low confidence
    m[2, 4, 4] = 1     # nipple, low confidence
    cls, conf = np.array([1, 0, 2]), np.array([0.9, 0.05, 0.1])
    label, st = instances_to_label_map(m, cls, conf, (h, w), conf=0.25, floor=0.01)
    assert label[5, 1] == 2 and label[0, 0] == 1 and label[4, 4] == 3
    assert st.rescued[1] == 1 and st.rescued[3] == 1
    cc, st2 = instances_to_label_map(m, cls, conf, (h, w), conf=0.25, floor=0.01, is_cc=True)
    assert cc[0, 0] == 2 and st2.rescued[1] == 0       # no pectoral rescue on a CC view
    off, st3 = instances_to_label_map(m, cls, conf, (h, w), conf=0.25, floor=0.0)
    assert off[4, 4] == 0 and st3.dropped[3] == 1
    empty, st4 = instances_to_label_map(None, None, None, (h, w))
    assert not empty.any() and st4.empty == 1


def test_fallback_farthest_point():
    tissue = np.zeros((40, 40), np.uint8)
    tissue[5:35, 0:25] = 1
    pts = tissue_contour(tissue, min_px=10)
    far = farthest_point(pts, lambda q: q[0])
    assert far[0] == 24
    assert tissue_contour(tissue, min_px=10_000) is None


GEOM = ImageGeometry(CanvasTransform(scale=0.5, crop_x1=0, crop_y1=0, pad_left=0, pad_top=0), spacing_mm=0.1,
                     laterality="L", width=200)


def test_measure_mlo_with_known_line(monkeypatch):
    label = np.zeros((100, 100), np.uint8)
    label[40:50, 60:70] = 3                       # nipple centroid (64.5, 44.5) canvas
    monkeypatch.setattr(measure_mod, "_pectoral_line", lambda lab: (np.array([0.0, 0.0]), np.array([0.0, 100.0])))
    gt = {"nipple": np.array([129.0, 89.0]), "pec_top": np.array([0.0, 0.0]), "pec_bot": np.array([0.0, 200.0])}
    row = measure_mlo(label, GEOM, gt)
    assert row["pnl_mm"] == pytest.approx(129 * 0.1)
    assert row["gt_pnl_mm"] == pytest.approx(12.9) and row["e_pnl"] == pytest.approx(0, abs=1e-9)
    assert row["e_pec_bot"] == pytest.approx(0, abs=1e-9)
    # no nipple: reference still written, no measurement unless the fallback is on
    label[label == 3] = 0
    label[10:90, 0:80] = 2
    row = measure_mlo(label, GEOM, gt)
    assert "pnl_mm" not in row and "gt_pnl_mm" in row
    row = measure_mlo(label, GEOM, gt, MeasureOptions(fallback=True))
    assert row["fallback"] and row["pnl_mm"] == pytest.approx(79 * 2 * 0.1)


def test_measure_mlo_with_real_line_fit():
    label = np.zeros((128, 128), np.uint8)
    label[:, :] = 0
    label[0:120, 0:100] = 2
    for y in range(0, 80):
        label[y, : max(1, 60 - int(0.75 * y))] = 1
    label[60:66, 90:96] = 3
    row = measure_mlo(label, GEOM, None, MeasureOptions(min_nipple_px=10))
    assert row.get("pnl_mm", 0) > 0


def test_measure_cc_landmark_mode_and_right_side():
    geom = ImageGeometry(GEOM.transform, 0.1, "R", 200.0)
    row = measure_cc(np.zeros((100, 100), np.uint8), geom, np.array([150.0, 40.0]),
                     landmarks={"nipple": np.array([70.0, 20.0])})
    assert row["cc_mm"] == pytest.approx((200 - 140) * 0.1)
    assert row["gt_cc_mm"] == pytest.approx(5.0) and row["e_nipple"] == pytest.approx(1.0)


def test_merge_oof_predictions(tmp_path):
    f0, f1 = tmp_path / "f0", tmp_path / "f1"
    for d, stems in ((f0, ["a", "b", "x"]), (f1, ["c"])):
        for s in stems:
            write_label_map(d / f"{s}.png", np.zeros((4, 4), np.uint8))
    rep = merge_oof_predictions({0: f0, 1: f1}, tmp_path / "all", {"a": 0, "b": 0, "c": 1, "x": 1}, strict=False)
    assert rep.n_images == 3 and rep.foreign == ["x"] and rep.missing == ["x"]
    with pytest.raises(FileNotFoundError):
        merge_oof_predictions({0: f0, 1: f1}, tmp_path / "all", {"a": 0, "b": 0, "c": 1, "x": 1})
