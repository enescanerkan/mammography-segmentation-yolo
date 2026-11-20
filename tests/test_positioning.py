"""Positioning logic: both measurement routes, verdicts, criteria, label validation and the written report."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from breast_seg.positioning import (
    CRITERIA_VERSION,
    CriteriaGrader,
    LabelError,
    PositioningAssessor,
    Rules,
    apply_labels,
    build_report,
    clean_labels,
    clean_rules,
    compare,
    masks_from_polygons,
    measure_landmark,
    measure_mask,
    pair_verdict,
    summarise,
)
from breast_seg.positioning.criteria import mlo_alone, nipple_position


def _square(x1, y1, x2, y2):
    return {"points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "confidence": 0.9, "rescued": False}


def _image(view="MLO", laterality="L", chest_side="L", points=None, polygons=None, mm=0.2):
    if points is None:
        points = {"nipple": [300.0, 320.0]}
        if view == "MLO":
            points.update(pectoral_top=[120.0, 0.0], pectoral_bottom=[0.0, 400.0])
    if polygons is None:
        polygons = {"breast-tissue": _square(0, 40, 300, 600), "nipple": _square(292, 312, 300, 328)}
        if view == "MLO":
            polygons["pectoral"] = {"points": [[0, 40], [120, 40], [0, 400]], "confidence": 0.9}
    return {
        "view_key": f"{laterality}-{view}", "view": view, "laterality": laterality, "chest_side": chest_side,
        "mm_per_px": mm, "landmarks": {"detected": True, "points": points}, "polygons": polygons,
        "quality": {"sharpness": 900.0, "clipped_fraction": 0.001}, "physics": {},
    }


def _measured(image):
    image = copy.deepcopy(image)
    image["masks"] = masks_from_polygons(image["polygons"])
    image["measures"] = {"landmark": measure_landmark(image), "mask": measure_mask(image)}
    return image


# ── measures ───────────────────────────────────────────────────────────────
def test_landmark_route_cc_and_mlo():
    cc = measure_landmark(_image("CC"))
    assert cc["ok"] and cc["px"] == 300.0 and cc["mm"] == pytest.approx(60.0) and cc["to"] == [0.0, 320.0]
    cc_right = measure_landmark(_image("CC", chest_side="R"))
    assert cc_right["px"] == 340.0
    mlo = measure_landmark(_image("MLO"))
    assert mlo["ok"] and mlo["line"] == [[120.0, 0.0], [0.0, 400.0]] and mlo["mm"] > 0


def test_landmark_route_missing_points():
    assert measure_landmark(_image(points={}))["reason"] == "nipple keypoint not detected"
    assert measure_landmark(_image(points={"nipple": [1, 1]}))["reason"] == "pectoral keypoints not detected"


def test_mask_route():
    image = _image("MLO")
    image["masks"] = masks_from_polygons(image["polygons"])
    mask = measure_mask(image)
    assert mask["ok"] and len(mask["line"]) == 2 and mask["mm"] > 0
    image = _image("CC")
    image["masks"] = masks_from_polygons(image["polygons"])
    depth = measure_mask(image)
    assert depth["ok"] and depth["px"] == pytest.approx(296.0, abs=2)


@pytest.mark.parametrize("drop,reason", [("nipple", "no nipple mask"), ("breast-tissue", "no breast tissue mask"),
                                         ("pectoral", "no pectoral mask")])
def test_mask_route_reasons(drop, reason):
    image = _image("MLO")
    del image["polygons"][drop]
    image["masks"] = masks_from_polygons(image["polygons"])
    assert measure_mask(image) == {"ok": False, "reason": reason}


def test_masks_from_polygons_skips_degenerate_shapes():
    masks = masks_from_polygons({"a": {"points": [[0, 0], [1, 1]]}, "b": _square(10, 10, 20, 20), "c": None})
    assert list(masks) == ["b"] and masks["b"].shape == (640, 640) and masks["b"].sum() == 121


def test_pair_verdict():
    good = pair_verdict({"ok": True, "mm": 50.0}, {"ok": True, "mm": 44.0})
    assert good["verdict"] == "good" and good["delta_mm"] == 6.0 and not good["severe"]
    bad = pair_verdict({"ok": True, "mm": 50.0}, {"ok": True, "mm": 30.0}, threshold_mm=10, severe_mm=15)
    assert bad["verdict"] == "bad" and bad["severe"]
    assert pair_verdict(None, {"ok": True, "mm": 1})["reason"] == "needs both MLO and CC"
    assert pair_verdict({"ok": False, "reason": "x"}, {"ok": True, "mm": 1})["reason"] == "x"


def test_compare_states():
    a = {"ok": True, "delta_mm": 4.0, "verdict": "good"}
    assert compare(a, {**a, "delta_mm": 5.0})["state"] == "agree"
    wide = compare(a, {**a, "delta_mm": 9.9})
    assert wide["state"] == "agree" and wide["wide_gap"] and "differ by 5.9 mm" in wide["note"]
    assert compare(a, {**a, "verdict": "bad"})["state"] == "disagree"
    assert compare(a, {"ok": False})["state"] == "partial"
    assert compare({"ok": False}, {"ok": False})["state"] == "none"


def test_summarise_study():
    images = [_measured(_image(view, side, chest_side=side)) for view in ("MLO", "CC") for side in ("L", "R")]
    summary = summarise(images, threshold_mm=10.0)
    assert [side["laterality"] for side in summary["sides"]] == ["L", "R"]
    assert summary["overall"] in ("good", "bad")
    assert summarise([], 10.0) == {"sides": [], "overall": None, "flagged": False}


# ── criteria ───────────────────────────────────────────────────────────────
def test_grading_primitives():
    grader = CriteriaGrader({"h": {"good": 10, "acceptable": 5}})
    assert [grader.higher_is_better(v, "h") for v in (11, 7, 3)] == ["good", "acceptable", "poor"]
    assert grader.higher_is_better(1, "missing") == "na"


def test_criteria_for_mlo_and_cc():
    grader = CriteriaGrader()
    mlo = _measured(_image("MLO"))
    items = grader.for_image(mlo, mlo["masks"])
    assert [item["id"] for item in items] == ["pectoral_angle", "pectoral_depth", "mlo_alone"]
    assert items[0]["grade"] == "info" and items[1]["grade"] in ("good", "acceptable", "poor")
    assert items[2]["grade"] == "na" and "geometry" in items[2]["detail"]   # no frame transform in this fixture
    cc = _measured(_image("CC"))
    items = grader.for_image(cc, cc["masks"])
    assert [item["id"] for item in items] == ["nipple_centred"] and items[0]["grade"] == "info"
    assert items[0]["unit"] == "mm" and items[0]["value"] >= 0


def test_nipple_centred_is_the_offset_from_the_middle_of_the_breast_height():
    breast = np.zeros((640, 640), np.uint8)
    breast[100:501, 0:300] = 1                      # rows 100..500: middle at row 300, half-height 200 px
    item = CriteriaGrader._nipple_centred(breast, [250.0, 350.0], 0.5, "L-CC")
    assert item["grade"] == "info" and item["value"] == 25.0 and "25% of the half-height" in item["detail"]
    assert CriteriaGrader._nipple_centred(None, [1.0, 1.0], 0.5, "L-CC")["grade"] == "na"


def _mlo_alone(line, nipple=(300.0, 300.0), size=(640, 640)):
    transform = {"original_width": size[0], "original_height": size[1], "crop_x1": 0, "crop_y1": 0,
                 "crop_x2": size[0], "crop_y2": size[1], "scale": 1.0, "pad_left": 0, "pad_top": 0,
                 "pad_right": 0, "pad_bottom": 0, "pixel_spacing": 0.1}
    image = {"view": "MLO", "measures": {"mask": {"line": line}}, "transform": transform}
    return mlo_alone(image, list(nipple), "L-MLO")


def test_mlo_alone_foot_inside_or_outside_the_image():
    inside = _mlo_alone([[100.0, 0.0], [0.0, 100.0]])                   # foot (50, 50)
    assert (inside["grade"], inside["value"]) == ("info", "adequate") and "inside the image" in inside["detail"]
    outside = _mlo_alone([[100.0, 0.0], [0.0, 100.0]], nipple=(600.0, 20.0))  # foot (340, -240): 24 mm above the top
    assert (outside["grade"], outside["value"]) == ("info", "inadequate")
    assert outside["detail"].endswith("outside the image, 24.0 mm beyond the edge")


def test_mlo_alone_margin_is_a_quarter_percent_of_the_longest_side():
    # 0.25% of 640 px = 1.6 px: a foot 1 px outside still counts as inside, 2 px outside does not
    assert _mlo_alone([[0.0, -1.0], [640.0, -1.0]])["value"] == "adequate"
    assert _mlo_alone([[0.0, -2.0], [640.0, -2.0]])["value"] == "inadequate"
    assert _mlo_alone([[641.5, 0.0], [641.5, 640.0]])["value"] == "adequate"
    assert _mlo_alone([[641.7, 0.0], [641.7, 640.0]])["value"] == "inadequate"


def test_mlo_alone_needs_a_line_a_nipple_and_the_geometry():
    assert _mlo_alone(None)["grade"] == "na"
    assert mlo_alone({"view": "MLO", "measures": {"mask": {"line": [[0, 0], [1, 1]]}}}, None, "L-MLO")["grade"] == "na"
    assert mlo_alone({"view": "MLO", "measures": {"mask": {"line": [[0, 0], [1, 1]]}}}, [5, 5], "L-MLO")["grade"] == "na"


def test_a_nipple_mask_under_30_pixels_falls_back_to_the_pose_point():
    mask = np.zeros((640, 640), np.uint8)
    mask[100:105, 100:105] = 1                                            # 25 px
    image = {"landmarks": {"points": {"nipple": [300.0, 320.0]}}}
    assert nipple_position(image, mask, 30) == [300.0, 320.0]
    mask[100:106, 100:106] = 1                                            # 36 px
    assert nipple_position(image, mask, 30) == pytest.approx([102.5, 102.5])


def test_header_gantry_is_shown_and_compression_is_not():
    grader = CriteriaGrader()
    image = _measured(_image("CC"))
    image["physics"] = {"compression_n": 100.0, "thickness_mm": 50.0, "gantry_deg": 45.0, "entrance_dose_mgy": 1.2}
    items = grader.for_image(image, image["masks"])
    assert [item["id"] for item in items] == ["nipple_centred", "gantry"]
    assert items[1]["grade"] == "info" and items[1]["value"] == 45.0


def test_no_study_level_criteria():
    images = [_measured(_image("MLO", "L")), _measured(_image("MLO", "R"))]
    assert CriteriaGrader().for_study(images) == []


def test_counts_ignore_estimates_and_information():
    items = [{"grade": "good"}, {"grade": "poor"}, {"grade": "poor", "estimated": True}, {"grade": "na"},
             {"grade": "info"}]
    assert CriteriaGrader.summarise(items) == {"counts": {"good": 1, "acceptable": 0, "poor": 1, "na": 1}, "graded": 2}


# ── labels and rules ───────────────────────────────────────────────────────
def test_clean_labels():
    cleaned = clean_labels({"images": [
        {"view_key": "L-MLO",
         "polygons": {"nipple": {"points": [[1, 1], [9999, 5], [5, -3], ["x", 1]]}, "unknown": _square(0, 0, 9, 9)},
         "landmarks": {"points": {"nipple": [700, 2], "other": [1, 1], "pectoral_top": [1]}},
         "lesions": [{"id": "keep-me_1", "label": "Calcification", "box": [50, 60, 10, 20]},
                     {"id": "bad id!", "box": [0, 0, 1, 1]},
                     {"id": "../x", "box": [0, 0, 30, 30]}]},
        "garbage",
        {"view_key": 7},
    ]})
    entry = cleaned["L-MLO"]
    assert entry["polygons"]["nipple"]["points"] == [[1.0, 1.0], [640.0, 5.0], [5.0, 0.0]]
    assert "unknown" not in entry["polygons"]
    assert entry["points"] == {"nipple": [640.0, 2.0]}
    first, second = entry["lesions"]
    assert first == {"id": "keep-me_1", "label": "Calcification", "box": [10.0, 20.0, 50.0, 60.0],
                     "confidence": 1.0, "edited": True}
    assert second["id"] != "../x" and second["label"] == "Mass"


@pytest.mark.parametrize("payload,message", [
    (None, "Expected an object"),
    ({"images": []}, "No images"),
    ({"images": [{"view_key": "L-CC", "polygons": ["not", "a", "dict"]}]}, "'polygons' has the wrong type"),
    ({"images": [{"view_key": "L-CC", "lesions": {"a": 1}}]}, "'lesions' has the wrong type"),
])
def test_clean_labels_rejects_wrong_shapes(payload, message):
    with pytest.raises(LabelError, match=message):
        clean_labels(payload)


def test_apply_labels_marks_edits():
    record = {"images": [_image("CC", "L")]}
    assert apply_labels(record, {"R-CC": {"polygons": {}, "points": {}, "lesions": []}}) == 0 and "edited" not in record
    assert apply_labels(record, {"L-CC": {"polygons": {}, "points": {"nipple": [1.0, 2.0]}, "lesions": []}}) == 1
    landmarks = record["images"][0]["landmarks"]
    assert landmarks["edited"] and landmarks["detected"] and record["edited"] > 0


def test_clean_rules():
    current = {"threshold_mm": 10.0, "severe_mm": 15.0}
    assert clean_rules(current, {"threshold_mm": 20}) == {"threshold_mm": 20.0, "severe_mm": 20.0}
    assert clean_rules(current, {"severe_mm": None}) == current
    for bad in ({"threshold_mm": 0.5}, {"severe_mm": 500}, {"threshold_mm": "x"}, ["x"]):
        with pytest.raises(LabelError):
            clean_rules(current, bad)


# ── assessor and report ────────────────────────────────────────────────────
def _record(threshold=10.0):
    assessor = PositioningAssessor(Rules(threshold_mm=threshold))
    images = []
    for view in ("MLO", "CC"):
        for side in ("L", "R"):
            meta = {"view": view, "laterality": side, "chest_side": side, "pixel_spacing_known": True,
                    "manufacturer": "Phantom", "model_name": "", "original_width": 900, "original_height": 1200}
            base = _image(view, side, chest_side=side)
            analysis = {"mm_per_px": 0.2, "landmarks": base["landmarks"], "polygons": base["polygons"],
                        "lesions": [], "quality": base["quality"]}
            images.append(assessor.build_image(f"{side}-{view}", meta, analysis))
    return assessor, assessor.new_record("abc123", images, ["x.txt: skipped — a .txt file, not an image"],
                                         {"seg_v4": "best.pt"}, 1280, created=0.0)


def test_assessor_builds_a_complete_record():
    _, record = _record()
    assert record["rules"] == {"threshold_mm": 10.0, "severe_mm": 15.0}
    assert len(record["summary"]["sides"]) == 2
    assert {"study", "summary", "version"} == set(record["criteria"])
    for image in record["images"]:
        assert "masks" not in image and set(image["measures"]) == {"landmark", "mask"}
        assert image["criteria"]


def test_moving_the_threshold_redecides():
    assessor, record = _record()
    record["rules"]["threshold_mm"] = 1.0
    record["rules"]["severe_mm"] = 1.0
    assessor.recompute(record)
    verdicts = {side["landmark"]["verdict"] for side in record["summary"]["sides"] if side["landmark"]["ok"]}
    assert verdicts <= {"good", "bad"} and record["rules"]["threshold_mm"] == 1.0


def test_report_languages():
    _, record = _record()
    english = build_report(record, "en")
    turkish = build_report(record, "tr")
    assert english["lang"] == "en" and english["title"] == "Mammography Positioning Report"
    assert turkish["title"] == "Mamografi Pozisyonlama Raporu"
    assert build_report(record, "xx")["lang"] == "en"
    assert [s["heading"] for s in english["sections"]][:4] == [
        "Mammography Positioning Report", "Verdict", "Positioning criteria", "Conclusion"]
    assert english["sections"][-1]["heading"] == "Upload notes"
    assert "ΔD threshold" in english["text"] and "10 mm" in english["text"]
    # the numbers are identical across languages
    delta = record["summary"]["sides"][0]["landmark"]["delta_mm"]
    assert f"ΔD {delta:.1f} mm" in english["text"] and f"ΔD {delta:.1f} mm" in turkish["text"]


def test_report_translates_missing_reasons():
    _, record = _record()
    record["summary"]["sides"][0]["landmark"] = {"ok": False, "reason": "nipple keypoint not detected"}
    assert "meme ucu noktası bulunamadı" in build_report(record, "tr")["text"]
    assert "nipple keypoint not detected" in build_report(record, "en")["text"]


def test_report_conclusion_names_the_failing_route():
    _, record = _record()
    side = record["summary"]["sides"][0]
    side["mask"] = {**side["mask"], "ok": True, "verdict": "bad", "delta_mm": 12.3, "threshold_mm": 10.0,
                    "pnl_mm": 40.0, "cc_mm": 27.7}
    text = build_report(record, "en")["text"]
    assert "Left breast (Segmentation): ΔD 12.3 mm > 10 mm — inadequate." in text


def _set_criterion(record, view_key, criterion_id, **fields):
    image = next(image for image in record["images"] if image["view_key"] == view_key)
    entry = next(entry for entry in image["criteria"] if entry["id"] == criterion_id)
    entry.update(fields)


def test_poor_pectoral_depth_asks_for_a_repeat():
    _, record = _record()
    _set_criterion(record, "L-MLO", "pectoral_depth", grade="poor", value=-16.6)
    text = build_report(record, "en")["text"]
    assert "A repeat L-MLO is advised." in text and "Pectoral reaches nipple level (-16.6mm)" in text


def test_information_values_never_ask_for_a_repeat():
    _, record = _record()
    for image in record["images"]:
        for entry in image["criteria"]:
            if entry["grade"] != "info":
                entry["grade"] = "good"
    _set_criterion(record, "L-MLO", "pectoral_angle", value=70.0)   # far outside any guidance band, still info
    report = build_report(record, "en")
    text = report["text"]
    assert "repeat" not in text and "70.0°" in text and "70.0° ·" not in text  # value only, no grade word


def test_regrade_rebuilds_criteria_only():
    assessor, record = _record()
    record["criteria"]["version"] = 1
    record["images"][0]["criteria"] = [{"id": "sharpness", "name": "Focus", "grade": "poor", "value": 1.0}]
    record["images"][0]["measures"]["landmark"]["mm"] = 999.0       # a stored number a re-measure would change
    summary, rules = copy.deepcopy(record["summary"]), copy.deepcopy(record["rules"])
    assessor.regrade(record)
    assert record["criteria"]["version"] == CRITERIA_VERSION
    assert [entry["id"] for entry in record["images"][0]["criteria"]] == ["pectoral_angle", "pectoral_depth",
                                                                         "mlo_alone"]
    assert record["images"][0]["measures"]["landmark"]["mm"] == 999.0
    assert record["summary"] == summary and record["rules"] == rules


def test_masks_are_uint8_binary():
    masks = masks_from_polygons({"x": _square(0, 0, 3, 3)})
    assert masks["x"].dtype == np.uint8 and set(np.unique(masks["x"])) == {0, 1}
