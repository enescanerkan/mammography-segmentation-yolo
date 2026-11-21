"""Positioning logic: both measurement routes, verdicts, criteria, label validation and the written report."""
from __future__ import annotations

import copy
import re
from pathlib import Path

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
from breast_seg.positioning.criteria import mlo_quality, nipple_position


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
    result = measure_mask(image)
    assert (result["ok"], result["reason"]) == (False, reason)
    # the pectoral line needs only the pectoral and the breast: it is returned although the distance is not measured
    assert ("line" in result) == (drop == "nipple")


def test_mlo_criteria_fall_back_to_the_pose_nipple_when_the_nipple_mask_is_missing():
    assessor = PositioningAssessor(Rules())
    meta = {"view": "MLO", "laterality": "L", "chest_side": "L", "pixel_spacing_known": True,
            "manufacturer": "Phantom", "model_name": "", "original_width": 640, "original_height": 640}
    base = _image("MLO", "L", chest_side="L")
    del base["polygons"]["nipple"]                                 # segmentation found no nipple at all
    transform = {"original_width": 640, "original_height": 640, "crop_x1": 0, "crop_y1": 0, "crop_x2": 640,
                 "crop_y2": 640, "scale": 1.0, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0,
                 "pixel_spacing": 0.2}
    analysis = {"mm_per_px": 0.2, "landmarks": base["landmarks"], "polygons": base["polygons"], "lesions": [],
                "quality": base["quality"], "transform": transform}
    record = assessor.new_record("abc", [assessor.build_image("L-MLO", meta, analysis)], [], {}, 1280, created=0.0)
    image = record["images"][0]
    assert image["measures"]["mask"]["ok"] is False and image["measures"]["mask"]["line"]
    by_id = {entry["id"]: entry for entry in image["criteria"]}
    assert by_id["pectoral_angle"]["grade"] == "info" and by_id["pectoral_angle"]["value"] > 0
    assert by_id["mlo_quality"]["grade"] in ("good", "poor")        # from the pectoral line and the pose nipple


# ── no nipple found: the contour fallback, marked estimated ─────────────────
def _without_nipple(view, drop=("nipple",)):
    image = _image(view)
    for name in drop:
        image["polygons"].pop(name, None)
    image["masks"] = masks_from_polygons(image["polygons"])
    return image


def test_without_a_nipple_the_fallback_measures_at_the_farthest_tissue_contour_point():
    from breast_seg import geometry

    mlo = _without_nipple("MLO")
    assert measure_mask(mlo)["reason"] == "no nipple mask"              # off: no measurement, as before
    estimate = measure_mask(mlo, contour_fallback=True)
    assert estimate["ok"] and estimate["estimated"] is True and estimate["nipple"] == "contour"
    assert estimate["missing"] == "no nipple mask" and estimate["line"]
    # the contour point farthest from the pectoral line taken as an infinite line (as the fallback was evaluated),
    # then measured as the route measures a nipple there
    contour = geometry.tissue_contour(mlo["masks"]["breast-tissue"])
    line = geometry.fit_pectoral_line(mlo["masks"]["pectoral"], mlo["masks"]["breast-tissue"], (640, 640))
    a, b = np.asarray(line.point_a, float), np.asarray(line.point_b, float)
    scores = [geometry.perpendicular_distance(a, b, point)[0] for point in contour]
    tip = contour[int(np.argmax(scores))]
    assert estimate["from"] == [float(tip[0]), float(tip[1])]
    measured = geometry.compute_pnl((int(tip[0]), int(tip[1])), line).distance_px
    assert estimate["mm"] == pytest.approx(measured * 0.2)
    # CC: farthest from the chest wall (the left edge of this breast), measured as a nipple there would be
    cc = measure_mask(_without_nipple("CC"), contour_fallback=True)
    assert cc["ok"] and cc["estimated"] and cc["from"][0] == 300.0 and cc["to"][0] == 0.0
    assert cc["from"][1] == cc["to"][1] and cc["mm"] == pytest.approx(60.0)
    right = _image("CC", chest_side="R", polygons={"breast-tissue": _square(340, 40, 639, 600)})
    right["masks"] = masks_from_polygons(right["polygons"])
    estimate = measure_mask(right, contour_fallback=True)                  # the wall on the right: the left edge
    assert estimate["from"][0] == 340.0 and estimate["to"][0] == 639.0


def test_a_found_nipple_is_never_replaced_by_the_fallback():
    for view in ("MLO", "CC"):
        image = _image(view)
        image["masks"] = masks_from_polygons(image["polygons"])
        with_fallback = measure_mask(image, contour_fallback=True)
        assert with_fallback == measure_mask(image) and "estimated" not in with_fallback


def test_the_fallback_needs_tissue_and_on_an_mlo_a_pectoral_line():
    tiny = _image("CC", polygons={"breast-tissue": _square(10, 10, 14, 14)})
    tiny["masks"] = masks_from_polygons(tiny["polygons"])
    assert measure_mask(tiny, contour_fallback=True)["reason"] == "no nipple mask"
    no_line = _without_nipple("MLO", drop=("nipple", "pectoral"))
    assert measure_mask(no_line, contour_fallback=True)["reason"] == "no nipple mask"
    no_breast = _without_nipple("MLO", drop=("nipple", "breast-tissue"))
    assert measure_mask(no_breast, contour_fallback=True)["reason"] == "no nipple mask"


def _left_breast(assessor, nipple_on_mlo=True):
    images = []
    for view in ("MLO", "CC"):
        meta = {"view": view, "laterality": "L", "chest_side": "L", "pixel_spacing_known": True,
                "manufacturer": "Phantom", "model_name": "", "original_width": 900, "original_height": 1200}
        base = _image(view, "L", chest_side="L")
        if view == "MLO" and not nipple_on_mlo:
            del base["polygons"]["nipple"]
        analysis = {"mm_per_px": 0.2, "landmarks": base["landmarks"], "polygons": base["polygons"], "lesions": [],
                    "quality": base["quality"]}
        images.append(assessor.build_image(f"L-{view}", meta, analysis))
    return assessor.new_record("est001", images, [], {}, 1280, created=0.0)


def test_an_estimated_measurement_enters_the_verdict_and_says_so():
    record = _left_breast(PositioningAssessor(Rules(), contour_fallback=True), nipple_on_mlo=False)
    side = record["summary"]["sides"][0]
    assert side["mask"]["ok"] and side["mask"]["verdict"] in ("good", "bad") and side["mask"]["estimated"] == ["MLO"]
    assert "estimated" not in side["landmark"] and side["comparison"]["state"] in ("agree", "disagree")
    english, turkish = build_report(record, "en")["text"], build_report(record, "tr")["text"]
    assert "L-MLO: nipple not found — the segmentation PNL is estimated" in english
    assert re.search(r"PNL \(MLO\) [\d.]+ mm \(estimated\)", english)
    assert "L-MLO: meme ucu bulunamadı — segmentasyon PNL’si" in turkish and "(tahmini)" in turkish
    # without the fallback the same breast has no segmentation verdict, as before
    plain = _left_breast(PositioningAssessor(Rules()), nipple_on_mlo=False)["summary"]["sides"][0]
    assert plain["mask"] == {"ok": False, "reason": "no nipple mask"}
    # with the nipple found, nothing is estimated
    found = _left_breast(PositioningAssessor(Rules(), contour_fallback=True))
    assert "estimated" not in found["summary"]["sides"][0]["mask"]
    assert "estimated" not in build_report(found, "en")["text"]


def test_upload_notes_are_written_in_the_report_language_by_their_codes():
    _, record = _record()
    record["problems"] = ["x.png: skipped — this application reads DICOM (.dcm) only; PNG/JPG files have no pixel "
                          "spacing or view information, so mm cannot be measured.", "y.dcm: a newer reason"]
    record["problem_codes"] = [{"text": record["problems"][0], "code": "raster_image", "params": {"file": "x.png"}},
                               {"text": record["problems"][1], "code": "a_code_this_server_does_not_know",
                                "params": {"file": "y.dcm"}}]
    notes = lambda lang: [row["text"] for row in build_report(record, lang)["sections"][-1]["rows"]]  # noqa: E731
    assert notes("en") == record["problems"]
    assert notes("tr") == ["x.png: atlandı — bu uygulama yalnız DICOM (.dcm) okur; PNG/JPG’de piksel aralığı ve "
                           "görünüm bilgisi olmadığı için mm ölçülemez.", "y.dcm: a newer reason"]
    del record["problem_codes"]                                         # a record stored before the codes were kept
    assert notes("tr") == record["problems"]


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
def test_criteria_for_mlo_and_cc():
    grader = CriteriaGrader()
    mlo = _measured(_image("MLO"))
    items = grader.for_image(mlo, mlo["masks"])
    assert [item["id"] for item in items] == ["pectoral_angle", "mlo_quality"]   # one graded MLO item, no depth
    assert items[0]["grade"] == "info"
    assert items[1]["grade"] == "na" and "geometry" in items[1]["detail"]   # no frame transform in this fixture
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


def _mlo_quality(line, nipple=(300.0, 300.0), size=(640, 640)):
    transform = {"original_width": size[0], "original_height": size[1], "crop_x1": 0, "crop_y1": 0,
                 "crop_x2": size[0], "crop_y2": size[1], "scale": 1.0, "pad_left": 0, "pad_top": 0,
                 "pad_right": 0, "pad_bottom": 0, "pixel_spacing": 0.1}
    image = {"view": "MLO", "measures": {"mask": {"line": line}}, "transform": transform}
    return mlo_quality(image, list(nipple), "L-MLO")


def test_mlo_quality_is_good_inside_and_poor_outside_the_image():
    inside = _mlo_quality([[100.0, 0.0], [0.0, 100.0]])                   # foot (50, 50)
    assert (inside["grade"], inside["value"]) == ("good", "good") and "inside the image" in inside["detail"]
    outside = _mlo_quality([[100.0, 0.0], [0.0, 100.0]], nipple=(600.0, 20.0))  # foot (340, -240): 24 mm above
    assert (outside["grade"], outside["value"]) == ("poor", "poor")
    assert outside["detail"].endswith("outside the image, 24.0 mm beyond the edge")


def test_mlo_quality_margin_is_a_quarter_percent_of_the_longest_side():
    # 0.25% of 640 px = 1.6 px: a foot 1 px outside still counts as inside, 2 px outside does not
    assert _mlo_quality([[0.0, -1.0], [640.0, -1.0]])["value"] == "good"
    assert _mlo_quality([[0.0, -2.0], [640.0, -2.0]])["value"] == "poor"
    assert _mlo_quality([[641.5, 0.0], [641.5, 640.0]])["value"] == "good"
    assert _mlo_quality([[641.7, 0.0], [641.7, 640.0]])["value"] == "poor"


def test_mlo_quality_needs_a_line_a_nipple_and_the_geometry():
    assert _mlo_quality(None)["grade"] == "na"
    line = {"view": "MLO", "measures": {"mask": {"line": [[0, 0], [1, 1]]}}}
    assert mlo_quality(line, None, "L-MLO")["grade"] == "na"
    assert mlo_quality(line, [5, 5], "L-MLO")["grade"] == "na"


def test_the_nipple_is_the_mask_centroid_whatever_the_size_of_the_mask():
    """No pixel limit: a nipple mask of any size is the nipple, and the pose point stands in only when there is no
    mask at all."""
    image = {"landmarks": {"points": {"nipple": [300.0, 320.0]}}}
    mask = np.zeros((640, 640), np.uint8)
    assert nipple_position(image, None) == [300.0, 320.0] and nipple_position(image, mask) == [300.0, 320.0]
    mask[100:103, 100:103] = 1                                            # 9 px
    assert nipple_position(image, mask) == [101.0, 101.0]
    mask[100:105, 100:105] = 1                                            # 25 px, under the limit there used to be
    assert nipple_position(image, mask) == [102.0, 102.0]
    assert nipple_position({}, None) is None


def test_the_mlo_quality_is_judged_from_the_nipple_the_distance_is_measured_from():
    """One nipple per view: with a small nipple mask far from the pose point, the PNL and the MLO quality both use
    the mask centroid (they used to part ways under 30 px)."""
    grader = CriteriaGrader()
    small = [[400.0, 300.0], [404.0, 300.0], [404.0, 304.0], [400.0, 304.0]]            # about 20 px
    image = _image("MLO", "L", points={"nipple": [250.0, 520.0]})
    image["polygons"]["nipple"]["points"] = small
    image["transform"] = {"original_width": 640, "original_height": 640, "crop_x1": 0, "crop_y1": 0, "crop_x2": 640,
                          "crop_y2": 640, "scale": 1.0, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0,
                          "pixel_spacing": 0.2}
    image = _measured(image)
    measured = image["measures"]["mask"]
    quality = next(entry for entry in grader.for_image(image, image["masks"]) if entry["id"] == "mlo_quality")
    assert measured["ok"] and quality["grade"] in ("good", "poor")
    assert quality["basis"]["nipple"] == [float(value) for value in measured["from"]]
    assert np.hypot(quality["basis"]["nipple"][0] - 250.0, quality["basis"]["nipple"][1] - 520.0) > 100   # not the pose


def test_header_gantry_is_shown_and_compression_is_not():
    grader = CriteriaGrader()
    image = _measured(_image("CC"))
    image["physics"] = {"compression_n": 100.0, "thickness_mm": 50.0, "gantry_deg": 45.0, "entrance_dose_mgy": 1.2}
    items = grader.for_image(image, image["masks"])
    assert [item["id"] for item in items] == ["nipple_centred", "gantry"]
    assert items[1]["grade"] == "info" and items[1]["value"] == 45.0


def test_counts_ignore_information():
    items = [{"grade": "good"}, {"grade": "poor"}, {"grade": "na"}, {"grade": "info"}]
    assert CriteriaGrader.summarise(items) == {"counts": {"good": 1, "acceptable": 0, "poor": 1, "na": 1}, "graded": 2}


def test_a_criterion_item_carries_nothing_that_is_never_read():
    image = _measured(_image("MLO"))
    entry = CriteriaGrader().for_image(image, image["masks"])[0]
    assert set(entry) == {"id", "name", "grade", "value", "unit", "detail", "code", "params", "view_key"}


def test_a_graded_mlo_quality_says_what_it_was_judged_from():
    graded = _mlo_quality([[100.0, 0.0], [0.0, 100.0]], nipple=(80.0, 80.0))     # foot (50, 50) on x + y = 100
    assert graded["basis"] == {"line": [[100.0, 0.0], [0.0, 100.0]], "nipple": [80.0, 80.0], "foot": [50.0, 50.0]}
    outside = _mlo_quality([[100.0, 0.0], [0.0, 100.0]], nipple=(600.0, 20.0))
    assert outside["grade"] == "poor" and outside["basis"]["foot"] == [340.0, -240.0]   # also when it is off the image
    assert "basis" not in _mlo_quality(None)                                          # nothing was judged


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


def test_apply_labels_leaves_an_unchanged_view_unmarked():
    record = {"images": [_image("MLO", "L")]}
    image = record["images"][0]
    sent = clean_labels({"images": [{"view_key": "L-MLO", "polygons": copy.deepcopy(image["polygons"]),
                                     "landmarks": {"points": copy.deepcopy(image["landmarks"]["points"])}}]})
    assert apply_labels(record, sent) == 1                       # matched ...
    assert "edited" not in record and not image["landmarks"].get("edited")   # ... but nothing changed
    moved = copy.deepcopy(sent)
    moved["L-MLO"]["points"]["nipple"] = [moved["L-MLO"]["points"]["nipple"][0] + 3.0,
                                          moved["L-MLO"]["points"]["nipple"][1]]
    assert apply_labels(record, moved) == 1 and record["edited"] > 0 and image["landmarks"]["edited"]


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
    assert {"summary", "version"} == set(record["criteria"])
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
    # the same numbers in both languages, written the way each language writes them
    delta = record["summary"]["sides"][0]["landmark"]["delta_mm"]
    assert f"ΔD {delta:.1f} mm" in english["text"]
    assert "ΔD {} mm".format(f"{delta:.1f}".replace(".", ",")) in turkish["text"]


def test_turkish_report_uses_a_decimal_comma_a_minus_sign_and_a_space_before_units():
    _, record = _record()
    _set_criterion(record, "L-CC", "nipple_centred", value=75.6)
    _set_criterion(record, "L-MLO", "pectoral_angle", value=-47.2)
    side = record["summary"]["sides"][0]
    side["mask"] = {**side["mask"], "ok": True, "verdict": "bad", "delta_mm": 12.3, "threshold_mm": 10.0,
                    "pnl_mm": 40.0, "cc_mm": 27.7}
    side["comparison"] = {"state": "disagree", "gap_mm": 6.78, "wide_gap": False}
    turkish, english = build_report(record, "tr")["text"], build_report(record, "en")["text"]
    assert "75,6 mm" in turkish and "\u221247,2°" in turkish and "ΔD 12,3 mm > 10 mm" in turkish
    assert "(6,8 mm fark)" in turkish and "75.6 mm" in english and "-47.2°" in english
    assert "6.8 mm apart" in english and "6.78" not in english


def test_report_lists_views_and_breasts_in_the_grid_order():
    _, record = _record()
    record["images"].reverse()                                         # upload order differs from the grid
    text = build_report(record, "en")["text"]
    header = next(line for line in text.splitlines() if line.startswith("Views:"))
    assert header.endswith("R-MLO, L-MLO, R-CC, L-CC")
    criteria = text[text.index("POSITIONING CRITERIA"):text.index("CONCLUSION")]
    assert [view for view in ("R-MLO", "L-MLO", "R-CC", "L-CC")] == sorted(
        ("R-MLO", "L-MLO", "R-CC", "L-CC"), key=criteria.index)
    verdicts = text[text.index("VERDICT"):text.index("POSITIONING CRITERIA")]
    assert verdicts.index("Right breast") < verdicts.index("Left breast")


def test_report_grade_words_match_the_review_screen():
    # one word per grade, the same in the drawer (i18n.js grade_*) and in the report
    from breast_seg.positioning.report import WORDS
    assert WORDS["tr"]["grades"]["poor"] == "kötü" and WORDS["tr"]["grades"]["good"] == "iyi"
    assert WORDS["en"]["grades"]["poor"] == "poor" and WORDS["en"]["grades"]["good"] == "good"


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


def test_poor_mlo_quality_is_stated_without_advising_a_repeat():
    _, record = _record()
    _set_criterion(record, "L-MLO", "mlo_quality", grade="poor", value="poor")
    english, turkish = build_report(record, "en"), build_report(record, "tr")
    assert "· L-MLO — MLO quality: poor" in english["text"] and "· L-MLO — MLO kalitesi: kötü" in turkish["text"]
    assert "repeat" not in english["text"].lower() and "tekrar" not in turkish["text"].lower()
    row = next(row for section in english["sections"] for row in section["rows"]
               if row.get("label", "").strip() == "MLO quality")
    assert (row["value"], row["grade"]) == ("poor", "poor")          # the value word only; red by its grade
    _set_criterion(record, "L-MLO", "mlo_quality", grade="good", value="good")
    assert "— MLO kalitesi" not in build_report(record, "tr")["text"]   # a good one is not a finding
    rows = [row for section in build_report(record, "tr")["sections"] for row in section["rows"]]
    assert any(row.get("label", "").strip() == "MLO kalitesi" and row["value"] == "iyi" for row in rows)


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
    assert [entry["id"] for entry in record["images"][0]["criteria"]] == ["pectoral_angle", "mlo_quality"]
    assert record["images"][0]["measures"]["landmark"]["mm"] == 999.0
    assert record["summary"] == summary and record["rules"] == rules


# ── one rounding, everywhere ───────────────────────────────────────────────
def test_a_number_is_rounded_once_by_one_rule():
    from breast_seg.positioning.numbers import fixed, rounded

    # the exact value, half away from zero, as the interface's toFixed (test_ui_js.py checks the two against each
    # other); 1.005 is 1.00499… in binary, 47.95 is 47.9500…03
    assert [fixed(value) for value in (47.949377, 47.95, 47.25, 47.75, -47.25, 0.05, 75.6, 10)] == [
        "47.9", "48.0", "47.3", "47.8", "-47.3", "0.1", "75.6", "10.0"]
    assert [fixed(2.5, 0), fixed(24.4, 0), fixed(1.005, 2), fixed(0.125, 2)] == ["3", "24", "1.00", "0.13"]
    assert rounded(47.949377) == 47.9 and rounded(29.25) == 29.3 and rounded(24.5, 0) == 25.0


def _with_cc_distance(record, mm=47.949377):
    """``record`` with the left CC's landmark distance set to ``mm`` and the verdicts decided again from it."""
    cc = next(image for image in record["images"] if image["view_key"] == "L-CC")
    cc["measures"]["landmark"]["mm"] = mm
    record["summary"] = summarise(record["images"], 10.0)
    return cc, next(side for side in record["summary"]["sides"] if side["laterality"] == "L")


def test_a_measurement_reads_the_same_under_its_image_on_its_card_and_in_the_report():
    """47.949 mm is 47.9 mm wherever it is written. Rounded to 47.95 on its way into the summary, it read 47.9 mm
    under the image and 48.0 mm on the card and in the report."""
    _, record = _record()
    cc, left = _with_cc_distance(record)
    # the card reads the summary, the image its own measurement: one unrounded number, rounded where it is shown
    assert left["landmark"]["cc_mm"] == cc["measures"]["landmark"]["mm"] == 47.949377
    assert left["landmark"]["delta_mm"] == abs(left["landmark"]["pnl_mm"] - 47.949377)
    assert left["comparison"]["gap_mm"] == abs(left["landmark"]["delta_mm"] - left["mask"]["delta_mm"])
    english, turkish = build_report(record, "en")["text"], build_report(record, "tr")["text"]
    assert "(CC) 47.9 mm" in english and "(CC) 47,9 mm" in turkish
    assert "(CC) 48.0 mm" not in english and "(CC) 48,0 mm" not in turkish
    # a tie goes the way the page's toFixed takes it (47.25 is exact in binary; Python's own format says 47.2)
    _with_cc_distance(record, 47.25)
    assert "(CC) 47.3 mm" in build_report(record, "en")["text"]


def test_a_criterion_writes_its_numbers_by_the_same_rule():
    from breast_seg.positioning.criteria import item

    beyond = item("mlo_quality", "MLO quality", "poor", "poor", "", "mlo_outside_by", "L-MLO", mm=7.25)
    assert beyond["params"] == {"mm": 7.3} and "7.3 mm beyond the edge" in beyond["detail"]
    # a share is written without decimals, so it is kept without them: 24.5 is 25% in English and in Turkish alike
    centred = item("nipple_centred", "Nipple centred", "info", 6.0, "mm", "nipple_centred", "L-CC", share=24.5)
    assert centred["params"] == {"share": 25.0} and "(25% of the half-height)" in centred["detail"]


def test_an_older_record_gets_its_unrounded_millimetres_back_when_it_is_regraded():
    assessor, record = _record()
    _with_cc_distance(record)
    exact = copy.deepcopy(record["summary"])
    for side in record["summary"]["sides"]:                       # as summaries were stored: two decimals
        for route in ("landmark", "mask"):
            for key in ("pnl_mm", "cc_mm", "delta_mm"):
                side[route][key] = round(side[route][key], 2)
        side["comparison"]["gap_mm"] = round(side["comparison"]["gap_mm"], 2)
    record["criteria"]["version"] = 7
    left = next(side for side in record["summary"]["sides"] if side["laterality"] == "L")
    assert left["landmark"]["cc_mm"] == 47.95 and record["summary"] != exact
    assessor.regrade(record)
    assert record["summary"] == exact and record["criteria"]["version"] == CRITERIA_VERSION
    # a summary that does not come from the measurements stored next to it is left as it is, verdict and numbers
    left["mask"]["pnl_mm"] = 12.34
    stored = copy.deepcopy(record["summary"])
    assessor.regrade(record)
    assert record["summary"] == stored


def test_masks_are_uint8_binary():
    masks = masks_from_polygons({"x": _square(0, 0, 3, 3)})
    assert masks["x"].dtype == np.uint8 and set(np.unique(masks["x"])) == {0, 1}


def test_a_brightened_nipple_is_named_in_the_report_and_survives_an_edit():
    _, record = _record()
    image = next(image for image in record["images"] if image["view_key"] == "R-CC")
    image["polygons"]["nipple"]["brightened"] = True
    assert "R-CC: nipple found after brightening the front of the breast — check its position." in \
        build_report(record, "en")["text"]
    assert "R-CC: meme ucu memenin ön ucu parlatılınca bulundu" in build_report(record, "tr")["text"]
    assert "brightening" not in build_report(_record()[1], "en")["text"]
    cleaned = clean_labels({"images": [{"view_key": "R-CC", "polygons": {"nipple": image["polygons"]["nipple"]}}]})
    assert cleaned["R-CC"]["polygons"]["nipple"]["brightened"] is True


def test_a_lesion_whose_label_or_confidence_changed_is_a_change():
    lesion = {"id": "a1", "label": "Mass", "box": [10.0, 20.0, 60.0, 80.0], "confidence": 0.7}
    record = {"images": [{**_image("CC", "L"), "lesions": [dict(lesion)]}]}
    image = record["images"][0]

    def sent(**changes):
        return clean_labels({"images": [{"view_key": "L-CC", "polygons": copy.deepcopy(image["polygons"]),
                                         "landmarks": {"points": copy.deepcopy(image["landmarks"]["points"])},
                                         "lesions": [{**lesion, **changes}]}]})

    assert apply_labels(record, sent()) == 1 and "edited" not in record            # sent back as stored
    assert apply_labels(record, sent(label="Calcification")) == 1 and record.pop("edited") > 0
    assert image["lesions"][0]["label"] == "Calcification"
    assert apply_labels(record, sent(label="Calcification", confidence=0.2)) == 1 and record.pop("edited") > 0
    assert image["lesions"][0]["confidence"] == 0.2


def test_every_criterion_detail_carries_a_code_the_interface_can_translate():
    from breast_seg.positioning.criteria import DETAILS, PARAM_DIGITS
    from breast_seg.positioning.numbers import fixed

    # the codes the grader emits are keys of DETAILS, and the English sentence is built from them
    grader = CriteriaGrader()
    emitted = {}
    for view in ("MLO", "CC"):
        image = _measured(_image(view))
        image["physics"] = {"gantry_deg": 45.0}
        for entry in grader.for_image(image, image["masks"]) + grader.for_image({**image, "measures": {}}, {}):
            emitted[entry["code"]] = entry
    emitted.update({entry["code"]: entry for entry in (
        _mlo_quality([[100.0, 0.0], [0.0, 100.0]]), _mlo_quality([[100.0, 0.0], [0.0, 100.0]], nipple=(600.0, 20.0)),
        _mlo_quality(None))})
    assert set(emitted) <= set(DETAILS) and {"mlo_inside", "mlo_outside_by", "nipple_centred", "from_header"} <= set(emitted)
    assert emitted["mlo_outside_by"]["params"] == {"mm": 24.0} and "24.0 mm beyond" in emitted["mlo_outside_by"]["detail"]
    # the sentence writes each parameter as the interface does (the same rounding rule, the same number of decimals)
    written = lambda params: {key: fixed(value, PARAM_DIGITS[key]) for key, value in params.items()}  # noqa: E731
    assert all(entry["detail"] == DETAILS[code].format(**written(entry["params"])) for code, entry in emitted.items())

    # every code has a Turkish sentence in the interface, with the same placeholders
    source = (Path(__file__).resolve().parents[1] / "src/breast_seg/apps/positioning/static/js/i18n.js").read_text()
    block = source[source.index("const DETAILS_TR = {"):]
    block = block[:block.index("};")]
    turkish = dict(re.findall(r"^\s*(\w+): '([^']*)',?$", block, flags=re.M))
    assert set(turkish) == set(DETAILS)
    for code, english in DETAILS.items():
        assert set(re.findall(r"\{(\w+)", english)) == set(re.findall(r"\{(\w+)\}", turkish[code])), code


def test_a_corrected_nipple_contour_is_no_longer_flagged_for_checking():
    record = {"images": [_image("CC", "L")]}
    image = record["images"][0]
    image["polygons"]["nipple"].update(brightened=True, rescued=True)

    def sent(nipple, breast=None):
        polygons = copy.deepcopy(image["polygons"])
        polygons["nipple"]["points"] = nipple
        if breast:
            polygons["breast-tissue"]["points"] = breast
        return clean_labels({"images": [{"view_key": "L-CC", "polygons": polygons,
                                         "landmarks": {"points": copy.deepcopy(image["landmarks"]["points"])}}]})

    stored = copy.deepcopy(image["polygons"]["nipple"]["points"])
    moved_breast = [[x + 1.0, y] for x, y in image["polygons"]["breast-tissue"]["points"]]
    apply_labels(record, sent(stored, moved_breast))                 # another contour was corrected, not the nipple
    assert image["polygons"]["nipple"]["brightened"] and image["polygons"]["nipple"]["rescued"]
    apply_labels(record, sent([[x + 2.0, y] for x, y in stored]))    # the reviewer set the nipple: nothing to check
    assert not image["polygons"]["nipple"]["brightened"] and not image["polygons"]["nipple"]["rescued"]
    assert "found after brightening" not in build_report({**_record()[1], "images": [
        {**entry, "polygons": image["polygons"]} for entry in _record()[1]["images"]]}, "en")["text"]


def test_a_contour_sent_back_with_the_same_points_stays_the_stored_one():
    """A vertex dragged away and dropped back arrives as an edited contour with the same points. Next to a real edit
    in the same view it must not turn the model's contour into an edited one, change its confidence or drop the marks
    that ask the reader to check its position."""
    record = {"images": [_image("CC", "L")]}
    image = record["images"][0]
    image["polygons"]["nipple"].update(brightened=True, rescued=True, confidence=0.31)
    stored = copy.deepcopy(image["polygons"])

    polygons = copy.deepcopy(image["polygons"])
    polygons["nipple"] = {"points": copy.deepcopy(stored["nipple"]["points"]), "edited": True, "confidence": 1.0}
    polygons["breast-tissue"]["points"] = [[x + 1.0, y] for x, y in stored["breast-tissue"]["points"]]   # a real edit
    apply_labels(record, clean_labels({"images": [{
        "view_key": "L-CC", "polygons": polygons,
        "landmarks": {"points": copy.deepcopy(image["landmarks"]["points"])}}]}))

    assert image["polygons"]["nipple"] == stored["nipple"]                # as stored: flags, confidence, no "edited"
    assert image["polygons"]["breast-tissue"]["points"] != stored["breast-tissue"]["points"]
    assert image["landmarks"]["edited"] is True                           # the view itself did change


def test_saving_a_view_leaves_untouched_lesions_as_they_were_stored():
    lesions = [{"label": "Mass", "box": [10.0, 20.0, 60.0, 80.0], "confidence": 0.7},           # no id, as detected
               {"id": "k2", "label": "Calcification", "box": [100.0, 100.0, 140.0, 150.0], "confidence": 0.4}]
    record = {"images": [{**_image("CC", "L"), "lesions": copy.deepcopy(lesions)}]}
    image = record["images"][0]
    points = copy.deepcopy(image["landmarks"]["points"])
    points["nipple"] = [points["nipple"][0] + 3.0, points["nipple"][1]]                # only the landmark moved
    apply_labels(record, clean_labels({"images": [{
        "view_key": "L-CC", "polygons": copy.deepcopy(image["polygons"]), "landmarks": {"points": points},
        "lesions": copy.deepcopy(lesions) + [{"label": "Mass", "box": [200.0, 200.0, 240.0, 260.0]}]}]}))
    assert image["lesions"][:2] == lesions                           # no new id, no "edited" mark
    assert image["lesions"][2]["edited"] is True and image["lesions"][2]["id"]         # the one that was added
