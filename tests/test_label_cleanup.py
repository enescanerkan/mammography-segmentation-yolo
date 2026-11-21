"""The label-cleaning rules of the editor's "Smooth / clean labels" (breast_seg.positioning.label_cleanup), on
synthetic label maps: islands, holes, the pectoral-breast strip of an MLO, the pectoral of a CC, which nipple is
kept, smoothing, the contours of a view that are already clean coming back as they were, and the editor's
button deciding on label maps, a second click changing nothing."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("scipy")

from breast_seg.positioning.label_cleanup import (  # noqa: E402
    BREAST,
    NIPPLE,
    PECTORAL,
    clean_label_map,
    clean_polygons,
    clean_view,
    label_map,
    smooth,
)
from breast_seg.positioning.labels import clean_polygon  # noqa: E402

S = 200


def _mlo():
    """A left-chest-wall MLO: breast tissue as a half ellipse, a pectoral triangle at the top left, and a nipple on
    the front edge."""
    label = np.zeros((S, S), np.uint8)
    cv2.ellipse(label, (0, 100), (150, 90), 0, -90, 90, BREAST, -1)
    cv2.fillPoly(label, [np.array([[0, 10], [70, 10], [0, 140]], np.int32)], PECTORAL)
    cv2.circle(label, (148, 100), 5, NIPPLE, -1)
    return label


def _band(x1, x2):
    return {"points": [[x1, 0], [x2, 0], [x2, S - 1], [x1, S - 1]], "confidence": 0.9, "rescued": False}


def test_islands_are_removed_and_holes_are_filled():
    label = _mlo()
    cv2.circle(label, (175, 20), 4, BREAST, -1)          # breast tissue in the air
    cv2.circle(label, (180, 180), 3, PECTORAL, -1)       # pectoral in the air
    cv2.circle(label, (90, 100), 6, 0, -1)               # a hole in the breast tissue
    out = clean_label_map(label, "MLO")
    assert out.dtype == np.uint8 and out.shape == label.shape
    assert (out[12:28, 167:183] == 0).all() and (out[174:186, 174:186] == 0).all()
    assert (out[95:106, 85:96] == BREAST).all()
    for value in (PECTORAL, BREAST, NIPPLE):              # one component per class
        assert cv2.connectedComponents((out == value).astype(np.uint8), connectivity=8)[0] == 2


def test_discarded_pixels_inside_take_the_nearest_kept_class():
    label = _mlo()
    cv2.circle(label, (15, 60), 3, BREAST, -1)           # breast tissue inside the pectoral muscle
    cv2.circle(label, (100, 120), 3, PECTORAL, -1)       # a pectoral speck inside the breast tissue
    out = clean_label_map(label, "MLO")
    assert (out[57:64, 12:19] == PECTORAL).all()
    assert (out[117:124, 97:104] == BREAST).all()


def _strip():
    """Pectoral and breast tissue with an unlabelled 6 px strip between them, open to the top edge (so filling holes
    cannot close it), and the image under it."""
    label = np.zeros((S, S), np.uint8)
    label[:, :40] = PECTORAL
    label[:, 46:150] = BREAST
    label[95:105, 146:152] = NIPPLE
    image = np.zeros((S, S), np.uint8)
    image[:, :150] = 90
    return label, image


def test_the_strip_between_pectoral_and_breast_is_filled_where_the_image_shows_tissue():
    label, image = _strip()
    out = clean_label_map(label, "MLO", image)
    strip = out[:, 40:46]
    assert (strip > 0).all()                                            # nearest class on each side
    assert (strip[:, :3] == PECTORAL).all() and (strip[:, 3:] == BREAST).all()
    assert (clean_label_map(label, "MLO")[:, 40:46] == 0).all()         # no image: the rule does not apply
    assert (clean_label_map(label, "MLO", np.full((S, S), 15, np.uint8))[:, 40:46] == 0).all()   # air or padding
    # a render of the same frame at another size is reduced to the label map's size first
    larger = cv2.resize(image, (2 * S, 2 * S), interpolation=cv2.INTER_NEAREST)
    assert np.array_equal(clean_label_map(label, "MLO", larger), out)


def test_a_cc_has_no_pectoral_class():
    label = np.zeros((S, S), np.uint8)
    cv2.ellipse(label, (S, 100), (140, 90), 0, 90, 270, BREAST, -1)
    cv2.ellipse(label, (S, 100), (25, 50), 0, 90, 270, PECTORAL, -1)
    out = clean_label_map(label, "CC")
    assert not (out == PECTORAL).any()
    assert (out[90:111, 185:200] == BREAST).all()
    # on an MLO the same pixels stay pectoral
    assert (clean_label_map(label, "MLO")[90:111, 185:200] == PECTORAL).all()


def test_the_nipple_on_the_front_edge_is_kept_and_otherwise_the_nearest():
    label = _mlo()
    cv2.circle(label, (100, 180), 9, NIPPLE, -1)         # larger, but away from the breast tissue
    out = clean_label_map(label, "MLO")
    assert (out[96:105, 144:153] == NIPPLE).any() and not (out[171:190, 91:110] == NIPPLE).any()

    far = _mlo()
    far[far == NIPPLE] = BREAST
    cv2.circle(far, (170, 100), 4, NIPPLE, -1)           # 20 px in front of the breast tissue
    cv2.circle(far, (190, 190), 6, NIPPLE, -1)           # larger, farther
    out = clean_label_map(far, "MLO")
    assert (out[96:105, 166:175] == NIPPLE).any() and not (out[185:196, 185:196] == NIPPLE).any()


def test_smoothing_rounds_a_jagged_edge_and_keeps_the_shape():
    label = np.zeros((S, S), np.uint8)
    label[:, :100] = BREAST
    for row in range(0, S, 4):                            # a 2 px zig-zag along the edge
        label[row:row + 2, 100:102] = BREAST
    out = clean_label_map(label, "CC")
    before, after = label == BREAST, out == BREAST
    assert (before & after).sum() / (before | after).sum() > 0.98
    edge = [int(np.nonzero(out[row] == BREAST)[0].max()) for row in range(S)]
    assert len(set(edge)) == 1                            # one straight edge, no staircase
    assert np.array_equal(smooth(smooth(out)), out)


def test_clean_contours_come_back_unchanged():
    clean = {"breast-tissue": _band(0, 119), "pectoral": _band(0, 59), "an unknown class": {"points": [[1, 1]]}}
    out = clean_polygons(clean, "MLO", size=S)
    assert out.keys() == clean.keys() and all(out[name] is clean[name] for name in clean)


def test_a_changed_contour_is_a_copy_with_new_points_and_its_flags():
    nipple = {"points": [[160, 90], [175, 90], [175, 110], [160, 110]], "confidence": 0.31, "rescued": True,
              "brightened": True}
    polygons = {"breast-tissue": _band(0, 119), "nipple": nipple}
    out = clean_polygons(polygons, "CC", size=S)
    assert out["breast-tissue"] is polygons["breast-tissue"]          # untouched by the cleaning
    changed = out["nipple"]
    assert changed is not nipple and changed["points"] != nipple["points"] and len(changed["points"]) >= 3
    assert {key: changed[key] for key in ("confidence", "rescued", "brightened")} == \
        {"confidence": 0.31, "rescued": True, "brightened": True}
    assert "edited" not in changed and nipple["points"][0] == [160, 90]   # the input is not modified


def test_a_cc_pectoral_contour_is_merged_into_the_breast_tissue():
    out = clean_polygons({"breast-tissue": _band(0, 119), "pectoral": _band(0, 20)}, "CC", size=S)
    assert set(out) == {"breast-tissue"}
    assert (label_map(out, S)[:, :120] == BREAST).all()
    alone = clean_polygons({"pectoral": _band(0, 59)}, "CC", size=S)   # no breast contour: the pixels become one
    assert set(alone) == {"breast-tissue"}
    assert alone["breast-tissue"]["confidence"] == 1.0 and alone["breast-tissue"]["rescued"] is False


def test_label_map_paints_breast_then_pectoral_then_nipple():
    painted = label_map({"nipple": _band(50, 70), "pectoral": _band(0, 59), "breast-tissue": _band(0, 119),
                         "short": {"points": [[0, 0], [5, 5]]}}, S)
    assert painted[100, 30] == PECTORAL and painted[100, 65] == NIPPLE and painted[100, 100] == BREAST
    assert painted[100, 150] == 0


def _traced(label):
    """The contours of a label map as the engine traces model masks."""
    from breast_seg.inference.engine import largest_polygon

    return {name: {"points": largest_polygon((label == value).astype(np.uint8)), "confidence": 0.9, "rescued": False}
            for name, value in (("breast-tissue", BREAST), ("pectoral", PECTORAL), ("nipple", NIPPLE))
            if (label == value).any()}


def test_the_button_decides_on_label_maps(monkeypatch):
    """Cleaned contours that paint the same pixels as the ones sent are no change, whatever their points: the
    contours come back exactly as sent and nothing is named as changed."""
    import breast_seg.positioning.label_cleanup as cleanup

    sent = {"breast-tissue": _band(0, 119), "pectoral": _band(0, 59)}
    rotated = {name: {**entry, "points": entry["points"][1:] + entry["points"][:1]} for name, entry in sent.items()}
    monkeypatch.setattr(cleanup, "clean_polygons", lambda polygons, view, image=None, size=S: rotated)
    polygons, changed = clean_view(sent, "MLO", size=S)
    assert changed == [] and polygons == sent and polygons["pectoral"] is sent["pectoral"]


def test_a_second_click_changes_nothing_until_a_contour_is_edited():
    """The smoothing is not idempotent (run on its own result it shaves a few more pixels), so the contours a clean
    returns carry a fingerprint of their points: cleaning them again, also after a save, is a no-op; an edit of any
    of them makes the view cleaned for real again."""
    jagged = _mlo()
    jagged[95:106, 150:162] = BREAST                                    # a bump on the front edge
    jagged[30:34, 120:124] = BREAST                                     # an island
    first, changed = clean_view(_traced(jagged), "MLO", size=S)
    assert "breast-tissue" in changed and all(entry["cleaned"] for entry in first.values())
    once = label_map(first, S)
    assert not np.array_equal(clean_label_map(once, "MLO"), once)      # the rules would change it again ...
    second, changed = clean_view(first, "MLO", size=S)
    assert changed == [] and second == first                           # ... the button does not
    stored = {name: clean_polygon(entry) for name, entry in first.items()}   # as Save stores it
    assert clean_view(stored, "MLO", size=S)[1] == []

    moved = {**stored, "breast-tissue": {**stored["breast-tissue"]}}
    points = [list(point) for point in moved["breast-tissue"]["points"]]
    points[len(points) // 2][0] += 12.0                                 # a vertex dragged out
    moved["breast-tissue"]["points"] = points
    assert "breast-tissue" in clean_view(moved, "MLO", size=S)[1]
    assert clean_polygon({**stored["nipple"], "cleaned": "not a mark"}).get("cleaned") is None
