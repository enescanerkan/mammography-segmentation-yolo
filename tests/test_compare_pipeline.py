"""Compare pipeline pieces with fakes: dataset pairing, geometry, inference mapping, metrics and the orchestrator."""
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from breast_seg.compare import (
    GeometryUtils,
    InferenceEngine,
    MammographyDataset,
    MetricsCalculator,
    ModelFactory,
    PipelineOrchestrator,
    ResultVisualizer,
)
from breast_seg.compare.interfaces import PosePredictor
from tests.test_datasets_builder import make_cohort


def compare_root(tmp: Path) -> Path:
    """Arrange the synthetic cohort in the compare layout (labels/, MLO/, CC/ with images/test)."""
    paths = make_cohort(tmp / "src", n_studies=3)
    root = tmp / "compare"
    shutil.copytree(paths.resolve("${mlo_labels}").parent, root / "labels")
    for view, pool in (("MLO", "${mlo_label_pool}"), ("CC", "${cc_label_pool}")):
        (root / view / "images/test").mkdir(parents=True)
        shutil.copyfile(paths.resolve(f"${{{view.lower()}_metadata}}"), root / view / "metadata.csv")
        for png in paths.resolve(pool).glob("*.png"):
            shutil.copyfile(png, root / view / "images/test" / png.name)
    return root


class FakePose:
    def __init__(self, kpts):
        self.kpts = None if kpts is None else np.asarray(kpts, float)

    def predict(self, image_path, min_keypoints=1):
        return self.kpts


def test_geometry_round_trip():
    row = pd.Series({"scale": 0.5, "crop_x1": 10, "crop_y1": 20, "pad_left": 4, "pad_top": 0})
    ox, oy = GeometryUtils.inverse_transform_640(14, 10, row)
    assert (ox, oy) == (30.0, 40.0) and GeometryUtils.dicom_to_640(ox, oy, row) == (14.0, 10.0)
    assert GeometryUtils.pnl_infinite_line_mm((5, 50), (0, 0), (0, 10), 0.1) == pytest.approx(0.5)
    assert GeometryUtils.foot_on_infinite_line((5, 50), (0, 0), (0, 10)) == (0.0, 50.0)
    assert GeometryUtils.cc_chest_mm_from_nipple_dicom((100, 0), "R", 640, 0.1) == pytest.approx(54.0)


def test_dataset_pairs_and_ground_truth(tmp_path):
    ds = MammographyDataset.from_root(compare_root(tmp_path), split="Test")
    pairs = ds.get_test_pairs()
    assert {(p.study_uid, p.laterality, p.mlo_sop, p.cc_sop) for p in pairs} == {("st0", "L", "m0L", "c0L"),
                                                                                ("st0", "R", "m0R", "c0R")}
    kp = ds.get_mlo_gt_keypoints("m0L")
    assert kp["nipple"] == (580.0, 320.0) and kp["pectoral_bottom"] == (0.0, 400.0)
    assert ds.get_cc_gt_nipple("c0L") == (550.0, 320.0)
    assert ds.get_qualitative_label("c0L", "CC") in ("good", "bad")
    assert ds.get_mlo_metadata("m0L")["sop_uid"] == "m0L" and ds.get_cc_metadata("nope") is None
    assert len(MammographyDataset.from_root(compare_root(tmp_path / "b"), split="Train").get_test_pairs(limit=1)) == 1


def test_engine_maps_pose_and_segmentation_to_original(tmp_path):
    meta = pd.Series({"scale": 0.1, "crop_x1": 0, "crop_y1": 0, "pad_left": 0, "pad_top": 0, "laterality": "R",
                      "original_width": 640})
    line = SimpleNamespace(point_a=(0, 0), point_b=(0, 40))
    analysis = SimpleNamespace(has_nipple=True, nipple_mask=np.zeros((64, 64), np.uint8), pectoral_line=line)
    analysis.nipple_mask[30:34, 50:54] = 1
    seg = SimpleNamespace(predict=lambda path, cfg: SimpleNamespace(masks=None, boxes=None))
    engine = InferenceEngine(FakePose([[58, 32], [0, 0], [0, 40]]), FakePose([[55, 32]]), seg,
                             analyzer=SimpleNamespace(analyze_single=lambda res: analysis))
    pm = engine.run_mlo_pose(Path("x.png"), meta)
    assert pm["kpts_dicom"][0] == (580.0, 320.0) and pm["foot_dicom"] == (0.0, 320.0)
    pc = engine.run_cc_pose(Path("x.png"), meta)
    assert pc["wall_dicom"] == (639.0, 320.0)
    sm = engine.run_mlo_seg(Path("x.png"), meta)
    assert sm["pec_b_dicom"] == (0.0, 400.0) and sm["yolo_masks"] is None
    assert engine.run_cc_seg(Path("x.png"), meta)["nipple_dicom"][1] > 0
    assert InferenceEngine(FakePose(None), FakePose(None), None).run_mlo_pose(Path("x"), meta) is None
    assert isinstance(FakePose(None), PosePredictor)


def test_metrics_calculator_counts_unanswered_pairs():
    df = pd.DataFrame({"gt_pnl_mm": [50, 50, 50], "gt_chest_mm": [45, 70, 48], "pose_pnl_mm": [50, 50, None],
                       "pose_chest_mm": [44, 69, 47], "seg_pnl_mm": [50, 20, 50], "seg_chest_mm": [45, 70, 49],
                       "cc_qualitativeLabel": ["good", "bad", "good"], "mlo_pose_nip_err": [1.0, 2.0, None]})
    calc = MetricsCalculator(df)
    calc.compute_10mm_rules()
    assert list(df.gt_clinical_pass) == [True, False, True]
    t = calc.classification_table().set_index("Model")
    assert t.loc["Pose", "n_fail"] == 1 and t.loc["Pose", "accuracy"] == pytest.approx(100 * 2 / 3)
    assert t.loc["Seg", "accuracy"] == pytest.approx(100.0)
    lm = calc.landmark_table()
    assert lm[(lm.View == "MLO") & (lm.Metric == "Nipple")].Pose.iloc[0].startswith("1.50")


def test_orchestrator_with_fake_models(tmp_path):
    root = compare_root(tmp_path)
    ds = MammographyDataset.from_root(root, split="Test")
    engine = InferenceEngine(FakePose([[58, 32], [0, 0], [0, 40]]), FakePose([[55, 32]]), None)

    def fake_dicom(path):
        return np.zeros((640, 640, 3), np.uint8), 0.1, None

    orch = PipelineOrchestrator(ds, engine, tmp_path / "out", ResultVisualizer(), dicom_root=root / "test_dicom",
                                dicom_loader=fake_dicom)
    df = orch.run_compare()
    assert len(df) == 2 and df.pose_pnl_mm.iloc[0] == pytest.approx(58.0)
    assert df.mlo_pose_nip_err.iloc[0] == pytest.approx(0.0) and df.seg_pnl_mm.isna().all()
    for name in ("metrics_clinical.csv", "classification_metrics.csv", "landmark_summary.csv"):
        assert (tmp_path / "out" / name).is_file()
    assert len(list((tmp_path / "out/viz_dicom").glob("*.png"))) == 2


def test_model_factory_without_weights():
    f = ModelFactory(Path("does/not/exist"))
    pose = f.get_pose_model("MLO")
    assert pose.weights_path.name == "mlo-yolo26-pose-advanced.pt" and not pose.available
    assert pose.predict(Path("x.png")) is None
    with pytest.raises(FileNotFoundError):
        _ = pose.model
