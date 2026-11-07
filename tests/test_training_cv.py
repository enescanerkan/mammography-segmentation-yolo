"""Cross-validation runner end to end on the synthetic cohort with a fake trainer (no framework, no training)."""
from pathlib import Path

import cv2
import pandas as pd
import pytest

from breast_seg.datasets.labels import rasterise, read_polygons
from breast_seg.training import CrossValidationRunner, load_cv_config
from breast_seg.training.yolo import YoloSegTrainer
from tests.test_datasets_builder import DatasetBuilder, load_committed, make_cohort

ROOT = Path(__file__).resolve().parents[1]


class FakeFoldTrainer(YoloSegTrainer):
    """Fit writes a marker; predict writes the reference label map of each test image (from the pool labels)."""

    fitted = []

    def fit(self):
        self.prepare()
        FakeFoldTrainer.fitted.append(self.config.name)
        return None

    def predict(self, images, out_dir, weights=None):
        out = []
        for p in images:
            p = Path(p)
            lbl = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
            if not lbl.exists():                                    # unlabelled pool image
                lbl = p.parent / "labels" / f"{p.stem}.txt"
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            label, _ = rasterise(read_polygons(lbl), *img.shape)
            target = Path(out_dir) / f"{p.stem}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), label)
            out.append(target)
        return out


def factory(cfg, paths):
    return FakeFoldTrainer(cfg, paths)


@pytest.fixture()
def cohort(tmp_path):
    paths = make_cohort(tmp_path, n_studies=10)
    DatasetBuilder(load_committed("seg_v4.yaml", **{"exports.nnunet.enabled": False}), paths).build()
    labels = pd.read_csv(paths.resolve("${mlo_labels}"))
    rows = []
    for study in sorted(labels.StudyInstanceUID.unique()):
        j = int(study[2:])
        for side in ("L", "R"):
            if (j, side) == (4, "R"):
                continue                                            # pair missing from the table: placed by study
            rows.append({"StudyInstanceUID": study, "Side": side, "SOPInstanceUID_mlo": f"m{j}{side}",
                         "SOPInstanceUID_cc": f"c{j}{side}", "qualitativeLabel": "Good", "fold": j % 5})
    pd.DataFrame(rows).to_csv(paths.resolve("${cv10_fold_table}"), index=False)
    return paths


def test_cv_runner_on_committed_config(cohort):
    FakeFoldTrainer.fitted = []
    cfg = load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"n_folds": 5})
    runner = CrossValidationRunner(cfg, cohort, ROOT / "configs/cv", trainer_factory=factory)
    a = runner.assignment()
    assert a.fold_of["m4R"] == 4 and a.fold_of["cc_c4R"] == 4           # extended through its study
    assert not a.is_labelled("m7R")                                     # excluded from v4: predict only
    s = runner.split(2)
    assert "m7R" not in s.train + s.val and set(s.test) == {"m2L", "m2R", "cc_c2L", "cc_c2R", "m7L", "cc_c7L", "cc_c7R",
                                                            "m7R"}
    spec = runner.spec(2)
    assert spec.images["m7R"].parent == cohort.resolve("${mlo_label_pool}")
    report = runner.run()
    assert sorted(FakeFoldTrainer.fitted) == [f"cv10_yolo26m_fold{i}" for i in range(5)]
    assert report.oof.complete and report.oof.n_images == 40
    fold_tree = cohort.dir("runs_dir") / "cv/cv10_yolo26m/fold2"
    assert (fold_tree / "images/test/m7R.png").is_symlink() and not (fold_tree / "labels/test/m7R.txt").exists()
    assert (cohort.dir("runs_dir") / "cv10_yolo26m_fold2/data.yaml").is_file()
    ev = report.evaluation
    assert len(ev.pairs) == 20 and ev.views.fold.notna().all()
    assert len(ev.summary["per_fold"]) == 5 and "decision" in ev.summary
    assert ev.summary["dice"]["MLO"]["pectoral"] == pytest.approx(1.0)
    res = cohort.dir("results_dir")
    assert (res / "cv10_yolo26m_pairs.csv").is_file() and (res / "cv10_yolo26m_summary.json").is_file()
    # a second run skips finished folds
    FakeFoldTrainer.fitted = []
    runner.run()
    assert FakeFoldTrainer.fitted == []


def test_cv_runner_rejects_unknown_fold_and_missing_sections(cohort):
    from breast_seg.config import ConfigError

    with pytest.raises(ConfigError):
        load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"run_folds": [12]})
    with pytest.raises(ConfigError):
        load_cv_config(ROOT / "configs/cv/cv10_yolo26m.yaml", {"folds": None})
