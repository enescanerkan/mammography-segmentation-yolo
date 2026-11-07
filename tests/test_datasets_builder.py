"""Dataset builder on a tiny synthetic cohort (64x64 canvases, fake ids); also hosts the cohort helper."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from breast_seg.config import PathsConfig, apply_overrides, from_dict, read_yaml
from breast_seg.datasets import DatasetAuditError, DatasetBuilder, load_dataset_config
from breast_seg.datasets.config import DatasetConfig

ROOT = Path(__file__).resolve().parents[1]
SIZE = 64
SCALE = 0.1          # canvas = original * 0.1 -> original images are 640 x 640
SPACING = 0.1

MLO_POLY = {0: [(0, 0), (0.4, 0), (0, 0.5)],
            1: [(0, 0), (0.85, 0.05), (0.95, 0.5), (0.85, 0.95), (0, 1)],
            2: [(0.84, 0.44), (0.96, 0.44), (0.96, 0.56), (0.84, 0.56)]}
CC_POLY = {1: [(0, 0.05), (0.8, 0.1), (0.9, 0.5), (0.8, 0.9), (0, 0.95)],
           2: [(0.8, 0.44), (0.92, 0.44), (0.92, 0.56), (0.8, 0.56)]}


def poly_text(polys, skip=()):
    return "".join(f"{c} " + " ".join(f"{x} {y}" for x, y in pts) + "\n" for c, pts in polys.items() if c not in skip)


def canvas(seed):
    rng = np.random.default_rng(seed)
    img = (rng.random((SIZE, SIZE)) * 40).astype(np.uint8)
    img[:, : SIZE * 3 // 4] += 120
    return img


def nipple_box(cx, cy):
    return str({"x": cx - 5.0, "y": cy - 5.0, "width": 10.0, "height": 10.0})


def make_cohort(tmp: Path, n_studies: int = 8):
    """Fake cohort: per study two breasts, each with an MLO and a CC canvas, labels, tables and metadata.

    Returns a :class:`PathsConfig` whose ``extra`` entries match ``configs/paths.example.yaml``.
    """
    pool_mlo, pool_cc, pseudo = tmp / "pool/MLO", tmp / "pool/CC", tmp / "pool/CC_pseudo/labels"
    for d in (pool_mlo / "labels", pool_cc, pseudo, tmp / "labels", tmp / "canvas/MLO", tmp / "canvas/CC"):
        d.mkdir(parents=True, exist_ok=True)
    mlo_rows, cc_rows, mlo_meta, cc_meta, header = [], [], [], [], []
    seed = 0
    for j in range(n_studies):
        study = f"st{j}"
        split = "Test" if j == 0 else "Validation" if j == 1 else "Train"
        for side in ("L", "R"):
            m, c = f"m{j}{side}", f"c{j}{side}"
            seed += 1
            cv2.imwrite(str(pool_mlo / f"{m}.png"), canvas(seed))
            (pool_mlo / "labels" / f"{m}.txt").write_text(poly_text(MLO_POLY))
            seed += 1
            cv2.imwrite(str(pool_cc / f"{c}.png"), canvas(seed))
            (pseudo / f"{c}.txt").write_text(poly_text(CC_POLY))
            nip_m, nip_c = (580.0, 320.0), (550.0 - 10 * j, 320.0)
            pnl = nip_m[0] * SPACING                                    # pectoral line on x = 0
            dcc = (nip_c[0] if side == "L" else 640 - nip_c[0]) * SPACING
            quality = "Good" if abs(pnl - dcc) <= 10 else "Bad"
            common = {"StudyInstanceUID": study, "annotationMode": "x", "qualitativeLabel": quality, "height": 640,
                      "width": 640, "Split": split}
            mlo_rows.append({**common, "SOPInstanceUID": m, "labelName": "Nipple", "data": nipple_box(*nip_m),
                             "SeriesDescription": f"{side}-MLO"})
            mlo_rows.append({**common, "SOPInstanceUID": m, "labelName": "Pectoralis",
                             "data": str({"vertices": [[0.0, 0.0], [0.0, 400.0]]}), "SeriesDescription": f"{side}-MLO"})
            cc_rows.append({**common, "SOPInstanceUID": c, "labelName": "Nipple", "data": nipple_box(*nip_c),
                            "SeriesDescription": f"{side}-CC"})
            for sop, rows in ((m, mlo_meta), (c, cc_meta)):
                rows.append({"sop_uid": sop, "split": split.lower(), "laterality": side, "original_width": 640,
                             "original_height": 640, "crop_x1": 0, "crop_y1": 0, "crop_x2": 640, "crop_y2": 640,
                             "scale": SCALE, "pad_left": 0, "pad_top": 0, "pad_right": 0, "pad_bottom": 0,
                             "pixel_spacing": SPACING})
                header.append({"SOP Instance UID": sop, "Series Instance UID": f"se{j}"})
    pd.DataFrame(mlo_rows).to_csv(tmp / "labels/mlo_labels.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(tmp / "labels/cc_labels.csv", index=False)
    pd.DataFrame(header).to_csv(tmp / "labels/metadata.csv", index=False)
    pd.DataFrame(mlo_meta).to_csv(tmp / "canvas/MLO/metadata.csv", index=False)
    pd.DataFrame(cc_meta).to_csv(tmp / "canvas/CC/metadata.csv", index=False)
    (tmp / "exclude.txt").write_text("# broken annotations\nm7R\n")
    extra = {"mlo_labels": "labels/mlo_labels.csv", "cc_labels": "labels/cc_labels.csv",
             "mlo_metadata": "canvas/MLO/metadata.csv", "cc_metadata": "canvas/CC/metadata.csv",
             "vindr_metadata": "labels/metadata.csv", "mlo_label_pool": "pool/MLO", "cc_label_pool": "pool/CC",
             "cc_pseudo_labels": "pool/CC_pseudo/labels", "mlo_exclude_list": "exclude.txt",
             "completion_dir": "completion", "embed_test_csv": "embed_test.csv", "cv10_fold_table": "folds.csv"}
    paths_file = tmp / "paths.yaml"
    paths_file.write_text(yaml.safe_dump({"root": ".", "extra": extra}))
    return PathsConfig.load(paths_file)


def small_audit(cfg_data):
    return apply_overrides(cfg_data, {"audit.image_shape": [SIZE, SIZE]})


def load_committed(name, **overrides):
    data = small_audit(read_yaml(ROOT / "configs/datasets" / name))
    return from_dict(DatasetConfig, apply_overrides(data, overrides))


def test_committed_seg_v4_config_builds_synthetic_cohort(tmp_path):
    paths = make_cohort(tmp_path)
    report = DatasetBuilder(load_committed("seg_v4.yaml"), paths).build()
    out = paths.resolve("${data_root}/seg-dataset_v4")
    assert report.ok and report.data_yaml == out / "data.yaml"
    data = yaml.safe_load(report.data_yaml.read_text())
    assert data["names"] == ["pectoral", "breast-tissue", "nipple"] and Path(data["path"]) == out
    idx = pd.read_csv(out / "index.csv")
    assert len(idx) == 8 * 4 - 1                                  # one excluded MLO
    assert "m7R" not in set(idx.stem)
    assert set(idx[idx.view == "CC"].stem.str[:3]) == {"cc_"}
    assert set(idx[idx.split == "test"].stem) == {"m0L", "m0R", "cc_c0L", "cc_c0R"}
    # corrected labels never carry a pectoral polygon on CC and every image keeps breast tissue
    for row in idx.itertuples():
        text = (out / "labels" / row.split / f"{row.stem}.txt").read_text()
        classes = {ln.split()[0] for ln in text.splitlines()}
        assert "1" in classes and (row.view == "MLO" or "0" not in classes)
    # semantic export
    sem = paths.resolve("${data_root}/semantic_v4")
    sidx = pd.read_csv(sem / "index.csv")
    mask = cv2.imread(str(sem / "masks/train" / f"{sidx.stem[sidx.split == 'train'].iloc[0]}.png"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(mask)) <= {0, 1, 2, 3} and sidx.group.str.startswith("study:").all()
    # nnU-Net export: train+val in imagesTr, test in imagesTs, fold 0 + 5 group-level folds
    ds = paths.dir("nnunet_raw") / "Dataset601_BreastSeg"
    assert len(list((ds / "imagesTr").glob("*_0000.png"))) == len(idx[idx.split != "test"])
    assert len(list((ds / "imagesTs").glob("*_0000.png"))) == 4
    splits = yaml.safe_load((paths.dir("nnunet_preprocessed") / "Dataset601_BreastSeg/splits_final.json").read_text())
    assert len(splits) == 6
    group = dict(zip(sidx.stem, sidx.group))
    for sp in splits[1:]:
        assert not {group[s] for s in sp["train"]} & {group[s] for s in sp["val"]}


def test_leaking_split_map_fails_the_audit(tmp_path):
    paths = make_cohort(tmp_path)
    labels = paths.resolve("${cc_labels}")
    df = pd.read_csv(labels)
    df.loc[df.SOPInstanceUID == "c3L", "Split"] = "Test"          # one breast of study st3 moves to test
    df.to_csv(labels, index=False)
    with pytest.raises(DatasetAuditError, match="audit failed"):
        DatasetBuilder(load_committed("seg_v4.yaml"), paths).build()
    report = DatasetBuilder(load_committed("seg_v4.yaml"), paths).build(strict=False)
    assert not report.ok and report.data_yaml is None
    assert any("group shared" in p for p in report.problems)


def test_cc_pectoral_and_missing_breast_are_reported(tmp_path):
    paths = make_cohort(tmp_path, n_studies=3)
    (paths.resolve("${cc_pseudo_labels}") / "c2L.txt").write_text(poly_text({**CC_POLY, 0: MLO_POLY[0]}))
    (paths.resolve("${mlo_label_pool}") / "labels" / "m2L.txt").write_text(poly_text(MLO_POLY, skip=(1,)))
    cfg = load_committed("seg_v4.yaml", **{"sources": [
        {"name": "mlo", "images": "${mlo_label_pool}", "labels": "${mlo_label_pool}/labels", "split": "map"},
        {"name": "cc", "images": "${cc_label_pool}", "labels": "${cc_pseudo_labels}", "prefix": "cc_", "view": "CC",
         "split": "map"}], "exports.semantic.enabled": False, "exports.nnunet.enabled": False})
    report = DatasetBuilder(cfg, paths).build(strict=False)
    assert any("pectoral polygon on a CC view" in p for p in report.problems)
    assert any("no breast tissue" in p for p in report.problems)


def make_completion(paths: PathsConfig):
    """Manifest with human / auto / rejected buckets, an external-test image and a same-patient image."""
    root = paths.resolve("${completion_dir}")
    rows = []
    specs = [("mendeley", "a1", "MLO", "insan", "se9"), ("mendeley", "a2", "CC", "otomatik", "se9"),
             ("inbreast", "b1", "MLO", "otomatik", "p1"), ("inbreast", "b2", "MLO", "radyologa", "p1"),
             ("embed", "e1", "MLO", "insan", "pe1"), ("embed", "e2", "CC", "insan", "pe1"),
             ("embed", "e3", "MLO", "insan", ""), ("embed", "e4", "MLO", "otomatik", "pe2")]
    for k, (src, key, view, bucket, group) in enumerate(specs):
        d = root / bucket
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "images" / f"{src}_{key}.png"), canvas(1000 + k))
        (d / "labels" / f"{src}_{key}.txt").write_text(poly_text(MLO_POLY if view == "MLO" else CC_POLY))
        rows.append({"src": src, "id": key, "grup": group, "view": view, "kova": bucket})
    pd.DataFrame(rows).to_csv(root / "meta.csv", index=False)
    pd.DataFrame({"sop": ["e1"]}).to_csv(paths.resolve("${embed_test_csv}"), index=False)


def test_committed_seg_v5_config_keeps_v4_splits_and_filters_completion(tmp_path):
    paths = make_cohort(tmp_path)
    DatasetBuilder(load_committed("seg_v4.yaml", **{"exports.nnunet.enabled": False}), paths).build()
    make_completion(paths)
    report = DatasetBuilder(load_committed("seg_v5.yaml", **{"exports.nnunet.enabled": False}), paths).build()
    v4 = pd.read_csv(paths.resolve("${data_root}/seg-dataset_v4/index.csv"))
    v5 = pd.read_csv(paths.resolve("${data_root}/seg-dataset_v5/index.csv"))
    assert report.ok
    for split in ("val", "test"):
        assert set(v5[v5.split == split].stem) == set(v4[v4.split == split].stem)
    new = set(v5[v5.source != "v4"].stem)
    # radyologa bucket out; e1 is external test, e2 shares its patient, e3 has no patient id
    assert new == {"mendeley_a1", "cc_mendeley_a2", "inbreast_b1", "embed_e4"}
    assert (v5[v5.source != "v4"].split == "train").all()


def test_series_leak_into_held_out_split_is_caught(tmp_path):
    paths = make_cohort(tmp_path)
    DatasetBuilder(load_committed("seg_v4.yaml", **{"exports.nnunet.enabled": False}), paths).build()
    make_completion(paths)
    meta = paths.resolve("${completion_dir}/meta.csv")
    df = pd.read_csv(meta, keep_default_na=False)
    df.loc[df.id == "a1", "grup"] = "se0"                        # the series of the v4 test study
    df.to_csv(meta, index=False)
    report = DatasetBuilder(load_committed("seg_v5.yaml", **{"exports.nnunet.enabled": False}), paths).build(strict=False)
    assert any("group shared" in p for p in report.problems)


def test_strict_classes_reject_incomplete_new_images(tmp_path):
    paths = make_cohort(tmp_path, n_studies=3)
    DatasetBuilder(load_committed("seg_v4.yaml", **{"exports.nnunet.enabled": False}), paths).build()
    make_completion(paths)
    lbl = paths.resolve("${completion_dir}/insan/labels/mendeley_a1.txt")
    lbl.write_text(poly_text(MLO_POLY, skip=(2,)))
    report = DatasetBuilder(load_committed("seg_v5.yaml", **{"exports.nnunet.enabled": False}), paths).build(strict=False)
    assert any("missing class(es) [2]" in p for p in report.problems)


def test_ratio_split_is_joint_and_oversampling_only_in_train(tmp_path):
    paths = make_cohort(tmp_path, n_studies=10)
    cfg = from_dict(DatasetConfig, {
        "name": "ratio", "output_dir": "${data_root}/ratio", "ratio_seed": 1,
        "groups": [{"csv": ["${mlo_labels}", "${cc_labels}"], "key": "SOPInstanceUID", "group": "StudyInstanceUID"}],
        "sources": [{"name": "mlo", "images": "${mlo_label_pool}", "labels": "${mlo_label_pool}/labels", "split": "ratio",
                     "corrections": "auto"},
                    {"name": "cc", "images": "${cc_label_pool}", "labels": "${cc_pseudo_labels}", "prefix": "cc_",
                     "view": "CC", "split": "ratio", "oversample": 3, "corrections": "auto"}],
        "audit": {"image_shape": [SIZE, SIZE], "unique_pixels": True}})
    report = DatasetBuilder(cfg, paths).build()
    idx = pd.read_csv(paths.resolve("${data_root}/ratio/index.csv"), keep_default_na=False)
    assert report.ok
    by_group = idx.groupby("groups").split.nunique()
    assert (by_group == 1).all()                                  # a study never straddles two splits
    replicas = idx[idx.replica > 0]
    assert len(replicas) and (replicas.split == "train").all() and replicas.stem.str.contains("_r").all()


def test_dataset_config_validation():
    base = {"name": "x", "output_dir": "o", "sources": [{"name": "a", "images": "i", "labels": "l", "split": "train"}]}
    from_dict(DatasetConfig, base)
    from breast_seg.config import ConfigError

    for bad in ({"sources": [{"name": "a", "images": "i", "labels": "l"}]},             # split map missing
                {"sources": [{"name": "a", "type": "manifest", "split": "train"}]},      # templates missing
                {"ratios": [0.5, 0.5]},
                {"bogus": 1}):
        with pytest.raises(ConfigError):
            from_dict(DatasetConfig, {**base, **bad})
    for p in sorted((ROOT / "configs/datasets").glob("*.yaml")):
        assert isinstance(load_dataset_config(p), DatasetConfig)


def test_builder_refuses_to_overwrite_a_source(tmp_path):
    paths = make_cohort(tmp_path, n_studies=3)
    cfg = load_committed("seg_v4.yaml", output_dir="${mlo_label_pool}")
    with pytest.raises(ValueError, match="overwrite"):
        DatasetBuilder(cfg, paths).build()
    assert (paths.resolve("${mlo_label_pool}") / "m0L.png").is_file()
