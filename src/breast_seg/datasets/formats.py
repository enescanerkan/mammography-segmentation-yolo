"""Exporters from a built YOLO dataset to the layouts the other model families read.

* YOLO: ``images/<split>/<stem>.png``, ``labels/<split>/<stem>.txt`` and ``data.yaml`` (:func:`write_yolo_data_yaml`).
* Semantic (SegFormer, U-Net, AttUNet): ``images/<split>/<stem>.png`` (grey), ``masks/<split>/<stem>.png``
  (label map ``0 bg / 1 pectoral / 2 breast-tissue / 3 nipple``) and ``index.csv`` (:class:`SemanticExporter`).
* nnU-Net v2 raw: ``Dataset<ID>_<name>/{imagesTr,labelsTr,imagesTs,labelsTs}`` with ``<stem>_0000.png`` inputs,
  ``dataset.json`` and ``splits_final.json`` (:class:`NnUNetExporter`).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
import pandas as pd
import yaml

from breast_seg.config import LABEL_VALUES, YOLO_CLASSES
from breast_seg.datasets.labels import rasterise, read_polygons

PathLike = Union[str, Path]


def write_yolo_data_yaml(path: PathLike, root: PathLike, names: Sequence[str] = tuple(YOLO_CLASSES.values()),
                         kpt_shape: Optional[Sequence[int]] = None, flip_idx: Optional[Sequence[int]] = None,
                         splits: Mapping[str, Optional[str]] = None) -> Path:
    """Write an Ultralytics ``data.yaml`` (``path`` = absolute ``root``; the file stays local, it is never committed)."""
    splits = dict(splits or {"train": "images/train", "val": "images/val", "test": "images/test"})
    data: Dict[str, object] = {"path": str(Path(root).resolve())}
    data.update({k: v for k, v in splits.items() if v})
    data["nc"] = len(names)
    data["names"] = {i: n for i, n in enumerate(names)} if kpt_shape else list(names)
    if kpt_shape:
        data["kpt_shape"] = list(kpt_shape)
    if flip_idx:
        data["flip_idx"] = list(flip_idx)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def link_or_copy(src: PathLike, dst: PathLike, mode: str = "symlink") -> None:
    """Place ``src`` at ``dst`` by ``symlink`` (absolute target), ``hardlink`` or ``copy``."""
    src, dst = Path(src).resolve(), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copyfile(src, dst)
    else:
        raise ValueError(f"unknown link mode {mode!r}")


def view_of(stem: str, cc_prefix: str = "cc_") -> str:
    """Dataset naming rule: CC files carry the ``cc_`` prefix, everything else is MLO."""
    return "CC" if stem.startswith(cc_prefix) else "MLO"


@dataclass
class SemanticExporter:
    """YOLO polygons -> semantic label maps (painted breast, pectoral, nipple) next to grey copies of the images."""

    output_dir: Path

    def export(self, items: Iterable[Mapping[str, object]]) -> pd.DataFrame:
        """``items``: mappings with ``stem``, ``split``, ``image``, ``label`` and optionally ``view`` / ``source`` /
        ``group``.

        Returns (and writes) ``index.csv``: stem, split, view, source, group, height, width, ``has_<class>``,
        ``px_<class>``.
        """
        rows = []
        names = {v: n for v, n in LABEL_VALUES.items() if v}
        for it in items:
            stem, split = str(it["stem"]), str(it["split"])
            img = cv2.imread(str(it["image"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"unreadable image for {split}/{stem}")
            h, w = img.shape
            label, _ = rasterise(read_polygons(it["label"]), h, w)
            for sub, arr in (("images", img), ("masks", label)):
                d = self.output_dir / sub / split
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / f"{stem}.png"), arr)
            present = set(np.unique(label).tolist())
            rows.append({"stem": stem, "split": split, "view": it.get("view") or view_of(stem),
                         "source": it.get("source", ""), "group": it.get("group", ""), "height": h, "width": w,
                         **{f"has_{n}": int(v in present) for v, n in names.items()},
                         **{f"px_{n}": int((label == v).sum()) for v, n in names.items()}})
        df = pd.DataFrame(rows)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_dir / "index.csv", index=False)
        return df


def semantic_paths(root: PathLike, index: pd.DataFrame) -> pd.DataFrame:
    """Add absolute ``image`` / ``mask`` columns to a semantic ``index.csv`` frame (keeps existing ones)."""
    root = Path(root)
    df = index.copy()
    if "image" not in df:
        df["image"] = [str(root / "images" / sp / f"{s}.png") for s, sp in zip(df.stem, df.split)]
    if "mask" not in df:
        df["mask"] = [str(root / "masks" / sp / f"{s}.png") for s, sp in zip(df.stem, df.split)]
    return df


def study_folds(stems: Sequence[str], group_of: Callable[[str], str], n_folds: int, seed: int = 42) -> List[Dict[str, List[str]]]:
    """Group-level k-fold over ``stems`` (shuffled sorted groups, ``numpy.array_split``) -> nnU-Net split dicts."""
    pool = sorted(stems)
    groups = sorted({group_of(s) for s in pool})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    chunks = np.array_split(np.asarray(groups, dtype=object), n_folds)
    out = []
    for k in range(n_folds):
        held = set(chunks[k].tolist())
        out.append({"train": [s for s in pool if group_of(s) not in held], "val": [s for s in pool if group_of(s) in held]})
    return out


@dataclass
class NnUNetExporter:
    """Semantic dataset -> nnU-Net v2 raw dataset.

    Train and validation images go to ``imagesTr``; which of them validate is decided by ``splits_final.json``
    (written to ``<preprocessed>/<dataset>/``). Fold 0 is the dataset's own train/val split; ``cv_folds`` extra
    group-level folds over train+val may follow. Test images go to ``imagesTs`` and are never preprocessed.
    """

    raw_dir: Path
    preprocessed_dir: Path
    dataset_id: int
    dataset_name: str
    description: str = ""
    link_mode: str = "copy"

    @property
    def dataset_folder(self) -> str:
        return f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / self.dataset_folder

    def dataset_json(self, n_training: int) -> Dict[str, object]:
        return {"channel_names": {"0": "greyscale"},
                "labels": {name: value for value, name in LABEL_VALUES.items()},
                "numTraining": n_training, "file_ending": ".png",
                "overwrite_image_reader_writer": "NaturalImage2DIO",
                "description": self.description}

    def export(self, index: pd.DataFrame, cv_folds: int = 0, cv_seed: int = 42,
               group_of: Optional[Mapping[str, str]] = None, splits: Optional[List[Dict[str, List[str]]]] = None,
               extra_test_dirs: Optional[Mapping[str, Sequence[str]]] = None) -> Path:
        """Write the raw dataset from an index with ``stem, split, image, mask`` columns.

        ``splits`` replaces the default fold list (used by cross-validation); ``extra_test_dirs`` maps a folder
        name (e.g. ``imagesTest_fold3``) to the stems linked into it (unlabelled images allowed, ``mask`` may be
        empty for them).
        """
        ds = self.dataset_dir
        if ds.exists():
            shutil.rmtree(ds)
        for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
            (ds / sub).mkdir(parents=True)
        labelled = index[index["mask"].fillna("").astype(str) != ""] if "mask" in index else index
        by_split = {sp: sorted(labelled.stem[labelled.split == sp]) for sp in ("train", "val", "test")}
        for r in labelled.itertuples(index=False):
            tr = "Ts" if r.split == "test" else "Tr"
            link_or_copy(r.image, ds / f"images{tr}" / f"{r.stem}_0000.png", self.link_mode)
            link_or_copy(r.mask, ds / f"labels{tr}" / f"{r.stem}.png", self.link_mode)
        image_of = dict(zip(index.stem, index.image))
        for folder, stems in (extra_test_dirs or {}).items():
            for s in stems:
                link_or_copy(image_of[s], ds / folder / f"{s}_0000.png", "symlink")
        n_tr = len(by_split["train"]) + len(by_split["val"])
        if splits is None:
            splits = [{"train": by_split["train"], "val": by_split["val"]}]
            if cv_folds:
                groups = dict(group_of or {})
                splits += study_folds(by_split["train"] + by_split["val"], lambda s: groups.get(s) or s, cv_folds, cv_seed)
        else:
            tr_stems = set(by_split["train"]) | set(by_split["val"])
            for k, sp in enumerate(splits):
                stray = (set(sp["train"]) | set(sp["val"])) - tr_stems
                if stray:
                    raise ValueError(f"split {k} names {len(stray)} image(s) that are not in imagesTr")
        (ds / "dataset.json").write_text(json.dumps(self.dataset_json(n_tr), indent=2))
        pre = self.preprocessed_dir / self.dataset_folder
        pre.mkdir(parents=True, exist_ok=True)
        (pre / "splits_final.json").write_text(json.dumps(splits, indent=2))
        return ds
