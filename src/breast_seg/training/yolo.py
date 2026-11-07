"""Ultralytics YOLO families: ``yolo_seg`` (breast / pectoral / nipple instance segmentation), ``yolo_pose``
(MLO / CC landmark models) and ``yolo_detect`` (lesion detector)."""
from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

from breast_seg.datasets.formats import view_of, write_yolo_data_yaml
from breast_seg.datasets.labels import rasterise, read_polygons
from breast_seg.datasets.splits import SPLITS
from breast_seg.training.augment import ultralytics_albumentations, validate_specs
from breast_seg.training.base import SplitSpec, Trainer, TrainResult, register_trainer
from breast_seg.training.common import dice_report, materialise_split, stems_of
from breast_seg.training.config import YoloTrainConfig


class YoloTrainer(Trainer):
    """Shared Ultralytics logic; subclasses fix ``task`` and the prediction format."""

    config_class = YoloTrainConfig
    task: ClassVar[str] = ""
    config: YoloTrainConfig

    # ── inputs ─────────────────────────────────────────────────────────────────────────────────────────────

    @property
    def dataset_dir(self) -> Path:
        return self.resolve(self.config.data.dataset_dir)

    @property
    def data_yaml(self) -> Path:
        return self.run_dir / "data.yaml"

    def model_source(self) -> str:
        """Base weights: a file under ``weights_dir`` (or a resolvable path) if present, else the bare name, which
        Ultralytics downloads."""
        m = self.config.model
        for cand in (self.paths.dir("weights_dir") / m, self.resolve(m)):
            if cand.is_file():
                return str(cand)
        return m

    def train_kwargs(self) -> Dict[str, Any]:
        """Exactly the keyword arguments passed to ``YOLO.train``."""
        c = self.config
        hyp = {k: v for k, v in dataclasses.asdict(c.hyp).items() if k != "extra" and v is not None}
        kw: Dict[str, Any] = {"data": str(self.data_yaml), "device": c.device, "workers": c.workers, "seed": c.seed,
                              "project": str(self.run_dir.parent), "name": c.name, "exist_ok": True}
        kw.update(hyp)
        kw.update(c.hyp.extra)
        return kw

    def prepare(self) -> Path:
        d = self.config.data
        root = self.dataset_dir
        for sub in (d.train, d.val):
            if not (root / sub).is_dir():
                raise FileNotFoundError(f"{self.config.name}: dataset split folder {sub} not found under data.dataset_dir")
        if self.config.albumentations.enabled:
            validate_specs(self.config.albumentations.transforms)
        splits = {"train": d.train, "val": d.val, "test": d.test if d.test and (root / d.test).is_dir() else None}
        write_yolo_data_yaml(self.data_yaml, root, d.names, d.kpt_shape, d.flip_idx, splits)
        self.save_config()
        return self.data_yaml

    # ── training ───────────────────────────────────────────────────────────────────────────────────────────

    def fit(self) -> TrainResult:
        self.prepare()
        from ultralytics import YOLO

        if self.config.cudnn_benchmark is not None:
            import torch

            torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        model = YOLO(self.model_source())
        if self.config.albumentations.enabled:
            with ultralytics_albumentations(self.config.albumentations.transforms):
                results = model.train(**self.train_kwargs())
        else:
            results = model.train(**self.train_kwargs())
        metrics = dict(getattr(results, "results_dict", {}) or {})
        return TrainResult(self.run_dir, self.best_weights(), metrics)

    def best_weights(self) -> Path:
        return self.run_dir / "weights" / f"{self.config.export.checkpoint}.pt"

    def _model(self, weights: Optional[Path]):
        from ultralytics import YOLO

        return YOLO(str(weights or self.best_weights()))

    def ultralytics_val(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, float]:
        if not self.data_yaml.is_file():
            self.prepare()
        r = self._model(weights).val(data=str(self.data_yaml), split=split, imgsz=self.config.predict.imgsz,
                                     device=self.config.device, plots=False, verbose=False)
        return {k: float(v) for k, v in (getattr(r, "results_dict", {}) or {}).items()}

    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        return {"ultralytics": self.ultralytics_val(split, weights)}

    def split_images(self, split: str) -> List[Path]:
        sub = {"train": self.config.data.train, "val": self.config.data.val, "test": self.config.data.test}[split]
        return sorted((self.dataset_dir / sub).glob("*.png")) if sub else []

    # ── cross-validation ───────────────────────────────────────────────────────────────────────────────────

    def use_split(self, spec: SplitSpec, work_dir: Path) -> YoloTrainer:
        """Link the fold's images / labels into ``work_dir`` (Ultralytics finds a label by replacing ``/images/``
        with ``/labels/`` in the image path, so both trees are linked) and point the config at it. Unlabelled
        images are only allowed in ``test``."""
        for stem, split in spec.split_of.items():
            if split in ("train", "val") and stem not in spec.labels:
                raise ValueError(f"{stem}: unlabelled image in {split}")
        materialise_split(work_dir, spec.split_of, spec.images, "images", ".png")
        materialise_split(work_dir, spec.split_of, spec.labels, "labels", ".txt")
        for split in SPLITS:
            (work_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        data = dataclasses.replace(self.config.data, dataset_dir=str(work_dir), train="images/train",
                                   val="images/val", test="images/test")
        return type(self)(dataclasses.replace(self.config, data=data), self.paths)


@register_trainer("yolo_seg")
class YoloSegTrainer(YoloTrainer):
    """Instance segmentation (pectoral, breast tissue, nipple); predictions are flattened to label maps."""

    task = "segment"

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None) -> List[Path]:
        from breast_seg.evaluation.predictions import FlattenStats, instances_to_label_map, write_label_map

        p = self.config.predict
        model = self._model(weights)
        query = min(p.conf, p.floor) if p.floor > 0 else p.conf
        out_dir = Path(out_dir)
        written, stats = [], FlattenStats()
        images = [Path(i) for i in images]
        for i in range(0, len(images), p.batch):
            chunk = images[i:i + p.batch]
            kwargs = {"conf": query, "imgsz": p.imgsz, "device": self.config.device, "verbose": False}
            if p.iou is not None:
                kwargs["iou"] = p.iou
            results = model.predict([str(x) for x in chunk], **kwargs)
            if len(results) != len(chunk):
                raise RuntimeError(f"{len(chunk)} images in, {len(results)} results out: refusing to misalign stems")
            for path, res in zip(chunk, results):
                masks = res.masks.data.cpu().numpy() if res.masks is not None else None
                cls = res.boxes.cls.cpu().numpy() if res.boxes is not None else None
                conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else None
                label, st = instances_to_label_map(masks, cls, conf, tuple(res.orig_shape), p.conf, p.floor,
                                                   is_cc=view_of(path.stem) == "CC")
                stats.add(st)
                written.append(write_label_map(out_dir / f"{path.stem}.png", label))
        (out_dir / "flatten_stats.csv").write_text(
            "value,rescued,dropped\n" + "".join(f"{v},{stats.rescued[v]},{stats.dropped[v]}\n" for v in stats.rescued))
        return written

    def reference_map(self, split: str):
        root, sub = self.dataset_dir, {"train": "labels/train", "val": "labels/val", "test": "labels/test"}[split]
        img_sub = {"train": self.config.data.train, "val": self.config.data.val, "test": self.config.data.test}[split]

        def load(stem: str):
            import cv2

            lbl, img = root / sub / f"{stem}.txt", root / img_sub / f"{stem}.png"
            if not lbl.is_file():
                return None
            shape = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE).shape
            return rasterise(read_polygons(lbl), *shape)[0]

        return load

    def evaluate(self, split: str = "val", weights: Optional[Path] = None) -> Dict[str, Any]:
        """Mean per-image Dice by view from the flattened label maps, plus Ultralytics' own validation metrics."""
        images = self.split_images(split)
        out = self.run_dir / "predictions" / split
        self.predict(images, out, weights)
        return {"dice": dice_report(out, stems_of(images), self.reference_map(split)),
                "ultralytics": self.ultralytics_val(split, weights)}


@register_trainer("yolo_pose")
class YoloPoseTrainer(YoloTrainer):
    """Landmark models (one ``Breast`` box with keypoints). Trained on wavelet-enhanced 640 canvases."""

    task = "pose"

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None) -> List[Path]:
        """``landmarks.csv`` (``stem, role, x, y, conf``; canvas pixels) from the most confident detection."""
        p = self.config.predict
        names = self.keypoint_names()
        model = self._model(weights)
        rows = []
        for path in [Path(i) for i in images]:
            res = model.predict(str(path), conf=p.conf, imgsz=p.imgsz, device=self.config.device, verbose=False)[0]
            if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
                continue
            best = int(res.boxes.conf.argmax().item()) if res.boxes is not None and len(res.boxes) else 0
            kp = res.keypoints.xy[best].cpu().numpy()
            score = float(res.boxes.conf[best]) if res.boxes is not None and len(res.boxes) else float("nan")
            rows += [{"stem": path.stem, "role": names[k], "x": float(kp[k][0]), "y": float(kp[k][1]), "conf": score}
                     for k in range(min(len(names), len(kp)))]
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "landmarks.csv"
        with open(target, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["stem", "role", "x", "y", "conf"])
            w.writeheader()
            w.writerows(rows)
        return [target]

    def keypoint_names(self) -> List[str]:
        names = list(self.config.data.keypoint_names)
        n = self.config.data.kpt_shape[0] if self.config.data.kpt_shape else len(names)
        return names if len(names) == n else [f"kp{i}" for i in range(n)]


@register_trainer("yolo_detect")
class YoloDetectTrainer(YoloTrainer):
    """Box detector (lesion classes)."""

    task = "detect"

    def predict(self, images: Sequence[Path], out_dir: Path, weights: Optional[Path] = None) -> List[Path]:
        """``detections.csv`` (``stem, class, conf, x0, y0, x1, y1``; image pixels)."""
        p = self.config.predict
        model = self._model(weights)
        rows = []
        for path in [Path(i) for i in images]:
            res = model.predict(str(path), conf=p.conf, imgsz=p.imgsz, device=self.config.device, verbose=False)[0]
            if res.boxes is None:
                continue
            for xyxy, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                rows.append({"stem": path.stem, "class": self.config.data.names[int(c)], "conf": float(s),
                             "x0": float(xyxy[0]), "y0": float(xyxy[1]), "x1": float(xyxy[2]), "y1": float(xyxy[3])})
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "detections.csv"
        with open(target, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["stem", "class", "conf", "x0", "y0", "x1", "y1"])
            w.writeheader()
            w.writerows(rows)
        return [target]
