"""Deterministic stand-ins for the Ultralytics models — tests and demos without any weights.

:func:`fake_loader` is a drop-in ``loader`` for :class:`~breast_seg.inference.LandmarkEngine`. The fake models
look at the image they are given (tissue = 8-bit level above 12, chest wall = the edge with more tissue) and
answer with plausible geometry: the nipple at the anterior apex, the pectoral wedge at the top of the chest wall,
the tissue as the breast mask. They expose exactly the result attributes the engine reads, as numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from ..registry import ModelSpec, Registry

TISSUE_LEVEL = 12


@dataclass
class FakeBoxes:
    conf: np.ndarray
    cls: np.ndarray
    xyxy: np.ndarray

    def __len__(self) -> int:
        return int(len(self.conf))


@dataclass
class FakeKeypoints:
    xy: np.ndarray  # (N, K, 2)


@dataclass
class FakeMasks:
    data: np.ndarray  # (N, H, W) float in [0, 1]


@dataclass
class FakeResult:
    boxes: FakeBoxes
    keypoints: Optional[FakeKeypoints] = None
    masks: Optional[FakeMasks] = None


def _empty() -> FakeResult:
    return FakeResult(boxes=FakeBoxes(np.zeros(0), np.zeros(0), np.zeros((0, 4))))


@dataclass
class _Anatomy:
    tissue: np.ndarray
    chest_side: str
    nipple: np.ndarray  # (x, y)
    pectoral_top: np.ndarray
    pectoral_bottom: np.ndarray
    box: np.ndarray  # tissue bounding box (x1, y1, x2, y2)


def find_anatomy(image: np.ndarray) -> Optional[_Anatomy]:
    """Crude landmarks of a padded 640 render (gray or BGR); None when there is no tissue."""
    gray = image[..., 0] if image.ndim == 3 else image
    tissue = gray > TISSUE_LEVEL
    if tissue.sum() < 500:
        return None
    ys, xs = np.nonzero(tissue)
    width = gray.shape[1]
    chest_side = "L" if tissue[:, :8].sum() >= tissue[:, -8:].sum() else "R"
    wall_x = float(xs.min()) if chest_side == "L" else float(xs.max())
    distance = np.abs(xs - wall_x)
    far = distance >= distance.max() - 2
    nipple = np.array([float(np.median(xs[far])), float(np.median(ys[far]))])
    top_y, bottom_y = float(ys.min()), float(ys.max())
    span = float(distance.max())
    reach = 0.42 * span
    top_x = wall_x + reach if chest_side == "L" else wall_x - reach
    pectoral_top = np.array([min(max(top_x, 0.0), width - 1.0), top_y])
    pectoral_bottom = np.array([wall_x, top_y + 0.62 * (bottom_y - top_y)])
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=float)
    return _Anatomy(tissue, chest_side, nipple, pectoral_top, pectoral_bottom, box)


@dataclass
class FakeModel:
    """A model with the slice of the Ultralytics interface the engine uses.

    Args:
        task: ``landmark`` | ``segmentation`` | ``detection``.
        names: class names (``model.names``).
        keypoints: keypoint names of a landmark model, in output order.
        nipple_conf: confidence the segmentation model gives its nipple instance (below the normal threshold
            to exercise the nipple rescue).
        find_nothing: answer with no detection at all (to exercise the mirror fallback / missing labels).
        only_mirrored: find landmarks only when the chest wall is on the right (the mirrored presentation).
    """

    task: str
    names: Dict[int, str] = field(default_factory=dict)
    keypoints: Sequence[str] = ()
    nipple_conf: float = 0.8
    find_nothing: bool = False
    only_mirrored: bool = False
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def predict(self, image: np.ndarray, **kwargs: Any) -> List[FakeResult]:
        self.calls.append(dict(kwargs, shape=tuple(image.shape)))
        anatomy = None if self.find_nothing else find_anatomy(image)
        if anatomy is None or (self.only_mirrored and anatomy.chest_side != "R"):
            return [_empty()]
        if self.task == "landmark":
            return [self._pose(anatomy)]
        if self.task == "segmentation":
            return [self._segment(anatomy, float(kwargs.get("conf", 0.25)), image.shape[:2])]
        return [self._detect(anatomy, float(kwargs.get("conf", 0.25)))]

    def _pose(self, anatomy: _Anatomy) -> FakeResult:
        named = {"nipple": anatomy.nipple, "pectoral_top": anatomy.pectoral_top,
                 "pectoral_bottom": anatomy.pectoral_bottom}
        points = np.array([[named.get(name, np.zeros(2)) for name in self.keypoints]], dtype=np.float32)
        return FakeResult(boxes=FakeBoxes(np.array([0.9]), np.array([0.0]), anatomy.box[None, :]),
                          keypoints=FakeKeypoints(points))

    def _segment(self, anatomy: _Anatomy, conf: float, shape: Sequence[int]) -> FakeResult:
        height, width = shape
        index = {name: i for i, name in self.names.items()}
        masks, classes, scores = [], [], []

        breast = anatomy.tissue.astype(np.float32)
        masks.append(breast)
        classes.append(index.get("breast-tissue", 1))
        scores.append(0.95)

        wedge = np.zeros((height, width), np.uint8)
        wall = (anatomy.pectoral_bottom[0], anatomy.pectoral_top[1])
        triangle = np.array([wall, anatomy.pectoral_top, anatomy.pectoral_bottom]).round().astype(np.int32)
        cv2.fillPoly(wedge, [triangle], 1)
        masks.append((wedge.astype(bool) & anatomy.tissue).astype(np.float32))
        classes.append(index.get("pectoral", 0))
        scores.append(0.9)

        nipple = np.zeros((height, width), np.uint8)
        cv2.circle(nipple, (int(round(anatomy.nipple[0])), int(round(anatomy.nipple[1]))), 6, 1, -1)
        masks.append(nipple.astype(np.float32))
        classes.append(index.get("nipple", 2))
        scores.append(self.nipple_conf)

        keep = [i for i, score in enumerate(scores) if score >= conf]
        data = np.stack([masks[i] for i in keep]) if keep else np.zeros((0, height, width), np.float32)
        return FakeResult(
            boxes=FakeBoxes(np.array([scores[i] for i in keep]), np.array([classes[i] for i in keep], dtype=float),
                            np.repeat(anatomy.box[None, :], len(keep), axis=0)),
            masks=FakeMasks(data) if keep else None)

    def _detect(self, anatomy: _Anatomy, conf: float) -> FakeResult:
        score = 0.6
        if score < conf:
            return _empty()
        x1, y1, x2, y2 = anatomy.box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box = np.array([[cx - 20, cy - 15, cx + 20, cy + 15]], dtype=float)
        return FakeResult(boxes=FakeBoxes(np.array([score]), np.array([0.0]), box))


def fake_loader(**options: Any):
    """A ``loader`` building a :class:`FakeModel` per spec; ``options`` are passed to every segmentation model
    (``nipple_conf``) or landmark model (``find_nothing``, ``only_mirrored``) as relevant."""
    built: Dict[str, FakeModel] = {}

    def load(spec: ModelSpec) -> FakeModel:
        if spec.task == "landmark":
            model = FakeModel("landmark", names={0: "breast"}, keypoints=spec.keypoints,
                              find_nothing=bool(options.get("find_nothing", False)),
                              only_mirrored=bool(options.get("only_mirrored", False)))
        elif spec.task == "segmentation":
            names = dict(enumerate(spec.classes or ("pectoral", "breast-tissue", "nipple")))
            model = FakeModel("segmentation", names=names, nipple_conf=float(options.get("nipple_conf", 0.8)))
        else:
            model = FakeModel("detection", names=dict(enumerate(spec.classes or ("Mass", "Calcification"))))
        built[spec.id] = model
        return model

    load.models = built  # type: ignore[attr-defined]
    return load


def placeholder_weights(registry_path: Union[str, Path], root: Union[str, Path]) -> Path:
    """Create an empty file at every weights path of the registry under ``root`` (so every model is "available"
    to a :func:`fake_loader` engine); returns ``root``."""
    root = Path(root)
    for spec in Registry.load(registry_path, weights_root=root).models:
        spec.weights_path.parent.mkdir(parents=True, exist_ok=True)
        spec.weights_path.touch()
    return root
