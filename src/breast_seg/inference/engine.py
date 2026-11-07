"""Model loading and inference for the positioning pipeline.

The models the engine needs are picked from the registry by *role* — the MLO and CC landmark (pose) models, one
segmentation model and an optional lesion detector — loaded once and kept resident. Ultralytics predictors are not
re-entrant, so a single lock serialises all inference; callers may run :meth:`LandmarkEngine.analyse` from a
thread pool.

Everything the engine returns is in the 640 model frame (see :mod:`breast_seg.imaging.preprocess`), except
:meth:`LandmarkEngine.analyse_landmarks`, which maps the landmark record back to original pixel coordinates.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np

from ..imaging.preprocess import TARGET_SIZE, Transform, preprocess_pair
from ..imaging.render import image_quality, render_display
from ..registry import ModelSpec, Registry
from .loader import ModelLoader, ultralytics_loader

log = logging.getLogger(__name__)

MIN_MASK_AREA_PX = 20.0  # drop slivers, the same threshold the dataset builder used
POLYGON_EPS = 0.002  # Douglas-Peucker tolerance, fraction of the contour perimeter

# role -> (task, view) looked up with Registry.select
ROLES: Dict[str, Tuple[str, Optional[str]]] = {
    "pose_mlo": ("landmark", "MLO"),
    "pose_cc": ("landmark", "CC"),
    "segmentation": ("segmentation", None),
    "detection": ("detection", None),
}
REQUIRED_ROLES = ("pose_mlo", "pose_cc", "segmentation")


class EngineError(RuntimeError):
    """Inference cannot run (models not ready); the message is user-facing."""


def _np(value: Any) -> np.ndarray:
    """A torch tensor or array-like as a numpy array."""
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


class LandmarkEngine:
    """Holds the loaded models and runs one image through all of them.

    Args:
        registry: the model registry.
        device: Ultralytics device string — ``"cpu"``, ``"0"``, ``"cuda:1"`` …
        preview_size: pixel size of the square display render returned by :meth:`analyse`.
        loader: ``(spec) -> model``; defaults to :func:`~breast_seg.inference.loader.ultralytics_loader`.
    """

    def __init__(self, registry: Registry, device: str = "cpu", preview_size: int = 1280,
                 loader: Optional[ModelLoader] = None) -> None:
        self.registry = registry
        self.device = str(device)
        self.preview_size = int(preview_size)
        self._loader = loader or ultralytics_loader
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self.roles: Dict[str, Optional[ModelSpec]] = {
            role: registry.select(task, view) for role, (task, view) in ROLES.items()}

    # ── lifecycle ──────────────────────────────────────────────────────────
    def load(self) -> LandmarkEngine:
        """Load every model that fills a role. Failures are recorded in :attr:`errors`, never raised, so a service
        still starts and its health endpoint can say exactly which model is missing."""
        for role in REQUIRED_ROLES:
            if self.roles[role] is None:
                task, view = ROLES[role]
                self.errors[role] = f"no enabled {task} model{' for ' + view if view else ''} in the registry"

        for spec in {s.id: s for s in self.roles.values() if s is not None}.values():
            if spec.id in self._models:
                continue
            if not spec.available:
                log.error("%s: weights file not found: %s", spec.id, spec.weights_path)
                self.errors[spec.id] = f"weights file not found: {spec.weights} (under the weights root)"
                continue
            try:
                self._models[spec.id] = self._loader(spec)
                self.errors.pop(spec.id, None)
            except Exception as exc:  # noqa: BLE001 - recorded and reported through status()
                log.exception("%s: failed to load", spec.id)
                self.errors[spec.id] = f"failed to load: {exc}"
        return self

    @property
    def ready(self) -> bool:
        """Both pose models and the segmentation model are loaded."""
        return all(self.roles[r] is not None and self.roles[r].id in self._models for r in REQUIRED_ROLES)

    @property
    def lock(self) -> threading.Lock:
        """The inference lock (held while any model runs)."""
        return self._lock

    def status(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "device": self.device,
            "loaded": sorted(self._models),
            "errors": dict(self.errors),
            "roles": {role: (spec.id if spec else None) for role, spec in self.roles.items()},
            "models": [spec.to_dict() for spec in self.registry.models],
        }

    def loaded_weights(self) -> Dict[str, str]:
        """``{model id: weights file name}`` of the loaded models (recorded with each study)."""
        return {spec.id: spec.weights_path.name for spec in self.registry.models if spec.id in self._models}

    # ── inference ──────────────────────────────────────────────────────────
    def analyse(self, pixels: np.ndarray, meta: Mapping[str, Any]) -> Dict[str, Any]:
        """One image -> display render, quality, landmarks, masks, polygons and lesions, all in 640 space.

        ``pixels`` are as read from the file (photometric of ``meta``); ``meta`` is the dict produced by
        :func:`breast_seg.imaging.read_dicom` (``view``, ``photometric``, ``chest_side``, ``pixel_spacing``).
        """
        if not self.ready:
            raise EngineError("models are not ready")

        baseline, wavelet, transform = preprocess_pair(pixels, meta)
        view = meta["view"]

        with self._lock:
            landmarks = self._run_pose(wavelet, view)
            masks, polygons = self._run_seg(baseline, landmarks, transform)
            lesions = self._run_lesions(baseline)

        return {
            "transform": transform.as_dict(),
            "mm_per_px": transform.mm_per_640px,
            "display": render_display(pixels, meta, transform, self.preview_size),
            "quality": image_quality(baseline),
            "landmarks": landmarks,
            "masks": masks,
            "polygons": polygons,
            "lesions": lesions,
        }

    def analyse_landmarks(self, pixels: np.ndarray, meta: Mapping[str, Any]) -> Dict[str, Any]:
        """Pose + segmentation for one image -> the landmark record, in ORIGINAL pixel coordinates.

        Returns ``{"points": {name: [x, y]}, "conf": pose confidence, "polygons": {class: {"p": [[x, y], ...],
        "conf": c, "rescued": bool}}, "chest_side", "lat", "view", "W", "H"}`` — the record a cross-view matcher
        stores per image. No display render, no lesion pass.
        """
        if not self.ready:
            raise EngineError("models are not ready")
        baseline, wavelet, transform = preprocess_pair(pixels, meta)
        with self._lock:
            landmarks = self._run_pose(wavelet, meta["view"])
            _, polygons = self._run_seg(baseline, landmarks, transform)
        record = landmark_record(landmarks, polygons, transform)
        record.update(chest_side=meta["chest_side"], lat=meta["laterality"], view=meta["view"],
                      W=int(pixels.shape[1]), H=int(pixels.shape[0]))
        return record

    # ── individual models ──────────────────────────────────────────────────
    def _model(self, role: str) -> Tuple[Optional[ModelSpec], Any]:
        spec = self.roles.get(role)
        return spec, (self._models.get(spec.id) if spec else None)

    def _run_pose(self, image640: np.ndarray, view: str) -> Dict[str, Any]:
        spec, model = self._model("pose_mlo" if view == "MLO" else "pose_cc")
        points, conf, mirrored = self._predict_pose(model, image640)
        if points is None:
            return {"detected": False, "mirrored": False, "confidence": 0.0, "points": {}}

        named: Dict[str, List[float]] = {}
        for index, name in enumerate(spec.keypoints):
            if index < len(points):
                x, y = float(points[index][0]), float(points[index][1])
                # A keypoint the model could not place comes back as (0, 0).
                if x > 0.5 or y > 0.5:
                    named[name] = [round(x, 2), round(y, 2)]
        return {
            "detected": bool(named),
            "mirrored": mirrored,
            "confidence": round(conf, 4),
            "points": named,
        }

    def _predict_pose(self, model: Any, image640: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool]:
        """Predict, and retry on the mirrored image when nothing is found.

        A mirrored mammogram is an equally valid presentation, but the checkpoints are occasionally brittle on one
        of the two orientations; the mirror recovers those. Points are mapped back so the caller never sees the
        flip.
        """
        points, conf = self._pose_once(model, image640)
        if points is not None:
            return points, conf, False

        points, conf = self._pose_once(model, cv2.flip(image640, 1))
        if points is None:
            return None, 0.0, False
        points = np.asarray(points, dtype=np.float64).copy()
        points[:, 0] = (TARGET_SIZE - 1) - points[:, 0]
        return points, conf, True

    def _pose_once(self, model: Any, image640: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        result = model.predict(cv2.cvtColor(image640, cv2.COLOR_GRAY2BGR),
                               imgsz=TARGET_SIZE, device=self.device, verbose=False)[0]
        if result.keypoints is None or len(result.boxes) == 0:
            return None, 0.0
        confidences = _np(result.boxes.conf)
        best = int(np.argmax(confidences))
        return _np(result.keypoints.xy)[best], float(confidences[best])

    def _run_seg(self, image640: np.ndarray, landmarks: Mapping[str, Any], transform: Transform
                 ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        spec, model = self._model("segmentation")
        best = self._seg_pass(model, image640, spec.conf, spec.iou)

        rescue = spec.nipple_rescue
        if rescue.enabled and "nipple" not in best:
            anchor = (landmarks.get("points") or {}).get("nipple")
            if anchor:
                best_low = self._seg_pass(model, image640, rescue.conf, spec.iou)
                candidate = best_low.get("nipple")
                if candidate is not None:
                    centre = np.argwhere(candidate["mask"] > 0).mean(axis=0)[::-1]  # (x, y)
                    distance_mm = float(np.linalg.norm(centre - np.array(anchor)) * transform.mm_per_640px)
                    if distance_mm <= rescue.max_mm:
                        candidate["rescued"] = True
                        candidate["rescue_distance_mm"] = round(distance_mm, 2)
                        best["nipple"] = candidate

        masks = {name: entry["mask"] for name, entry in best.items()}
        polygons = {
            name: {
                "points": entry["polygon"],
                "confidence": entry["conf"],
                "rescued": entry.get("rescued", False),
            }
            for name, entry in best.items()
        }
        return masks, polygons

    def _seg_pass(self, model: Any, image640: np.ndarray, conf: float, iou: float) -> Dict[str, Dict[str, Any]]:
        result = model.predict(cv2.cvtColor(image640, cv2.COLOR_GRAY2BGR),
                               imgsz=TARGET_SIZE, device=self.device, conf=conf, iou=iou,
                               retina_masks=True, verbose=False)[0]
        if result.masks is None:
            return {}

        raw = _np(result.masks.data)
        classes = _np(result.boxes.cls).astype(int)
        confidences = _np(result.boxes.conf)

        # One instance per class: the anatomy is single-object, so keep the largest and let the rest go rather
        # than showing the annotator duplicate blobs.
        best: Dict[str, Dict[str, Any]] = {}
        for mask, class_id, confidence in zip(raw, classes, confidences):
            name = model.names[int(class_id)]
            binary = (mask > 0.5).astype(np.uint8)
            if binary.shape != (TARGET_SIZE, TARGET_SIZE):
                binary = cv2.resize(binary, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
            area = float(binary.sum())
            if area < MIN_MASK_AREA_PX:
                continue
            if name in best and best[name]["area"] >= area:
                continue
            best[name] = {
                "mask": binary,
                "area": area,
                "conf": round(float(confidence), 4),
                "polygon": largest_polygon(binary),
            }
        return {name: entry for name, entry in best.items() if entry["polygon"]}

    def _run_lesions(self, image640: np.ndarray) -> List[Dict[str, Any]]:
        spec, model = self._model("detection")
        if model is None or spec is None:
            return []
        result = model.predict(cv2.cvtColor(image640, cv2.COLOR_GRAY2BGR), imgsz=TARGET_SIZE,
                               device=self.device, conf=spec.conf, verbose=False)[0]
        boxes = []
        for box, class_id, confidence in zip(_np(result.boxes.xyxy), _np(result.boxes.cls).astype(int),
                                             _np(result.boxes.conf)):
            boxes.append({
                "label": model.names[int(class_id)],
                "box": [round(float(v), 1) for v in box],
                "confidence": round(float(confidence), 4),
            })
        return boxes


def landmark_record(landmarks: Mapping[str, Any], polygons: Mapping[str, Mapping[str, Any]],
                    transform: Transform) -> Dict[str, Any]:
    """Pose points and polygons in 640 space -> the same, mapped to original pixels and rounded to 0.1 px."""
    points = {name: transform.to_original(np.array([p]))[0].round(1).tolist()
              for name, p in (landmarks.get("points") or {}).items()}
    shapes: Dict[str, Dict[str, Any]] = {}
    for name, entry in polygons.items():
        if entry["points"]:
            shapes[name] = {
                "p": transform.to_original(np.array(entry["points"], dtype=float)).round(1).tolist(),
                "conf": entry["confidence"],
                "rescued": entry.get("rescued", False),
            }
    return {"points": points, "conf": landmarks.get("confidence"), "polygons": shapes}


def largest_polygon(binary: np.ndarray) -> List[List[float]]:
    """Outer contour of the largest blob of ``binary``, simplified to a draggable number of vertices."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).astype(np.float32)
    epsilon = POLYGON_EPS * cv2.arcLength(contour, True)
    reduced = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(reduced) < 3:
        reduced = contour.reshape(-1, 2)
    return [[round(float(x), 1), round(float(y), 1)] for x, y in reduced]
