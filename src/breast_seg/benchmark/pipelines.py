"""The timed pipelines of the runtime benchmark, from one MLO-CC pair of DICOM files to the adequacy decision.

Stages of one pair (both views):

``read``   pydicom + pixel decoding + VOI LUT (:func:`read_dicom_file`, shared by every pipeline)
``pre``    the pipeline's released preprocessing and its model input
``infer``  the forward passes including host <-> device copies (CUDA synchronised) and the decoding of the network
           output into label maps (YOLO: Ultralytics' own pre- and post-processing is inside)
``post``   landmarks -> millimetres -> ``delta = |PNL - D_CC|`` -> decision (``delta <= 10 mm`` adequate)

The four segmentation pipelines share the 640 canvas (:func:`breast_seg.imaging.preprocess.preprocess` without
enhancement; chest-wall side as the app's DICOM reader resolves it) and the measurement rule of the paper's
evaluation: pectoral line by :func:`breast_seg.geometry.fit_pectoral_line`, nipple by
:func:`breast_seg.evaluation.landmarks.nipple_point` with the nipple probability of the same forward pass (YOLO: its
low-confidence nipple instance), PNL = distance to the infinite pectoral line, D_CC = distance to the chest-wall
edge of the CC image (the side of the pixel data, so a mirrored image with ``FieldOfViewHorizontalFlip=YES`` is
measured from the correct edge).

Where one model serves both views, the two views (and their flips) go through one batched forward pass;
``Setting.batched = False`` runs every image (and every flip) on its own. Weights are loaded once.

Precision is that of the evaluation code the paper timed: on the GPU the training config's autocast (SegFormer and
U-Net: ``optim.amp_dtype``, bfloat16), nnU-Net's float16 autocast, YOLO in float32; on the CPU everything in
float32 (the trainers' own ``predict`` would autocast on the CPU too; ``cpu_autocast`` reproduces that).

Where the nipple fallback is counted (only views whose arg-max map has no nipple pixel; the stage split follows the
research script and is informational):

* SegFormer-B3, U-Net: the nipple probability stays on the device after ``infer``; it is copied to the host, cast to
  float16 and binarised inside ``post``.
* nnU-Net: the probabilities are exported (``return_probabilities``) inside ``post``.
* YOLO26m-seg: the second pass at :data:`YOLO_NIPPLE_FALLBACK_CONF` runs inside ``infer``.
* CRAUNet regresses its points and has no fallback.

A probability map that is zero everywhere after the float16 cast carries no nipple: the view has no measurement.
"""
from __future__ import annotations

import contextlib
import itertools
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from breast_seg.benchmark.protocol import Pair, Setting
from breast_seg.evaluation.landmarks import chest_wall_mm, nipple_point, pnl_mm
from breast_seg.evaluation.rule import THRESHOLD_MM
from breast_seg.geometry import fit_pectoral_line
from breast_seg.imaging.dicom import DEFAULT_PIXEL_SPACING_MM, _apply_voi_lut, resolve_chest_side, resolve_laterality
from breast_seg.imaging.preprocess import TARGET_SIZE, Transform, chest_wall_side, preprocess

#: Confidence of the second YOLO pass that looks for a nipple instance when the label map has none.
YOLO_NIPPLE_FALLBACK_CONF = 0.001

NippleFallback = Callable[[int], Optional[np.ndarray]]
Sources = Tuple[str, str]


# ── shared stages ─────────────────────────────────────────────────────────────────────────────────────────────

def read_dicom_file(path: str) -> Tuple[np.ndarray, Any]:
    """``(float32 pixels after the VOI LUT, dataset)`` of one DICOM file."""
    import pydicom

    ds = pydicom.dcmread(str(path))
    return _apply_voi_lut(ds.pixel_array, ds).astype(np.float32), ds


def canvas_meta(pixels: np.ndarray, ds: Any, filename: str = "") -> Dict[str, Any]:
    """What :func:`preprocess` needs, as :func:`breast_seg.imaging.read_dicom` builds it: photometric, pixel spacing
    (mm) and the chest-wall side of the pixel data (:func:`breast_seg.imaging.dicom.resolve_chest_side`)."""
    spacing = getattr(ds, "ImagerPixelSpacing", None) or getattr(ds, "PixelSpacing", None)
    side = chest_wall_side(ds)  # from ImageLaterality and the flip flag; the same answer resolve_chest_side gives
    if side not in ("L", "R"):  # no tag: the reader's fallback (pixels after MONOCHROME1 inversion, then laterality)
        side = resolve_chest_side(ds, pixels, resolve_laterality(ds, filename)[0] or "U")
    return {"photometric": str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")),
            "pixel_spacing": float(spacing[0]) if spacing else DEFAULT_PIXEL_SPACING_MM, "chest_side": side}


def canvas(pixels: np.ndarray, ds: Any, filename: str = "") -> Tuple[np.ndarray, Transform, Dict[str, Any]]:
    """The released 640 canvas of the segmentation models (no enhancement), its transform and the header meta."""
    meta = canvas_meta(pixels, ds, filename)
    img, transform = preprocess(pixels, meta, enhance=False)
    return img, transform, meta


def measure_pair(label_mlo: np.ndarray, t_mlo: Transform, label_cc: np.ndarray, t_cc: Transform, chest_side_cc: str,
                 fallback: NippleFallback) -> Tuple[float, float, Sources]:
    """``(PNL mm, D_CC mm, (nipple source MLO, CC))``; NaN where a landmark is missing. ``chest_side_cc`` is the
    chest-wall side of the CC pixel data ('L' = left edge). ``fallback(i)`` is called only for a view (0 = MLO,
    1 = CC) whose label map has no nipple pixel; it may return None (no fallback)."""
    pnl = dcc = float("nan")
    nip, src_mlo = nipple_point(label_mlo) if (label_mlo == 3).any() else nipple_point(label_mlo, fallback(0))
    line = fit_pectoral_line((label_mlo == 1).astype(np.uint8), (label_mlo == 2).astype(np.uint8), label_mlo.shape)
    if line is not None and nip is not None:
        f = t_mlo.to_original
        pnl = pnl_mm(f(nip)[0], f(line.point_a)[0], f(line.point_b)[0], t_mlo.pixel_spacing)
    nc, src_cc = nipple_point(label_cc) if (label_cc == 3).any() else nipple_point(label_cc, fallback(1))
    if nc is not None:
        dcc = chest_wall_mm(t_cc.to_original(nc)[0], chest_side_cc, t_cc.original_width, t_cc.pixel_spacing)
    return pnl, dcc, (src_mlo, src_cc)


def probability_or_none(q: np.ndarray) -> Optional[np.ndarray]:
    """A float16 nipple-probability map, or None when it is zero everywhere (then it carries no nipple)."""
    return q if q.any() else None


def decide(delta: float) -> str:
    """``adequate`` (delta <= 10 mm), ``inadequate`` or ``no measurement``."""
    if not np.isfinite(delta):
        return "no measurement"
    return "adequate" if delta <= THRESHOLD_MM else "inadequate"


# ── device and model helpers ──────────────────────────────────────────────────────────────────────────────────

class Device:
    """The session's torch device (``cuda:0`` of the visible GPUs, or the CPU) and its thread setting."""

    def __init__(self, gpu: bool, threads: int):
        import torch

        self.gpu, self.threads = gpu, int(threads)
        self.torch = torch.device("cuda:0" if gpu else "cpu")
        self.name = torch.cuda.get_device_name(0) if gpu else f"CPU, {self.threads} threads"
        torch.backends.cudnn.benchmark = True

    def set_threads(self) -> None:
        import cv2
        import torch

        torch.set_num_threads(self.threads)
        cv2.setNumThreads(self.threads)

    def threads_now(self) -> Dict[str, int]:
        import cv2
        import torch

        return {"torch": torch.get_num_threads(), "cv2": cv2.getNumThreads()}

    def sync(self) -> None:
        if self.gpu:
            import torch

            torch.cuda.synchronize()

    def memory_allocated(self) -> int:
        if not self.gpu:
            return 0
        import torch

        return torch.cuda.memory_allocated()

    def reset_peak(self) -> None:
        if self.gpu:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def max_memory_allocated(self) -> int:
        if not self.gpu:
            return 0
        import torch

        return torch.cuda.max_memory_allocated()


def n_params(model: Any) -> float:
    """Parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def file_mb(*paths: Path) -> float:
    total = 0
    for p in paths:
        p = Path(p)
        total += sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size
    return total / 1e6


def count_gflops(run: Callable[[], Any], enabled: bool = True) -> Optional[float]:
    """GFLOPs of one call of ``run`` (torch's FlopCounterMode), or None."""
    if not enabled:
        return None
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode

        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            run()
        return fc.get_total_flops() / 1e9
    except Exception as exc:  # noqa: BLE001 - a FLOP count is informative only
        print("FLOP count failed:", exc, flush=True)
        return None


class Pipeline:
    """One pipeline: ``load`` once, then per pair ``prepare`` (pre), ``infer`` and ``measure`` (post)."""

    name: str = ""
    family: str = ""
    #: True when the pipeline has a with- and a without-flip setting.
    flip_option: bool = False

    def load(self) -> Dict[str, Any]:
        """Load the model(s); return their description (``params_M`` is required)."""
        raise NotImplementedError

    def prepare(self, raw: Tuple[Tuple[np.ndarray, Any], Tuple[np.ndarray, Any]], pair: Pair) -> Any:
        raise NotImplementedError

    def infer(self, prepared: Any, setting: Setting) -> Any:
        raise NotImplementedError

    def measure(self, output: Any, prepared: Any, pair: Pair) -> Tuple[float, float, Sources]:
        raise NotImplementedError

    def setting_info(self, setting: Setting) -> Dict[str, Any]:
        """Forward passes of one pair under ``setting``."""
        return {"tta": setting.tta, "batched": setting.batched, "forwards_per_pair": 4 if setting.tta else 2,
                "forward_calls_per_pair": 1 if setting.batched else (4 if setting.tta else 2)}


class CanvasPipeline(Pipeline):
    """A segmentation model on the shared 640 canvas, measured by :func:`measure_pair`."""

    def __init__(self, name: str, trainer: Any, weights: Path, device: Device, flops: bool = True,
                 cpu_autocast: bool = False):
        self.name, self.trainer, self.weights, self.device, self.flops = name, trainer, Path(weights), device, flops
        self.cpu_autocast = cpu_autocast

    def model_input(self, im: np.ndarray, ic: np.ndarray) -> Any:
        raise NotImplementedError

    def prepare(self, raw, pair):
        (pm, dm), (pc, dc) = raw
        im, tm, _ = canvas(pm, dm, pair.mlo)
        ic, tc, meta_cc = canvas(pc, dc, pair.cc)
        return self.model_input(im, ic), tm, tc, meta_cc["chest_side"]

    def infer(self, prepared, setting):
        return self.run_model(prepared[0], setting)

    def run_model(self, x: Any, setting: Setting) -> Tuple[np.ndarray, np.ndarray, NippleFallback]:
        """Model input of both views -> (label map MLO, label map CC, nipple fallback of view i)."""
        raise NotImplementedError

    def measure(self, output, prepared, pair):
        lab_mlo, lab_cc, fallback = output
        chest = prepared[3] if prepared[3] in ("L", "R") else pair.side
        return measure_pair(lab_mlo, prepared[1], lab_cc, prepared[2], chest, fallback)

    def autocast(self):
        """The training config's autocast (``optim.amp_dtype``) on the GPU; float32 on the CPU unless
        ``cpu_autocast`` (then the config's dtype there too, as the trainers' ``predict`` does)."""
        import torch

        dt = self.trainer.config.optim.amp_dtype
        if dt != "none" and (self.device.gpu or self.cpu_autocast):
            return torch.autocast("cuda" if self.device.gpu else "cpu", dtype=getattr(torch, dt))
        return torch.autocast("cpu", enabled=False)

    def precision(self) -> str:
        dt = self.trainer.config.optim.amp_dtype
        return f"{dt} autocast" if dt != "none" and (self.device.gpu or self.cpu_autocast) else "float32"


# ── SegFormer-B3 and U-Net (breast_seg.training.semantic) ─────────────────────────────────────────────────────

class SegFormerPipeline(CanvasPipeline):
    """SegFormer: grey canvas on three ImageNet-normalised channels, logits upsampled to the canvas; flip TTA
    averages the softmax of the image and of its horizontally flipped pass."""

    family = "segformer"
    flip_option = True

    def load(self):
        import torch

        self.model = self.trainer.load(self.weights).to(self.device.torch).eval()
        self.to_input = self.trainer.to_input()
        z = torch.zeros(1, 3, TARGET_SIZE, TARGET_SIZE, device=self.device.torch)
        weights = self.weights / "model.safetensors" if (self.weights / "model.safetensors").is_file() else self.weights
        return {"params_M": n_params(self.model), "weights_file_MB": file_mb(weights),
                "gflops_per_forward": count_gflops(lambda: self.model(pixel_values=z), self.flops),
                "models_per_pair": 1, "input": "640 x 640, 3 channels", "precision": self.precision()}

    def model_input(self, im, ic):
        from breast_seg.training.torch_data import stack_inputs

        return stack_inputs([im, ic], self.to_input)

    def _forward(self, x):
        return self.trainer.forward(self.model, x).float()

    def run_model(self, x, setting):
        import torch

        with torch.no_grad():
            xt = x.to(self.device.torch)
            with self.autocast():
                if setting.batched:
                    lg = self._forward(torch.cat([xt, xt.flip(-1)]) if setting.tta else xt)
                    p = (lg[:2].softmax(1) + lg[2:].flip(-1).softmax(1)) / 2 if setting.tta else lg
                else:
                    ps = []
                    for i in range(2):
                        lg = self._forward(xt[i:i + 1])
                        if setting.tta:
                            lg = (lg.softmax(1) + self._forward(xt[i:i + 1].flip(-1)).flip(-1).softmax(1)) / 2
                        ps.append(lg)
                    p = torch.cat(ps)
                lab = p.argmax(1).to(torch.uint8).cpu().numpy()
        tta = setting.tta

        def nipple_probability(i: int) -> Optional[np.ndarray]:
            q = p[i, 3] if tta else p[i].softmax(0)[3]
            return probability_or_none(q.float().cpu().numpy().astype(np.float16))

        return lab[0], lab[1], nipple_probability


class UNetPipeline(CanvasPipeline):
    """Plain U-Net on the single-channel ``[0, 1]`` canvas; flip TTA averages the two softmax maps."""

    family = "unet"
    flip_option = True

    def load(self):
        import torch

        self.model = self.trainer.load(self.weights).to(self.device.torch).eval()
        z = torch.zeros(1, 1, TARGET_SIZE, TARGET_SIZE, device=self.device.torch)
        return {"params_M": n_params(self.model), "weights_file_MB": file_mb(self.weights),
                "gflops_per_forward": count_gflops(lambda: self.model(z), self.flops),
                "models_per_pair": 1, "input": "640 x 640, 1 channel", "precision": self.precision()}

    def model_input(self, im, ic):
        from breast_seg.training.torch_data import grey01, stack_inputs

        return stack_inputs([im, ic], grey01)

    def _probabilities(self, x):
        return self.trainer.forward(self.model, x).float().softmax(1)

    def run_model(self, x, setting):
        import torch

        with torch.no_grad():
            xt = x.to(self.device.torch)
            with self.autocast():
                if setting.batched:
                    o = self._probabilities(torch.cat([xt, xt.flip(-1)]) if setting.tta else xt)
                    p = (o[:2] + o[2:].flip(-1)) / 2 if setting.tta else o
                else:
                    ps = []
                    for i in range(2):
                        o = self._probabilities(xt[i:i + 1])
                        if setting.tta:
                            o = (o + self._probabilities(xt[i:i + 1].flip(-1)).flip(-1)) / 2
                        ps.append(o)
                    p = torch.cat(ps)
                lab = p.argmax(1).to(torch.uint8).cpu().numpy()
        return lab[0], lab[1], lambda i: probability_or_none(p[i, 3].float().cpu().numpy().astype(np.float16))


# ── nnU-Net ResEnc-M (breast_seg.training.nnunet) ─────────────────────────────────────────────────────────────

class NnUNetPipeline(CanvasPipeline):
    """nnU-Net as ``nnUNetv2_predict`` runs it on one 640 canvas: its own preprocessing (crop to non-zero, z-score),
    a single sliding window, mirroring as ``nnUNetPredictor`` does it (the average over the image and every
    non-empty combination of the mirroring axes stored in the checkpoint; the shipped checkpoint mirrors the
    horizontal axis only), its autocast (float16) on the GPU, its half-precision Gaussian accumulation and its export
    function. ``weights`` is the trained model folder (``<trainer>__<plans>__<configuration>``)."""

    family = "nnunet"
    flip_option = True

    def load(self):
        import torch

        for key, value in self.trainer.env().items():
            os.environ.setdefault(key, value)
        from acvl_utils.cropping_and_padding.padding import pad_nd_image
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian

        c = self.trainer.config
        pr = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                             perform_everything_on_device=self.device.gpu, device=self.device.torch, verbose=False,
                             verbose_preprocessing=False, allow_tqdm=False)
        try:
            pr.initialize_from_trained_model_folder(str(self.weights), use_folds=(c.fold,), checkpoint_name=c.checkpoint)
        except Exception as exc:  # noqa: BLE001 - add the one hint nnU-Net does not give
            if c.trainer in str(exc):
                raise RuntimeError(f"{exc} (install it with breast_seg.training.nnunet.install_trainer_variants())") from exc
            raise
        if len(pr.list_of_parameters) != 1:
            raise RuntimeError("expected the weights of exactly one fold")
        net = pr.network
        getattr(net, "_orig_mod", net).load_state_dict(pr.list_of_parameters[0])  # the fold's weights, explicitly
        self.net = net.to(self.device.torch).eval()
        self.pm, self.cm, self.dj, self.lm = pr.plans_manager, pr.configuration_manager, pr.dataset_json, pr.label_manager
        self.pre = self.cm.preprocessor_class(verbose=False)
        if tuple(self.cm.patch_size) != (TARGET_SIZE, TARGET_SIZE):
            raise RuntimeError(f"patch size {tuple(self.cm.patch_size)}: the benchmark expects one 640 x 640 window")
        self.mirroring_axes = tuple(pr.allowed_mirroring_axes or ())
        self.combos = mirror_combinations(self.mirroring_axes)
        self.gauss = compute_gaussian(tuple(self.cm.patch_size), sigma_scale=1. / 8, value_scaling_factor=10,
                                      device=self.device.torch)
        self._pad, self._export = pad_nd_image, convert_predicted_logits_to_segmentation_with_correct_shape
        z = torch.zeros(1, 1, TARGET_SIZE, TARGET_SIZE, device=self.device.torch)
        return {"params_M": n_params(self.net), "weights_file_MB": file_mb(self.weights / f"fold_{c.fold}" / c.checkpoint),
                "weights_file_note": "nnU-Net checkpoint: network weights plus optimiser state",
                "gflops_per_forward": count_gflops(lambda: self.net(z), self.flops), "models_per_pair": 1,
                "input": "640 x 640 patch, 1 channel, z-score", "mirroring_axes": list(self.mirroring_axes),
                "mirroring_combinations": [list(c) for c in self.combos],
                "precision": "float16 autocast (nnU-Net)" if self.device.gpu else "float32"}

    def setting_info(self, setting):
        n = 1 + len(self.combos) if setting.tta else 1
        return {"tta": setting.tta, "batched": setting.batched, "forwards_per_pair": 2 * n,
                "forward_calls_per_pair": 1 if setting.batched else 2 * n}

    def model_input(self, im, ic):
        import torch

        out = []
        for img in (im, ic):  # as nnU-Net's NaturalImage2DIO reads a grey PNG: (1, 1, H, W) float32, spacing (999, 1, 1)
            d, _, props = self.pre.run_case_npy(img[None, None].astype(np.float32), None, {"spacing": (999, 1, 1)},
                                                self.pm, self.cm, self.dj)
            t, sl = self._pad(torch.from_numpy(d), self.cm.patch_size, "constant", {"value": 0}, True, None)
            if tuple(t.shape[1:]) != (1, TARGET_SIZE, TARGET_SIZE):
                raise RuntimeError(f"nnU-Net input {tuple(t.shape)} is not a single 640 x 640 window")
            out.append((t, sl, props))
        return out

    def run_model(self, g, setting):
        import torch

        net, combos = self.net, (self.combos if setting.tta else [])
        with torch.inference_mode():
            with (torch.autocast("cuda", enabled=True) if self.device.gpu else contextlib.nullcontext()):
                x = torch.stack([g[0][0][:, 0], g[1][0][:, 0]]).to(self.device.torch)  # (2, 1, 640, 640)
                if setting.batched:
                    if combos:  # image and every mirrored copy in one forward
                        o = net(torch.cat([x] + [torch.flip(x, c) for c in combos]))
                        pred = o[:2] + torch.flip(o[2:4], combos[0])
                        for k, c in enumerate(combos[1:], start=2):
                            pred += torch.flip(o[2 * k:2 * k + 2], c)
                        pred /= (len(combos) + 1)
                    else:
                        pred = net(x)
                else:
                    ps = []
                    for i in range(2):
                        q = net(x[i:i + 1])
                        if combos:
                            for c in combos:
                                q += torch.flip(net(torch.flip(x[i:i + 1], c)), c)
                            q /= (len(combos) + 1)
                        ps.append(q)
                    pred = torch.cat(ps)
                # nnUNetPredictor's accumulation of the single window: Gaussian weighting in half precision
                pred *= self.gauss
                lg = torch.zeros(pred.shape, dtype=torch.half, device=self.device.torch)
                lg += pred
                nrm = torch.zeros(pred.shape[-2:], dtype=torch.half, device=self.device.torch)
                nrm += self.gauss
                torch.div(lg, nrm, out=lg)
                if torch.any(torch.isinf(lg)):
                    raise RuntimeError("inf in nnU-Net logits")
            lg = lg.to("cpu")
            parts = [lg[i][:, None][(slice(None), *g[i][1][1:])] for i in range(2)]  # revert the padding, (C, 1, h, w)
            lab = [np.asarray(self._export(parts[i], self.pm, self.cm, self.lm, g[i][2], return_probabilities=False,
                                           num_threads_torch=self.device.threads)[0], np.uint8) for i in range(2)]

        def nipple_probability(i: int) -> Optional[np.ndarray]:
            with torch.inference_mode():
                pr = self._export(parts[i], self.pm, self.cm, self.lm, g[i][2], return_probabilities=True,
                                  num_threads_torch=self.device.threads)[1]
            return probability_or_none(pr.reshape(pr.shape[0], *pr.shape[-2:])[3].astype(np.float16))

        return lab[0], lab[1], nipple_probability


# ── YOLO26m-seg (breast_seg.training.yolo) ────────────────────────────────────────────────────────────────────

class YoloPipeline(CanvasPipeline):
    """YOLO instance segmentation flattened to a label map exactly as ``YoloSegTrainer.predict`` (inference at
    ``min(conf, floor)``, instances >= ``conf`` merged per class, else the best instance >= ``floor``, no pectoral
    rescue on CC); FP32, no TTA. When the map has no nipple, the best nipple instance of a pass at
    :data:`YOLO_NIPPLE_FALLBACK_CONF` is the fallback (inside the timed inference)."""

    family = "yolo_seg"
    flip_option = False

    def load(self):
        import torch

        p = self.trainer.config.predict
        self.query = min(p.conf, p.floor) if p.floor > 0 else p.conf
        self.kwargs = {"imgsz": p.imgsz, "device": self.device.torch, "verbose": False}
        if p.iou is not None:
            self.kwargs["iou"] = p.iou
        import ultralytics  # noqa: F401  (switches OpenCV to one thread on import; restored below)

        self.device.set_threads()
        self.model = self.trainer._model(self.weights)
        unfused = n_params(self.model.model)  # before Ultralytics fuses convolution + batch norm for inference
        self.model.predict([np.zeros((TARGET_SIZE, TARGET_SIZE, 3), np.uint8)] * 2, conf=self.query, **self.kwargs)
        self.device.set_threads()
        try:
            net = self.model.predictor.model.model
        except AttributeError:
            net = self.model.model
        z = torch.zeros(1, 3, TARGET_SIZE, TARGET_SIZE, device=self.device.torch, dtype=next(net.parameters()).dtype)
        return {"params_M": n_params(net), "params_M_unfused": unfused, "weights_file_MB": file_mb(self.weights),
                "params_note": "params_M: fused inference model (as run); params_M_unfused: the checkpoint as loaded",
                "weights_file_note": "Ultralytics checkpoint (half-precision weights)",
                "gflops_per_forward": count_gflops(lambda: net(z), self.flops), "models_per_pair": 1,
                "input": "640 x 640, 3 channels",
                "precision": "float32 (Ultralytics default), fused convolution + batch normalisation"}

    def setting_info(self, setting):
        info = super().setting_info(setting)
        info.update(tta=False, forwards_per_pair=2, forward_calls_per_pair=1 if setting.batched else 2)
        return info

    def model_input(self, im, ic):  # as cv2.imread reads a grey PNG for Ultralytics: three equal BGR channels
        import cv2

        return [cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), cv2.cvtColor(ic, cv2.COLOR_GRAY2BGR)]

    def _flatten(self, res: Any, is_cc: bool) -> np.ndarray:
        from breast_seg.evaluation.predictions import instances_to_label_map

        if res.masks is None:
            return np.zeros(tuple(res.orig_shape), np.uint8)
        p = self.trainer.config.predict
        label, _ = instances_to_label_map(res.masks.data.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                          res.boxes.conf.cpu().numpy(), tuple(res.orig_shape), p.conf, p.floor,
                                          is_cc=is_cc)
        return label

    def _nipple_instance(self, bgr: np.ndarray) -> np.ndarray:
        """Mask of the most confident nipple instance at :data:`YOLO_NIPPLE_FALLBACK_CONF` (empty if none)."""
        import cv2

        from breast_seg.config import VALUE_TO_YOLO

        res = self.model.predict(bgr, conf=YOLO_NIPPLE_FALLBACK_CONF, **self.kwargs)[0]
        h, w = res.orig_shape
        m = np.zeros((h, w), np.uint8)
        if res.masks is not None:
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            k = np.nonzero(cls == VALUE_TO_YOLO[3])[0]
            if len(k):
                d = res.masks.data[k[int(np.argmax(cf[k]))]].cpu().numpy()
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
                m = (d > 0.5).astype(np.uint8)
        return m

    def run_model(self, g, setting):
        if setting.batched:
            res = self.model.predict(g, conf=self.query, **self.kwargs)
        else:
            res = [self.model.predict(x, conf=self.query, **self.kwargs)[0] for x in g]
        if len(res) != 2:
            raise RuntimeError(f"2 images in, {len(res)} results out")
        lab = [self._flatten(res[0], False), self._flatten(res[1], True)]
        fallback: Dict[int, np.ndarray] = {i: self._nipple_instance(g[i]) for i in range(2) if not (lab[i] == 3).any()}
        return lab[0], lab[1], lambda i: fallback[i]


def mirror_combinations(axes: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """Tensor dims to flip, as ``nnUNetPredictor`` mirrors: every non-empty combination of the spatial ``axes``."""
    dims = [a + 2 for a in axes]
    return [c for i in range(len(dims)) for c in itertools.combinations(dims, i + 1)]


FAMILIES = {"segformer": SegFormerPipeline, "unet": UNetPipeline, "nnunet": NnUNetPipeline, "yolo_seg": YoloPipeline}


def describe(info: Dict[str, Any], pipeline: Pipeline, setting: Setting) -> Dict[str, Any]:
    """Model description plus the forward passes of one configuration."""
    out = dict(info)
    out.update(pipeline.setting_info(setting))
    g = out.get("gflops_per_forward")
    if out.get("gflops_per_pair") is None:
        out["gflops_per_pair"] = None if g is None or isinstance(g, list) else g * out["forwards_per_pair"]
    return out
