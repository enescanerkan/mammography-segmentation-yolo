"""CRAUNet (the landmark-regression baseline) as released, timed through the user's own copy of its code and weights.

CRAUNet is not part of this package. Give the root of its released repository (``code_dir``, the one with
``rule-based-model/``) and either the folder its cross-validation writes (``weights_dir`` with
``mlo/fold<k>/mlo_model.pth`` and ``cc/fold<k>/cc_model.pth``; default ``<code_dir>/../cv_results``) or the two
weight files. The pipeline is then:

* **pre**: the released ``crop_pad_resize`` preprocessing, as :func:`released_preprocess` reproduces it: VOI-LUT
  pixels, MONOCHROME1 inversion, min-max to ``[0, 1]``, threshold at the mean, morphological opening with a disk of
  radius 3, the bounding box of the largest connected region (``rmax + 1`` / ``cmax + 1`` included, as released),
  square padding away from the chest wall, bilinear resize to 512 x 512;
* **infer**: the view's model (MLO: 6 outputs = pectoral points and nipple, CC: 2 outputs = nipple), float32;
* **post**: CRAUNet's own measurement: PNL = perpendicular distance of the nipple to the line through the two
  pectoral points in the 512 space, D_CC = nipple x to the chest-wall edge, both scaled back with the padded size
  and the pixel spacing.

Its two models cannot be batched (one per view). The padding side and the D_CC edge are the breast's side, as in
the released code (which reads it from the series description). Needs scikit-image, SciPy and torchvision (CRAUNet's
own dependencies).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from breast_seg.benchmark.pipelines import Device, Pair, Pipeline, Setting, count_gflops, file_mb, n_params

SIZE = 512
OUTPUTS = {"MLO": 6, "CC": 2}


def _normalise(x: np.ndarray, ds: Any) -> np.ndarray:
    """MONOCHROME1 inversion and min-max to [0, 1], in place on the float32 pixels as released."""
    if ds.PhotometricInterpretation == "MONOCHROME1":
        x = np.max(x) - x
    x -= np.min(x)
    x /= np.max(x)
    return x


def _crop_pad_resize(img: np.ndarray, rmin: int, cmin: int, rmax: int, cmax: int, chest_side: str,
                     ds: Any) -> Tuple[np.ndarray, Dict[str, float]]:
    import cv2

    crop = img[rmin:rmax + 1, cmin:cmax + 1]
    h, w = crop.shape
    m = max(h, w)
    tp = m - w
    pt = pb = (m - h) // 2
    if chest_side == "L":
        pl, pr = 0, tp
    elif chest_side == "R":
        pl, pr = tp, 0
    else:
        pl = pr = tp // 2
    pad = np.pad(crop, ((pt, pb), (pl, pr)), "constant")
    rs = cv2.resize(pad, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    return rs, {"padded_w": pad.shape[1], "L": max(rmax - rmin, cmax - cmin),
                "spacing": float(str(ds.ImagerPixelSpacing[0]))}


def released_preprocess(x: np.ndarray, ds: Any, chest_side: str) -> Tuple[np.ndarray, Dict[str, float]]:
    """512 x 512 float32 model input and the geometry for millimetres. ``x`` (VOI-LUT pixels) is modified in place,
    as in the released code; ``chest_side`` is the breast's side (``L`` pads on the right)."""
    from scipy import ndimage
    from skimage import measure, morphology

    img = _normalise(x, ds)
    b = img > img.mean()
    b = morphology.opening(b, morphology.disk(3))
    lab, _ = ndimage.label(b)
    rg = max(measure.regionprops(lab), key=lambda r: r.area)
    rmin, cmin, rmax, cmax = rg.bbox
    return _crop_pad_resize(img, rmin, cmin, rmax, cmax, chest_side, ds)


def perpendicular_distance(p1: np.ndarray, p2: np.ndarray, n: np.ndarray) -> float:
    """CRAUNet's distance of ``n`` to the line through ``p1`` and ``p2`` (projection form, in the inputs' dtype)."""
    line, v = p2 - p1, n - p1
    length = np.linalg.norm(line)
    return float(np.linalg.norm(v - (v @ line) / length ** 2 * line)) if length else float(np.linalg.norm(v))


def model_source(code_dir: Path, view: str) -> Path:
    return Path(code_dir) / "rule-based-model" / f"{view.lower()}-landmark-detection" / "code" / "regression" / "main" \
        / "utils" / "models.py"


def weight_file(weights_dir: Path, view: str, fold: int) -> Path:
    return Path(weights_dir) / view.lower() / f"fold{fold}" / f"{view.lower()}_model.pth"


class CraunetPipeline(Pipeline):
    """CRAUNet: fold ``fold``'s MLO model + CC model, 512 x 512, float32, released preprocessing."""

    name = "craunet"
    family = "craunet"
    flip_option = False

    def __init__(self, code_dir: Path, device: Device, weights_dir: Optional[Path] = None, fold: int = 0,
                 mlo_weights: Optional[Path] = None, cc_weights: Optional[Path] = None, flops: bool = True):
        self.code_dir, self.device, self.fold, self.flops = Path(code_dir), device, int(fold), flops
        weights_dir = Path(weights_dir) if weights_dir else self.code_dir.parent / "cv_results"
        self.weights = {"MLO": Path(mlo_weights) if mlo_weights else weight_file(weights_dir, "MLO", self.fold),
                        "CC": Path(cc_weights) if cc_weights else weight_file(weights_dir, "CC", self.fold)}

    def missing(self):
        """Files the pipeline needs that do not exist."""
        need = [model_source(self.code_dir, v) for v in OUTPUTS] + list(self.weights.values())
        return [str(p) for p in need if not p.is_file()]

    def load(self):
        import torch

        self.models: Dict[str, Any] = {}
        for view, n_out in OUTPUTS.items():
            spec = importlib.util.spec_from_file_location(f"_craunet_{view.lower()}_models", model_source(self.code_dir, view))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            m = module.CRAUNet(in_channels=1, out_features=n_out)
            m.load_state_dict(torch.load(str(self.weights[view]), map_location="cpu", weights_only=True))
            self.models[view] = m.to(self.device.torch).eval()
        z = torch.zeros(1, 1, SIZE, SIZE, device=self.device.torch)
        g = [count_gflops(lambda v=v: self.models[v](z), self.flops) for v in OUTPUTS]
        return {"params_M": sum(n_params(m) for m in self.models.values()),
                "params_M_per_model": [n_params(self.models[v]) for v in OUTPUTS],
                "weights_file_MB": file_mb(*self.weights.values()), "gflops_per_forward": g,
                "gflops_per_pair": None if None in g else sum(g), "models_per_pair": 2, "fold": self.fold,
                "input": "512 x 512, 1 channel", "precision": "float32"}

    def setting_info(self, setting: Setting) -> Dict[str, Any]:
        return {"tta": False, "batched": False, "forwards_per_pair": 2, "forward_calls_per_pair": 2}

    def prepare(self, raw, pair: Pair):
        import torch

        (im, dsm), (ic, dsc) = raw
        rm, gm = released_preprocess(im, dsm, pair.side)
        rc, gc = released_preprocess(ic, dsc, pair.side)
        return torch.from_numpy(rm).float()[None, None], torch.from_numpy(rc).float()[None, None], gm, gc

    def infer(self, prepared, setting: Setting):
        import torch

        xm, xc = prepared[0], prepared[1]
        with torch.no_grad():
            om = self.models["MLO"](xm.to(self.device.torch)).cpu().numpy().reshape(-1, 2) * float(SIZE)
            oc = self.models["CC"](xc.to(self.device.torch)).cpu().numpy().reshape(-1, 2) * float(SIZE)
        return om, oc

    def measure(self, output, prepared, pair: Pair):
        om, oc = output
        gm, gc = prepared[2], prepared[3]
        pnl = perpendicular_distance(om[0], om[1], om[2]) * gm["spacing"] * gm["padded_w"] / float(SIZE)
        nx = oc[0][0]
        # kept in the network's float32, as CRAUNet's evaluation computes it (delta inherits that precision)
        dcc = (nx if pair.side == "L" else SIZE - nx) * gc["L"] / float(SIZE) * gc["spacing"]
        return pnl, dcc, ("", "")
