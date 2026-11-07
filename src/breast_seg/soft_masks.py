"""Per-class continuous probability maps from a YOLO26 segmentation model.

Ultralytics returns binary masks: each instance mask is ``sigmoid(mask_coefficients @ prototypes)`` thresholded at
0.5. :func:`soft_masks` stops one step earlier and keeps the sigmoid, giving a ``[0, 1]`` value at every pixel for
every class.

The channels are independent sigmoids, not a softmax over classes: they do not sum to 1 (pectoral and breast tissue
overlap, background is near zero everywhere). A class without any instance above ``conf`` comes back as an all-zero
channel, which means "no instance emitted", not "confidently absent". Several instances of one class are merged by
their pixel-wise maximum.

The raw head output read here is that of the NMS-free YOLO26 segmentation head (rows ``x1 y1 x2 y2 conf cls
coefficients...``); other heads are not supported.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import numpy as np

SIZE = 640


def soft_masks(model: Any, image: np.ndarray, conf: float = 0.25, size: int = SIZE) -> np.ndarray:
    """``(num_classes, size, size)`` float32 probabilities for a ``size x size`` image (grey or BGR ``uint8``).

    ``model`` is an ``ultralytics.YOLO`` segmentation model.
    """
    import cv2
    import torch
    from ultralytics.utils import ops

    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = x.to(next(model.model.parameters()).device)
    with torch.no_grad():
        (det, protos), _ = model.model(x)
    det, protos = det[0], protos[0]
    det = det[det[:, 4] > conf]
    out = torch.zeros(len(model.names), size, size, device=det.device)
    if not len(det):
        return out.cpu().numpy()
    c, mh, mw = protos.shape
    soft = (det[:, 6:] @ protos.view(c, -1)).sigmoid().view(-1, mh, mw)
    soft = torch.nn.functional.interpolate(soft[None], (size, size), mode="bilinear", align_corners=False)[0]
    soft = ops.crop_mask(soft, det[:, :4])
    for i, cls in enumerate(det[:, 5].int().tolist()):
        out[cls] = torch.maximum(out[cls], soft[i])
    return out.cpu().numpy()


def soft_masks_from_file(weights: Union[str, Path], image_path: Union[str, Path], conf: float = 0.25) -> np.ndarray:
    """Load ``weights`` and an image file, return :func:`soft_masks` of it."""
    import cv2
    from ultralytics import YOLO

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"unreadable image {Path(image_path).name}")
    return soft_masks(YOLO(str(weights)), img, conf)
