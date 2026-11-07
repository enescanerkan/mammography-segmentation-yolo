"""Synthetic mammogram phantoms — for tests, demos and smoke runs that must not touch patient data.

A phantom is a smooth half-ellipse of "tissue" against the chest-wall edge with a little texture, a brighter
pectoral wedge on MLO views and a small nipple bump at the anterior apex. It is not anatomy; it only has to look
enough like a mammogram for the crop, the padding and the fake models of :mod:`breast_seg.inference.testing`.

:func:`phantom_dicom` wraps one into an in-memory DICOM file (fresh random UIDs, no patient module).
"""
from __future__ import annotations

import io
from typing import Any, Dict, Optional

import numpy as np


def phantom_mammogram(width: int = 900, height: int = 1200, chest_side: str = "L", view: str = "MLO",
                      seed: int = 0, level: float = 1800.0) -> np.ndarray:
    """A float32 ``(height, width)`` phantom, tissue bright on a dark background (MONOCHROME2 sense).

    Args:
        chest_side: 'L' puts the chest wall on the left edge of the pixel data, 'R' on the right.
        view: 'MLO' adds the pectoral wedge at the top of the chest wall.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x_from_wall = xs if chest_side == "L" else (width - 1) - xs

    centre_y = height * 0.5
    semi_x, semi_y = width * 0.62, height * 0.42
    inside = (x_from_wall / semi_x) ** 2 + ((ys - centre_y) / semi_y) ** 2 <= 1.0
    texture = rng.normal(0.0, 1.0, (height // 8 + 1, width // 8 + 1)).astype(np.float32)
    texture = np.kron(texture, np.ones((8, 8), np.float32))[:height, :width]
    falloff = np.clip(1.0 - x_from_wall / semi_x, 0.0, 1.0)
    image = np.where(inside, level * (0.55 + 0.45 * falloff) + 90.0 * texture, 0.0)

    nipple_x, nipple_r = semi_x, max(6.0, width * 0.018)
    nipple = (x_from_wall - nipple_x) ** 2 + (ys - centre_y) ** 2 <= nipple_r ** 2
    image = np.where(nipple, level * 0.7, image)

    if view == "MLO":
        # The wedge spans the top of the chest wall: (0, 0) -> (0.42 W, 0) -> (0, 0.55 H).
        wedge = x_from_wall / (width * 0.42) + ys / (height * 0.55) <= 1.0
        image = np.where(wedge & inside, level * 1.35 + 60.0 * texture, image)
        image = np.where(wedge & ~inside, level * 1.25, image)

    image += rng.normal(0.0, 4.0, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 4095.0).astype(np.float32)


def phantom_dicom(view: str = "MLO", laterality: str = "L", photometric: str = "MONOCHROME2",
                  flipped: bool = False, spacing: Optional[float] = 0.1, width: int = 900, height: int = 1200,
                  seed: int = 0, attributes: Optional[Dict[str, Any]] = None,
                  include_view: bool = True, include_laterality: bool = True) -> bytes:
    """An in-memory DICOM (bytes) of a phantom.

    Args:
        view, laterality: written to ViewPosition / ImageLaterality unless ``include_*`` is False.
        photometric: 'MONOCHROME1' stores inverted values, as such scanners do.
        flipped: FieldOfViewHorizontalFlip=YES with the pixels mirrored (chest wall on the opposite edge).
        spacing: ImagerPixelSpacing in mm; None leaves both spacing tags out.
        attributes: extra top-level attributes (e.g. ``{"CompressionForce": 90.0}``).
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    chest_side = laterality if not flipped else ("R" if laterality == "L" else "L")
    pixels = phantom_mammogram(width, height, chest_side=chest_side, view=view, seed=seed)
    stored = pixels.max() - pixels if photometric == "MONOCHROME1" else pixels
    data = np.round(stored).astype(np.uint16)

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.2"  # Digital Mammography X-Ray Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "MG"
    ds.Manufacturer = "Phantom"
    ds.ManufacturerModelName = "breast-seg phantom"
    if include_view:
        ds.ViewPosition = view
    if include_laterality:
        ds.ImageLaterality = laterality
    ds.FieldOfViewHorizontalFlip = "YES" if flipped else "NO"
    if spacing:
        ds.ImagerPixelSpacing = [float(spacing), float(spacing)]
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.Rows, ds.Columns = data.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.PixelData = data.tobytes()
    for key, value in (attributes or {}).items():
        setattr(ds, key, value)

    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, ds, write_like_original=False)
    return buffer.getvalue()
