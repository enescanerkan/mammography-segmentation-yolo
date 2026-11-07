"""DICOM bytes in, pixels plus a non-identifying header summary out. Nothing is written to disk.

The bytes live in memory for the length of one call. What the caller gets back is the pixel array and a small dict
of acquisition facts (view, side, spacing, vendor, physics) — no name, ID, accession, UID or date is read.

View and laterality are the two header fields the pipeline cannot do without, and plenty of exports lose them: an
anonymiser strips ViewPosition, a research extract keeps only ViewCodeSequence, a PACS dump carries the view in
the series description. Each is therefore resolved through a chain of sources ending at the file name, and the
answer records which source it came from — a value guessed from a file name is a different kind of fact from one
read out of (0018,5101).
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

from .preprocess import chest_wall_side

# SNOMED codes DICOM uses for mammography projections, in both the older SRT scheme and the current SCT one.
VIEW_CODES = {
    "R-10242": "CC", "399162004": "CC",
    "R-10226": "MLO", "399368009": "MLO",
    "R-10228": "ML", "399260004": "ML",
    "R-10214": "MLO",
}

# Assumed when the header carries neither ImagerPixelSpacing nor PixelSpacing; every millimetre is then an estimate.
DEFAULT_PIXEL_SPACING_MM = 0.085

TEXT_FIELDS = ("SeriesDescription", "ProtocolName", "StudyDescription",
               "ImageComments", "PerformedProcedureStepDescription", "AcquisitionDeviceProcessingDescription")

# Field names that count as header facts rather than guesses (the UI flags the rest).
HEADER_SOURCES = frozenset({"ViewPosition", "ViewCodeSequence", "ImageLaterality", "Laterality", "FrameLaterality"})

_NOT_IMAGES = ("txt", "json", "csv", "xml", "yaml", "yml", "md", "pdf", "log",
               "png", "jpg", "jpeg", "tif", "tiff", "zip", "gz")

# "L-MLO", "l_mlo", "lcc" — but not the "cc" inside "accession".
_VIEW_IN_TEXT = re.compile(r"(?<![a-z])(mlo|cc|ml|lm|xccl)(?![a-z])", re.I)
# The glued spelling radiographers actually type: LMLO, RCC, RMLO.
_VIEW_GLUED = re.compile(r"(?<![a-z])[lr][\W_]*(mlo|cc)(?![a-z])", re.I)

# Side next to a view token, in either order. Written as explicit alternatives rather than a lookbehind: Python
# only allows fixed-width ones, and "left" and "l" are not the same width.
_SIDE_PATTERNS = (
    (re.compile(r"(?<![a-z])(l|left)[\W_]*(?:mlo|cc|ml)(?![a-z])", re.I), "L"),
    (re.compile(r"(?<![a-z])(r|right)[\W_]*(?:mlo|cc|ml)(?![a-z])", re.I), "R"),
    (re.compile(r"(?<![a-z])(?:mlo|cc|ml)[\W_]*(l|left)(?![a-z])", re.I), "L"),
    (re.compile(r"(?<![a-z])(?:mlo|cc|ml)[\W_]*(r|right)(?![a-z])", re.I), "R"),
    (re.compile(r"(?<![a-z])left(?![a-z])", re.I), "L"),
    (re.compile(r"(?<![a-z])right(?![a-z])", re.I), "R"),
)


class DicomError(ValueError):
    """Raised with a message meant to be shown to the user as-is."""


def _apply_voi_lut(array: np.ndarray, dataset: Any) -> np.ndarray:
    try:  # pydicom >= 3
        from pydicom.pixels import apply_voi_lut
    except ImportError:  # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    return apply_voi_lut(array, dataset)


def read_dicom(data: bytes, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Decode one single-frame mammogram.

    Args:
        data: the DICOM file bytes.
        filename: the upload name or relative path (``case/RMLO/IM0001.dcm``); used for messages and as the last
            source of view / side.

    Returns:
        ``(pixels, meta)``: float32 pixels after the VOI LUT, in the file's photometric interpretation, and the
        header summary consumed by :func:`breast_seg.imaging.preprocess` and the engine.

    Raises:
        DicomError: with a user-facing message (not DICOM, undecodable, multi-frame, unsupported or unknown view).
    """
    import pydicom

    # Say plainly what a text file is, instead of quoting the parser at someone who only dropped a folder and let
    # it pick up a labels/ sidecar.
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename.rsplit("/", 1)[-1] else ""
    if suffix in _NOT_IMAGES:
        raise DicomError(f"{filename}: skipped — a .{suffix} file, not an image")

    try:
        dataset = pydicom.dcmread(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - any parser failure is a user-facing "not DICOM"
        raise DicomError(f"{filename}: not readable as DICOM ({exc})") from None

    try:
        pixels = _apply_voi_lut(dataset.pixel_array, dataset).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        raise DicomError(
            f"{filename}: pixel data could not be decoded — a compression codec may be missing ({exc})") from None

    if pixels.ndim != 2:
        raise DicomError(f"{filename}: expected a single-frame grayscale image")

    view, view_source = resolve_view(dataset, filename)
    laterality, laterality_source = resolve_laterality(dataset, filename)

    if view is None:
        raise DicomError(
            f"{filename}: could not tell CC from MLO. The header has no ViewPosition, "
            "ViewCodeSequence or series description naming the view — rename the file "
            "so it contains the view, e.g. L-MLO.dcm or R_CC.dcm.")
    if laterality is None:
        raise DicomError(
            f"{filename}: could not tell left from right. The header has no ImageLaterality or "
            "Laterality — rename the file so it contains the side, e.g. L-MLO.dcm.")

    spacing = getattr(dataset, "ImagerPixelSpacing", None) or getattr(dataset, "PixelSpacing", None)
    meta: Dict[str, Any] = {
        "filename": filename,
        "photometric": str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")),
        "laterality": laterality,
        "view": view,
        "view_key": f"{laterality}-{view}",
        "view_source": view_source,
        "laterality_source": laterality_source,
        "pixel_spacing": float(spacing[0]) if spacing else DEFAULT_PIXEL_SPACING_MM,
        "pixel_spacing_known": bool(spacing),
        "horizontal_flip": str(getattr(dataset, "FieldOfViewHorizontalFlip", "NO")).upper() == "YES",
        "chest_side": resolve_chest_side(dataset, pixels, laterality),
        "manufacturer": str(getattr(dataset, "Manufacturer", "") or ""),
        "model_name": str(getattr(dataset, "ManufacturerModelName", "") or ""),
        "original_width": int(pixels.shape[1]),
        "original_height": int(pixels.shape[0]),
        "physics": read_physics(dataset),
    }
    return pixels, meta


def display_pixels(pixels: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
    """``pixels`` with MONOCHROME1 inverted, so tissue is bright whatever the file's photometric."""
    if str(meta.get("photometric", "")).strip().upper() == "MONOCHROME1":
        return pixels.max() - pixels
    return pixels


def read_physics(dataset: Any) -> Dict[str, Any]:
    """Acquisition parameters the header already carries.

    They are what turns a compression force into a pressure, and what a service audits a technologist on.
    Everything is optional — a study whose header omits them simply reports fewer numbers.
    """
    def number(attribute: str) -> Optional[float]:
        value = getattr(dataset, attribute, None)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def text(attribute: str) -> Optional[str]:
        return str(getattr(dataset, attribute, "") or "") or None

    return {
        "compression_n": number("CompressionForce"),
        "thickness_mm": number("BodyPartThickness"),
        "gantry_deg": number("PositionerPrimaryAngle"),
        "kvp": number("KVP"),
        "exposure_uas": number("ExposureInuAs"),
        "entrance_dose_mgy": number("EntranceDoseInmGy"),
        "organ_dose": number("OrganDose"),
        "anode": text("AnodeTargetMaterial"),
        "filter": text("FilterMaterial"),
        "paddle": text("PaddleDescription"),
    }


# ── view ───────────────────────────────────────────────────────────────────
def resolve_view(dataset: Any, filename: str) -> Tuple[Optional[str], str]:
    """('MLO' | 'CC' | None, source). True laterals are rejected, not coerced into MLO."""
    raw = str(getattr(dataset, "ViewPosition", "") or "").strip().upper()
    if raw in ("MLO", "CC"):
        return raw, "ViewPosition"
    if raw in ("ML", "LM", "XCCL", "LMO"):
        # A true lateral is not an MLO; treating it as one would silently measure the wrong projection.
        raise DicomError(f"{filename}: ViewPosition '{raw}' — only MLO and CC are supported")

    coded = _view_from_code(dataset)
    if coded:
        return coded, "ViewCodeSequence"

    for field in TEXT_FIELDS:
        found = view_in_text(str(getattr(dataset, field, "") or ""))
        if found:
            return found, field

    for text, source in name_candidates(filename):
        found = view_in_text(text)
        if found:
            return found, source
    return None, "unknown"


def _view_from_code(dataset: Any) -> Optional[str]:
    sequence = getattr(dataset, "ViewCodeSequence", None)
    if not sequence:
        return None
    try:
        item = sequence[0]
    except (IndexError, TypeError):
        return None
    code = str(getattr(item, "CodeValue", "") or "").strip()
    if code in VIEW_CODES and VIEW_CODES[code] in ("MLO", "CC"):
        return VIEW_CODES[code]
    meaning = str(getattr(item, "CodeMeaning", "") or "").lower()
    if "cranio" in meaning or meaning.strip() in ("cc", "craniocaudal"):
        return "CC"
    if "oblique" in meaning:
        return "MLO"
    return None


def view_in_text(text: str) -> Optional[str]:
    """'MLO' or 'CC' named in free text (``LMLO``, ``r_cc``, ``L-MLO``), else None."""
    if not text:
        return None
    for pattern in (_VIEW_GLUED, _VIEW_IN_TEXT):
        match = pattern.search(text)
        if match:
            token = match.group(1).upper()
            if token in ("MLO", "CC"):
                return token
    return None


# ── laterality ─────────────────────────────────────────────────────────────
def resolve_laterality(dataset: Any, filename: str) -> Tuple[Optional[str], str]:
    """('L' | 'R' | None, source)."""
    for attribute in ("ImageLaterality", "Laterality", "FrameLaterality"):
        value = str(getattr(dataset, attribute, "") or "").strip().upper()
        if value in ("L", "R"):
            return value, attribute

    for field in TEXT_FIELDS:
        found = side_in_text(str(getattr(dataset, field, "") or ""))
        if found:
            return found, field

    for text, source in name_candidates(filename):
        found = side_in_text(text)
        if found:
            return found, source
    return None, "unknown"


def name_candidates(filename: str) -> Iterator[Tuple[str, str]]:
    """Places in an upload path where the view might be written, closest first.

    A folder upload arrives as ``case-042/RMLO/IM0001.dcm``. The folder is the part the operator actually named,
    and the file inside it is usually a scanner serial number, so the immediate parent is tried before the file
    itself — and the whole path last, in case the view is further up.
    """
    parts = [part for part in filename.replace("\\", "/").split("/") if part]
    if len(parts) > 1:
        yield parts[-2], "folder name"
    if parts:
        yield parts[-1], "file name"
    if len(parts) > 2:
        yield "/".join(parts[:-1]), "upload path"


def side_in_text(text: str) -> Optional[str]:
    """'L' or 'R' named next to a view token (or as a whole word) in free text, else None."""
    if not text:
        return None
    for pattern, side in _SIDE_PATTERNS:
        if pattern.search(text):
            return side
    return None


def resolve_chest_side(dataset: Any, pixels: np.ndarray, laterality: str) -> str:
    """Which side of the pixel data the chest wall sits on.

    :func:`chest_wall_side` answers from ImageLaterality and the flip flag when the tag is there; without it, it
    counts where the bright pixels are. That count assumes tissue brighter than air, so a MONOCHROME1 image is
    inverted first — otherwise the air wins, and the padding direction and the CC chest-wall reference both land
    on the wrong side. The resolved laterality (flipped when the pixels are mirrored) is the last resort.
    """
    if str(getattr(dataset, "PhotometricInterpretation", "")).strip().upper() == "MONOCHROME1":
        pixels = pixels.max() - pixels
    side = chest_wall_side(dataset, pixels)
    if side in ("L", "R"):
        return side
    flipped = str(getattr(dataset, "FieldOfViewHorizontalFlip", "NO")).upper() == "YES"
    if not flipped:
        return laterality
    return "R" if laterality == "L" else "L"
