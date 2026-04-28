from pathlib import Path
from typing import Optional, Any, Tuple
import cv2
import numpy as np

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except ImportError:
    pydicom = None
    apply_voi_lut = None

class DicomUtils:
    @staticmethod
    def spacing_mm_from_ds(ds) -> float:
        if ds is None: return 0.085
        try:
            if getattr(ds, "ImagerPixelSpacing", None) is not None:
                s = ds.ImagerPixelSpacing
                if isinstance(s, (list, tuple)) and len(s) > 0: return float(s[0])
        except Exception: pass
        try:
            if getattr(ds, "PixelSpacing", None) is not None:
                s = ds.PixelSpacing
                if isinstance(s, (list, tuple)) and len(s) > 0: return float(s[0])
        except Exception: pass
        return 0.085

    @staticmethod
    def load_dicom_bgr(path: Path) -> Tuple[Optional[np.ndarray], float, Any]:
        """Loads DICOM, applies VOI LUT, inverts MONOCHROME1, scales to uint8 BGR."""
        if pydicom is None or not path.is_file(): return None, 0.085, None
        try:
            ds = pydicom.dcmread(str(path), force=True)
            sp = DicomUtils.spacing_mm_from_ds(ds)
            arr = np.asarray(ds.pixel_array)
            arr = np.squeeze(arr)
            if arr.ndim == 3: arr = arr[0] if arr.shape[0] < arr.shape[1] else arr[:, :, 0]
            
            a = apply_voi_lut(arr, ds).astype(np.float32) if apply_voi_lut else np.asarray(arr, dtype=np.float32)
            photo = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2") or "MONOCHROME2")
            if photo == "MONOCHROME1" and a.size:
                a = float(np.max(a)) - a
                
            mn, mx = float(np.min(a)), float(np.max(a))
            g8 = np.clip((a - mn) / (mx - mn + 1e-8) * 255.0, 0.0, 255.0).astype(np.uint8) if mx > mn + 1e-8 else np.zeros(a.shape, dtype=np.uint8)
            bgr = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
            return bgr, sp, ds
        except Exception:
            return None, 0.085, None

