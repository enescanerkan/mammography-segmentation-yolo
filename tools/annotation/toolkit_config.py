"""Configuration for the annotation toolkit (weights, MedSAM2 code, output folders).

Every model file the toolkit needs is resolved here, nowhere else. Values come from,
in increasing priority:

1. built-in defaults (see ``DEFAULTS``), relative to the toolkit folder;
2. a YAML file: ``--config PATH`` on the command line, else ``$ANNOTATION_CONFIG``,
   else ``tools/annotation/annotation.yaml`` if it exists (copy
   ``annotation.example.yaml`` to start);
3. one environment variable per key (see ``ENV_VARS``).

Path rules
----------
* ``weights_root``, ``medsam2.repo``, ``positioning.results_dir`` and ``logs_dir``:
  a relative value is resolved against the folder of the YAML file (or the toolkit
  folder when no YAML is used).
* Weight files (``medsam2.sam2_base``, ``medsam2.checkpoint``, ``detector.weights``,
  ``positioning.*_weights``): a relative value is resolved against ``weights_root``.
* Paths given through environment variables that are relative resolve against the
  current working directory.

Nothing is auto-discovered: the fine-tuned MedSAM2 checkpoint is exactly the file named
by ``medsam2.checkpoint`` (or ``$MEDSAM2_CHECKPOINT``).
"""

from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

TOOLKIT_ROOT = Path(__file__).resolve().parent
LOCAL_CONFIG_NAME = "annotation.yaml"

DEFAULTS: Dict[str, Any] = {
    "weights_root": "weights",
    "logs_dir": "logs",
    "medsam2": {
        "repo": "vendor/MedSAM",
        "model_cfg": "sam2_hiera_t.yaml",
        "sam2_base": "medsam2/sam2_hiera_tiny.pt",
        "checkpoint": "medsam2/medsam_model_best.pth",
        "device": "auto",
    },
    "detector": {
        "weights": "detector/manual_detector.pt",
    },
    "positioning": {
        "mlo_weights": "positioning/mlo-yolo26-pose-advanced.pt",
        "cc_weights": "positioning/cc-yolo26-pose-advanced.pt",
        "results_dir": "results/positioning",
    },
}

# dotted key -> environment variable(s); the first one that is set wins.
ENV_VARS: Dict[str, tuple] = {
    "weights_root": ("ANNOTATION_WEIGHTS_ROOT",),
    "logs_dir": ("ANNOTATION_LOGS_DIR",),
    "medsam2.repo": ("MEDSAM2_REPO",),
    "medsam2.model_cfg": ("SAM2_MODEL_CFG",),
    "medsam2.sam2_base": ("MEDSAM2_SAM2_BASE",),
    # MEDSAM2_INTERACTIVE_CKPT is the name older instructions used; still accepted.
    "medsam2.checkpoint": ("MEDSAM2_CHECKPOINT", "MEDSAM2_INTERACTIVE_CKPT"),
    "medsam2.device": ("MEDSAM2_DEVICE",),
    "detector.weights": ("ANNOTATION_DETECTOR_WEIGHTS",),
    "positioning.mlo_weights": ("POSITIONING_MLO_WEIGHTS",),
    "positioning.cc_weights": ("POSITIONING_CC_WEIGHTS",),
    "positioning.results_dir": ("POSITIONING_RESULTS_DIR",),
}
CONFIG_ENV_VAR = "ANNOTATION_CONFIG"


@dataclass(frozen=True)
class ToolkitConfig:
    """Resolved, absolute settings. Build with :func:`load_config`."""

    weights_root: Path
    logs_dir: Path
    medsam2_repo: Path
    sam2_model_cfg: str
    sam2_base_weights: Path
    medsam2_checkpoint: Path
    device: str
    detector_weights: Path
    positioning_mlo_weights: Path
    positioning_cc_weights: Path
    positioning_results_dir: Path
    source: Optional[Path] = None  # YAML file that was read, if any

    def missing_medsam2_files(self) -> list:
        """Human-readable problems that stop SAM mode from starting (empty = OK)."""
        problems = []
        if not (self.medsam2_repo / "sam2").is_dir():
            problems.append(
                f"MedSAM2 code (MedSAM2 branch of bowang-lab/MedSAM) not found at {self.medsam2_repo} "
                "[medsam2.repo / $MEDSAM2_REPO]"
            )
        if not self.sam2_base_weights.is_file():
            problems.append(f"SAM2 base weights not found: {self.sam2_base_weights} [medsam2.sam2_base / $MEDSAM2_SAM2_BASE]")
        if not self.medsam2_checkpoint.is_file():
            problems.append(
                f"Fine-tuned MedSAM2 checkpoint not found: {self.medsam2_checkpoint} "
                "[medsam2.checkpoint / $MEDSAM2_CHECKPOINT]"
            )
        return problems

    def missing_positioning_files(self) -> list:
        return [
            str(p)
            for p in (self.positioning_mlo_weights, self.positioning_cc_weights)
            if not p.is_file()
        ]

    def describe(self) -> str:
        """Multi-line summary for logs and the help dialog."""
        rows = [
            ("config file", self.source or "(defaults only)"),
            ("weights_root", self.weights_root),
            ("medsam2.repo", self.medsam2_repo),
            ("medsam2.model_cfg", self.sam2_model_cfg),
            ("medsam2.sam2_base", self.sam2_base_weights),
            ("medsam2.checkpoint", self.medsam2_checkpoint),
            ("medsam2.device", self.device),
            ("detector.weights", self.detector_weights),
            ("positioning.mlo_weights", self.positioning_mlo_weights),
            ("positioning.cc_weights", self.positioning_cc_weights),
            ("positioning.results_dir", self.positioning_results_dir),
            ("logs_dir", self.logs_dir),
        ]
        return "\n".join(f"{k:24s} {v}" for k, v in rows)


# ── loading ──────────────────────────────────────────────────────────────────

_explicit_path: Optional[Path] = None
_cached: Optional[ToolkitConfig] = None


def set_config_path(path: Optional[os.PathLike]) -> None:
    """Use ``path`` as the YAML file for subsequent :func:`get_config` calls."""
    global _explicit_path, _cached
    _explicit_path = Path(path).expanduser().resolve() if path else None
    _cached = None


def get_config() -> ToolkitConfig:
    """Process-wide configuration (loaded once, then cached)."""
    global _cached
    if _cached is None:
        _cached = load_config(_explicit_path)
    return _cached


def reset_config_cache() -> None:
    global _cached
    _cached = None


def find_config_file(explicit: Optional[os.PathLike] = None, environ: Optional[Dict[str, str]] = None) -> Optional[Path]:
    """The YAML file to read: ``explicit``, else ``$ANNOTATION_CONFIG``, else a local ``annotation.yaml``."""
    env_map = os.environ if environ is None else environ
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p.resolve()
    env = env_map.get(CONFIG_ENV_VAR, "").strip()
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"${CONFIG_ENV_VAR} points to a missing file: {p}")
        return p.resolve()
    local = TOOLKIT_ROOT / LOCAL_CONFIG_NAME
    return local if local.is_file() else None


def load_config(path: Optional[os.PathLike] = None, environ: Optional[Dict[str, str]] = None) -> ToolkitConfig:
    """Resolve defaults + YAML + environment into a :class:`ToolkitConfig`."""
    env = os.environ if environ is None else environ
    source = find_config_file(path, env)
    raw = copy.deepcopy(DEFAULTS)
    base = TOOLKIT_ROOT
    if source is not None:
        _merge(raw, _read_yaml(source), source)
        base = source.parent

    def get(key: str) -> Any:
        node: Any = raw
        for part in key.split("."):
            node = node[part]
        return node

    from_env: Dict[str, str] = {}
    for key, names in ENV_VARS.items():
        for name in names:
            value = env.get(name, "").strip()
            if value:
                from_env[key] = value
                break

    def path_of(key: str, relative_to: Path) -> Path:
        if key in from_env:
            return Path(from_env[key]).expanduser().resolve()
        p = Path(str(get(key))).expanduser()
        return p if p.is_absolute() else (relative_to / p).resolve()

    weights_root = path_of("weights_root", base)

    def text_of(key: str) -> str:
        return from_env.get(key) or str(get(key))

    return ToolkitConfig(
        weights_root=weights_root,
        logs_dir=path_of("logs_dir", base),
        medsam2_repo=path_of("medsam2.repo", base),
        sam2_model_cfg=text_of("medsam2.model_cfg"),
        sam2_base_weights=path_of("medsam2.sam2_base", weights_root),
        medsam2_checkpoint=path_of("medsam2.checkpoint", weights_root),
        device=text_of("medsam2.device"),
        detector_weights=path_of("detector.weights", weights_root),
        positioning_mlo_weights=path_of("positioning.mlo_weights", weights_root),
        positioning_cc_weights=path_of("positioning.cc_weights", weights_root),
        positioning_results_dir=path_of("positioning.results_dir", base),
        source=source,
    )


def resolve_device(name: str) -> str:
    """Map ``auto`` to ``cuda`` when available, else ``cpu``; pass other values through."""
    if name.strip().lower() != "auto":
        return name.strip()
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def _merge(dst: Dict[str, Any], src: Dict[str, Any], source: Path, prefix: str = "") -> None:
    for key, value in src.items():
        dotted = f"{prefix}{key}"
        if key not in dst:
            log.warning("%s: unknown config key '%s' ignored", source, dotted)
            continue
        if isinstance(dst[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{source}: '{dotted}' must be a mapping")
            _merge(dst[key], value, source, prefix=f"{dotted}.")
        elif value is not None:
            dst[key] = value


def ensure_breast_seg_importable() -> None:
    """Make ``import breast_seg`` work from a source checkout without ``pip install -e .``.

    Adds ``<repo>/src`` to ``sys.path`` only when the package is not importable already.
    """
    try:
        import breast_seg  # noqa: F401
        return
    except ImportError:
        pass
    import sys

    src = TOOLKIT_ROOT.parents[1] / "src"
    if (src / "breast_seg").is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
