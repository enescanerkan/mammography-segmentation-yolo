"""Mammography Positioning Review — a FastAPI service with a build-free single-page client.

Upload a study (DICOM files or a study folder); pose landmarks and segmentation masks each yield the posterior
nipple line (MLO) and the nipple-to-chest-wall distance (CC); delta-D is decided per breast by both routes, the
criteria a radiographer checks are graded, every label is editable and a report is written in English or Turkish.
Uploaded DICOMs are never written to disk.

Usage::

    from breast_seg.apps.positioning import load_app_config, serve
    serve(load_app_config("configs/app.yaml"))

FastAPI and uvicorn come with the ``app`` extra; the models with the ``models`` extra.
"""
from .config import AppConfig, ConfigError, app_config_from_dict, load_app_config

__all__ = ["AppConfig", "ConfigError", "app_config_from_dict", "create_app", "load_app_config", "serve"]


def create_app(*args, **kwargs):
    """See :func:`breast_seg.apps.positioning.api.create_app` (imported lazily: needs FastAPI)."""
    from .api import create_app as _create_app

    return _create_app(*args, **kwargs)


def serve(*args, **kwargs):
    """See :func:`breast_seg.apps.positioning.api.serve` (imported lazily: needs FastAPI and uvicorn)."""
    from .api import serve as _serve

    return _serve(*args, **kwargs)
