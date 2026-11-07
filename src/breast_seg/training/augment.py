"""Albumentations pipelines from YAML specs, and the patch that injects them into Ultralytics.

A spec is ``{name: <class in albumentations>, <kwargs>}``. Lists become tuples (``clip_limit: [1, 4]``), strings
``"cv2.<NAME>"`` become OpenCV constants, and a ``transforms`` list nests (``OneOf``, ``Compose``...).
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Mapping, Sequence


def _value(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_value(x) for x in v)
    if isinstance(v, str) and v.startswith("cv2."):
        import cv2

        return getattr(cv2, v[4:])
    return v


def build_transform(spec: Mapping[str, Any], module: Any = None) -> Any:
    """One Albumentations transform from its spec."""
    if module is None:
        import albumentations as module  # type: ignore[no-redef]
    spec = dict(spec)
    try:
        name = spec.pop("name")
    except KeyError:
        raise ValueError(f"transform spec without a name: {spec}") from None
    cls = getattr(module, name, None)
    if cls is None:
        raise ValueError(f"albumentations has no transform {name!r}")
    kwargs: Dict[str, Any] = {}
    for k, v in spec.items():
        kwargs[k] = build_transforms(v, module) if k == "transforms" else _value(v)
    return cls(**kwargs)


def build_transforms(specs: Sequence[Mapping[str, Any]], module: Any = None) -> List[Any]:
    """A list of transforms (not yet composed)."""
    return [build_transform(s, module) for s in specs]


def validate_specs(specs: Sequence[Mapping[str, Any]]) -> None:
    """Structural check usable without Albumentations installed (names present, nested lists well formed)."""
    for i, s in enumerate(specs):
        if not isinstance(s, Mapping) or not isinstance(s.get("name"), str):
            raise ValueError(f"transform {i}: expected a mapping with a 'name'")
        if "transforms" in s:
            validate_specs(s["transforms"])


@contextlib.contextmanager
def ultralytics_albumentations(specs: Sequence[Mapping[str, Any]]) -> Iterator[None]:
    """Replace the default transform list of ``ultralytics.data.augment.Albumentations`` while the block runs.

    Ultralytics builds its ``Albumentations`` step with ``transforms=None`` and then uses its own list; with this
    patch the configured pixel-level list is used instead. Unknown keyword arguments of newer Ultralytics versions
    are passed through. The original constructor is restored on exit. Training must run in this process (single
    device): a DDP launch starts fresh interpreters that do not see the patch.
    """
    import ultralytics.data.augment as aug

    original = aug.Albumentations.__init__
    specs = [dict(s) for s in specs]

    def patched(self, p: float = 1.0, transforms=None, **kwargs):
        if transforms is None:
            transforms = build_transforms(specs)
        original(self, p=p, transforms=transforms, **kwargs)

    aug.Albumentations.__init__ = patched
    try:
        yield
    finally:
        aug.Albumentations.__init__ = original
