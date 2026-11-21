"""One way to round a number for display.

A measurement is kept unrounded in the study record and rounded once, where it is written out: by ``formatNumber``
in the interface (i18n.js) and by :func:`fixed` here (the report, the English notes and explanations, the values of
the criteria). Both round the exact value half away from zero, as JavaScript's ``Number.toFixed`` does, so the same
measurement is never written two ways on one screen. Rounding a value that was already rounded is what made a
47.949 mm distance read 47.9 mm under its image and 48.0 mm on its card.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal


def fixed(value: float, digits: int = 1) -> str:
    """``value`` written with ``digits`` decimals: its exact value rounded half away from zero (``toFixed``)."""
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return format(Decimal(number).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), "f")


def rounded(value: float, digits: int = 1) -> float:
    """``value`` rounded for display by the rule of :func:`fixed`, as a number (a criterion's stored value)."""
    return float(fixed(value, digits))
