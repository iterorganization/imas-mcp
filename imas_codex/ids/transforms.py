"""Transform execution engine for IDS field mappings.

Provides safe evaluation of transform_code expressions stored on
IMASMapping nodes, with access to math, numpy, and imas_codex
unit/COCOS utilities.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Lazy numpy import — not always available
_numpy = None


def _get_numpy():
    global _numpy
    if _numpy is None:
        try:
            import numpy

            _numpy = numpy
        except ImportError:
            _numpy = False
    return _numpy if _numpy is not False else None


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units using pint."""
    from imas_codex.units import unit_registry

    q = unit_registry.Quantity(value, from_unit)
    return q.to(to_unit).magnitude


def execute_transform(value: Any, transform_code: str | None) -> Any:
    """Execute a mapping's transform_code expression.

    The transform_code is a Python expression evaluated with 'value' as
    the input variable. The expression has access to math, numpy, and
    imas_codex utility functions.

    This follows the same trust model as DataAccess templates — the code
    is authored by agents/humans through the mapping lifecycle, not from
    untrusted external input.

    Args:
        value: Input value to transform.
        transform_code: Python expression string. If None or "value",
            returns the input unchanged (identity transform).

    Returns:
        Transformed value.
    """
    if not transform_code or transform_code == "value":
        return value

    np = _get_numpy()
    context: dict[str, Any] = {
        "value": value,
        "math": math,
        "abs": abs,
        "float": float,
        "int": int,
        "str": str,
        "len": len,
        "convert_units": convert_units,
    }
    if np is not None:
        context["np"] = np

    return eval(transform_code, {"__builtins__": {}}, context)  # noqa: S307


def set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value on an imas-python object using a dotted path.

    Handles nested structures like 'geometry.rectangle.r' by traversing
    getattr chain and setting the final attribute.
    """
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)
