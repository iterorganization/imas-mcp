"""Calibration dataset loader for quality review.

Single loading path for both mint review and benchmark scoring.
"""

from __future__ import annotations

import importlib.resources
import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CACHE: list[dict[str, Any]] | None = None


def load_calibration() -> list[dict[str, Any]]:
    """Load benchmark calibration entries (cached).

    Returns a list of dicts from benchmark_calibration.yaml,
    each with: name, tier, expected_score, description, etc.
    """
    global _CACHE  # noqa: PLW0603
    if _CACHE is not None:
        return _CACHE

    try:
        ref = importlib.resources.files("imas_codex.sn") / "benchmark_calibration.yaml"
        _CACHE = yaml.safe_load(ref.read_text()).get("entries", [])
    except Exception:
        logger.debug("Failed to load calibration entries", exc_info=True)
        _CACHE = []

    return _CACHE


def get_calibration_for_prompt() -> list[dict[str, Any]]:
    """Return calibration entries formatted for prompt rendering.

    Includes normalized 0-1 score alongside per-dimension integers.
    """
    return [
        {
            "name": e["name"],
            "tier": e["tier"],
            "score": round(e["expected_score"] / 120.0, 2),
            "reason": e["reason"],
        }
        for e in load_calibration()
    ]


def clear_calibration_cache() -> None:
    """Clear cached calibration data (for testing)."""
    global _CACHE  # noqa: PLW0603
    _CACHE = None
