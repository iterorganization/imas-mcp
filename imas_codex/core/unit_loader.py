"""
Unit utilities for IMAS Data Dictionary.

This module provides pint-based unit name and dimensionality lookup
for physical units used in the data dictionary.
"""

import logging

from imas_codex.units import unit_registry

logger = logging.getLogger(__name__)

# Map pint dimensionality to human-readable physics descriptions
DIMENSIONALITY_DESCRIPTIONS = {
    "[length]": "length",
    "[mass]": "mass",
    "[time]": "time",
    "[temperature]": "temperature",
    "[current]": "electric current",
    "[length] ** 2": "area",
    "[length] ** 3": "volume",
    "[length] ** -3": "number density",
    "1 / [length] ** 3": "number density",
    "[length] / [time]": "velocity",
    "[length] / [time] ** 2": "acceleration",
    "[mass] / [length] ** 3": "mass density",
    "[mass] / [length] / [time] ** 2": "pressure",
    "[mass] * [length] ** 2 / [time] ** 2": "energy",
    "[mass] * [length] ** 2 / [time] ** 3": "power",
    "[current] * [time]": "electric charge",
    "[mass] * [length] ** 2 / [current] / [time] ** 3": "voltage",
    "[mass] / [current] / [time] ** 2": "magnetic field",
    "[mass] / [time] ** 2 / [current]": "magnetic field",
    "[mass] * [length] ** 2 / [current] / [time] ** 2": "magnetic flux",
    "[length] ** -1": "spatial frequency",
    "[time] ** -1": "frequency",
    "1 / [time]": "frequency",
}


def get_unit_name(unit_str: str) -> str:
    """
    Get unit name for a unit using pint's compact format.

    Uses the `~P` format specifier for short, paper-style unit symbols
    (e.g., "eV" instead of "electron_volt", "Pa" instead of "pascal").

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Compact unit symbol from pint
    """
    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        # Parse the unit with pint
        unit = unit_registry(unit_str)

        # Use compact format (~P) for paper-style symbols like eV, Pa, T
        unit_name = f"{unit.units:~P}"
        return unit_name

    except Exception as e:
        logger.debug(f"Could not parse unit '{unit_str}' with pint: {e}")
        return ""


def get_unit_dimensionality(unit_str: str) -> str:
    """
    Get human-readable dimensionality description for a unit.

    Uses pint's dimensionality and maps to physics-meaningful descriptions
    like "energy", "pressure", "magnetic field" instead of raw dimension formulas.

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Human-readable dimensionality description or raw pint format
    """
    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        unit = unit_registry(unit_str)

        # Get dimensionality
        if hasattr(unit, "dimensionality"):
            dim = unit.dimensionality
        elif hasattr(unit, "units") and hasattr(unit.units, "dimensionality"):
            dim = unit.units.dimensionality
        else:
            logger.debug(
                f"Unit '{unit_str}' parsed as {type(unit)} but no dimensionality found"
            )
            return "unknown"

        dim_str = str(dim)

        # Map to human-readable description if available
        if dim_str in DIMENSIONALITY_DESCRIPTIONS:
            return DIMENSIONALITY_DESCRIPTIONS[dim_str]

        # Return raw dimensionality if no mapping found
        return dim_str

    except Exception as e:
        logger.debug(f"Could not get dimensionality for unit '{unit_str}': {e}")

    return ""
