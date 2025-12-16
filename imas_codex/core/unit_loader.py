"""
Unit utilities for IMAS Data Dictionary.

This module provides pint-based unit name and dimensionality lookup
for physical units used in the data dictionary.
"""

import logging

from imas_codex.units import unit_registry

logger = logging.getLogger(__name__)


def get_unit_name(unit_str: str) -> str:
    """
    Get unit name for a unit using pint with custom 'U' formatter.

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Unit name from pint formatter
    """
    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        # Parse the unit with pint
        unit = unit_registry(unit_str)

        # Use the custom 'U' formatter to get unit names
        unit_name = f"{unit.units:U}"
        return unit_name

    except Exception as e:
        logger.debug(f"Could not parse unit '{unit_str}' with pint: {e}")
        return ""


def get_unit_dimensionality(unit_str: str) -> str:
    """
    Get dimensionality description for a unit using pint's built-in formatting.

    Args:
        unit_str: Unit string (e.g., "T", "eV", "Pa")

    Returns:
        Pint's dimensionality string or empty string
    """
    try:
        # Handle special case for dimensionless units
        if unit_str in ("1", "dimensionless", ""):
            return "dimensionless"

        unit = unit_registry(unit_str)

        # Check if unit has dimensionality attribute (some parsed units might not)
        if hasattr(unit, "dimensionality"):
            dimensionality = str(unit.dimensionality)
        elif hasattr(unit, "units") and hasattr(unit.units, "dimensionality"):
            dimensionality = str(unit.units.dimensionality)
        else:
            # Fallback for cases where dimensionality is not accessible
            logger.debug(
                f"Unit '{unit_str}' parsed as {type(unit)} but no dimensionality found"
            )
            return "unknown"

        # Return pint's clean dimensionality formatting directly
        return dimensionality

    except Exception as e:
        logger.debug(f"Could not get dimensionality for unit '{unit_str}': {e}")

    return ""
