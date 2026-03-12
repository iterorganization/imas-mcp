"""Unit handling for IMAS Codex Server."""

import importlib.resources
import logging
from functools import lru_cache

import pint

logger = logging.getLogger(__name__)


# Initialize unit registry
unit_registry = pint.UnitRegistry()

# Load non-SI Data Dictionary unit aliases
with importlib.resources.as_file(
    importlib.resources.files("imas_codex.units").joinpath(
        "data_dictionary_unit_aliases.txt"
    )
) as resource_path:
    unit_registry.load_definitions(str(resource_path))


# Sentinel unit strings that are not physical units
_NON_UNIT_STRINGS = frozenset(
    {
        "-",
        "1",
        "mixed",
        "as parent",
        "as_parent",
        "as_parent_level_2",
        "Toroidal angle",
        "dimensionless",
        "",
    }
)


@lru_cache(maxsize=512)
def normalize_unit_symbol(raw: str) -> str | None:
    """Normalize a unit string to a canonical symbol via pint.

    Returns pint's default short notation for graph storage. Equivalent unit
    expressions (e.g., ``m.s^-1`` and ``m/s``) produce the same output.

    Uses ``*`` for multiplication and ``**`` for exponentiation per pint
    convention. Spaces around operators are stripped for compact storage.

    Examples:
        >>> normalize_unit_symbol("Ohm")
        'ohm'
        >>> normalize_unit_symbol("H.m^-1")
        'H/m'
        >>> normalize_unit_symbol("m.s^-1")
        'm/s'
        >>> normalize_unit_symbol("mixed")  # sentinel
        >>> normalize_unit_symbol("A/m^2")
        'A/m**2'
        >>> normalize_unit_symbol("kg.m.s^-2")
        'kg*m/s**2'

    Args:
        raw: Raw unit string from MDSplus or IMAS DD.

    Returns:
        Normalized symbol string, or None if unparseable/not a unit.
    """
    if not raw or raw in _NON_UNIT_STRINGS:
        return None
    if raw.startswith("units given") or raw.startswith("as_parent"):
        return None

    try:
        parsed = unit_registry.parse_expression(raw)
        # ~ gives short symbols (H, m, T), then compact spacing
        compact = f"{parsed.units:~}"
        compact = (
            compact.replace("Ω", "ohm")
            .replace(" ** ", "**")
            .replace(" * ", "*")
            .replace(" / ", "/")
        )
        return compact
    except Exception:
        logger.debug("Could not normalize unit '%s'", raw)
        return None


def validate_unit(unit_str: str) -> str | None:
    """Validate unit string against pint and return canonical short form.

    Used as a post-enrichment validation step: if the LLM-extracted unit
    is invalid, returns None (clear rather than store garbage). If valid,
    returns the pint-canonical short form.

    Args:
        unit_str: Raw unit string from LLM enrichment.

    Returns:
        Canonical short-form unit string, or None if invalid.
    """
    if not unit_str or not unit_str.strip():
        return None
    return normalize_unit_symbol(unit_str.strip())
