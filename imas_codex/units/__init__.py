"""Unit handling for IMAS Codex Server."""

import importlib.resources
import logging
from functools import lru_cache
from typing import Any

import pint

logger = logging.getLogger(__name__)


# register UDUNITS unit format with pint (guard against re-import)
def format_unit_simple(
    unit, registry: pint.UnitRegistry, **options: dict[str, Any]
) -> str:
    return ".".join(u if p == 1 else f"{u}^{p}" for u, p in unit.items())


if "U" not in pint.formatting.REGISTERED_FORMATTERS:
    pint.register_unit_format("U")(format_unit_simple)


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

    Returns a compact ASCII symbol for graph storage. Equivalent unit
    expressions (e.g., ``m.s^-1`` and ``m/s``) produce the same output.

    Examples:
        >>> normalize_unit_symbol("Ohm")
        'ohm'
        >>> normalize_unit_symbol("H.m^-1")
        'H/m'
        >>> normalize_unit_symbol("m.s^-1")
        'm/s'
        >>> normalize_unit_symbol("mixed")  # sentinel
        >>> normalize_unit_symbol("A/m^2")
        'A/m^2'

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
        # ~ gives short symbols (H, m, T), then convert to ASCII
        compact = f"{parsed.units:~}"
        compact = (
            compact.replace("Ω", "ohm")
            .replace(" ** ", "^")
            .replace(" * ", ".")
            .replace(" / ", "/")
        )
        return compact
    except Exception:
        logger.debug("Could not normalize unit '%s'", raw)
        return None
