"""Unit handling for IMAS MCP Server."""

import importlib.resources
from typing import Any

import pint


# register UDUNITS unit format with pint
@pint.register_unit_format("U")
def format_unit_simple(
    unit, registry: pint.UnitRegistry, **options: dict[str, Any]
) -> str:
    return ".".join(u if p == 1 else f"{u}^{p}" for u, p in unit.items())


# Initialize unit registry
unit_registry = pint.UnitRegistry()

# Load non-SI Data Dictionary unit aliases
with importlib.resources.as_file(
    importlib.resources.files("imas_mcp.units").joinpath(
        "data_dictionary_unit_aliases.txt"
    )
) as resource_path:
    unit_registry.load_definitions(str(resource_path))
