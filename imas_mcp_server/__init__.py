"""
IMAS MCP Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata

import pint

# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-standard-names")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
__all__ = ["__version__"]


# register UDUNITS unit format with pint
@pint.register_unit_format("F")
def format_unit_simple(unit, registry, **options):
    return ".".join(u if p == 1 else f"{u}^{p}" for u, p in unit.items())
