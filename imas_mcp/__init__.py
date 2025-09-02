"""
IMAS MCP Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata

# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
