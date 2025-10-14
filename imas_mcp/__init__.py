"""
IMAS MCP Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata
import os


def _get_dd_version() -> str:
    """
    Get DD version without expensive DD accessor creation.

    Checks environment variable first, then falls back to package __version__.
    This avoids costly XML parsing (560ms) during package import.

    Returns:
        Normalized DD version string (without git hash suffix).
    """
    # Check environment first (allows version override)
    if env_version := os.getenv("IMAS_DD_VERSION"):
        return env_version

    # Use package __version__ (fast - no XML parsing)
    try:
        import imas_data_dictionary

        version = imas_data_dictionary.__version__
        # Normalize: remove git hash suffix (e.g., '4.0.1.dev277+g8b28b0d89' -> '4.0.1.dev277')
        return version.split("+")[0] if "+" in version else version
    except ImportError:
        return "unknown"


# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Get DD version efficiently (no XML parsing)
dd_version = _get_dd_version()

__all__ = ["__version__", "dd_version"]
