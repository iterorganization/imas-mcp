"""
IMAS MCP Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata
import os


def _get_dd_version(cli_version: str | None = None) -> str:
    """
    Get DD version without expensive DD accessor creation.

    Checks CLI argument first, then environment variable, then falls back to package __version__,
    and finally defaults to "4.0.0" if neither is available.
    This avoids costly XML parsing (560ms) during package import.

    Args:
        cli_version: Optional DD version specified via CLI argument.

    Returns:
        Normalized DD version string (without git hash suffix).
    """
    # Check CLI argument first (highest priority)
    if cli_version:
        return cli_version

    # Check environment second (allows version override)
    if env_version := os.getenv("IMAS_DD_VERSION"):
        return env_version

    # Try to use imas-data-dictionary (git dev package) if available
    try:
        import imas_data_dictionary

        version = imas_data_dictionary.__version__
        # Normalize: remove git hash suffix (e.g., '4.0.1.dev277+g8b28b0d89' -> '4.0.1.dev277')
        return version.split("+")[0] if "+" in version else version
    except ImportError:
        # Default to 4.0.0 when using imas-data-dictionaries PyPI package
        # This is the stable version that imas-data-dictionaries provides
        return "4.0.0"


# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Get DD version efficiently (no XML parsing)
dd_version = _get_dd_version()

# Docs MCP server version - pinned to avoid npx download/verification delays
DOCS_MCP_SERVER_VERSION = "1.29.0"

__all__ = ["__version__", "dd_version", "DOCS_MCP_SERVER_VERSION"]
