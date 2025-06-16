"""
IMAS MCP Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata
import warnings


# Filter whoosh SyntaxError warnings
warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message="invalid escape sequence"
)
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message='"is" with \'int\' literal. Did you mean "=="?',
)

# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-mcp")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
