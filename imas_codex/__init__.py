"""
IMAS Codex Server - A server providing Model Context Protocol (MCP) access to IMAS data structures.
"""

import importlib.metadata
import os
import warnings
from functools import lru_cache
from pathlib import Path

# Suppress third-party deprecation warnings early, before any imports
# These are upstream issues in Pydantic, LlamaIndex, and Neo4j that we cannot fix
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Relying on Driver's destructor.*")


@lru_cache(maxsize=1)
def _get_default_dd_version_from_pyproject() -> str:
    """Read default-dd-version from pyproject.toml [tool.imas-codex] section."""
    try:
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
            return (
                data.get("tool", {})
                .get("imas-codex", {})
                .get("default-dd-version", "4.1.0")
            )
    except Exception:
        pass
    return "4.1.0"


def _get_max_available_dd_version() -> str | None:
    """Get the maximum DD version available from installed packages."""
    # Try imas-data-dictionary (git dev package) first
    try:
        import imas_data_dictionary

        version = imas_data_dictionary.__version__
        # Normalize: remove git hash suffix (e.g., '4.0.1.dev277+g8b28b0d89' -> '4.0.1')
        return version.split(".dev")[0] if ".dev" in version else version.split("+")[0]
    except ImportError:
        pass

    # Try imas-data-dictionaries (PyPI package)
    try:
        import imas_data_dictionaries

        # Get available versions from the package
        versions = imas_data_dictionaries.list_dd_versions()
        if versions:
            from packaging.version import Version

            return str(max(Version(v) for v in versions))
    except (ImportError, AttributeError):
        pass

    return None


def _validate_dd_version(requested: str, max_available: str | None) -> None:
    """Raise error if requested version exceeds available version."""
    if max_available is None:
        return  # Can't validate, allow any version

    from packaging.version import Version

    try:
        req_version = Version(requested.split(".dev")[0].split("+")[0])
        max_version = Version(max_available)
        if req_version > max_version:
            raise ValueError(
                f"Requested DD version {requested} exceeds maximum available version {max_available}. "
                f"Update imas-data-dictionaries dependency or use a lower version."
            )
    except Exception as e:
        if isinstance(e, ValueError) and "exceeds maximum" in str(e):
            raise
        # Ignore parsing errors


def _get_dd_version(cli_version: str | None = None) -> str:
    """
    Get DD version without expensive DD accessor creation.

    Priority order:
    1. CLI argument (if provided)
    2. IMAS_DD_VERSION environment variable
    3. default-dd-version from pyproject.toml [tool.imas-codex]
    4. imas-data-dictionary package __version__ (git dev package, fallback)

    Validates that the resolved version doesn't exceed the installed package version.

    Args:
        cli_version: Optional DD version specified via CLI argument.

    Returns:
        Normalized DD version string (without git hash suffix).

    Raises:
        ValueError: If requested version exceeds available package version.
    """
    # Get max available for validation
    max_available = _get_max_available_dd_version()

    # Check CLI argument first (highest priority)
    if cli_version:
        _validate_dd_version(cli_version, max_available)
        return cli_version

    # Check environment second (allows version override)
    if env_version := os.getenv("IMAS_DD_VERSION"):
        _validate_dd_version(env_version, max_available)
        return env_version

    # Use configured default from pyproject.toml
    default = _get_default_dd_version_from_pyproject()
    _validate_dd_version(default, max_available)
    return default


# import version from project metadata
try:
    __version__ = importlib.metadata.version("imas-codex")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Get DD version efficiently (no XML parsing)
dd_version = _get_dd_version()

__all__ = ["__version__", "dd_version", "_get_dd_version"]
