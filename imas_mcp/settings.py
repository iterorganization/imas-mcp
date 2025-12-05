"""Project settings loaded from pyproject.toml [tool.imas-mcp] section.

This module provides centralized access to project configuration defaults,
with environment variable overrides for runtime flexibility.
"""

import importlib.resources
import os
from functools import cache

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]


@cache
def _load_pyproject_settings() -> dict[str, str]:
    """Load settings from pyproject.toml [tool.imas-mcp] section.

    Returns:
        Dictionary of settings from pyproject.toml, empty dict if not found.
    """
    try:
        # Try package resources first (installed package)
        files = importlib.resources.files("imas_mcp")
        pyproject_path = files.joinpath("..", "pyproject.toml")

        # If package resource doesn't exist, try filesystem
        if not pyproject_path.is_file():  # type: ignore[union-attr]
            from pathlib import Path

            # Walk up to find pyproject.toml (for development)
            current = Path(__file__).resolve().parent
            while current != current.parent:
                candidate = current / "pyproject.toml"
                if candidate.exists():
                    pyproject_path = candidate
                    break
                current = current.parent
            else:
                return {}

        # Read and parse the TOML file
        if hasattr(pyproject_path, "read_text"):
            content = pyproject_path.read_text()  # type: ignore[union-attr]
        else:
            from pathlib import Path

            content = Path(pyproject_path).read_text()  # type: ignore[arg-type]

        data = tomllib.loads(content)
        return data.get("tool", {}).get("imas-mcp", {})
    except Exception:
        return {}


def get_imas_embedding_model() -> str:
    """Get the IMAS DD embedding model name.

    Priority:
        1. IMAS_MCP_EMBEDDING_MODEL environment variable
        2. pyproject.toml [tool.imas-mcp] imas-embedding-model
        3. Fallback default: all-MiniLM-L6-v2 (local model)

    Returns:
        Model name string.
    """
    if env_model := os.getenv("IMAS_MCP_EMBEDDING_MODEL"):
        return env_model

    settings = _load_pyproject_settings()
    if model := settings.get("imas-embedding-model"):
        return model

    return "all-MiniLM-L6-v2"


def get_docs_embedding_model() -> str:
    """Get the docs server embedding model name.

    Priority:
        1. DOCS_MCP_EMBEDDING_MODEL environment variable
        2. pyproject.toml [tool.imas-mcp] docs-embedding-model
        3. Fallback default: openai/text-embedding-3-small

    Returns:
        Model name string.
    """
    if env_model := os.getenv("DOCS_MCP_EMBEDDING_MODEL"):
        return env_model

    settings = _load_pyproject_settings()
    if model := settings.get("docs-embedding-model"):
        return model

    return "openai/text-embedding-3-small"


def _parse_bool(value: str | bool) -> bool:
    """Parse a boolean value from string or bool."""
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes")


def get_include_ggd() -> bool:
    """Get whether to include GGD (Grid Geometry Description) paths.

    Priority:
        1. IMAS_MCP_INCLUDE_GGD environment variable
        2. pyproject.toml [tool.imas-mcp] include-ggd
        3. Fallback default: True

    Returns:
        Boolean indicating whether to include GGD paths.
    """
    if env_val := os.getenv("IMAS_MCP_INCLUDE_GGD"):
        return _parse_bool(env_val)

    settings = _load_pyproject_settings()
    if (val := settings.get("include-ggd")) is not None:
        return _parse_bool(val)

    return True


def get_include_error_fields() -> bool:
    """Get whether to include error fields (_error_upper, _error_lower, etc.).

    Priority:
        1. IMAS_MCP_INCLUDE_ERROR_FIELDS environment variable
        2. pyproject.toml [tool.imas-mcp] include-error-fields
        3. Fallback default: False

    Returns:
        Boolean indicating whether to include error fields.
    """
    if env_val := os.getenv("IMAS_MCP_INCLUDE_ERROR_FIELDS"):
        return _parse_bool(env_val)

    settings = _load_pyproject_settings()
    if (val := settings.get("include-error-fields")) is not None:
        return _parse_bool(val)

    return False


# Computed defaults (for use in module-level constants)
IMAS_MCP_EMBEDDING_MODEL = get_imas_embedding_model()
DOCS_MCP_EMBEDDING_MODEL = get_docs_embedding_model()
INCLUDE_GGD = get_include_ggd()
INCLUDE_ERROR_FIELDS = get_include_error_fields()
