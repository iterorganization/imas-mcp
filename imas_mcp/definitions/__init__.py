"""
IMAS MCP Definitions Package

This package contains all data definitions, schemas, and templates used by the IMAS MCP.

Structure:
- physics/: Physics-related definitions (domains, units, constants)
- imas/: IMAS-specific definitions (data dictionary, workflows, metadata)
- validation/: JSON schemas for validation
- templates/: Template files for generating new definitions
"""

from importlib import resources
from pathlib import Path
from typing import Any

import yaml


def get_definitions_path() -> Path:
    """Get the path to the definitions directory."""
    definitions_resource = resources.files("imas_mcp.definitions")
    return Path(str(definitions_resource))


def load_definition_file(relative_path: str) -> dict[str, Any]:
    """Load a YAML definition file by relative path from definitions root."""
    definitions_path = get_definitions_path()
    file_path = definitions_path / relative_path

    if not file_path.exists():
        raise FileNotFoundError(f"Definition file not found: {relative_path}")

    with open(file_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["get_definitions_path", "load_definition_file"]
