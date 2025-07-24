"""
IMAS MCP Definitions Package

This package contains all data definitions, schemas, and templates used by the IMAS MCP.

Structure:
- physics/: Physics-related definitions (domains, units, constants)
- imas/: IMAS-specific definitions (data dictionary, workflows, metadata)
- validation/: JSON schemas for validation
- templates/: Template files for generating new definitions
"""

from pathlib import Path
from typing import Dict, Any
import yaml


def get_definitions_path() -> Path:
    """Get the path to the definitions directory."""
    return Path(__file__).parent


def load_definition_file(relative_path: str) -> Dict[str, Any]:
    """Load a YAML definition file by relative path from definitions root."""
    definitions_path = get_definitions_path()
    file_path = definitions_path / relative_path

    if not file_path.exists():
        raise FileNotFoundError(f"Definition file not found: {relative_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


__all__ = ["get_definitions_path", "load_definition_file"]
