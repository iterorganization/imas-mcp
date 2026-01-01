"""
Unified facility configuration loading.

Loads public (<facility>.yaml) and private (<facility>_private.yaml) data,
merging them into a single Facility model. Private fields are identified
via is_private annotations in the schema.

Key functions:
- get_facility(name) -> Facility: Load merged facility data
- save_private(name, data): Write private fields to *_private.yaml
- list_facilities() -> list[str]: List available facilities
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from imas_codex.graph.schema import get_schema

logger = logging.getLogger(__name__)

# Configure ruamel.yaml for comment-preserving round-trips
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120


def get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).parent.parent / "config"


def get_facilities_dir() -> Path:
    """Get the facilities config directory."""
    return get_config_dir() / "facilities"


def list_facilities() -> list[str]:
    """List all available facility configurations.

    Returns:
        List of facility IDs (e.g., ["epfl", "iter"])
    """
    facilities_dir = get_facilities_dir()
    if not facilities_dir.exists():
        return []
    return [
        p.stem for p in facilities_dir.glob("*.yaml") if not p.stem.endswith("_private")
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, updates: dict) -> dict:
    """Deep merge updates into base dict."""
    result = base.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _deep_merge_ruamel(base: CommentedMap, updates: dict[str, Any]) -> CommentedMap:
    """Deep merge updates into ruamel CommentedMap, preserving comments."""
    for key, value in updates.items():
        if key in base:
            if isinstance(base[key], CommentedMap) and isinstance(value, dict):
                _deep_merge_ruamel(base[key], value)
            elif isinstance(base[key], list | CommentedSeq) and isinstance(value, list):
                # Extend lists, avoiding duplicates
                for item in value:
                    if item not in base[key]:
                        base[key].append(item)
            else:
                base[key] = value
        else:
            base[key] = value
    return base


@lru_cache(maxsize=8)
def get_facility(facility_id: str) -> dict[str, Any]:
    """Load complete facility configuration (public + private).

    Merges data from:
    - <facility>.yaml (public, git-tracked)
    - <facility>_private.yaml (private, gitignored)

    Args:
        facility_id: Facility identifier (e.g., "epfl")

    Returns:
        Merged facility data as dict with all fields

    Raises:
        ValueError: If facility not found
    """
    facilities_dir = get_facilities_dir()
    public_path = facilities_dir / f"{facility_id}.yaml"
    private_path = facilities_dir / f"{facility_id}_private.yaml"

    if not public_path.exists():
        available = list_facilities()
        msg = f"Unknown facility: {facility_id}. Available: {available}"
        raise ValueError(msg)

    # Load both files
    public_data = _load_yaml(public_path)
    private_data = _load_yaml(private_path)

    # Merge private into public (private wins on conflict)
    merged = _deep_merge(public_data, private_data)

    # Ensure facility ID is set
    merged["id"] = facility_id
    if "facility" in merged:
        merged["id"] = merged.pop("facility")

    return merged


def get_facility_public(facility_id: str) -> dict[str, Any]:
    """Load only public facility fields (safe for graph/OCI).

    Uses schema introspection to filter out private fields.

    Args:
        facility_id: Facility identifier

    Returns:
        Dict with only public fields
    """
    full = get_facility(facility_id)
    schema = get_schema()
    private_slots = set(schema.get_private_slots("Facility"))

    return {k: v for k, v in full.items() if k not in private_slots}


def get_facility_private(facility_id: str) -> dict[str, Any]:
    """Load only private facility fields.

    Args:
        facility_id: Facility identifier

    Returns:
        Dict with only private fields
    """
    facilities_dir = get_facilities_dir()
    private_path = facilities_dir / f"{facility_id}_private.yaml"
    return _load_yaml(private_path)


def save_private(facility_id: str, data: dict[str, Any]) -> None:
    """Save or merge private data for a facility.

    Deep-merges the provided data into the existing private file,
    preserving comments and key order using ruamel.yaml.

    Args:
        facility_id: Facility identifier
        data: Data to merge into private file
    """
    facilities_dir = get_facilities_dir()
    private_path = facilities_dir / f"{facility_id}_private.yaml"

    # Load existing with ruamel for comment preservation
    if private_path.exists():
        with private_path.open() as f:
            existing = _yaml.load(f)
        if existing is None:
            existing = CommentedMap()
    else:
        existing = CommentedMap()

    # Deep merge preserving comments
    _deep_merge_ruamel(existing, data)

    # Write back with preserved formatting
    with private_path.open("w") as f:
        _yaml.dump(existing, f)

    # Clear cache since data changed
    get_facility.cache_clear()

    logger.info(f"Updated private data for {facility_id}")


def validate_no_private_fields(
    data: dict[str, Any], node_type: str = "Facility"
) -> list[str]:
    """Check if data contains any private fields.

    Used for release validation before OCI push.

    Args:
        data: Data to validate
        node_type: Schema class to check against

    Returns:
        List of private field names found in data (empty = OK)
    """
    schema = get_schema()
    private_slots = set(schema.get_private_slots(node_type))
    return [k for k in data.keys() if k in private_slots]


def filter_private_fields(
    data: dict[str, Any], node_type: str = "Facility"
) -> dict[str, Any]:
    """Remove private fields from data.

    Used by ingest_node to ensure private data never enters graph.

    Args:
        data: Data to filter
        node_type: Schema class to check against

    Returns:
        Data with private fields removed
    """
    schema = get_schema()
    private_slots = set(schema.get_private_slots(node_type))
    return {k: v for k, v in data.items() if k not in private_slots}
