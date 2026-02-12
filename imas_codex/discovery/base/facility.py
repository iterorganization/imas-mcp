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
from typing import TYPE_CHECKING, Any

import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from imas_codex.graph.schema import get_schema

if TYPE_CHECKING:
    from imas_codex.config.models import FacilityConfig

logger = logging.getLogger(__name__)

# Configure ruamel.yaml for comment-preserving round-trips
_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 120


def get_config_dir() -> Path:
    """Get the config directory path."""
    # facility.py is at imas_codex/discovery/base/facility.py
    # config is at imas_codex/config
    return Path(__file__).parent.parent.parent / "config"


def get_facilities_dir() -> Path:
    """Get the facilities config directory."""
    return get_config_dir() / "facilities"


def list_facilities() -> list[str]:
    """List all available facility configurations.

    Returns:
        List of facility IDs (e.g., ["tcv", "iter"])
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
    """Deep merge updates into ruamel CommentedMap, preserving comments.

    Recursively merges nested dicts, extends lists without duplicates.
    New keys are added, existing keys are updated additively.
    """
    for key, value in updates.items():
        if key in base:
            if isinstance(base[key], CommentedMap) and isinstance(value, dict):
                # Recursively merge nested dicts (additive, not replacement)
                _deep_merge_ruamel(base[key], value)
            elif isinstance(base[key], list | CommentedSeq) and isinstance(value, list):
                # Extend lists, avoiding duplicates
                for item in value:
                    if item not in base[key]:
                        base[key].append(item)
            else:
                base[key] = value
        else:
            # New key - add it, converting dicts to CommentedMap for consistency
            if isinstance(value, dict) and not isinstance(value, CommentedMap):
                new_map = CommentedMap()
                for k, v in value.items():
                    new_map[k] = v
                base[key] = new_map
            else:
                base[key] = value
    return base


@lru_cache(maxsize=8)
def get_facility(facility_id: str) -> dict[str, Any]:
    """Load complete facility configuration (public + private merged).

    Merges data from:
    - <facility>.yaml (public, git-tracked)
    - <facility>_private.yaml (private, gitignored)

    Use this when you need full context for exploration.
    For graph ingestion, use get_facility_metadata() instead.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")

    Returns:
        Merged facility data as dict with all fields

    Raises:
        ValueError: If facility not found

    Examples:
        >>> config = get_facility('iter')
        >>> print(config['paths'])  # Has both public and private paths
        >>> print(config['tools'])  # Tool availability (private)
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


def validate_facility_config(
    facility_id: str, *, raise_on_error: bool = False
) -> list[str]:
    """Validate facility config against LinkML-generated schema.

    Checks that the facility YAML files conform to the schema defined in
    imas_codex/schemas/facility_config.yaml. Unknown keys are allowed
    (for YAML anchors and forward compatibility).

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")
        raise_on_error: If True, raise ValidationError; if False, return error list

    Returns:
        List of validation error messages (empty = valid)

    Raises:
        pydantic.ValidationError: If raise_on_error=True and validation fails

    Examples:
        >>> errors = validate_facility_config('tcv')
        >>> if errors:
        ...     print(f"Config issues: {errors}")
    """
    from pydantic import ValidationError

    from imas_codex.config.models import FacilityConfig

    config = get_facility(facility_id)

    try:
        FacilityConfig.model_validate(config)
        return []
    except ValidationError as e:
        if raise_on_error:
            raise
        return [str(err) for err in e.errors()]


def get_facility_validated(facility_id: str) -> "FacilityConfig":
    """Load and validate facility config, returning typed Pydantic model.

    Use this when you need type-safe access to config fields.
    Raises ValidationError if config is invalid.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")

    Returns:
        FacilityConfig model instance with validated fields

    Raises:
        pydantic.ValidationError: If config doesn't match schema

    Examples:
        >>> config = get_facility_validated('tcv')
        >>> for root in config.discovery_roots or []:
        ...     print(root)
    """
    from imas_codex.config.models import FacilityConfig

    config = get_facility(facility_id)
    return FacilityConfig.model_validate(config)


def get_facility_metadata(facility_id: str) -> dict[str, Any]:
    """Load only public facility metadata (safe for graph/OCI).

    Uses schema introspection to filter out private fields.
    Use this before writing facility data to the graph.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")

    Returns:
        Dict with only public fields (name, machine, data_systems, etc.)

    Examples:
        >>> metadata = get_facility_metadata('iter')
        >>> print('ssh_host' in metadata)  # False - private field filtered
        >>> print('machine' in metadata)   # True - public field
    """
    full = get_facility(facility_id)
    schema = get_schema()
    private_slots = set(schema.get_private_slots("Facility"))

    return {k: v for k, v in full.items() if k not in private_slots}


def get_facility_infrastructure(facility_id: str) -> dict[str, Any]:
    """Load only private facility infrastructure data.

    Returns sensitive infrastructure data: hostnames, paths, tools, OS versions.
    This data is never stored in the graph or OCI artifacts.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")

    Returns:
        Dict with only private fields (hostnames, paths, tools, etc.)

    Examples:
        >>> infra = get_facility_infrastructure('iter')
        >>> print(infra['tools'])  # Tool versions
        >>> print(infra['paths'])  # File system paths
    """
    facilities_dir = get_facilities_dir()
    private_path = facilities_dir / f"{facility_id}_private.yaml"
    return _load_yaml(private_path)


def update_infrastructure(facility_id: str, data: dict[str, Any]) -> None:
    """Update private facility infrastructure data.

    Deep-merges the provided data into the existing private file,
    preserving comments and key order using ruamel.yaml.

    Use this for sensitive infrastructure data:
    - Tool versions and availability
    - File system paths
    - Hostnames and network info
    - OS and environment details
    - Exploration notes

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")
        data: Data to merge into private file

    Examples:
        >>> # Update tool availability
        >>> update_infrastructure('iter', {
        ...     'tools': {'rg': '14.1.1', 'fd': '10.2.0'}
        ... })

        >>> # Add exploration notes
        >>> update_infrastructure('iter', {
        ...     'exploration_notes': ['Found IMAS modules at /work/imas']
        ... })
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

    logger.info(f"Updated infrastructure data for {facility_id}")


def update_metadata(facility_id: str, data: dict[str, Any]) -> None:
    """Update public facility metadata.

    Updates the public facility YAML file with metadata that is safe
    to store in the graph and version control.

    Use this for public metadata:
    - Facility name and description
    - Machine name
    - Data system types
    - Wiki site URLs (if public)

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")
        data: Data to merge into public file

    Examples:
        >>> # Update facility description
        >>> update_metadata('iter', {
        ...     'description': 'ITER SDCC - Updated description'
        ... })
    """
    facilities_dir = get_facilities_dir()
    public_path = facilities_dir / f"{facility_id}.yaml"

    if not public_path.exists():
        msg = f"Public config not found: {public_path}"
        raise ValueError(msg)

    # Load existing with ruamel for comment preservation
    with public_path.open() as f:
        existing = _yaml.load(f)
    if existing is None:
        existing = CommentedMap()

    # Deep merge preserving comments
    _deep_merge_ruamel(existing, data)

    # Write back with preserved formatting
    with public_path.open("w") as f:
        _yaml.dump(existing, f)

    # Clear cache since data changed
    get_facility.cache_clear()

    logger.info(f"Updated metadata for {facility_id}")


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


def add_exploration_note(facility_id: str, note: str) -> list[str]:
    """Add a timestamped exploration note to facility's private data.

    Automatically adds ISO timestamp prefix to the note. Notes are stored
    in the exploration_notes list in the private config.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "iter")
        note: Exploration note to add

    Returns:
        Updated exploration_notes list

    Examples:
        >>> add_exploration_note("iter", "Found IMAS modules at /work/imas")
        >>> add_exploration_note("iter", "Discovered 50 Python files in /work/codes")
    """
    from datetime import UTC, datetime

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamped_note = f"[{timestamp}] {note}"

    # Load current notes
    infra = get_facility_infrastructure(facility_id)
    notes = infra.get("exploration_notes", [])

    # Add new note
    notes.append(timestamped_note)

    # Persist
    update_infrastructure(facility_id, {"exploration_notes": notes})

    logger.info(f"Added exploration note for {facility_id}: {note[:50]}...")
    return notes
