"""
Learning persistence for facility exploration.

Supports two modes:
1. Legacy mode: Merges untyped YAML into facility config knowledge section
2. Artifact mode: Validates against Pydantic models and stores typed JSON artifacts

Artifact mode is triggered when an artifact_type is specified.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from imas_codex.discovery.config import get_facilities_dir
from imas_codex.remote.session import discard_session, get_session_status


# Lazy import of artifact models to avoid circular imports
def _get_artifact_models() -> dict[str, type[BaseModel]]:
    """Get mapping of artifact type names to Pydantic model classes."""
    from imas_codex.discovery.models import (
        DataArtifact,
        EnvironmentArtifact,
        FilesystemArtifact,
        ToolsArtifact,
    )

    return {
        "environment": EnvironmentArtifact,
        "tools": ToolsArtifact,
        "filesystem": FilesystemArtifact,
        "data": DataArtifact,
    }


# Keys that indicate which artifact type the data belongs to
ARTIFACT_KEY_HINTS = {
    "python": "environment",
    "os": "environment",
    "compilers": "environment",
    "module_system": "environment",
    "tools": "tools",
    "root_paths": "filesystem",
    "tree": "filesystem",
    "important_paths": "filesystem",
    "mdsplus": "data",
    "hdf5": "data",
    "netcdf": "data",
    "data_formats": "data",
    "shot_ranges": "data",
}


def get_facility_config_path(facility: str) -> Path:
    """Get the path to a facility's config file."""
    return get_facilities_dir() / f"{facility}.yaml"


def get_artifacts_dir(facility: str) -> Path:
    """Get the artifacts directory for a facility."""
    cache_dir = Path.home() / ".cache" / "imas-codex" / facility / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def load_facility_yaml(facility: str) -> dict[str, Any]:
    """Load raw facility YAML config."""
    config_path = get_facility_config_path(facility)
    if not config_path.exists():
        raise ValueError(f"Facility config not found: {config_path}")

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def save_facility_yaml(facility: str, data: dict[str, Any]) -> None:
    """Save facility YAML config."""
    config_path = get_facility_config_path(facility)

    with open(config_path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def deep_merge(base: dict, updates: dict) -> dict:
    """
    Deep merge updates into base dictionary.

    For lists, extends rather than replaces.
    For dicts, recursively merges.
    For other values, replaces.
    """
    result = base.copy()

    for key, value in updates.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                # Extend list, avoiding duplicates
                existing = {str(x) for x in result[key]}
                for item in value:
                    if str(item) not in existing:
                        result[key].append(item)
                        existing.add(str(item))
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def parse_learnings(input_str: str) -> dict[str, Any]:
    """
    Parse learnings from YAML or JSON string.

    Accepts either format and returns a dictionary.
    """
    input_str = input_str.strip()
    if not input_str:
        return {}

    # Try JSON first (for inline arguments)
    if input_str.startswith("{"):
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            pass

    # Try YAML
    try:
        result = yaml.safe_load(input_str)
        if isinstance(result, dict):
            return result
        return {}
    except yaml.YAMLError:
        return {}


def read_stdin_if_available() -> str:
    """Read from stdin if data is available (non-blocking check)."""
    import select

    # Check if stdin has data (Unix only)
    if hasattr(select, "select"):
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read()
    return ""


def detect_artifact_type(data: dict[str, Any]) -> str | None:
    """
    Detect artifact type from data keys.

    Returns the artifact type name or None if no match.
    """
    for key in data.keys():
        if key in ARTIFACT_KEY_HINTS:
            return ARTIFACT_KEY_HINTS[key]
    return None


def save_artifact(
    facility: str,
    artifact_type: str,
    data: dict[str, Any],
) -> tuple[bool, str, BaseModel | None]:
    """
    Validate and save a typed artifact.

    Args:
        facility: Facility identifier
        artifact_type: Type of artifact (environment, tools, filesystem, data)
        data: Artifact data to validate and save

    Returns:
        Tuple of (success, message, validated_artifact)
    """
    models = _get_artifact_models()

    if artifact_type not in models:
        valid_types = ", ".join(models.keys())
        return (
            False,
            f"Unknown artifact type: {artifact_type}. Valid: {valid_types}",
            None,
        )

    model_class = models[artifact_type]

    # Add metadata fields
    data["facility"] = facility
    data["explored_at"] = datetime.now(UTC).isoformat()

    # Validate against Pydantic model
    try:
        artifact = model_class.model_validate(data)
    except ValidationError as e:
        error_msg = _format_validation_error(e)
        return False, f"Validation failed for {artifact_type}:\n{error_msg}", None

    # Save to cache
    artifacts_dir = get_artifacts_dir(facility)
    artifact_path = artifacts_dir / f"{artifact_type}.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2))

    # Update manifest
    _update_manifest(facility, artifact_type)

    return True, f"Saved {artifact_type} artifact to {artifact_path}", artifact


def _format_validation_error(e: ValidationError) -> str:
    """Format Pydantic validation error for user display."""
    lines = []
    for error in e.errors():
        loc = ".".join(str(x) for x in error["loc"])
        msg = error["msg"]
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def _update_manifest(facility: str, artifact_type: str) -> None:
    """Update the manifest.json with exploration state."""
    artifacts_dir = get_artifacts_dir(facility)
    manifest_path = artifacts_dir / "manifest.json"

    # Load existing manifest or create new
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "facility": facility,
            "artifacts": {},
        }

    # Update artifact entry
    manifest["artifacts"][artifact_type] = {
        "status": "explored",
        "updated": datetime.now(UTC).isoformat(),
    }
    manifest["last_exploration"] = datetime.now(UTC).isoformat()

    manifest_path.write_text(json.dumps(manifest, indent=2))


def finish_session(
    facility: str,
    artifact_type: str | None = None,
    learnings: dict[str, Any] | str | None = None,
) -> tuple[bool, str]:
    """
    Persist learnings to artifact storage or facility config.

    Args:
        facility: Facility identifier
        artifact_type: Type of artifact (environment, tools, filesystem, data)
                      If None, uses legacy mode (merges to facility YAML)
        learnings: Dict of learnings, YAML/JSON string, or None to read from stdin

    Returns:
        Tuple of (success, message)
    """
    # Check session exists
    status = get_session_status(facility)
    if not status.exists:
        return False, f"No active session for {facility}"

    # Parse learnings
    if learnings is None:
        stdin_content = read_stdin_if_available()
        if stdin_content:
            learnings = parse_learnings(stdin_content)
        else:
            learnings = {}
    elif isinstance(learnings, str):
        learnings = parse_learnings(learnings)

    if not learnings:
        discard_session(facility)
        return True, f"Session cleared for {facility} (no learnings provided)"

    # Auto-detect artifact type if not specified
    if artifact_type is None:
        artifact_type = detect_artifact_type(learnings)

    # If we have an artifact type, use artifact mode
    if artifact_type is not None:
        success, message, _artifact = save_artifact(facility, artifact_type, learnings)
        if success:
            discard_session(facility)
        return success, message

    # Legacy mode: merge into facility YAML
    try:
        config = load_facility_yaml(facility)
    except ValueError as e:
        return False, str(e)

    if "knowledge" not in config:
        config["knowledge"] = {}

    config["knowledge"] = deep_merge(config["knowledge"], learnings)
    save_facility_yaml(facility, config)
    discard_session(facility)

    categories = list(learnings.keys())
    return True, f"Persisted learnings to {facility}: {', '.join(categories)}"


def list_artifacts(facility: str) -> dict[str, Any]:
    """
    List all artifacts for a facility.

    Returns manifest contents or empty dict if no artifacts exist.
    """
    artifacts_dir = get_artifacts_dir(facility)
    manifest_path = artifacts_dir / "manifest.json"

    if not manifest_path.exists():
        return {"facility": facility, "artifacts": {}}

    return json.loads(manifest_path.read_text())


def load_artifact(facility: str, artifact_type: str) -> dict[str, Any] | None:
    """
    Load a specific artifact for a facility.

    Returns artifact data or None if not found.
    """
    artifacts_dir = get_artifacts_dir(facility)
    artifact_path = artifacts_dir / f"{artifact_type}.json"

    if not artifact_path.exists():
        return None

    return json.loads(artifact_path.read_text())
