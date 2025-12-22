"""
Artifact capture for facility exploration.

Validates learnings against Pydantic models and stores typed JSON artifacts.
Supports partial updates - new learnings are merged into existing artifacts.
Session remains open after capture to allow multiple writes.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError


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


def get_artifacts_dir(facility: str) -> Path:
    """Get the artifacts directory for a facility."""
    cache_dir = Path.home() / ".cache" / "imas-codex" / facility / "artifacts"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def deep_merge(base: dict, updates: dict) -> dict:
    """
    Deep merge updates into base dictionary.

    For lists, extends rather than replaces (avoiding duplicates).
    For dicts, recursively merges.
    For other values, replaces.
    """
    result = base.copy()

    for key, value in updates.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
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

    if input_str.startswith("{"):
        try:
            return json.loads(input_str)
        except json.JSONDecodeError:
            pass

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

    if hasattr(select, "select"):
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable:
            return sys.stdin.read()
    return ""


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


def save_artifact(
    facility: str,
    artifact_type: str,
    data: dict[str, Any],
) -> tuple[bool, str, BaseModel | None]:
    """
    Validate and save a typed artifact with partial update support.

    Loads existing artifact (if any), merges new data, validates the merged
    result against the Pydantic model, and saves.

    Args:
        facility: Facility identifier
        artifact_type: Type of artifact (environment, tools, filesystem, data)
        data: Artifact data to merge and validate

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

    existing = load_artifact(facility, artifact_type)
    if existing:
        if "facility" in existing:
            del existing["facility"]
        if "explored_at" in existing:
            del existing["explored_at"]
        merged_data = deep_merge(existing, data)
    else:
        merged_data = data

    merged_data["facility"] = facility
    merged_data["explored_at"] = datetime.now(UTC).isoformat()

    try:
        artifact = model_class.model_validate(merged_data)
    except ValidationError as e:
        error_msg = _format_validation_error(e)
        return False, f"Validation failed for {artifact_type}:\n{error_msg}", None

    artifacts_dir = get_artifacts_dir(facility)
    artifact_path = artifacts_dir / f"{artifact_type}.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2))

    _update_manifest(facility, artifact_type)

    action = "Updated" if existing else "Created"
    return True, f"{action} {artifact_type} artifact: {artifact_path}", artifact


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

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "facility": facility,
            "artifacts": {},
        }

    manifest["artifacts"][artifact_type] = {
        "status": "explored",
        "updated": datetime.now(UTC).isoformat(),
    }
    manifest["last_exploration"] = datetime.now(UTC).isoformat()

    manifest_path.write_text(json.dumps(manifest, indent=2))


def capture_artifact(
    facility: str,
    artifact_type: str,
    learnings: dict[str, Any] | str | None = None,
) -> tuple[bool, str]:
    """
    Capture learnings to typed artifact storage.

    Session remains open after capture to allow multiple writes.
    Use --discard to clear the session when done exploring.

    Args:
        facility: Facility identifier
        artifact_type: Type of artifact (environment, tools, filesystem, data)
        learnings: Dict of learnings, YAML/JSON string, or None to read from stdin

    Returns:
        Tuple of (success, message)
    """
    if learnings is None:
        stdin_content = read_stdin_if_available()
        if stdin_content:
            learnings = parse_learnings(stdin_content)
        else:
            learnings = {}
    elif isinstance(learnings, str):
        learnings = parse_learnings(learnings)

    if not learnings:
        return False, "No learnings provided"

    success, message, _artifact = save_artifact(facility, artifact_type, learnings)
    return success, message


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
