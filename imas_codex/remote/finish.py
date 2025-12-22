"""
Learning persistence for facility exploration.

Merges learnings from an exploration session into the facility
configuration file and clears the session log.
"""

import sys
from pathlib import Path
from typing import Any

import yaml

from imas_codex.discovery.config import get_facilities_dir
from imas_codex.remote.session import discard_session, get_session_status


def get_facility_config_path(facility: str) -> Path:
    """Get the path to a facility's config file."""
    return get_facilities_dir() / f"{facility}.yaml"


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

    # Preserve comments by reading and updating
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
    import json

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


def finish_session(
    facility: str,
    learnings: dict[str, Any] | str | None = None,
) -> tuple[bool, str]:
    """
    Persist learnings to facility config and clear session.

    Args:
        facility: Facility identifier
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
        # Try to read from stdin
        stdin_content = read_stdin_if_available()
        if stdin_content:
            learnings = parse_learnings(stdin_content)
        else:
            learnings = {}
    elif isinstance(learnings, str):
        learnings = parse_learnings(learnings)

    if not learnings:
        # No learnings provided - just clear session
        discard_session(facility)
        return True, f"Session cleared for {facility} (no learnings provided)"

    # Load current config
    try:
        config = load_facility_yaml(facility)
    except ValueError as e:
        return False, str(e)

    # Ensure knowledge section exists
    if "knowledge" not in config:
        config["knowledge"] = {}

    # Merge learnings into knowledge section
    config["knowledge"] = deep_merge(config["knowledge"], learnings)

    # Save updated config
    save_facility_yaml(facility, config)

    # Clear session
    discard_session(facility)

    # Build summary
    categories = list(learnings.keys())
    return True, f"Persisted learnings to {facility}: {', '.join(categories)}"
