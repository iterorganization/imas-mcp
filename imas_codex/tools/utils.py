"""Shared utilities for MCP tool parameter handling."""

import json

from imas_codex.core.paths import strip_path_annotations


def _try_parse_json_list(s: str) -> list[str] | None:
    """Try to parse a string as a JSON array of strings."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def normalize_ids_filter(ids_filter: str | list[str] | None) -> list[str] | None:
    """Normalize ids_filter to list format.

    Accepts:
        - None -> None
        - ["equilibrium", "magnetics"] -> ["equilibrium", "magnetics"]
        - "equilibrium magnetics" -> ["equilibrium", "magnetics"]
        - "equilibrium, magnetics" -> ["equilibrium", "magnetics"]
        - "equilibrium" -> ["equilibrium"]

    Args:
        ids_filter: IDS filter in various formats.

    Returns:
        Normalized list of IDS names, or None if input is None/empty.
    """
    if ids_filter is None:
        return None

    if isinstance(ids_filter, list):
        result = [s.strip() for s in ids_filter if s and s.strip()]
        return result if result else None

    if isinstance(ids_filter, str):
        ids_filter = ids_filter.strip()
        if not ids_filter:
            return None

        # Handle JSON array strings from MCP transport
        json_parsed = _try_parse_json_list(ids_filter)
        if json_parsed is not None:
            return json_parsed if json_parsed else None

        # Handle comma-delimited first (higher priority)
        if "," in ids_filter:
            result = [s.strip() for s in ids_filter.split(",") if s.strip()]
            return result if result else None

        # Handle space-delimited
        if " " in ids_filter:
            result = [s.strip() for s in ids_filter.split() if s.strip()]
            return result if result else None

        return [ids_filter]

    return None


def normalize_paths_input(paths: str | list[str]) -> list[str]:
    """Normalize paths parameter to list format.

    Accepts:
        - "path1 path2" -> ["path1", "path2"]
        - ["path1", "path2"] -> ["path1", "path2"]
        - '["path1","path2"]' -> ["path1", "path2"]
        - "path1" -> ["path1"]
        - "" -> []

    Args:
        paths: Paths input in various formats.

    Returns:
        Normalized list of path strings (may be empty).
    """
    if isinstance(paths, list):
        return [strip_path_annotations(p.strip()) for p in paths if p and p.strip()]

    if isinstance(paths, str):
        paths = paths.strip()
        if not paths:
            return []
        # Handle JSON array strings from MCP transport
        json_parsed = _try_parse_json_list(paths)
        if json_parsed is not None:
            return [strip_path_annotations(p) for p in json_parsed]
        return [strip_path_annotations(p.strip()) for p in paths.split() if p.strip()]

    return []


def validate_query(query: str | None, tool_name: str) -> tuple[bool, str | None]:
    """Validate a search query, return (is_valid, error_message).

    Args:
        query: The search query to validate.
        tool_name: Name of the tool (for error message context).

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    if query is None or not query.strip():
        return False, (
            f"Query cannot be empty for {tool_name}. "
            "Provide a search term like 'electron temperature' or 'equilibrium/time_slice'. "
            "Use get_dd_overview() to explore available IDS structures."
        )
    return True, None
