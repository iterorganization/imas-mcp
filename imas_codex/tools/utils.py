"""Shared utilities for MCP tool parameter handling."""


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
        - "path1" -> ["path1"]
        - "" -> []

    Args:
        paths: Paths input in various formats.

    Returns:
        Normalized list of path strings (may be empty).
    """
    if isinstance(paths, list):
        return [p.strip() for p in paths if p and p.strip()]

    if isinstance(paths, str):
        paths = paths.strip()
        if not paths:
            return []
        return [p.strip() for p in paths.split() if p.strip()]

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
            "Use get_imas_overview() to explore available IDS structures."
        )
    return True, None
