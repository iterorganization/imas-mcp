"""
Shared exclusion logic for IMAS path filtering.

This module provides a centralized ExclusionChecker that determines which paths
are excluded from the document store. Used by both xml_parser.py (build-time)
and migrations (runtime lookup).
"""

from dataclasses import dataclass, field

from imas_mcp.settings import get_include_error_fields, get_include_ggd

# Exclusion reason identifiers and their descriptions
EXCLUSION_REASONS: dict[str, str] = {
    "error_field": "Uncertainty bound fields (_error_upper, _error_lower, _error_index)",
    "ggd": "Grid Geometry Description nodes",
    "metadata": "Internal metadata fields (ids_properties, code)",
}


@dataclass
class ExclusionChecker:
    """
    Checks if IMAS paths should be excluded from the document store.

    Provides consistent exclusion logic for use during both build-time
    (XML parsing, migration generation) and runtime (path lookup).

    Args:
        include_ggd: Whether to include GGD paths. Default from settings.
        include_error_fields: Whether to include error fields. Default from settings.
        excluded_patterns: Patterns always excluded (metadata fields).
    """

    include_ggd: bool = field(default_factory=get_include_ggd)
    include_error_fields: bool = field(default_factory=get_include_error_fields)
    excluded_patterns: set[str] = field(
        default_factory=lambda: {"ids_properties", "code"}
    )

    def get_exclusion_reason(self, path: str) -> str | None:
        """
        Get the exclusion reason for a path, or None if not excluded.

        Args:
            path: Full IMAS path (e.g., "equilibrium/time_slice/boundary/psi_error_lower")

        Returns:
            Exclusion reason key (e.g., "error_field", "ggd") or None if path is indexed.
        """
        if not path:
            return None

        # Extract the final component name
        name = path.split("/")[-1] if "/" in path else path
        path_lower = path.lower()
        name_lower = name.lower()

        # Check metadata/excluded patterns first
        for pattern in self.excluded_patterns:
            if pattern in name or pattern in path:
                return "metadata"

        # Check GGD patterns (exclude if not included)
        if not self.include_ggd and self._is_ggd_path(path_lower, name_lower):
            return "ggd"

        # Check error field patterns (exclude if not included)
        if not self.include_error_fields and self._is_error_field(name):
            return "error_field"

        return None

    def is_excluded(self, path: str) -> bool:
        """Check if a path is excluded from the document store."""
        return self.get_exclusion_reason(path) is not None

    def _is_ggd_path(self, path_lower: str, name_lower: str) -> bool:
        """Check if path matches GGD (Grid Geometry Description) patterns."""
        return (
            "ggd" in name_lower
            or "/ggd/" in path_lower
            or "grids_ggd" in path_lower
            or path_lower.startswith("grids_ggd")
            or "/grids_ggd/" in path_lower
        )

    def _is_error_field(self, name: str) -> bool:
        """Check if name matches error field patterns."""
        return (
            "_error_" in name
            or name.endswith("_error_upper")
            or name.endswith("_error_lower")
            or name.endswith("_error_index")
            or "error_upper" in name
            or "error_lower" in name
            or "error_index" in name
        )


# Default singleton instance with standard exclusion settings
_default_checker: ExclusionChecker | None = None


def get_exclusion_checker() -> ExclusionChecker:
    """Get the default ExclusionChecker singleton with settings from pyproject.toml."""
    global _default_checker
    if _default_checker is None:
        _default_checker = ExclusionChecker()
    return _default_checker


__all__ = [
    "EXCLUSION_REASONS",
    "ExclusionChecker",
    "get_exclusion_checker",
]
