"""
Path mapping utilities for IMAS Data Dictionary version upgrades.

This module provides access to the build-time generated path map,
enabling path mapping suggestions and rename history lookups.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from imas_codex import dd_version
from imas_codex.core.exclusions import EXCLUSION_REASONS, ExclusionChecker
from imas_codex.resource_path_accessor import ResourcePathAccessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathMapping:
    """Information about a path mapping from old to new version."""

    new_path: str | None
    deprecated_in: str
    last_valid_version: str


@dataclass(frozen=True)
class RenameHistoryEntry:
    """Information about a path that was renamed to the current path."""

    old_path: str
    deprecated_in: str


class PathMap:
    """
    Provides access to path mapping data for version upgrades.

    Loads the build-time generated path map and provides lookup methods
    for both forward (old→new) and reverse (new→old) path mappings.
    """

    def __init__(
        self,
        dd_version: str = dd_version,
        mapping_data: dict | None = None,
    ):
        """
        Initialize the path map.

        Args:
            dd_version: The DD version to load mappings for.
            mapping_data: Optional pre-loaded mapping data (for testing).
        """
        self._dd_version = dd_version
        self._data: dict | None = mapping_data
        self._loaded = mapping_data is not None

    def _ensure_loaded(self) -> None:
        """Load mapping data from disk if not already loaded."""
        if self._loaded:
            return

        path_accessor = ResourcePathAccessor(dd_version=self._dd_version)
        mapping_file = path_accessor.mappings_dir / "path_mappings.json"

        if not mapping_file.exists():
            # Try to auto-build mappings
            if self._build_mappings_if_missing(mapping_file):
                # Retry loading after build
                pass
            else:
                logger.warning(
                    f"Path mappings file not found: {mapping_file}. "
                    "Run 'build-path-map' to generate it."
                )
                self._data = {
                    "old_to_new": {},
                    "new_to_old": {},
                    "metadata": {},
                    "exclusion_reasons": {},
                    "excluded_paths": {},
                }
                self._loaded = True
                return

        try:
            with open(mapping_file) as f:
                self._data = json.load(f)
            logger.debug(
                f"Loaded path map with {len(self._data.get('old_to_new', {}))} mappings"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load path mappings file: {e}")
            self._data = {
                "old_to_new": {},
                "new_to_old": {},
                "metadata": {},
                "exclusion_reasons": {},
                "excluded_paths": {},
            }

        self._loaded = True

    def _build_mappings_if_missing(self, mapping_file: Path) -> bool:
        """Attempt to auto-build path mappings if file is missing.

        Args:
            mapping_file: Path where mappings should be saved.

        Returns:
            True if mappings were built successfully, False otherwise.
        """
        try:
            from scripts.build_path_map import build_path_map

            logger.info(
                f"Path mappings not found. Building for DD {self._dd_version}..."
            )

            # Suppress imas library's verbose logging during build
            imas_logger = logging.getLogger("imas")
            imas_dd_logger = logging.getLogger("imas.dd_zip")
            original_imas_level = imas_logger.level
            original_dd_level = imas_dd_logger.level
            imas_logger.setLevel(logging.WARNING)
            imas_dd_logger.setLevel(logging.WARNING)

            try:
                # Build mappings (use_rich=None for auto-detection based on TTY)
                mapping_data = build_path_map(
                    target_version=self._dd_version,
                    verbose=False,
                    use_rich=None,  # Auto-detect: rich if TTY, logging otherwise
                )
            finally:
                # Restore original log levels
                imas_logger.setLevel(original_imas_level)
                imas_dd_logger.setLevel(original_dd_level)

            # Ensure directory exists
            mapping_file.parent.mkdir(parents=True, exist_ok=True)

            # Write mapping file
            with open(mapping_file, "w") as f:
                json.dump(mapping_data, f, indent=2)

            logger.info(
                f"✓ Built path map with "
                f"{mapping_data['metadata']['total_mappings']} mappings"
            )
            return True

        except Exception as e:
            logger.warning(f"Auto-build path mappings failed: {e}")
            return False

    def get_mapping(self, old_path: str) -> PathMapping | None:
        """
        Get mapping info for an old path.

        Args:
            old_path: The old path to look up (e.g., "equilibrium/time_slice/...").

        Returns:
            PathMapping with new_path, deprecated_in, and last_valid_version,
            or None if no mapping exists.
        """
        self._ensure_loaded()

        if self._data is None:
            return None

        entry = self._data.get("old_to_new", {}).get(old_path)
        if entry is None:
            return None

        return PathMapping(
            new_path=entry.get("new_path"),
            deprecated_in=entry.get("deprecated_in", ""),
            last_valid_version=entry.get("last_valid_version", ""),
        )

    def get_rename_history(self, new_path: str) -> list[RenameHistoryEntry]:
        """
        Get rename history for a current path.

        Args:
            new_path: The current path to look up.

        Returns:
            List of RenameHistoryEntry objects for paths that were renamed
            to this path, or empty list if no history.
        """
        self._ensure_loaded()

        if self._data is None:
            return []

        entries = self._data.get("new_to_old", {}).get(new_path, [])
        return [
            RenameHistoryEntry(
                old_path=entry.get("old_path", ""),
                deprecated_in=entry.get("deprecated_in", ""),
            )
            for entry in entries
        ]

    @property
    def metadata(self) -> dict:
        """Get path map metadata."""
        self._ensure_loaded()
        return self._data.get("metadata", {}) if self._data else {}

    @property
    def total_mappings(self) -> int:
        """Get total number of mappings in the map."""
        return self.metadata.get("total_mappings", 0)

    @property
    def target_version(self) -> str:
        """Get the target DD version for mappings."""
        return self.metadata.get("target_version", "")

    def get_exclusion_reason(self, path: str) -> str | None:
        """
        Get the exclusion reason for a path, or None if not excluded.

        First checks the pre-computed excluded_paths in the mapping data,
        then falls back to live ExclusionChecker for paths not in the map.

        Args:
            path: IMAS path to check.

        Returns:
            Exclusion reason key (e.g., "error_field", "ggd") or None.
        """
        self._ensure_loaded()

        if self._data is None:
            return None

        # First check pre-computed excluded paths (O(1) lookup)
        excluded_paths = self._data.get("excluded_paths", {})
        if path in excluded_paths:
            return excluded_paths[path]

        # Fallback to live checker for paths not in mapping map
        checker = ExclusionChecker()
        return checker.get_exclusion_reason(path)

    def get_exclusion_description(self, reason: str) -> str:
        """
        Get the human-readable description for an exclusion reason.

        Args:
            reason: Exclusion reason key (e.g., "error_field").

        Returns:
            Human-readable description or the reason key if not found.
        """
        self._ensure_loaded()

        if self._data is None:
            return EXCLUSION_REASONS.get(reason, reason)

        # Check pre-computed exclusion reasons first
        exclusion_reasons = self._data.get("exclusion_reasons", {})
        return exclusion_reasons.get(reason, EXCLUSION_REASONS.get(reason, reason))

    def is_excluded(self, path: str) -> bool:
        """Check if a path is excluded from the document store."""
        return self.get_exclusion_reason(path) is not None


@lru_cache(maxsize=1)
def get_path_map() -> PathMap:
    """
    Get the singleton PathMap instance.

    Returns:
        PathMap for the current DD version.
    """
    return PathMap()


__all__ = [
    "PathMapping",
    "RenameHistoryEntry",
    "PathMap",
    "get_path_map",
]
