"""
Path migration utilities for IMAS Data Dictionary version upgrades.

This module provides access to the build-time generated migration map,
enabling path migration suggestions and rename history lookups.
"""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache

from imas_mcp import dd_version
from imas_mcp.resource_path_accessor import ResourcePathAccessor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationEntry:
    """Information about a path migration from old to new version."""

    new_path: str | None
    deprecated_in: str
    last_valid_version: str


@dataclass(frozen=True)
class RenameHistoryEntry:
    """Information about a path that was renamed to the current path."""

    old_path: str
    deprecated_in: str


class PathMigrationMap:
    """
    Provides access to path migration data for version upgrades.

    Loads the build-time generated migration map and provides lookup methods
    for both forward (old→new) and reverse (new→old) path mappings.
    """

    def __init__(
        self,
        dd_version: str = dd_version,
        migration_data: dict | None = None,
    ):
        """
        Initialize the migration map.

        Args:
            dd_version: The DD version to load migrations for.
            migration_data: Optional pre-loaded migration data (for testing).
        """
        self._dd_version = dd_version
        self._data: dict | None = migration_data
        self._loaded = migration_data is not None

    def _ensure_loaded(self) -> None:
        """Load migration data from disk if not already loaded."""
        if self._loaded:
            return

        path_accessor = ResourcePathAccessor(dd_version=self._dd_version)
        migration_file = path_accessor.migrations_dir / "path_migrations.json"

        if not migration_file.exists():
            logger.warning(
                f"Migration file not found: {migration_file}. "
                "Run 'build-migrations' to generate it."
            )
            self._data = {"old_to_new": {}, "new_to_old": {}, "metadata": {}}
            self._loaded = True
            return

        try:
            with open(migration_file) as f:
                self._data = json.load(f)
            logger.debug(
                f"Loaded migration map with "
                f"{len(self._data.get('old_to_new', {}))} migrations"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load migration file: {e}")
            self._data = {"old_to_new": {}, "new_to_old": {}, "metadata": {}}

        self._loaded = True

    def get_migration(self, old_path: str) -> MigrationEntry | None:
        """
        Get migration info for an old path.

        Args:
            old_path: The old path to look up (e.g., "equilibrium/time_slice/...").

        Returns:
            MigrationEntry with new_path, deprecated_in, and last_valid_version,
            or None if no migration exists.
        """
        self._ensure_loaded()

        if self._data is None:
            return None

        entry = self._data.get("old_to_new", {}).get(old_path)
        if entry is None:
            return None

        return MigrationEntry(
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
        """Get migration map metadata."""
        self._ensure_loaded()
        return self._data.get("metadata", {}) if self._data else {}

    @property
    def total_migrations(self) -> int:
        """Get total number of migrations in the map."""
        return self.metadata.get("total_migrations", 0)

    @property
    def target_version(self) -> str:
        """Get the target DD version for migrations."""
        return self.metadata.get("target_version", "")


@lru_cache(maxsize=1)
def get_migration_map() -> PathMigrationMap:
    """
    Get the singleton PathMigrationMap instance.

    Returns:
        PathMigrationMap for the current DD version.
    """
    return PathMigrationMap()


__all__ = [
    "MigrationEntry",
    "RenameHistoryEntry",
    "PathMigrationMap",
    "get_migration_map",
]
