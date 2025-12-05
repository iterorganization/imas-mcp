"""
Custom build hooks for hatchling to initialize JSON data during wheel creation.
"""

import os
import sys
from pathlib import Path
from typing import Any

# hatchling is a build system for Python projects, and this hook will be used to
# create JSON data structures for the IMAS MCP server during the wheel build process.
from hatchling.builders.hooks.plugin.interface import (
    BuildHookInterface,  # type: ignore[import]
)


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to create JSON data structures during wheel building."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Initialize the build hook and create JSON data structures.

        Args:
            version: The version string for the build
            build_data: Dictionary containing build configuration data
        """
        # Add package root to sys.path temporarily to resolve internal imports
        package_root = Path(__file__).parent
        original_path = sys.path[:]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        try:
            from imas_mcp.core.xml_parser import DataDictionaryTransformer
            from scripts.build_migrations import build_migration_map

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

        # Get configuration options
        ids_filter = self.config.get("ids-filter", "")
        dd_version = self.config.get("imas-dd-version", "")

        # Allow environment variable override for ASV builds
        ids_filter = os.environ.get("IDS_FILTER", ids_filter)
        dd_version = os.environ.get("IMAS_DD_VERSION", dd_version)

        # Transform ids_filter from space-separated or comma-separated string to set
        ids_set = None
        if ids_filter:
            # Support both space-separated and comma-separated formats
            ids_set = set(ids_filter.replace(",", " ").split())

        # Create DD accessor based on version config
        # Note: Build process needs actual DD accessor for XML parsing,
        # unlike runtime which can use lazy loading with just dd_version
        dd_accessor = None
        if dd_version:
            # Use specific version from imas-data-dictionaries PyPI package
            from imas_mcp.dd_accessor import ImasDataDictionariesAccessor

            dd_accessor = ImasDataDictionariesAccessor(dd_version)
            print(f"Building with IMAS DD version: {dd_version}")
        else:
            # Use default imas-data-dictionary accessor
            from imas_mcp.dd_accessor import ImasDataDictionaryAccessor

            dd_accessor = ImasDataDictionaryAccessor()
            print(f"Building with IMAS DD version: {dd_accessor.get_version()}")

        # Build only JSON schemas (avoiding heavy relationship extraction)
        # Transformer determines its own version-aware output path
        json_transformer = DataDictionaryTransformer(
            dd_accessor=dd_accessor, ids_set=ids_set, use_rich=True
        )
        json_transformer.build()

        # Build path migration map for version upgrades
        # This enables migration suggestions for deprecated paths
        resolved_dd_version = dd_version or str(dd_accessor.get_version())
        print(f"Building path migration map for version: {resolved_dd_version}")

        from imas_mcp.resource_path_accessor import ResourcePathAccessor

        path_accessor = ResourcePathAccessor(dd_version=resolved_dd_version)
        migrations_dir = path_accessor.migrations_dir
        migration_file = migrations_dir / "path_migrations.json"

        import json

        migration_data = build_migration_map(
            target_version=resolved_dd_version,
            ids_filter=ids_set,
            verbose=True,
        )

        with open(migration_file, "w") as f:
            json.dump(migration_data, f, indent=2)

        print(
            f"Built migration map with "
            f"{migration_data['metadata']['total_migrations']} migrations"
        )
