"""
Custom build hooks for hatchling to initialize JSON data during wheel creation.

This hook is called during `uv sync` and wheel building. It skips regeneration
if the required resource files already exist, making incremental syncs fast.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any

# hatchling is a build system for Python projects, and this hook will be used to
# create JSON data structures for the IMAS Codex server during the wheel build process.
from hatchling.builders.hooks.plugin.interface import (
    BuildHookInterface,  # type: ignore[import]
)


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to create JSON data structures during wheel building."""

    PLUGIN_NAME = "imas-build-hook"

    def _trace(self, message: str) -> None:
        """Print trace message for debugging build hook execution."""
        print(f"[BUILD HOOK] {message}", flush=True)

    def _check_physics_domain_exists(self) -> bool:
        """Check if physics domain enum file exists and is up to date."""
        package_root = Path(__file__).parent
        generated_file = package_root / "imas_codex" / "core" / "physics_domain.py"
        schema_file = (
            package_root / "imas_codex" / "definitions" / "physics" / "domains.yaml"
        )

        if not generated_file.exists():
            return False

        if not schema_file.exists():
            return True  # No schema, nothing to generate

        # Check if schema is newer than generated file
        return generated_file.stat().st_mtime >= schema_file.stat().st_mtime

    def _generate_physics_domain(self, package_root: Path) -> None:
        """Generate physics domain enum from LinkML schema."""
        schema_file = (
            package_root / "imas_codex" / "definitions" / "physics" / "domains.yaml"
        )
        output_file = package_root / "imas_codex" / "core" / "physics_domain.py"

        if not schema_file.exists():
            self._trace(f"Schema file not found: {schema_file}")
            return

        # Import generator function
        original_path = sys.path[:]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        try:
            from scripts.gen_physics_domains import generate_enum_code

            code = generate_enum_code(schema_file)
            output_file.write_text(code)
            self._trace(f"Generated {output_file}")
        except Exception as e:
            self._trace(f"Failed to generate physics domain: {e}")
        finally:
            sys.path[:] = original_path

    def _check_schemas_exist(self, schemas_dir: Path) -> bool:
        """Check if schema files already exist and are valid."""
        catalog_path = schemas_dir / "ids_catalog.json"
        detailed_dir = schemas_dir / "detailed"
        exists = (
            catalog_path.exists()
            and detailed_dir.exists()
            and any(detailed_dir.glob("*.json"))
        )
        return exists

    def _check_path_map_exists(self, mappings_dir: Path) -> bool:
        """Check if path map file already exists."""
        mapping_file = mappings_dir / "path_mappings.json"
        return mapping_file.exists()

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """
        Initialize the build hook and create JSON data structures.

        Skips regeneration if files already exist to keep uv sync fast.

        Args:
            version: The version string for the build
            build_data: Dictionary containing build configuration data
        """
        start_time = time.time()
        self._trace(f"initialize() called with version={version}")

        # Add package root to sys.path temporarily to resolve internal imports
        package_root = Path(__file__).parent
        original_path = sys.path[:]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        try:
            # Lightweight import for path resolution
            from imas_codex.resource_path_accessor import ResourcePathAccessor
        finally:
            sys.path[:] = original_path

        # Get configuration options
        ids_filter = self.config.get("ids-filter", "")
        dd_version_config = self.config.get("imas-dd-version", "")

        # Allow environment variable override for ASV builds
        ids_filter = os.environ.get("IDS_FILTER", ids_filter)
        dd_version_config = os.environ.get("IMAS_DD_VERSION", dd_version_config)

        # Transform ids_filter from space-separated or comma-separated string to set
        ids_set = None
        if ids_filter:
            ids_set = set(ids_filter.replace(",", " ").split())

        # Determine DD version without loading heavy accessor
        if dd_version_config:
            resolved_dd_version = dd_version_config
        else:
            # Get version from default package
            if str(package_root) not in sys.path:
                sys.path.insert(0, str(package_root))
            try:
                from imas_codex import dd_version

                resolved_dd_version = dd_version
            finally:
                sys.path[:] = original_path

        self._trace(f"Using DD version: {resolved_dd_version}")

        # Check if physics domain enum needs generation
        physics_domain_exists = self._check_physics_domain_exists()
        self._trace(f"physics_domain_exists={physics_domain_exists}")

        if not physics_domain_exists:
            self._trace("Generating physics domain enum from LinkML schema...")
            self._generate_physics_domain(package_root)

        # Get resource paths for this version
        path_accessor = ResourcePathAccessor(dd_version=resolved_dd_version)
        schemas_dir = path_accessor.schemas_dir
        mappings_dir = path_accessor.mappings_dir

        # Check if all required files already exist
        schemas_exist = self._check_schemas_exist(schemas_dir)
        path_map_exists = self._check_path_map_exists(mappings_dir)

        self._trace(f"schemas_exist={schemas_exist}, path_map_exists={path_map_exists}")

        if schemas_exist and path_map_exists:
            elapsed = time.time() - start_time
            self._trace(f"Resources exist, skipping build. Total time: {elapsed:.2f}s")
            return

        # Need to build - import heavy modules now
        self._trace("Resources missing, starting build...")
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        try:
            from imas_codex.core.xml_parser import DataDictionaryTransformer
            from scripts.build_path_map import build_path_map
        finally:
            sys.path[:] = original_path

        # Create DD accessor based on version config
        dd_accessor = None
        if dd_version_config:
            from imas_codex.dd_accessor import ImasDataDictionariesAccessor

            dd_accessor = ImasDataDictionariesAccessor(dd_version_config)
            self._trace(f"Building with IMAS DD version: {dd_version_config}")
        else:
            from imas_codex.dd_accessor import ImasDataDictionaryAccessor

            dd_accessor = ImasDataDictionaryAccessor()
            self._trace(f"Building with IMAS DD version: {dd_accessor.get_version()}")

        # Build schemas only if they don't exist
        # IMPORTANT: Always build ALL schemas (ids_set=None), not a filtered subset.
        # The ids_set is for runtime filtering, not build-time filtering.
        if not schemas_exist:
            self._trace("Building schemas...")
            json_transformer = DataDictionaryTransformer(
                dd_accessor=dd_accessor, ids_set=None, use_rich=True
            )
            json_transformer.build()
            self._trace("Schemas built")
        else:
            self._trace("Schemas already exist, skipping")

        # Build path map only if it doesn't exist
        if not path_map_exists:
            import json

            self._trace("Building path map...")
            mapping_file = mappings_dir / "path_mappings.json"
            mapping_data = build_path_map(
                target_version=resolved_dd_version,
                ids_filter=ids_set,
                verbose=True,
            )
            with open(mapping_file, "w") as f:
                json.dump(mapping_data, f, indent=2)
            self._trace(
                f"Built path map with {mapping_data['metadata']['total_mappings']} mappings"
            )
        else:
            self._trace("Path map already exists, skipping")

        elapsed = time.time() - start_time
        self._trace(f"Build complete. Total time: {elapsed:.2f}s")
