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
            # from imas_mcp.structure.mermaid_generator import MermaidGraphGenerator

        finally:
            # Restore original sys.path
            sys.path[:] = original_path

        # Get configuration options
        ids_filter = self.config.get("ids-filter", "")

        # Allow environment variable override for ASV builds
        ids_filter = os.environ.get("IDS_FILTER", ids_filter)

        # Transform ids_filter from space-separated or comma-separated string to set
        ids_set = None
        if ids_filter:
            # Support both space-separated and comma-separated formats
            ids_set = set(ids_filter.replace(",", " ").split())

        # Build only JSON schemas (avoiding heavy relationship extraction)
        json_transformer = DataDictionaryTransformer(ids_set=ids_set, use_rich=True)
        json_transformer.build()

        # Skip Mermaid graph generation (requires relationships)
        # resources_dir = package_root / "imas_mcp" / "resources"
        # mermaid_generator = MermaidGraphGenerator(resources_dir)
        # mermaid_generator.build(ids_set)
