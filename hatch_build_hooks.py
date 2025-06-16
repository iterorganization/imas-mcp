"""
Custom build hooks for hatchling to initialize lexicographic index during wheel creation.
"""

import logging
from typing import Any, Dict

# Note: hatchling import may not resolve in development environment
# but will work correctly when hatchling loads the plugin
from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore

from imas_mcp_server.lexicographic_search import LexicographicSearch


class CustomBuildHook(BuildHookInterface):
    """Custom build hook to create lexicographic index during wheel building."""

    def initialize(self, version: str, build_data: Dict[str, Any]) -> None:
        """Initialize the build hook and initialize the lexicographic index."""
        # Configure logging to ensure progress messages are visible during build
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,  # Override any existing configuration
        )

        logger = logging.getLogger(__name__)
        logger.info("Initializing lexicographic index build hook")

        # Get configuration options
        verbose = self.config.get("verbose", False)
        ids_filter = self.config.get("ids-filter", "")  # Default builds full Dictionary

        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        logger.info(
            "Initializing lexicographic index as part of wheel creation"
        )  # Transform ids_filter from space-separated string to set
        ids_set = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.info(f"Using IDS filter: {ids_filter}")
        else:
            logger.info("Building index for all available IDS (no filter specified)")

        # Initialize the index (this will create the index structure if needed)
        logger.info("Starting lexicographic index creation...")
        index = LexicographicSearch(ids_set=ids_set)
        logger.info(
            f"Lexicographic index created successfully with {len(index)} documents"
        )
        logger.info("Build hook initialization completed")
