#!/usr/bin/env python3
"""
Build the lexicographic search index for the IMAS Data Dictionary.
This script builds the index and prints the index name that was created.
"""

import logging
import sys
from typing import Optional

import click
from imas_mcp.lexicographic_search import LexicographicSearch


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if index already exists"
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include in the index as a space-separated string (e.g., 'core_profiles equilibrium')",
)
def build_index(verbose: bool, quiet: bool, force: bool, ids_filter: str) -> int:
    """Build the lexicographic search index for the IMAS Data Dictionary.

    This command initializes a LexicographicSearch instance and builds the index
    if it doesn't exist. The index name is printed to stdout for use in CI/CD scripts.

    Examples:
        build-index                    # Build index with default settings
        build-index -v                 # Build with verbose logging
        build-index -f                 # Force rebuild even if exists
        build-index --ids-filter "core_profiles equilibrium"  # Build specific IDS only"""
    # Set up logging level
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    try:
        logger.info(
            "Starting index build process..."
        )  # Parse ids_filter string into a set if provided
        ids_set: Optional[set] = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building index for specific IDS: {sorted(ids_set)}")

        # Initialize the search class
        search = LexicographicSearch(ids_set=ids_set, auto_build=False)

        # Check if we need to build
        should_build = force or len(search) == 0

        if should_build:
            if force and len(search) > 0:
                logger.info(
                    f"Force rebuilding existing index with {len(search)} documents"
                )
            else:
                logger.info("Index does not exist, building new index...")

            search.build_index()
            logger.info(f"Index built successfully: {search.indexname}")
        else:
            index_count = len(search)
            logger.info(
                f"Index already exists with {index_count} documents: {search.indexname}"
            )

        # Print the index name for use in scripts/CI
        click.echo(search.indexname)
        return 0

    except Exception as e:
        logger.error(f"Error building index: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_index())
