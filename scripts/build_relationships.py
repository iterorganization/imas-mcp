#!/usr/bin/env python3
"""
Build relationships between IMAS data paths using optimized clustering.
This script takes the detailed JSON schemas as input and generates relationships.
"""

import logging
import sys
from pathlib import Path

import click

from imas_mcp.relationships import (
    RelationshipExtractionConfig,
    RelationshipExtractor,
)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.7,
    help="Similarity threshold for clustering (default: 0.7)",
)
@click.option(
    "--min-samples",
    type=int,
    default=3,
    help="Minimum samples for DBSCAN clustering (default: 3)",
)
@click.option(
    "--eps",
    type=float,
    default=0.25,
    help="Epsilon parameter for DBSCAN clustering (default: 0.25)",
)
def build_relationships(
    verbose: bool,
    quiet: bool,
    force: bool,
    similarity_threshold: float,
    min_samples: int,
    eps: float,
) -> int:
    """Build relationships between IMAS data paths using DBSCAN clustering.

    This command reads detailed IDS JSON files and generates semantic relationships
    between data paths using embedding-based clustering in the semantic space.

    Examples:
        build-relationships                              # Build with default settings
        build-relationships -v                           # Build with verbose logging
        build-relationships -f                           # Force rebuild even if exists
        build-relationships --eps 0.3 --min-samples 5   # Custom clustering parameters
        build-relationships --similarity-threshold 0.8  # Higher similarity threshold
    """
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

    # Hardcoded paths
    input_dir = Path("imas_mcp/resources/schemas/detailed")
    output_file = Path("imas_mcp/resources/schemas/relationships.json")

    try:
        logger.info("Starting relationship extraction process...")

        # Check if we need to build
        should_build = force or not output_file.exists()

        if should_build:
            if force and output_file.exists():
                logger.info("Force rebuilding existing relationships file")
            else:
                logger.info("Relationships file does not exist, building new file...")

            # Create configuration
            config = RelationshipExtractionConfig(
                input_dir=input_dir,
                output_file=output_file,
                eps=eps,
                min_samples=min_samples,
                similarity_threshold=similarity_threshold,
                use_rich=not quiet,
            )

            # Build relationships using the extractor
            extractor = RelationshipExtractor(config)
            relationships = extractor.extract_relationships(force_rebuild=force)

            # Save relationships
            extractor.save_relationships(relationships)

            logger.info("Relationships built successfully")
            click.echo(f"Built relationships file: {output_file}")

        else:
            logger.info("Relationships file already exists at %s", output_file)
            click.echo(f"Relationships already exist at {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Error building relationships: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_relationships())
