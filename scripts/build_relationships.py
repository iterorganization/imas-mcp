#!/usr/bin/env python3
"""
Build relationships between IMAS data paths using optimized clustering.
This script takes the detailed JSON schemas as input and generates relationships.

OPTIMAL CLUSTERING PARAMETERS (Latin Hypercube Optimization):
- cross_ids_eps = 0.0751 (cross-IDS clustering epsilon)
- cross_ids_min_samples = 2 (cross-IDS minimum samples)
- intra_ids_eps = 0.0319 (intra-IDS clustering epsilon)
- intra_ids_min_samples = 2 (intra-IDS minimum samples)

Optimization achieved 79% improvement over initial parameters (Score: 5436.17)
"""

import logging
import sys
from pathlib import Path

import click

from imas_mcp.core.relationships import Relationships


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--cross-ids-eps",
    type=float,
    default=0.0751,
    help="Epsilon parameter for cross-IDS DBSCAN clustering (default: 0.0751, optimized via LHC)",
)
@click.option(
    "--cross-ids-min-samples",
    type=int,
    default=2,
    help="Minimum samples for cross-IDS DBSCAN clustering (default: 2)",
)
@click.option(
    "--intra-ids-eps",
    type=float,
    default=0.0319,
    help="Epsilon parameter for intra-IDS DBSCAN clustering (default: 0.0319, optimized via LHC)",
)
@click.option(
    "--intra-ids-min-samples",
    type=int,
    default=2,
    help="Minimum samples for intra-IDS DBSCAN clustering (default: 2, optimized)",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
def build_relationships(
    verbose: bool,
    quiet: bool,
    force: bool,
    cross_ids_eps: float,
    cross_ids_min_samples: int,
    intra_ids_eps: float,
    intra_ids_min_samples: int,
    ids_filter: str,
) -> int:
    """Build relationships between IMAS data paths using multi-membership DBSCAN clustering.

    This command reads detailed IDS JSON files and generates semantic relationships
    between data paths using embedding-based clustering. It performs separate clustering
    for cross-IDS relationships (paths that span multiple IDS) and intra-IDS
    relationships (paths within the same IDS).

    Examples:
        build-relationships                              # Build with default settings
        build-relationships -v                           # Build with verbose logging
        build-relationships -f                           # Force rebuild even if exists
        build-relationships --ids-filter "core_profiles equilibrium"  # Build specific IDS only
        build-relationships --cross-ids-eps 0.0751 --intra-ids-eps 0.0319  # Optimized clustering parameters
        build-relationships --cross-ids-min-samples 2   # Custom minimum samples for cross-IDS
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
    output_file = Path("imas_mcp/resources/schemas/relationships.json")
    embeddings_dir = Path("imas_mcp/resources/embeddings")
    schemas_dir = Path("imas_mcp/resources/schemas/detailed")

    try:
        logger.info("Starting relationship extraction process...")

        # Parse ids_filter string into a set if provided
        ids_set: set | None = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building relationships for specific IDS: {sorted(ids_set)}")
        else:
            logger.info("Building relationships for all available IDS")

        # Check if we need to build with cache busting strategy
        should_build = force or not output_file.exists()

        # Use the unified relationships manager to check if rebuild is needed
        if not should_build and not force:
            relationships = Relationships(output_file)
            if relationships.needs_rebuild():
                should_build = True
                logger.info(
                    "Cache busting: dependencies are newer than relationships file"
                )
                cache_info = relationships.get_cache_info()
                logger.debug(f"Cache status: {cache_info}")

        if not should_build and output_file.exists():
            # Check if any source files are newer than the relationships file
            relationships_mtime = output_file.stat().st_mtime

            # Check if embeddings cache files are newer
            newer_embeddings = False
            if embeddings_dir.exists():
                for embedding_file in embeddings_dir.glob("*.pkl"):
                    if embedding_file.stat().st_mtime > relationships_mtime:
                        newer_embeddings = True
                        logger.info(
                            f"Found newer embedding file: {embedding_file.name}"
                        )
                        break

            # Check if schema files are newer
            newer_schemas = False
            if schemas_dir.exists():
                for schema_file in schemas_dir.glob("*.json"):
                    if schema_file.stat().st_mtime > relationships_mtime:
                        newer_schemas = True
                        logger.info(f"Found newer schema file: {schema_file.name}")
                        break

            # Force rebuild if dependencies are newer
            if newer_embeddings or newer_schemas:
                should_build = True
                if newer_embeddings and newer_schemas:
                    logger.info(
                        "Cache busting: embeddings and schema files are newer than relationships file"
                    )
                elif newer_embeddings:
                    logger.info(
                        "Cache busting: embedding files are newer than relationships file"
                    )
                else:
                    logger.info(
                        "Cache busting: schema files are newer than relationships file"
                    )

        if should_build:
            if force and output_file.exists():
                logger.info("Force rebuilding existing relationships file")
            else:
                logger.info("Relationships file does not exist, building new file...")

            # Use the unified relationships manager to build
            relationships = Relationships(output_file)
            config_overrides = {
                "cross_ids_eps": cross_ids_eps,
                "cross_ids_min_samples": cross_ids_min_samples,
                "intra_ids_eps": intra_ids_eps,
                "intra_ids_min_samples": intra_ids_min_samples,
                "use_rich": not quiet,
                "ids_set": ids_set,
            }

            relationships.build(force=force, **config_overrides)

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
