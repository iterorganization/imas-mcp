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

import click

from imas_mcp.core.relationships import Relationships
from imas_mcp.embeddings.config import EncoderConfig


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

    # Parse ids_filter early
    ids_set_parsed = set(ids_filter.split()) if ids_filter else None

    if ids_set_parsed:
        logger.info(
            f"Building relationships for filtered IDS: {sorted(ids_set_parsed)}"
        )
    else:
        logger.info("Building full dataset relationships file")

    try:
        logger.info("Starting relationship extraction process...")

        # Use the already parsed ids_set from above
        ids_set = ids_set_parsed

        # Create encoder config to determine the correct output file path
        encoder_config_temp = EncoderConfig(
            model_name=None,
            batch_size=250,
            normalize_embeddings=True,
            use_half_precision=False,
            enable_cache=True,
            cache_dir="embeddings",
            ids_set=ids_set,
            use_rich=False,
        )

        # Create Relationships instance - this will determine the correct filename via __post_init__
        relationships_temp = Relationships(encoder_config=encoder_config_temp)
        output_file = relationships_temp.file_path

        logger.info(f"Output file: {output_file}")

        # Check if we need to build with cache busting strategy
        should_build = force or not output_file.exists()

        # Use the unified relationships manager to check if rebuild is needed
        if not should_build and not force:
            # Reuse the temp relationships instance we already created
            relationships = relationships_temp
            if relationships.needs_rebuild():
                should_build = True
                logger.info(
                    "Cache busting: dependencies are newer than relationships file"
                )
                cache_info = relationships.get_cache_info()
                logger.debug(f"Cache status: {cache_info}")

        # The Relationships manager already handles dependency checking

        if should_build:
            if force and output_file.exists():
                logger.info("Force rebuilding existing relationships file")
            else:
                logger.info("Relationships file does not exist, building new file...")

            # Reuse the temp relationships instance (already has correct encoder_config and file path)
            relationships = relationships_temp
            config_overrides = {
                "cross_ids_eps": cross_ids_eps,
                "cross_ids_min_samples": cross_ids_min_samples,
                "intra_ids_eps": intra_ids_eps,
                "intra_ids_min_samples": intra_ids_min_samples,
                "use_rich": not quiet,
                # ids_set is already in encoder_config, no need to pass again
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
