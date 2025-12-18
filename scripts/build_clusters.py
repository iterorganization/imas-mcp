#!/usr/bin/env python3
"""
Build semantic clusters of IMAS data paths using HDBSCAN clustering.

This script uses a semantic-first approach with HDBSCAN, which automatically
adapts to varying density in the embedding space without requiring epsilon tuning.
Cross-IDS vs intra-IDS cluster properties are derived post-hoc from results.

HDBSCAN Parameters:
- min_cluster_size = 2 (allows pairs of related paths)
- min_samples = 2 (core sample threshold)
- cluster_selection_method = 'eom' or 'leaf' ('eom' for broader, 'leaf' for finer)
"""

import logging
import sys
from typing import Literal

import click
from dotenv import load_dotenv

from imas_codex.clusters.label_cache import LabelCache
from imas_codex.core.clusters import Clusters
from imas_codex.embeddings.config import EncoderConfig

# Load environment variables from .env file, overriding any existing values
load_dotenv(override=True)


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    help="Minimum cluster size for HDBSCAN (default: 2, allows pairs)",
)
@click.option(
    "--min-samples",
    type=int,
    default=2,
    help="Minimum samples for HDBSCAN core points (default: 2)",
)
@click.option(
    "--cluster-method",
    type=click.Choice(["eom", "leaf"]),
    default="eom",
    help="HDBSCAN cluster selection method: 'eom' for broader clusters, 'leaf' for finer (default: eom)",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
def build_clusters(
    verbose: bool,
    quiet: bool,
    force: bool,
    min_cluster_size: int,
    min_samples: int,
    cluster_method: Literal["eom", "leaf"],
    ids_filter: str,
) -> int:
    """Build semantic clusters of IMAS data paths using HDBSCAN clustering.

    This command reads detailed IDS JSON files and generates semantic clusters
    using embedding-based HDBSCAN clustering. The algorithm automatically adapts
    to varying density without epsilon tuning. Cross-IDS vs intra-IDS properties
    are derived from cluster membership after clustering completes.

    Examples:
        build-clusters                              # Build with default settings
        build-clusters -v                           # Build with verbose logging
        build-clusters -f                           # Force rebuild even if exists
        build-clusters --ids-filter "core_profiles equilibrium"  # Build specific IDS only
        build-clusters --cluster-method leaf        # Use finer-grained clustering
        build-clusters --min-cluster-size 3         # Require at least 3 paths per cluster
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
        logger.info(f"Building clusters for filtered IDS: {sorted(ids_set_parsed)}")
    else:
        logger.info("Building full dataset clusters file")

    try:
        logger.info("Starting cluster extraction process...")

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

        # Create Clusters instance - this will determine the correct filename via __post_init__
        clusters_manager = Clusters(encoder_config=encoder_config_temp)
        output_file = clusters_manager.file_path

        logger.info(f"Output file: {output_file}")

        # Check if we need to build with cache busting strategy
        should_build = force or not output_file.exists()

        # Use the unified clusters manager to check if rebuild is needed
        if not should_build and not force:
            if clusters_manager.needs_rebuild():
                should_build = True
                logger.info("Cache busting: dependencies are newer than clusters file")
                cache_info = clusters_manager.get_cache_info()
                logger.debug(f"Cache status: {cache_info}")

        if should_build:
            if force and output_file.exists():
                logger.info("Force rebuilding existing clusters file")
            else:
                logger.info("Clusters file does not exist, building new file...")

            config_overrides = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_method,
                "use_rich": not quiet,
            }

            clusters_manager.build(force=force, **config_overrides)

            logger.info("Clusters built successfully")
            click.echo(f"Built clusters file: {output_file}")

            # Export labels to definitions for version control
            label_cache = LabelCache()
            exported = label_cache.export_labels()
            click.echo(f"Exported {len(exported)} labels to definitions")

        else:
            logger.info("Clusters file already exists at %s", output_file)
            click.echo(f"Clusters already exist at {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Error building clusters: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_clusters())
