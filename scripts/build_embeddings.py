#!/usr/bin/env python3
"""
Build the document store and semantic search embeddings for the IMAS Data Dictionary.

This script creates the in-memory document store from JSON data and generates
sentence transformer embeddings optimized for semantic search.
"""

import logging
import sys

import click

from imas_mcp.search.document_store import DocumentStore
from imas_mcp.search.semantic_search import SemanticSearch, SemanticSearchConfig


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force rebuild even if cache files already exist",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
@click.option(
    "--model-name",
    type=str,
    default="all-MiniLM-L6-v2",
    help="Sentence transformer model name (default: all-MiniLM-L6-v2)",
)
@click.option(
    "--batch-size",
    type=int,
    default=250,
    help="Batch size for embedding generation (default: 250)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching of embeddings",
)
@click.option(
    "--half-precision",
    is_flag=True,
    help="Use half precision (float16) to reduce memory usage",
)
@click.option(
    "--no-normalize",
    is_flag=True,
    help="Disable embedding normalization (enabled by default for faster cosine similarity)",
)
@click.option(
    "--similarity-threshold",
    type=float,
    default=0.0,
    help="Similarity threshold for search results (default: 0.0)",
)
@click.option(
    "--device",
    type=str,
    help="Device to use for model (auto-detect if not specified)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check if embeddings exist, don't build them",
)
@click.option(
    "--profile",
    is_flag=True,
    help="Enable memory and time profiling",
)
@click.option(
    "--list-caches",
    is_flag=True,
    help="List all cache files and exit",
)
@click.option(
    "--cleanup-caches",
    type=int,
    metavar="KEEP_COUNT",
    help="Remove old cache files, keeping only KEEP_COUNT most recent",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress display and use plain logging",
)
def build_embeddings(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    model_name: str,
    batch_size: int,
    no_cache: bool,
    half_precision: bool,
    no_normalize: bool,
    similarity_threshold: float,
    device: str | None,
    check_only: bool,
    profile: bool,
    list_caches: bool,
    cleanup_caches: int | None,
    no_rich: bool,
) -> int:
    """Build the document store and semantic search embeddings.

    This command creates an in-memory document store from the JSON data and
    generates sentence transformer embeddings for semantic search capabilities.

    The embeddings are cached for fast subsequent loads. Use --force to rebuild
    the cache by overwriting the existing file (safe for cancellation).

    Examples:
        build-embeddings                          # Build with default settings
        build-embeddings -v                       # Build with verbose logging
        build-embeddings -f                       # Force rebuild cache
        build-embeddings --ids-filter "core_profiles equilibrium"  # Build specific IDS
        build-embeddings --model-name "all-mpnet-base-v2"  # Use different model
        build-embeddings --half-precision         # Use float16 to reduce memory
        build-embeddings --no-cache               # Don't cache embeddings
        build-embeddings --no-normalize           # Disable embedding normalization
        build-embeddings --check-only             # Check if embeddings exist
        build-embeddings --profile                # Enable performance profiling
        build-embeddings --device cuda            # Force GPU usage
        build-embeddings --list-caches            # List all cache files
        build-embeddings --cleanup-caches 3       # Keep only 3 most recent caches
        build-embeddings --no-rich                # Disable rich progress and use plain logging
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

    try:
        logger.info("Starting document store and embeddings build process...")

        # Handle cache management operations FIRST - before any heavy initialization
        if list_caches or cleanup_caches is not None:
            if list_caches:
                cache_files = SemanticSearch.list_all_cache_files()
                if not cache_files:
                    click.echo("No cache files found")
                    return 0

                click.echo(f"Found {len(cache_files)} cache files:")
                for cache_info in cache_files:
                    click.echo(
                        f"  {cache_info['filename']}: {cache_info['size_mb']:.1f} MB, "
                        f"modified {cache_info['modified']}"
                    )
                return 0

            if cleanup_caches is not None:
                removed_count = SemanticSearch.cleanup_all_old_caches(
                    keep_count=cleanup_caches
                )
                if removed_count > 0:
                    click.echo(f"Removed {removed_count} old cache files")
                else:
                    click.echo("No old cache files to remove")
                return 0

        # Parse ids_filter string into a set if provided
        ids_set: set | None = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building embeddings for specific IDS: {sorted(ids_set)}")
        else:
            logger.info("Building embeddings for all available IDS")

        # Create semantic search configuration
        config = SemanticSearchConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            enable_cache=not no_cache,
            use_half_precision=half_precision,
            normalize_embeddings=not no_normalize,  # Default True, inverted flag
            similarity_threshold=similarity_threshold,
            ids_set=ids_set,
            auto_initialize=not (
                check_only or force
            ),  # Defer initialization for check-only or force mode
            use_rich=not no_rich,  # Use rich display unless disabled
        )

        logger.info("Configuration:")
        logger.info(f"  - Model: {config.model_name}")
        logger.info(f"  - Device: {config.device or 'auto-detect'}")
        logger.info(f"  - Batch size: {config.batch_size}")
        logger.info(f"  - Caching: {'disabled' if no_cache else 'enabled'}")
        logger.info(f"  - Half precision: {config.use_half_precision}")
        logger.info(f"  - Normalize embeddings: {config.normalize_embeddings}")
        logger.info(f"  - Similarity threshold: {config.similarity_threshold}")
        logger.info(f"  - Rich progress: {'disabled' if no_rich else 'enabled'}")

        # Build document store from JSON data
        logger.info("Building document store from JSON data...")
        if ids_set:
            logger.info(f"Creating document store with IDS filter: {list(ids_set)}")
            document_store = DocumentStore(ids_set=ids_set)
        else:
            logger.info("Creating document store with all available IDS")
            document_store = DocumentStore()

        document_count = len(document_store.get_all_documents())
        logger.info(f"Document store built with {document_count} documents")

        # Check if embeddings already exist
        semantic_search = SemanticSearch(config=config, document_store=document_store)

        # If check-only mode, just report status and exit
        if check_only:
            status_info = semantic_search.cache_status()
            status = status_info.get("status")

            if status == "valid_cache":
                click.echo(
                    f"Embeddings exist: {status_info['document_count']} documents"
                )
                click.echo(f"Model: {status_info['model_name']}")
                click.echo(f"Cache size: {status_info['cache_file_size_mb']:.1f} MB")
                click.echo(f"Created: {status_info['created_at']}")
                return 0
            elif status in ["no_cache_file", "invalid_cache", "invalid_cache_file"]:
                if status == "no_cache_file":
                    click.echo("Embeddings do not exist: No cache file found")
                elif status == "invalid_cache_file":
                    click.echo(
                        "Embeddings are invalid: Cache file is corrupted or incompatible"
                    )
                elif status == "invalid_cache":
                    reason = status_info.get("reason", "Unknown reason")
                    click.echo(f"Embeddings are invalid: {reason}")
                return 1
            elif status == "cache_disabled":
                click.echo("Embeddings cache is disabled")
                return 1
            else:
                click.echo(f"Cache status: {status}")
                if "error" in status_info:
                    click.echo(f"Error: {status_info['error']}")
                return 1

        # If force rebuild, regenerate embeddings by overwriting cache
        if force:
            logger.info("Force rebuild requested, regenerating embeddings...")
            # Ensure embeddings are initialized before rebuilding
            if not semantic_search._initialized:
                semantic_search._initialize()
            semantic_search.rebuild_embeddings()
        else:
            # Normal initialization (will use cache if valid)
            if not semantic_search._initialized:
                semantic_search._initialize()

        # Get embeddings info
        info = semantic_search.get_embeddings_info()

        if info.get("status") == "not_initialized":
            logger.error("Failed to initialize embeddings")
            return 1

        # Log success information
        logger.info("Embeddings built successfully:")
        logger.info(f"  - Model: {info['model_name']}")
        logger.info(f"  - Documents: {info['document_count']}")
        logger.info(f"  - Dimensions: {info['embedding_dimension']}")
        logger.info(f"  - Data type: {info['dtype']}")
        logger.info(f"  - Memory usage: {info['memory_usage_mb']:.1f} MB")

        if "cache_file_path" in info:
            logger.info(f"  - Cache file: {info['cache_file_path']}")
            logger.info(f"  - Cache size: {info['cache_file_size_mb']:.1f} MB")
        elif config.enable_cache:
            logger.warning("  - Cache file not found (embeddings may not be cached)")

        # Print summary for scripts/CI
        click.echo(
            f"Built embeddings for {document_count} documents using {model_name}"
        )
        if "cache_file_size_mb" in info:
            click.echo(f"Cache size: {info['cache_file_size_mb']:.1f} MB")

        return 0

    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_embeddings())
