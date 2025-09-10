#!/usr/bin/env python3
"""
Build the document store and embeddings for the IMAS Data Dictionary.

This script creates the in-memory document store from JSON data and generates
sentence transformer embeddings using the core embedding management system.
"""

import logging
import sys
from pathlib import Path

import click

from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder
from imas_mcp.search.document_store import DocumentStore


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
    """Build the document store and embeddings.

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

        # Parse ids_filter string into a set if provided
        ids_set: set | None = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building embeddings for specific IDS: {sorted(ids_set)}")
        else:
            logger.info("Building embeddings for all available IDS")

        # Create embedding configuration
        config = EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            enable_cache=not no_cache,
            use_half_precision=half_precision,
            normalize_embeddings=not no_normalize,  # Default True, inverted flag
            ids_set=ids_set,
            use_rich=not no_rich,  # Use rich display unless disabled
        )
        # Create encoder directly (no global registry)
        encoder = Encoder(config)

        # Handle cache management operations FIRST - before any heavy initialization
        if list_caches or cleanup_caches is not None:
            if list_caches:
                cache_files = encoder.list_cache_files()
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
                removed_count = encoder.cleanup_old_caches(keep_count=cleanup_caches)
                if removed_count > 0:
                    click.echo(f"Removed {removed_count} old cache files")
                else:
                    click.echo("No old cache files to remove")
                return 0

        logger.info("Configuration:")
        logger.info(f"  - Model: {config.model_name}")
        logger.info(f"  - Device: {config.device or 'auto-detect'}")
        logger.info(f"  - Batch size: {config.batch_size}")
        logger.info(f"  - Caching: {'disabled' if no_cache else 'enabled'}")
        logger.info(f"  - Half precision: {config.use_half_precision}")
        logger.info(f"  - Normalize embeddings: {config.normalize_embeddings}")
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

        # Get all documents and prepare texts for embedding
        documents = document_store.get_all_documents()
        texts = [doc.embedding_text for doc in documents]
        identifiers = [doc.metadata.path_id for doc in documents]

        # Generate cache key using centralized config method
        cache_key = config.generate_cache_key()

        # Get source data directory for validation
        source_data_dir = None
        try:
            import importlib.resources as resources

            resources_dir = Path(str(resources.files("imas_mcp") / "resources"))
            source_data_dir = resources_dir / "schemas"
        except Exception:
            pass

        # If check-only mode, just report status and exit
        if check_only:
            # Try to check if embeddings already exist using the same manager and cache key
            try:
                # Set the cache path for the manager
                encoder._set_cache_path(cache_key)  # noqa: SLF001 (intentional internal use)

                # Try to load existing cache
                if encoder._try_load_cache(texts, identifiers, source_data_dir):  # noqa: SLF001
                    # Cache was successfully loaded
                    cache_info = encoder.get_cache_info()
                    click.echo(
                        f"Embeddings exist: {cache_info.get('document_count', 0)} documents"
                    )
                    click.echo(f"Model: {cache_info['model_name']}")
                    if "cache_file_size_mb" in cache_info:
                        click.echo(
                            f"Cache size: {cache_info['cache_file_size_mb']:.1f} MB"
                        )
                    if "created_at" in cache_info:
                        click.echo(f"Created: {cache_info['created_at']}")
                    return 0
                else:
                    # Check if cache file exists but is invalid
                    cache_path = encoder._cache_path  # noqa: SLF001
                    if cache_path and cache_path.exists():
                        click.echo(
                            "Embeddings exist but are invalid: Cache validation failed"
                        )
                        return 1
                    else:
                        click.echo("Embeddings do not exist: No cache file found")
                        return 1

            except Exception as e:
                if verbose:
                    logger.exception("Error checking cache:")
                click.echo(f"Error checking cache: {e}")
                return 1

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} documents...")

        try:
            embeddings, result_identifiers, was_cached = (
                encoder.build_document_embeddings(
                    texts=texts,
                    identifiers=identifiers,
                    cache_key=cache_key,
                    force_rebuild=force,
                    source_data_dir=source_data_dir,
                )
            )

            # Get embeddings info
            info = encoder.get_cache_info()

            # Log success information
            cache_status = "loaded from cache" if was_cached else "built successfully"
            logger.info(f"Embeddings {cache_status}:")
            logger.info(f"  - Model: {info.get('model_name', 'unknown')}")
            logger.info(f"  - Documents: {info.get('document_count', len(embeddings))}")
            logger.info(
                f"  - Dimensions: {info.get('embedding_dimension', embeddings.shape[1] if len(embeddings.shape) > 1 else 0)}"
            )
            logger.info(f"  - Data type: {info.get('dtype', str(embeddings.dtype))}")
            logger.info(
                f"  - Memory usage: {info.get('memory_usage_mb', embeddings.nbytes / (1024 * 1024)):.1f} MB"
            )

            if "cache_file_path" in info:
                logger.info(f"  - Cache file: {info['cache_file_path']}")
                logger.info(f"  - Cache size: {info['cache_file_size_mb']:.1f} MB")
            elif config.enable_cache:
                logger.warning(
                    "  - Cache file not found (embeddings may not be cached)"
                )

            # Print summary for scripts/CI with accurate messaging
            action = "Loaded" if was_cached else "Built"
            click.echo(
                f"{action} embeddings for {document_count} documents using {model_name}"
            )
            if "cache_file_size_mb" in info:
                click.echo(f"Cache size: {info['cache_file_size_mb']:.1f} MB")

            return 0

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            if verbose:
                logger.exception("Full traceback:")
            return 1

    except Exception as e:
        logger.error(f"Error building embeddings: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_embeddings())
