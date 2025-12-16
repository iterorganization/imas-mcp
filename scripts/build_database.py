#!/usr/bin/env python3
"""
Build the SQLite FTS database for the IMAS Data Dictionary.

This script creates the SQLite full-text search database from JSON schema data
for fast lexical search capabilities. The database is persisted to disk for
reuse across server sessions.
"""

import logging
import sys

import click

from imas_codex.search.document_store import DocumentStore


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force rebuild even if database already exists",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check if database exists, don't build it",
)
def build_database(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    check_only: bool,
) -> int:
    """Build the SQLite FTS database for lexical search.

    This command creates a SQLite full-text search (FTS5) database from the
    JSON schema data. The database enables fast lexical search capabilities
    and is persisted to disk for reuse.

    The database is cached and will only be rebuilt if the source data changes
    or if forced with --force.

    Examples:
        build-database                          # Build with default settings
        build-database -v                       # Build with verbose logging
        build-database -f                       # Force rebuild database
        build-database --ids-filter "core_profiles equilibrium"  # Build specific IDS
        build-database --check-only             # Check if database exists
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
        logger.info("Starting SQLite FTS database build process...")

        # Parse ids_filter string into a set if provided
        ids_set: set | None = set(ids_filter.split()) if ids_filter else None
        if ids_set:
            logger.info(f"Building database for specific IDS: {sorted(ids_set)}")
        else:
            logger.info("Building database for all available IDS")

        # Create document store with ids_set if provided
        if ids_set:
            document_store = DocumentStore(ids_set=ids_set)
        else:
            document_store = DocumentStore()

        # Check if database exists
        sqlite_path = document_store._sqlite_path  # noqa: SLF001

        if check_only:
            if sqlite_path.exists():
                # Get cache info
                cache_info = document_store.get_cache_info()
                if cache_info.get("cached"):
                    click.echo(
                        f"Database exists: {cache_info.get('document_count', 0)} documents"
                    )
                    click.echo(f"IDS count: {cache_info.get('ids_count', 0)}")
                    if "file_size_mb" in cache_info:
                        click.echo(f"File size: {cache_info['file_size_mb']:.1f} MB")
                    if "created_at" in cache_info:
                        click.echo(f"Created: {cache_info['created_at']}")
                    click.echo(f"Location: {sqlite_path}")
                    return 0
                else:
                    click.echo("Database file exists but is invalid or incomplete")
                    return 1
            else:
                click.echo("Database does not exist")
                return 1

        # Load documents (this will trigger database build if needed)
        logger.info("Loading documents into memory...")
        document_store.load_all_documents(force_rebuild_index=force)

        # Get statistics
        stats = document_store.get_statistics()
        cache_info = stats.get("cache", {})

        # Log success information
        logger.info("Database build completed successfully:")
        logger.info(f"  - Documents: {stats['total_documents']}")
        logger.info(f"  - IDS count: {stats['total_ids']}")
        logger.info(f"  - Physics domains: {stats['physics_domains']}")
        logger.info(f"  - Database file: {sqlite_path}")

        if "file_size_mb" in cache_info:
            logger.info(f"  - File size: {cache_info['file_size_mb']:.1f} MB")

        # Print summary for scripts/CI
        action = "Rebuilt" if force else "Built"
        click.echo(
            f"{action} database with {stats['total_documents']} documents from {stats['total_ids']} IDS"
        )
        if "file_size_mb" in cache_info:
            click.echo(f"Database size: {cache_info['file_size_mb']:.1f} MB")
        click.echo(f"Location: {sqlite_path}")

        return 0

    except Exception as e:
        logger.error(f"Error building database: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_database())
