#!/usr/bin/env python3
"""
Build the path migration map for IMAS Data Dictionary version upgrades.

This script creates a JSON file mapping old paths to new paths across DD versions,
enabling the MCP server to suggest path migrations for deprecated paths and
provide rename history for current paths.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import click
import imas

from imas_mcp import dd_version
from imas_mcp.resource_path_accessor import ResourcePathAccessor


def build_migration_map(
    target_version: str,
    ids_filter: set[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Build bidirectional path migration map from all older versions to target.

    Args:
        target_version: The DD version to migrate paths to.
        ids_filter: Optional set of IDS names to include.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with metadata, old_to_new, and new_to_old mappings.
    """
    logger = logging.getLogger(__name__)

    # Get all available DD versions
    all_versions = imas.dd_zip.dd_xml_versions()
    source_versions = [v for v in all_versions if v < target_version]

    if verbose:
        logger.info(f"Building migration map to {target_version}")
        logger.info(f"Source versions: {len(source_versions)} versions")

    target_factory = imas.IDSFactory(target_version)
    target_ids_names = set(target_factory.ids_names())

    # Apply IDS filter if provided
    if ids_filter:
        target_ids_names = target_ids_names & ids_filter

    # Track migrations with version info
    old_to_new: dict[str, dict] = {}
    new_to_old: dict[str, list[dict]] = defaultdict(list)

    # Track the earliest version where each path was deprecated
    deprecation_versions: dict[str, str] = {}
    # Track the latest version where each old path was valid
    last_valid_versions: dict[str, str] = {}

    for source_version in sorted(source_versions):
        if verbose:
            logger.debug(f"Processing version {source_version}")

        source_factory = imas.IDSFactory(source_version)

        for ids_name in source_factory.ids_names():
            if ids_name not in target_ids_names:
                continue

            try:
                version_map, _ = imas.ids_convert.dd_version_map_from_factories(
                    ids_name, source_factory, target_factory
                )
            except Exception as e:
                logger.warning(
                    f"Failed to get version map for {ids_name} "
                    f"from {source_version}: {e}"
                )
                continue

            for old_path, new_path in version_map.old_to_new.path.items():
                full_old = f"{ids_name}/{old_path}"
                full_new = f"{ids_name}/{new_path}" if new_path else None

                # Skip if path unchanged
                if full_old == full_new:
                    continue

                # Track last valid version for this old path
                # (the version before it was deprecated)
                if full_old not in last_valid_versions:
                    last_valid_versions[full_old] = source_version

                # Update old_to_new mapping
                if full_old not in old_to_new:
                    old_to_new[full_old] = {
                        "new_path": full_new,
                        "deprecated_in": target_version,  # Will refine below
                        "last_valid_version": source_version,
                    }

                # Track deprecation version (first version where path changed)
                if full_old not in deprecation_versions:
                    deprecation_versions[full_old] = source_version

                # Build reverse mapping (new_to_old)
                if full_new:
                    entry = {
                        "old_path": full_old,
                        "deprecated_in": deprecation_versions.get(
                            full_old, target_version
                        ),
                    }
                    # Avoid duplicates
                    existing_old_paths = [e["old_path"] for e in new_to_old[full_new]]
                    if full_old not in existing_old_paths:
                        new_to_old[full_new].append(entry)

    # Refine deprecation versions based on tracking
    for old_path, info in old_to_new.items():
        if old_path in deprecation_versions:
            # Find the next version after last_valid
            last_valid = last_valid_versions.get(old_path, info["last_valid_version"])
            info["last_valid_version"] = last_valid

            # Deprecated in is the target version (current DD)
            # since that's when it's no longer valid
            if old_path in deprecation_versions:
                info["deprecated_in"] = target_version

    # Build final structure
    migration_data = {
        "metadata": {
            "target_version": target_version,
            "source_versions": sorted(source_versions),
            "generated_at": datetime.now(UTC).isoformat(),
            "total_migrations": len(old_to_new),
            "paths_with_history": len(new_to_old),
        },
        "old_to_new": old_to_new,
        "new_to_old": dict(new_to_old),
    }

    return migration_data


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force rebuild even if migration file already exists",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include (space-separated)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check if migration file exists, don't build it",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Override output path for migration file",
)
def build_migrations(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    check_only: bool,
    output: str | None,
) -> int:
    """Build the path migration map for IMAS DD version upgrades.

    This command creates a JSON file mapping old paths to new paths,
    enabling migration suggestions for deprecated paths and rename
    history for current paths.

    Examples:
        build-migrations                        # Build with default settings
        build-migrations -v                     # Build with verbose logging
        build-migrations -f                     # Force rebuild
        build-migrations --ids-filter "equilibrium core_profiles"
    """
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            output_path = path_accessor.migrations_dir / "path_migrations.json"

        # Check-only mode
        if check_only:
            if output_path.exists():
                with open(output_path) as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                click.echo(f"Migration file exists: {output_path}")
                click.echo(f"Target version: {metadata.get('target_version')}")
                click.echo(f"Total migrations: {metadata.get('total_migrations')}")
                click.echo(f"Generated at: {metadata.get('generated_at')}")
                return 0
            else:
                click.echo("Migration file does not exist")
                return 1

        # Check if rebuild needed
        if output_path.exists() and not force:
            logger.info(f"Migration file already exists: {output_path}")
            logger.info("Use --force to rebuild")
            return 0

        logger.info(f"Building migration map for DD version {dd_version}...")

        # Parse IDS filter
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.info(f"Filtering to IDS: {sorted(ids_set)}")

        # Build migration map
        migration_data = build_migration_map(
            target_version=dd_version,
            ids_filter=ids_set,
            verbose=verbose,
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write migration file
        with open(output_path, "w") as f:
            json.dump(migration_data, f, indent=2)

        # Report results
        metadata = migration_data["metadata"]
        logger.info("Migration map built successfully:")
        logger.info(f"  - Target version: {metadata['target_version']}")
        logger.info(f"  - Source versions: {len(metadata['source_versions'])}")
        logger.info(f"  - Total migrations: {metadata['total_migrations']}")
        logger.info(f"  - Paths with history: {metadata['paths_with_history']}")
        logger.info(f"  - Output: {output_path}")

        click.echo(
            f"Built migration map with {metadata['total_migrations']} migrations"
        )
        click.echo(f"Output: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Error building migration map: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_migrations())
