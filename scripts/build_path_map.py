#!/usr/bin/env python3
"""
Build the path map for IMAS Data Dictionary version upgrades.

This script creates a JSON file mapping old paths to new paths across DD versions,
enabling the MCP server to suggest path mappings for deprecated paths and
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
from imas_mcp.core.exclusions import EXCLUSION_REASONS, ExclusionChecker
from imas_mcp.core.progress_monitor import create_progress_monitor
from imas_mcp.resource_path_accessor import ResourcePathAccessor

logger = logging.getLogger(__name__)


def _collect_all_paths(ids_def, ids_name: str, prefix: str = "") -> list[str]:
    """Recursively collect all paths from an IDS definition."""
    paths = []
    try:
        metadata = ids_def.metadata
        for child_name in metadata.children:
            child_path = f"{prefix}{child_name}" if prefix else child_name
            full_path = f"{ids_name}/{child_path}"
            paths.append(full_path)

            # Recurse into child
            try:
                child = getattr(ids_def, child_name)
                if hasattr(child, "metadata"):
                    paths.extend(_collect_all_paths(child, ids_name, f"{child_path}/"))
            except Exception:
                pass
    except Exception:
        pass
    return paths


def _find_deprecation_version(
    path: str,
    last_valid: str,
    sorted_versions: list[str],
    paths_by_version: dict[str, set[str]],
) -> str:
    """Find the first version where a path no longer exists."""
    # Find index of last_valid version
    try:
        start_idx = sorted_versions.index(last_valid)
    except ValueError:
        return sorted_versions[-1]

    # Look for the first version after last_valid where path doesn't exist
    for version in sorted_versions[start_idx + 1 :]:
        if path not in paths_by_version.get(version, set()):
            return version

    # If path exists in all subsequent versions, it's deprecated in target
    return sorted_versions[-1]


def build_path_map(
    target_version: str,
    ids_filter: set[str] | None = None,
    verbose: bool = False,
    use_rich: bool | None = None,
) -> dict:
    """
    Build bidirectional path map from all older versions to target.

    Args:
        target_version: The DD version to map paths to.
        ids_filter: Optional set of IDS names to include.
        verbose: Enable verbose logging.
        use_rich: Force rich progress (True), logging (False), or auto-detect (None).

    Returns:
        Dictionary with metadata, old_to_new, and new_to_old mappings.
    """
    # Get all available DD versions
    all_versions = imas.dd_zip.dd_xml_versions()
    source_versions = [v for v in all_versions if v < target_version]

    if verbose:
        logger.info(f"Building path map to {target_version}")
        logger.info(f"Source versions: {len(source_versions)} versions")

    target_factory = imas.IDSFactory(target_version)
    target_ids_names = set(target_factory.ids_names())

    # Apply IDS filter if provided
    if ids_filter:
        target_ids_names = target_ids_names & ids_filter

    # Track mappings with version info
    old_to_new: dict[str, dict] = {}
    new_to_old: dict[str, list[dict]] = defaultdict(list)

    # Track paths that exist in each version (to find when they disappear)
    # We need to find the first version where a path no longer exists
    sorted_versions = sorted(source_versions) + [target_version]

    # Create progress monitor for version processing
    progress = create_progress_monitor(
        use_rich=use_rich,
        logger=logger,
        item_names=sorted_versions,
        description_template="Indexing: {item}",
    )

    # First pass: collect all paths that exist in each version
    paths_by_version: dict[str, set[str]] = {}
    progress.start_processing(sorted_versions, "Indexing IDS paths")

    for version in sorted_versions:
        progress.set_current_item(version)
        factory = imas.IDSFactory(version)
        paths_by_version[version] = set()

        for ids_name in factory.ids_names():
            if ids_name not in target_ids_names:
                continue
            try:
                ids_def = factory.new(ids_name)
                for path in _collect_all_paths(ids_def, ids_name):
                    paths_by_version[version].add(path)
            except Exception as e:
                if verbose:
                    logger.debug(
                        f"Could not collect paths for {ids_name} in {version}: {e}"
                    )
        progress.update_progress(version)

    progress.finish_processing()

    # Create progress monitor for mapping building
    progress = create_progress_monitor(
        use_rich=use_rich,
        logger=logger,
        item_names=sorted(source_versions),
        description_template="Mapping: {item}",
    )

    # Second pass: build path mappings with correct deprecation versions
    progress.start_processing(sorted(source_versions), "Building path mappings")

    for source_version in sorted(source_versions):
        progress.set_current_item(source_version)

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

                # Only record first occurrence (earliest source version)
                if full_old in old_to_new:
                    continue

                # Find the first version where this path no longer exists
                deprecated_in = _find_deprecation_version(
                    full_old, source_version, sorted_versions, paths_by_version
                )

                old_to_new[full_old] = {
                    "new_path": full_new,
                    "deprecated_in": deprecated_in,
                    "last_valid_version": source_version,
                }

                # Build reverse mapping (new_to_old)
                if full_new:
                    entry = {
                        "old_path": full_old,
                        "deprecated_in": deprecated_in,
                    }
                    existing_old_paths = [e["old_path"] for e in new_to_old[full_new]]
                    if full_old not in existing_old_paths:
                        new_to_old[full_new].append(entry)

        progress.update_progress(source_version)

    progress.finish_processing()

    # Build excluded paths section: paths that are mapping targets but excluded from index
    exclusion_checker = ExclusionChecker()
    excluded_paths: dict[str, str] = {}

    for new_path in new_to_old:
        reason = exclusion_checker.get_exclusion_reason(new_path)
        if reason:
            excluded_paths[new_path] = reason

    # Build final structure
    mapping_data = {
        "metadata": {
            "target_version": target_version,
            "source_versions": sorted(source_versions),
            "generated_at": datetime.now(UTC).isoformat(),
            "total_mappings": len(old_to_new),
            "paths_with_history": len(new_to_old),
            "excluded_target_paths": len(excluded_paths),
        },
        "exclusion_reasons": EXCLUSION_REASONS,
        "excluded_paths": excluded_paths,
        "old_to_new": old_to_new,
        "new_to_old": dict(new_to_old),
    }

    return mapping_data


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force rebuild even if path map file already exists",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include (space-separated)",
)
@click.option(
    "--check-only",
    is_flag=True,
    help="Only check if path map file exists, don't build it",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Override output path for path map file",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress bar, use plain logging",
)
def build_path_map_cli(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    check_only: bool,
    output: str | None,
    no_rich: bool,
) -> int:
    """Build the path map for IMAS DD version upgrades.

    This command creates a JSON file mapping old paths to new paths,
    enabling mapping suggestions for deprecated paths and rename
    history for current paths.

    Examples:
        build-path-map                        # Build with default settings
        build-path-map -v                     # Build with verbose logging
        build-path-map -f                     # Force rebuild
        build-path-map --ids-filter "equilibrium core_profiles"
    """
    # Set up logging - suppress imas library's verbose DD parsing logs
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

    # Suppress imas library's verbose logging (dd_zip parsing messages)
    logging.getLogger("imas").setLevel(logging.WARNING)
    logging.getLogger("imas.dd_zip").setLevel(logging.WARNING)

    try:
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            path_accessor = ResourcePathAccessor(dd_version=dd_version)
            output_path = path_accessor.mappings_dir / "path_mappings.json"

        # Check-only mode
        if check_only:
            if output_path.exists():
                with open(output_path) as f:
                    data = json.load(f)
                metadata = data.get("metadata", {})
                click.echo(f"Path map file exists: {output_path}")
                click.echo(f"Target version: {metadata.get('target_version')}")
                click.echo(f"Total mappings: {metadata.get('total_mappings')}")
                click.echo(f"Generated at: {metadata.get('generated_at')}")
                return 0
            else:
                click.echo("Path map file does not exist")
                return 1

        # Check if rebuild needed
        if output_path.exists() and not force:
            logger.info(f"Path map file already exists: {output_path}")
            logger.info("Use --force to rebuild")
            return 0

        logger.info(f"Building path map for DD version {dd_version}...")

        # Parse IDS filter
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.info(f"Filtering to IDS: {sorted(ids_set)}")

        # Determine rich usage
        use_rich: bool | None = None
        if no_rich:
            use_rich = False

        # Build path map
        mapping_data = build_path_map(
            target_version=dd_version,
            ids_filter=ids_set,
            verbose=verbose,
            use_rich=use_rich,
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write path map file
        with open(output_path, "w") as f:
            json.dump(mapping_data, f, indent=2)

        # Report results
        metadata = mapping_data["metadata"]
        logger.info("Path map built successfully:")
        logger.info(f"  - Target version: {metadata['target_version']}")
        logger.info(f"  - Source versions: {len(metadata['source_versions'])}")
        logger.info(f"  - Total mappings: {metadata['total_mappings']}")
        logger.info(f"  - Paths with history: {metadata['paths_with_history']}")
        logger.info(f"  - Excluded target paths: {metadata['excluded_target_paths']}")
        logger.info(f"  - Output: {output_path}")

        click.echo(f"Built path map with {metadata['total_mappings']} mappings")
        click.echo(f"Output: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Error building path map: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_path_map_cli())
