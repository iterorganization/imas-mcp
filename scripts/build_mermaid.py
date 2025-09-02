#!/usr/bin/env python3
"""
Build Mermaid graph visualizations for IMAS IDS structures.

This script generates Mermaid diagrams representing the hierarchical structure
and relationships within IMAS IDS data, optimized for LLM consumption.
"""

import json
import logging
import sys
from pathlib import Path

import click

from imas_mcp.structure.mermaid_generator import MermaidGraphGenerator


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--force", "-f", is_flag=True, help="Force rebuild even if files already exist"
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'core_profiles equilibrium')",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Custom output directory (defaults to imas_mcp/resources)",
)
@click.option(
    "--schemas-dir",
    type=click.Path(path_type=Path),
    help="Directory containing schema files (defaults to imas_mcp/resources/schemas)",
)
def build_mermaid(
    verbose: bool,
    quiet: bool,
    force: bool,
    ids_filter: str,
    output_dir: Path | None,
    schemas_dir: Path | None,
) -> int:
    """Build Mermaid graph visualizations for IMAS IDS structures.

    This command generates Mermaid diagrams representing the hierarchical structure
    and relationships within IMAS IDS data. The graphs are optimized for LLM
    consumption and provide visual understanding of data organization.

    Examples:
        build-mermaid                    # Build graphs with default settings
        build-mermaid -v                 # Build with verbose logging
        build-mermaid -f                 # Force rebuild even if exists
        build-mermaid --ids-filter "core_profiles equilibrium"  # Build specific IDS only
        build-mermaid --output-dir /path/to/custom/dir  # Use custom output directory
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
        logger.info("Starting Mermaid graph build process...")

        # Determine output directory
        if output_dir is None:
            # Default to imas_mcp/resources
            output_dir = Path(__file__).parent.parent / "imas_mcp" / "resources"

        if schemas_dir is None:
            schemas_dir = output_dir / "schemas" / "detailed"

        # Validate schemas directory exists
        if not schemas_dir.exists():
            logger.error(f"Schemas directory not found: {schemas_dir}")
            logger.error("Please run 'build-schemas' first to generate schema data")
            return 1

        # Load IDS catalog
        catalog_file = schemas_dir.parent / "ids_catalog.json"
        if not catalog_file.exists():
            logger.error(f"IDS catalog not found: {catalog_file}")
            logger.error("Please run 'build-schemas' first to generate schema data")
            return 1

        with open(catalog_file, encoding="utf-8") as f:
            catalog_data = json.load(f)

        ids_catalog = catalog_data.get("ids_catalog", {})
        if not ids_catalog:
            logger.error("No IDS data found in catalog")
            return 1

        # Apply IDS filter if provided
        ids_set = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            filtered_catalog = {
                ids_name: data
                for ids_name, data in ids_catalog.items()
                if ids_name in ids_set
            }

            # Check for missing IDS
            missing_ids = ids_set - set(filtered_catalog.keys())
            if missing_ids:
                logger.warning(f"IDS not found in catalog: {sorted(missing_ids)}")

            ids_catalog = filtered_catalog
            logger.info(
                f"Building graphs for specific IDS: {sorted(ids_catalog.keys())}"
            )
        else:
            logger.info(f"Building graphs for all {len(ids_catalog)} IDS")

        if not ids_catalog:
            logger.error("No valid IDS found to process")
            return 1

        # Initialize Mermaid generator
        mermaid_generator = MermaidGraphGenerator(output_dir)

        # Check if we need to build
        mermaid_dir = output_dir / "mermaid"
        overview_file = mermaid_dir / "ids_overview.md"
        should_build = force or not overview_file.exists()

        if should_build:
            if force and overview_file.exists():
                logger.info("Force rebuilding existing Mermaid graph files")
                # Clean up existing files when force rebuilding
                import shutil

                if mermaid_dir.exists():
                    shutil.rmtree(mermaid_dir)
            else:
                logger.info("Mermaid graph files do not exist, building new files...")

            # Generate all graphs using the build method
            logger.info("Starting Mermaid graph generation...")
            mermaid_generator.build(ids_set)

            # Log success
            logger.info("Mermaid graphs built successfully:")
            logger.info(f"  - Output directory: {mermaid_dir}")

            # Count generated files from both directories
            detailed_dir = mermaid_dir / "detailed"
            overview_files = list(mermaid_dir.glob("*.md"))
            hierarchy_files = list(detailed_dir.glob("*_hierarchy.md"))
            physics_files = list(detailed_dir.glob("*_physics_domains.md"))
            complexity_files = list(detailed_dir.glob("*_complexity.md"))

            total_files = (
                len(overview_files)
                + len(hierarchy_files)
                + len(physics_files)
                + len(complexity_files)
            )

            logger.info(f"  - Total graph files: {total_files}")
            logger.info(f"  - Overview graphs: {len(overview_files)}")
            logger.info(f"  - Hierarchy graphs: {len(hierarchy_files)}")
            logger.info(f"  - Physics domain graphs: {len(physics_files)}")
            logger.info(f"  - Complexity graphs: {len(complexity_files)}")

            # Generate summary for each IDS
            for ids_name in ids_catalog.keys():
                available_graphs = mermaid_generator.get_available_graphs(ids_name)
                if available_graphs:
                    logger.debug(f"  - {ids_name}: {', '.join(available_graphs)}")

            # Print summary for scripts/CI
            click.echo(f"Built {total_files} Mermaid graph files in {mermaid_dir}")
        else:
            logger.info(f"Mermaid graph files already exist in {mermaid_dir}")
            click.echo(f"Mermaid graphs already exist in {mermaid_dir}")

        return 0

    except Exception as e:
        logger.error(f"Error building Mermaid graphs: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_mermaid())
