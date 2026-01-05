#!/usr/bin/env python3
"""
Discover MDSplus tree structure epochs and ingest to Neo4j.

This script discovers structural epochs (when tree structure changed) and
builds the "super tree" - TreeNodes with applicability ranges (first_shot/last_shot).

Usage:
    # Full discovery with metadata extraction (default)
    uv run discover-mdsplus epfl results

    # Use legacy sequential mode (slower, rough boundaries)
    uv run discover-mdsplus epfl results --sequential --step 100

    # Refine existing rough boundaries
    uv run discover-mdsplus epfl results --refine

    # Skip metadata extraction (faster but less complete)
    uv run discover-mdsplus epfl results --skip-metadata

    # Dry run to preview
    uv run discover-mdsplus epfl results --dry-run

    # Resume interrupted discovery
    uv run discover-mdsplus epfl results --checkpoint discovery.json

    # Specify shot range
    uv run discover-mdsplus epfl results --start 80000 --end 89000
"""

import logging
import sys
from pathlib import Path

import click

from imas_codex.graph import GraphClient
from imas_codex.mdsplus import (
    cleanup_legacy_nodes,
    discover_epochs,
    discover_epochs_optimized,
    enrich_graph_metadata,
    ingest_epochs,
    ingest_super_tree,
    merge_legacy_metadata,
    refine_boundaries,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.argument("tree_name")
@click.option("--start", "-s", type=int, help="Start shot (default: 3000)")
@click.option("--end", "-e", type=int, help="End shot (default: current)")
@click.option(
    "--coarse-step", type=int, default=1000, help="Coarse scan step (batch mode)"
)
@click.option("--step", type=int, default=500, help="Shot step (sequential mode)")
@click.option("--sequential", is_flag=True, help="Use sequential mode (slower)")
@click.option("--checkpoint", type=click.Path(), help="Checkpoint file for resume")
@click.option("--dry-run", is_flag=True, help="Preview without ingesting")
@click.option("--epochs-only", is_flag=True, help="Only ingest epochs, skip super tree")
@click.option(
    "--full", is_flag=True, help="Force full scan, ignoring existing graph data"
)
@click.option(
    "--refine", is_flag=True, help="Refine existing rough boundaries (for legacy data)"
)
@click.option("--skip-metadata", is_flag=True, help="Skip metadata extraction (faster)")
@click.option(
    "--clean",
    is_flag=True,
    help="Merge and cleanup legacy nodes after ingestion",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
def main(
    facility: str,
    tree_name: str,
    start: int | None,
    end: int | None,
    coarse_step: int,
    step: int,
    sequential: bool,
    checkpoint: str | None,
    dry_run: bool,
    epochs_only: bool,
    full: bool,
    refine: bool,
    skip_metadata: bool,
    clean: bool,
    verbose: bool,
    quiet: bool,
) -> int:
    """Discover MDSplus tree structure epochs and build super tree.

    FACILITY is the SSH host alias (e.g., "epfl").
    TREE_NAME is the MDSplus tree name (e.g., "results").

    By default uses optimized batch mode with exact boundaries and metadata extraction.
    Use --sequential for the legacy per-shot scanning mode (rough boundaries).

    This creates:
    - TreeModelVersion nodes for each structural epoch
    - TreeNode nodes with first_shot/last_shot applicability ranges
    - INTRODUCED_IN/REMOVED_IN relationships linking nodes to versions
    - Real units and descriptions extracted from MDSplus

    Examples:
        discover-mdsplus epfl results
        discover-mdsplus epfl results --sequential --step 100
        discover-mdsplus epfl results --checkpoint discovery.json
        discover-mdsplus epfl results --dry-run -v
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
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Check Neo4j connectivity upfront (unless dry-run)
    if not dry_run:
        try:
            with GraphClient() as client:
                client.query("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            click.echo(
                f"✗ Neo4j is not available: {e}\n"
                "Start Neo4j with: uv run imas-codex neo4j start\n"
                "Or use --dry-run to skip ingestion.",
                err=True,
            )
            return 1

    try:
        # Special mode: refine-only (just refine and enrich existing data)
        if refine:
            click.echo(f"Refining existing data for {facility}:{tree_name}...")

            with GraphClient() as client:
                # Get current shot for metadata
                from imas_codex.mdsplus import BatchDiscovery

                discovery = BatchDiscovery(facility, tree_name)
                current_shot = discovery.get_current_shot()

                # Refine boundaries
                click.echo("\nRefining epoch boundaries...")
                refine_result = refine_boundaries(client, facility, tree_name, dry_run)
                if refine_result["boundaries_refined"] > 0:
                    click.echo(
                        f"✓ Refined {refine_result['boundaries_refined']} of "
                        f"{refine_result['epochs_checked']} boundaries"
                    )
                else:
                    click.echo("✓ All boundaries already at exact precision")

                # Enrich metadata (unless skip)
                if not skip_metadata:
                    click.echo(f"\nExtracting metadata from shot {current_shot}...")
                    meta_result = enrich_graph_metadata(
                        client, facility, tree_name, current_shot, dry_run
                    )
                    click.echo(
                        f"✓ Updated {meta_result['units_updated']} nodes with units, "
                        f"{meta_result['descriptions_updated']} with descriptions"
                    )

            return 0

        # Phase 1: Discover epochs
        click.echo(f"Discovering epochs for {facility}:{tree_name}...")

        if sequential:
            # Legacy sequential mode
            click.echo(f"Using sequential mode (step={step})")
            epochs, structures = discover_epochs(
                facility=facility,
                tree_name=tree_name,
                start_shot=start,
                end_shot=end,
                step=step,
            )
        else:
            # Optimized batch mode with optional incremental support
            mode = "full" if full else "incremental"
            click.echo(
                f"Using optimized batch mode ({mode}, coarse_step={coarse_step})"
            )
            checkpoint_path = Path(checkpoint) if checkpoint else None

            # Get graph client for incremental mode (unless --full)
            graph_client = None
            if not full and not dry_run:
                try:
                    graph_client = GraphClient()
                except Exception:
                    pass  # Will do full scan if graph unavailable

            epochs, structures = discover_epochs_optimized(
                facility=facility,
                tree_name=tree_name,
                start_shot=start,
                end_shot=end,
                coarse_step=coarse_step,
                checkpoint_path=checkpoint_path,
                client=graph_client,
            )

            if graph_client is not None:
                graph_client.close()

        if not epochs:
            click.echo("No epochs discovered")
            return 0

        # Display epoch summary with new/existing breakdown
        new_epochs = [e for e in epochs if e.get("is_new", True)]
        existing_epochs = [e for e in epochs if not e.get("is_new", True)]

        click.echo(f"\nDiscovered {len(epochs)} epochs:")
        if existing_epochs:
            click.echo(f"  - {len(existing_epochs)} already in graph (skipped scan)")
        if new_epochs:
            click.echo(f"  - {len(new_epochs)} newly scanned")
        click.echo()

        for epoch in epochs:
            shot_range = f"{epoch['first_shot']}"
            if epoch.get("last_shot"):
                shot_range += f"-{epoch['last_shot']}"
            else:
                shot_range += "-current"

            delta = ""
            if epoch.get("nodes_added"):
                delta += f"+{epoch['nodes_added']}"
            if epoch.get("nodes_removed"):
                delta += f" -{epoch['nodes_removed']}"

            subtrees = ""
            if epoch.get("added_subtrees"):
                subtrees = f" [{', '.join(epoch['added_subtrees'][:3])}...]"

            click.echo(
                f"  v{epoch['version']:2d}: {shot_range:15s} "
                f"{epoch['node_count']:5d} nodes {delta:10s}{subtrees}"
            )

        if dry_run:
            click.echo("\n(dry run - not ingesting)")
            # Show super tree stats
            all_paths = set()
            for epoch in epochs:
                all_paths.update(epoch.get("added_paths", []))
            click.echo(f"Super tree would have {len(all_paths)} unique TreeNodes")
            return 0

        # Phase 2: Ingest epochs
        click.echo("\nIngesting TreeModelVersion nodes...")
        with GraphClient() as client:
            epoch_count = ingest_epochs(client, epochs)
            click.echo(f"✓ Ingested {epoch_count} epochs")

            # Phase 3: Build super tree (unless epochs-only)
            if not epochs_only:
                click.echo("\nBuilding super tree with applicability ranges...")
                node_count = ingest_super_tree(
                    client, facility, tree_name, epochs, structures
                )
                click.echo(f"✓ Ingested {node_count} TreeNodes")

                # Phase 4: Refine boundaries (if requested or if sequential mode)
                if refine or sequential:
                    click.echo("\nRefining epoch boundaries...")
                    refine_result = refine_boundaries(client, facility, tree_name)
                    if refine_result["boundaries_refined"] > 0:
                        click.echo(
                            f"✓ Refined {refine_result['boundaries_refined']} boundaries"
                        )
                    else:
                        click.echo("✓ All boundaries already at exact precision")

                # Phase 5: Enrich with metadata (unless skipped)
                if not skip_metadata:
                    # Use most recent shot for metadata
                    latest_shot = max(e["first_shot"] for e in epochs)
                    click.echo(f"\nExtracting metadata from shot {latest_shot}...")
                    meta_result = enrich_graph_metadata(
                        client, facility, tree_name, latest_shot
                    )
                    click.echo(
                        f"✓ Updated {meta_result['units_updated']} nodes with units, "
                        f"{meta_result['descriptions_updated']} with descriptions"
                    )
                else:
                    click.echo(
                        "\n(skipping metadata extraction - use without --skip-metadata)"
                    )

                # Phase 6: Merge and cleanup legacy nodes (if --clean)
                if clean:
                    click.echo("\nMerging legacy node metadata...")
                    merge_result = merge_legacy_metadata(
                        client, facility, tree_name, dry_run
                    )
                    click.echo(
                        f"✓ Merged {merge_result['descriptions_merged']} descriptions, "
                        f"{merge_result['physics_domains_merged']} physics_domains"
                    )

                    click.echo("\nCleaning up superseded legacy nodes...")
                    cleanup_result = cleanup_legacy_nodes(
                        client, facility, tree_name, dry_run
                    )
                    click.echo(
                        f"✓ Deleted {cleanup_result['deleted']} legacy nodes, "
                        f"kept {cleanup_result['to_keep']} (no epoch equivalent)"
                    )
            else:
                click.echo(
                    "\n(skipping super tree - use without --epochs-only to build)"
                )

        return 0

    except Exception as e:
        logger.exception("Discovery/ingestion failed")
        click.echo(f"✗ Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
