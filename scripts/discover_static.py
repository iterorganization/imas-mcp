#!/usr/bin/env python3
"""
Discover and ingest MDSplus static/machine-description trees.

Static trees contain time-invariant constructional data: coil positions,
vessel geometry, tile contours, magnetic probe positions, flux loop
positions, Green's functions, and poloidal meshes. They are versioned
by machine configuration (e.g., baffle changes), not by shot number.

Usage:
    # Discover all configured static trees for TCV
    uv run discover-static tcv

    # Specify a particular tree
    uv run discover-static tcv --tree static

    # Extract actual numerical values (R/Z, matrices)
    uv run discover-static tcv --values

    # Only specific versions
    uv run discover-static tcv --versions 1,2,3

    # Dry run
    uv run discover-static tcv --dry-run -v
"""

import logging
import sys

import click

from imas_codex.graph import GraphClient
from imas_codex.mdsplus.static import (
    discover_static_tree,
    get_static_tree_config,
    ingest_static_tree,
)
from imas_codex.mdsplus.static_mapping import (
    build_mapping_proposals,
    persist_mapping_proposals,
)

logger = logging.getLogger(__name__)


@click.command()
@click.argument("facility")
@click.option(
    "--tree", "-t", "tree_name", help="Static tree name (default: from config)"
)
@click.option(
    "--versions",
    help="Comma-separated version numbers (default: all from config)",
)
@click.option(
    "--values/--no-values",
    "extract_values",
    default=None,
    help="Extract numerical values (R/Z, matrices). Default: from config.",
)
@click.option("--dry-run", is_flag=True, help="Preview without ingesting")
@click.option("--timeout", type=int, default=300, help="SSH timeout in seconds")
@click.option(
    "--map-imas/--no-map-imas", default=True, help="Generate IMAS mapping proposals"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
def main(
    facility: str,
    tree_name: str | None,
    versions: str | None,
    extract_values: bool | None,
    dry_run: bool,
    timeout: int,
    map_imas: bool,
    verbose: bool,
    quiet: bool,
) -> int:
    """Discover and ingest static/machine-description MDSplus trees.

    FACILITY is the SSH host alias (e.g., "tcv").

    Static trees contain time-invariant constructional data versioned by
    machine configuration changes. This command extracts tree structure,
    tags, metadata, and optionally numerical values, then ingests them
    into the knowledge graph.

    Creates:
    - TreeModelVersion nodes for each configuration version
    - TreeNode nodes with first_shot/last_shot applicability ranges
    - INTRODUCED_IN/REMOVED_IN relationships linking nodes to versions
    - AT_FACILITY relationships

    Examples:
        discover-static tcv
        discover-static tcv --values
        discover-static tcv --tree static --versions 1,3,8
        discover-static tcv --dry-run -v
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

    # Load static tree configs
    configs = get_static_tree_config(facility)
    if not configs:
        click.echo(
            f"No static_trees configured for {facility}. "
            f"Add static_trees to data_sources.mdsplus in the facility YAML.",
            err=True,
        )
        return 1

    # Filter to specific tree if requested
    if tree_name:
        configs = [c for c in configs if c.get("tree_name") == tree_name]
        if not configs:
            click.echo(
                f"No static tree named '{tree_name}' in {facility} config", err=True
            )
            return 1

    # Check Neo4j connectivity upfront (unless dry-run)
    if not dry_run:
        try:
            with GraphClient() as client:
                client.query("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            click.echo(
                f"Neo4j is not available: {e}\nUse --dry-run to skip ingestion.",
                err=True,
            )
            return 1

    # Process each configured static tree
    for cfg in configs:
        tname = cfg["tree_name"]
        click.echo(f"\nProcessing static tree: {facility}:{tname}")

        # Determine versions
        ver_list: list[int] | None = None
        if versions:
            ver_list = [int(v.strip()) for v in versions.split(",")]
        else:
            ver_configs = cfg.get("versions", [])
            if ver_configs:
                ver_list = [v["version"] for v in ver_configs]

        # Determine value extraction
        do_extract = extract_values
        if do_extract is None:
            do_extract = cfg.get("extract_values", False)

        click.echo(f"  Versions: {ver_list or 'auto'}")
        click.echo(f"  Extract values: {do_extract}")

        # Phase 1: Extract from facility
        click.echo(f"  Extracting from {facility}...")
        try:
            data = discover_static_tree(
                facility=facility,
                tree_name=tname,
                versions=ver_list,
                extract_values=do_extract,
                timeout=timeout,
            )
        except Exception as e:
            click.echo(f"  Extraction failed: {e}", err=True)
            logger.exception("Static tree extraction failed")
            continue

        # Show summary
        for ver_str, ver_data in sorted(
            data.get("versions", {}).items(), key=lambda x: int(x[0])
        ):
            if "error" in ver_data:
                click.echo(f"  v{ver_str}: ERROR — {ver_data['error'][:80]}")
            else:
                tags = len(ver_data.get("tags", {}))
                nodes = ver_data.get("node_count", 0)
                click.echo(f"  v{ver_str}: {nodes} nodes, {tags} tags")

        # Show diffs
        diff = data.get("diff", {})
        for ver_str in sorted(diff.get("added", {}).keys(), key=int):
            n_added = len(diff["added"][ver_str])
            click.echo(f"  v{ver_str} added {n_added} nodes")
        for ver_str in sorted(diff.get("removed", {}).keys(), key=int):
            n_removed = len(diff["removed"][ver_str])
            click.echo(f"  v{ver_str} removed {n_removed} nodes")

        # Phase 2: Ingest to graph
        version_config = cfg.get("versions")

        if dry_run:
            click.echo("  [DRY RUN] Ingestion preview:")
            ingest_static_tree(
                client=None,  # type: ignore[arg-type]
                facility=facility,
                data=data,
                version_config=version_config,
                dry_run=True,
            )
        else:
            with GraphClient() as client:
                stats = ingest_static_tree(
                    client=client,
                    facility=facility,
                    data=data,
                    version_config=version_config,
                )
                click.echo(
                    f"  Ingested: {stats['versions_created']} versions, "
                    f"{stats['nodes_created']} nodes, "
                    f"{stats['values_stored']} values"
                )

        # Phase 3: IMAS mapping proposals
        if map_imas:
            click.echo("  Building IMAS mapping proposals...")
            mapping_result = build_mapping_proposals(facility, data)
            click.echo(
                f"  Mapped {mapping_result.stats.get('mapped', 0)} nodes → "
                f"{mapping_result.stats.get('unique_imas_targets', 0)} IMAS targets, "
                f"{mapping_result.stats.get('unmapped', 0)} unmapped"
            )

            if mapping_result.proposals:
                if dry_run:
                    click.echo("  [DRY RUN] Mapping proposals preview:")
                    for p in mapping_result.proposals[:10]:
                        click.echo(
                            f"    {p['source_tag']:15s} → {p['target_path'][:50]:50s} "
                            f"(confidence={p['confidence']:.2f})"
                        )
                    if len(mapping_result.proposals) > 10:
                        click.echo(
                            f"    ... and {len(mapping_result.proposals) - 10} more"
                        )
                else:
                    with GraphClient() as client:
                        n_proposals = persist_mapping_proposals(client, mapping_result)
                        click.echo(f"  Persisted {n_proposals} mapping proposals")

            if mapping_result.unmapped_nodes:
                click.echo(
                    f"  {len(mapping_result.unmapped_nodes)} data nodes without "
                    f"IMAS mapping (Green's functions, eigenmodes, mesh, derived)"
                )

    click.echo("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
