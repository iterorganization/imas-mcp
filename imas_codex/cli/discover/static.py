"""Static tree discovery: extract and ingest machine-description MDSplus trees."""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output except errors")
def static(
    facility: str,
    tree_name: str | None,
    versions: str | None,
    extract_values: bool | None,
    dry_run: bool,
    timeout: int,
    verbose: bool,
    quiet: bool,
) -> None:
    """Discover and ingest static/machine-description MDSplus trees.

    FACILITY is the SSH host alias (e.g., "tcv").

    Static trees contain time-invariant constructional data versioned by
    machine configuration changes. This command extracts tree structure,
    tags, metadata, and optionally numerical values, then ingests them
    into the knowledge graph as TreeModelVersion and TreeNode nodes.

    \b
    Examples:
      imas-codex discover static tcv
      imas-codex discover static tcv --values
      imas-codex discover static tcv --tree static --versions 1,3,8
      imas-codex discover static tcv --dry-run -v
    """
    from imas_codex.graph import GraphClient
    from imas_codex.mdsplus.static import (
        discover_static_tree,
        get_static_tree_config,
        ingest_static_tree,
    )

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

    console = Console()

    # Load static tree configs
    configs = get_static_tree_config(facility)
    if not configs:
        console.print(
            f"[red]No static_trees configured for {facility}.[/red] "
            f"Add static_trees to data_sources.mdsplus in the facility YAML.",
        )
        raise SystemExit(1)

    # Filter to specific tree if requested
    if tree_name:
        configs = [c for c in configs if c.get("tree_name") == tree_name]
        if not configs:
            console.print(
                f"[red]No static tree named '{tree_name}' in {facility} config[/red]"
            )
            raise SystemExit(1)

    # Check Neo4j connectivity upfront (unless dry-run)
    if not dry_run:
        try:
            with GraphClient() as client:
                client.query("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            console.print(
                f"[red]Neo4j is not available:[/red] {e}\n"
                "Use --dry-run to skip ingestion.",
            )
            raise SystemExit(1) from e

    # Process each configured static tree
    for cfg in configs:
        tname = cfg["tree_name"]
        console.print(f"\n[bold]Processing static tree:[/bold] {facility}:{tname}")

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

        console.print(f"  Versions: {ver_list or 'auto'}")
        console.print(f"  Extract values: {do_extract}")

        # Phase 1: Extract from facility
        console.print(f"  Extracting from {facility}...")
        try:
            data = discover_static_tree(
                facility=facility,
                tree_name=tname,
                versions=ver_list,
                extract_values=do_extract,
                timeout=timeout,
            )
        except Exception as e:
            console.print(f"  [red]Extraction failed:[/red] {e}")
            logger.exception("Static tree extraction failed")
            continue

        # Show version summary table
        table = Table(title=f"Versions — {facility}:{tname}")
        table.add_column("Version", justify="right")
        table.add_column("Nodes", justify="right")
        table.add_column("Tags", justify="right")
        table.add_column("Status")

        for ver_str, ver_data in sorted(
            data.get("versions", {}).items(), key=lambda x: int(x[0])
        ):
            if "error" in ver_data:
                table.add_row(
                    f"v{ver_str}", "—", "—", f"[red]{ver_data['error'][:60]}[/red]"
                )
            else:
                tags = len(ver_data.get("tags", {}))
                nodes = ver_data.get("node_count", 0)
                table.add_row(f"v{ver_str}", str(nodes), str(tags), "[green]OK[/green]")

        console.print(table)

        # Phase 2: Ingest to graph
        if not dry_run:
            console.print("  Ingesting to graph...")
            with GraphClient() as client:
                stats = ingest_static_tree(client, facility, data, dry_run=False)
                console.print(
                    f"  [green]✓[/green] {stats['versions_created']} versions, "
                    f"{stats['nodes_created']} nodes, "
                    f"{stats['values_stored']} values"
                )
        else:
            stats = ingest_static_tree(None, facility, data, dry_run=True)
            console.print(
                f"  [dim][DRY RUN][/dim] {stats['versions_created']} versions, "
                f"{stats['nodes_created']} nodes"
            )
