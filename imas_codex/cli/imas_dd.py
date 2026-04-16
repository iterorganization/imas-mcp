"""IMAS commands: Data Dictionary, mappings, and IDS assembly."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from imas_codex.core.node_categories import SEARCHABLE_CATEGORIES

logger = logging.getLogger(__name__)
console = Console()


@click.group()
def imas() -> None:
    """IMAS data dictionary, mappings, and IDS assembly.

    \b
    Data Dictionary:
      imas-codex imas dd build          Build/update DD graph
      imas-codex imas dd status         Show DD graph statistics
      imas-codex imas dd search         Semantic search for paths

    \b
    Mapping:
      imas-codex imas map <facility>                          Map all achievable IDS
      imas-codex imas map <f> -d magnetic_field_systems       Map by physics domain
      imas-codex imas map <f> -i pf_active                    Map specific IDS
      imas-codex imas map status <f>                          Show mapping status

    \b
    Assembly:
      imas-codex imas list [facility]   List IDS with mappings
      imas-codex imas show <f> <i>      Show IDS details + epochs
      imas-codex imas export <f> <i>    Export IDS to file
      imas-codex imas epochs <f>        List structural epochs
    """
    pass


@click.group("dd")
def dd() -> None:
    """IMAS Data Dictionary graph management."""
    pass


imas.add_command(dd)

# Register map subgroup under imas
from imas_codex.cli.map import map_cmd  # noqa: E402

imas.add_command(map_cmd, "map")


@dd.command("build")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "-c",
    "--current-only",
    is_flag=True,
    help="Process only current DD version (default: all versions)",
)
@click.option(
    "--from-version",
    type=str,
    help="Start from a specific version (for incremental updates)",
)
@click.option(
    "--reset-to",
    type=click.Choice(
        ["extracted", "built", "enriched", "refined"], case_sensitive=False
    ),
    default=None,
    help=(
        "Reset nodes to a target state before rebuilding. "
        "extracted: delete all nodes and rebuild from DD XML. "
        "built: clear enrichments and re-enrich all nodes. "
        "enriched: clear refinements and re-refine all nodes. "
        "refined: clear embeddings and re-embed all nodes."
    ),
)
@click.option(
    "--ids-filter",
    type=str,
    help="Filter to specific IDS (space-separated, for testing)",
)
@click.option(
    "--dry-run", is_flag=True, help="Preview changes without writing to graph"
)
@click.option(
    "--model",
    type=str,
    default=None,
    help=(
        "Override the LLM model for enrichment and refinement "
        "(e.g., 'openrouter/anthropic/claude-sonnet-4.6'). "
        "Default: uses [tool.imas-codex.language] model from pyproject.toml."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "Force a complete rebuild: reset all nodes to extracted state, "
        "skip all hash checks, re-enrich and re-embed everything."
    ),
)
def imas_build(
    verbose: bool,
    quiet: bool,
    current_only: bool,
    from_version: str | None,
    reset_to: str | None,
    ids_filter: str | None,
    dry_run: bool,
    model: str | None,
    force: bool,
) -> None:
    """Build the IMAS Data Dictionary Knowledge Graph.

    Populates Neo4j with complete IMAS DD structure including:

    \b
    - DDVersion nodes with version tracking (PREDECESSOR relationships)
    - IDS nodes for top-level structures (core_profiles, equilibrium, etc.)
    - IMASNode nodes with hierarchical relationships (PARENT, IDS)
    - Unit nodes with HAS_UNIT relationships
    - IMASCoordinateSpec nodes with HAS_COORDINATE relationships
    - IMASNodeChange nodes for metadata evolution between versions
    - RENAMED_TO relationships for path migrations
    - HAS_ERROR relationships linking data paths to error fields
    - LLM-enriched descriptions for semantic understanding (ON by default)
    - Vector embeddings for semantic search (current version only)
    - IMASSemanticCluster nodes with centroids for cluster-based search

    \b
    Examples:
        imas-codex imas dd build                  # Build all DD versions (default)
        imas-codex imas dd build --current-only   # Build current version only
        imas-codex imas dd build --from-version 4.0.0  # Incremental from 4.0.0
        imas-codex imas dd build --reset-to extracted  # Full rebuild from DD XML
        imas-codex imas dd build --reset-to built      # Re-enrich (prompt/model change)
        imas-codex imas dd build --reset-to enriched   # Re-refine only
        imas-codex imas dd build --reset-to refined    # Re-embed (embedding model change)
        imas-codex imas dd build --model openrouter/anthropic/claude-sonnet-4.6  # Override model
        imas-codex imas dd build --dry-run -v     # Preview without writing
        imas-codex imas dd build --ids-filter "core_profiles equilibrium"  # Test subset
        imas-codex imas dd build --force          # Full rebuild, no skips, no hash matches
    """
    # On air-gapped nodes, prevent LiteLLM import-time remote fetches
    from imas_codex.discovery.base.llm import set_litellm_offline_env

    set_litellm_offline_env()

    from imas_codex import dd_version as current_dd_version
    from imas_codex.cli.discover.common import (
        DiscoveryConfig,
        make_log_print,
        run_discovery,
        setup_logging,
        use_rich_output,
    )
    from imas_codex.graph.build_dd import get_all_dd_versions

    use_rich = use_rich_output()
    console_obj = setup_logging("imas_dd", "dd", use_rich, verbose=verbose)
    log_print = make_log_print("imas_dd", console_obj)

    # Suppress imas library's verbose logging
    logging.getLogger("imas").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    try:
        # Determine versions to process
        available_versions = get_all_dd_versions()

        if current_only:
            versions = [current_dd_version]
        elif from_version:
            try:
                start_idx = available_versions.index(from_version)
                versions = available_versions[start_idx:]
            except ValueError as e:
                click.echo(f"Error: Unknown version {from_version}", err=True)
                click.echo(
                    f"Available: {', '.join(available_versions[:5])}...", err=True
                )
                raise SystemExit(1) from e
        else:
            versions = available_versions

        # Parse IDS filter
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())

        log_print("\n[bold]IMAS DD Build[/bold]")
        log_print(f"  Versions: {len(versions)} ({versions[0]} → {versions[-1]})")
        if ids_set:
            log_print(f"  IDS filter: {sorted(ids_set)}")
        if model:
            log_print(f"  Model: {model}")
        if dry_run:
            log_print("  Mode: dry run")
        if force:
            reset_to = "extracted"
            log_print("  Mode: [bold red]FORCE[/bold red] (full rebuild, no skips)")
        if reset_to:
            log_print(f"  Reset: nodes → {reset_to}")
        log_print("")

        # Execute reset before building
        if reset_to and not dry_run:
            from imas_codex.graph.dd_graph_ops import reset_imas_nodes

            reset_count = reset_imas_nodes(reset_to, ids_filter=ids_set)
            if reset_count > 0:
                log_print(
                    f"  [yellow]Reset {reset_count:,} nodes to '{reset_to}'[/yellow]"
                )
            else:
                log_print(f"  [dim]No nodes to reset to '{reset_to}'[/dim]")

        # Build state
        from imas_codex.graph.dd_workers import DDBuildState, run_dd_build_engine

        state = DDBuildState(
            facility="imas",
            cost_limit=100.0,  # DD enrichment is bounded by path count, not budget
            versions=versions,
            ids_filter=ids_set,
            dry_run=dry_run,
            reset_to=reset_to,
            force=force,
            model=model,
        )

        # Build display for rich mode
        display = None
        if use_rich:
            from imas_codex.graph.dd_progress import create_dd_build_display

            display = create_dd_build_display(
                state,
                console=console_obj,
                mode_label="DRY RUN" if dry_run else None,
            )
            display.set_engine_state(state)

        disc_config = DiscoveryConfig(
            domain="imas_dd",
            facility="imas",
            facility_config={},
            display=display,
            check_graph=True,
            check_embed=True,
            check_ssh=False,
            check_auth=False,
            check_model=True,
            suppress_loggers=[
                "imas_codex.embeddings",
                "imas_codex.graph.build_dd",
                "imas_codex.graph.dd_enrichment",
                "imas",
            ],
            verbose=verbose,
            force_exit_on_complete=use_rich and "PYTEST_CURRENT_TEST" not in os.environ,
        )

        async def async_main(stop_event, service_monitor):
            if service_monitor:
                state.service_monitor = service_monitor
            await run_dd_build_engine(
                state,
                stop_event=stop_event,
                on_worker_status=(display.update_worker_status if display else None),
            )
            return state.stats

        def _on_complete(results):
            if not results:
                return
            stats = results
            if stats.get("skipped"):
                log_print("\n[yellow]Build skipped (graph already up-to-date)[/yellow]")
                return

        result = run_discovery(disc_config, async_main, on_complete=_on_complete)

        # Plain-mode summary (rich mode shows via display)
        if not use_rich and result:
            _print_build_summary(result)

    except SystemExit:
        raise
    except TypeError as e:
        if "unexpected keyword argument" in str(e):
            click.echo(
                f"Error: {e}\n\n"
                "This usually means the installed package is out of sync "
                "with the source code.\n"
                "Run 'uv sync' to update, then retry.",
                err=True,
            )
        else:
            click.echo(f"Error: {e}", err=True)
        if verbose:
            logger.exception("Full traceback:")
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(f"Error building DD graph: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


def _print_build_summary(stats: dict) -> None:
    """Print plain-text build summary to stdout."""
    if stats.get("skipped"):
        click.echo("\n=== Build Skipped (graph already up-to-date) ===")
        click.echo(f"Versions verified: {stats['versions_processed']}")
        return

    click.echo("\n=== Build Complete ===")
    click.echo(f"Versions processed: {stats['versions_processed']}")
    click.echo(f"IDS types: {stats['ids_created']}")
    click.echo(f"IMASNode nodes (across all versions): {stats['paths_created']}")
    click.echo(f"Unit nodes: {stats['units_created']}")
    click.echo(f"IMASNodeChange nodes: {stats['path_changes_created']}")
    click.echo(f"Definitions changed (documentation): {stats['definitions_changed']}")
    click.echo(f"Paths enriched (LLM): {stats.get('enriched_llm', 0)}")
    click.echo(f"Paths enriched (template): {stats.get('enriched_template', 0)}")
    click.echo(f"Enrichment cached: {stats.get('enrichment_cached', 0)}")
    if stats.get("identifier_schemas_total", 0):
        click.echo(
            "Identifier schemas enriched: "
            f"{stats.get('identifier_schemas_enriched', 0)} "
            f"(cached: {stats.get('identifier_schemas_cached', 0)})"
        )
    if stats.get("ids_total", 0):
        click.echo(
            f"IDS enriched: {stats.get('ids_enriched', 0)} "
            f"(cached: {stats.get('ids_cached', 0)})"
        )
    if stats.get("enrichment_cost", 0) > 0:
        click.echo(f"Enrichment cost: ${stats['enrichment_cost']:.4f}")
    if stats.get("paths_filtered"):
        click.echo(
            f"Paths excluded from embedding (error/metadata): {stats['paths_filtered']}"
        )
    click.echo(f"HAS_ERROR relationships: {stats.get('error_relationships', 0)}")
    click.echo(f"Embeddings updated: {stats.get('embeddings_updated', 0)}")
    click.echo(f"Embeddings cached: {stats.get('embeddings_cached', 0)}")
    if stats.get("identifier_schemas_total", 0):
        click.echo(
            "Identifier schema embeddings: "
            f"{stats.get('identifier_embeddings_updated', 0)} "
            f"(cached: {stats.get('identifier_embeddings_cached', 0)})"
        )
    if stats.get("ids_total", 0):
        click.echo(
            f"IDS embeddings: {stats.get('ids_embeddings_updated', 0)} "
            f"(cached: {stats.get('ids_embeddings_cached', 0)})"
        )
    click.echo(f"Cluster nodes: {stats['clusters_created']}")
    if stats.get("elapsed_seconds", 0) > 0:
        click.echo(f"Elapsed: {stats['elapsed_seconds']:.1f}s")


@dd.command("status")
@click.option(
    "--version", "-v", "version_filter", help="Show details for specific version"
)
def imas_status(version_filter: str | None) -> None:
    """Show IMAS DD graph statistics.

    Displays summary of DD graph content including version coverage,
    path counts, relationship statistics, and embedding status.

    \b
    Examples:
        imas-codex imas dd status             # Overall summary
        imas-codex imas dd status -v 4.1.0    # Details for specific version
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Get version summary
        versions = gc.query("""
            MATCH (v:DDVersion)
            OPTIONAL MATCH (v)-[:HAS_PREDECESSOR]->(prev:DDVersion)
            RETURN v.id AS version, v.is_current AS is_current, prev.id AS predecessor
            ORDER BY v.id
        """)

        if not versions:
            console.print("[yellow]No DD versions in graph.[/yellow]")
            console.print("Build with: imas-codex imas dd build")
            return

        # Version table
        version_table = Table(title="DD Versions in Graph")
        version_table.add_column("Version", style="cyan")
        version_table.add_column("Current", justify="center")
        version_table.add_column("Predecessor")

        for v in versions:
            current = "✓" if v["is_current"] else ""
            version_table.add_row(v["version"], current, v["predecessor"] or "—")

        console.print(version_table)

        # Get detailed stats for specific version or overall
        if version_filter:
            stats = gc.query(
                """
                MATCH (p:IMASNode)-[:INTRODUCED_IN]->(v:DDVersion {id: $version})
                WITH count(p) AS paths
                OPTIONAL MATCH (p2:IMASNode)-[:INTRODUCED_IN]->(:DDVersion {id: $version})
                WHERE p2.embedding IS NOT NULL
                RETURN paths, count(p2) AS with_embeddings
            """,
                version=version_filter,
            )

            if stats:
                console.print(f"\n[bold]Version {version_filter}:[/bold]")
                console.print(f"  Paths introduced: {stats[0]['paths']}")
                console.print(f"  With embeddings: {stats[0]['with_embeddings']}")
        else:
            # Overall stats
            overall = gc.query("""
                MATCH (p:IMASNode) WITH count(p) AS total_paths
                MATCH (i:IDS) WITH total_paths, count(i) AS ids_count
                MATCH (u:Unit) WITH total_paths, ids_count, count(u) AS unit_count
                MATCH (c:IMASCoordinateSpec) WITH total_paths, ids_count, unit_count, count(c) AS coord_count
                MATCH (p2:IMASNode) WHERE p2.embedding IS NOT NULL
                WITH total_paths, ids_count, unit_count, coord_count, count(p2) AS with_embeddings
                OPTIONAL MATCH (:IMASNode)-[r:HAS_UNIT]->(:Unit)
                WITH total_paths, ids_count, unit_count, coord_count, with_embeddings, count(r) AS unit_rels
                OPTIONAL MATCH (:IMASNode)-[r2:HAS_COORDINATE]->()
                RETURN total_paths, ids_count, unit_count, coord_count, with_embeddings, unit_rels, count(r2) AS coord_rels
            """)

            if overall:
                s = overall[0]
                stats_table = Table(title="Graph Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Count", justify="right")

                stats_table.add_row("IMASNode nodes", str(s["total_paths"]))
                stats_table.add_row("IDS nodes", str(s["ids_count"]))
                stats_table.add_row("Unit nodes", str(s["unit_count"]))
                stats_table.add_row("IMASCoordinateSpec nodes", str(s["coord_count"]))
                stats_table.add_row("Paths with embeddings", str(s["with_embeddings"]))
                stats_table.add_row("HAS_UNIT relationships", str(s["unit_rels"]))
                stats_table.add_row(
                    "HAS_COORDINATE relationships", str(s["coord_rels"])
                )

                console.print()
                console.print(stats_table)

            # Cluster stats
            cluster_stats = gc.query("""
                MATCH (c:IMASSemanticCluster)
                WITH count(c) AS total
                OPTIONAL MATCH (c2:IMASSemanticCluster)
                WHERE c2.label IS NOT NULL
                WITH total, count(c2) AS with_labels
                OPTIONAL MATCH (c3:IMASSemanticCluster)
                WHERE c3.embedding IS NOT NULL
                RETURN total, with_labels, count(c3) AS with_embeddings
            """)
            if cluster_stats and cluster_stats[0]["total"] > 0:
                cs = cluster_stats[0]
                console.print(
                    f"\nIMASSemanticCluster nodes: {cs['total']} "
                    f"({cs['with_labels']} labeled, "
                    f"{cs['with_embeddings']} with embeddings)"
                )

            # IMASNodeChange stats
            change_stats = gc.query("""
                MATCH (pc:IMASNodeChange)
                WITH count(pc) AS total
                OPTIONAL MATCH (pc2:IMASNodeChange {change_type: 'documentation'})
                RETURN total, count(pc2) AS definitions_changed
            """)
            if change_stats and change_stats[0]["total"] > 0:
                cs = change_stats[0]
                console.print(
                    f"IMASNodeChange nodes: {cs['total']} "
                    f"({cs['definitions_changed']} definition changes)"
                )


@dd.command("search")
@click.argument("query")
@click.option("-n", "--limit", default=10, help="Max results (default: 10)")
@click.option("--ids", help="Filter to specific IDS")
@click.option("--version", "-v", "version_filter", help="Filter to DD version")
@click.option(
    "--include-deprecated",
    is_flag=True,
    help="Include deprecated paths in search results (default: active only)",
)
def imas_search(
    query: str,
    limit: int,
    ids: str | None,
    version_filter: str | None,
    include_deprecated: bool,
) -> None:
    """Semantic search for IMAS paths.

    Uses vector embeddings to find paths matching natural language queries.
    By default, only active (non-deprecated) paths are returned.

    \b
    Examples:
        imas-codex imas dd search "electron temperature"
        imas-codex imas dd search "magnetic field boundary" --ids equilibrium
        imas-codex imas dd search "plasma current" -n 20
        imas-codex imas dd search "plasma current" --include-deprecated
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph import GraphClient

    config = EncoderConfig(normalize_embeddings=True, use_rich=False)
    encoder = Encoder(config=config)
    embedding = encoder.embed_texts([query])[0].tolist()

    # Build all WHERE conditions as post-filters after SCORE AS score
    all_where: list[str] = ["node.node_category IN $searchable_categories"]
    if ids:
        all_where.append(f"node.id STARTS WITH '{ids}/'")
    if not include_deprecated:
        all_where.append("NOT (node)-[:DEPRECATED_IN]->(:DDVersion)")

    # Build version filter using relationship-based INTRODUCED_IN
    dd_version_join = ""
    extra_params: dict[str, Any] = {}
    if version_filter:
        dd_major = int(version_filter.split(".")[0])
        dd_version_join = (
            "WITH node, score "
            "MATCH (node)-[:INTRODUCED_IN]->(iv:DDVersion) "
            "WHERE toInteger(split(iv.id, '.')[0]) <= $dd_version "
        )
        extra_params["dd_version"] = dd_major

    search_block = (
        "CYPHER 25\n"
        "MATCH (node:IMASNode)\n"
        "SEARCH node IN (\n"
        "  VECTOR INDEX imas_node_embedding\n"
        "  FOR $embedding\n"
        "  LIMIT $search_k\n"
        f") SCORE AS score\n"
        f"WHERE {' AND '.join(all_where)}"
    )

    with GraphClient() as gc:
        results = gc.query(
            f"""
            {search_block}
            {dd_version_join}
            RETURN node.id AS path, score, node.unit AS unit, node.documentation AS doc
            LIMIT $limit
        """,
            embedding=embedding,
            limit=limit,
            search_k=limit * 2,
            searchable_categories=list(SEARCHABLE_CATEGORIES),
            **extra_params,
        )

    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        return

    table = Table(title=f"Search: '{query}'")
    table.add_column("Score", style="dim", width=6)
    table.add_column("Path", style="cyan")
    table.add_column("Units", width=8)

    for r in results:
        units = r["unit"] or ""
        table.add_row(f"{r['score']:.3f}", r["path"], units)

    console.print(table)


def _resolve_version(version_spec: str) -> str:
    """Resolve incomplete version specification to a full version."""
    from imas_codex.graph.build_dd import get_all_dd_versions

    all_versions = get_all_dd_versions()

    # If exact match, return it
    if version_spec in all_versions:
        return version_spec

    # Parse the version spec
    parts = version_spec.split(".")
    if len(parts) == 1:
        # Major only (e.g., "4") - find latest in that major
        major = parts[0]
        matching = [v for v in all_versions if v.startswith(f"{major}.")]
        if matching:
            return matching[-1]  # sorted, so last is highest
    elif len(parts) == 2:
        # Major.minor (e.g., "4.1") - find latest patch in that minor
        prefix = f"{parts[0]}.{parts[1]}."
        matching = [v for v in all_versions if v.startswith(prefix)]
        if matching:
            return matching[-1]

    raise ValueError(f"No DD version matching '{version_spec}'")


@dd.command("version")
@click.argument("version", required=False)
@click.option(
    "--available",
    "-a",
    is_flag=True,
    help="Show all available versions from imas-python",
)
@click.option(
    "--list",
    "-l",
    "list_versions",
    is_flag=True,
    help="List all versions in the graph",
)
def imas_version(version: str | None, available: bool, list_versions: bool) -> None:
    """Show DD version info (defaults to latest/current version).

    Without arguments, shows details for the current (latest) DD version.
    Incomplete version numbers are resolved to the latest matching version.

    \b
    Examples:
        imas-codex imas dd version              # Show current version details
        imas-codex imas dd version 4            # Show latest 4.x version
        imas-codex imas dd version 4.0.0        # Show specific version
        imas-codex imas dd version --list       # List all versions in graph
        imas-codex imas dd version --available  # All available versions
    """
    if available:
        from imas_codex.graph.build_dd import get_all_dd_versions

        versions_list = get_all_dd_versions()
        console.print(f"[bold]Available DD versions ({len(versions_list)}):[/bold]")
        # Group by major version
        major_groups: dict[str, list[str]] = {}
        for v in versions_list:
            major = v.split(".")[0]
            major_groups.setdefault(major, []).append(v)

        for major, vers in sorted(major_groups.items()):
            console.print(f"  {major}.x: {', '.join(vers)}")
        return

    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        if list_versions:
            _show_versions_summary(gc)
        elif version:
            # Resolve incomplete version and show details
            try:
                resolved = _resolve_version(version)
                if resolved != version:
                    console.print(f"[dim]Resolved '{version}' → {resolved}[/dim]")
                _show_version_details(gc, resolved)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
        else:
            # Default: show details for current version
            current = gc.query(
                "MATCH (v:DDVersion {is_current: true}) RETURN v.id AS version"
            )
            if current:
                _show_version_details(gc, current[0]["version"])
            else:
                console.print("[yellow]No current version set in graph.[/yellow]")
                console.print("Build with: imas-codex imas dd build")


def _show_version_details(gc, version: str) -> None:
    """Show detailed statistics for a specific DD version."""
    # Check version exists
    check = gc.query(
        "MATCH (v:DDVersion {id: $version}) RETURN v",
        version=version,
    )
    if not check:
        console.print(f"[red]Version {version} not found in graph.[/red]")
        return

    # Get version metadata
    meta = gc.query(
        """
        MATCH (v:DDVersion {id: $version})
        RETURN v.is_current AS is_current,
               v.embeddings_built_at AS embeddings_built_at,
               v.embeddings_model AS embeddings_model,
               v.embeddings_count AS embeddings_count
        """,
        version=version,
    )[0]

    # Get path statistics
    stats = gc.query(
        """
        MATCH (v:DDVersion {id: $version})
        OPTIONAL MATCH (introduced:IMASNode)-[:INTRODUCED_IN]->(v)
        WITH v, count(introduced) AS paths_introduced
        OPTIONAL MATCH (deprecated:IMASNode)-[:DEPRECATED_IN]->(v)
        WITH v, paths_introduced, count(deprecated) AS paths_deprecated
        OPTIONAL MATCH (embedded:IMASNode)-[:INTRODUCED_IN]->(v)
        WHERE embedded.embedding IS NOT NULL
        RETURN paths_introduced, paths_deprecated, count(embedded) AS paths_embedded
        """,
        version=version,
    )[0]

    # Get IMASNodeChange statistics
    path_changes = gc.query(
        """
        MATCH (pc:IMASNodeChange)-[:IN_VERSION]->(v:DDVersion {id: $version})
        RETURN pc.change_type AS change_type, count(*) AS count
        ORDER BY count DESC
        LIMIT 10
        """,
        version=version,
    )

    # Display version header
    current_marker = " [green](current)[/green]" if meta["is_current"] else ""
    console.print(f"\n[bold]DD Version {version}{current_marker}[/bold]\n")

    # Path statistics table
    table = Table(title="Path Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")

    table.add_row("Paths introduced", str(stats["paths_introduced"]))
    table.add_row("Paths deprecated", str(stats["paths_deprecated"]))
    table.add_row("Paths embedded", str(stats["paths_embedded"]))

    console.print(table)

    # IMASNodeChange statistics if any
    if path_changes:
        pc_table = Table(title="Metadata Changes", show_header=True)
        pc_table.add_column("Change Type", style="magenta")
        pc_table.add_column("Count", justify="right")

        for pc in path_changes:
            pc_table.add_row(pc["change_type"], str(pc["count"]))

        console.print(pc_table)

    # Embeddings metadata
    if meta["embeddings_built_at"]:
        console.print(
            f"\n[dim]Embeddings: {meta['embeddings_count']} paths, "
            f"model: {meta['embeddings_model']}, "
            f"built: {meta['embeddings_built_at']}[/dim]"
        )


def _show_versions_summary(gc) -> None:
    """Show summary of all DD versions in graph."""
    # Count paths INTRODUCED_IN each version
    versions = gc.query("""
        MATCH (v:DDVersion)
        OPTIONAL MATCH (p:IMASNode)-[:INTRODUCED_IN]->(v)
        WITH v, count(p) AS introduced
        OPTIONAL MATCH (p2:IMASNode)-[:INTRODUCED_IN]->(v) WHERE p2.embedding IS NOT NULL
        RETURN v.id AS version, v.is_current AS is_current, introduced, count(p2) AS embedded
        ORDER BY v.id
    """)

    if not versions:
        console.print("[yellow]No versions in graph.[/yellow]")
        console.print("Build with: imas-codex imas dd build")
        return

    console.print("[bold]DD versions in graph:[/bold]")
    for v in versions:
        current = " [green](current)[/green]" if v["is_current"] else ""
        embedded = f", {v['embedded']} embedded" if v["embedded"] else ""
        console.print(
            f"  {v['version']}: {v['introduced']} paths introduced{embedded}{current}"
        )


@dd.command("clear")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--dump-first",
    is_flag=True,
    help="Create a graph dump before clearing (recommended)",
)
def imas_clear(force: bool, dump_first: bool) -> None:
    """Delete all IMAS Data Dictionary nodes from the graph.

    Removes ALL DD-specific nodes: DDVersion, IDS, IMASNode, IMASNodeChange,
    IMASSemanticCluster, IMASCoordinateSpec, IdentifierSchema, and
    EmbeddingChange. Orphaned Unit nodes are also cleaned up.

    Cross-references from facility nodes (IMASMapping, MENTIONS_IMAS) are
    detached before deletion. DD-specific vector indexes are dropped.

    \b
    Examples:
      imas-codex imas dd clear             # Interactive confirmation
      imas-codex imas dd clear --force     # Skip confirmation
      imas-codex imas dd clear --dump-first  # Backup before clearing
    """
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.graph import GraphClient
    from imas_codex.graph.build_dd import clear_dd_graph

    configure_cli_logging("imas_dd")

    with GraphClient() as gc:
        # Count nodes that will be deleted
        counts = gc.query("""
            OPTIONAL MATCH (p:IMASNode)
            WITH count(p) AS paths
            OPTIONAL MATCH (v:DDVersion)
            WITH paths, count(v) AS versions
            OPTIONAL MATCH (i:IDS)
            WITH paths, versions, count(i) AS ids
            OPTIONAL MATCH (c:IMASSemanticCluster)
            WITH paths, versions, ids, count(c) AS clusters
            OPTIONAL MATCH (ch:IMASNodeChange)
            RETURN paths, versions, ids, clusters, count(ch) AS changes
        """)

        if not counts or counts[0]["paths"] == 0:
            console.print("[yellow]No DD nodes found in graph[/yellow]")
            return

        c = counts[0]
        console.print("[bold red]This will delete:[/bold red]")
        console.print(f"  {c['paths']:,} IMASNode nodes")
        console.print(f"  {c['versions']} DDVersion nodes")
        console.print(f"  {c['ids']} IDS nodes")
        console.print(f"  {c['clusters']:,} IMASSemanticCluster nodes")
        console.print(f"  {c['changes']:,} IMASNodeChange nodes")
        console.print("  + IMASCoordinateSpec, IdentifierSchema, orphaned Units")
        console.print("  + DD vector indexes (imas_node_embedding, cluster_embedding)")

        if not force:
            click.confirm(
                "\nThis action is irreversible. Continue?",
                abort=True,
            )

        if dump_first:
            console.print("[dim]Creating graph dump...[/dim]")
            try:
                import subprocess

                result = subprocess.run(
                    ["uv", "run", "imas-codex", "neo4j", "dump"],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    console.print("[green]Dump created successfully[/green]")
                else:
                    raise RuntimeError(result.stderr.strip() or "dump failed")
            except Exception as e:
                console.print(f"[red]Dump failed: {e}[/red]")
                if not force:
                    click.confirm("Continue without dump?", abort=True)

        console.print("[dim]Clearing DD graph...[/dim]")
        results = clear_dd_graph(gc)

        # Display results
        table = Table(title="DD Graph Cleared")
        table.add_column("Node Type", style="cyan")
        table.add_column("Deleted", justify="right", style="red")

        label_map = {
            "paths": "IMASNode",
            "versions": "DDVersion",
            "ids_nodes": "IDS",
            "clusters": "IMASSemanticCluster",
            "path_changes": "IMASNodeChange",
            "embedding_changes": "EmbeddingChange",
            "identifier_schemas": "IdentifierSchema",
            "coordinate_specs": "IMASCoordinateSpec",
            "orphaned_units": "Unit (orphaned)",
        }

        total = 0
        for key, label in label_map.items():
            count = results.get(key, 0)
            if count > 0:
                table.add_row(label, f"{count:,}")
                total += count

        table.add_row("[bold]Total[/bold]", f"[bold]{total:,}[/bold]")
        console.print(table)
        console.print("[green]DD graph cleared successfully[/green]")


@dd.command("path-history")
@click.argument("path")
@click.option(
    "--type",
    "-t",
    "change_type",
    help="Filter by change type (documentation, units, data_type, node_type)",
)
def imas_path_history(path: str, change_type: str | None) -> None:
    """Show all definition changes for a specific IMAS path.

    Displays the version-by-version history of metadata changes including
    documentation, units, data_type, and node_type modifications.

    \b
    Examples:
        imas-codex imas dd path-history core_profiles/profiles_1d/electrons/temperature
        imas-codex imas dd path-history equilibrium/time_slice/boundary/psi -t documentation
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Check the path exists
        exists = gc.query(
            "MATCH (p:IMASNode {id: $path}) RETURN p.id AS id",
            path=path,
        )
        if not exists:
            console.print(f"[red]Path not found: {path}[/red]")
            # Suggest similar paths
            suggestions = gc.query(
                """
                MATCH (p:IMASNode)
                WHERE p.id CONTAINS $fragment
                RETURN p.id AS id
                LIMIT 5
                """,
                fragment=path.split("/")[-1],
            )
            if suggestions:
                console.print("[dim]Did you mean:[/dim]")
                for s in suggestions:
                    console.print(f"  {s['id']}")
            return

        # Build query with optional type filter
        where = "WHERE pc.change_type = $change_type" if change_type else ""
        params = {"path": path}
        if change_type:
            params["change_type"] = change_type

        changes = gc.query(
            f"""
            MATCH (pc:IMASNodeChange)-[:FOR_IMAS_PATH]->(p:IMASNode {{id: $path}})
            MATCH (pc)-[:IN_VERSION]->(v:DDVersion)
            {where}
            RETURN v.id AS version, pc.change_type AS change_type,
                   pc.old_value AS old_value, pc.new_value AS new_value,
                   pc.semantic_type AS semantic_type,
                   pc.keywords_detected AS keywords_detected
            ORDER BY v.id
            """,
            **params,
        )

        # Get introduction version
        intro = gc.query(
            """
            MATCH (p:IMASNode {id: $path})-[:INTRODUCED_IN]->(v:DDVersion)
            RETURN v.id AS version
            """,
            path=path,
        )

        # Get deprecation version if any
        depr = gc.query(
            """
            MATCH (p:IMASNode {id: $path})-[:DEPRECATED_IN]->(v:DDVersion)
            RETURN v.id AS version
            """,
            path=path,
        )

        console.print(f"\n[bold]History for [cyan]{path}[/cyan][/bold]\n")

        if intro:
            console.print(f"Introduced in: [green]{intro[0]['version']}[/green]")
        if depr:
            console.print(f"Deprecated in: [red]{depr[0]['version']}[/red]")

        if not changes:
            console.print("\n[dim]No metadata changes recorded.[/dim]")
            return

        console.print(f"\n[bold]{len(changes)} change(s):[/bold]\n")

        table = Table(show_header=True)
        table.add_column("Version", style="cyan", width=8)
        table.add_column("Field", style="magenta", width=15)
        table.add_column("Old Value", style="red", max_width=40, overflow="fold")
        table.add_column("New Value", style="green", max_width=40, overflow="fold")
        table.add_column("Semantic", style="yellow", width=25)

        for c in changes:
            semantic = c["semantic_type"] or ""
            if c["keywords_detected"]:
                try:
                    kw = json.loads(c["keywords_detected"])
                    if kw:
                        semantic += f" ({', '.join(kw)})"
                except (json.JSONDecodeError, TypeError):
                    pass

            # Truncate long values for display
            old = (c["old_value"] or "")[:120]
            new = (c["new_value"] or "")[:120]
            table.add_row(c["version"], c["change_type"], old, new, semantic)

        console.print(table)


# --- IDS Assembly commands (registered directly on imas group) ---


@imas.command("list")
@click.argument("facility", required=False)
def ids_list(facility: str | None) -> None:
    """List available IDS recipes."""
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.ids.assembler import list_recipes

    configure_cli_logging("ids", facility=facility or "all")

    recipes = list_recipes(facility)
    if not recipes:
        click.echo("No recipes found.")
        return

    click.echo(f"{'Facility':<12} {'IDS':<20} {'DD Version':<12} {'Source':<8}")
    click.echo("-" * 52)
    for r in recipes:
        click.echo(
            f"{r['facility']:<12} {r['ids_name']:<20} "
            f"{r['dd_version']:<12} {r.get('source', 'yaml'):<8}"
        )


@imas.command("show")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--epoch", "-e", help="Epoch to show summary for (e.g., p68613).")
def ids_show(facility: str, ids_name: str, epoch: str | None) -> None:
    """Show IDS mapping details and available epochs."""
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.ids.assembler import IDSAssembler

    configure_cli_logging("ids", facility=facility)

    assembler = IDSAssembler(facility, ids_name)

    click.echo(f"IDS: {ids_name}")
    click.echo(f"Facility: {facility}")
    click.echo(f"DD version: {assembler.recipe['dd_version']}")
    if provider := assembler.recipe.get("provider"):
        click.echo(f"Provider: {provider}")
    click.echo()

    # List epochs
    epochs = assembler.list_epochs()
    if epochs:
        click.echo(f"Available epochs ({len(epochs)}):")
        for ep in epochs:
            shot_range = f"shots {ep['first_shot']}"
            if ep.get("last_shot"):
                shot_range += f"-{ep['last_shot']}"
            else:
                shot_range += "+"
            desc = ep.get("description", "")
            epoch_short = ep["id"].split(":")[-1]
            click.echo(f"  {epoch_short:<12} {shot_range:<20} {desc}")
    else:
        click.echo("No epochs found in graph.")

    # Show summary for specific epoch
    if epoch:
        click.echo()
        summary = assembler.summary(epoch)
        click.echo(f"Assembly summary for epoch {epoch}:")
        for array_name, stats in summary.get("arrays", {}).items():
            line = f"  {array_name}: {stats['count']} entries"
            if elements := stats.get("total_elements"):
                line += f" ({elements} total elements)"
            click.echo(line)


@imas.command("export")
@click.argument("facility")
@click.argument("ids_name")
@click.option("--epoch", "-e", required=True, help="Epoch version (e.g., p68613).")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path (without extension). Default: {facility}_{ids_name}_{epoch}.",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["hdf5", "netcdf"]),
    default="hdf5",
    help="Storage backend.",
)
def ids_export(
    facility: str,
    ids_name: str,
    epoch: str,
    output: str | None,
    backend: str,
) -> None:
    """Assemble and export an IDS to file."""
    from pathlib import Path

    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.ids.assembler import IDSAssembler

    configure_cli_logging("ids", facility=facility)

    if output is None:
        output = f"{facility}_{ids_name}_{epoch}"

    assembler = IDSAssembler(facility, ids_name)

    # Show summary first
    summary = assembler.summary(epoch)
    for array_name, stats in summary.get("arrays", {}).items():
        line = f"  {array_name}: {stats['count']} entries"
        if elements := stats.get("total_elements"):
            line += f" ({elements} total elements)"
        click.echo(line)

    # Export
    out_path = assembler.export(Path(output), epoch, backend=backend)
    click.echo(f"Exported {ids_name} to {out_path}")


@imas.command("epochs")
@click.argument("facility")
def ids_epochs(facility: str) -> None:
    """List structural epochs for a facility."""
    from imas_codex.cli.logging import configure_cli_logging
    from imas_codex.graph.client import GraphClient

    configure_cli_logging("ids", facility=facility)

    with GraphClient() as gc:
        epochs = list(
            gc.query(
                """
                MATCH (se:SignalEpoch {facility_id: $facility})
                OPTIONAL MATCH (se)<-[:INTRODUCED_IN]-(dn:SignalNode)
                WITH se, count(dn) AS node_count
                RETURN se.id AS id,
                       se.first_shot AS first_shot,
                       se.last_shot AS last_shot,
                       se.description AS description,
                       se.data_source_name AS source,
                       node_count
                ORDER BY se.first_shot
                """,
                facility=facility,
            )
        )

    if not epochs:
        click.echo(f"No epochs found for {facility}.")
        return

    click.echo(f"{'Epoch':<35} {'Shots':<20} {'Nodes':>6}  {'Description'}")
    click.echo("-" * 90)
    for ep in epochs:
        epoch_id = ep["id"]
        shot_range = str(ep.get("first_shot", "?"))
        if ep.get("last_shot"):
            shot_range += f"-{ep['last_shot']}"
        else:
            shot_range += "+"
        desc = ep.get("description", "")[:40]
        click.echo(f"{epoch_id:<35} {shot_range:<20} {ep['node_count']:>6}  {desc}")
