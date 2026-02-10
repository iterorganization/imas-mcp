"""IMAS Data Dictionary commands: Build, status, search, version."""

from __future__ import annotations

import logging

import click
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


@click.group()
def imas() -> None:
    """Manage IMAS Data Dictionary graph.

    \b
      imas-codex imas build    Build/update DD graph from imas-python
      imas-codex imas status   Show DD graph statistics
      imas-codex imas search   Semantic search for paths
      imas-codex imas versions List available DD versions
    """
    pass


@imas.command("build")
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
    "-f", "--force", is_flag=True, help="Force regenerate all embeddings (ignore cache)"
)
@click.option(
    "--skip-clusters", is_flag=True, help="Skip importing semantic clusters into graph"
)
@click.option(
    "--skip-embeddings",
    is_flag=True,
    help="Skip embedding generation for current version paths",
)
@click.option(
    "--embedding-model",
    type=str,
    default=None,
    help="Embedding model (defaults to configured model from settings)",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Filter to specific IDS (space-separated, for testing)",
)
@click.option(
    "--dry-run", is_flag=True, help="Preview changes without writing to graph"
)
def imas_build(
    verbose: bool,
    quiet: bool,
    current_only: bool,
    from_version: str | None,
    force: bool,
    skip_clusters: bool,
    skip_embeddings: bool,
    embedding_model: str,
    ids_filter: str | None,
    dry_run: bool,
) -> None:
    """Build the IMAS Data Dictionary Knowledge Graph.

    Populates Neo4j with complete IMAS DD structure including:

    \b
    - DDVersion nodes with version tracking (PREDECESSOR relationships)
    - IDS nodes for top-level structures (core_profiles, equilibrium, etc.)
    - IMASPath nodes with hierarchical relationships (PARENT, IDS)
    - Unit nodes with HAS_UNIT relationships
    - IMASCoordinateSpec nodes with HAS_COORDINATE relationships
    - IMASPathChange nodes for metadata evolution between versions
    - RENAMED_TO relationships for path migrations
    - HAS_ERROR relationships linking data paths to error fields
    - Vector embeddings for semantic search (current version only)
    - IMASSemanticCluster nodes with centroids for cluster-based search

    \b
    Examples:
        imas-codex imas build                  # Build all DD versions (default)
        imas-codex imas build --current-only   # Build current version only
        imas-codex imas build --from-version 4.0.0  # Incremental from 4.0.0
        imas-codex imas build --force          # Regenerate all embeddings
        imas-codex imas build --dry-run -v     # Preview without writing
        imas-codex imas build --ids-filter "core_profiles equilibrium"  # Test subset
    """
    from imas_codex import dd_version as current_dd_version
    from imas_codex.graph.build_dd import build_dd_graph, get_all_dd_versions

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

    # Suppress imas library's verbose logging
    logging.getLogger("imas").setLevel(logging.WARNING)
    # Suppress httpx/httpcore per-request INFO logs (noisy during embedding)
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
            # Default: all versions
            versions = available_versions

        logger.debug(f"Processing {len(versions)} DD versions")
        if len(versions) > 1:
            logger.debug(f"Versions: {versions[0]} → {versions[-1]}")

        # Parse IDS filter
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.debug(f"Filtering to IDS: {sorted(ids_set)}")

        if dry_run:
            click.echo("DRY RUN - no changes will be written to graph")

        # Build graph
        from imas_codex.graph import GraphClient

        with GraphClient() as client:
            stats = build_dd_graph(
                client=client,
                versions=versions,
                ids_filter=ids_set,
                include_clusters=not skip_clusters,
                include_embeddings=not skip_embeddings,
                dry_run=dry_run,
                embedding_model=embedding_model,
                force_embeddings=force,
            )

        # Report results
        click.echo("\n=== Build Complete ===")
        click.echo(f"Versions processed: {stats['versions_processed']}")
        click.echo(f"IDS nodes: {stats['ids_created']}")
        click.echo(f"IMASPath nodes created: {stats['paths_created']}")
        click.echo(f"Unit nodes: {stats['units_created']}")
        click.echo(f"IMASPathChange nodes: {stats['path_changes_created']}")
        if not skip_embeddings:
            click.echo(f"Paths filtered (error/metadata): {stats['paths_filtered']}")
            click.echo(f"HAS_ERROR relationships: {stats['error_relationships']}")
            click.echo(f"Embeddings updated: {stats['embeddings_updated']}")
            click.echo(f"Embeddings cached: {stats['embeddings_cached']}")
            if stats.get("definitions_changed", 0) > 0:
                click.echo(
                    f"Definitions changed: {stats['definitions_changed']} "
                    "(deprecated paths cleaned)"
                )
        if not skip_clusters:
            click.echo(f"Cluster nodes: {stats['clusters_created']}")

    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Error building DD graph: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from e


@imas.command("status")
@click.option(
    "--version", "-v", "version_filter", help="Show details for specific version"
)
def imas_status(version_filter: str | None) -> None:
    """Show IMAS DD graph statistics.

    Displays summary of DD graph content including version coverage,
    path counts, relationship statistics, and embedding status.

    \b
    Examples:
        imas-codex imas status             # Overall summary
        imas-codex imas status -v 4.1.0    # Details for specific version
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        # Get version summary
        versions = gc.query("""
            MATCH (v:DDVersion)
            OPTIONAL MATCH (v)-[:PREDECESSOR]->(prev:DDVersion)
            RETURN v.id AS version, v.is_current AS is_current, prev.id AS predecessor
            ORDER BY v.id
        """)

        if not versions:
            console.print("[yellow]No DD versions in graph.[/yellow]")
            console.print("Build with: imas-codex imas build")
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
                MATCH (p:IMASPath)-[:INTRODUCED_IN]->(v:DDVersion {id: $version})
                WITH count(p) AS paths
                OPTIONAL MATCH (p2:IMASPath)-[:INTRODUCED_IN]->(:DDVersion {id: $version})
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
                MATCH (p:IMASPath) WITH count(p) AS total_paths
                MATCH (i:IDS) WITH total_paths, count(i) AS ids_count
                MATCH (u:Unit) WITH total_paths, ids_count, count(u) AS unit_count
                MATCH (c:IMASCoordinateSpec) WITH total_paths, ids_count, unit_count, count(c) AS coord_count
                MATCH (p2:IMASPath) WHERE p2.embedding IS NOT NULL
                WITH total_paths, ids_count, unit_count, coord_count, count(p2) AS with_embeddings
                OPTIONAL MATCH (:IMASPath)-[r:HAS_UNIT]->(:Unit)
                WITH total_paths, ids_count, unit_count, coord_count, with_embeddings, count(r) AS unit_rels
                OPTIONAL MATCH (:IMASPath)-[r2:HAS_COORDINATE]->()
                RETURN total_paths, ids_count, unit_count, coord_count, with_embeddings, unit_rels, count(r2) AS coord_rels
            """)

            if overall:
                s = overall[0]
                stats_table = Table(title="Graph Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Count", justify="right")

                stats_table.add_row("IMASPath nodes", str(s["total_paths"]))
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
            clusters = gc.query(
                "MATCH (c:IMASSemanticCluster) RETURN count(c) AS count"
            )
            if clusters and clusters[0]["count"] > 0:
                console.print(f"\nIMASSemanticCluster nodes: {clusters[0]['count']}")


@imas.command("search")
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
        imas-codex imas search "electron temperature"
        imas-codex imas search "magnetic field boundary" --ids equilibrium
        imas-codex imas search "plasma current" -n 20
        imas-codex imas search "plasma current" --include-deprecated
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph import GraphClient

    config = EncoderConfig(normalize_embeddings=True, use_rich=False)
    encoder = Encoder(config=config)
    embedding = encoder.embed_texts([query])[0].tolist()

    # Build filter clause
    # By default, exclude deprecated paths unless --include-deprecated flag is used
    where_clauses = []
    if not include_deprecated:
        where_clauses.append("NOT (node)-[:DEPRECATED_IN]->(:DDVersion)")
    if ids:
        where_clauses.append(f"node.id STARTS WITH '{ids}/'")
    if version_filter:
        where_clauses.append(f"node.dd_version = '{version_filter}'")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    with GraphClient() as gc:
        results = gc.query(
            f"""
            CALL db.index.vector.queryNodes("imas_path_embedding", $limit * 2, $embedding)
            YIELD node, score
            {where_clause}
            RETURN node.id AS path, score, node.units AS units, node.documentation AS doc
            LIMIT $limit
        """,
            embedding=embedding,
            limit=limit,
        )

    if not results:
        console.print(f"[yellow]No results for '{query}'[/yellow]")
        return

    table = Table(title=f"Search: '{query}'")
    table.add_column("Score", style="dim", width=6)
    table.add_column("Path", style="cyan")
    table.add_column("Units", width=8)

    for r in results:
        units = r["units"] or ""
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


@imas.command("version")
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
        imas-codex imas version              # Show current version details
        imas-codex imas version 4            # Show latest 4.x version
        imas-codex imas version 4.0.0        # Show specific version
        imas-codex imas version --list       # List all versions in graph
        imas-codex imas version --available  # All available versions
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
                console.print("Build with: imas-codex imas build")


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
        OPTIONAL MATCH (introduced:IMASPath)-[:INTRODUCED_IN]->(v)
        WITH v, count(introduced) AS paths_introduced
        OPTIONAL MATCH (deprecated:IMASPath)-[:DEPRECATED_IN]->(v)
        WITH v, paths_introduced, count(deprecated) AS paths_deprecated
        OPTIONAL MATCH (embedded:IMASPath)-[:INTRODUCED_IN]->(v)
        WHERE embedded.embedding IS NOT NULL
        RETURN paths_introduced, paths_deprecated, count(embedded) AS paths_embedded
        """,
        version=version,
    )[0]

    # Get IMASPathChange statistics
    path_changes = gc.query(
        """
        MATCH (pc:IMASPathChange)-[:VERSION]->(v:DDVersion {id: $version})
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

    # IMASPathChange statistics if any
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
        OPTIONAL MATCH (p:IMASPath)-[:INTRODUCED_IN]->(v)
        WITH v, count(p) AS introduced
        OPTIONAL MATCH (p2:IMASPath)-[:INTRODUCED_IN]->(v) WHERE p2.embedding IS NOT NULL
        RETURN v.id AS version, v.is_current AS is_current, introduced, count(p2) AS embedded
        ORDER BY v.id
    """)

    if not versions:
        console.print("[yellow]No versions in graph.[/yellow]")
        console.print("Build with: imas-codex imas build")
        return

    console.print("[bold]DD versions in graph:[/bold]")
    for v in versions:
        current = " [green](current)[/green]" if v["is_current"] else ""
        embedded = f", {v['embedded']} embedded" if v["embedded"] else ""
        console.print(
            f"  {v['version']}: {v['introduced']} paths introduced{embedded}{current}"
        )
