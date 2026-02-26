"""Clusters commands: Build, label, sync, and status for semantic clusters."""

from __future__ import annotations

import logging

import click

logger = logging.getLogger(__name__)


@click.group()
def clusters() -> None:
    """Manage semantic clusters of IMAS data paths.

    \b
      imas-codex imas clusters build    Build HDBSCAN clusters from embeddings
      imas-codex imas clusters label    Generate LLM labels for clusters
      imas-codex imas clusters sync     Sync clusters to Neo4j graph
      imas-codex imas clusters status   Show cluster statistics
    """
    pass


@clusters.command("build")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("-f", "--force", is_flag=True, help="Force rebuild even if files exist")
@click.option(
    "--min-cluster-size",
    type=int,
    default=2,
    help="Minimum cluster size for HDBSCAN (default: 2)",
)
@click.option(
    "--min-samples",
    type=int,
    default=2,
    help="Minimum samples for HDBSCAN core points (default: 2)",
)
@click.option(
    "--cluster-method",
    type=click.Choice(["eom", "leaf"]),
    default="eom",
    help="HDBSCAN cluster selection method: 'eom' for broader, 'leaf' for finer",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include (space-separated)",
)
def clusters_build(
    verbose: bool,
    quiet: bool,
    force: bool,
    min_cluster_size: int,
    min_samples: int,
    cluster_method: str,
    ids_filter: str | None,
) -> None:
    """Build semantic clusters of IMAS data paths using HDBSCAN.

    This command creates clusters based on semantic embeddings of path
    documentation. It does NOT generate LLM labels - use 'clusters label'
    for that step.

    \b
    Examples:
      imas-codex imas clusters build                    # Build with defaults
      imas-codex imas clusters build -v -f              # Force rebuild, verbose
      imas-codex imas clusters build --ids-filter "core_profiles equilibrium"
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

    from imas_codex.core.clusters import Clusters
    from imas_codex.embeddings.config import EncoderConfig

    ids_set = set(ids_filter.split()) if ids_filter else None

    if ids_set:
        click.echo(f"Building clusters for IDS: {sorted(ids_set)}")
    else:
        click.echo("Building clusters for all IDS...")

    try:
        encoder_config = EncoderConfig(
            ids_set=ids_set,
            use_rich=not quiet,
        )
        clusters_manager = Clusters(encoder_config=encoder_config)
        output_file = clusters_manager.file_path

        should_build = force or not output_file.exists()
        if not should_build and clusters_manager.needs_rebuild():
            should_build = True
            click.echo("Dependencies changed, rebuilding...")

        if should_build:
            config_overrides = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": cluster_method,
                "use_rich": not quiet,
            }
            clusters_manager.build(force=force, **config_overrides)
            click.echo(f"✓ Built clusters: {output_file}")
        else:
            click.echo(f"Clusters already exist: {output_file}")

    except Exception as e:
        click.echo(f"Error building clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


def _sync_labels_to_graph(label_cache, cluster_data: list[dict], click) -> None:
    """Write cached labels directly to graph cluster nodes.

    Matches clusters by content hash and updates label/description
    on IMASSemanticCluster nodes. The graph is the source of truth
    for labels — this replaces the old _sync_labels_from_cache flow.
    """
    from imas_codex.graph.build_dd import _compute_cluster_content_hash

    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
    except Exception as e:
        logger.debug("Graph not available for label sync: %s", e)
        click.echo("⚠ Graph not available — labels saved to cache only")
        return

    label_batch = []
    for cluster in cluster_data:
        paths = sorted(cluster.get("paths", []))
        cached = label_cache.get_label(paths)
        if not cached:
            continue

        scope = cluster.get("scope", "global")
        cluster_id = _compute_cluster_content_hash(paths, scope)
        label_batch.append(
            {
                "id": cluster_id,
                "label": cached.label,
                "description": cached.description,
            }
        )

    if not label_batch:
        return

    for i in range(0, len(label_batch), 1000):
        batch = label_batch[i : i + 1000]
        client.query(
            """
            UNWIND $batch AS b
            MATCH (c:IMASSemanticCluster {id: b.id})
            SET c.label = b.label, c.description = b.description
            """,
            batch=batch,
        )

    click.echo(f"✓ Synced {len(label_batch)} labels to graph")


@clusters.command("label")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("-f", "--force", is_flag=True, help="Force regenerate all labels")
@click.option(
    "--cost-limit",
    type=float,
    default=10.0,
    help="Maximum cost in USD for LLM requests (default: $10)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Number of clusters per LLM batch (default: from settings)",
)
@click.option(
    "--export/--no-export",
    default=True,
    help="Export labels to JSON for version control (default: True)",
)
def clusters_label(
    verbose: bool,
    quiet: bool,
    force: bool,
    cost_limit: float,
    batch_size: int | None,
    export: bool,
) -> None:
    """Generate LLM labels for semantic clusters.

    Uses the configured language model to generate human-readable labels
    and descriptions for each cluster. Labels are cached to avoid
    regenerating existing ones unless --force is used.

    \b
    Examples:
      imas-codex imas clusters label                # Label unlabeled clusters
      imas-codex imas clusters label -f             # Force regenerate all labels
      imas-codex imas clusters label --cost-limit 5 # Limit to $5 USD
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

    import json

    from imas_codex import dd_version
    from imas_codex.clusters.label_cache import LabelCache
    from imas_codex.clusters.labeler import ClusterLabeler
    from imas_codex.resource_path_accessor import ResourcePathAccessor

    try:
        # Load clusters directly from JSON — no dependency checking or
        # auto-rebuild.  The label command only needs cluster data, not
        # embeddings.  Run 'clusters build' separately if clusters are stale.
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        clusters_file = path_accessor.clusters_dir / "clusters.json"

        if not clusters_file.exists():
            click.echo("No clusters found. Run 'clusters build' first.", err=True)
            raise SystemExit(1)

        with clusters_file.open() as f:
            cluster_data = json.load(f).get("clusters", [])
        click.echo(f"Found {len(cluster_data)} clusters")

        # Initialize cache and labeler
        label_cache = LabelCache()
        labeler = ClusterLabeler()

        # Get cached and uncached clusters
        if force:
            uncached = cluster_data
            cached = {}
            click.echo("Force mode: regenerating all labels")
        else:
            cached, uncached = label_cache.get_many(cluster_data)
            click.echo(f"Cached: {len(cached)}, Need labeling: {len(uncached)}")

        if not uncached:
            click.echo("All clusters already labeled")
            if export:
                exported = label_cache.export_labels()
                click.echo(f"Exported {len(exported)} labels to definitions")
                if label_cache.commit_labels():
                    click.echo("\u2713 Auto-committed labels.json")
            return

        # Generate labels with cost tracking
        click.echo(f"Generating labels (cost limit: ${cost_limit:.2f})...")
        labels = labeler.label_clusters(uncached, batch_size=batch_size)

        # Store in cache
        label_tuples = [
            (c.get("paths", []), lbl.label, lbl.description)
            for c, lbl in zip(uncached, labels, strict=False)
            if lbl
        ]
        stored = label_cache.set_many(label_tuples)
        click.echo(f"✓ Cached {stored} new labels")

        # Write labels directly to graph nodes (graph is source of truth)
        _sync_labels_to_graph(label_cache, cluster_data, click)

        # Export to JSON for version control
        if export:
            exported = label_cache.export_labels()
            click.echo(f"✓ Exported {len(exported)} labels to definitions")
            if label_cache.commit_labels():
                click.echo("✓ Auto-committed labels.json")

    except Exception as e:
        click.echo(f"Error labeling clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("sync")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing")
def clusters_sync(verbose: bool, quiet: bool, dry_run: bool) -> None:
    """Sync semantic clusters to Neo4j knowledge graph.

    Creates/updates IMASSemanticCluster nodes and IN_CLUSTER relationships
    linking IMASPath nodes to their clusters.

    \b
    Examples:
      imas-codex imas clusters sync              # Sync clusters to graph
      imas-codex imas clusters sync --dry-run    # Preview without changes
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

    from imas_codex.graph.build_dd import import_semantic_clusters
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()

        if dry_run:
            click.echo("Dry run - previewing cluster sync...")
        else:
            click.echo("Syncing clusters to graph...")

        count = import_semantic_clusters(client, dry_run=dry_run)

        if dry_run:
            click.echo(f"Would sync {count} clusters")
        else:
            click.echo(f"✓ Synced {count} clusters to graph")

    except Exception as e:
        click.echo(f"Error syncing clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("embed")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("-q", "--quiet", is_flag=True, help="Suppress all logging except errors")
def clusters_embed(verbose: bool, quiet: bool) -> None:
    """Embed cluster labels and descriptions for semantic search.

    Generates vector embeddings from LLM-generated label and description
    text. Creates two vector indexes for natural language cluster search.

    Run after 'clusters label' and 'clusters sync'.

    \b
    Examples:
      imas-codex imas clusters embed           # Embed labels + descriptions
      imas-codex imas clusters embed -v        # Verbose output
    """
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

    from imas_codex.graph.build_dd import _embed_cluster_text
    from imas_codex.graph.client import GraphClient

    try:
        with GraphClient() as client:
            click.echo("Embedding cluster labels and descriptions...")
            _embed_cluster_text(client, use_rich=not quiet)
            click.echo("✓ Cluster text embeddings complete")
    except Exception as e:
        click.echo(f"Error embedding clusters: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        raise SystemExit(1) from e


@clusters.command("status")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed statistics")
def clusters_status(verbose: bool) -> None:
    """Show cluster statistics and cache status.

    \b
    Examples:
      imas-codex imas clusters status     # Basic stats
      imas-codex imas clusters status -v  # Detailed stats
    """
    from imas_codex.clusters.label_cache import LabelCache
    from imas_codex.core.clusters import Clusters
    from imas_codex.embeddings.config import EncoderConfig

    try:
        # Cluster file status
        encoder_config = EncoderConfig()
        clusters_manager = Clusters(encoder_config=encoder_config)

        click.echo("=== Cluster Status ===")
        if clusters_manager.is_available():
            cluster_data = clusters_manager.get_clusters()
            click.echo(f"Clusters file: {clusters_manager.file_path}")
            click.echo(f"Total clusters: {len(cluster_data)}")

            if verbose:
                # Count cross-IDS vs intra-IDS
                cross_ids = sum(1 for c in cluster_data if c.get("cross_ids", False))
                click.echo(f"Cross-IDS clusters: {cross_ids}")
                click.echo(f"Intra-IDS clusters: {len(cluster_data) - cross_ids}")
        else:
            click.echo("No clusters file found")

        # Label cache status
        click.echo("\n=== Label Cache ===")
        label_cache = LabelCache()
        stats = label_cache.get_stats()
        click.echo(f"Cache file: {stats['cache_file']}")
        click.echo(f"Total labels: {stats['total_labels']}")
        click.echo(f"Cache size: {stats['cache_size_mb']:.2f} MB")

        if verbose and stats["by_model"]:
            click.echo("Labels by model:")
            for model, count in stats["by_model"].items():
                click.echo(f"  {model}: {count}")

    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        raise SystemExit(1) from e
