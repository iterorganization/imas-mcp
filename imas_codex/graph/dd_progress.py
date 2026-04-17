"""Progress display factory for the IMAS DD build pipeline.

Uses ``DataDrivenProgressDisplay`` directly — no custom subclass
needed since each phase has its own worker group.  The base class's
``_count_group_workers()`` and ``_worker_complete()`` work naturally
with one worker per group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imas_codex.core.node_categories import (
    EMBEDDABLE_CATEGORIES,
    ENRICHABLE_CATEGORIES,
)
from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    StageDisplaySpec,
    format_count,
)

if TYPE_CHECKING:
    from imas_codex.graph.dd_workers import DDBuildState


def _build_stats(state: DDBuildState) -> list[tuple[str, str, str]]:
    """Build STATS row entries from live build state."""
    stats: list[tuple[str, str, str]] = []
    s = state.stats
    counts = state.imas_node_status_counts

    versions = s.get("versions_processed", 0)
    if versions:
        stats.append(("versions", str(versions), "blue"))

    paths = s.get("paths_created", 0)
    if paths:
        stats.append(("paths", format_count(paths), "green"))

    enriched = s.get("enriched_llm", 0) + s.get("enriched_template", 0)
    if enriched:
        stats.append(("enriched", format_count(enriched), "green"))

    cached = s.get("enrichment_cached", 0)
    if cached:
        stats.append(("cached", format_count(cached), "dim"))

    embedded = counts.get("embedded", 0) or state.embed_stats.processed
    if embedded <= 0:
        embedded = s.get("embeddings_updated", 0) + s.get("embeddings_cached", 0)
    if embedded:
        stats.append(("embedded", format_count(embedded), "magenta"))

    # Identifier and IDS completion: show a single field each.
    # Use the max of enrichment and embedding counts to avoid showing 0
    # when a prior run completed everything (enrichment returns 0 for
    # both enriched/cached when the WHERE filter finds no work).
    identifier_total = s.get("identifier_schemas_total", 0)
    if identifier_total:
        ident_enriched = s.get("identifier_schemas_enriched", 0) + s.get(
            "identifier_schemas_cached", 0
        )
        ident_embedded = s.get("identifier_embeddings_updated", 0) + s.get(
            "identifier_embeddings_cached", 0
        )
        ident_done = max(ident_enriched, ident_embedded)
        stats.append(
            (
                "identifiers",
                f"{format_count(ident_done)}/{format_count(identifier_total)}",
                "cyan",
            )
        )

    ids_total = s.get("ids_total", 0)
    if ids_total:
        ids_enriched = s.get("ids_enriched", 0) + s.get("ids_cached", 0)
        ids_embedded = s.get("ids_embeddings_updated", 0) + s.get(
            "ids_embeddings_cached", 0
        )
        ids_done = max(ids_enriched, ids_embedded)
        stats.append(
            ("ids", f"{format_count(ids_done)}/{format_count(ids_total)}", "green")
        )

    clusters = s.get("clusters_created", 0)
    if clusters:
        stats.append(("clusters", format_count(clusters), "cyan"))

    cost = state.total_cost
    if cost > 0:
        stats.append(("cost", f"${cost:.2f}", "yellow"))

    return stats


def _build_pending(state: DDBuildState) -> list[tuple[str, int]]:
    """Build pending work counts from live build state."""
    counts = state.imas_node_status_counts
    pending: list[tuple[str, int]] = []

    pending_enrich = counts.get("built", 0)
    pending.append(("enrich", pending_enrich))

    pending_refine = counts.get("enriched", 0)
    pending.append(("refine", pending_refine))

    # Embed only applies to EMBEDDABLE_CATEGORIES (quantity, geometry)
    # — coordinate nodes stop at refined. Use embeddable-scoped counts
    # when available, falling back to enrichable-scoped refined count.
    embed_counts = state.embeddable_status_counts
    if embed_counts:
        pending_embed = embed_counts.get("refined", 0)
    else:
        pending_embed = counts.get("refined", 0)
    pending.append(("embed", pending_embed))

    # Cluster is a single-shot phase: pending if embed is done but cluster hasn't run
    cluster_pending = 0
    if (
        state.embed_phase.is_done
        and not state.cluster_phase.is_done
        and state.cluster_stats.processed == 0
    ):
        cluster_pending = 1
    pending.append(("cluster", cluster_pending))

    return pending


def _get_accumulated_build_time() -> float:
    """Query accumulated build duration from DDVersion nodes."""
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                "MATCH (v:DDVersion) "
                "WHERE v.build_duration IS NOT NULL "
                "RETURN sum(v.build_duration) AS total_time"
            )
        return float(result[0]["total_time"] or 0.0) if result else 0.0
    except Exception:
        return 0.0


def _graph_refresh(state: DDBuildState, _facility: str) -> None:
    """Refresh enrich/refine/embed progress and LLM costs from graph.

    Queries ``count_imas_nodes_by_status`` for progress and per-node
    atomic cost fields for authoritative cost tracking.  Graph costs
    are synced into ``WorkerStats.cost`` so the display always reflects
    the source of truth — no in-memory accumulators needed.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.dd_graph_ops import count_imas_nodes_by_status

    try:
        counts = count_imas_nodes_by_status(node_categories=ENRICHABLE_CATEGORIES)
    except Exception:
        return

    state.imas_node_status_counts = counts
    total = counts.get("total", 0)
    if total <= 0:
        return

    # Enrich: built → enriched/refined/embedded.  Processed = not-built.
    built = counts.get("built", 0)
    enrich_processed = total - built
    state.enrich_stats.total = total
    state.enrich_stats.set_baseline(enrich_processed)
    state.enrich_stats.processed = enrich_processed

    # Refine: enriched → refined/embedded.  Processed = refined + embedded.
    refined = counts.get("refined", 0)
    embedded = counts.get("embedded", 0)
    refine_processed = refined + embedded
    state.refine_stats.total = total
    state.refine_stats.set_baseline(refine_processed)
    state.refine_stats.processed = refine_processed

    # Embed uses EMBEDDABLE_CATEGORIES (quantity+geometry, excludes coordinate)
    try:
        embed_counts = count_imas_nodes_by_status(node_categories=EMBEDDABLE_CATEGORIES)
        state.embeddable_status_counts = embed_counts
        embed_total = embed_counts.get("total", 0)
        embed_done = embed_counts.get("embedded", 0)
        state.embed_stats.total = embed_total
        state.embed_stats.set_baseline(embed_done)
        state.embed_stats.processed = embed_done
    except Exception:
        # Fallback: use enrichable counts (slight overcount from coordinates)
        state.embed_stats.total = total
        state.embed_stats.set_baseline(embedded)
        state.embed_stats.processed = embedded

    # Sync LLM costs from per-node atomic fields (source of truth)
    try:
        with GraphClient() as gc:
            cost_result = gc.query(
                """
                MATCH (p:IMASNode)
                WHERE p.enrich_llm_cost IS NOT NULL
                   OR p.refine_llm_cost IS NOT NULL
                RETURN
                    coalesce(sum(p.enrich_llm_cost), 0) AS enrich_cost,
                    coalesce(sum(p.refine_llm_cost), 0) AS refine_cost
                """
            )
        if cost_result:
            state.enrich_stats.cost = float(cost_result[0]["enrich_cost"] or 0.0)
            state.refine_stats.cost = float(cost_result[0]["refine_cost"] or 0.0)
    except Exception:
        pass


def create_dd_build_display(
    state: DDBuildState,
    *,
    cost_limit: float = 0.0,
    console: Any | None = None,
    mode_label: str | None = None,
) -> DataDrivenProgressDisplay:
    """Create a progress display for the DD build pipeline.

    Configures ``DataDrivenProgressDisplay`` with stages matching the
    five sequential build phases.

    Args:
        state: Live build state — used for STATS summary row.
        cost_limit: LLM cost limit (for resource gauge).
        console: Rich Console instance (auto-created if None).
        mode_label: Optional mode label for the header (e.g. "DRY RUN").

    Returns:
        Configured display ready for ``set_engine_state()`` and use
        as a context manager.
    """
    stages = [
        StageDisplaySpec(
            name="EXTRACT",
            style="bold blue",
            group="extract",
            stats_attr="extract_stats",
            phase_attr="extract_phase",
        ),
        StageDisplaySpec(
            name="BUILD",
            style="bold green",
            group="build",
            stats_attr="build_stats",
            phase_attr="build_phase",
        ),
        StageDisplaySpec(
            name="ENRICH",
            style="bold yellow",
            group="enrich",
            stats_attr="enrich_stats",
            phase_attr="enrich_phase",
        ),
        StageDisplaySpec(
            name="REFINE",
            style="bold red",
            group="refine",
            stats_attr="refine_stats",
            phase_attr="refine_phase",
        ),
        StageDisplaySpec(
            name="EMBED",
            style="bold magenta",
            group="embed",
            stats_attr="embed_stats",
            phase_attr="embed_phase",
        ),
        StageDisplaySpec(
            name="CLUSTER",
            style="bold cyan",
            group="cluster",
            stats_attr="cluster_stats",
            phase_attr="cluster_phase",
        ),
    ]

    return DataDrivenProgressDisplay(
        facility="IMAS",
        cost_limit=cost_limit,
        stages=stages,
        console=console,
        title_suffix="DD Build",
        mode_label=mode_label,
        graph_refresh_fn=lambda f: _graph_refresh(state, f),
        stats_fn=lambda: _build_stats(state),
        pending_fn=lambda: _build_pending(state),
        accumulated_time_fn=lambda: _get_accumulated_build_time(),
    )
