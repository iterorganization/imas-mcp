"""Progress display factory for the IMAS DD build pipeline.

Uses ``DataDrivenProgressDisplay`` directly — no custom subclass
needed since each phase has its own worker group.  The base class's
``_count_group_workers()`` and ``_worker_complete()`` work naturally
with one worker per group.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.progress import (
    DataDrivenProgressDisplay,
    StageDisplaySpec,
)

if TYPE_CHECKING:
    from imas_codex.graph.dd_workers import DDBuildState


def _build_stats(state: DDBuildState) -> list[tuple[str, str, str]]:
    """Build STATS row entries from live build state."""
    stats: list[tuple[str, str, str]] = []
    s = state.stats

    versions = s.get("versions_processed", 0)
    if versions:
        stats.append(("versions", str(versions), "blue"))

    paths = s.get("paths_created", 0)
    if paths:
        stats.append(("paths", f"{paths:,}", "green"))

    enriched = s.get("enriched_llm", 0) + s.get("enriched_template", 0)
    if enriched:
        stats.append(("enriched", f"{enriched:,}", "green"))

    cached = s.get("enrichment_cached", 0)
    if cached:
        stats.append(("cached", f"{cached:,}", "dim"))

    embedded = s.get("embeddings_updated", 0) + s.get("embeddings_cached", 0)
    if embedded:
        stats.append(("embedded", f"{embedded:,}", "magenta"))

    clusters = s.get("clusters_created", 0)
    if clusters:
        stats.append(("clusters", f"{clusters:,}", "cyan"))

    return stats


def create_dd_build_display(
    state: DDBuildState,
    *,
    cost_limit: float = 0.0,
    console: Any | None = None,
    skip_enrichment: bool = False,
    skip_embeddings: bool = False,
    skip_clusters: bool = False,
    mode_label: str | None = None,
) -> DataDrivenProgressDisplay:
    """Create a progress display for the DD build pipeline.

    Configures ``DataDrivenProgressDisplay`` with stages matching the
    five sequential build phases.  Disabled stages show as "skipped".

    Args:
        state: Live build state — used for STATS summary row.
        cost_limit: LLM cost limit (for resource gauge).
        console: Rich Console instance (auto-created if None).
        skip_enrichment: Disable the ENRICH stage.
        skip_embeddings: Disable the EMBED stage.
        skip_clusters: Disable the CLUSTER stage.
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
            disabled=skip_enrichment,
            disabled_msg="skipped",
        ),
        StageDisplaySpec(
            name="EMBED",
            style="bold magenta",
            group="embed",
            stats_attr="embed_stats",
            phase_attr="embed_phase",
            disabled=skip_embeddings,
            disabled_msg="skipped",
        ),
        StageDisplaySpec(
            name="CLUSTER",
            style="bold cyan",
            group="cluster",
            stats_attr="cluster_stats",
            phase_attr="cluster_phase",
            disabled=skip_clusters,
            disabled_msg="skipped",
        ),
    ]

    return DataDrivenProgressDisplay(
        facility="IMAS",
        cost_limit=cost_limit,
        stages=stages,
        console=console,
        title_suffix="DD Build",
        mode_label=mode_label,
        stats_fn=lambda: _build_stats(state),
    )
