"""Async workers for the IMAS DD build pipeline.

Per-phase workers using the standard discovery engine pattern with
``depends_on`` for sequential phase coordination.

Architecture:
- Five independent workers: extract, build, enrich, embed, cluster
- Each worker runs one phase via ``asyncio.to_thread()``
- Sequential dependencies via ``WorkerSpec.depends_on``
- Each phase has its own worker group for per-row display
- Phase functions from ``build_dd`` are called directly

Pipeline: EXTRACT → BUILD → ENRICH → EMBED → CLUSTER
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.state import DiscoveryStateBase
from imas_codex.discovery.base.supervision import PipelinePhase

logger = logging.getLogger(__name__)


# =============================================================================
# Discovery State
# =============================================================================


@dataclass
class DDBuildState(DiscoveryStateBase):
    """Shared state for the DD build pipeline.

    Each phase has its own ``WorkerStats`` and ``PipelinePhase``.
    Workers update stats and shared data as they progress.
    """

    # Build configuration
    versions: list[str] = field(default_factory=list)
    ids_filter: set[str] | None = None

    # Feature flags
    include_clusters: bool = True
    include_embeddings: bool = True
    include_enrichment: bool = True
    skip_cluster_labels: bool = False
    dry_run: bool = False
    force: bool = False
    no_hash: bool = False

    # Model overrides
    embedding_model: str | None = None
    enrichment_model: str | None = None

    # Shared data (extract → build/embed)
    version_data: dict[str, dict] = field(default_factory=dict)
    all_units: set[str] = field(default_factory=set)
    build_hash: str = ""

    # Build results
    stats: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    # Per-phase progress (observed by display)
    extract_stats: WorkerStats = field(default_factory=WorkerStats)
    build_stats: WorkerStats = field(default_factory=WorkerStats)
    enrich_stats: WorkerStats = field(default_factory=WorkerStats)
    embed_stats: WorkerStats = field(default_factory=WorkerStats)
    cluster_stats: WorkerStats = field(default_factory=WorkerStats)

    # Pipeline phases
    extract_phase: PipelinePhase = field(init=False)
    build_phase: PipelinePhase = field(init=False)
    enrich_phase: PipelinePhase = field(init=False)
    embed_phase: PipelinePhase = field(init=False)
    cluster_phase: PipelinePhase = field(init=False)

    def __post_init__(self) -> None:
        self.extract_phase = PipelinePhase("extract")
        self.build_phase = PipelinePhase("build")
        self.enrich_phase = PipelinePhase("enrich")
        self.embed_phase = PipelinePhase("embed")
        self.cluster_phase = PipelinePhase("cluster")

    @property
    def total_cost(self) -> float:
        return self.enrich_stats.cost


# =============================================================================
# Workers
# =============================================================================


async def extract_worker(state: DDBuildState, **_kwargs) -> None:
    """Extract paths from DD XML for all versions."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="extract_worker")
    wlog.info("Starting extraction for %d versions", len(state.versions))

    state.extract_stats.total = len(state.versions)

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_extract

        version_data, all_units = phase_extract(
            state.versions, state.ids_filter
        )
        state.version_data = version_data
        state.all_units = all_units
        state.extract_stats.processed = len(version_data)

    await asyncio.to_thread(_run)

    total_paths = sum(len(d["paths"]) for d in state.version_data.values())
    wlog.info(
        "Extraction complete: %d versions, %d paths",
        len(state.version_data),
        total_paths,
    )
    state.extract_phase.mark_done()


async def build_worker(state: DDBuildState, **_kwargs) -> None:
    """Create graph nodes from extracted data."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="build_worker")
    wlog.info("Starting graph build for %d versions", len(state.versions))

    state.build_stats.total = len(state.versions)

    def _run() -> None:
        from imas_codex.graph.build_dd import (
            _check_graph_up_to_date,
            _compute_build_hash,
            _create_version_nodes,
            _ensure_indexes,
            phase_build,
        )
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            # Hash-based idempotency check
            build_hash = _compute_build_hash(
                state.versions,
                state.ids_filter,
                state.embedding_model,
                state.include_clusters,
                state.include_embeddings,
            )
            state.build_hash = build_hash

            if not state.dry_run and not state.force:
                if _check_graph_up_to_date(
                    client,
                    build_hash,
                    state.versions,
                    state.include_embeddings,
                    state.include_clusters,
                ):
                    wlog.info("Graph already up-to-date — skipping")
                    state.skipped = True
                    state.stats["skipped"] = True
                    state.stats["versions_processed"] = len(state.versions)
                    return

            # Preflight: indexes + version nodes
            if not state.dry_run:
                _ensure_indexes(client)
                _create_version_nodes(client, state.versions)

            state.stats["versions_processed"] = len(state.versions)

            # Build graph nodes
            build_stats = phase_build(
                client,
                state.versions,
                state.version_data,
                state.all_units,
                dry_run=state.dry_run,
            )
            state.stats.update(build_stats)
            state.build_stats.processed = build_stats.get("paths_created", 0)

    await asyncio.to_thread(_run)

    if state.skipped:
        wlog.info("Build skipped (graph up-to-date)")
        # Mark all downstream phases done
        state.stop_requested = True
    else:
        wlog.info(
            "Build complete: %d paths created",
            state.stats.get("paths_created", 0),
        )
    state.build_phase.mark_done()


async def enrich_worker(state: DDBuildState, **_kwargs) -> None:
    """LLM enrichment of path descriptions."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_worker")

    if state.dry_run:
        wlog.info("Enrichment skipped (dry run)")
        state.enrich_phase.mark_done()
        return

    wlog.info("Starting LLM enrichment")

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_enrich
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            enrich_stats = phase_enrich(
                client,
                model=state.enrichment_model,
                ids_filter=state.ids_filter,
                force=state.no_hash,
            )
            state.stats.update(enrich_stats)
            state.enrich_stats.processed = int(
                enrich_stats.get("enriched_llm", 0)
                + enrich_stats.get("enriched_template", 0)
            )
            state.enrich_stats.cost = enrich_stats.get("enrichment_cost", 0.0)

    await asyncio.to_thread(_run)

    wlog.info(
        "Enrichment complete: %d LLM, %d template",
        state.stats.get("enriched_llm", 0),
        state.stats.get("enriched_template", 0),
    )
    state.enrich_phase.mark_done()


async def embed_worker(state: DDBuildState, **_kwargs) -> None:
    """Generate vector embeddings for DD paths."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="embed_worker")

    if state.dry_run:
        wlog.info("Embedding skipped (dry run)")
        state.embed_phase.mark_done()
        return

    wlog.info("Starting embedding generation")

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_embed
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            embed_stats = phase_embed(
                client,
                state.versions,
                state.version_data,
                include_enrichment=state.include_enrichment,
                enriched_llm_count=state.stats.get("enriched_llm", 0),
                embedding_model=state.embedding_model,
                force=state.force,
                no_hash=state.no_hash,
            )
            state.stats.update(embed_stats)
            state.embed_stats.processed = (
                embed_stats.get("embeddings_updated", 0)
                + embed_stats.get("embeddings_cached", 0)
            )

    await asyncio.to_thread(_run)

    wlog.info(
        "Embedding complete: %d updated, %d cached",
        state.stats.get("embeddings_updated", 0),
        state.stats.get("embeddings_cached", 0),
    )
    state.embed_phase.mark_done()


async def cluster_worker(state: DDBuildState, **_kwargs) -> None:
    """Import semantic clusters."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="cluster_worker")
    wlog.info("Starting cluster import")

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_cluster
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            cluster_count = phase_cluster(
                client,
                dry_run=state.dry_run,
                no_hash=state.no_hash,
                skip_labels=state.skip_cluster_labels,
            )
            state.stats["clusters_created"] = cluster_count
            state.cluster_stats.processed = cluster_count

    await asyncio.to_thread(_run)

    wlog.info(
        "Cluster import complete: %d clusters",
        state.stats.get("clusters_created", 0),
    )
    state.cluster_phase.mark_done()


# =============================================================================
# Engine Entry Point
# =============================================================================


async def run_dd_build_engine(
    state: DDBuildState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the DD build pipeline as a discovery engine.

    Each phase is a separate worker with ``depends_on`` declarations
    for sequential coordination.  The display shows one row per phase,
    with workers starting automatically when dependencies complete.
    """
    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            extract_worker,
        ),
        WorkerSpec(
            "build",
            "build_phase",
            build_worker,
            depends_on=["extract_phase"],
        ),
        WorkerSpec(
            "enrich",
            "enrich_phase",
            enrich_worker,
            depends_on=["build_phase"],
            enabled=state.include_enrichment,
        ),
        WorkerSpec(
            "embed",
            "embed_phase",
            embed_worker,
            depends_on=["build_phase", "enrich_phase"],
            enabled=state.include_embeddings,
        ),
        WorkerSpec(
            "cluster",
            "cluster_phase",
            cluster_worker,
            depends_on=["build_phase", "enrich_phase", "embed_phase"],
            enabled=state.include_clusters,
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
    )
