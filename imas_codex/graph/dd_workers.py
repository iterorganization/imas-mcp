"""Async workers for the IMAS DD build pipeline.

Polling workers using the standard discovery engine pattern with
graph-backed status tracking and ``has_work_fn`` phase wiring.

Architecture:
- Five workers: extract, build, enrich, embed, cluster
- extract/build run once (extract XML, write nodes with status=built)
- enrich/embed are polling loops that claim batches from the graph
- cluster waits until embed phase is fully done

IMASNode lifecycle::

    built → enriched → embedded

Pipeline flow::

    EXTRACT → BUILD ──→ ENRICH ──→ CLUSTER
                   └──→ EMBED ──↗

Build writes nodes with status=built.  As soon as the first batch
lands, both enrich and embed workers can start claiming work.
Enrich claims built paths, enriches them, sets status=enriched.
Embed claims enriched paths, embeds them, sets status=embedded.
Cluster requires all embedding to complete.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
    dry_run: bool = False
    force: bool = False
    no_hash: bool = False

    # Shared data (extract → build/embed)
    version_data: dict[str, dict] = field(default_factory=dict)
    all_units: set[str] = field(default_factory=set)
    build_hash: str = ""

    # Build results
    stats: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    build_start_time: float = field(default_factory=time.time)

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

    def _on_progress(processed: int, total: int) -> None:
        state.extract_stats.total = total
        state.extract_stats.processed = processed
        if processed < total:
            state.extract_stats.status_text = f"v{state.versions[processed]}"
        else:
            state.extract_stats.status_text = ""

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_extract

        version_data, all_units = phase_extract(
            state.versions,
            state.ids_filter,
            on_progress=_on_progress,
        )
        state.version_data = version_data
        state.all_units = all_units

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

    def _on_progress(processed: int, total: int) -> None:
        state.build_stats.total = total
        state.build_stats.processed = processed
        if processed < total:
            state.build_stats.status_text = f"v{state.versions[processed]}"
        else:
            state.build_stats.status_text = ""

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
            )
            state.build_hash = build_hash

            if not state.dry_run and not state.force:
                if _check_graph_up_to_date(
                    client,
                    build_hash,
                    state.versions,
                ):
                    wlog.info("Graph already up-to-date — skipping")
                    state.skipped = True
                    state.stats["skipped"] = True
                    state.stats["versions_processed"] = len(state.versions)
                    _on_progress(len(state.versions), len(state.versions))
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
                on_progress=_on_progress,
            )
            state.stats.update(build_stats)

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
    """LLM enrichment polling loop: claim built → enrich → mark enriched.

    Polls the graph for IMASNodes with status=built, claims a batch,
    enriches them with LLM-generated descriptions, then sets
    status=enriched.  Exits when the phase is marked done by the
    supervision loop (idle + no pending work).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="enrich_worker")

    if state.dry_run:
        wlog.info("Enrichment skipped (dry run)")
        return

    wlog.info("Starting LLM enrichment polling loop")

    from imas_codex.graph.dd_graph_ops import (
        claim_paths_for_enrichment,
        release_enrichment_claims,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    while not state.should_stop():
        # Claim a batch of built paths
        paths = await asyncio.to_thread(
            claim_paths_for_enrichment,
            50,
            ids_filter=state.ids_filter,
        )

        if not paths:
            state.enrich_phase.record_idle()
            await asyncio.sleep(1.0)
            continue

        state.enrich_phase.record_activity(len(paths))
        path_ids = [p["id"] for p in paths]

        try:
            updates = await _enrich_batch(paths, model, state)

            if updates:
                await _batch_update_enrichments_with_graph(updates)
                state.enrich_stats.processed += len(updates)
                state.enrich_stats.record_batch(len(updates))

                llm_count = sum(
                    1 for u in updates if u.get("enrichment_source") == "llm"
                )
                template_count = len(updates) - llm_count
                state.stats["enriched_llm"] = (
                    state.stats.get("enriched_llm", 0) + llm_count
                )
                state.stats["enriched_template"] = (
                    state.stats.get("enriched_template", 0) + template_count
                )

        except Exception:
            wlog.exception("Error enriching batch, releasing claims")
            await asyncio.to_thread(release_enrichment_claims, path_ids)

    wlog.info(
        "Enrichment complete: %d LLM, %d template",
        state.stats.get("enriched_llm", 0),
        state.stats.get("enriched_template", 0),
    )


async def embed_worker(state: DDBuildState, **_kwargs) -> None:
    """Embedding polling loop: claim enriched → embed → mark embedded.

    Polls the graph for IMASNodes with status=enriched, claims a batch,
    generates embeddings, then sets status=embedded.  Exits when the
    phase is marked done (idle + no pending work).
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="embed_worker")

    if state.dry_run:
        wlog.info("Embedding skipped (dry run)")
        return

    wlog.info("Starting embedding polling loop")

    from imas_codex.graph.dd_graph_ops import (
        claim_paths_for_embedding,
        mark_paths_embedded,
        release_embedding_claims,
    )

    while not state.should_stop():
        # Claim a batch of enriched paths
        paths = await asyncio.to_thread(
            claim_paths_for_embedding,
            500,
        )

        if not paths:
            state.embed_phase.record_idle()
            await asyncio.sleep(1.0)
            continue

        state.embed_phase.record_activity(len(paths))
        path_ids = [p["id"] for p in paths]

        try:
            embedded_ids = await _embed_batch(paths, state)

            if embedded_ids:
                await asyncio.to_thread(mark_paths_embedded, embedded_ids)
                state.embed_stats.processed += len(embedded_ids)
                state.embed_stats.record_batch(len(embedded_ids))
                state.stats["embeddings_updated"] = state.stats.get(
                    "embeddings_updated", 0
                ) + len(embedded_ids)

        except Exception:
            wlog.exception("Error embedding batch, releasing claims")
            await asyncio.to_thread(release_embedding_claims, path_ids)

    wlog.info(
        "Embedding complete: %d updated",
        state.stats.get("embeddings_updated", 0),
    )


async def cluster_worker(state: DDBuildState, **_kwargs) -> None:
    """Import semantic clusters."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="cluster_worker")

    if state.dry_run:
        wlog.info("Clustering skipped (dry run)")
        state.cluster_phase.mark_done()
        return

    wlog.info("Starting cluster import")

    def _on_progress(processed: int, total: int) -> None:
        state.cluster_stats.total = total
        state.cluster_stats.processed = processed
        if processed > 0:
            state.cluster_stats.status_text = f"{processed:,} clusters"
        else:
            state.cluster_stats.status_text = ""

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_cluster
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            cluster_count = phase_cluster(
                client,
                dry_run=state.dry_run,
                no_hash=state.no_hash,
                on_progress=_on_progress,
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
# Batch Processing Helpers
# =============================================================================


async def _enrich_batch(
    paths: list[dict],
    model: str,
    state: DDBuildState,
) -> list[dict]:
    """Enrich a claimed batch of IMASNode paths.

    Async — uses ``acall_llm_structured`` for non-blocking LLM calls
    and ``asyncio.to_thread`` for sync graph reads.

    Returns list of update dicts ready for ``_batch_update_enrichments``.
    """
    import time as _time

    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.graph.dd_enrichment import (
        IMASPathEnrichmentBatch,
        build_enrichment_messages,
        compute_enrichment_hash,
        gather_path_context,
        generate_template_description,
        is_boilerplate_path,
    )

    updates: list[dict] = []

    # Separate boilerplate vs LLM paths
    boilerplate = [p for p in paths if is_boilerplate_path(p["id"])]
    llm_paths = [p for p in paths if not is_boilerplate_path(p["id"])]

    # Template enrichment (no LLM call)
    for path in boilerplate:
        template = generate_template_description(path["id"], path)
        template_hash = compute_enrichment_hash(
            f"{path.get('documentation', '')}", "template"
        )
        updates.append(
            {
                "id": path["id"],
                "description": template["description"],
                "keywords": template["keywords"],
                "enrichment_hash": template_hash,
                "enrichment_model": "template",
                "enrichment_source": "template",
            }
        )

    # LLM enrichment
    if llm_paths:

        def _gather_context():
            from imas_codex.graph.client import GraphClient

            with GraphClient() as client:
                ids_info_result = client.query(
                    "MATCH (i:IDS) RETURN i.id AS id, i.description AS description, "
                    "i.physics_domain AS physics_domain"
                )
                ids_info = {r["id"]: r for r in ids_info_result}
                batch_contexts = gather_path_context(client, llm_paths, ids_info)
            return ids_info, batch_contexts

        ids_info, batch_contexts = await asyncio.to_thread(_gather_context)

        # Check hashes — skip already-enriched (hash match)
        to_enrich = []
        for ctx in batch_contexts:
            ctx_str = (
                f"{ctx['id']}:{ctx.get('documentation', '')}:{ctx.get('siblings', [])}"
            )
            expected_hash = compute_enrichment_hash(ctx_str, model)
            if not state.no_hash and ctx.get("enrichment_hash") == expected_hash:
                continue  # Already enriched with same context
            ctx["_expected_hash"] = expected_hash
            to_enrich.append(ctx)

        if to_enrich:
            messages = build_enrichment_messages(to_enrich, ids_info)
            try:
                batch_start = _time.time()
                result, cost, tokens = await acall_llm_structured(
                    model=model,
                    messages=messages,
                    response_model=IMASPathEnrichmentBatch,
                )
                state.enrich_stats.cost += cost
                batch_time = _time.time() - batch_start

                for enrichment in result.results:
                    if enrichment.path_index < 1 or enrichment.path_index > len(
                        to_enrich
                    ):
                        continue
                    ctx = to_enrich[enrichment.path_index - 1]
                    update = {
                        "id": ctx["id"],
                        "description": enrichment.description,
                        "keywords": enrichment.keywords[:5],
                        "enrichment_hash": ctx["_expected_hash"],
                        "enrichment_model": model,
                        "enrichment_source": "llm",
                    }
                    if enrichment.physics_domain:
                        update["physics_domain"] = enrichment.physics_domain
                    updates.append(update)

                # Stream items for display
                if updates:
                    stream_items = [
                        {"primary_text": u["id"]}
                        for u in updates
                        if u.get("enrichment_source") == "llm"
                    ]
                    if stream_items:
                        state.enrich_stats.stream_queue.add(
                            stream_items,
                            last_batch_time=batch_time,
                        )

            except Exception:
                logger.exception("LLM enrichment failed for batch")
                # Return what we have (boilerplate updates)

    return updates


async def _batch_update_enrichments_with_graph(updates: list[dict]) -> None:
    """Write enrichment updates to graph (sets status=enriched)."""

    def _write():
        from imas_codex.graph.client import GraphClient
        from imas_codex.graph.dd_enrichment import _batch_update_enrichments

        with GraphClient() as client:
            _batch_update_enrichments(client, updates)

    await asyncio.to_thread(_write)


async def _embed_batch(
    paths: list[dict],
    state: DDBuildState,
) -> list[str]:
    """Generate embeddings for a claimed batch of enriched IMASNode paths.

    Async — uses ``asyncio.to_thread`` for CPU-bound encoding and
    sync graph writes.

    Returns list of path IDs that were successfully embedded.
    """
    from imas_codex.graph.build_dd import (
        compute_embedding_hash,
        filter_embeddable_paths,
        generate_embedding_text,
        generate_embeddings_batch,
    )
    from imas_codex.settings import get_embedding_model

    resolved_model = get_embedding_model()

    # Build path data dict from claimed results
    paths_data = {}
    for r in paths:
        pid = r["id"]
        paths_data[pid] = {k: v for k, v in r.items() if v is not None}

    # Filter to embeddable paths (excludes STRUCTURE, error fields, etc.)
    embeddable, _ = filter_embeddable_paths(paths_data)
    if not embeddable:
        return []

    # Generate embedding text and hashes
    embedding_texts = {}
    content_hashes = {}
    for path_id, path_info in embeddable.items():
        text = generate_embedding_text(path_id, path_info, {})
        embedding_texts[path_id] = text
        content_hashes[path_id] = compute_embedding_hash(text, resolved_model)

    # Skip paths where hash hasn't changed (already embedded correctly)
    path_ids = list(embeddable.keys())
    paths_to_embed = []
    for pid in path_ids:
        existing_hash = paths_data[pid].get("embedding_hash")
        if not state.force and existing_hash == content_hashes[pid]:
            continue  # Already embedded with same content
        paths_to_embed.append(pid)

    if not paths_to_embed:
        return path_ids  # All cached — still mark as embedded

    # Generate embeddings (CPU-bound — run in thread)
    texts = [embedding_texts[pid] for pid in paths_to_embed]
    embeddings = await asyncio.to_thread(
        generate_embeddings_batch,
        texts,
        resolved_model,
    )

    # Store embeddings in graph (sync driver — run in thread)
    batch_data = []
    for i, pid in enumerate(paths_to_embed):
        batch_data.append(
            {
                "path_id": pid,
                "embedding_text": embedding_texts[pid],
                "embedding": embeddings[i].tolist(),
                "embedding_hash": content_hashes[pid],
            }
        )

    def _write_embeddings():
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            client.query(
                """
                UNWIND $batch AS b
                MATCH (p:IMASNode {id: b.path_id})
                SET p.embedding_text = b.embedding_text,
                    p.embedding = b.embedding,
                    p.embedding_hash = b.embedding_hash
                """,
                batch=batch_data,
            )

    await asyncio.to_thread(_write_embeddings)

    # Stream items for display
    stream_items = [{"primary_text": pid} for pid in paths_to_embed]
    if stream_items:
        state.embed_stats.stream_queue.add(stream_items, last_batch_time=0.0)

    return path_ids


async def run_dd_build_engine(
    state: DDBuildState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the DD build pipeline as a discovery engine.

    Uses graph-backed ``has_work_fn`` wiring so workers start as soon
    as their upstream phase produces work — no blocking on full phase
    completion.  Build writes nodes with ``status=built``, enrich polls
    for built paths, embed polls for enriched paths.

    Pipeline::

        EXTRACT → BUILD ──→ ENRICH ──→ CLUSTER
                       └──→ EMBED ──↗
    """
    from imas_codex.discovery.base.engine import OrphanRecoverySpec
    from imas_codex.graph import dd_graph_ops

    # --- Wire has_work_fn for graph-backed phase completion ---
    state.enrich_phase.set_has_work_fn(
        lambda: (
            dd_graph_ops.has_pending_enrichment(ids_filter=state.ids_filter)
            or not state.build_phase.done
        )
    )
    state.embed_phase.set_has_work_fn(
        lambda: (dd_graph_ops.has_pending_embedding() or not state.enrich_phase.done)
    )

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
        ),
        WorkerSpec(
            "embed",
            "embed_phase",
            embed_worker,
        ),
        WorkerSpec(
            "cluster",
            "cluster_phase",
            cluster_worker,
            depends_on=["enrich_phase", "embed_phase"],
        ),
    ]

    orphan_specs = [
        OrphanRecoverySpec("IMASNode"),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
        orphan_specs=orphan_specs,
    )

    # Persist build metadata to graph
    if not state.dry_run and not state.skipped:
        _write_build_metadata(state)


def _write_build_metadata(state: DDBuildState) -> None:
    """Write build timing, cost, and hash to the current DDVersion node."""
    from imas_codex import dd_version as current_dd_version
    from imas_codex.graph.client import GraphClient

    duration = time.time() - state.build_start_time
    total_cost = state.total_cost

    try:
        with GraphClient() as client:
            client.query(
                """
                MATCH (v:DDVersion {id: $version})
                SET v.build_hash = $build_hash,
                    v.build_completed_at = datetime(),
                    v.build_duration = $duration,
                    v.build_cost = $total_cost,
                    v.enrichment_cost = $enrichment_cost
                """,
                version=current_dd_version,
                build_hash=state.build_hash,
                duration=duration,
                total_cost=total_cost,
                enrichment_cost=state.enrich_stats.cost,
            )
    except Exception:
        logger.warning("Failed to write build metadata to graph", exc_info=True)
