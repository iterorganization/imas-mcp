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

from imas_codex.core.node_categories import EMBEDDABLE_CATEGORIES, ENRICHABLE_CATEGORIES
from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.progress import WorkerStats, format_count
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
    reset_to: str | None = None
    force: bool = False

    @property
    def skip_build_hash(self) -> bool:
        """Bypass build-level hash check (re-extract/rebuild)."""
        return self.force or self.reset_to == "extracted"

    @property
    def skip_enrichment_hash(self) -> bool:
        """Bypass per-path enrichment hash check (re-enrich all)."""
        return self.force or self.reset_to in ("extracted", "built")

    @property
    def skip_embedding_hash(self) -> bool:
        """Bypass per-path embedding hash check (re-embed all)."""
        return self.force or self.reset_to is not None

    # Shared data (extract → build/embed)
    version_data: dict[str, dict] = field(default_factory=dict)
    all_units: set[str] = field(default_factory=set)
    build_hash: str = ""

    # Build results
    stats: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    build_start_time: float = field(default_factory=time.time)

    # Status breakdown from graph (for pending display)
    imas_node_status_counts: dict[str, int] = field(default_factory=dict)

    # Accumulated cost from prior builds (queried from graph)
    accumulated_cost: float = 0.0

    # Auxiliary IDS / identifier enrichment and embedding run once per build
    aux_enrichment_done: bool = False
    aux_embedding_done: bool = False
    aux_enrichment_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    aux_embedding_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

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


def _style_stream_items(items: list[dict], primary_text_style: str) -> list[dict]:
    """Attach display style metadata to DD stream items."""
    return [
        {
            **item,
            "primary_text_style": item.get("primary_text_style", primary_text_style),
        }
        for item in items
    ]


# =============================================================================
# Workers
# =============================================================================


async def extract_worker(state: DDBuildState, **_kwargs) -> None:
    """Extract paths from DD XML for all versions.

    Skips extraction if build nodes already exist in the graph with
    matching build hash — the version_data is not needed when
    build_worker will also skip.
    """
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="extract_worker")
    wlog.info("Starting extraction for %d versions", len(state.versions))

    def _on_progress(processed: int, total: int) -> None:
        state.extract_stats.total = total
        prev = state.extract_stats.processed
        state.extract_stats.processed = processed
        if processed > prev:
            state.extract_stats.record_batch(processed - prev)
        if processed < total:
            state.extract_stats.status_text = f"v{state.versions[processed]}"
        else:
            state.extract_stats.status_text = ""

    def _run() -> None:
        from imas_codex.graph.build_dd import (
            _check_build_nodes_exist,
            _compute_build_hash,
            phase_extract,
        )
        from imas_codex.graph.client import GraphClient

        # Skip extraction entirely when only re-enriching or re-embedding
        if state.reset_to in ("built", "enriched"):
            wlog.info("Extraction skipped (reset-to %s)", state.reset_to)
            _on_progress(len(state.versions), len(state.versions))
            return

        # Quick graph check before expensive XML parsing
        if not state.dry_run and not state.skip_build_hash:
            build_hash = _compute_build_hash(state.versions, state.ids_filter)
            try:
                with GraphClient() as client:
                    if _check_build_nodes_exist(client, build_hash, state.versions):
                        wlog.info("Build nodes already exist — skipping XML extraction")
                        _on_progress(len(state.versions), len(state.versions))
                        return
            except Exception:
                wlog.debug("Graph check failed, proceeding with extraction")

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
    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()


async def build_worker(state: DDBuildState, **_kwargs) -> None:
    """Create graph nodes from extracted data."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="build_worker")
    wlog.info("Starting graph build for %d versions", len(state.versions))

    def _on_progress(processed: int, total: int) -> None:
        state.build_stats.total = total
        prev = state.build_stats.processed
        state.build_stats.processed = processed
        if processed > prev:
            state.build_stats.record_batch(processed - prev)

    def _on_version(version: str) -> None:
        state.build_stats.status_text = f"v{version}"

    def _run() -> None:
        from imas_codex.graph.build_dd import (
            _check_build_nodes_exist,
            _check_graph_up_to_date,
            _compute_build_hash,
            _create_version_nodes,
            _ensure_indexes,
            phase_build,
        )
        from imas_codex.graph.client import GraphClient

        # Always compute build hash (needed by downstream phases)
        build_hash = _compute_build_hash(state.versions, state.ids_filter)
        state.build_hash = build_hash

        # Skip build entirely when only re-enriching or re-embedding
        if state.reset_to in ("built", "enriched"):
            state.stats["skipped_build"] = True
            state.stats["versions_processed"] = len(state.versions)
            wlog.info("Build skipped (reset-to %s)", state.reset_to)
            _on_progress(len(state.versions), len(state.versions))
            return

        with GraphClient() as client:
            if not state.dry_run and not state.skip_build_hash:
                # Full check: everything including embeddings + clusters
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

                # Lighter check: build nodes exist, skip extract+build
                # but let enrich/embed/cluster continue
                if _check_build_nodes_exist(
                    client,
                    build_hash,
                    state.versions,
                ):
                    wlog.info(
                        "Build nodes already exist — skipping extract/build, "
                        "downstream phases will continue"
                    )
                    state.stats["skipped_build"] = True
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
                on_version=_on_version,
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
    state.build_stats.freeze_rate()
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
        count_imas_nodes_by_status,
        release_enrichment_claims,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    # Set initial totals from graph so progress bar shows real denominator
    # and initialize processed counts from already-completed work so that
    # restarting the CLI doesn't reset progress to 0%.
    try:
        status_counts = await asyncio.to_thread(
            count_imas_nodes_by_status, node_categories=ENRICHABLE_CATEGORIES
        )
        state.imas_node_status_counts = status_counts
        total_nodes = status_counts.get("total", 0)
        if total_nodes > 0:
            state.enrich_stats.total = total_nodes
            state.embed_stats.total = total_nodes
            # Nodes already past enrichment (enriched or embedded)
            already_enriched = status_counts.get("enriched", 0) + status_counts.get(
                "embedded", 0
            )
            state.enrich_stats.set_baseline(already_enriched)
            state.enrich_stats.processed = already_enriched
            # Nodes already past embedding
            already_embedded = status_counts.get("embedded", 0)
            state.embed_stats.set_baseline(already_embedded)
            state.embed_stats.processed = already_embedded
    except Exception:
        wlog.debug("Could not fetch initial node counts", exc_info=True)

    while not state.should_stop():
        # Claim a batch of built paths
        paths = await asyncio.to_thread(
            claim_paths_for_enrichment,
            50,
            ids_filter=state.ids_filter,
        )

        if not paths:
            if not state.aux_enrichment_done:
                state.enrich_stats.status_text = "IDS/identifier"
                await _run_aux_enrichment(state)
                state.enrich_stats.status_text = ""
                continue

            state.enrich_phase.record_idle()
            if state.enrich_phase.done:
                break
            # Refresh totals while idle (build may still be adding nodes)
            try:
                status_counts = await asyncio.to_thread(
                    count_imas_nodes_by_status, node_categories=ENRICHABLE_CATEGORIES
                )
                state.imas_node_status_counts = status_counts
                total_nodes = status_counts.get("total", 0)
                if total_nodes > 0:
                    state.enrich_stats.total = total_nodes
                    state.embed_stats.total = total_nodes
                    # Keep processed in sync with graph state
                    already_enriched = status_counts.get(
                        "enriched", 0
                    ) + status_counts.get("embedded", 0)
                    state.enrich_stats.processed = max(
                        state.enrich_stats.processed, already_enriched
                    )
                    already_embedded = status_counts.get("embedded", 0)
                    state.embed_stats.processed = max(
                        state.embed_stats.processed, already_embedded
                    )
            except Exception:
                pass
            await asyncio.sleep(1.0)
            continue

        state.enrich_phase.record_activity(len(paths))
        path_ids = [p["id"] for p in paths]

        # Check stop AFTER claiming — release claims and exit promptly
        if state.should_stop():
            await asyncio.to_thread(release_enrichment_claims, path_ids)
            break

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

            # Release claims for paths not in the updates (LLM failures)
            updated_ids = {u["id"] for u in updates} if updates else set()
            failed_ids = [pid for pid in path_ids if pid not in updated_ids]
            if failed_ids:
                await asyncio.to_thread(release_enrichment_claims, failed_ids)

        except Exception:
            wlog.exception("Error enriching batch, releasing claims")
            await asyncio.to_thread(release_enrichment_claims, path_ids)

    await _run_aux_enrichment(state)

    wlog.info(
        "Enrichment complete: %d LLM, %d template",
        state.stats.get("enriched_llm", 0),
        state.stats.get("enriched_template", 0),
    )
    state.enrich_stats.freeze_rate()


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
        count_imas_nodes_by_status,
        mark_paths_embedded,
        release_embedding_claims,
    )

    # Initialize processed count from graph state so restarting doesn't
    # show 0% for already-embedded work.
    try:
        status_counts = await asyncio.to_thread(
            count_imas_nodes_by_status, node_categories=EMBEDDABLE_CATEGORIES
        )
        total_nodes = status_counts.get("total", 0)
        if total_nodes > 0:
            state.embed_stats.total = max(state.embed_stats.total, total_nodes)
            already_embedded = status_counts.get("embedded", 0)
            state.embed_stats.set_baseline(already_embedded)
            state.embed_stats.processed = max(
                state.embed_stats.processed, already_embedded
            )
    except Exception:
        wlog.debug("Could not fetch initial embed counts", exc_info=True)

    while not state.should_stop():
        # Claim a batch of enriched paths
        paths = await asyncio.to_thread(
            claim_paths_for_embedding,
            500,
        )

        if not paths:
            if state.enrich_phase.done and not state.aux_embedding_done:
                state.embed_stats.status_text = "IDS/identifier"
                await _run_aux_embedding(state)
                state.embed_stats.status_text = ""
                continue

            state.embed_phase.record_idle()
            if state.embed_phase.done:
                break
            await asyncio.sleep(1.0)
            continue

        state.embed_phase.record_activity(len(paths))
        path_ids = [p["id"] for p in paths]

        # Check stop AFTER claiming — release claims and exit promptly
        # so Ctrl+C doesn't block waiting for a full embed batch.
        if state.should_stop():
            await asyncio.to_thread(release_embedding_claims, path_ids)
            break

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

    await _run_aux_embedding(state)

    wlog.info(
        "Embedding complete: %d updated",
        state.stats.get("embeddings_updated", 0),
    )
    state.embed_stats.freeze_rate()


async def cluster_worker(state: DDBuildState, **_kwargs) -> None:
    """Import semantic clusters."""
    from imas_codex.cli.logging import WorkerLogAdapter

    wlog = WorkerLogAdapter(logger, worker_name="cluster_worker")

    if state.dry_run:
        wlog.info("Clustering skipped (dry run)")
        state.cluster_phase.mark_done()
        return

    # Check stop BEFORE entering the long-running clustering thread
    if state.should_stop():
        wlog.info("Clustering skipped (stop requested)")
        state.cluster_phase.mark_done()
        return

    wlog.info("Starting cluster import")

    def _on_progress(processed: int, total: int) -> None:
        state.cluster_stats.total = total
        state.cluster_stats.processed = processed
        if processed > 0:
            state.cluster_stats.status_text = f"{format_count(processed)} clusters"
        else:
            state.cluster_stats.status_text = ""

    def _run() -> None:
        from imas_codex.graph.build_dd import phase_cluster
        from imas_codex.graph.client import GraphClient

        with GraphClient() as client:
            cluster_count = phase_cluster(
                client,
                dry_run=state.dry_run,
                force_reembed=state.reset_to is not None,
                on_progress=_on_progress,
                stop_check=state.should_stop,
            )
            state.stats["clusters_created"] = cluster_count
            state.cluster_stats.processed = cluster_count

    await asyncio.to_thread(_run)

    wlog.info(
        "Cluster import complete: %d clusters",
        state.stats.get("clusters_created", 0),
    )
    state.cluster_stats.freeze_rate()
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
        is_accessor_terminal,
        is_boilerplate_path,
    )

    updates: list[dict] = []

    # Separate boilerplate/accessor terminals vs LLM paths
    template_paths = [
        p for p in paths if is_accessor_terminal(p["id"], p["id"].split("/")[-1])
    ]
    llm_paths = [
        p for p in paths if not is_accessor_terminal(p["id"], p["id"].split("/")[-1])
    ]

    # Template enrichment (no LLM call) — boilerplate + accessor terminals
    if template_paths:
        # Query parent info for accessor terminals that need parent context
        accessor_ids = [
            p["id"] for p in template_paths if not is_boilerplate_path(p["id"])
        ]
        parent_map: dict[str, dict] = {}
        if accessor_ids:

            def _fetch_parents():
                from imas_codex.graph.client import GraphClient

                with GraphClient() as client:
                    results = client.query(
                        """
                        UNWIND $path_ids AS pid
                        MATCH (n:IMASNode {id: pid})-[:HAS_PARENT]->(parent:IMASNode)
                        RETURN pid AS path_id, parent.name AS parent_name,
                               coalesce(parent.description, parent.documentation) AS parent_doc
                        """,
                        path_ids=accessor_ids,
                    )
                return {
                    r["path_id"]: {
                        "name": r["parent_name"],
                        "documentation": r["parent_doc"],
                    }
                    for r in results
                }

            parent_map = await asyncio.to_thread(_fetch_parents)

        for path in template_paths:
            parent_info = parent_map.get(path["id"])
            template = generate_template_description(
                path["id"], path, parent_info=parent_info
            )
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
        hash_match_updates = []
        for ctx in batch_contexts:
            ctx_str = (
                f"{ctx['id']}:{ctx.get('documentation', '')}:{ctx.get('siblings', [])}"
            )
            expected_hash = compute_enrichment_hash(ctx_str, model)
            if (
                state.reset_to != "built"
                and ctx.get("enrichment_hash") == expected_hash
            ):
                # Hash matches — content unchanged. Still need to mark as
                # enriched so the node advances from built → enriched.
                hash_match_updates.append({"id": ctx["id"]})
                continue
            ctx["_expected_hash"] = expected_hash
            to_enrich.append(ctx)

        # Advance hash-matched nodes without re-enriching
        if hash_match_updates:
            updates.extend(hash_match_updates)

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
                        {
                            "primary_text": u["id"],
                            "physics_domain": u.get("physics_domain", ""),
                            "description": u.get("description", ""),
                        }
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
    from imas_codex.graph.dd_graph_ops import mark_paths_enriched

    await asyncio.to_thread(mark_paths_enriched, updates)


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

    # Generate embedding text and hashes for all paths
    embedding_texts = {}
    content_hashes = {}
    for path_id, path_info in paths_data.items():
        text = generate_embedding_text(path_id, path_info, {})
        embedding_texts[path_id] = text
        content_hashes[path_id] = compute_embedding_hash(text, resolved_model)

    # Skip paths where hash hasn't changed (already embedded correctly)
    path_ids = list(paths_data.keys())
    paths_to_embed = []
    for pid in path_ids:
        existing_hash = paths_data[pid].get("embedding_hash")
        if not state.skip_embedding_hash and existing_hash == content_hashes[pid]:
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


async def _run_aux_enrichment(state: DDBuildState) -> None:
    """Run IDS and identifier enrichment once per DD build."""
    async with state.aux_enrichment_lock:
        if state.aux_enrichment_done or state.dry_run:
            return

        loop = asyncio.get_running_loop()

        def _publish_items(
            items: list[dict],
            batch_time: float,
            *,
            primary_text_style: str,
        ) -> None:
            loop.call_soon_threadsafe(
                lambda: state.enrich_stats.stream_queue.add(
                    _style_stream_items(items, primary_text_style),
                    last_batch_time=batch_time,
                )
            )

        def _on_identifier_items(items: list[dict], batch_time: float) -> None:
            _publish_items(items, batch_time, primary_text_style="bold cyan")

        def _on_ids_items(items: list[dict], batch_time: float) -> None:
            _publish_items(items, batch_time, primary_text_style="bold green")

        def _run() -> tuple[int, int, dict[str, Any], dict[str, Any]]:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.dd_identifier_enrichment import (
                enrich_identifier_schemas,
            )
            from imas_codex.graph.dd_ids_enrichment import enrich_ids_nodes
            from imas_codex.settings import get_model

            with GraphClient() as client:
                identifier_total = client.query(
                    "MATCH (s:IdentifierSchema) RETURN count(s) AS total"
                )[0]["total"]
                ids_total = client.query("MATCH (i:IDS) RETURN count(i) AS total")[0][
                    "total"
                ]
                ident_stats = enrich_identifier_schemas(
                    client,
                    model=get_model("language"),
                    force=state.skip_enrichment_hash,
                    on_items=_on_identifier_items,
                )
                ids_stats = enrich_ids_nodes(
                    client,
                    model=get_model("language"),
                    force=state.skip_enrichment_hash,
                    on_items=_on_ids_items,
                )
            return identifier_total, ids_total, ident_stats, ids_stats

        identifier_total, ids_total, ident_stats, ids_stats = await asyncio.to_thread(
            _run
        )
        state.stats["identifier_schemas_total"] = identifier_total
        state.stats["ids_total"] = ids_total
        state.stats["identifier_schemas_enriched"] = ident_stats.get("enriched", 0)
        state.stats["ids_enriched"] = ids_stats.get("enriched", 0)
        state.stats["identifier_schemas_cached"] = ident_stats.get("cached", 0)
        state.stats["ids_cached"] = ids_stats.get("cached", 0)
        state.enrich_stats.cost += ident_stats.get("cost", 0.0)
        state.enrich_stats.cost += ids_stats.get("cost", 0.0)
        state.aux_enrichment_done = True


async def _run_aux_embedding(state: DDBuildState) -> None:
    """Run IDS and identifier embedding once per DD build."""
    async with state.aux_embedding_lock:
        if state.aux_embedding_done or state.dry_run:
            return

        loop = asyncio.get_running_loop()

        def _publish_items(
            items: list[dict],
            batch_time: float,
            *,
            primary_text_style: str,
        ) -> None:
            loop.call_soon_threadsafe(
                lambda: state.embed_stats.stream_queue.add(
                    _style_stream_items(items, primary_text_style),
                    last_batch_time=batch_time,
                )
            )

        def _on_identifier_items(items: list[dict], batch_time: float) -> None:
            _publish_items(items, batch_time, primary_text_style="bold cyan")

        def _on_ids_items(items: list[dict], batch_time: float) -> None:
            _publish_items(items, batch_time, primary_text_style="bold green")

        def _run() -> tuple[dict[str, int], dict[str, int]]:
            from imas_codex.graph.client import GraphClient
            from imas_codex.graph.dd_identifier_enrichment import (
                embed_identifier_schemas,
            )
            from imas_codex.graph.dd_ids_enrichment import embed_ids_nodes

            with GraphClient() as client:
                ident_stats = embed_identifier_schemas(
                    client,
                    force_reembed=state.skip_embedding_hash,
                    on_items=_on_identifier_items,
                )
                ids_stats = embed_ids_nodes(
                    client,
                    force_reembed=state.skip_embedding_hash,
                    on_items=_on_ids_items,
                )
            return ident_stats, ids_stats

        ident_stats, ids_stats = await asyncio.to_thread(_run)
        state.stats["identifier_embeddings_updated"] = ident_stats.get("updated", 0)
        state.stats["identifier_embeddings_cached"] = ident_stats.get("cached", 0)
        state.stats["ids_embeddings_updated"] = ids_stats.get("updated", 0)
        state.stats["ids_embeddings_cached"] = ids_stats.get("cached", 0)
        state.aux_embedding_done = True


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
        lambda: dd_graph_ops.has_pending_embedding() or not state.enrich_phase.done
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
            depends_on=["build_phase"],
        ),
        WorkerSpec(
            "embed",
            "embed_phase",
            embed_worker,
            count=4,
            depends_on=["build_phase"],
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

    state.stats["elapsed_seconds"] = time.time() - state.build_start_time

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
