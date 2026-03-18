"""Parallel code discovery engine.

Main entry point for code discovery with async workers. Orchestrates:
- Scan: SSH code file enumeration (FacilityPaths → CodeFile nodes)
- Triage: Per-dimension LLM triage (discovered → triaged | skipped)
- Enrich: rg pattern matching + preview extraction (triaged → enriched)
- Score: Full LLM scoring (enriched → scored)
- Code: Fetch, tree-sitter chunk, embed code files (scored → ingested)

Use ``run_parallel_code_discovery()`` as the main entry point.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine
from imas_codex.discovery.base.supervision import OrphanRecoverySpec
from imas_codex.graph import GraphClient

from .graph_ops import (
    has_pending_code_work,
    has_pending_enrich_work,
    has_pending_link_work,
    has_pending_scan_work,
    has_pending_score_work,
    has_pending_triage_work,
    reset_orphaned_file_claims,
)
from .state import FileDiscoveryState
from .workers import (
    code_worker,
    enrich_worker,
    link_worker,
    scan_worker,
    score_worker,
    triage_worker,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.discovery.base.supervision import SupervisedWorkerGroup

logger = logging.getLogger(__name__)


async def run_parallel_code_discovery(
    facility: str,
    ssh_host: str,
    *,
    cost_limit: float = 5.0,
    min_score: float | None = None,
    max_paths: int = 100,
    focus: str | None = None,
    num_scan_workers: int = 2,
    num_triage_workers: int = 2,
    num_enrich_workers: int = 2,
    num_score_workers: int = 2,
    num_code_workers: int = 2,
    scan_only: bool = False,
    score_only: bool = False,
    triage_batch_size: int | None = None,
    deadline: float | None = None,
    on_scan_progress: Callable | None = None,
    on_triage_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_code_progress: Callable | None = None,
    on_embed_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
    stop_event: asyncio.Event | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run parallel code discovery with async workers.

    Orchestrates workers through the code discovery pipeline:
    1. Scan workers: SSH code file enumeration (depth=1)
    2. Triage workers: Per-dimension LLM triage (discovered → triaged)
    3. Enrich workers: rg pattern matching + preview (triaged → enriched)
    4. Score workers: Full LLM scoring (enriched → scored)
    5. Code workers: Fetch, tree-sitter chunk, embed (scored → ingested)
    6. Link worker: Code evidence → signal propagation

    Args:
        facility: Facility ID
        ssh_host: SSH host alias for remote operations
        cost_limit: Maximum LLM spend in USD
        min_score: Minimum FacilityPath score for scanning
        max_paths: Maximum paths to scan per batch
        focus: Natural language focus for scoring
        num_scan_workers: Number of parallel scan workers
        num_triage_workers: Number of parallel triage workers
        num_enrich_workers: Number of parallel enrich workers
        num_score_workers: Number of parallel score workers
        num_code_workers: Number of parallel code workers
        scan_only: Only scan, skip triage/enrichment/scoring/ingestion
        score_only: Only triage/enrich/score, skip scanning and ingestion
        deadline: Absolute time (epoch) when discovery should stop
        on_scan_progress: Callback for scan worker progress
        on_triage_progress: Callback for triage worker progress
        on_score_progress: Callback for score worker progress
        on_enrich_progress: Callback for enrich worker progress
        on_code_progress: Callback for code worker progress
        on_embed_progress: Callback for chunk embedding worker progress
        on_worker_status: Callback for worker status updates
        service_monitor: ServiceMonitor for health monitoring

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

    # Resolve thresholds lazily so pyproject.toml changes take effect
    if min_score is None:
        from imas_codex.settings import get_discovery_threshold

        min_score = get_discovery_threshold()

    from imas_codex.settings import get_triage_threshold

    min_triage_score = get_triage_threshold()

    # Release orphaned claims from previous runs
    reset_orphaned_file_claims(facility, silent=True)

    # Ensure Facility node exists
    with GraphClient() as gc:
        gc.ensure_facility(facility)

    state = FileDiscoveryState(
        facility=facility,
        ssh_host=ssh_host,
        service_monitor=service_monitor,
        cost_limit=cost_limit,
        min_score=min_score,
        min_triage_score=min_triage_score,
        max_paths=max_paths,
        focus=focus,
        deadline=deadline,
        scan_only=scan_only,
        score_only=score_only,
    )

    # Wire up graph-backed has_work_fn on each phase.
    # Pipeline: scan → triage → enrich → score → code → link
    state.scan_phase.set_has_work_fn(lambda: has_pending_scan_work(facility, min_score))
    state.triage_phase.set_has_work_fn(
        lambda: has_pending_triage_work(facility) or not state.scan_phase.done
    )
    state.enrich_phase.set_has_work_fn(
        lambda: (
            has_pending_enrich_work(facility, min_triage_score)
            or not state.triage_phase.done
        )
    )
    state.score_phase.set_has_work_fn(
        lambda: has_pending_score_work(facility) or not state.enrich_phase.done
    )
    state.code_phase.set_has_work_fn(
        lambda: has_pending_code_work(facility, min_score) or not state.score_phase.done
    )
    # Link phase depends on code phase (propagates evidence after ingestion)
    state.link_phase.set_has_work_fn(
        lambda: has_pending_link_work(facility) or not state.code_phase.done
    )

    # Pre-warm SSH ControlMaster
    logger.info("Pre-warming SSH ControlMaster to %s...", ssh_host)
    try:
        await asyncio.to_thread(
            subprocess.run,
            ["ssh", "-O", "check", ssh_host],
            capture_output=True,
            timeout=10,
        )
        logger.info("SSH ControlMaster active for %s", ssh_host)
    except Exception:
        try:
            await asyncio.to_thread(
                subprocess.run,
                ["ssh", ssh_host, "true"],
                capture_output=True,
                timeout=30,
            )
            logger.info("SSH ControlMaster established for %s", ssh_host)
        except Exception as e:
            logger.warning("Failed to pre-warm SSH to %s: %s", ssh_host, e)

    # Declare workers
    workers = [
        WorkerSpec(
            "scan",
            "scan_phase",
            scan_worker,
            count=num_scan_workers,
            enabled=not score_only,
            on_progress=on_scan_progress,
        ),
        WorkerSpec(
            "triage",
            "triage_phase",
            triage_worker,
            count=num_triage_workers,
            enabled=not scan_only,
            on_progress=on_triage_progress,
            kwargs={"batch_size": triage_batch_size}
            if triage_batch_size is not None
            else {},
        ),
        WorkerSpec(
            "enrich",
            "enrich_phase",
            enrich_worker,
            count=num_enrich_workers,
            enabled=not scan_only,
            on_progress=on_enrich_progress,
        ),
        WorkerSpec(
            "score",
            "score_phase",
            score_worker,
            count=num_score_workers,
            enabled=not scan_only,
            on_progress=on_score_progress,
        ),
        WorkerSpec(
            "code",
            "code_phase",
            code_worker,
            count=num_code_workers,
            enabled=not scan_only and not score_only,
            on_progress=on_code_progress,
        ),
        WorkerSpec(
            "link",
            "link_phase",
            link_worker,
            enabled=not scan_only and not score_only,
            on_progress=on_code_progress,
        ),
    ]

    # --- Embed workers ---
    from imas_codex.discovery.base.embed_worker import (
        embed_description_worker,
        embed_text_worker,
    )

    # Description embeddings for CodeExample nodes
    workers.append(
        WorkerSpec(
            "embed",
            "enrich_phase",
            embed_description_worker,
            group="embed",
            kwargs={
                "labels": ["CodeExample"],
                "done_check": lambda: state.enrich_phase.done,
            },
        )
    )

    # Chunk text embeddings — picks up CodeChunk nodes written by the
    # ingestion pipeline with embedding=null and embeds them
    # asynchronously on the GPU.  Multiple workers keep the embed
    # server busy across GPUs; 2 per CLI instance × N concurrent
    # facility CLIs saturates the server without excessive Neo4j
    # contention.
    workers.append(
        WorkerSpec(
            "chunk_embed",
            "code_phase",
            embed_text_worker,
            group="embed",
            count=2,
            on_progress=on_embed_progress,
            kwargs={
                "labels": ["CodeChunk"],
                "done_check": lambda: state.code_phase.done,
            },
        )
    )

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        orphan_specs=[
            OrphanRecoverySpec("CodeFile"),
            OrphanRecoverySpec("FacilityPath", timeout_seconds=300),
        ],
        on_worker_status=on_worker_status,
    )

    elapsed = time.time() - start_time

    return {
        "scanned": state.scan_stats.processed,
        "triaged": state.triage_stats.processed,
        "scored": state.score_stats.processed,
        "enriched": state.enrich_stats.processed,
        "code_ingested": state.code_stats.processed,
        "signals_linked": state.link_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_errors": state.scan_stats.errors,
        "triage_errors": state.triage_stats.errors,
        "score_errors": state.score_stats.errors,
        "enrich_errors": state.enrich_stats.errors,
        "code_errors": state.code_stats.errors,
        "link_errors": state.link_stats.errors,
    }


def get_code_discovery_stats(
    facility: str,
    min_score: float | None = None,
    min_triage_score: float | None = None,
) -> dict[str, int | float]:
    """Get code discovery statistics from graph for progress display.

    Args:
        facility: Facility ID
        min_score: Minimum score threshold for pending counts.
            Defaults to ``get_discovery_threshold()``.
        min_triage_score: Minimum triage composite for enrich pending.
            Defaults to ``get_triage_threshold()``.
    """
    if min_score is None:
        from imas_codex.settings import get_discovery_threshold

        min_score = get_discovery_threshold()
    if min_triage_score is None:
        from imas_codex.settings import get_triage_threshold

        min_triage_score = get_triage_threshold()
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WITH cf.status AS status, cf.language AS language,
                 cf.score_composite AS score
            RETURN status, language,
                   count(*) AS count,
                   avg(score) AS avg_score
            """,
            facility=facility,
        )

        stats: dict[str, int | float] = {
            "total": 0,
            "discovered": 0,
            "triaged": 0,
            "scored": 0,
            "ingested": 0,
            "failed": 0,
            "skipped": 0,
            "pending_triage": 0,
            "pending_enrich": 0,
            "pending_score": 0,
            "pending_ingest": 0,
        }

        for r in result:
            status = r["status"]
            count = r["count"]
            stats["total"] += count

            if status in stats:
                stats[status] += count

            # Language counts
            lang = r["language"] or "unknown"
            lang_key = f"{lang}_files"
            stats[lang_key] = stats.get(lang_key, 0) + count

        # Pending triage: discovered without triage_composite
        triage_result = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.status = 'discovered' AND cf.triage_composite IS NULL
            RETURN count(cf) AS pending
            """,
            facility=facility,
        )
        stats["pending_triage"] = triage_result[0]["pending"] if triage_result else 0

        # Pending enrich: triaged but not enriched (above triage threshold)
        enrich_pending = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.status = 'triaged'
              AND cf.triage_composite >= $min_triage
              AND coalesce(cf.is_enriched, false) = false
            RETURN count(cf) AS pending
            """,
            facility=facility,
            min_triage=min_triage_score,
        )
        stats["pending_enrich"] = enrich_pending[0]["pending"] if enrich_pending else 0

        # Pending score: triaged + enriched
        score_pending = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.status = 'triaged'
              AND cf.is_enriched = true
            RETURN count(cf) AS pending
            """,
            facility=facility,
        )
        stats["pending_score"] = score_pending[0]["pending"] if score_pending else 0

        # Pending ingest: scored code files (consistent with claim filters)
        ingest_result = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.status = 'scored'
              AND cf.score_composite >= $min_score
              AND coalesce(cf.line_count, 0) <= 10000
            RETURN count(cf) AS pending
            """,
            facility=facility,
            min_score=min_score,
        )
        stats["pending_ingest"] = ingest_result[0]["pending"] if ingest_result else 0

        # Enriched count
        enriched_result = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.is_enriched = true
            RETURN count(cf) AS enriched
            """,
            facility=facility,
        )
        stats["enriched_count"] = (
            enriched_result[0]["enriched"] if enriched_result else 0
        )

        # Accumulated LLM cost from graph (source of truth across runs)
        cost_result = gc.query(
            """
            MATCH (cf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE cf.score_cost IS NOT NULL
            RETURN sum(cf.score_cost) AS accumulated_cost
            """,
            facility=facility,
        )
        stats["accumulated_cost"] = (
            cost_result[0]["accumulated_cost"] or 0.0 if cost_result else 0.0
        )

        # Chunk embedding stats — only count chunks from CodeFiles
        # meeting the score threshold so pending counts are consistent
        # with what the embed workers will actually process.
        # Exclude whitespace-only chunks from pending (they can't be
        # meaningfully embedded and the workers skip them).
        embed_result = gc.query(
            """
            MATCH (cc:CodeChunk)-[:AT_FACILITY]->(f:Facility {id: $facility})
            MATCH (cc)<-[:HAS_CHUNK]-(:CodeExample)<-[:HAS_EXAMPLE]-(cf:CodeFile)
            WHERE cf.score_composite >= $min_score
            RETURN count(cc) AS total,
                   count(cc.embedding) AS embedded,
                   count(CASE WHEN cc.embedding IS NULL
                              AND cc.embed_failed_at IS NULL
                              AND cc.text IS NOT NULL
                              AND trim(cc.text) <> ''
                         THEN 1 END) AS pending
            """,
            facility=facility,
            min_score=min_score,
        )
        if embed_result:
            total_chunks = embed_result[0]["total"]
            embedded_chunks = embed_result[0]["embedded"]
            stats["total_chunks"] = total_chunks
            stats["embedded_chunks"] = embedded_chunks
            stats["pending_embed"] = embed_result[0]["pending"]
        else:
            stats["total_chunks"] = 0
            stats["embedded_chunks"] = 0
            stats["pending_embed"] = 0

        return stats
