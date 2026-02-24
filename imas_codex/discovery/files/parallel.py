"""Parallel file discovery engine.

Main entry point for file discovery with async workers. Orchestrates:
- Scan: SSH file enumeration from scored FacilityPaths
- Score: LLM batch scoring of discovered SourceFiles
- Code: Fetch, chunk, embed code files (replaces ``ingest run``)
- Docs: Ingest non-code files (documents, notebooks, configs)

Use ``run_parallel_file_discovery()`` as the main entry point.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.supervision import (
    OrphanRecoverySpec,
    SupervisedWorkerGroup,
    make_orphan_recovery_tick,
    run_supervised_loop,
    supervised_worker,
)
from imas_codex.graph import GraphClient

from .graph_ops import reset_orphaned_file_claims
from .state import FileDiscoveryState
from .workers import code_worker, docs_worker, scan_worker, score_worker

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


async def run_parallel_file_discovery(
    facility: str,
    ssh_host: str,
    *,
    cost_limit: float = 5.0,
    min_score: float = 0.5,
    max_paths: int = 100,
    focus: str | None = None,
    num_scan_workers: int = 2,
    num_score_workers: int = 2,
    num_code_workers: int = 2,
    num_docs_workers: int = 1,
    scan_only: bool = False,
    score_only: bool = False,
    deadline: float | None = None,
    on_scan_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_code_progress: Callable | None = None,
    on_docs_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
) -> dict[str, Any]:
    """Run parallel file discovery with async workers.

    Orchestrates four worker types through the file discovery pipeline:
    1. Scan workers: SSH file enumeration from scored FacilityPaths
    2. Score workers: LLM batch scoring of discovered SourceFiles
    3. Code workers: Fetch, chunk, embed code files (ingestion)
    4. Docs workers: Ingest non-code files (documents, notebooks, configs)

    Args:
        facility: Facility ID
        ssh_host: SSH host alias for remote operations
        cost_limit: Maximum LLM spend in USD
        min_score: Minimum FacilityPath score for scanning
        max_paths: Maximum paths to scan per batch
        focus: Natural language focus for scoring
        num_scan_workers: Number of parallel scan workers
        num_score_workers: Number of parallel score workers
        num_code_workers: Number of parallel code workers
        num_docs_workers: Number of parallel docs workers
        scan_only: Only scan, skip scoring and ingestion
        score_only: Only score, skip scanning and ingestion
        deadline: Absolute time (epoch) when discovery should stop
        on_scan_progress: Callback for scan worker progress
        on_score_progress: Callback for score worker progress
        on_code_progress: Callback for code worker progress
        on_docs_progress: Callback for docs worker progress
        on_worker_status: Callback for worker status updates
        service_monitor: ServiceMonitor for health monitoring

    Returns:
        Dict with discovery statistics
    """
    start_time = time.time()

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
        max_paths=max_paths,
        focus=focus,
        deadline=deadline,
        scan_only=scan_only,
        score_only=score_only,
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

    worker_group = SupervisedWorkerGroup()

    # --- Scan workers (skip in score_only mode) ---
    if not score_only:
        for i in range(num_scan_workers):
            worker_name = f"scan_worker_{i}"
            status = worker_group.create_status(worker_name, group="scan")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        scan_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_scan_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.scan_phase.mark_done()

    # --- Score workers ---
    if not scan_only:
        for i in range(num_score_workers):
            worker_name = f"score_worker_{i}"
            status = worker_group.create_status(worker_name, group="triage")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        score_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_score_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.score_phase.mark_done()

    # --- Code workers (skip in scan_only and score_only modes) ---
    if not scan_only and not score_only:
        for i in range(num_code_workers):
            worker_name = f"code_worker_{i}"
            status = worker_group.create_status(worker_name, group="code")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        code_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_code_progress,
                        status_tracker=status,
                    )
                )
            )

        # --- Docs workers ---
        for i in range(num_docs_workers):
            worker_name = f"docs_worker_{i}"
            status = worker_group.create_status(worker_name, group="docs")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        docs_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_docs_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.code_phase.mark_done()
        state.docs_phase.mark_done()

    logger.info(
        "Started %d workers: scan=%d score=%d code=%d docs=%d "
        "scan_only=%s score_only=%s",
        worker_group.get_active_count(),
        num_scan_workers if not score_only else 0,
        num_score_workers if not scan_only else 0,
        num_code_workers if not (scan_only or score_only) else 0,
        num_docs_workers if not (scan_only or score_only) else 0,
        scan_only,
        score_only,
    )

    # Periodic orphan recovery (every 60s)
    orphan_tick = make_orphan_recovery_tick(
        facility,
        [
            OrphanRecoverySpec("SourceFile"),
            OrphanRecoverySpec(
                "FacilityPath",
                timeout_seconds=300,
            ),
        ],
    )

    await run_supervised_loop(
        worker_group,
        state.should_stop,
        on_worker_status=on_worker_status,
        on_tick=orphan_tick,
    )
    state.stop_requested = True

    elapsed = time.time() - start_time

    return {
        "scanned": state.scan_stats.processed,
        "scored": state.score_stats.processed,
        "code_ingested": state.code_stats.processed,
        "docs_ingested": state.docs_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_errors": state.scan_stats.errors,
        "score_errors": state.score_stats.errors,
        "code_errors": state.code_stats.errors,
    }


def get_file_discovery_stats(facility: str) -> dict[str, int | float]:
    """Get file discovery statistics from graph for progress display."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WITH sf.status AS status, sf.file_category AS category,
                 sf.interest_score AS score
            RETURN status, category,
                   count(*) AS count,
                   avg(score) AS avg_score
            """,
            facility=facility,
        )

        stats: dict[str, int | float] = {
            "total": 0,
            "discovered": 0,
            "ingested": 0,
            "failed": 0,
            "skipped": 0,
            "pending_score": 0,
            "pending_ingest": 0,
            "code_files": 0,
            "document_files": 0,
            "notebook_files": 0,
            "config_files": 0,
        }

        for r in result:
            status = r["status"]
            category = r["category"] or "unknown"
            count = r["count"]
            stats["total"] += count

            if status in stats:
                stats[status] += count

            # Category counts
            cat_key = f"{category}_files"
            if cat_key in stats:
                stats[cat_key] += count

        # Pending score: discovered without interest_score
        score_result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered' AND sf.interest_score IS NULL
            RETURN count(sf) AS pending
            """,
            facility=facility,
        )
        stats["pending_score"] = score_result[0]["pending"] if score_result else 0

        # Pending ingest: scored code files not yet ingested
        ingest_result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.interest_score >= 0.3
              AND sf.file_category = 'code'
            RETURN count(sf) AS pending
            """,
            facility=facility,
        )
        stats["pending_ingest"] = ingest_result[0]["pending"] if ingest_result else 0

        return stats
