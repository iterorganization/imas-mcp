"""Parallel file discovery engine.

Main entry point for file discovery with async workers. Orchestrates:
- Scan: SSH file enumeration from scored FacilityPaths
- Score: LLM batch scoring of discovered SourceFiles
- Enrich: rg pattern matching on individual scored files
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

from .graph_ops import (
    has_pending_code_work,
    has_pending_docs_work,
    has_pending_enrich_work,
    has_pending_image_score_work,
    has_pending_image_work,
    has_pending_scan_work,
    has_pending_score_work,
    reset_orphaned_file_claims,
)
from .state import FileDiscoveryState
from .workers import (
    code_worker,
    docs_worker,
    enrich_worker,
    image_score_worker,
    image_worker,
    scan_worker,
    score_worker,
)

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
    num_enrich_workers: int = 1,
    num_code_workers: int = 2,
    num_docs_workers: int = 1,
    num_image_workers: int = 1,
    num_image_score_workers: int = 1,
    scan_only: bool = False,
    score_only: bool = False,
    store_images: bool = False,
    deadline: float | None = None,
    on_scan_progress: Callable | None = None,
    on_score_progress: Callable | None = None,
    on_enrich_progress: Callable | None = None,
    on_code_progress: Callable | None = None,
    on_docs_progress: Callable | None = None,
    on_image_progress: Callable | None = None,
    on_image_score_progress: Callable | None = None,
    on_worker_status: Callable[[SupervisedWorkerGroup], None] | None = None,
    service_monitor: Any = None,
) -> dict[str, Any]:
    """Run parallel file discovery with async workers.

    Orchestrates six worker types through the file discovery pipeline:
    1. Scan workers: SSH file enumeration from scored FacilityPaths
    2. Score workers: LLM batch scoring of discovered SourceFiles
    3. Enrich workers: rg pattern matching on scored files
    4. Code workers: Fetch, chunk, embed code files (ingestion)
    5. Docs workers: Ingest non-code files (documents, notebooks, configs)
    6. Image workers: Downsample and persist standalone image files
    7. Image score workers: VLM captioning + scoring of ingested images

    Args:
        facility: Facility ID
        ssh_host: SSH host alias for remote operations
        cost_limit: Maximum LLM spend in USD
        min_score: Minimum FacilityPath score for scanning
        max_paths: Maximum paths to scan per batch
        focus: Natural language focus for scoring
        num_scan_workers: Number of parallel scan workers
        num_score_workers: Number of parallel score workers
        num_enrich_workers: Number of parallel enrich workers
        num_code_workers: Number of parallel code workers
        num_docs_workers: Number of parallel docs workers
        num_image_score_workers: Number of parallel VLM image scoring workers
        scan_only: Only scan, skip scoring and ingestion
        score_only: Only score, skip scanning and ingestion
        store_images: Keep image bytes in graph after VLM scoring
        deadline: Absolute time (epoch) when discovery should stop
        on_scan_progress: Callback for scan worker progress
        on_score_progress: Callback for score worker progress
        on_enrich_progress: Callback for enrich worker progress
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
        store_images=store_images,
    )

    # Wire up graph-backed has_work_fn on each phase.
    # Downstream phases also check if upstream is still producing work,
    # preventing premature exit when upstream hasn't flushed yet.
    state.scan_phase.set_has_work_fn(lambda: has_pending_scan_work(facility, min_score))
    state.score_phase.set_has_work_fn(
        lambda: has_pending_score_work(facility) or not state.scan_phase.done
    )
    state.enrich_phase.set_has_work_fn(
        lambda: has_pending_enrich_work(facility) or not state.score_phase.done
    )
    state.code_phase.set_has_work_fn(
        lambda: has_pending_code_work(facility) or not state.enrich_phase.done
    )
    state.docs_phase.set_has_work_fn(
        lambda: has_pending_docs_work(facility) or not state.enrich_phase.done
    )
    state.image_phase.set_has_work_fn(
        lambda: has_pending_image_work(facility) or not state.enrich_phase.done
    )
    state.image_score_phase.set_has_work_fn(
        lambda: has_pending_image_score_work(facility) or not state.image_phase.done
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
        # --- Enrich workers ---
        for i in range(num_enrich_workers):
            worker_name = f"enrich_worker_{i}"
            status = worker_group.create_status(worker_name, group="enrich")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        enrich_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_enrich_progress,
                        status_tracker=status,
                    )
                )
            )

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

        # --- Image workers ---
        for i in range(num_image_workers):
            worker_name = f"image_worker_{i}"
            status = worker_group.create_status(worker_name, group="image")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        image_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_image_progress,
                        status_tracker=status,
                    )
                )
            )

        # --- Image score workers (VLM captioning) ---
        for i in range(num_image_score_workers):
            worker_name = f"image_score_worker_{i}"
            status = worker_group.create_status(worker_name, group="vlm")
            worker_group.add_task(
                asyncio.create_task(
                    supervised_worker(
                        image_score_worker,
                        worker_name,
                        state,
                        state.should_stop,
                        on_progress=on_image_score_progress,
                        status_tracker=status,
                    )
                )
            )
    else:
        state.code_phase.mark_done()
        state.docs_phase.mark_done()
        state.enrich_phase.mark_done()
        state.image_phase.mark_done()
        state.image_score_phase.mark_done()

    logger.info(
        "Started %d workers: scan=%d score=%d enrich=%d code=%d docs=%d "
        "image=%d vlm=%d scan_only=%s score_only=%s",
        worker_group.get_active_count(),
        num_scan_workers if not score_only else 0,
        num_score_workers if not scan_only else 0,
        num_enrich_workers if not (scan_only or score_only) else 0,
        num_code_workers if not (scan_only or score_only) else 0,
        num_docs_workers if not (scan_only or score_only) else 0,
        num_image_workers if not (scan_only or score_only) else 0,
        num_image_score_workers if not (scan_only or score_only) else 0,
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
        "enriched": state.enrich_stats.processed,
        "code_ingested": state.code_stats.processed,
        "docs_ingested": state.docs_stats.processed,
        "images_ingested": state.image_stats.processed,
        "images_scored": state.image_score_stats.processed,
        "cost": state.total_cost,
        "elapsed_seconds": elapsed,
        "scan_errors": state.scan_stats.errors,
        "score_errors": state.score_stats.errors,
        "enrich_errors": state.enrich_stats.errors,
        "code_errors": state.code_stats.errors,
        "image_errors": state.image_stats.errors,
        "image_score_errors": state.image_score_stats.errors,
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

        # Scored and enriched counts for progress display
        enrich_result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.interest_score IS NOT NULL
            RETURN count(sf) AS scored,
                   count(CASE WHEN coalesce(sf.is_enriched, false) = true
                              THEN 1 END) AS enriched
            """,
            facility=facility,
        )
        if enrich_result:
            stats["scored_count"] = enrich_result[0]["scored"]
            stats["enriched_count"] = enrich_result[0]["enriched"]
        else:
            stats["scored_count"] = 0
            stats["enriched_count"] = 0

        # Image counts for progress display
        image_result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            RETURN count(img) AS total_images,
                   count(CASE WHEN img.description IS NOT NULL
                              THEN 1 END) AS scored_images
            """,
            facility=facility,
        )
        if image_result:
            stats["total_images"] = image_result[0]["total_images"]
            stats["scored_images"] = image_result[0]["scored_images"]
        else:
            stats["total_images"] = 0
            stats["scored_images"] = 0

        return stats
