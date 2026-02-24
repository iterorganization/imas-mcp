"""Async workers for parallel file discovery.

Four supervised workers that process source files through the pipeline:
- scan_worker: SSH file enumeration (discovered FacilityPaths → SourceFile nodes)
- score_worker: LLM batch scoring (discovered → scored SourceFiles)
- code_worker: Code ingestion — fetch, chunk, embed (scored → ingested)
- docs_worker: Document/notebook/config ingestion (scored → ingested)

Workers coordinate through graph_ops claim/mark functions using claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .state import FileDiscoveryState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Scan Worker
# ============================================================================


async def scan_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Scan worker: SSH file enumeration from scored FacilityPaths.

    Claims FacilityPaths via files_claimed_at, runs batched SSH file listing
    (single SSH call per batch), creates SourceFile nodes.
    """
    from imas_codex.discovery.files.graph_ops import (
        claim_paths_for_file_scan,
        mark_path_file_scanned,
        release_path_file_scan_claim,
    )
    from imas_codex.discovery.files.scanner import (
        _persist_discovered_files,
        async_scan_remote_paths_batch,
    )
    from imas_codex.graph import GraphClient

    # Ensure Facility node exists
    with GraphClient() as gc:
        gc.ensure_facility(state.facility)

    ssh_retry_count = 0
    max_ssh_retries = 5

    while not state.should_stop():
        # Claim paths atomically
        paths = await asyncio.to_thread(
            claim_paths_for_file_scan,
            state.facility,
            min_score=state.min_score,
            limit=batch_size,
        )

        if not paths:
            state.scan_phase.record_idle()
            if on_progress:
                on_progress("idle", state.scan_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.scan_phase.record_activity(len(paths))

        # Build path list for batched SSH scan
        path_map = {p["path"]: p for p in paths}
        path_list = [p["path"] for p in paths]

        if on_progress:
            scores = [f"{p.get('score', 0):.2f}" for p in paths]
            on_progress(
                f"scanning {len(paths)} paths (scores: {', '.join(scores[:3])}...)",
                state.scan_stats,
                None,
            )

        try:
            # Single SSH call for the entire batch
            result_map = await async_scan_remote_paths_batch(
                state.facility,
                path_list,
                ssh_host=state.ssh_host,
            )

            # Reset retry count on success
            ssh_retry_count = 0

            # Process results per path
            for path, files in result_map.items():
                if state.should_stop():
                    break

                path_info = path_map.get(path)
                path_id = path_info["id"] if path_info else None

                if files:
                    persist_result = await asyncio.to_thread(
                        _persist_discovered_files,
                        state.facility,
                        files,
                        source_path_id=path_id,
                    )
                    state.scan_stats.processed += persist_result.get("discovered", 0)

                    # Mark path as scanned with file count
                    if path_id:
                        await asyncio.to_thread(
                            mark_path_file_scanned, path_id, len(files)
                        )

                    if on_progress:
                        on_progress(
                            f"found {persist_result.get('discovered', 0)} files in {path}",
                            state.scan_stats,
                            [{"path": f["path"]} for f in files[:5]],
                        )
                else:
                    # Mark path as scanned even with 0 files to prevent re-scanning
                    if path_id:
                        await asyncio.to_thread(mark_path_file_scanned, path_id, 0)
                    if on_progress:
                        on_progress(f"no files in {path}", state.scan_stats, None)

                # Release claim after processing
                if path_id:
                    await asyncio.to_thread(release_path_file_scan_claim, path_id)

        except Exception as e:
            ssh_retry_count += 1
            logger.warning(
                "SSH scan failed (%d/%d): %s", ssh_retry_count, max_ssh_retries, e
            )
            state.scan_stats.errors += len(paths)

            # Release all claims on error
            for p in paths:
                await asyncio.to_thread(release_path_file_scan_claim, p["id"])

            if ssh_retry_count >= max_ssh_retries:
                logger.error(
                    "SSH connection failed after %d attempts. Scan worker stopping.",
                    max_ssh_retries,
                )
                state.scan_phase.mark_done()
                if on_progress:
                    on_progress(
                        f"SSH failed: {str(e)[:100]}",
                        state.scan_stats,
                        None,
                    )
                break

            backoff = min(2**ssh_retry_count, 32)
            if on_progress:
                on_progress(
                    f"SSH retry {ssh_retry_count} in {backoff}s",
                    state.scan_stats,
                    None,
                )
            await asyncio.sleep(backoff)
            continue

        await asyncio.sleep(0.1)


# ============================================================================
# Score Worker
# ============================================================================


async def score_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 50,
) -> None:
    """Score worker: LLM batch scoring of discovered SourceFiles.

    Claims SourceFiles with status='discovered' and no interest_score,
    scores them via LLM, updates graph.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.files.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.discovery.files.scorer import (
        BatchScoreResult,
        _apply_scores,
        _build_scoring_prompt,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Claim files
        files = await asyncio.to_thread(
            claim_files_for_scoring, state.facility, limit=batch_size
        )

        if not files:
            state.score_phase.record_idle()
            if on_progress:
                on_progress("idle", state.score_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.score_phase.record_activity(len(files))

        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]

        if on_progress:
            on_progress(f"scoring {len(files)} files", state.score_stats, None)

        prompt = _build_scoring_prompt(files, state.facility, focus=state.focus)

        try:
            parsed, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=BatchScoreResult,
                temperature=0.1,
            )
            state.score_stats.cost += cost

            result = await asyncio.to_thread(_apply_scores, parsed.scores, file_id_map)
            state.score_stats.processed += result.get("scored", 0)

            await asyncio.to_thread(release_file_score_claims, batch_ids)

            if on_progress:
                score_results = [
                    {
                        "path": s.path,
                        "score": s.interest_score,
                        "category": s.file_category,
                    }
                    for s in parsed.scores[:5]
                ]
                on_progress(
                    f"scored {result.get('scored', 0)} (${cost:.3f})",
                    state.score_stats,
                    score_results,
                )

        except Exception as e:
            logger.error("Score batch failed: %s", e)
            state.score_stats.errors += 1
            await asyncio.to_thread(release_file_score_claims, batch_ids)

        await asyncio.sleep(0.1)


# ============================================================================
# Code Worker (ingestion)
# ============================================================================


def _claim_files_for_ingestion(
    facility: str,
    file_category: str = "code",
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Claim scored SourceFiles for ingestion.

    Claims files with status='discovered' that have been scored
    (interest_score is not null) and match the target category.
    """
    from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS
    from imas_codex.graph import GraphClient

    cutoff = f"PT{DEFAULT_CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:SourceFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.interest_score >= $min_score
              AND sf.file_category = $category
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY sf.interest_score DESC LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.file_category AS file_category,
                   sf.interest_score AS interest_score
            """,
            facility=facility,
            min_score=min_score,
            category=file_category,
            limit=limit,
            cutoff=cutoff,
        )
        return list(result)


def _mark_files_ingested(file_ids: list[str]) -> int:
    """Mark SourceFiles as ingested after successful processing."""
    from imas_codex.graph import GraphClient

    if not file_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:SourceFile {id: fid})
            SET sf.status = 'ingested',
                sf.ingested_at = datetime(),
                sf.claimed_at = null
            RETURN count(sf) AS updated
            """,
            ids=file_ids,
        )
        return result[0]["updated"] if result else 0


def _mark_file_failed(file_id: str, error: str) -> None:
    """Mark a single SourceFile as failed."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sf:SourceFile {id: $id})
            SET sf.status = 'failed',
                sf.error = $error,
                sf.claimed_at = null
            """,
            id=file_id,
            error=error[:200],
        )


async def code_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Code worker: Fetch, chunk, embed, and link code files.

    Claims scored SourceFiles with file_category='code', runs the
    ingestion pipeline (tree-sitter chunking, embedding, entity extraction).
    Transitions: discovered (scored) → ingested | failed

    This replaces the `ingest run` CLI command's core loop.
    """
    from imas_codex.ingestion.pipeline import ingest_files

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        # Claim code files for ingestion
        files = await asyncio.to_thread(
            _claim_files_for_ingestion,
            state.facility,
            file_category="code",
            limit=batch_size,
        )

        if not files:
            state.code_phase.record_idle()
            if on_progress:
                on_progress("idle", state.code_stats, None)
            await asyncio.sleep(3.0)
            continue

        state.code_phase.record_activity(len(files))

        if on_progress:
            on_progress(f"ingesting {len(files)} code files", state.code_stats, None)

        remote_paths = [f["path"] for f in files]
        file_id_map = {f["path"]: f["id"] for f in files}

        try:
            stats = await ingest_files(
                facility=state.facility,
                remote_paths=remote_paths,
                force=False,
            )

            ingested_ids = [file_id_map[p] for p in remote_paths if p in file_id_map]
            if ingested_ids:
                await asyncio.to_thread(_mark_files_ingested, ingested_ids)

            state.code_stats.processed += stats.get("files", 0)

            if on_progress:
                on_progress(
                    f"ingested {stats.get('files', 0)} "
                    f"({stats.get('chunks', 0)} chunks)",
                    state.code_stats,
                    [
                        {"path": p, "chunks": stats.get("chunks", 0)}
                        for p in remote_paths[:3]
                    ],
                )

        except Exception as e:
            logger.error("Code ingestion batch failed: %s", e)
            state.code_stats.errors += 1
            # Mark individual files as failed
            for f in files:
                await asyncio.to_thread(_mark_file_failed, f["id"], str(e)[:200])

        await asyncio.sleep(0.5)


# ============================================================================
# Docs Worker (documents, notebooks, configs)
# ============================================================================


async def docs_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Docs worker: Ingest non-code files (documents, notebooks, configs).

    Claims scored SourceFiles with file_category in ('document', 'notebook',
    'config'), runs appropriate ingestion pipeline.
    Transitions: discovered (scored) → ingested | failed
    """
    from imas_codex.ingestion.pipeline import ingest_files

    # Doc categories to process (everything except 'code')
    doc_categories = ["document", "notebook", "config"]

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        # Try each category
        files: list[dict] = []
        for category in doc_categories:
            if files:
                break
            files = await asyncio.to_thread(
                _claim_files_for_ingestion,
                state.facility,
                file_category=category,
                limit=batch_size,
            )

        if not files:
            state.docs_phase.record_idle()
            if on_progress:
                on_progress("idle", state.docs_stats, None)
            await asyncio.sleep(3.0)
            continue

        state.docs_phase.record_activity(len(files))
        category = files[0].get("file_category", "document")

        if on_progress:
            on_progress(
                f"ingesting {len(files)} {category} files",
                state.docs_stats,
                None,
            )

        remote_paths = [f["path"] for f in files]
        file_id_map = {f["path"]: f["id"] for f in files}

        try:
            stats = await ingest_files(
                facility=state.facility,
                remote_paths=remote_paths,
                force=False,
            )

            ingested_ids = [file_id_map[p] for p in remote_paths if p in file_id_map]
            if ingested_ids:
                await asyncio.to_thread(_mark_files_ingested, ingested_ids)

            state.docs_stats.processed += stats.get("files", 0)

            if on_progress:
                on_progress(
                    f"ingested {stats.get('files', 0)} {category} files",
                    state.docs_stats,
                    [{"path": p, "category": category} for p in remote_paths[:3]],
                )

        except Exception as e:
            logger.error("Docs ingestion batch failed: %s", e)
            state.docs_stats.errors += 1
            for f in files:
                await asyncio.to_thread(_mark_file_failed, f["id"], str(e)[:200])

        await asyncio.sleep(0.5)
