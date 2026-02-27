"""Async workers for parallel code discovery.

Workers that process code files through the pipeline:
- scan_worker: SSH file enumeration + rg enrichment (FacilityPaths → CodeFile nodes)
- score_worker: Dual-pass LLM scoring (triage → score) of discovered CodeFiles
- code_worker: Code ingestion — fetch, chunk, embed (scored → ingested)
- enrich_worker: rg pattern matching on scored CodeFiles (legacy backfill)

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
    (single SSH call per batch), creates CodeFile nodes.
    """
    from imas_codex.discovery.code.graph_ops import (
        claim_paths_for_file_scan,
        mark_path_file_scanned,
        release_path_file_scan_claim,
    )
    from imas_codex.discovery.code.scanner import (
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
            if state.scan_phase.done:
                break
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
                            f"found {persist_result.get('discovered', 0)} files",
                            state.scan_stats,
                            [
                                {
                                    "path": path,
                                    "files_found": persist_result.get("discovered", 0),
                                }
                            ],
                        )
                else:
                    # Mark path as scanned even with 0 files to prevent re-scanning
                    if path_id:
                        await asyncio.to_thread(mark_path_file_scanned, path_id, 0)
                    if on_progress:
                        on_progress(
                            "no files",
                            state.scan_stats,
                            [{"path": path, "files_found": 0}],
                        )

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
    """Score worker: Dual-pass LLM scoring of discovered CodeFiles.

    Pass 1 (Triage): Fast keep/skip classification from path + per-file
    enrichment evidence (rg matches). Filters ~70-80% of noise.

    Pass 2 (Score): Full multi-dimensional scoring for kept files.

    Claims CodeFiles with status='discovered' and no interest_score,
    groups by parent FacilityPath for context.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.discovery.code.scorer import (
        FileScoreBatch,
        FileTriageBatch,
        _build_system_prompt,
        _build_triage_system_prompt,
        _build_triage_user_prompt,
        _build_user_prompt,
        _group_files_by_parent,
        apply_file_scores,
        apply_triage_results,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    # Build system prompts once for prefix caching
    triage_system_prompt = _build_triage_system_prompt(focus=state.focus)
    score_system_prompt = _build_system_prompt(focus=state.focus)

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Claim files (joined with parent FacilityPath enrichment data)
        files = await asyncio.to_thread(
            claim_files_for_scoring, state.facility, limit=batch_size
        )

        if not files:
            state.score_phase.record_idle()
            if state.score_phase.done:
                break
            if on_progress:
                on_progress("idle", state.score_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.score_phase.record_activity(len(files))

        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]

        # Group by parent path for enrichment context
        file_groups = _group_files_by_parent(files)

        if on_progress:
            on_progress(
                f"triaging {len(files)} files ({len(file_groups)} dirs)",
                state.score_stats,
                None,
            )

        # --- Pass 1: Triage ---
        try:
            triage_user_prompt = _build_triage_user_prompt(file_groups)
            triage_parsed, triage_cost, _ = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[
                    {"role": "system", "content": triage_system_prompt},
                    {"role": "user", "content": triage_user_prompt},
                ],
                response_model=FileTriageBatch,
                temperature=0.1,
            )
            state.score_stats.cost += triage_cost

            triage_applied = await asyncio.to_thread(
                apply_triage_results, triage_parsed.results, file_id_map
            )

            kept_ids = set(triage_applied["kept_ids"])
            skipped_count = triage_applied["skipped"]

            if on_progress and skipped_count > 0:
                on_progress(
                    f"triage: kept {len(kept_ids)}, skipped {skipped_count} (${triage_cost:.3f})",
                    state.score_stats,
                    None,
                )

            if not kept_ids:
                # All files triaged out
                state.score_stats.processed += skipped_count
                await asyncio.to_thread(release_file_score_claims, batch_ids)
                if on_progress:
                    on_progress(
                        f"all {len(files)} files triaged out",
                        state.score_stats,
                        None,
                    )
                await asyncio.sleep(0.1)
                continue

        except Exception as e:
            logger.error("Triage batch failed: %s — scoring all files", e)
            state.score_stats.errors += 1
            # Fail-open: score all files if triage fails
            kept_ids = set(file_id_map.values())

        # --- Pass 2: Score kept files ---
        kept_groups = []
        for group in file_groups:
            kept_files = [f for f in group["files"] if f["id"] in kept_ids]
            if kept_files:
                kept_groups.append({**group, "files": kept_files})

        if not kept_groups:
            await asyncio.to_thread(release_file_score_claims, batch_ids)
            await asyncio.sleep(0.1)
            continue

        if on_progress:
            kept_count = sum(len(g["files"]) for g in kept_groups)
            on_progress(
                f"scoring {kept_count} kept files ({len(kept_groups)} dirs)",
                state.score_stats,
                None,
            )

        try:
            score_user_prompt = _build_user_prompt(kept_groups)
            parsed, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[
                    {"role": "system", "content": score_system_prompt},
                    {"role": "user", "content": score_user_prompt},
                ],
                response_model=FileScoreBatch,
                temperature=0.1,
            )
            state.score_stats.cost += cost

            result = await asyncio.to_thread(
                apply_file_scores, parsed.results, file_id_map
            )
            state.score_stats.processed += result.get("scored", 0)

            await asyncio.to_thread(release_file_score_claims, batch_ids)

            if on_progress:
                score_results = [
                    {
                        "path": s.path,
                        "score": s.interest_score,
                        "category": s.file_category,
                        "description": s.description,
                        "skipped": s.skip,
                    }
                    for s in parsed.results
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


def _claim_code_files_for_ingestion(
    facility: str,
    limit: int = 20,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """Claim scored CodeFiles for ingestion.

    Claims CodeFiles with status='discovered' that have been scored.
    """
    from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS
    from imas_codex.graph import GraphClient

    cutoff = f"PT{DEFAULT_CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
              AND sf.interest_score IS NOT NULL
              AND sf.interest_score >= $min_score
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY sf.interest_score DESC LIMIT $limit
            SET sf.claimed_at = datetime()
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.interest_score AS interest_score
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
            cutoff=cutoff,
        )
        return list(result)


def _mark_files_ingested(file_ids: list[str]) -> int:
    """Mark CodeFiles as ingested after successful processing."""
    from imas_codex.graph import GraphClient

    if not file_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:CodeFile {id: fid})
            SET sf.status = 'ingested',
                sf.ingested_at = datetime(),
                sf.claimed_at = null
            RETURN count(sf) AS updated
            """,
            ids=file_ids,
        )
        return result[0]["updated"] if result else 0


def _mark_file_failed(file_id: str, error: str) -> None:
    """Mark a single CodeFile as failed."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sf:CodeFile {id: $id})
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

    Claims scored CodeFiles with file_category='code', runs the
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
            _claim_code_files_for_ingestion,
            state.facility,
            limit=batch_size,
        )

        if not files:
            state.code_phase.record_idle()
            if state.code_phase.done:
                break
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
                    f"ingested {stats.get('files', 0)} code files",
                    state.code_stats,
                    [
                        {
                            "path": f["path"],
                            "language": f.get("language", ""),
                            "file_type": "code",
                        }
                        for f in files
                    ],
                )

        except Exception as e:
            logger.error("Code ingestion batch failed: %s", e)
            state.code_stats.errors += 1
            # Mark individual files as failed
            for f in files:
                await asyncio.to_thread(_mark_file_failed, f["id"], str(e)[:200])

        await asyncio.sleep(0.1)


# ============================================================================
# Enrich Worker (rg pattern matching on individual files)
# ============================================================================


async def enrich_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 100,
) -> None:
    """Enrich worker: rg pattern matching on scored CodeFiles.

    Claims scored CodeFiles that haven't been enriched yet, runs
    batched rg pattern matching via SSH, stores per-file pattern
    evidence on CodeFile nodes.

    Runs AFTER scoring, in parallel with code/docs ingestion workers.
    Enrichment data improves ingestion prioritization — files with
    actual pattern evidence are ingested first.
    """
    from imas_codex.discovery.code.enrichment import (
        enrich_files,
        persist_file_enrichment,
    )
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_enrichment,
        release_file_enrich_claims,
    )

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        files = await asyncio.to_thread(
            claim_files_for_enrichment,
            state.facility,
            limit=batch_size,
        )

        if not files:
            state.enrich_phase.record_idle()
            if state.enrich_phase.done:
                break
            if on_progress:
                on_progress("idle", state.enrich_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.enrich_phase.record_activity(len(files))
        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]
        file_paths = [f["path"] for f in files]

        if on_progress:
            on_progress(
                f"enriching {len(files)} files",
                state.enrich_stats,
                None,
            )

        try:
            results = await enrich_files(
                state.facility,
                file_paths,
            )

            enriched = await asyncio.to_thread(
                persist_file_enrichment, results, file_id_map
            )

            state.enrich_stats.processed += enriched

            # Release claims
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)

            if on_progress:
                # Stream all enriched files (with and without patterns)
                on_progress(
                    f"enriched {enriched}",
                    state.enrich_stats,
                    [
                        {
                            "path": r["path"],
                            "patterns": r.get("total_pattern_matches", 0),
                        }
                        for r in results
                    ],
                )

        except Exception as e:
            logger.error("File enrichment batch failed: %s", e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)

        await asyncio.sleep(0.1)
