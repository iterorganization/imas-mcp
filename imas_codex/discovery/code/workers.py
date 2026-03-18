"""Async workers for parallel code discovery.

Workers that process code files through the pipeline:
- scan_worker: SSH file enumeration (FacilityPaths → CodeFile nodes)
- triage_worker: LLM dimension triage (discovered → triaged | skipped)
- enrich_worker: rg pattern matching + preview extraction (triaged → enriched)
- score_worker: LLM full scoring (enriched → scored)
- code_worker: Code ingestion — fetch, chunk, embed (scored → ingested)

Workers coordinate through graph_ops claim/mark functions using claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.base.supervision import is_infrastructure_error

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
        _persist_code_files,
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
                        _persist_code_files,
                        state.facility,
                        files,
                        source_path_id=path_id,
                    )
                    state.scan_stats.processed += persist_result.get("discovered", 0)
                    state.scan_stats.record_batch(persist_result.get("discovered", 0))

                    # Mark path as scanned with file count
                    if path_id:
                        await asyncio.to_thread(
                            mark_path_file_scanned, path_id, len(files)
                        )

                    if on_progress:
                        path_info = path_map.get(path, {})
                        on_progress(
                            f"found {persist_result.get('discovered', 0)} files",
                            state.scan_stats,
                            [
                                {
                                    "path": path,
                                    "files_found": persist_result.get("discovered", 0),
                                    "score_composite": path_info.get("score"),
                                }
                            ],
                        )
                else:
                    # Mark path as scanned even with 0 files to prevent re-scanning
                    if path_id:
                        await asyncio.to_thread(mark_path_file_scanned, path_id, 0)
                    if on_progress:
                        path_info = path_map.get(path, {})
                        on_progress(
                            "no files",
                            state.scan_stats,
                            [
                                {
                                    "path": path,
                                    "files_found": 0,
                                    "score_composite": path_info.get("score"),
                                }
                            ],
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
# Triage Worker
# ============================================================================


async def triage_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 50,
) -> None:
    """Triage worker: Per-dimension LLM triage of discovered CodeFiles.

    Claims CodeFiles with status='discovered', groups by parent
    FacilityPath for batch context, calls LLM for per-dimension triage
    scores.  Files scoring above a composite threshold are set to
    status='triaged'; below threshold → status='skipped'.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_triage,
        release_file_triage_claims,
    )
    from imas_codex.discovery.code.scorer import (
        FileTriageBatch,
        _build_triage_system_prompt,
        _build_triage_user_prompt,
        _group_files_by_parent,
        apply_triage_results,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    import time as _time

    _prompt_built_at = _time.monotonic()
    _PROMPT_REBUILD_INTERVAL = 60.0

    triage_system_prompt = _build_triage_system_prompt(
        facility=state.facility, focus=state.focus
    )

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.triage_stats, None)
            break

        # Rebuild system prompt periodically (picks up new calibration)
        if (_time.monotonic() - _prompt_built_at) > _PROMPT_REBUILD_INTERVAL:
            triage_system_prompt = _build_triage_system_prompt(
                facility=state.facility, focus=state.focus
            )
            _prompt_built_at = _time.monotonic()

        files = await asyncio.to_thread(
            claim_files_for_triage, state.facility, limit=batch_size
        )

        if not files:
            state.triage_phase.record_idle()
            if state.triage_phase.done:
                break
            if on_progress:
                on_progress("idle", state.triage_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.triage_phase.record_activity(len(files))

        file_id_map = {f["path"]: f["id"] for f in files}
        batch_ids = [f["id"] for f in files]

        # Group by parent path; include sibling names for context
        file_groups = _group_files_by_parent(files, include_siblings=True)

        if on_progress:
            on_progress(
                f"triaging {len(files)} files ({len(file_groups)} dirs)",
                state.triage_stats,
                None,
            )

        batch_start = _time.monotonic()

        try:
            triage_user_prompt = _build_triage_user_prompt(file_groups)
            triage_raw, triage_cost, _ = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[
                    {"role": "system", "content": triage_system_prompt},
                    {"role": "user", "content": triage_user_prompt},
                ],
                response_model=FileTriageBatch,
                temperature=0.1,
            )
            assert isinstance(triage_raw, FileTriageBatch)
            triage_parsed = triage_raw
            state.triage_stats.cost += triage_cost

            triage_applied = await asyncio.to_thread(
                apply_triage_results,
                triage_parsed.results,
                file_id_map,
                threshold=state.min_triage_score,
                batch_cost=triage_cost,
            )

            triaged = triage_applied["triaged"]
            skipped = triage_applied["skipped"]
            batch_total = triaged + skipped
            state.triage_stats.processed += batch_total
            state.triage_stats.last_batch_time = _time.monotonic() - batch_start
            state.triage_stats.record_batch(batch_total)

            if on_progress:
                # Stream per-file triage results with scores and descriptions
                triage_results = []
                for r in triage_parsed.results:
                    composite = r.triage_composite
                    # Find top dimension
                    dim_scores = {
                        "modeling": r.score_modeling_code,
                        "analysis": r.score_analysis_code,
                        "operations": r.score_operations_code,
                        "data_access": r.score_data_access,
                        "workflow": r.score_workflow,
                        "visualization": r.score_visualization,
                        "documentation": r.score_documentation,
                        "imas": r.score_imas,
                        "convention": r.score_convention,
                    }
                    top_dim = max(dim_scores, key=lambda k: dim_scores[k])
                    triage_results.append(
                        {
                            "path": r.path,
                            "triage_composite": round(composite, 3),
                            "description": r.description,
                            "category": top_dim,
                            "skipped": composite < state.min_triage_score,
                        }
                    )
                on_progress(
                    f"triaged {triaged}, skipped {skipped} (${triage_cost:.3f})",
                    state.triage_stats,
                    triage_results,
                )

            # Release claims (apply_triage_results already clears claimed_at
            # for triaged/skipped files, but release any unmatched ones)
            unmatched = set(batch_ids) - {
                f["id"]
                for f in files
                if f["path"] in {r.path for r in triage_parsed.results}
            }
            if unmatched:
                await asyncio.to_thread(release_file_triage_claims, list(unmatched))

        except Exception as e:
            logger.error("Triage batch failed: %s", e)
            state.triage_stats.errors += 1
            await asyncio.to_thread(release_file_triage_claims, batch_ids)
            if is_infrastructure_error(e):
                raise

        await asyncio.sleep(0.1)


# ============================================================================
# Score Worker
# ============================================================================


async def score_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 50,
) -> None:
    """Score worker: Full LLM scoring of enriched CodeFiles.

    Claims CodeFiles that have been triaged AND enriched
    (status='triaged', is_enriched=true).  The score prompt receives
    enrichment evidence (pattern matches, preview text) and
    the triage description (qualitative, NO triage numeric scores).
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.discovery.code.scorer import (
        FileScoreBatch,
        _build_score_system_prompt,
        _build_score_user_prompt,
        _group_files_by_parent,
        apply_file_scores,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    import time as _time

    _prompt_built_at = _time.monotonic()
    _PROMPT_REBUILD_INTERVAL = 60.0

    score_system_prompt = _build_score_system_prompt(
        facility=state.facility, focus=state.focus
    )

    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.score_stats, None)
            break

        # Rebuild system prompt periodically
        if (_time.monotonic() - _prompt_built_at) > _PROMPT_REBUILD_INTERVAL:
            score_system_prompt = _build_score_system_prompt(
                facility=state.facility, focus=state.focus
            )
            _prompt_built_at = _time.monotonic()

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

        # Group by parent path (no siblings needed — enrichment provides context)
        file_groups = _group_files_by_parent(files, include_siblings=False)

        if on_progress:
            on_progress(
                f"scoring {len(files)} files ({len(file_groups)} dirs)",
                state.score_stats,
                None,
            )

        batch_start = _time.monotonic()

        try:
            score_user_prompt = _build_score_user_prompt(file_groups)
            parsed_raw, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[
                    {"role": "system", "content": score_system_prompt},
                    {"role": "user", "content": score_user_prompt},
                ],
                response_model=FileScoreBatch,
                temperature=0.1,
            )
            assert isinstance(parsed_raw, FileScoreBatch)
            parsed = parsed_raw
            state.score_stats.cost += cost

            result = await asyncio.to_thread(
                apply_file_scores,
                parsed.results,
                file_id_map,
                batch_cost=cost,
            )
            batch_total = result.get("scored", 0) + result.get("skipped", 0)
            state.score_stats.processed += batch_total
            state.score_stats.last_batch_time = _time.monotonic() - batch_start
            state.score_stats.record_batch(batch_total)

            await asyncio.to_thread(release_file_score_claims, batch_ids)

            if on_progress:
                # Stream per-file score results with composite, category, description
                score_results = []
                for r in parsed.results:
                    score_results.append(
                        {
                            "path": r.path,
                            "score_composite": round(r.score_composite, 3),
                            "category": r.file_category,
                            "description": r.description,
                            "skipped": r.skip,
                        }
                    )
                on_progress(
                    f"scored {result.get('scored', 0)}, skipped {result.get('skipped', 0)} (${cost:.3f})",
                    state.score_stats,
                    score_results,
                )

        except Exception as e:
            logger.error("Score batch failed: %s", e)
            state.score_stats.errors += 1
            await asyncio.to_thread(release_file_score_claims, batch_ids)
            if is_infrastructure_error(e):
                raise

        await asyncio.sleep(0.1)


# ============================================================================
# Code Worker (ingestion)
# ============================================================================


@retry_on_deadlock()
def _claim_code_files_for_ingestion(
    facility: str,
    limit: int = 20,
    min_score: float | None = None,
    max_line_count: int = 10000,
) -> list[dict[str, Any]]:
    """Claim scored CodeFiles for ingestion.

    Claims CodeFiles with status='scored' above the minimum interest
    score threshold. Skips files exceeding max_line_count to avoid
    tree-sitter hangs on very large auto-generated files.

    Dedup is handled *after* claiming — see ``_filter_duplicates()``.
    Keeping the claim query simple avoids expensive correlated subqueries
    that scale as O(candidates × ingested_hashes).

    Uses anti-deadlock patterns: ORDER BY rand(), claim_token two-step
    verify, and @retry_on_deadlock decorator.
    """
    if min_score is None:
        from imas_codex.settings import get_discovery_threshold

        min_score = get_discovery_threshold()
    import uuid

    from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS
    from imas_codex.graph import GraphClient

    token = str(uuid.uuid4())
    cutoff = f"PT{DEFAULT_CLAIM_TIMEOUT_SECONDS}S"
    with GraphClient() as gc:
        # Step 1: Claim with random ordering and token
        gc.query(
            """
            MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WHERE sf.status = 'scored'
              AND sf.score_composite IS NOT NULL
              AND sf.score_composite >= $min_score
              AND coalesce(sf.line_count, 0) <= $max_line_count
              AND (sf.claimed_at IS NULL
                   OR sf.claimed_at < datetime() - duration($cutoff))
            WITH sf ORDER BY rand() LIMIT $limit
            SET sf.claimed_at = datetime(), sf.claim_token = $token
            """,
            facility=facility,
            min_score=min_score,
            max_line_count=max_line_count,
            limit=limit,
            cutoff=cutoff,
            token=token,
        )
        # Step 2: Read back by token to confirm claims
        result = gc.query(
            """
            MATCH (sf:CodeFile {claim_token: $token})
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.score_composite AS score_composite,
                   sf.content_hash AS content_hash
            """,
            token=token,
        )
        return list(result)


def _filter_duplicates(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out files whose content_hash is already ingested.

    This is a cheap post-claim dedup check: collects the unique hashes
    from the claimed batch, does a single ``IN $hashes`` lookup against
    ingested CodeFiles, and returns only the files that need processing.

    Files filtered out are immediately marked 'skipped' so they won't
    be claimed again.

    Returns:
        List of files that still need ingestion.
    """
    from imas_codex.graph import GraphClient

    # Collect hashes from the batch (skip files without a hash)
    hashed = {f["content_hash"]: f for f in files if f.get("content_hash")}
    if not hashed:
        return files  # nothing to dedup

    with GraphClient() as gc:
        # Single indexed lookup: which of these hashes are already ingested?
        result = gc.query(
            """
            UNWIND $hashes AS h
            MATCH (dup:CodeFile {content_hash: h})
            WHERE dup.status = 'ingested'
            RETURN DISTINCT h AS hash
            """,
            hashes=list(hashed.keys()),
        )
        already_ingested = {r["hash"] for r in result}

    if not already_ingested:
        return files

    # Split into keep vs skip
    keep = []
    skip_ids = []
    for f in files:
        h = f.get("content_hash")
        if h and h in already_ingested:
            skip_ids.append(f["id"])
        else:
            keep.append(f)

    # Mark skipped files so they aren't reclaimed
    if skip_ids:
        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS fid
                MATCH (sf:CodeFile {id: fid})
                SET sf.status = 'skipped',
                    sf.skip_reason = 'content already ingested',
                    sf.claimed_at = null,
                    sf.claim_token = null
                """,
                ids=skip_ids,
            )

    return keep


def _mark_files_ingested(file_ids: list[str]) -> int:
    """Mark CodeFiles as ingested after successful processing.

    Also marks content-identical duplicates (same content_hash) as
    skipped, since their content is now represented by the ingested copy.
    """
    from imas_codex.graph import GraphClient

    if not file_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:CodeFile {id: fid})
            WHERE sf.status <> 'failed'
            SET sf.status = 'ingested',
                sf.ingested_at = datetime(),
                sf.claimed_at = null,
                sf.claim_token = null
            RETURN count(sf) AS updated
            """,
            ids=file_ids,
        )

        # Mark content-identical duplicates as skipped
        gc.query(
            """
            UNWIND $ids AS fid
            MATCH (sf:CodeFile {id: fid})
            WHERE sf.content_hash IS NOT NULL
            WITH sf
            MATCH (dup:CodeFile {content_hash: sf.content_hash})
            WHERE dup.id <> sf.id AND dup.status IN ['scored', 'triaged']
            SET dup.status = 'skipped',
                dup.skip_reason = 'duplicate of ' + sf.id,
                dup.claimed_at = null,
                dup.claim_token = null
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
                sf.claimed_at = null,
                sf.claim_token = null
            """,
            id=file_id,
            error=error[:200],
        )


async def code_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Code worker: Fetch, chunk, and link code files.

    Claims scored CodeFiles, runs the ingestion pipeline (tree-sitter
    chunking, entity extraction, graph writes).  Embedding is deferred
    to the ``embed_text_worker`` which populates embeddings
    asynchronously on the GPU.
    Transitions: scored → ingested | failed

    Uses small claim batches (default 10) so progress is reported
    frequently — this keeps the streamer display flowing and the
    rate calculation accurate.
    """
    import time as _time

    from imas_codex.ingestion.pipeline import ingest_files

    logger.warning(
        "code_worker started (facility=%s, batch_size=%d, scan_only=%s, score_only=%s)",
        state.facility,
        batch_size,
        state.scan_only,
        state.score_only,
    )

    idle_log_interval = 10  # log every Nth consecutive idle poll
    consecutive_idle = 0
    batches_processed = 0

    while not state.should_stop():
        if state.scan_only or state.score_only:
            logger.warning(
                "code_worker exiting: scan_only=%s, score_only=%s",
                state.scan_only,
                state.score_only,
            )
            break

        # Claim code files for ingestion (wrapped in try/except to survive
        # transient Neo4j errors without crashing the worker)
        try:
            files = await asyncio.to_thread(
                _claim_code_files_for_ingestion,
                state.facility,
                limit=batch_size,
                min_score=state.min_score,
            )
        except Exception as e:
            logger.warning("Code claim failed: %s", e)
            if is_infrastructure_error(e):
                raise
            await asyncio.sleep(2.0)
            continue

        if not files:
            consecutive_idle += 1
            state.code_phase.record_idle()
            if state.code_phase.done:
                logger.warning(
                    "code_worker exiting: phase done after %d batches "
                    "(%d files processed, %d errors)",
                    batches_processed,
                    state.code_stats.processed,
                    state.code_stats.errors,
                )
                break
            if consecutive_idle == 1 or consecutive_idle % idle_log_interval == 0:
                logger.warning(
                    "code_worker idle (poll #%d, phase.idle=%s, "
                    "score_phase.done=%s, processed=%d)",
                    consecutive_idle,
                    state.code_phase.idle,
                    state.score_phase.done,
                    state.code_stats.processed,
                )
            if on_progress:
                on_progress("idle", state.code_stats, None)
            await asyncio.sleep(3.0)
            continue

        # Post-claim dedup: filter out files whose content is already ingested.
        # This is O(batch_size) with index rather than O(candidates × ingested)
        # in the claim query itself.
        try:
            files = await asyncio.to_thread(_filter_duplicates, files)
        except Exception as e:
            logger.warning("Dedup filter failed (proceeding with full batch): %s", e)
            if is_infrastructure_error(e):
                raise

        if not files:
            # Entire batch was duplicates — count as activity but skip processing
            state.code_phase.record_activity(0)
            continue

        consecutive_idle = 0
        state.code_phase.record_activity(len(files))

        if on_progress:
            on_progress(f"ingesting {len(files)} code files", state.code_stats, None)

        remote_paths = [f["path"] for f in files]
        all_ids = [f["id"] for f in files]
        scores = [f.get("score_composite", 0) for f in files]

        logger.warning(
            "code_worker claimed %d files (scores %.2f–%.2f): %s",
            len(files),
            min(scores),
            max(scores),
            ", ".join(f["path"].rsplit("/", 1)[-1] for f in files[:3])
            + ("..." if len(files) > 3 else ""),
        )

        batch_start = _time.monotonic()

        try:
            ingest_stats = await ingest_files(
                facility=state.facility,
                remote_paths=remote_paths,
                force=False,
            )

            batch_elapsed = _time.monotonic() - batch_start

            ingested_count = ingest_stats.get("files", 0)
            skipped_count = ingest_stats.get("skipped", 0)
            chunks_count = ingest_stats.get("chunks", 0)

            # Mark ALL claimed files as ingested — either their content
            # was just processed or was already present (dedup-skipped).
            # _mark_files_ingested skips files already marked 'failed'
            # by ingest_files, so individual failures are preserved.
            if ingested_count > 0 or skipped_count > 0:
                await asyncio.to_thread(_mark_files_ingested, all_ids)

            batch_total = ingested_count + skipped_count
            batches_processed += 1
            state.code_stats.processed += batch_total
            state.code_stats.last_batch_time = batch_elapsed
            state.code_stats.record_batch(batch_total)

            logger.warning(
                "code_worker batch #%d: ingested=%d skipped=%d chunks=%d elapsed=%.1fs",
                batches_processed,
                ingested_count,
                skipped_count,
                chunks_count,
                batch_elapsed,
            )

            if on_progress:
                avg_chunks = chunks_count // max(ingested_count, 1)
                on_progress(
                    f"ingested {ingested_count}, {chunks_count} chunks",
                    state.code_stats,
                    [
                        {
                            "path": f["path"],
                            "language": f.get("language", ""),
                            "score_composite": f.get("score_composite"),
                            "chunks": avg_chunks,
                            "file_type": "code",
                        }
                        for f in files
                    ],
                )

        except Exception as e:
            if is_infrastructure_error(e):
                logger.warning(
                    "Code ingestion batch hit infrastructure failure (%d files): %s",
                    len(files),
                    e,
                )
                raise
            logger.error("Code ingestion batch failed (%d files): %s", len(files), e)
            state.code_stats.errors += 1
            # Mark individual files as failed
            for f in files:
                await asyncio.to_thread(_mark_file_failed, f["id"], str(e)[:200])

        await asyncio.sleep(0.1)

    logger.warning(
        "code_worker stopped (facility=%s, batches=%d, "
        "processed=%d, errors=%d, should_stop=%s)",
        state.facility,
        batches_processed,
        state.code_stats.processed,
        state.code_stats.errors,
        state.should_stop(),
    )


# ============================================================================
# Enrich Worker (rg pattern matching on individual files)
# ============================================================================


async def enrich_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 100,
) -> None:
    """Enrich worker: rg pattern matching + preview extraction on triaged files.

    Claims triaged CodeFiles above the triage composite threshold,
    runs batched rg pattern matching and preview text extraction
    via SSH.  Stores pattern evidence on CodeFile nodes; preview text
    is NOT persisted but is available for the subsequent score worker
    via the graph claim query.

    Runs AFTER triage, BEFORE scoring.
    """
    import time as _time  # noqa: PLC0415

    from imas_codex.discovery.code.enrichment import (
        enrich_files,
        persist_file_enrichment,
    )
    from imas_codex.discovery.code.graph_ops import (
        claim_files_for_enrichment,
        release_file_enrich_claims,
    )

    while not state.should_stop():
        if state.scan_only:
            break

        files = await asyncio.to_thread(
            claim_files_for_enrichment,
            state.facility,
            limit=batch_size,
            min_triage_composite=state.min_triage_score,
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

        batch_start = _time.monotonic()

        try:
            results = await enrich_files(
                state.facility,
                file_paths,
            )

            enrich_counts = await asyncio.to_thread(
                persist_file_enrichment, results, file_id_map
            )
            enriched = enrich_counts["enriched"]
            failed = enrich_counts["failed"]
            processed = enriched + failed

            state.enrich_stats.processed += processed
            state.enrich_stats.last_batch_time = _time.monotonic() - batch_start
            state.enrich_stats.record_batch(processed)

            # Release claims
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)

            if on_progress:
                # Stream enriched files with line count + pattern categories + preview
                enrich_results = []
                for r in results:
                    cats = r.get("pattern_categories", {})
                    # Find top pattern categories by count
                    top_cats = (
                        sorted(cats.items(), key=lambda x: x[1], reverse=True)[:4]
                        if cats
                        else []
                    )
                    # Extract a meaningful preview snippet (first non-blank, non-comment line)
                    preview = r.get("preview_text", "")
                    snippet = ""
                    if preview:
                        for line in preview.splitlines():
                            stripped = line.strip()
                            if stripped and not stripped.startswith(
                                ("#", "//", "/*", "*", "!", "C ", "c ")
                            ):
                                snippet = stripped[:80]
                                break
                    # Look up triage_composite from the claim data
                    file_id = file_id_map.get(r["path"])
                    triage_score = None
                    for f in files:
                        if f["id"] == file_id:
                            triage_score = f.get("triage_composite")
                            break
                    enrich_results.append(
                        {
                            "path": r["path"],
                            "triage_composite": triage_score,
                            "patterns": r.get("total_pattern_matches", 0),
                            "line_count": r.get("line_count", 0),
                            "pattern_categories": dict(top_cats),
                            "preview_snippet": snippet,
                        }
                    )
                on_progress(
                    f"enriched {enriched}, failed {failed}",
                    state.enrich_stats,
                    enrich_results,
                )

        except Exception as e:
            logger.error("File enrichment batch failed: %s", e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)
            if is_infrastructure_error(e):
                raise

        await asyncio.sleep(0.1)


# ============================================================================
# Link Worker (code evidence → signal propagation)
# ============================================================================


async def link_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Link worker: Propagate code evidence to FacilitySignals.

    After code ingestion creates DataReference → SignalNode links, this
    worker propagates evidence to FacilitySignals via the chain:
      DataReference → RESOLVES_TO_NODE → SignalNode ← HAS_DATA_SOURCE_NODE ← FacilitySignal

    Sets code_evidence_count and has_code_evidence on matched signals.
    Runs periodically while code workers are active, then one final pass.
    """
    from imas_codex.discovery.code.graph_ops import (
        has_pending_link_work,
        link_code_evidence_to_signals,
    )

    last_linked = 0

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        has_work = await asyncio.to_thread(has_pending_link_work, state.facility)

        if not has_work:
            # If code phase is done, we're done too
            if state.code_phase.done:
                break
            await asyncio.sleep(5.0)
            continue

        if on_progress:
            on_progress("linking code evidence to signals", state.link_stats, None)

        try:
            result = await asyncio.to_thread(
                link_code_evidence_to_signals, state.facility
            )

            signals_linked = result.get("signals_linked", 0)
            refs_resolved = result.get("refs_resolved", 0)
            state.link_stats.processed += signals_linked
            last_linked = signals_linked

            if on_progress:
                on_progress(
                    f"linked {signals_linked} signals ({refs_resolved} refs resolved)",
                    state.link_stats,
                    None,
                )

        except Exception as e:
            logger.error("Code evidence linking failed: %s", e)
            state.link_stats.errors += 1
            if is_infrastructure_error(e):
                raise

        # Link is cheap, run every 10s
        await asyncio.sleep(10.0)

    # Final pass after all code ingestion is done.
    # Skip it when the graph is already drained; otherwise the worker can
    # spend minutes in a no-op relinking query after the UI has gone idle.
    if not (state.scan_only or state.score_only):
        try:
            final_has_work = await asyncio.to_thread(
                has_pending_link_work, state.facility
            )
            if final_has_work:
                result = await asyncio.to_thread(
                    link_code_evidence_to_signals, state.facility
                )
                final_linked = result.get("signals_linked", 0)
                if final_linked > last_linked and on_progress:
                    on_progress(
                        f"final link: {final_linked} signals",
                        state.link_stats,
                        None,
                    )
        except Exception as e:
            logger.error("Final code evidence linking failed: %s", e)
            if is_infrastructure_error(e):
                raise
