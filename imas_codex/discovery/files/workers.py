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
    groups by parent FacilityPath, scores via LLM with enrichment context.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.discovery.files.graph_ops import (
        claim_files_for_scoring,
        release_file_score_claims,
    )
    from imas_codex.discovery.files.scorer import (
        FileScoreBatch,
        _build_system_prompt,
        _build_user_prompt,
        _group_files_by_parent,
        apply_file_scores,
    )
    from imas_codex.settings import get_model

    model = get_model("language")

    # Build system prompt once for prefix caching
    system_prompt = _build_system_prompt(focus=state.focus)

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
                f"scoring {len(files)} files ({len(file_groups)} dirs)",
                state.score_stats,
                None,
            )

        user_prompt = _build_user_prompt(file_groups)

        try:
            parsed, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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
                    }
                    for s in parsed.results[:5]
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


# ============================================================================
# Enrich Worker (rg pattern matching on individual files)
# ============================================================================


async def enrich_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 100,
) -> None:
    """Enrich worker: rg pattern matching on scored SourceFiles.

    Claims scored SourceFiles that haven't been enriched yet, runs
    batched rg pattern matching via SSH, stores per-file pattern
    evidence on SourceFile nodes.

    Runs AFTER scoring, in parallel with code/docs ingestion workers.
    Enrichment data improves ingestion prioritization — files with
    actual pattern evidence are ingested first.
    """
    from imas_codex.discovery.files.enrichment import (
        enrich_files,
        persist_file_enrichment,
    )
    from imas_codex.discovery.files.graph_ops import (
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
                # Summarize pattern findings
                files_with_patterns = sum(
                    1 for r in results if r.get("total_pattern_matches", 0) > 0
                )
                on_progress(
                    f"enriched {enriched} ({files_with_patterns} with patterns)",
                    state.enrich_stats,
                    [
                        {
                            "path": r["path"],
                            "patterns": r.get("total_pattern_matches", 0),
                        }
                        for r in results[:5]
                        if r.get("total_pattern_matches", 0) > 0
                    ],
                )

        except Exception as e:
            logger.error("File enrichment batch failed: %s", e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_file_enrich_claims, batch_ids)

        await asyncio.sleep(0.1)


# ============================================================================
# Image Worker (standalone image files → Image nodes)
# ============================================================================


async def image_worker(
    state: FileDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Image worker: Fetch, downsample, and persist standalone image files.

    Claims scored SourceFiles with file_category='image', fetches image
    bytes via SCP, downsamples to WebP, and creates Image nodes linked
    to the SourceFile. Image nodes are then available for VLM captioning.

    Transitions: discovered (scored) → ingested | failed
    """
    import subprocess
    import tempfile
    from pathlib import Path

    from imas_codex.discovery.base.image import (
        downsample_image,
        make_image_id,
        persist_images,
    )

    while not state.should_stop():
        if state.scan_only or state.score_only:
            break

        # Claim image files for processing
        files = await asyncio.to_thread(
            _claim_files_for_ingestion,
            state.facility,
            file_category="image",
            limit=batch_size,
        )

        if not files:
            state.image_phase.record_idle()
            if on_progress:
                on_progress("idle", state.image_stats, None)
            await asyncio.sleep(3.0)
            continue

        state.image_phase.record_activity(len(files))

        if on_progress:
            on_progress(f"processing {len(files)} images", state.image_stats, None)

        images_to_persist: list[dict[str, Any]] = []
        ingested_ids: list[str] = []
        failed_ids: list[tuple[str, str]] = []

        # Batch-fetch images via tar (reuses SSH connection)
        remote_paths = [f["path"] for f in files]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Fetch all image files in a single SSH tar
            try:
                tar_cmd = (
                    "tar cf - "
                    + " ".join(f"'{p}'" for p in remote_paths)
                    + " 2>/dev/null"
                )
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["ssh", state.ssh_host, tar_cmd],
                    capture_output=True,
                    timeout=60,
                )

                if result.returncode == 0 and result.stdout:
                    import io
                    import tarfile

                    tar = tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:")
                    tar.extractall(path=tmpdir, filter="data")
                    tar.close()

                    for f in files:
                        remote_path = f["path"]
                        sf_id = f["id"]
                        # tar preserves full path
                        local_path = Path(tmpdir) / remote_path.lstrip("/")

                        if not local_path.exists():
                            failed_ids.append((sf_id, "File not found in tar archive"))
                            continue

                        try:
                            image_bytes = local_path.read_bytes()
                        except Exception as e:
                            failed_ids.append((sf_id, str(e)[:200]))
                            continue

                        if len(image_bytes) < 512:
                            failed_ids.append((sf_id, "Image too small (<512 bytes)"))
                            continue

                        ds_result = downsample_image(image_bytes)
                        if ds_result is None:
                            failed_ids.append(
                                (sf_id, "Downsample failed (too small or unreadable)")
                            )
                            continue

                        b64_data, stored_w, stored_h, orig_w, orig_h = ds_result
                        image_id = make_image_id(state.facility, remote_path)
                        filename = Path(remote_path).name

                        images_to_persist.append(
                            {
                                "id": image_id,
                                "facility_id": state.facility,
                                "source_url": remote_path,
                                "source_type": "filesystem",
                                "status": "ingested",
                                "filename": filename,
                                "image_format": "webp",
                                "width": stored_w,
                                "height": stored_h,
                                "original_width": orig_w,
                                "original_height": orig_h,
                                "content_hash": None,
                                "image_data": b64_data,
                                "page_title": None,
                                "section": None,
                                "alt_text": None,
                                "surrounding_text": None,
                                "source_file_id": sf_id,
                            }
                        )
                        ingested_ids.append(sf_id)
                else:
                    # tar failed — mark all as failed
                    for f in files:
                        failed_ids.append((f["id"], "SSH tar fetch failed"))

            except Exception as e:
                logger.error("Image batch fetch failed: %s", e)
                state.image_stats.errors += 1
                for f in files:
                    await asyncio.to_thread(_mark_file_failed, f["id"], str(e)[:200])
                continue

        # Persist Image nodes
        if images_to_persist:
            await asyncio.to_thread(
                persist_images,
                images_to_persist,
                parent_label="SourceFile",
                parent_id_key="source_file_id",
            )

        # Mark SourceFiles as ingested
        if ingested_ids:
            await asyncio.to_thread(_mark_files_ingested, ingested_ids)

        # Mark failures
        for sf_id, error in failed_ids:
            await asyncio.to_thread(_mark_file_failed, sf_id, error)

        state.image_stats.processed += len(ingested_ids)

        if on_progress:
            on_progress(
                f"processed {len(ingested_ids)} images ({len(failed_ids)} failed)",
                state.image_stats,
                [
                    {
                        "path": img["source_url"],
                        "size": f"{img['width']}x{img['height']}",
                    }
                    for img in images_to_persist[:5]
                ],
            )

        await asyncio.sleep(0.5)
