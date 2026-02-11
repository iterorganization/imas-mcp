"""Async workers for parallel wiki discovery.

Five supervised workers that process wiki content through the pipeline:
- score_worker: LLM content-aware scoring (scanned → scored)
- ingest_worker: Chunk and embed high-value pages (scored → ingested)
- artifact_worker: Download and ingest scored artifacts (scored → ingested)
- artifact_score_worker: LLM scoring for artifacts (discovered → scored)
- image_score_worker: VLM captioning and scoring (ingested → captioned)

Workers are supervised via base.supervision for automatic restart on crash.
They coordinate through graph_ops claim/mark functions using claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from imas_codex.graph import GraphClient
from imas_codex.graph.models import WikiPageStatus

from .graph_ops import (
    IMAGE_ARTIFACT_TYPES,
    INGESTABLE_ARTIFACT_TYPES,
    _release_claimed_images,
    _release_claimed_pages,
    claim_artifacts_for_ingesting,
    claim_artifacts_for_scoring,
    claim_images_for_scoring,
    claim_pages_for_ingesting,
    claim_pages_for_scoring,
    mark_artifact_deferred,
    mark_artifact_failed,
    mark_artifacts_ingested,
    mark_artifacts_scored,
    mark_images_scored,
    mark_page_failed,
    mark_pages_ingested,
    mark_pages_scored,
)
from .scoring import (
    _extract_artifact_preview,
    _fetch_and_summarize,
    _fetch_html,
    _score_artifacts_batch,
    _score_images_batch,
    _score_pages_batch,
)
from .state import WikiDiscoveryState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


async def score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Score worker: Content-aware LLM scoring in single pass.

    Transitions: scanned → scored

    Fetches page content preview, then scores with LLM.
    Uses centralized LLM access via get_model_for_task().
    Cost is tracked from actual OpenRouter response.
    """
    from imas_codex.agentic.agents import get_model_for_task

    worker_id = id(asyncio.current_task())
    logger.info(f"score_worker started (task={worker_id})")

    # Use effective semaphore: SSH semaphore (4) for SSH-based sites,
    # HTTP semaphore (30) for direct HTTP. Prevents overwhelming remote
    # hosts with concurrent SSH subprocess calls.
    fetch_semaphore = state.effective_fetch_semaphore

    # Get shared async wiki client for Tequila auth (native async HTTP)
    logger.debug(f"score_worker {worker_id}: getting async wiki client")
    shared_async_wiki_client = (
        await state.get_async_wiki_client()
        if state.auth_type in ("tequila", "session")
        else None
    )
    # Get shared Keycloak client for keycloak auth (OIDC via oauth2-proxy)
    shared_keycloak_client = (
        await state.get_keycloak_client() if state.auth_type == "keycloak" else None
    )
    # Get shared HTTP Basic auth client (e.g. JET wikis)
    shared_basic_auth_client = (
        await state.get_basic_auth_client() if state.auth_type == "basic" else None
    )
    logger.debug(f"score_worker {worker_id}: got async wiki client")

    async def fetch_content_for_page(page: dict) -> dict:
        """Fetch content preview for a single page."""
        async with fetch_semaphore:
            url = page.get("url", "")
            try:
                preview = await _fetch_and_summarize(
                    url,
                    state.ssh_host,
                    auth_type=state.auth_type,
                    credential_service=state.credential_service,
                    max_chars=1500,  # Reduced for scoring
                    async_wiki_client=shared_async_wiki_client,  # Native async HTTP
                    keycloak_client=shared_keycloak_client,
                    basic_auth_client=shared_basic_auth_client,
                )
                return {
                    "id": page["id"],
                    "title": page.get("title", ""),
                    "url": url,
                    "preview_text": preview,
                    "fetch_error": None,
                }
            except Exception as e:
                logger.debug("Failed to fetch %s: %s", url, e)
                return {
                    "id": page["id"],
                    "title": page.get("title", ""),
                    "url": url,
                    "preview_text": "",
                    "fetch_error": str(e),
                }

    while not state.should_stop_scoring():
        # Increased batch size from 25 to 50 for better LLM throughput
        # Run blocking Neo4j call in thread pool to avoid blocking event loop
        logger.debug(f"score_worker {worker_id}: claiming pages...")
        try:
            pages = await asyncio.to_thread(claim_pages_for_scoring, state.facility, 50)
        except Exception as e:
            # Neo4j connection error - backoff and retry
            logger.warning("score_worker %s: claim failed: %s", worker_id, e)
            await asyncio.sleep(5.0)
            continue
        logger.debug(f"score_worker {worker_id}: claimed {len(pages)} pages")

        if not pages:
            state.score_idle_count += 1
            if on_progress:
                on_progress("idle", state.score_stats)
            await asyncio.sleep(1.0)
            continue

        state.score_idle_count = 0

        if on_progress:
            on_progress(f"fetching {len(pages)} pages", state.score_stats)

        # Step 1: Fetch content for all pages in parallel
        fetch_tasks = [fetch_content_for_page(page) for page in pages]
        fetched_pages = await asyncio.gather(*fetch_tasks)
        logger.debug(f"score_worker {worker_id}: fetched {len(fetched_pages)} pages")

        # Step 1b: Separate pages with content from those where fetch failed.
        # Pages without preview_text cannot be scored meaningfully — release
        # them back so they can be retried when the server is responsive.
        pages_with_content = [p for p in fetched_pages if p.get("preview_text")]
        pages_no_content = [p for p in fetched_pages if not p.get("preview_text")]

        if pages_no_content:
            logger.info(
                "score_worker %s: %d/%d pages had no content (fetch failed), "
                "releasing for retry",
                worker_id,
                len(pages_no_content),
                len(fetched_pages),
            )
            await asyncio.to_thread(
                _release_claimed_pages, [p["id"] for p in pages_no_content]
            )

        if not pages_with_content:
            logger.debug(
                f"score_worker {worker_id}: no pages with content, skipping LLM"
            )
            continue

        if on_progress:
            on_progress(f"scoring {len(pages_with_content)} pages", state.score_stats)

        try:
            # Step 2: Score batch with LLM (only pages that have content)
            model = get_model_for_task("discovery")
            logger.debug(f"score_worker {worker_id}: starting LLM scoring...")
            results, cost = await _score_pages_batch(
                pages_with_content, model, state.focus
            )
            logger.debug(
                f"score_worker {worker_id}: LLM scored {len(results)} pages, cost=${cost:.4f}"
            )

            # Add preview_text to results for persistence
            for r in results:
                matching_page = next(
                    (p for p in pages_with_content if p["id"] == r["id"]), {}
                )
                r["preview_text"] = matching_page.get("preview_text", "")
                r["score_cost"] = cost / len(results) if results else 0.0

            # Run blocking Neo4j call in thread pool to avoid blocking event loop
            await asyncio.to_thread(mark_pages_scored, state.facility, results)
            state.score_stats.processed += len(results)
            state.score_stats.cost += cost

            if on_progress:
                on_progress(
                    f"scored {len(results)} pages", state.score_stats, results=results
                )

        except ValueError as e:
            # LLM validation error (e.g., truncated JSON) - release pages and continue
            # Don't stop the whole process; pages will be reclaimed after timeout
            logger.warning(
                "LLM failed for batch of %d pages: %s. "
                "Pages reverted to scanned status for retry.",
                len(pages),
                e,
            )
            state.score_stats.errors = getattr(state.score_stats, "errors", 0) + 1
            # Release pages by clearing claimed_at (not marking as failed)
            # Run blocking Neo4j call in thread pool
            await asyncio.to_thread(_release_claimed_pages, [p["id"] for p in pages])
            # Continue processing - don't stop the whole discovery
            continue
        except Exception as e:
            logger.error("Error in scoring batch: %s", e)
            # Run blocking Neo4j calls in thread pool
            for page in pages:
                await asyncio.to_thread(
                    mark_page_failed,
                    page["id"],
                    str(e),
                    WikiPageStatus.scanned.value,
                )


async def ingest_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
    min_score: float = 0.5,
) -> None:
    """Ingest worker: Chunk and embed high-value scored pages.

    Transitions: scored → ingested

    Claims scored pages with score >= min_score, fetches full content,
    chunks it, and creates embeddings.

    The ingest worker continues running even after cost limit is reached
    to drain the ingest queue. This ensures all scored content gets ingested.

    PERF: Pages are processed in parallel using asyncio.gather() with a
    semaphore to limit concurrency. This provides ~5x speedup over sequential.
    """
    # Get shared async wiki client for Tequila auth (native async HTTP)
    shared_async_wiki_client = (
        await state.get_async_wiki_client()
        if state.auth_type in ("tequila", "session")
        else None
    )
    # Get shared Keycloak client for keycloak auth (OIDC via oauth2-proxy)
    shared_keycloak_client = (
        await state.get_keycloak_client() if state.auth_type == "keycloak" else None
    )
    # Get shared HTTP Basic auth client (e.g. JET wikis)
    shared_basic_auth_client = (
        await state.get_basic_auth_client() if state.auth_type == "basic" else None
    )

    # Use effective semaphore: SSH semaphore (4) for SSH-based sites,
    # HTTP semaphore (30) for direct HTTP. Prevents overwhelming remote
    # hosts with concurrent SSH subprocess calls.
    http_semaphore = state.effective_fetch_semaphore

    async def process_single_page(page: dict) -> dict | None:
        """Process a single page with semaphore-limited concurrency."""
        page_id = page["id"]
        url = page.get("url", "")

        async with http_semaphore:
            try:
                chunk_count = await _ingest_page(
                    url=url,
                    page_id=page_id,
                    facility=state.facility,
                    site_type=state.site_type,
                    ssh_host=state.ssh_host,
                    auth_type=state.auth_type,
                    credential_service=state.credential_service,
                    async_wiki_client=shared_async_wiki_client,
                    keycloak_client=shared_keycloak_client,
                    basic_auth_client=shared_basic_auth_client,
                )
                return {
                    "id": page_id,
                    "chunk_count": chunk_count,
                    "score": page.get("score"),
                    "description": page.get("description", ""),
                    "physics_domain": page.get("physics_domain"),
                }
            except Exception as e:
                logger.warning("Error ingesting %s: %s", page_id, e)
                # Run blocking Neo4j call in thread pool
                await asyncio.to_thread(
                    mark_page_failed, page_id, str(e), WikiPageStatus.scored.value
                )
                return None

    while not state.should_stop_ingesting():
        # Increased batch size from 10 to 20 for better embedding throughput
        # Run blocking Neo4j call in thread pool to avoid blocking event loop
        try:
            pages = await asyncio.to_thread(
                claim_pages_for_ingesting, state.facility, min_score, 20
            )
        except Exception as e:
            # Neo4j connection error - backoff and retry
            logger.warning("ingest_worker: claim failed: %s", e)
            await asyncio.sleep(5.0)
            continue

        if not pages:
            state.ingest_idle_count += 1
            if on_progress:
                on_progress("idle", state.ingest_stats)
            await asyncio.sleep(1.0)
            continue

        state.ingest_idle_count = 0

        if on_progress:
            on_progress(f"ingesting {len(pages)} pages", state.ingest_stats)

        # Process all pages in parallel with semaphore-limited concurrency
        tasks = [process_single_page(page) for page in pages]
        results_raw = await asyncio.gather(*tasks)

        # Filter out None results (failed pages)
        results = [r for r in results_raw if r is not None]

        # Run blocking Neo4j call in thread pool
        await asyncio.to_thread(mark_pages_ingested, state.facility, results)
        state.ingest_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"ingested {len(results)} pages", state.ingest_stats, results=results
            )


async def artifact_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
    max_size_mb: float = 5.0,
) -> None:
    """Artifact worker: Download and ingest wiki artifacts.

    Transitions: scored → ingested

    Routes artifacts by type:
    - Text-extractable (pdf, document, presentation, spreadsheet, notebook, json):
      Download, extract text, chunk, embed. Also extract embedded images from PDF/PPTX.
    - Image artifacts: Download, downsample, create Image nodes for VLM pipeline.
    - Oversized files: Deferred with DEBUG-level logging (node preserved with metadata).

    Args:
        state: Shared discovery state
        on_progress: Progress callback (msg, stats, results=None)
        max_size_mb: Maximum artifact size in MB
    """
    from imas_codex.discovery.wiki.pipeline import (
        WikiArtifactPipeline,
        fetch_artifact_content,
        fetch_artifact_size,
    )

    max_size_bytes = int(max_size_mb * 1024 * 1024)
    pipeline = WikiArtifactPipeline(
        facility_id=state.facility,
        max_size_mb=max_size_mb,
        use_rich=False,
    )

    while not state.should_stop_artifact_worker():
        # Run blocking Neo4j call in thread pool to avoid blocking event loop
        try:
            artifacts = await asyncio.to_thread(
                claim_artifacts_for_ingesting, state.facility, limit=8
            )
        except Exception as e:
            logger.warning("artifact_worker: claim failed: %s", e)
            await asyncio.sleep(5.0)
            continue

        if not artifacts:
            state.artifact_idle_count += 1
            if on_progress:
                on_progress("idle", state.artifact_stats)
            await asyncio.sleep(1.0)
            continue

        state.artifact_idle_count = 0

        if on_progress:
            on_progress(f"ingesting {len(artifacts)} artifacts", state.artifact_stats)

        results = []
        for artifact in artifacts:
            artifact_id = artifact["id"]
            artifact_type = artifact.get("artifact_type", "unknown")
            url = artifact.get("url", "")
            filename = artifact.get("filename", "unknown")

            try:
                # Route image artifacts to Image node pipeline
                if artifact_type.lower() in IMAGE_ARTIFACT_TYPES:
                    await _ingest_image_artifact(
                        artifact_id=artifact_id,
                        url=url,
                        filename=filename,
                        facility=state.facility,
                        ssh_host=getattr(state, "ssh_host", None),
                    )
                    results.append(
                        {
                            "id": artifact_id,
                            "chunk_count": 0,
                            "filename": filename,
                            "artifact_type": artifact_type,
                            "score": artifact.get("score"),
                            "physics_domain": artifact.get("physics_domain"),
                            "description": artifact.get("description", ""),
                        }
                    )
                    continue

                # Check size before downloading text-extractable artifacts
                size_bytes = fetch_artifact_size(url, facility=state.facility)

                if size_bytes is not None and size_bytes > max_size_bytes:
                    size_mb = size_bytes / (1024 * 1024)
                    reason = (
                        f"File size {size_mb:.1f} MB exceeds limit {max_size_mb:.1f} MB"
                    )
                    # Demote to DEBUG — node is preserved with metadata, just not ingested
                    logger.debug(
                        "Deferring oversized artifact %s: %s", filename, reason
                    )
                    await asyncio.to_thread(mark_artifact_deferred, artifact_id, reason)
                    continue

                # Check if type is text-extractable
                if artifact_type.lower() not in INGESTABLE_ARTIFACT_TYPES:
                    reason = f"Artifact type '{artifact_type}' not text-extractable"
                    logger.debug(
                        "Deferring non-ingestable artifact %s: %s", filename, reason
                    )
                    await asyncio.to_thread(mark_artifact_deferred, artifact_id, reason)
                    continue

                # Download and ingest
                _, content = await fetch_artifact_content(url, facility=state.facility)
                stats = await pipeline.ingest_artifact(
                    artifact_id, content, artifact_type
                )

                # If pipeline extracted images (PDF/PPTX), create Image nodes
                extracted_images = stats.get("extracted_images", [])
                if extracted_images:
                    await _persist_document_figures(
                        extracted_images,
                        artifact_id=artifact_id,
                        facility=state.facility,
                    )

                results.append(
                    {
                        "id": artifact_id,
                        "chunk_count": stats["chunks"],
                        "filename": filename,
                        "artifact_type": artifact_type,
                        "score": artifact.get("score"),
                        "physics_domain": artifact.get("physics_domain"),
                        "description": artifact.get("description", ""),
                    }
                )

            except Exception as e:
                # Unwrap common error wrappers to get the actual error message
                error_msg = str(e)
                if hasattr(e, "__cause__") and e.__cause__:
                    error_msg = str(e.__cause__)
                elif "RetryError" in type(e).__name__ and hasattr(e, "last_attempt"):
                    # tenacity RetryError - get the underlying exception
                    try:
                        error_msg = str(e.last_attempt.exception())
                    except Exception:
                        pass
                logger.warning(
                    "Error ingesting artifact %s: %s", artifact_id, error_msg
                )
                # Run blocking Neo4j call in thread pool
                await asyncio.to_thread(mark_artifact_failed, artifact_id, error_msg)

        # Run blocking Neo4j call in thread pool
        await asyncio.to_thread(mark_artifacts_ingested, state.facility, results)
        state.artifact_stats.processed += len(results)

        if on_progress:
            on_progress(
                f"ingested {len(results)} artifacts",
                state.artifact_stats,
                results=results,
            )


async def artifact_score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Artifact score worker: LLM scoring with text preview extraction.

    Transitions: discovered → scored

    Claims discovered artifacts with supported types, downloads a small
    portion of content to extract a text preview, then scores with LLM.
    After scoring, artifacts with score >= 0.5 become eligible for full ingestion.

    Uses the same scoring dimensions as wiki page scoring for consistency.
    """
    from imas_codex.agentic.agents import get_model_for_task

    worker_id = id(asyncio.current_task())
    logger.info(f"artifact_score_worker started (task={worker_id})")

    while not state.should_stop_artifact_scoring():
        # Claim artifacts for scoring
        try:
            artifacts = await asyncio.to_thread(
                claim_artifacts_for_scoring, state.facility, 10
            )
        except Exception as e:
            logger.warning("artifact_score_worker %s: claim failed: %s", worker_id, e)
            await asyncio.sleep(5.0)
            continue

        if not artifacts:
            state.artifact_score_idle_count += 1
            if on_progress:
                on_progress("idle", state.artifact_score_stats)
            await asyncio.sleep(1.0)
            continue

        state.artifact_score_idle_count = 0

        if on_progress:
            on_progress(
                f"extracting text from {len(artifacts)} artifacts",
                state.artifact_score_stats,
            )

        # Step 1: Extract text preview from each artifact
        artifacts_with_text = []
        for artifact in artifacts:
            artifact_id = artifact["id"]
            url = artifact.get("url", "")
            filename = artifact.get("filename", "")
            artifact_type = artifact.get("artifact_type", "unknown")

            try:
                preview_text = await _extract_artifact_preview(
                    url=url,
                    artifact_type=artifact_type,
                    facility=state.facility,
                    max_chars=1500,
                )
                artifacts_with_text.append(
                    {
                        "id": artifact_id,
                        "filename": filename,
                        "url": url,
                        "artifact_type": artifact_type,
                        "size_bytes": artifact.get("size_bytes"),
                        "preview_text": preview_text,
                    }
                )
            except Exception as e:
                logger.debug("Failed to extract preview for %s: %s", filename, e)
                # Track failed extractions for release
                artifacts_with_text.append(
                    {
                        "id": artifact_id,
                        "filename": filename,
                        "url": url,
                        "artifact_type": artifact_type,
                        "size_bytes": artifact.get("size_bytes"),
                        "preview_text": "",
                    }
                )

        # Separate artifacts with content from those where extraction failed.
        # For types that can't have text preview (data, archive, image), use
        # filename-based metadata as the preview rather than releasing for retry.
        artifacts_to_score = []
        for a in artifacts_with_text:
            if a.get("preview_text"):
                artifacts_to_score.append(a)
            else:
                # Generate metadata-only preview from filename and type
                at = a.get("artifact_type", "unknown")
                fn = a.get("filename", "unknown")
                size = a.get("size_bytes")
                size_str = f" ({size / (1024 * 1024):.1f} MB)" if size else ""
                a["preview_text"] = (
                    f"[No text content available — metadata only]\n"
                    f"Filename: {fn}\n"
                    f"Type: {at}{size_str}\n"
                    f"This is a {at} file attached to a facility wiki page."
                )
                artifacts_to_score.append(a)

        if not artifacts_to_score:
            logger.debug(
                "artifact_score_worker %s: no artifacts with content, skipping LLM",
                worker_id,
            )
            continue

        if on_progress:
            on_progress(
                f"scoring {len(artifacts_to_score)} artifacts",
                state.artifact_score_stats,
            )

        try:
            # Step 2: Score batch with LLM (only artifacts that have content)
            model = get_model_for_task("discovery")
            results, cost = await _score_artifacts_batch(
                artifacts_to_score, model, state.focus
            )

            # Add preview_text to results for persistence
            for r in results:
                matching = next(
                    (a for a in artifacts_to_score if a["id"] == r["id"]), {}
                )
                r["preview_text"] = matching.get("preview_text", "")[:500]
                r["score_cost"] = cost / len(results) if results else 0.0

            # Persist scores to graph
            await asyncio.to_thread(mark_artifacts_scored, state.facility, results)
            state.artifact_score_stats.processed += len(results)
            state.artifact_score_stats.cost += cost

            if on_progress:
                on_progress(
                    f"scored {len(results)} artifacts",
                    state.artifact_score_stats,
                    results=results,
                )

        except ValueError as e:
            logger.warning(
                "LLM failed for artifact batch of %d: %s. "
                "Artifacts reverted to discovered status for retry.",
                len(artifacts),
                e,
            )
            state.artifact_score_stats.errors = (
                getattr(state.artifact_score_stats, "errors", 0) + 1
            )
            # Release artifacts by clearing claimed_at
            try:
                with GraphClient() as gc:
                    gc.query(
                        """
                        UNWIND $ids AS id
                        MATCH (wa:WikiArtifact {id: id})
                        SET wa.claimed_at = null
                        """,
                        ids=[a["id"] for a in artifacts],
                    )
            except Exception:
                pass
            continue
        except Exception as e:
            logger.error("Error in artifact scoring batch: %s", e)
            for artifact in artifacts:
                await asyncio.to_thread(mark_artifact_failed, artifact["id"], str(e))


async def image_score_worker(
    state: WikiDiscoveryState,
    on_progress: Callable | None = None,
) -> None:
    """Image score worker: VLM captioning + scoring in single pass.

    Transitions: ingested → captioned

    Claims images that have been ingested (image_data stored) but not yet
    captioned/scored. Sends image bytes + context to VLM, receives
    caption + scoring in one pass.
    """
    from imas_codex.agentic.agents import get_model_for_task

    worker_id = id(asyncio.current_task())
    logger.info(f"image_score_worker started (task={worker_id})")

    while not state.should_stop_image_scoring():
        try:
            images = await asyncio.to_thread(
                claim_images_for_scoring, state.facility, 10
            )
        except Exception as e:
            logger.warning("image_score_worker %s: claim failed: %s", worker_id, e)
            await asyncio.sleep(5.0)
            continue

        if not images:
            state.image_idle_count += 1
            if on_progress:
                on_progress("idle", state.image_stats)
            await asyncio.sleep(2.0)
            continue

        state.image_idle_count = 0

        # Filter out images without stored image_data
        images_with_data = [img for img in images if img.get("image_data")]
        images_no_data = [img for img in images if not img.get("image_data")]

        if images_no_data:
            logger.info(
                "image_score_worker: %d/%d images have no image_data, releasing",
                len(images_no_data),
                len(images),
            )
            await asyncio.to_thread(
                _release_claimed_images, [img["id"] for img in images_no_data]
            )

        if not images_with_data:
            continue

        if on_progress:
            on_progress(f"scoring {len(images_with_data)} images", state.image_stats)

        try:
            model = get_model_for_task("vlm")
            results, cost = await _score_images_batch(
                images_with_data, model, state.focus
            )

            # Persist to graph
            await asyncio.to_thread(mark_images_scored, state.facility, results)
            state.image_stats.processed += len(results)
            state.image_stats.cost += cost

            if on_progress:
                on_progress(
                    f"scored {len(results)} images",
                    state.image_stats,
                    results=results,
                )

        except ValueError as e:
            logger.warning(
                "VLM failed for batch of %d images: %s. Images released for retry.",
                len(images_with_data),
                e,
            )
            state.image_stats.errors = getattr(state.image_stats, "errors", 0) + 1
            await asyncio.to_thread(
                _release_claimed_images,
                [img["id"] for img in images_with_data],
            )


async def _ingest_page(
    url: str,
    page_id: str,
    facility: str,
    site_type: str,
    ssh_host: str | None,
    auth_type: str | None = None,
    credential_service: str | None = None,
    async_wiki_client: Any = None,
    keycloak_client: Any = None,
    basic_auth_client: Any = None,
) -> int:
    """Ingest a page: fetch content, chunk, and embed.

    Uses the WikiIngestionPipeline for proper chunking and embedding.

    Args:
        url: Page URL to fetch
        page_id: Unique page identifier
        facility: Facility ID (e.g., 'tcv', 'jet')
        site_type: Site type ('mediawiki', 'confluence', 'twiki')
        ssh_host: Optional SSH host for proxied fetching
        auth_type: Authentication type (tequila, session, keycloak, basic, etc.)
        credential_service: Keyring service for credentials
        async_wiki_client: Optional shared AsyncMediaWikiClient for Tequila auth
        keycloak_client: Optional shared httpx.AsyncClient for Keycloak auth
        basic_auth_client: Optional shared httpx.AsyncClient with HTTP Basic auth

    Returns:
        Number of chunks created
    """
    from imas_codex.discovery.wiki.pipeline import WikiIngestionPipeline
    from imas_codex.discovery.wiki.scraper import WikiPage

    # Extract page name from URL or page_id
    page_name = page_id.split(":", 1)[1] if ":" in page_id else page_id

    # Handle TWiki raw content (ssh:// URLs → read file via SSH, convert markup)
    if url and url.startswith("ssh://"):
        import asyncio

        from imas_codex.discovery.wiki.adapters import fetch_twiki_raw_content
        from imas_codex.discovery.wiki.pipeline import twiki_markup_to_html

        # Parse ssh://host/path
        parts = url[len("ssh://") :]
        slash_idx = parts.index("/")
        raw_ssh_host = parts[:slash_idx]
        filepath = parts[slash_idx:]

        # Read raw TWiki markup via SSH (blocking I/O → thread pool)
        raw_markup = await asyncio.to_thread(
            fetch_twiki_raw_content, raw_ssh_host, filepath
        )
        if not raw_markup or len(raw_markup) < 50:
            logger.warning("Insufficient TWiki content for %s", page_id)
            return 0

        # Convert TWiki markup to minimal HTML
        html = twiki_markup_to_html(raw_markup)
    else:
        # Fetch HTML content with auth (standard HTTP path)
        html = await _fetch_html(
            url,
            ssh_host,
            auth_type=auth_type,
            credential_service=credential_service,
            async_wiki_client=async_wiki_client,
            keycloak_client=keycloak_client,
            basic_auth_client=basic_auth_client,
        )
    if not html or len(html) < 100:
        logger.warning("Insufficient content for %s", page_id)
        return 0

    # Extract title from HTML
    import re

    title_match = re.search(r"<title>([^<]+)</title>", html)
    title = title_match.group(1) if title_match else page_name

    # Clean up title (remove wiki suffix)
    for suffix in [" - SPCwiki", " - Wikipedia", " - Confluence"]:
        if title.endswith(suffix):
            title = title[: -len(suffix)]

    # Create WikiPage object (fields from dataclass in scraper.py)
    page = WikiPage(
        url=url,
        title=title,
        content_html=html,
        content_text="",  # Will be extracted by pipeline
        sections={},
        mdsplus_paths=[],  # Will be extracted by pipeline
        imas_paths=[],
        units=[],
        conventions=[],
    )

    # Use the ingestion pipeline
    pipeline = WikiIngestionPipeline(
        facility_id=facility,
        use_rich=False,  # No progress display in worker
    )

    try:
        stats = await pipeline.ingest_page(page)
        chunks = stats.get("chunks", 0)

        # Extract and persist images from HTML for VLM scoring
        if html and chunks > 0:
            try:
                image_count = await _extract_and_persist_images(
                    html=html,
                    page_url=url,
                    page_id=page_id,
                    page_title=title,
                    facility=facility,
                    ssh_host=ssh_host,
                )
                if image_count:
                    logger.debug("Extracted %d images from %s", image_count, page_id)
            except Exception as img_err:
                logger.debug("Image extraction failed for %s: %s", page_id, img_err)

        return chunks
    except Exception as e:
        logger.warning("Failed to ingest %s: %s", page_id, e)
        return 0


# =============================================================================
# Image Artifact Ingestion (standalone wiki image files)
# =============================================================================


async def _ingest_image_artifact(
    artifact_id: str,
    url: str,
    filename: str,
    facility: str,
    ssh_host: str | None = None,
) -> None:
    """Convert an image-type WikiArtifact into an Image node.

    Downloads the image, downsamples to WebP, creates an Image node with
    status='ingested', and links it to both the WikiArtifact (HAS_IMAGE)
    and any linked WikiPages.

    Args:
        artifact_id: WikiArtifact node ID
        url: Image download URL
        filename: Original filename
        facility: Facility ID
        ssh_host: Optional SSH host for proxied fetching
    """
    from imas_codex.discovery.wiki.image import downsample_image, make_image_id

    # Fetch image bytes
    image_bytes = await _fetch_image_bytes(url, ssh_host)
    if not image_bytes or len(image_bytes) < 512:
        logger.debug(
            "Image artifact %s: insufficient bytes (%d)",
            filename,
            len(image_bytes) if image_bytes else 0,
        )
        return

    # Downsample to WebP
    result = downsample_image(image_bytes)
    if result is None:
        logger.debug("Image artifact %s: downsample failed", filename)
        return

    b64_data, stored_w, stored_h, orig_w, orig_h = result
    image_id = make_image_id(facility, url)

    # Persist Image node and link to WikiArtifact + facility
    with GraphClient() as gc:
        gc.query(
            """
            MERGE (i:Image {id: $image_id})
            ON CREATE SET i.facility_id = $facility,
                          i.source_url = $url,
                          i.source_type = 'wiki_file',
                          i.status = 'ingested',
                          i.filename = $filename,
                          i.image_data = $b64_data,
                          i.image_format = 'webp',
                          i.image_width = $stored_w,
                          i.image_height = $stored_h,
                          i.original_width = $orig_w,
                          i.original_height = $orig_h,
                          i.ingested_at = datetime()
            WITH i
            MATCH (wa:WikiArtifact {id: $artifact_id})
            MERGE (wa)-[:HAS_IMAGE]->(i)
            WITH i
            MATCH (f:Facility {id: $facility})
            MERGE (i)-[:FACILITY_ID]->(f)
            """,
            image_id=image_id,
            facility=facility,
            url=url,
            filename=filename,
            artifact_id=artifact_id,
            b64_data=b64_data,
            stored_w=stored_w,
            stored_h=stored_h,
            orig_w=orig_w,
            orig_h=orig_h,
        )

        # Also link to pages that reference this artifact
        gc.query(
            """
            MATCH (wa:WikiArtifact {id: $artifact_id})<-[:HAS_ARTIFACT]-(wp:WikiPage)
            MATCH (i:Image {id: $image_id})
            MERGE (wp)-[:HAS_IMAGE]->(i)
            """,
            artifact_id=artifact_id,
            image_id=image_id,
        )

    logger.debug("Created Image node %s from artifact %s", image_id, filename)


async def _persist_document_figures(
    extracted_images: list[dict[str, Any]],
    artifact_id: str,
    facility: str,
) -> int:
    """Persist images extracted from PDF/PPTX as Image nodes.

    Creates Image nodes with source_type='document_figure' and links
    them to the parent WikiArtifact via HAS_IMAGE.

    Args:
        extracted_images: List from pipeline._extract_pdf_images or _extract_pptx_images
        artifact_id: WikiArtifact node ID
        facility: Facility ID

    Returns:
        Number of Image nodes created
    """
    import hashlib

    from imas_codex.discovery.wiki.image import downsample_image

    images_to_persist: list[dict[str, Any]] = []

    for img_data in extracted_images:
        img_bytes = img_data.get("image_bytes")
        if not img_bytes or len(img_bytes) < 2048:
            continue

        # Content-addressed ID from bytes hash
        content_hash = hashlib.sha256(img_bytes).hexdigest()[:16]
        image_id = f"{facility}:{content_hash}"

        result = downsample_image(img_bytes)
        if result is None:
            continue

        b64_data, stored_w, stored_h, orig_w, orig_h = result

        # Build context from extraction metadata
        page_num = img_data.get("page_num")
        slide_num = img_data.get("slide_num")
        name = img_data.get("name", "")
        source_url = f"{artifact_id}#{'page' if page_num else 'slide'}{page_num or slide_num or 0}"

        images_to_persist.append(
            {
                "id": image_id,
                "facility_id": facility,
                "source_url": source_url,
                "source_type": "document_figure",
                "status": "ingested",
                "filename": name,
                "image_data": b64_data,
                "image_format": "webp",
                "image_width": stored_w,
                "image_height": stored_h,
                "original_width": orig_w,
                "original_height": orig_h,
                "content_hash": content_hash,
                "artifact_id": artifact_id,
            }
        )

    if not images_to_persist:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $images AS img
            MERGE (i:Image {id: img.id})
            ON CREATE SET i.facility_id = img.facility_id,
                          i.source_url = img.source_url,
                          i.source_type = img.source_type,
                          i.status = img.status,
                          i.filename = img.filename,
                          i.image_data = img.image_data,
                          i.image_format = img.image_format,
                          i.image_width = img.image_width,
                          i.image_height = img.image_height,
                          i.original_width = img.original_width,
                          i.original_height = img.original_height,
                          i.content_hash = img.content_hash,
                          i.ingested_at = datetime()
            WITH i, img
            MATCH (wa:WikiArtifact {id: img.artifact_id})
            MERGE (wa)-[:HAS_IMAGE]->(i)
            WITH i
            MATCH (f:Facility {id: i.facility_id})
            MERGE (i)-[:FACILITY_ID]->(f)
            """,
            images=images_to_persist,
        )

    logger.debug(
        "Created %d document figure Image nodes from %s",
        len(images_to_persist),
        artifact_id,
    )
    return len(images_to_persist)


# =============================================================================
# Image Extraction from Wiki Pages
# =============================================================================

# Image file extensions worth processing
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}

# Patterns in filenames/paths that indicate decorative/UI images (skip)
_SKIP_PATTERNS = frozenset(
    [
        "favicon",
        "logo",
        "icon",
        "bullet",
        "spacer",
        "arrow",
        "button",
        "badge",
        "spinner",
        "loading",
        "pixel",
        "blank",
        "thumb_",
        "/skins/",
        "/skin/",
    ]
)

# Maximum images to extract per page
_MAX_IMAGES_PER_PAGE = 20


def _extract_image_refs(
    html: str,
    page_url: str,
    page_title: str,
) -> list[dict[str, str]]:
    """Extract image references from HTML content.

    Parses <img> tags, resolves relative URLs, filters decorative images,
    and captures surrounding context for VLM scoring.

    Args:
        html: Raw HTML content of the page
        page_url: Canonical URL of the page (for resolving relative paths)
        page_title: Page title for context

    Returns:
        List of dicts with keys: src, alt_text, section, surrounding_text,
        width, height
    """
    import re
    import urllib.parse

    if not html:
        return []

    # Parse base URL components for resolving relative paths
    parsed_page = urllib.parse.urlparse(page_url)
    origin = f"{parsed_page.scheme}://{parsed_page.netloc}"
    # Directory of the page URL (for relative paths)
    page_dir = page_url.rsplit("/", 1)[0] if "/" in page_url else page_url

    refs: list[dict[str, str]] = []

    # Build a simple position → section map from headings
    heading_pattern = re.compile(
        r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL
    )
    img_pattern = re.compile(
        r'<img\s[^>]*?src=["\']([^"\']+)["\'][^>]*?>',
        re.IGNORECASE | re.DOTALL,
    )
    alt_pattern = re.compile(r'alt=["\']([^"\']*)["\']', re.IGNORECASE)
    width_pattern = re.compile(r'width=["\']?(\d+)', re.IGNORECASE)
    height_pattern = re.compile(r'height=["\']?(\d+)', re.IGNORECASE)

    # Build section map: sorted list of (position, section_name)
    section_map: list[tuple[int, str]] = []
    for m in heading_pattern.finditer(html):
        heading_text = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        if heading_text:
            section_map.append((m.start(), heading_text))

    def _get_section_at(pos: int) -> str:
        """Get the section heading active at a given position."""
        current = ""
        for sec_pos, sec_name in section_map:
            if sec_pos > pos:
                break
            current = sec_name
        return current

    # Strip HTML tags for surrounding text extraction
    text_only = re.sub(r"<[^>]+>", " ", html)
    text_only = re.sub(r"\s+", " ", text_only)

    for m in img_pattern.finditer(html):
        src_raw = m.group(1).strip()

        # Skip data URIs
        if src_raw.startswith("data:"):
            continue

        # Resolve URL
        if src_raw.startswith("//"):
            src = f"https:{src_raw}"
        elif src_raw.startswith("/"):
            src = f"{origin}{src_raw}"
        elif src_raw.startswith("http://") or src_raw.startswith("https://"):
            src = src_raw
        else:
            # Relative path
            src = f"{page_dir}/{src_raw}"

        # Check extension — only process known image types
        # Strip query params for extension check
        path_part = urllib.parse.urlparse(src).path.lower()
        ext = "." + path_part.rsplit(".", 1)[-1] if "." in path_part else ""
        if ext not in _IMAGE_EXTENSIONS:
            continue

        # Check skip patterns in the full URL path
        path_lower = src.lower()
        if any(pat in path_lower for pat in _SKIP_PATTERNS):
            continue

        # Parse width/height from tag attributes
        tag_text = m.group(0)
        w_match = width_pattern.search(tag_text)
        h_match = height_pattern.search(tag_text)
        width = int(w_match.group(1)) if w_match else 0
        height = int(h_match.group(1)) if h_match else 0

        # Skip tiny images (1x1 pixels, tracking pixels)
        if (width > 0 and width < 32) or (height > 0 and height < 32):
            continue

        # Extract alt text
        alt_match = alt_pattern.search(tag_text)
        alt_text = alt_match.group(1) if alt_match else ""

        # Get section context
        section = _get_section_at(m.start())

        # Extract surrounding text (500 chars around the image position)
        # Map HTML position to approximate text position
        html_before = html[: m.start()]
        text_pos = len(re.sub(r"<[^>]+>", " ", html_before))
        text_pos = min(text_pos, len(text_only))
        start_ctx = max(0, text_pos - 250)
        end_ctx = min(len(text_only), text_pos + 250)
        surrounding = text_only[start_ctx:end_ctx].strip()

        refs.append(
            {
                "src": src,
                "alt_text": alt_text,
                "section": section,
                "surrounding_text": surrounding,
                "width": str(width),
                "height": str(height),
            }
        )

    return refs[:_MAX_IMAGES_PER_PAGE]


async def _extract_and_persist_images(
    html: str,
    page_url: str,
    page_id: str,
    page_title: str,
    facility: str,
    ssh_host: str | None = None,
) -> int:
    """Extract images from page HTML, downsample, and persist as Image nodes.

    Creates Image nodes with status='ingested' and HAS_IMAGE relationships
    to the WikiPage. Images are then available for VLM captioning by
    image_score_worker.

    Args:
        html: Raw HTML of the ingested page
        page_url: Page URL (for resolving relative image paths)
        page_id: WikiPage node ID
        page_title: Page title for context
        facility: Facility ID
        ssh_host: Optional SSH host for fetching images

    Returns:
        Number of images persisted
    """

    from imas_codex.discovery.wiki.image import downsample_image, make_image_id

    refs = _extract_image_refs(html, page_url, page_title)
    if not refs:
        return 0

    # Deduplicate by source URL
    seen_urls: set[str] = set()
    unique_refs = []
    for ref in refs:
        if ref["src"] not in seen_urls:
            seen_urls.add(ref["src"])
            unique_refs.append(ref)

    images_to_persist: list[dict[str, Any]] = []

    for ref in unique_refs:
        src = ref["src"]
        image_id = make_image_id(facility, src)

        try:
            # Fetch image bytes
            image_bytes = await _fetch_image_bytes(src, ssh_host)
            if not image_bytes or len(image_bytes) < 512:
                continue

            # Downsample to WebP
            result = downsample_image(image_bytes)
            if result is None:
                continue

            b64_data, stored_w, stored_h, orig_w, orig_h = result

            images_to_persist.append(
                {
                    "id": image_id,
                    "facility_id": facility,
                    "source_url": src,
                    "source_type": "wiki_inline",
                    "status": "ingested",
                    "image_data": b64_data,
                    "image_format": "webp",
                    "width": stored_w,
                    "height": stored_h,
                    "original_width": orig_w,
                    "original_height": orig_h,
                    "page_title": page_title,
                    "section": ref.get("section", ""),
                    "alt_text": ref.get("alt_text", ""),
                    "surrounding_text": ref.get("surrounding_text", "")[:500],
                    "page_id": page_id,
                }
            )
        except Exception as e:
            logger.debug("Failed to fetch/process image %s: %s", src, e)
            continue

    if not images_to_persist:
        return 0

    # Persist to graph in batch
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $images AS img
            MERGE (i:Image {id: img.id})
            ON CREATE SET i.facility_id = img.facility_id,
                          i.source_url = img.source_url,
                          i.source_type = img.source_type,
                          i.status = img.status,
                          i.image_data = img.image_data,
                          i.image_format = img.image_format,
                          i.width = img.width,
                          i.height = img.height,
                          i.original_width = img.original_width,
                          i.original_height = img.original_height,
                          i.page_title = img.page_title,
                          i.section = img.section,
                          i.alt_text = img.alt_text,
                          i.surrounding_text = img.surrounding_text,
                          i.ingested_at = datetime()
            WITH i, img
            MATCH (wp:WikiPage {id: img.page_id})
            MERGE (wp)-[:HAS_IMAGE]->(i)
            WITH i
            MATCH (f:Facility {id: i.facility_id})
            MERGE (i)-[:FACILITY_ID]->(f)
            """,
            images=images_to_persist,
        )

    return len(images_to_persist)


async def _fetch_image_bytes(
    url: str,
    ssh_host: str | None = None,
    timeout: int = 15,
) -> bytes | None:
    """Fetch image bytes from URL, optionally via SSH proxy.

    Args:
        url: Image URL
        ssh_host: Optional SSH host for proxied fetching
        timeout: Timeout in seconds

    Returns:
        Raw image bytes or None on failure
    """
    import subprocess

    import httpx

    if ssh_host:
        # Fetch via SSH proxy using curl
        cmd = f'curl -sk --noproxy "*" -m {timeout} "{url}" 2>/dev/null'
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ssh", ssh_host, cmd],
                capture_output=True,
                timeout=timeout + 10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            return None
        except Exception:
            return None
    else:
        # Direct HTTP fetch
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(float(timeout)),
                follow_redirects=True,
                verify=False,
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.content
                return None
        except Exception:
            return None
