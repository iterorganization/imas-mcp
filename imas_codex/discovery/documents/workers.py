"""Async workers for document discovery.

Workers:
- image_fetch_worker: Fetch image Documents, downsample, create Image nodes
- image_score_worker: VLM captioning + scoring of Image nodes
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from .pipeline import DocumentDiscoveryState

logger = logging.getLogger(__name__)


def _claim_image_documents(
    facility: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim Document nodes with document_type='image' for processing."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (d:Document {facility_id: $facility, document_type: 'image'})
            WHERE d.status = 'discovered'
              AND d.claimed_at IS NULL
            WITH d ORDER BY d.discovered_at ASC LIMIT $limit
            SET d.claimed_at = datetime()
            RETURN d.id AS id, d.path AS path, d.document_type AS document_type
            """,
            facility=facility,
            limit=limit,
        )
        return list(result)


def _mark_documents_ingested(doc_ids: list[str]) -> int:
    """Mark Document nodes as ingested."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS id
            MATCH (d:Document {id: id})
            SET d.status = 'ingested',
                d.claimed_at = null,
                d.ingested_at = datetime()
            RETURN count(d) AS updated
            """,
            ids=doc_ids,
        )
        return result[0]["updated"] if result else 0


def _mark_document_failed(doc_id: str, error: str) -> None:
    """Mark a Document node as failed."""
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (d:Document {id: $id})
            SET d.status = 'failed',
                d.claimed_at = null,
                d.error = $error
            """,
            id=doc_id,
            error=error[:200],
        )


async def image_fetch_worker(
    state: DocumentDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """Fetch image Documents, downsample, and create Image nodes.

    Claims Document nodes with document_type='image', fetches via SSH tar,
    downsamples, and creates linked Image nodes.

    Transitions Document: discovered → ingested | failed
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
        if state.scan_only:
            break

        docs = await asyncio.to_thread(
            _claim_image_documents,
            state.facility,
            limit=batch_size,
        )

        if not docs:
            state.image_phase.record_idle()
            if state.image_phase.done:
                break
            if on_progress:
                on_progress("idle", state.image_stats, None)
            await asyncio.sleep(3.0)
            continue

        state.image_phase.record_activity(len(docs))

        if on_progress:
            on_progress(f"processing {len(docs)} images", state.image_stats, None)

        images_to_persist: list[dict[str, Any]] = []
        ingested_ids: list[str] = []
        failed_ids: list[tuple[str, str]] = []

        remote_paths = [d["path"] for d in docs]

        with tempfile.TemporaryDirectory() as tmpdir:
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

                    for d in docs:
                        remote_path = d["path"]
                        doc_id = d["id"]
                        local_path = Path(tmpdir) / remote_path.lstrip("/")

                        if not local_path.exists():
                            failed_ids.append((doc_id, "File not found in tar archive"))
                            continue

                        try:
                            image_bytes = local_path.read_bytes()
                        except Exception as e:
                            failed_ids.append((doc_id, str(e)[:200]))
                            continue

                        if len(image_bytes) < 512:
                            failed_ids.append((doc_id, "Image too small (<512 bytes)"))
                            continue

                        ds_result = downsample_image(image_bytes)
                        if ds_result is None:
                            failed_ids.append(
                                (doc_id, "Downsample failed (too small or unreadable)")
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
                                "image_data": None,
                                "page_title": None,
                                "section": None,
                                "alt_text": None,
                                "surrounding_text": None,
                                "document_id": doc_id,
                            }
                        )
                        ingested_ids.append(doc_id)
                else:
                    for d in docs:
                        failed_ids.append((d["id"], "SSH tar fetch failed"))

            except Exception as e:
                logger.error("Image batch fetch failed: %s", e)
                state.image_stats.errors += 1
                for d in docs:
                    await asyncio.to_thread(
                        _mark_document_failed, d["id"], str(e)[:200]
                    )
                continue

        # Persist Image nodes linked to Document
        if images_to_persist:
            await asyncio.to_thread(
                persist_images,
                images_to_persist,
                parent_label="Document",
                parent_id_key="document_id",
            )

        if ingested_ids:
            await asyncio.to_thread(_mark_documents_ingested, ingested_ids)

        for doc_id, error in failed_ids:
            await asyncio.to_thread(_mark_document_failed, doc_id, error)

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

        await asyncio.sleep(0.1)


async def image_score_worker(
    state: DocumentDiscoveryState,
    on_progress: Callable | None = None,
    batch_size: int = 10,
) -> None:
    """VLM captioning + scoring of Image nodes.

    Claims Image nodes with status='ingested', fetches bytes on-demand,
    sends to VLM for caption + scoring.

    Transitions Image: ingested → captioned
    """
    from imas_codex.discovery.base.image import (
        claim_images_for_scoring,
        downsample_image,
        fetch_image_bytes,
        mark_images_scored,
        release_claimed_images,
        score_images_batch,
    )
    from imas_codex.settings import get_model

    worker_id = id(asyncio.current_task())
    logger.info("image_score_worker started (task=%s)", worker_id)

    facility_access_patterns: dict[str, Any] | None = None
    try:
        from imas_codex.discovery.base.facility import get_facility

        facility_config = get_facility(state.facility)
        facility_access_patterns = facility_config.get("data_access_patterns")
    except Exception as e:
        logger.debug("image_score_worker: no data_access_patterns: %s", e)

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    while not state.should_stop():
        if state.scan_only:
            break

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "image_score_worker %s: %d consecutive VLM failures, stopping",
                worker_id,
                consecutive_failures,
            )
            break

        try:
            images = await asyncio.to_thread(
                claim_images_for_scoring, state.facility, batch_size
            )
        except Exception as e:
            logger.warning("image_score_worker: claim failed: %s", e)
            await asyncio.sleep(5.0)
            continue

        if not images:
            state.image_score_phase.record_idle()
            if state.image_score_phase.done:
                break
            if on_progress:
                on_progress("idle", state.image_score_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.image_score_phase.record_activity(len(images))

        images_ready: list[dict[str, Any]] = []
        images_unfetchable: list[str] = []

        for img in images:
            source_url = img.get("source_url")
            stored_data = img.get("image_data")
            if stored_data:
                images_ready.append(img)
                continue
            if not source_url:
                images_unfetchable.append(img["id"])
                continue
            try:
                raw_bytes = await fetch_image_bytes(source_url, state.ssh_host)
                if not raw_bytes or len(raw_bytes) < 512:
                    images_unfetchable.append(img["id"])
                    continue
                result = downsample_image(raw_bytes)
                if result is None:
                    images_unfetchable.append(img["id"])
                    continue
                b64_data, _w, _h, _ow, _oh = result
                img["image_data"] = b64_data
                images_ready.append(img)
            except Exception as e:
                logger.debug(
                    "image_score_worker: failed to fetch %s: %s", source_url, e
                )
                images_unfetchable.append(img["id"])

        if images_unfetchable:
            logger.info(
                "image_score_worker: %d/%d images unfetchable, releasing",
                len(images_unfetchable),
                len(images),
            )
            await asyncio.to_thread(release_claimed_images, images_unfetchable)

        if not images_ready:
            continue

        if on_progress:
            on_progress(
                f"scoring {len(images_ready)} images",
                state.image_score_stats,
                None,
            )

        try:
            model = get_model("vision")
            results, cost = await score_images_batch(
                images_ready,
                model,
                state.focus,
                facility_access_patterns,
                facility_id=state.facility,
            )

            await asyncio.to_thread(
                mark_images_scored,
                state.facility,
                results,
                store_images=state.store_images,
            )
            state.image_score_stats.processed += len(results)
            state.image_score_stats.cost += cost
            consecutive_failures = 0

            if on_progress:
                on_progress(
                    f"scored {len(results)} images (${cost:.3f})",
                    state.image_score_stats,
                    [
                        {
                            "path": r.get("source_url", r["id"]),
                            "score": f"{r.get('score', 0):.2f}",
                        }
                        for r in results[:5]
                    ],
                )

        except ValueError as e:
            consecutive_failures += 1
            logger.warning(
                "VLM failed for batch of %d images: %s (failure %d/%d)",
                len(images_ready),
                e,
                consecutive_failures,
                MAX_CONSECUTIVE_FAILURES,
            )
            state.image_score_stats.errors += 1
            await asyncio.to_thread(
                release_claimed_images,
                [img["id"] for img in images_ready],
            )

        await asyncio.sleep(0.1)
