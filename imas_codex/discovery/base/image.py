"""Shared image processing utilities for discovery pipelines.

Source-agnostic image handling used by both wiki and files discovery:
- Downsampling to WebP for graph storage
- Content-addressed ID generation for deduplication
- Batch persistence of Image nodes to Neo4j
- Image extraction from documents (PDF, PPTX)
- Image claim/mark/release graph operations for VLM scoring
- Fetch image bytes from URL or filesystem via SSH

Design:
- Images are downsampled to WebP format, max 768px longest side, quality 80
- Stored as base64-encoded strings in Neo4j (no filesystem dependencies)
- Content-addressed IDs: facility:sha256(source_url)[:16] for dedup
- SVG images are rasterized via Pillow's SVG support or stored as-is

Third-party dependencies:
- Pillow (PIL): Image loading, resizing, format conversion, WebP encoding
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import uuid
from typing import Any

from PIL import Image

from imas_codex.discovery.base.claims import DEFAULT_CLAIM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Downsample configuration
MAX_DIMENSION = 768  # Max pixels on longest side
WEBP_QUALITY = 80  # WebP compression quality (0-100)
IMAGE_FORMAT = "webp"  # Target format for storage

# Minimum image size to process (skip tiny icons/bullets)
MIN_DIMENSION = 32  # Skip images smaller than 32x32
MIN_BYTES = 512  # Skip images smaller than 512 bytes

# Image file extensions worth processing
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".svg",
}


def make_image_id(facility_id: str, source_url: str) -> str:
    """Generate content-addressed image ID.

    Format: facility:sha256(source_url)[:16]
    Deduplicates same image referenced from multiple pages.

    Args:
        facility_id: Facility identifier (e.g., "tcv", "jet")
        source_url: Original URL or path of the image

    Returns:
        Deterministic ID string
    """
    url_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16]
    return f"{facility_id}:{url_hash}"


def downsample_image(
    image_bytes: bytes,
    *,
    max_dimension: int = MAX_DIMENSION,
    quality: int = WEBP_QUALITY,
) -> tuple[str, int, int, int, int] | None:
    """Downsample image bytes to WebP format for graph storage.

    Args:
        image_bytes: Raw image bytes (any format Pillow can read)
        max_dimension: Maximum pixels on longest side
        quality: WebP compression quality (0-100)

    Returns:
        Tuple of (base64_data, stored_width, stored_height, orig_width, orig_height)
        or None if image is too small or unreadable.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception:
        logger.debug("Failed to open image with Pillow")
        return None

    # Get original dimensions
    orig_width, orig_height = img.size

    # Skip tiny images (icons, bullets, spacers)
    if orig_width < MIN_DIMENSION or orig_height < MIN_DIMENSION:
        logger.debug(
            "Skipping tiny image: %dx%d < %dx%d",
            orig_width,
            orig_height,
            MIN_DIMENSION,
            MIN_DIMENSION,
        )
        return None

    # Convert to RGB if necessary (WebP doesn't support all modes)
    if img.mode in ("RGBA", "LA", "PA"):
        # Preserve alpha by compositing on white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background.convert("RGB")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Downsample if larger than max_dimension
    longest_side = max(orig_width, orig_height)
    if longest_side > max_dimension:
        scale = max_dimension / longest_side
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else:
        new_width, new_height = orig_width, orig_height

    # Encode to WebP
    buffer = io.BytesIO()
    img.save(buffer, format="WEBP", quality=quality)
    webp_bytes = buffer.getvalue()

    # Base64 encode for Neo4j string storage
    b64_data = base64.b64encode(webp_bytes).decode("ascii")

    return b64_data, new_width, new_height, orig_width, orig_height


def content_hash(image_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw image bytes for dedup verification.

    Args:
        image_bytes: Raw image bytes

    Returns:
        Full SHA-256 hex digest
    """
    return hashlib.sha256(image_bytes).hexdigest()


def guess_mime_type(filename: str) -> str:
    """Guess MIME type from filename extension.

    Args:
        filename: Image filename (e.g., "plot.png")

    Returns:
        MIME type string (e.g., "image/png")
    """
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "svg": "image/svg+xml",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "ico": "image/x-icon",
    }
    return mime_map.get(ext, "image/unknown")


def persist_images(
    images: list[dict[str, Any]],
    *,
    parent_label: str | None = None,
    parent_id_key: str | None = None,
) -> int:
    """Batch-persist Image nodes to the graph.

    Creates or merges Image nodes and links them to the parent facility.
    Optionally links to a parent node (WikiPage, WikiArtifact, SourceFile, etc.)
    via HAS_IMAGE relationship.

    Args:
        images: List of image dicts. Required keys: id, facility_id, source_url,
            source_type, status. Optional: image_data, width, height, page_title, etc.
        parent_label: Optional Neo4j label for the parent node (e.g., "SourceFile")
        parent_id_key: Key in image dict that holds the parent node's ID
            (e.g., "source_file_id"). Required when parent_label is set.

    Returns:
        Number of images persisted
    """
    if not images:
        return 0

    from imas_codex.graph import GraphClient

    # Build Cypher — conditionally include parent linking
    set_clause = """
        ON CREATE SET i.facility_id = img.facility_id,
                      i.source_url = img.source_url,
                      i.source_type = img.source_type,
                      i.status = img.status,
                      i.filename = img.filename,
                      i.image_format = img.image_format,
                      i.width = img.width,
                      i.height = img.height,
                      i.original_width = img.original_width,
                      i.original_height = img.original_height,
                      i.content_hash = img.content_hash,
                      i.page_title = img.page_title,
                      i.section = img.section,
                      i.alt_text = img.alt_text,
                      i.surrounding_text = img.surrounding_text,
                      i.ingested_at = datetime()
        ON MATCH SET i.status = CASE WHEN i.status = 'failed' THEN img.status ELSE i.status END
    """

    # Store image_data only if present
    cypher = f"""
        UNWIND $images AS img
        MERGE (i:Image {{id: img.id}})
        {set_clause}
        WITH i, img
        FOREACH (_ IN CASE WHEN img.image_data IS NOT NULL THEN [1] ELSE [] END |
            SET i.image_data = img.image_data
        )
        WITH i
        MATCH (f:Facility {{id: i.facility_id}})
        MERGE (i)-[:AT_FACILITY]->(f)
    """

    with GraphClient() as gc:
        gc.query(cypher, images=images)

        # Link to parent if specified
        if parent_label and parent_id_key:
            parent_ids = [
                (img["id"], img[parent_id_key])
                for img in images
                if img.get(parent_id_key)
            ]
            if parent_ids:
                gc.query(
                    f"""
                    UNWIND $pairs AS pair
                    MATCH (i:Image {{id: pair[0]}})
                    MATCH (p:{parent_label} {{id: pair[1]}})
                    MERGE (p)-[:HAS_IMAGE]->(i)
                    """,
                    pairs=parent_ids,
                )

    return len(images)


def persist_document_figures(
    extracted_images: list[dict[str, Any]],
    parent_id: str,
    parent_label: str,
    facility: str,
) -> int:
    """Persist images extracted from documents (PDF/PPTX) as Image nodes.

    Downsamples image bytes, creates content-addressed Image nodes,
    and links them to the parent document node.

    Args:
        extracted_images: List from _extract_pdf_images or _extract_pptx_images.
            Each dict has: image_bytes, page_num or slide_num, name
        parent_id: ID of the parent node (WikiArtifact or SourceFile)
        parent_label: Neo4j label of parent ("WikiArtifact" or "SourceFile")
        facility: Facility ID

    Returns:
        Number of Image nodes created
    """
    images_to_persist: list[dict[str, Any]] = []

    for img_data in extracted_images:
        img_bytes = img_data.get("image_bytes")
        if not img_bytes or len(img_bytes) < 2048:
            continue

        # Content-addressed ID from bytes hash
        img_hash = content_hash(img_bytes)[:16]
        image_id = f"{facility}:{img_hash}"

        result = downsample_image(img_bytes)
        if result is None:
            continue

        b64_data, stored_w, stored_h, orig_w, orig_h = result

        # Build context from extraction metadata
        page_num = img_data.get("page_num")
        slide_num = img_data.get("slide_num")
        name = img_data.get("name", "")
        source_url = (
            f"{parent_id}#{'page' if page_num else 'slide'}{page_num or slide_num or 0}"
        )

        images_to_persist.append(
            {
                "id": image_id,
                "facility_id": facility,
                "source_url": source_url,
                "source_type": "document_figure",
                "status": "ingested",
                "filename": name,
                "image_format": "webp",
                "width": stored_w,
                "height": stored_h,
                "original_width": orig_w,
                "original_height": orig_h,
                "content_hash": img_hash,
                "image_data": b64_data,
                "page_title": None,
                "section": None,
                "alt_text": None,
                "surrounding_text": None,
                "parent_id": parent_id,
            }
        )

    if not images_to_persist:
        return 0

    return persist_images(
        images_to_persist,
        parent_label=parent_label,
        parent_id_key="parent_id",
    )


# =============================================================================
# Image Claim/Mark/Release Graph Operations (shared by wiki + files)
# =============================================================================


def has_pending_image_work(facility: str) -> bool:
    """Check if there are images pending VLM scoring (status=ingested, no description)."""
    try:
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            result = gc.query(
                """
                MATCH (img:Image {facility_id: $facility})
                WHERE img.status = 'ingested'
                  AND img.description IS NULL
                RETURN count(img) > 0 AS has_work
                """,
                facility=facility,
            )
            rows = list(result)
            return rows[0]["has_work"] if rows else False
    except Exception:
        return False


def claim_images_for_scoring(
    facility: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Claim ingested images for VLM scoring.

    Images with status='ingested' and no description are ready for VLM processing.
    Uses claim token pattern with timeout-based orphan recovery.
    """
    from imas_codex.graph import GraphClient

    cutoff = f"PT{DEFAULT_CLAIM_TIMEOUT_SECONDS}S"
    claim_token = str(uuid.uuid4())

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (img:Image {facility_id: $facility})
            WHERE img.status = 'ingested'
              AND img.description IS NULL
              AND (img.claimed_at IS NULL
                   OR img.claimed_at < datetime() - duration($cutoff))
            WITH img
            ORDER BY rand()
            LIMIT $limit
            SET img.claimed_at = datetime(), img.claim_token = $token
            """,
            facility=facility,
            cutoff=cutoff,
            limit=limit,
            token=claim_token,
        )

        result = gc.query(
            """
            MATCH (img:Image {facility_id: $facility, claim_token: $token})
            OPTIONAL MATCH (sibling:Image {facility_id: $facility,
                                           page_title: img.page_title})
              WHERE img.page_title IS NOT NULL
            WITH img, count(sibling) AS page_image_count
            RETURN img.id AS id,
                   img.source_url AS source_url,
                   img.source_type AS source_type,
                   img.image_format AS image_format,
                   img.page_title AS page_title,
                   img.section AS section,
                   img.surrounding_text AS surrounding_text,
                   img.alt_text AS alt_text,
                   img.image_data AS image_data,
                   page_image_count
            """,
            facility=facility,
            token=claim_token,
        )
        claimed = list(result)

        logger.debug(
            "claim_images_for_scoring: requested %d, won %d (token=%s)",
            limit,
            len(claimed),
            claim_token[:8],
        )
        return claimed


def mark_images_scored(
    facility: str,
    results: list[dict[str, Any]],
    *,
    store_images: bool = False,
) -> int:
    """Mark images as scored with VLM results.

    Updates image status to 'captioned' and persists description + scoring fields.
    When store_images is False (default), clears image_data to free graph storage.
    """
    if not results:
        return 0

    from imas_codex.graph import GraphClient

    clear_data = "" if store_images else ", img.image_data = null"

    with GraphClient() as gc:
        gc.query(
            f"""
            UNWIND $batch AS item
            MATCH (img:Image {{id: item.id}})
            SET img.status = 'captioned',
                img.mermaid_diagram = item.mermaid_diagram,
                img.ocr_text = item.ocr_text,
                img.ocr_mdsplus_paths = item.ocr_mdsplus_paths,
                img.ocr_imas_paths = item.ocr_imas_paths,
                img.ocr_ppf_paths = item.ocr_ppf_paths,
                img.ocr_tool_mentions = item.ocr_tool_mentions,
                img.purpose = item.purpose,
                img.description = item.description,
                img.score = item.score,
                img.score_data_documentation = item.score_data_documentation,
                img.score_physics_content = item.score_physics_content,
                img.score_code_documentation = item.score_code_documentation,
                img.score_data_access = item.score_data_access,
                img.score_calibration = item.score_calibration,
                img.score_imas_relevance = item.score_imas_relevance,
                img.reasoning = item.reasoning,
                img.keywords = item.keywords,
                img.physics_domain = item.physics_domain,
                img.should_ingest = item.should_ingest,
                img.skip_reason = item.skip_reason,
                img.score_cost = item.score_cost,
                img.scored_at = datetime(),
                img.captioned_at = datetime(),
                img.claimed_at = null
                {clear_data}
            """,
            batch=results,
        )

    logger.info(
        "mark_images_scored: updated %d images to captioned for %s",
        len(results),
        facility,
    )
    return len(results)


def release_claimed_images(image_ids: list[str]) -> None:
    """Release claimed images back to pool (e.g., on VLM failure)."""
    if not image_ids:
        return
    try:
        from imas_codex.graph import GraphClient

        with GraphClient() as gc:
            gc.query(
                """
                UNWIND $ids AS id
                MATCH (img:Image {id: id})
                SET img.claimed_at = null
                """,
                ids=image_ids,
            )
    except Exception as e:
        logger.warning("Could not release %d claimed images: %s", len(image_ids), e)


# =============================================================================
# Fetch image bytes (URL or filesystem via SSH)
# =============================================================================


async def fetch_image_bytes(
    source: str,
    ssh_host: str | None = None,
    timeout: int = 15,
    session: Any = None,
) -> bytes | None:
    """Fetch image bytes from URL or filesystem path, optionally via SSH.

    Dispatches based on source format:
    - Absolute paths (starting with /): read via SSH cat
    - URLs (http/https): fetch via HTTP, SSH curl proxy, or auth session

    Args:
        source: Image URL or absolute filesystem path
        ssh_host: SSH host for proxied fetching or remote filesystem access
        timeout: Timeout in seconds
        session: Optional authenticated requests.Session (bypasses SSH)

    Returns:
        Raw image bytes or None on failure
    """
    import subprocess

    # Filesystem path — read via SSH
    if source.startswith("/"):
        if not ssh_host:
            logger.debug("Cannot fetch filesystem path without ssh_host: %s", source)
            return None
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["ssh", ssh_host, f"cat '{source}'"],
                capture_output=True,
                timeout=timeout + 10,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
            return None
        except Exception:
            return None

    # Authenticated session (e.g. Confluence)
    if session:
        import requests as _req

        dl_session = _req.Session()
        dl_session.cookies.update(session.cookies)
        dl_session.headers["Accept"] = "*/*"
        try:
            resp = await asyncio.to_thread(
                dl_session.get, source, timeout=timeout, verify=False
            )
            if resp.status_code == 200:
                return resp.content
            return None
        except Exception:
            return None

    # SSH-proxied HTTP fetch
    if ssh_host:
        cmd = f'curl -sk --noproxy "*" -m {timeout} "{source}" 2>/dev/null'
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

    # Direct HTTP fetch
    import httpx

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(float(timeout)),
            follow_redirects=True,
            verify=False,
        ) as client:
            resp = await client.get(source)
            if resp.status_code == 200:
                return resp.content
            return None
    except Exception:
        return None


# =============================================================================
# VLM Scoring
# =============================================================================


async def score_images_batch(
    images: list[dict[str, Any]],
    model: str,
    focus: str | None = None,
    data_access_patterns: dict[str, Any] | None = None,
    facility_id: str | None = None,
) -> tuple[list[dict[str, Any]], float]:
    """Score a batch of images using VLM with structured output.

    Sends image bytes + context to VLM and receives caption + scoring
    in a single pass. Uses ImageScoreBatch Pydantic model.

    Args:
        images: List of image dicts with id, image_data, page_title, etc.
        model: Model identifier from get_model()
        focus: Optional focus area for scoring
        data_access_patterns: Optional facility-specific data access patterns.
            When provided, injected into the prompt template so the VLM can
            recognize facility-specific path formats and tool references.
        facility_id: Facility identifier for entity extraction (e.g., 'tcv')

    Returns:
        (results, cost) tuple
    """
    import os
    import re

    from imas_codex.discovery.base.llm import set_litellm_offline_env

    set_litellm_offline_env()
    import litellm

    from imas_codex.agentic.prompt_loader import render_prompt
    from imas_codex.discovery.wiki.models import (
        ImageScoreBatch,
        grounded_image_score,
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    model_id = model
    if not model_id.startswith("openrouter/"):
        model_id = f"openrouter/{model_id}"

    # Build system prompt
    context: dict[str, Any] = {}
    if focus:
        context["focus"] = focus
    if data_access_patterns:
        context["data_access_patterns"] = data_access_patterns
    system_prompt = render_prompt("wiki/image-captioner", context)

    # Build user message with image content
    user_content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Score and caption these {len(images)} images "
                f"from fusion facility documentation.\n"
            ),
        }
    ]

    for i, img in enumerate(images, 1):
        context_parts = [f"\n## Image {i}", f"ID: {img['id']}"]
        if img.get("page_title"):
            context_parts.append(f"Page: {img['page_title']}")
        if img.get("section"):
            context_parts.append(f"Section: {img['section']}")
        if img.get("surrounding_text"):
            context_parts.append(f"Context: {img['surrounding_text'][:500]}")
        if img.get("alt_text"):
            context_parts.append(f"Alt text: {img['alt_text']}")

        user_content.append({"type": "text", "text": "\n".join(context_parts)})

        img_format = img.get("image_format", "webp")
        mime_type = f"image/{img_format}"
        user_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img['image_data']}",
                },
            }
        )

    user_content.append(
        {
            "type": "text",
            "text": "\n\nReturn results for each image in order. "
            "The response format is enforced by the schema.",
        }
    )

    # Retry loop
    max_retries = 5
    retry_base_delay = 4.0
    last_error = None
    total_cost = 0.0

    from imas_codex.discovery.base.llm import (
        _supports_cache_control,
        inject_cache_control,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if _supports_cache_control(model):
        messages = inject_cache_control(messages)

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(
                model=model_id,
                api_key=api_key,
                response_format=ImageScoreBatch,
                messages=messages,
                temperature=0.3,
                max_tokens=32000,
                timeout=180,
            )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = response._hidden_params["response_cost"]
            else:
                cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

            total_cost += cost

            content = response.choices[0].message.content
            if not content:
                return [], total_cost

            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
            content = content.encode("utf-8", errors="surrogateescape").decode(
                "utf-8", errors="replace"
            )

            batch = ImageScoreBatch.model_validate_json(content)
            llm_results = batch.results
            break

        except Exception as e:
            last_error = e
            error_msg = str(e).lower()

            from imas_codex.discovery.base.llm import (
                ProviderBudgetExhausted,
                _is_budget_exhausted,
            )

            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {str(e)[:200]}"
                ) from e

            is_retryable = any(
                x in error_msg
                for x in [
                    "overloaded",
                    "rate",
                    "429",
                    "503",
                    "timeout",
                    "eof",
                    "json",
                    "truncated",
                    "validation",
                ]
            )
            if is_retryable and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "VLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    str(e)[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                raise ValueError(f"VLM failed after {attempt + 1} attempts: {e}") from e
    else:
        raise last_error  # type: ignore[misc]

    # Convert to result dicts with grounded scoring
    cost_per_image = total_cost / len(images) if images else 0.0
    results: list[dict[str, Any]] = []

    # Load facility key_tools for OCR text pattern matching
    _facility_key_tools: list[str] | None = None
    _facility_code_patterns: list[str] | None = None
    if data_access_patterns:
        _facility_key_tools = data_access_patterns.get("key_tools")
        _facility_code_patterns = data_access_patterns.get("code_import_patterns")

    # Build facility-aware extractor for OCR entity extraction
    _extractor = None
    if facility_id:
        from imas_codex.discovery.wiki.entity_extraction import (
            FacilityEntityExtractor,
        )

        _extractor = FacilityEntityExtractor(facility_id)
    else:
        from imas_codex.discovery.wiki.entity_extraction import (
            extract_facility_tool_mentions,
            extract_mdsplus_paths,
        )

    for r in llm_results[: len(images)]:
        scores = {
            "score_data_documentation": r.score_data_documentation,
            "score_physics_content": r.score_physics_content,
            "score_code_documentation": r.score_code_documentation,
            "score_data_access": r.score_data_access,
            "score_calibration": r.score_calibration,
            "score_imas_relevance": r.score_imas_relevance,
        }
        combined_score = grounded_image_score(scores, r.purpose)

        # Extract structured entities from VLM OCR text
        ocr_mdsplus_paths: list[str] = []
        ocr_tool_mentions: list[str] = []
        ocr_imas_paths: list[str] = []
        ocr_ppf_paths: list[str] = []
        if r.ocr_text:
            if _extractor:
                ocr_entities = _extractor.extract(r.ocr_text)
                ocr_mdsplus_paths = ocr_entities.mdsplus_paths
                ocr_imas_paths = ocr_entities.imas_paths
                ocr_ppf_paths = ocr_entities.ppf_paths
                ocr_tool_mentions = ocr_entities.tool_mentions
            else:
                ocr_mdsplus_paths = extract_mdsplus_paths(r.ocr_text)
                ocr_tool_mentions = extract_facility_tool_mentions(
                    r.ocr_text, _facility_key_tools, _facility_code_patterns
                )

        results.append(
            {
                "id": r.id,
                "mermaid_diagram": r.mermaid_diagram,
                "ocr_text": r.ocr_text,
                "ocr_mdsplus_paths": ocr_mdsplus_paths,
                "ocr_imas_paths": ocr_imas_paths,
                "ocr_ppf_paths": ocr_ppf_paths,
                "ocr_tool_mentions": ocr_tool_mentions,
                "purpose": r.purpose.value,
                "description": r.description,
                "score": combined_score,
                "score_data_documentation": r.score_data_documentation,
                "score_physics_content": r.score_physics_content,
                "score_code_documentation": r.score_code_documentation,
                "score_data_access": r.score_data_access,
                "score_calibration": r.score_calibration,
                "score_imas_relevance": r.score_imas_relevance,
                "reasoning": r.reasoning,
                "keywords": r.keywords[:5],
                "physics_domain": r.physics_domain.value if r.physics_domain else None,
                "should_ingest": r.should_ingest,
                "skip_reason": r.skip_reason or None,
                "score_cost": cost_per_image,
            }
        )

    return results, total_cost
