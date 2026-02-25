"""Shared image processing utilities for discovery pipelines.

Source-agnostic image handling used by both wiki and files discovery:
- Downsampling to WebP for graph storage
- Content-addressed ID generation for deduplication
- Batch persistence of Image nodes to Neo4j
- Image extraction from documents (PDF, PPTX)

Design:
- Images are downsampled to WebP format, max 768px longest side, quality 80
- Stored as base64-encoded strings in Neo4j (no filesystem dependencies)
- Content-addressed IDs: facility:sha256(source_url)[:16] for dedup
- SVG images are rasterized via Pillow's SVG support or stored as-is

Third-party dependencies:
- Pillow (PIL): Image loading, resizing, format conversion, WebP encoding
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
from typing import Any

from PIL import Image

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
