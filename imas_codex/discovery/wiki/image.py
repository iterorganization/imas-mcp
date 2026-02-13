"""Image processing utilities for wiki discovery pipeline.

Handles downloading, downsampling, and preparing images for graph
storage and VLM captioning.

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

from PIL import Image

logger = logging.getLogger(__name__)

# Downsample configuration
MAX_DIMENSION = 768  # Max pixels on longest side
WEBP_QUALITY = 80  # WebP compression quality (0-100)
IMAGE_FORMAT = "webp"  # Target format for storage

# Minimum image size to process (skip tiny icons/bullets)
MIN_DIMENSION = 32  # Skip images smaller than 32x32
MIN_BYTES = 512  # Skip images smaller than 512 bytes


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
