"""Image processing utilities — re-exports from shared base module.

All image utilities now live in ``imas_codex.discovery.base.image``
so they can be shared across wiki and files discovery pipelines.
This module re-exports them for backward compatibility.
"""

from imas_codex.discovery.base.image import (
    IMAGE_EXTENSIONS,
    IMAGE_FORMAT,
    MAX_DIMENSION,
    MIN_BYTES,
    MIN_DIMENSION,
    WEBP_QUALITY,
    claim_images_for_scoring,
    content_hash,
    downsample_image,
    fetch_image_bytes,
    guess_mime_type,
    has_pending_image_work,
    make_image_id,
    mark_images_scored,
    persist_document_figures,
    persist_images,
    release_claimed_images,
    score_images_batch,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "IMAGE_FORMAT",
    "MAX_DIMENSION",
    "MIN_BYTES",
    "MIN_DIMENSION",
    "WEBP_QUALITY",
    "claim_images_for_scoring",
    "content_hash",
    "downsample_image",
    "fetch_image_bytes",
    "guess_mime_type",
    "has_pending_image_work",
    "make_image_id",
    "mark_images_scored",
    "persist_document_figures",
    "persist_images",
    "release_claimed_images",
    "score_images_batch",
]
