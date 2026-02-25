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
    content_hash,
    downsample_image,
    guess_mime_type,
    make_image_id,
    persist_document_figures,
    persist_images,
)

__all__ = [
    "IMAGE_EXTENSIONS",
    "IMAGE_FORMAT",
    "MAX_DIMENSION",
    "MIN_BYTES",
    "MIN_DIMENSION",
    "WEBP_QUALITY",
    "content_hash",
    "downsample_image",
    "guess_mime_type",
    "make_image_id",
    "persist_document_figures",
    "persist_images",
]
