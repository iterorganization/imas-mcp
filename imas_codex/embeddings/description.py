"""Batch embedding utilities for description fields.

Provides efficient batch embedding that integrates with UNWIND-based
graph persistence patterns. Workers collect descriptions, embed in batch,
then persist with embeddings in a single graph operation.
"""

import logging
from typing import Any

import numpy as np

from imas_codex.embeddings.encoder import Encoder

logger = logging.getLogger(__name__)

# Module-level encoder - lazily initialized
_encoder: Encoder | None = None


def _get_encoder() -> Encoder:
    """Get or create the module-level encoder."""
    global _encoder
    if _encoder is None:
        _encoder = Encoder()
    return _encoder


def embed_descriptions_batch(
    items: list[dict[str, Any]],
    text_field: str = "description",
    embedding_field: str = "embedding",
) -> list[dict[str, Any]]:
    """Add embeddings to a batch of items for UNWIND-based graph persistence.

    This function is designed to integrate with existing batch processing patterns.
    It extracts text from each item, embeds all texts in a single batch call,
    then adds the embeddings back to each item dict for graph persistence.

    Items without a text field or with empty text get None for embedding.

    Args:
        items: List of dicts to enrich with embeddings
        text_field: Key containing text to embed (default: "description")
        embedding_field: Key to store embedding (default: "embedding")

    Returns:
        Same list of items with embedding_field added to each

    Example:
        >>> signals = [
        ...     {"id": "tcv:eq/ip", "description": "Plasma current"},
        ...     {"id": "tcv:eq/q95", "description": "Edge safety factor"},
        ... ]
        >>> enriched = embed_descriptions_batch(signals)
        >>> # Now each signal has embedding ready for UNWIND
        >>> gc.query('''
        ...     UNWIND $signals AS sig
        ...     MATCH (s:FacilitySignal {id: sig.id})
        ...     SET s.description = sig.description,
        ...         s.embedding = sig.embedding
        ... ''', signals=enriched)
    """
    if not items:
        return items

    # Extract texts and track which items have valid text
    texts_to_embed: list[str] = []
    text_indices: list[int] = []  # Maps embedding index -> item index

    for i, item in enumerate(items):
        text = item.get(text_field)
        if text and isinstance(text, str) and text.strip():
            texts_to_embed.append(text.strip())
            text_indices.append(i)
        else:
            # No text - set embedding to None
            item[embedding_field] = None

    if not texts_to_embed:
        return items

    # Batch embed all texts at once
    try:
        encoder = _get_encoder()
        embeddings = encoder.embed_texts(texts_to_embed)

        # Add embeddings back to items
        for emb_idx, item_idx in enumerate(text_indices):
            embedding = embeddings[emb_idx]
            # Convert to list for JSON/Cypher compatibility
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            items[item_idx][embedding_field] = embedding

    except Exception as e:
        logger.warning("Failed to batch embed descriptions: %s", e)
        # On failure, set all pending embeddings to None
        for item_idx in text_indices:
            items[item_idx][embedding_field] = None

    return items


def embed_description(text: str | None) -> list[float] | None:
    """Embed a single description text.

    Convenience wrapper for single-item cases. For batch processing,
    use embed_descriptions_batch() instead.

    Args:
        text: Description text to embed, or None

    Returns:
        Embedding as list of floats, or None if text is empty/None
    """
    if not text or not isinstance(text, str) or not text.strip():
        return None

    try:
        encoder = _get_encoder()
        embeddings = encoder.embed_texts([text.strip()])
        embedding = embeddings[0]
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return list(embedding)
    except Exception as e:
        logger.warning("Failed to embed description: %s", e)
        return None


__all__ = ["embed_descriptions_batch", "embed_description"]
