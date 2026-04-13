"""Standard name search helpers for pipeline use.

Provides structured dict results (not formatted strings) for programmatic
use in compose and review context.  Wraps the same embedding + vector index
infrastructure as the ``search_standard_names`` MCP tool.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def search_similar_names(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Find existing StandardName nodes similar to *query* text.

    Uses :class:`~imas_codex.embeddings.encoder.Encoder` to embed the query,
    then runs a vector search on the ``standard_name_desc_embedding`` index.

    Returns a list of dicts with keys: ``id``, ``description``, ``kind``,
    ``unit``, ``score``.

    Returns an empty list when the graph or embedding service is unavailable.
    """
    if not query or not query.strip():
        return []

    try:
        from imas_codex.embeddings.config import EncoderConfig
        from imas_codex.embeddings.encoder import Encoder
        from imas_codex.graph.client import GraphClient

        encoder = Encoder(EncoderConfig())
        result = encoder.embed_texts([query])[0]
        embedding = result.tolist() if hasattr(result, "tolist") else list(result)

        with GraphClient() as gc:
            rows = gc.query(
                """
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $k, $embedding
                )
                YIELD node AS sn, score
                WHERE sn.id IS NOT NULL
                OPTIONAL MATCH (sn)-[:CANONICAL_UNITS]->(u:Unit)
                RETURN sn.id AS id,
                       sn.description AS description,
                       sn.kind AS kind,
                       coalesce(u.id, sn.canonical_units) AS unit,
                       score
                ORDER BY score DESC
                """,
                embedding=embedding,
                k=k,
            )
            return [dict(r) for r in rows] if rows else []
    except Exception:
        logger.debug("Similar name search unavailable", exc_info=True)
        return []
