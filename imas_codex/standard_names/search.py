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
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS id,
                       sn.description AS description,
                       sn.kind AS kind,
                       coalesce(u.id, sn.unit) AS unit,
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


def search_similar_sns_with_full_docs(
    description_query: str,
    k: int = 5,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Find existing StandardName nodes with full documentation for exemplar use.

    Like :func:`search_similar_names`, but returns richer records including
    ``documentation`` for use as few-shot exemplars in compose
    prompts.

    Args:
        description_query: Natural-language query text.
        k: Maximum number of results (fetches k + extras to allow exclusion).
        exclude_ids: StandardName IDs to exclude from results (e.g. names
            already in the current compose batch).

    Returns:
        List of dicts with keys: ``name``, ``description``, ``documentation``,
        ``unit``.
    """
    if not description_query or not description_query.strip():
        return []

    exclude_set = set(exclude_ids) if exclude_ids else set()
    # Fetch extra to allow for exclusions
    fetch_k = k + len(exclude_set) + 5

    try:
        from imas_codex.embeddings.config import EncoderConfig
        from imas_codex.embeddings.encoder import Encoder
        from imas_codex.graph.client import GraphClient

        encoder = Encoder(EncoderConfig())
        result = encoder.embed_texts([description_query])[0]
        embedding = result.tolist() if hasattr(result, "tolist") else list(result)

        with GraphClient() as gc:
            rows = gc.query(
                """
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $k, $embedding
                )
                YIELD node AS sn, score
                WHERE sn.id IS NOT NULL
                  AND sn.validation_status = 'valid'
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS name,
                       sn.description AS description,
                       sn.documentation AS documentation,
                       coalesce(u.id, sn.unit) AS unit,
                       score
                ORDER BY score DESC
                """,
                embedding=embedding,
                k=fetch_k,
            )
            if not rows:
                return []

            results = []
            for r in rows:
                name = r.get("name", "")
                if name in exclude_set:
                    continue
                results.append(
                    {
                        "name": name,
                        "description": r.get("description") or "",
                        "documentation": r.get("documentation") or "",
                        "unit": r.get("unit") or "1",
                    }
                )
                if len(results) >= k:
                    break
            return results
    except Exception:
        logger.debug("Similar SN full-doc search unavailable", exc_info=True)
        return []
