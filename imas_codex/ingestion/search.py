"""Semantic search over ingested code chunks.

Uses direct Cypher vector queries with optional facility and IDS filtering.
"""

import logging
from dataclasses import dataclass

from imas_codex.embeddings import Encoder
from imas_codex.graph import GraphClient
from imas_codex.graph.vector_search import build_vector_search

logger = logging.getLogger(__name__)


@dataclass
class ChunkSearchResult:
    """A chunk search result with relevance score."""

    chunk_id: str
    content: str
    function_name: str | None
    source_file: str
    facility_id: str
    related_ids: list[str]
    score: float
    start_line: int | None = None
    end_line: int | None = None


def search_code_chunks(
    query: str,
    top_k: int = 10,
    facility: str | None = None,
    ids_filter: list[str] | None = None,
    min_score: float = 0.5,
) -> list[ChunkSearchResult]:
    """Search code chunks using vector similarity via direct Cypher.

    Args:
        query: Natural language search query
        top_k: Maximum number of results to return
        facility: Optional facility ID to filter by
        ids_filter: Optional list of IDS names to filter by
        min_score: Minimum similarity score threshold

    Returns:
        List of ChunkSearchResult objects ordered by relevance
    """
    encoder = Encoder()
    embeddings = encoder.embed_texts([query])
    embedding = embeddings[0].tolist()

    where_clauses: list[str] = []
    params: dict = {"embedding": embedding, "k": top_k}

    if facility:
        where_clauses.append("node.facility_id = $facility")
        params["facility"] = facility
    if ids_filter:
        where_clauses.append("any(ids IN node.related_ids WHERE ids IN $ids_filter)")
        params["ids_filter"] = ids_filter

    search_block = build_vector_search(
        "code_chunk_embedding",
        "CodeChunk",
        where_clauses=where_clauses or None,
        node_alias="node",
    )

    with GraphClient() as gc:
        results = gc.query(
            f"""
            {search_block}
            RETURN node.id AS chunk_id,
                   node.text AS content,
                   node.function_name AS function_name,
                   node.source_file AS source_file,
                   node.facility_id AS facility_id,
                   node.related_ids AS related_ids,
                   node.start_line AS start_line,
                   node.end_line AS end_line,
                   score
            ORDER BY score DESC
            """,
            **params,
        )

    return [ChunkSearchResult(**r) for r in results if r["score"] >= min_score]


__all__ = [
    "ChunkSearchResult",
    "search_code_chunks",
]
