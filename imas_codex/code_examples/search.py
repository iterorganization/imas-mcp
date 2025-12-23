"""Semantic search over code examples.

Uses Neo4j vector index for similarity search with optional
IDS and facility filtering.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from imas_codex.embeddings.encoder import Encoder
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchResult:
    """A code search result with relevance score."""

    chunk_id: str
    content: str
    function_name: str | None
    source_file: str
    facility_id: str
    related_ids: list[str]
    score: float
    start_line: int | None = None
    end_line: int | None = None


@dataclass
class CodeExampleSearch:
    """Search code examples using vector similarity and graph filtering."""

    encoder: Encoder = field(default_factory=Encoder)
    graph_client: GraphClient = field(default_factory=GraphClient)

    def search(
        self,
        query: str,
        top_k: int = 10,
        ids_filter: list[str] | None = None,
        facility: str | None = None,
        min_score: float = 0.5,
    ) -> list[CodeSearchResult]:
        """Search for code examples matching the query.

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            ids_filter: Optional list of IDS names to filter by
            facility: Optional facility ID to filter by
            min_score: Minimum similarity score threshold

        Returns:
            List of CodeSearchResult objects ordered by relevance
        """
        # Generate query embedding
        query_embedding = self.encoder.embed_texts([query])[0]

        # Build Cypher query with optional filters
        where_clauses = []
        params: dict = {
            "embedding": query_embedding.tolist(),
            "top_k": top_k,
        }

        if facility:
            where_clauses.append("e.facility_id = $facility")
            params["facility"] = facility

        # Use vector index for similarity search
        # Then optionally filter by IDS
        if ids_filter:
            # Filter chunks that have relationships to specified IDS
            ids_filter_lower = [ids.lower() for ids in ids_filter]
            params["ids_filter"] = ids_filter_lower

            cypher = """
                CALL db.index.vector.queryNodes('code_chunk_embedding', $top_k * 2, $embedding)
                YIELD node AS c, score
                MATCH (c)-[:HAS_CHUNK]-(e:CodeExample)
                WHERE (c)-[:RELATED_PATHS]->(:IMASPath)
                WITH c, e, score
                MATCH (c)-[:RELATED_PATHS]->(p:IMASPath)
                WHERE p.ids IN $ids_filter
                WITH c, e, score, collect(DISTINCT p.ids) AS related_ids
            """
        else:
            cypher = """
                CALL db.index.vector.queryNodes('code_chunk_embedding', $top_k, $embedding)
                YIELD node AS c, score
                MATCH (c)-[:HAS_CHUNK]-(e:CodeExample)
                OPTIONAL MATCH (c)-[:RELATED_PATHS]->(p:IMASPath)
                WITH c, e, score, collect(DISTINCT p.ids) AS related_ids
            """

        # Add facility filter if specified
        if where_clauses:
            cypher += f" WHERE {' AND '.join(where_clauses)}"

        # Return results
        cypher += """
            RETURN
                c.id AS chunk_id,
                c.content AS content,
                c.function_name AS function_name,
                c.start_line AS start_line,
                c.end_line AS end_line,
                e.source_file AS source_file,
                e.facility_id AS facility_id,
                related_ids,
                score
            ORDER BY score DESC
            LIMIT $top_k
        """

        with self.graph_client:
            try:
                results = self.graph_client.query(cypher, **params)
            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text match: {e}")
                return self._fallback_search(query, top_k, ids_filter, facility)

        # Convert to result objects
        return [
            CodeSearchResult(
                chunk_id=r["chunk_id"],
                content=r["content"],
                function_name=r["function_name"],
                source_file=r["source_file"],
                facility_id=r["facility_id"],
                related_ids=r["related_ids"] or [],
                score=r["score"],
                start_line=r.get("start_line"),
                end_line=r.get("end_line"),
            )
            for r in results
            if r["score"] >= min_score
        ]

    def _fallback_search(
        self,
        query: str,
        top_k: int,
        ids_filter: list[str] | None = None,
        facility: str | None = None,
    ) -> list[CodeSearchResult]:
        """Fallback to in-memory similarity search if vector index unavailable."""
        # Fetch all chunks with embeddings
        params: dict = {}

        cypher = """
            MATCH (c:CodeChunk)-[:HAS_CHUNK]-(e:CodeExample)
            WHERE c.embedding IS NOT NULL
        """

        if facility:
            cypher += " AND e.facility_id = $facility"
            params["facility"] = facility

        if ids_filter:
            cypher += """
                AND EXISTS {
                    MATCH (c)-[:RELATED_PATHS]->(p:IMASPath)
                    WHERE p.ids IN $ids_filter
                }
            """
            params["ids_filter"] = [ids.lower() for ids in ids_filter]

        cypher += """
            OPTIONAL MATCH (c)-[:RELATED_PATHS]->(p:IMASPath)
            RETURN
                c.id AS chunk_id,
                c.content AS content,
                c.function_name AS function_name,
                c.start_line AS start_line,
                c.end_line AS end_line,
                c.embedding AS embedding,
                e.source_file AS source_file,
                e.facility_id AS facility_id,
                collect(DISTINCT p.ids) AS related_ids
        """

        with self.graph_client:
            results = self.graph_client.query(cypher, **params)

        if not results:
            return []

        # Generate query embedding
        query_embedding = self.encoder.embed_texts([query])[0]

        # Compute similarities
        scored_results = []
        for r in results:
            if r["embedding"]:
                chunk_embedding = np.array(r["embedding"])
                # Cosine similarity (embeddings are normalized)
                score = float(np.dot(query_embedding, chunk_embedding))
                scored_results.append((r, score))

        # Sort by score and return top_k
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return [
            CodeSearchResult(
                chunk_id=r["chunk_id"],
                content=r["content"],
                function_name=r["function_name"],
                source_file=r["source_file"],
                facility_id=r["facility_id"],
                related_ids=r["related_ids"] or [],
                score=score,
                start_line=r.get("start_line"),
                end_line=r.get("end_line"),
            )
            for r, score in scored_results[:top_k]
        ]

    def get_example_by_id(self, example_id: str) -> dict | None:
        """Get full code example by ID."""
        cypher = """
            MATCH (e:CodeExample {id: $id})
            OPTIONAL MATCH (e)<-[:HAS_CHUNK]-(c:CodeChunk)
            RETURN e, collect(c) AS chunks
        """
        with self.graph_client:
            results = self.graph_client.query(cypher, id=example_id)
            if not results:
                return None

            r = results[0]
            return {
                "example": dict(r["e"]),
                "chunks": [dict(c) for c in r["chunks"]],
            }

    def list_examples(
        self,
        facility: str | None = None,
        ids_filter: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List code examples with optional filtering."""
        where_clauses = []
        params: dict = {"limit": limit}

        if facility:
            where_clauses.append("e.facility_id = $facility")
            params["facility"] = facility

        if ids_filter:
            where_clauses.append("any(ids IN e.related_ids WHERE ids IN $ids_filter)")
            params["ids_filter"] = [ids.lower() for ids in ids_filter]

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        cypher = f"""
            MATCH (e:CodeExample)
            {where_clause}
            RETURN e
            ORDER BY e.ingested_at DESC
            LIMIT $limit
        """

        with self.graph_client:
            results = self.graph_client.query(cypher, **params)
            return [dict(r["e"]) for r in results]
