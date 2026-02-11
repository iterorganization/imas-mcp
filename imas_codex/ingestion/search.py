"""Semantic search over ingested code chunks.

Uses Neo4jVectorStore for similarity search with optional
IDS and facility filtering via Neo4j's native metadata filtering.
"""

import logging
from dataclasses import dataclass, field

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from imas_codex.embeddings import get_embed_model
from imas_codex.settings import get_embedding_dimension

logger = logging.getLogger(__name__)


def _create_vector_store(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "imas-codex",
) -> Neo4jVectorStore:
    """Create Neo4jVectorStore for code chunks."""
    return Neo4jVectorStore(
        username=neo4j_user,
        password=neo4j_password,
        url=neo4j_uri,
        embedding_dimension=get_embedding_dimension(),
        index_name="code_chunk_embedding",
        node_label="CodeChunk",
        text_node_property="content",
        embedding_node_property="embedding",
    )


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


@dataclass
class ChunkSearch:
    """Search ingested code chunks using vector similarity.

    Uses the embedding-backend config to choose local or remote embedding.
    Replaces CodeExampleSearch.
    """

    embed_model: BaseEmbedding = field(default_factory=get_embed_model)
    vector_store: Neo4jVectorStore | None = None
    _index: VectorStoreIndex | None = field(default=None, init=False)

    def _get_index(self) -> VectorStoreIndex:
        """Get or create the VectorStoreIndex."""
        if self._index is None:
            vs = self.vector_store or _create_vector_store()
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vs,
                embed_model=self.embed_model,
            )
        return self._index

    def search(
        self,
        query: str,
        top_k: int = 10,
        ids_filter: list[str] | None = None,
        facility: str | None = None,
        min_score: float = 0.5,
    ) -> list[ChunkSearchResult]:
        """Search for code chunks matching the query.

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            ids_filter: Optional list of IDS names to filter by
            facility: Optional facility ID to filter by
            min_score: Minimum similarity score threshold

        Returns:
            List of ChunkSearchResult objects ordered by relevance
        """
        index = self._get_index()

        filters = self._build_filters(facility, ids_filter)

        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )

        try:
            nodes_with_scores = retriever.retrieve(query)
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            return []

        results = []
        for node_with_score in nodes_with_scores:
            node = node_with_score.node
            score = node_with_score.score or 0.0

            if score < min_score:
                continue

            metadata = node.metadata or {}

            results.append(
                ChunkSearchResult(
                    chunk_id=node.node_id,
                    content=node.get_content(),
                    function_name=metadata.get("function_name"),
                    source_file=metadata.get("source_file", ""),
                    facility_id=metadata.get("facility_id", ""),
                    related_ids=metadata.get("related_ids", []),
                    score=score,
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                )
            )

        return results

    def _build_filters(
        self,
        facility: str | None,
        ids_filter: list[str] | None,
    ) -> MetadataFilters | None:
        """Build MetadataFilters for Neo4jVectorStore."""
        if not facility and not ids_filter:
            return None

        filter_list: list[MetadataFilter] = []

        if facility:
            filter_list.append(
                MetadataFilter(
                    key="facility_id",
                    value=facility,
                    operator=FilterOperator.EQ,
                )
            )

        if ids_filter:
            filter_list.append(
                MetadataFilter(
                    key="related_ids",
                    value=ids_filter,
                    operator=FilterOperator.ANY,
                )
            )

        return MetadataFilters(
            filters=filter_list,
            condition=FilterCondition.AND
            if len(filter_list) > 1
            else FilterCondition.OR,
        )


__all__ = [
    "ChunkSearch",
    "ChunkSearchResult",
]
