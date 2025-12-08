"""
Clusters tool for searching semantically related IMAS paths.

Provides search over clusters of related paths using:
- Path lookup (exact match)
- Natural language queries (semantic search on labels/descriptions)
- IDS filtering
"""

import logging
from typing import Any

from fastmcp import Context

from imas_mcp.clusters.search import ClusterSearcher, ClusterSearchResult
from imas_mcp.core.clusters import Clusters
from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder
from imas_mcp.models.error_models import ToolError
from imas_mcp.search.decorators import cache_results, handle_errors, mcp_tool

from .base import BaseTool

logger = logging.getLogger(__name__)


class ClustersTool(BaseTool):
    """
    Search tool for discovering related IMAS data paths via semantic clusters.

    Clusters group paths with similar physics meaning, enabling discovery of:
    - Cross-IDS clusters (same concept across different IDS)
    - Intra-IDS clusters (related paths within an IDS)
    """

    def __init__(self, *args, **kwargs):
        """Initialize clusters tool."""
        super().__init__(*args, **kwargs)

        # Create encoder config
        ids_set = (
            self.document_store.ids_set
            if hasattr(self.document_store, "ids_set")
            else None
        )

        self._encoder_config = EncoderConfig(
            model_name=None,
            device=None,
            batch_size=250,
            normalize_embeddings=True,
            enable_cache=True,
            ids_set=ids_set,
            use_rich=False,
        )

        # Use the Clusters class for unified management with auto-build
        self._clusters: Clusters | None = None
        self._searcher: ClusterSearcher | None = None
        self._encoder: Encoder | None = None

        self._initialize_clusters()

    def _initialize_clusters(self) -> None:
        """Initialize clusters using the Clusters class with auto-build support."""
        try:
            self._clusters = Clusters(encoder_config=self._encoder_config)

            # Check if clusters are available (will auto-build if needed)
            if self._clusters.is_available():
                clusters_data = self._clusters.get_clusters()
                self._searcher = ClusterSearcher(clusters=clusters_data)
                logger.info(f"Loaded {len(clusters_data)} clusters")
            else:
                logger.warning("Clusters not available")
                self._searcher = None
        except Exception as e:
            logger.error(f"Failed to initialize clusters: {e}")
            self._clusters = None
            self._searcher = None

    def _get_encoder(self) -> Encoder:
        """Get or create encoder for query embedding."""
        if self._encoder is None:
            self._encoder = Encoder(self._encoder_config)
        return self._encoder

    @property
    def tool_name(self) -> str:
        """Return the name of this tool."""
        return "search_imas_clusters"

    @cache_results(ttl=600, key_strategy="path_based")
    @handle_errors(fallback="cluster_suggestions")
    @mcp_tool(
        "Search for semantically related IMAS path clusters. "
        "Accepts paths (e.g., 'equilibrium/time_slice/profiles_2d/b_field_r') "
        "or natural language queries (e.g., 'electron temperature measurements')."
    )
    async def search_imas_clusters(
        self,
        query: str,
        ctx: Context | None = None,
    ) -> dict[str, Any] | ToolError:
        """
        Search for semantically related IMAS path clusters.

        Automatically detects query type:
        - Paths (contain '/') → exact cluster lookup
        - Natural language → semantic search on cluster labels/descriptions

        Args:
            query: Path or natural language query
            ctx: MCP context

        Returns:
            Dict with matching clusters and their paths

        Examples:
            search_imas_clusters(query="core_profiles/profiles_1d/electrons/density")
            search_imas_clusters(query="electron temperature measurements")
        """
        if not self._searcher:
            return ToolError(
                error="Cluster search not available",
                suggestions=[
                    "Ensure clusters.json exists",
                    "Try rebuilding with: build-clusters",
                ],
                context={"tool": "search_imas_clusters"},
            )

        try:
            # Detect query type and search
            is_path = "/" in query and " " not in query

            if is_path:
                # Path lookup
                results = self._searcher.search_by_path(query)
            else:
                # Natural language search
                encoder = self._get_encoder()
                results = self._searcher.search_by_text(
                    query=query,
                    encoder=encoder,
                    top_k=10,
                    similarity_threshold=0.3,
                )

            if not results:
                return ToolError(
                    error=f"No clusters found for query: {query}",
                    suggestions=[
                        "Try a broader search term",
                        "Use search_imas_paths() for direct path search",
                        "Check available IDS with get_imas_overview()",
                    ],
                    context={"query": query},
                )

            # Format response
            return self._format_results(query, results, is_path)

        except Exception as e:
            logger.error(f"Cluster search failed: {e}")
            return ToolError(
                error=str(e),
                suggestions=[
                    "Try a different query",
                    "Use search_imas_paths() for path search",
                ],
                context={"query": query, "tool": "search_imas_clusters"},
            )

    def _format_results(
        self,
        query: str,
        results: list[ClusterSearchResult],
        is_path: bool,
    ) -> dict[str, Any]:
        """Format search results for output."""
        clusters = []

        for result in results:
            cluster_data = {
                "id": result.cluster_id,
                "label": result.label or f"Cluster {result.cluster_id}",
                "description": result.description,
                "type": "cross_ids" if result.is_cross_ids else "intra_ids",
                "ids": result.ids_names,
                "similarity": round(result.similarity_score, 3),
                "paths": result.paths[:20],  # Limit paths in response
            }

            if len(result.paths) > 20:
                cluster_data["total_paths"] = len(result.paths)
                cluster_data["paths_truncated"] = True

            clusters.append(cluster_data)

        return {
            "query": query,
            "query_type": "path" if is_path else "semantic",
            "clusters_found": len(clusters),
            "clusters": clusters,
        }
