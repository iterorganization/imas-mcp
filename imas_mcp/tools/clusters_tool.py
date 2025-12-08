"""
Clusters tool for searching semantically related IMAS paths.

Provides search over clusters of related paths using:
- Path lookup (exact match)
- Natural language queries (semantic search on labels/descriptions)
- IDS filtering
"""

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import Context

from imas_mcp import dd_version
from imas_mcp.embeddings.config import EncoderConfig
from imas_mcp.embeddings.encoder import Encoder
from imas_mcp.models.error_models import ToolError
from imas_mcp.relationships.search import ClusterSearcher, ClusterSearchResult
from imas_mcp.resource_path_accessor import ResourcePathAccessor
from imas_mcp.search.decorators import cache_results, handle_errors, mcp_tool

from .base import BaseTool

logger = logging.getLogger(__name__)


class ClustersTool(BaseTool):
    """
    Search tool for discovering related IMAS data paths via semantic clusters.

    Clusters group paths with similar physics meaning, enabling discovery of:
    - Cross-IDS relationships (same concept across different IDS)
    - Intra-IDS relationships (related paths within an IDS)
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

        self._clusters_data: dict[str, Any] | None = None
        self._searcher: ClusterSearcher | None = None
        self._encoder: Encoder | None = None

        self._load_clusters()

    def _get_clusters_path(self) -> Path:
        """Get path to clusters.json file."""
        path_accessor = ResourcePathAccessor(dd_version=dd_version)
        return path_accessor.schemas_dir / "clusters.json"

    def _load_clusters(self) -> None:
        """Load clusters data from clusters.json."""
        clusters_path = self._get_clusters_path()

        if not clusters_path.exists():
            # Fall back to relationships.json for backwards compatibility
            relationships_path = clusters_path.parent / "relationships.json"
            if relationships_path.exists():
                logger.info(f"clusters.json not found, using {relationships_path}")
                clusters_path = relationships_path
            else:
                logger.warning("No clusters or relationships file found")
                self._clusters_data = {"clusters": []}
                return

        try:
            with open(clusters_path, encoding="utf-8") as f:
                self._clusters_data = json.load(f)

            clusters = self._clusters_data.get("clusters", [])
            self._searcher = ClusterSearcher(clusters=clusters)

            logger.info(f"Loaded {len(clusters)} clusters from {clusters_path.name}")
        except Exception as e:
            logger.error(f"Failed to load clusters: {e}")
            self._clusters_data = {"clusters": []}

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
                    "Ensure clusters.json or relationships.json exists",
                    "Try rebuilding with: build-relationships",
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
