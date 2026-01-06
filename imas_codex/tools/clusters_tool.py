"""
Clusters tool for searching semantically related IMAS paths.

Provides search over hierarchical clusters of related paths using:
- Path lookup (exact match)
- Natural language queries (semantic search on labels/descriptions)
- Scope filtering (global, domain, ids)
- Composite relevance ranking
"""

import logging
from typing import Any, Literal

from fastmcp import Context

from imas_codex.clusters.models import ClusterScope
from imas_codex.clusters.search import ClusterSearcher, ClusterSearchResult
from imas_codex.core.clusters import Clusters
from imas_codex.embeddings.config import EncoderConfig
from imas_codex.embeddings.encoder import Encoder
from imas_codex.models.error_models import ToolError
from imas_codex.search.decorators import cache_results, handle_errors, mcp_tool

from .base import BaseTool
from .utils import normalize_ids_filter, validate_query

logger = logging.getLogger(__name__)


class ClustersTool(BaseTool):
    """
    Search tool for discovering related IMAS data paths via semantic clusters.

    Clusters are organized hierarchically:
    - Global: Discovered from full embedding space (all IDS)
    - Domain: Per physics domain (e.g., transport, equilibrium)
    - IDS: Per individual IDS (e.g., core_profiles, magnetics)

    Clusters include enrichment metadata:
    - physics_concepts: Normalized physics quantities
    - data_type: Data structure classification
    - tags: Classification tags
    - mapping_relevance: Usefulness for data mapping (high/medium/low)

    Results are ranked by composite relevance score blending semantic
    similarity with mapping relevance.
    """

    def __init__(self, *args, **kwargs):
        """Initialize clusters tool with lazy loading."""
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

        # Lazy-loaded state - defer heavy initialization until first use
        self._clusters: Clusters | None = None
        self._searcher: ClusterSearcher | None = None
        self._encoder: Encoder | None = None
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of clusters - called on first use."""
        if self._initialized:
            return
        self._initialized = True
        try:
            self._clusters = Clusters(encoder_config=self._encoder_config)

            # Check if clusters are available (will auto-build if needed)
            if self._clusters.is_available():
                # Use get_cluster_searcher() which properly loads embeddings
                self._searcher = self._clusters.get_cluster_searcher()
                logger.info(
                    f"Loaded {len(self._clusters.get_clusters())} clusters with embeddings"
                )
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
        "query (required): Natural language (e.g., 'electron density measurements') or IMAS path. "
        "scope: Filter by hierarchy level - 'global', 'domain', or 'ids'. "
        "ids_filter: Limit to specific IDS (space/comma-delimited). "
        "Returns: Clusters ranked by relevance with enrichment metadata (physics_concepts, data_type, tags, mapping_relevance)."
    )
    async def search_imas_clusters(
        self,
        query: str,
        scope: Literal["global", "domain", "ids"] | None = None,
        ids_filter: str | list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any] | ToolError:
        """
        Search for semantically related IMAS path clusters.

        Automatically detects query type:
        - Paths (contain '/') → exact cluster lookup
        - Natural language → semantic search on cluster labels/descriptions

        Results are ranked by composite relevance score blending semantic
        similarity with mapping relevance, cluster type, and scope.

        Args:
            query: Path or natural language query (required)
            scope: Filter by hierarchy level:
                   - "global": Clusters from full embedding space
                   - "domain": Clusters within physics domains
                   - "ids": Clusters within individual IDS
                   - None: Search all levels (default)
            ids_filter: Limit results to specific IDS. Accepts:
                       - Space-delimited string: "equilibrium transport core_profiles"
                       - Comma-delimited string: "equilibrium, transport, core_profiles"
                       - List of IDS names: ["equilibrium", "transport"]
            ctx: MCP context

        Returns:
            Dict with matching clusters, enrichment metadata, and relevance scores

        Examples:
            search_imas_clusters(query="core_profiles/profiles_1d/electrons/density")
            search_imas_clusters(query="electron temperature", scope="global")
            search_imas_clusters(query="magnetic field", scope="domain")
            search_imas_clusters(query="transport", ids_filter="core_profiles")
        """
        # Validate query is not empty
        is_valid, error_message = validate_query(query, "search_imas_clusters")
        if not is_valid:
            return ToolError(
                error="Query cannot be empty",
                suggestions=[
                    "Try 'electron temperature' for temperature-related clusters",
                    "Try 'magnetic field measurements' for B-field data",
                    "Try a path like 'equilibrium/time_slice/profiles_2d' for related paths",
                    "Use get_imas_overview() to explore available IDS structures",
                ],
                context={"tool": "search_imas_clusters"},
            )

        # Lazy initialization - defer heavy loading until first use
        self._ensure_initialized()

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
            # Normalize ids_filter using utility function
            normalized_filter = normalize_ids_filter(ids_filter)
            ids_set: set[str] | None = (
                set(normalized_filter) if normalized_filter else None
            )

            # Cast scope to ClusterScope type if provided
            scope_filter: ClusterScope | None = scope  # type: ignore[assignment]

            # Detect query type and search
            is_path = "/" in query and " " not in query

            if is_path:
                # Path lookup
                results = self._searcher.search_by_path(query)
                # Apply scope filter for path lookups
                if scope_filter:
                    results = [r for r in results if r.scope == scope_filter]
            else:
                # Natural language search with scope filtering
                encoder = self._get_encoder()
                results = self._searcher.search_by_text(
                    query=query,
                    encoder=encoder,
                    top_k=10,
                    similarity_threshold=0.3,
                    scope=scope_filter,
                )

            # Apply IDS filter if specified
            if ids_set and results:
                results = [
                    r
                    for r in results
                    if any(ids_name in ids_set for ids_name in r.ids_names)
                ]

            if not results:
                error_context: dict[str, Any] = {"query": query}
                if ids_set:
                    error_context["ids_filter"] = list(ids_set)
                if scope:
                    error_context["scope"] = scope
                return ToolError(
                    error=f"No clusters found for query: {query}",
                    suggestions=[
                        "Try a broader search term",
                        "Try without scope filter to search all levels",
                        "Use search_imas_paths() for direct path search",
                        "Check available IDS with get_imas_overview()",
                    ],
                    context=error_context,
                )

            # Format response with enrichment fields
            return self._format_results(query, results, is_path, ids_set, scope)

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
        ids_filter: set[str] | None = None,
        scope_filter: str | None = None,
    ) -> dict[str, Any]:
        """Format search results with enrichment fields."""
        clusters = []

        for result in results:
            # Filter paths to only include those from requested IDS
            paths = result.paths
            if ids_filter:
                paths = [p for p in paths if p.split("/")[0] in ids_filter]

            cluster_data: dict[str, Any] = {
                "id": result.cluster_id,
                "label": result.label or f"Cluster {result.cluster_id[:8]}",
                "description": result.description,
                "type": "cross_ids" if result.is_cross_ids else "intra_ids",
                "scope": result.scope,
                "ids": result.ids_names,
                "similarity": round(result.similarity_score, 3),
                "relevance_score": round(result.relevance_score or 0, 3),
                "paths": paths[:20],  # Limit paths in response
            }

            # Add scope_detail if present
            if result.scope_detail:
                cluster_data["scope_detail"] = result.scope_detail

            # Add enrichment fields if present
            if result.physics_concepts:
                cluster_data["physics_concepts"] = result.physics_concepts
            if result.data_type:
                cluster_data["data_type"] = result.data_type
            if result.tags:
                cluster_data["tags"] = result.tags
            if result.mapping_relevance:
                cluster_data["mapping_relevance"] = result.mapping_relevance

            if len(paths) > 20:
                cluster_data["total_paths"] = len(paths)
                cluster_data["paths_truncated"] = True

            clusters.append(cluster_data)

        response: dict[str, Any] = {
            "query": query,
            "query_type": "path" if is_path else "semantic",
            "clusters_found": len(clusters),
            "clusters": clusters,
        }

        if ids_filter:
            response["ids_filter"] = sorted(ids_filter)

        if scope_filter:
            response["scope_filter"] = scope_filter

        return response
