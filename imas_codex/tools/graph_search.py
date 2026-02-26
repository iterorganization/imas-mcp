"""Graph-backed tool implementations for graph-native MCP server.

Replaces DocumentStore, SemanticSearch, and ClusterSearcher with
GraphClient queries against Neo4j vector indexes and Cypher traversals.
"""

import json
import logging
from typing import Any, Literal

from fastmcp import Context

from imas_codex.core.data_model import IdsNode
from imas_codex.graph.client import GraphClient
from imas_codex.models.constants import SearchMode
from imas_codex.models.result_models import (
    CheckPathsResult,
    CheckPathsResultItem,
    FetchPathsResult,
    GetIdentifiersResult,
    GetOverviewResult,
    ListPathsResult,
    ListPathsResultItem,
    NotFoundPathInfo,
    SearchPathsResult,
)
from imas_codex.search.decorators import (
    cache_results,
    handle_errors,
    mcp_tool,
    measure_performance,
)
from imas_codex.search.search_strategy import SearchHit
from imas_codex.tools.utils import normalize_ids_filter, validate_query

logger = logging.getLogger(__name__)


class GraphSearchTool:
    """Graph-backed semantic search for IMAS paths."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "search_imas_paths"

    @cache_results(ttl=300, key_strategy="semantic")
    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="search_suggestions")
    @mcp_tool(
        "Find IMAS IDS entries using semantic and lexical search. "
        "query (required): Natural language description or physics term (e.g., 'electron temperature', 'magnetic field boundary', 'plasma current'). "
        "Common abbreviations supported: Te (electron temp), Ti (ion temp), ne (electron density), Ip (plasma current). "
        "ids_filter: Limit to specific IDS (space/comma-delimited: 'equilibrium magnetics' or 'equilibrium, core_profiles'). "
        "search_mode: 'auto' (default), 'semantic', 'lexical', or 'hybrid'. "
        "response_profile: 'minimal', 'standard' (default), or 'detailed'."
    )
    async def search_imas_paths(
        self,
        query: str,
        ids_filter: str | list[str] | None = None,
        max_results: int = 50,
        search_mode: str | SearchMode = "auto",
        response_profile: str = "standard",
        ctx: Context | None = None,
    ) -> SearchPathsResult:
        """Search IMAS paths using graph vector index."""
        is_valid, error_message = validate_query(query, "search_imas_paths")
        if not is_valid:
            return SearchPathsResult(
                hits=[],
                summary={"error": "Query cannot be empty.", "query": query or ""},
                query=query or "",
                search_mode=SearchMode.AUTO,
                physics_domains=[],
                error="Query cannot be empty.",
            )

        normalized_filter = normalize_ids_filter(ids_filter)
        embedding = self._embed_query(query)

        # Build the Cypher query with optional IDS filter
        filter_clause = ""
        params: dict[str, Any] = {"embedding": embedding, "k": min(max_results, 100)}
        if normalized_filter:
            filter_clause = "WHERE path.ids IN $ids_filter"
            params["ids_filter"] = (
                normalized_filter
                if isinstance(normalized_filter, list)
                else [normalized_filter]
            )

        results = self._gc.query(
            f"""
            CALL db.index.vector.queryNodes('imas_path_embedding', $k, $embedding)
            YIELD node AS path, score
            {filter_clause}
            OPTIONAL MATCH (path)-[:HAS_UNIT]->(u:Unit)
            RETURN path.id AS id, path.name AS name, path.ids AS ids,
                   path.documentation AS documentation, path.data_type AS data_type,
                   path.physics_domain AS physics_domain, u.id AS units,
                   path.node_type AS node_type, score
            ORDER BY score DESC
            LIMIT $k
            """,
            **params,
        )

        mode = (
            SearchMode(search_mode)
            if isinstance(search_mode, str) and search_mode in SearchMode.__members__
            else SearchMode.AUTO
        )

        hits = []
        physics_domains = set()
        for rank, r in enumerate(results or [], start=1):
            hits.append(
                SearchHit(
                    path=r["id"],
                    ids_name=r["ids"] or "",
                    documentation=r["documentation"] or "",
                    data_type=r["data_type"],
                    units=r["units"] or "",
                    physics_domain=r["physics_domain"],
                    node_type=r["node_type"],
                    score=round(r["score"], 4),
                    rank=rank,
                    search_mode=mode,
                )
            )
            if r["physics_domain"]:
                physics_domains.add(r["physics_domain"])

        return SearchPathsResult(
            hits=hits,
            summary={
                "query": query,
                "search_mode": str(mode),
                "hits_returned": len(hits),
                "ids_coverage": sorted({h.ids_name for h in hits if h.ids_name}),
            },
            query=query,
            search_mode=mode,
            physics_domains=sorted(physics_domains),
        )

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text using the Encoder."""
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder()
        return encoder.embed_texts([query])[0].tolist()


class GraphPathTool:
    """Graph-backed path validation and fetching."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "path_tool"

    @measure_performance(include_metrics=True, slow_threshold=0.5)
    @handle_errors(fallback="check_suggestions")
    @mcp_tool(
        "Validate exact IMAS paths and check their existence. "
        "paths (required): One or more exact IMAS paths (e.g., 'equilibrium/time_slice/profiles_1d/psi'). "
        "ids: Optional IDS prefix to prepend (e.g., ids='equilibrium' with paths='time_slice/profiles_1d/psi')."
    )
    async def check_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        ctx: Context | None = None,
    ) -> CheckPathsResult:
        """Validate IMAS paths against graph."""
        path_list = _normalize_paths(paths)
        if ids:
            path_list = [
                f"{ids}/{p}" if "/" not in p or not p.startswith(ids) else p
                for p in path_list
            ]

        results = []
        found = 0
        for path in path_list:
            row = self._gc.query(
                """
                MATCH (p:IMASPath {id: $path})
                OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
                RETURN p.id AS id, p.ids AS ids, p.data_type AS data_type,
                       u.id AS units
                """,
                path=path,
            )
            if row:
                r = row[0]
                results.append(
                    CheckPathsResultItem(
                        path=r["id"],
                        exists=True,
                        ids_name=r["ids"],
                        data_type=r["data_type"],
                        units=r["units"] or "",
                    )
                )
                found += 1
            else:
                # Check for deprecated/renamed path
                renamed = self._gc.query(
                    """
                    MATCH (old:IMASPath {id: $path})-[:RENAMED_TO]->(new:IMASPath)
                    RETURN old.id AS old_path, new.id AS new_path
                    """,
                    path=path,
                )
                if renamed:
                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                            renamed_from=renamed[0]["old_path"],
                            migration=f"Renamed to {renamed[0]['new_path']}",
                            suggestion=renamed[0]["new_path"],
                        )
                    )
                else:
                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                        )
                    )

        return CheckPathsResult(
            results=results,
            summary={
                "total": len(path_list),
                "found": found,
                "not_found": len(path_list) - found,
            },
        )

    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="fetch_suggestions")
    @mcp_tool(
        "Retrieve detailed IMAS path data including documentation, units, coordinates, and cluster membership. "
        "paths (required): One or more exact IMAS paths. "
        "ids: Optional IDS prefix to prepend."
    )
    async def fetch_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        ctx: Context | None = None,
    ) -> FetchPathsResult:
        """Fetch detailed path information from graph."""
        path_list = _normalize_paths(paths)
        if ids:
            path_list = [
                f"{ids}/{p}" if "/" not in p or not p.startswith(ids) else p
                for p in path_list
            ]

        nodes = []
        not_found = []
        deprecated = []

        for path in path_list:
            row = self._gc.query(
                """
                MATCH (p:IMASPath {id: $path})
                OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
                RETURN p.id AS id, p.name AS name, p.ids AS ids,
                       p.documentation AS documentation, p.data_type AS data_type,
                       p.node_type AS node_type, p.physics_domain AS physics_domain,
                       p.ndim AS ndim,
                       u.id AS units,
                       collect(DISTINCT c.label) AS cluster_labels,
                       collect(DISTINCT coord.id) AS coordinates
                """,
                path=path,
            )
            if row and row[0]["id"]:
                r = row[0]
                node = IdsNode(
                    path=r["id"],
                    ids_name=r["ids"],
                    name=r["name"],
                    documentation=r["documentation"] or "",
                    data_type=r["data_type"],
                    units=r["units"] or "",
                    physics_domain=r["physics_domain"],
                    coordinates=r["coordinates"] or [],
                    cluster_labels=[cl for cl in r["cluster_labels"] if cl],
                )
                nodes.append(node)
            else:
                not_found.append(NotFoundPathInfo(path=path, reason="not_found"))

        return FetchPathsResult(
            nodes=nodes,
            not_found_paths=not_found,
            deprecated_paths=deprecated,
            excluded_paths=[],
            summary={
                "total_requested": len(path_list),
                "fetched": len(nodes),
                "not_found": len(not_found),
            },
        )


class GraphListTool:
    """Graph-backed path listing."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "list_imas_paths"

    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="list_suggestions")
    @mcp_tool(
        "List all IMAS data paths within an IDS or subtree. "
        "paths (required): IDS name(s) or path prefix(es), space-separated (e.g., 'equilibrium' or 'equilibrium/time_slice'). "
        "format: Output format - 'yaml' (default), 'flat', 'json', or 'tree'. "
        "leaf_only: If true, return only leaf nodes (data endpoints, not structures). "
        "max_paths: Maximum number of paths to return per query."
    )
    async def list_imas_paths(
        self,
        paths: str,
        format: str = "yaml",
        leaf_only: bool = False,
        include_ids_prefix: bool = True,
        max_paths: int | None = None,
        ctx: Context | None = None,
    ) -> ListPathsResult:
        """List paths from graph."""
        queries = paths.strip().split()
        results = []

        for query in queries:
            # Determine if this is an IDS name or a path prefix
            if "/" in query:
                ids_name = query.split("/")[0]
                prefix = query
            else:
                ids_name = query
                prefix = query

            # Verify IDS exists
            ids_exists = self._gc.query(
                "MATCH (i:IDS {name: $name}) RETURN i.name",
                name=ids_name,
            )
            if not ids_exists:
                results.append(
                    ListPathsResultItem(
                        query=query,
                        error=f"IDS '{ids_name}' not found",
                        path_count=0,
                        paths=[],
                    )
                )
                continue

            # Query paths
            leaf_filter = (
                "AND NOT p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']"
                if leaf_only
                else ""
            )
            limit_clause = f"LIMIT {max_paths}" if max_paths else ""

            path_results = self._gc.query(
                f"""
                MATCH (p:IMASPath)
                WHERE p.id STARTS WITH $prefix
                {leaf_filter}
                RETURN p.id AS id
                ORDER BY p.id
                {limit_clause}
                """,
                prefix=prefix + ("/" if "/" not in prefix else ""),
            )

            # Also include the prefix itself if it's an exact IDS
            if "/" not in prefix:
                path_results = self._gc.query(
                    f"""
                    MATCH (p:IMASPath)
                    WHERE p.ids = $ids_name
                    {leaf_filter}
                    RETURN p.id AS id
                    ORDER BY p.id
                    {limit_clause}
                    """,
                    ids_name=ids_name,
                )

            path_ids = [r["id"] for r in (path_results or [])]
            if not include_ids_prefix:
                path_ids = [
                    p[len(ids_name) + 1 :] if p.startswith(ids_name + "/") else p
                    for p in path_ids
                ]

            truncated = max_paths if max_paths and len(path_ids) >= max_paths else None

            if format == "flat":
                formatted = sorted(path_ids)
            else:
                formatted = sorted(path_ids)

            results.append(
                ListPathsResultItem(
                    query=query,
                    path_count=len(path_ids),
                    truncated_to=truncated,
                    paths=formatted,
                )
            )

        return ListPathsResult(
            format=format,
            results=results,
            summary={
                "queries": len(queries),
                "total_paths": sum(r.path_count for r in results),
            },
        )


class GraphOverviewTool:
    """Graph-backed IDS overview."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "get_imas_overview"

    @cache_results(ttl=3600)
    @handle_errors(fallback="overview_error")
    @mcp_tool(
        "Get an overview of available IMAS Interface Data Structures (IDS). "
        "Returns IDS names, descriptions, path counts, and physics domains. "
        "query: Optional filter to narrow results (e.g., 'magnetics' or 'plasma equilibrium')."
    )
    async def get_imas_overview(
        self,
        query: str | None = None,
        ctx: Context | None = None,
    ) -> GetOverviewResult:
        """Get overview from graph."""
        import importlib.metadata

        # Query IDS nodes from graph
        ids_results = self._gc.query(
            """
            MATCH (i:IDS)
            OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASPath)
            WITH i, count(p) AS path_count
            RETURN i.name AS name, i.description AS description,
                   i.physics_domain AS physics_domain,
                   i.lifecycle_status AS lifecycle_status,
                   path_count
            ORDER BY path_count DESC
            """
        )

        # Query DD version info
        version_results = self._gc.query(
            """
            MATCH (v:DDVersion {is_current: true})
            RETURN v.id AS version
            """
        )
        current_version = (
            version_results[0]["version"] if version_results else "unknown"
        )

        all_ids = []
        ids_statistics = {}
        physics_domains = set()

        for r in ids_results or []:
            ids_name = r["name"]
            # Apply query filter
            if query:
                query_lower = query.lower()
                name_match = query_lower in ids_name.lower()
                desc_match = (r["description"] or "").lower().find(query_lower) >= 0
                domain_match = (r["physics_domain"] or "").lower().find(
                    query_lower
                ) >= 0
                if not (name_match or desc_match or domain_match):
                    continue

            all_ids.append(ids_name)
            ids_statistics[ids_name] = {
                "path_count": r["path_count"],
                "description": r["description"] or "",
                "physics_domain": r["physics_domain"] or "",
            }
            if r["physics_domain"]:
                physics_domains.add(r["physics_domain"])

        # Build tools list
        mcp_tools = [
            "search_imas_paths",
            "check_imas_paths",
            "fetch_imas_paths",
            "list_imas_paths",
            "get_imas_overview",
            "search_imas_clusters",
            "get_imas_identifiers",
            "query_imas_graph",
            "get_dd_graph_schema",
            "get_dd_versions",
        ]

        try:
            version = importlib.metadata.version("imas-codex")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"

        total_paths = sum(s["path_count"] for s in ids_statistics.values())

        return GetOverviewResult(
            content=f"IMAS Data Dictionary v{current_version}: {len(all_ids)} IDS, {total_paths} total paths",
            available_ids=all_ids,
            query=query,
            physics_domains=sorted(physics_domains),
            ids_statistics=ids_statistics,
            mcp_tools=mcp_tools,
            dd_version=current_version,
            mcp_version=version,
            total_leaf_nodes=total_paths,
        )


class GraphClustersTool:
    """Graph-backed cluster search using vector indexes."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "search_imas_clusters"

    @cache_results(ttl=600)
    @handle_errors(fallback="cluster_error")
    @mcp_tool(
        "Search semantic clusters of related IMAS data paths. "
        "query (required): Natural language description (e.g., 'boundary geometry', 'transport coefficients') "
        "or exact IMAS path to find its cluster membership. "
        "scope: Filter by cluster scope - 'global', 'domain', or 'ids'. "
        "ids_filter: Limit to clusters containing paths from specific IDS."
    )
    async def search_imas_clusters(
        self,
        query: str,
        scope: Literal["global", "domain", "ids"] | None = None,
        ids_filter: str | list[str] | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search clusters using graph vector indexes."""
        is_valid, _ = validate_query(query, "search_imas_clusters")
        if not is_valid:
            return {
                "error": "Query cannot be empty.",
                "clusters_found": 0,
                "clusters": [],
            }

        normalized_filter = normalize_ids_filter(ids_filter)

        # Detect query type: path lookup vs semantic search
        if "/" in query and " " not in query:
            return self._search_by_path(query, scope)

        return self._search_by_text(query, scope, normalized_filter)

    def _search_by_path(self, path: str, scope: str | None) -> dict[str, Any]:
        """Find clusters containing a specific path."""
        scope_filter = "AND c.scope = $scope" if scope else ""
        params: dict[str, Any] = {"path": path}
        if scope:
            params["scope"] = scope

        results = self._gc.query(
            f"""
            MATCH (p:IMASPath {{id: $path}})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            {scope_filter}
            OPTIONAL MATCH (member:IMASPath)-[:IN_CLUSTER]->(c)
            WITH c, collect(DISTINCT member.id) AS paths
            RETURN c.id AS id, c.label AS label, c.description AS description,
                   c.scope AS scope, c.cross_ids AS cross_ids,
                   c.ids_names AS ids_names, c.similarity_score AS similarity,
                   paths
            ORDER BY c.scope
            """,
            **params,
        )

        clusters = self._format_clusters(results or [])
        return {
            "query": path,
            "query_type": "path",
            "clusters_found": len(clusters),
            "clusters": clusters,
        }

    def _search_by_text(
        self,
        query: str,
        scope: str | None,
        ids_filter: str | list[str] | None,
    ) -> dict[str, Any]:
        """Semantic search over cluster embeddings."""
        embedding = self._embed_query(query)

        scope_filter = "AND cluster.scope = $scope" if scope else ""
        ids_filter_clause = ""
        params: dict[str, Any] = {"embedding": embedding, "k": 10}
        if scope:
            params["scope"] = scope
        if ids_filter:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            ids_filter_clause = (
                "AND any(ids_name IN cluster.ids_names WHERE ids_name IN $ids_filter)"
            )
            params["ids_filter"] = filter_list

        results = self._gc.query(
            f"""
            CALL db.index.vector.queryNodes(
                'cluster_description_embedding', $k, $embedding
            )
            YIELD node AS cluster, score
            WHERE score > 0.3
            {scope_filter}
            {ids_filter_clause}
            OPTIONAL MATCH (member:IMASPath)-[:IN_CLUSTER]->(cluster)
            WITH cluster, score, collect(DISTINCT member.id) AS paths
            RETURN cluster.id AS id, cluster.label AS label,
                   cluster.description AS description,
                   cluster.scope AS scope, cluster.cross_ids AS cross_ids,
                   cluster.ids_names AS ids_names,
                   cluster.similarity_score AS similarity,
                   score AS relevance_score, paths
            ORDER BY relevance_score DESC
            """,
            **params,
        )

        clusters = self._format_clusters(results or [], include_score=True)
        return {
            "query": query,
            "query_type": "semantic",
            "clusters_found": len(clusters),
            "clusters": clusters,
            "ids_filter": ids_filter,
            "scope_filter": scope,
        }

    def _format_clusters(
        self, results: list[dict], include_score: bool = False
    ) -> list[dict]:
        """Format cluster results with path truncation."""
        clusters = []
        for r in results:
            paths = r.get("paths", []) or []
            cluster = {
                "id": r["id"],
                "label": r.get("label", ""),
                "description": r.get("description", ""),
                "type": "cross_ids" if r.get("cross_ids") else "intra_ids",
                "scope": r.get("scope", "global"),
                "ids": r.get("ids_names", []) or [],
                "similarity": round(r.get("similarity", 0) or 0, 4),
                "paths": paths[:20],
            }
            if len(paths) > 20:
                cluster["total_paths"] = len(paths)
            if include_score and "relevance_score" in r:
                cluster["relevance_score"] = round(r["relevance_score"], 4)
            clusters.append(cluster)
        return clusters

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text using the Encoder."""
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder()
        return encoder.embed_texts([query])[0].tolist()


class GraphIdentifiersTool:
    """Graph-backed identifier schemas."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "get_imas_identifiers"

    @cache_results(ttl=3600)
    @handle_errors(fallback="identifiers_error")
    @mcp_tool(
        "Get IMAS identifier/enumeration schemas. "
        "These define valid values for typed fields like coordinate systems, "
        "probe types, and grid types. "
        "query: Optional filter (e.g., 'coordinate' or 'magnetics')."
    )
    async def get_imas_identifiers(
        self,
        query: str | None = None,
        ctx: Context | None = None,
    ) -> GetIdentifiersResult:
        """Get identifier schemas from graph."""
        results = self._gc.query(
            """
            MATCH (s:IdentifierSchema)
            RETURN s.name AS name, s.description AS description,
                   s.option_count AS option_count, s.options AS options,
                   s.field_count AS field_count, s.source AS source
            ORDER BY s.name
            """
        )

        schemas = []
        for r in results or []:
            # Apply query filter
            if query:
                query_lower = query.lower()
                name_match = query_lower in (r["name"] or "").lower()
                desc_match = query_lower in (r["description"] or "").lower()
                opts_match = query_lower in (r["options"] or "").lower()
                if not (name_match or desc_match or opts_match):
                    continue

            options = []
            if r["options"]:
                try:
                    options = json.loads(r["options"])
                except (json.JSONDecodeError, TypeError):
                    pass

            option_count = r["option_count"] or len(options)
            schemas.append(
                {
                    "path": r["name"],
                    "schema_path": r["source"] or "",
                    "option_count": option_count,
                    "branching_significance": _classify_significance(option_count),
                    "options": options,
                    "description": r["description"] or "",
                }
            )

        # Compute analytics
        total_schemas = len(schemas)
        total_options = sum(s["option_count"] for s in schemas)

        return GetIdentifiersResult(
            schemas=schemas,
            paths=[],
            analytics={
                "total_schemas": total_schemas,
                "total_paths": 0,
                "enumeration_space": total_options,
                "query_context": query,
            },
        )


def _normalize_paths(paths: str | list[str]) -> list[str]:
    """Normalize paths input to a flat list."""
    if isinstance(paths, str):
        return [p.strip() for p in paths.replace(",", " ").split() if p.strip()]
    return list(paths)


def _classify_significance(option_count: int) -> str:
    """Classify branching significance based on option count."""
    if option_count >= 20:
        return "CRITICAL"
    elif option_count >= 10:
        return "HIGH"
    elif option_count >= 5:
        return "MODERATE"
    return "MINIMAL"
