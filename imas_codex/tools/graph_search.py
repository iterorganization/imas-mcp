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


def _dd_version_clause(
    alias: str = "p",
    dd_version: int | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Return a Cypher WHERE fragment for DD major version validity filtering.

    When dd_version is None, returns empty string (no filter).

    A path is **valid in DD major N** if:
    - introduced in any version with major ≤ N
    - NOT deprecated in any version with major ≤ N

    The DDVersion.id is a semver string like "3.42.2" or "4.0.0".
    Major version is extracted via ``toInteger(split(id, '.')[0])``.

    If *params* dict is provided, adds ``dd_version`` to it.
    """
    if dd_version is None:
        return ""
    if params is not None:
        params["dd_version"] = dd_version
    return (
        f"AND EXISTS {{ "
        f"  MATCH ({alias})-[:INTRODUCED_IN]->(iv:DDVersion) "
        f"  WHERE toInteger(split(iv.id, '.')[0]) <= $dd_version "
        f"}} "
        f"AND NOT EXISTS {{ "
        f"  MATCH ({alias})-[:DEPRECATED_IN]->(dv:DDVersion) "
        f"  WHERE toInteger(split(dv.id, '.')[0]) <= $dd_version "
        f"}}"
    )


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
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions. "
        "facility: Optional facility for cross-references (e.g., 'tcv'). Returns signal, wiki, and code references. "
        "include_version_context: Include DD version change history for each result path. "
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
        dd_version: int | None = None,
        facility: str | None = None,
        include_version_context: bool = False,
        ctx: Context | None = None,
    ) -> SearchPathsResult:
        """Search IMAS paths using hybrid vector + text search."""
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

        # --- Vector search ---
        filter_clause = ""
        params: dict[str, Any] = {
            "embedding": embedding,
            "k": min(max_results * 5, 500),
            "vector_limit": min(max_results * 3, 150),
        }
        if normalized_filter:
            filter_clause = "AND path.ids IN $ids_filter"
            params["ids_filter"] = (
                normalized_filter
                if isinstance(normalized_filter, list)
                else [normalized_filter]
            )

        dd_clause = _dd_version_clause("path", dd_version, params)

        vector_results = self._gc.query(
            f"""
            CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
            YIELD node AS path, score
            WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
              AND path.node_category = 'data'
            {filter_clause}
            {dd_clause}
            RETURN path.id AS id, score
            ORDER BY score DESC
            LIMIT $vector_limit
            """,
            **params,
        )

        scores: dict[str, float] = {}
        for r in vector_results or []:
            pid = r["id"]
            scores[pid] = round(r["score"], 4)

        # --- Text search ---
        text_results = _text_search_imas_paths(
            self._gc,
            query,
            min(max_results * 3, 150),
            normalized_filter,
            dd_version=dd_version,
        )
        for r in text_results:
            pid = r["id"]
            text_score = round(r["score"], 4)
            if pid in scores:
                scores[pid] = round(max(scores[pid], text_score) + 0.05, 4)
            else:
                scores[pid] = text_score

        # --- Path segment boost ---
        # Boost paths whose segments match query words for better relevance
        query_words = [w.lower() for w in query.split() if len(w) > 2]
        if query_words:
            for pid in scores:
                segments = pid.lower().split("/")
                match_count = sum(
                    1 for w in query_words if any(w in seg for seg in segments)
                )
                if match_count > 0:
                    scores[pid] = round(scores[pid] + 0.03 * match_count, 4)

        # Rank and limit
        sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)[
            :max_results
        ]

        mode = (
            SearchMode(search_mode)
            if isinstance(search_mode, str) and search_mode in SearchMode.__members__
            else SearchMode.AUTO
        )

        if not sorted_ids:
            return SearchPathsResult(
                hits=[],
                summary={
                    "query": query,
                    "search_mode": str(mode),
                    "hits_returned": 0,
                    "ids_coverage": [],
                },
                query=query,
                search_mode=mode,
                physics_domains=[],
            )

        # --- Enrich with full metadata ---
        enriched = self._gc.query(
            """
            UNWIND $path_ids AS pid
            MATCH (path:IMASNode {id: pid})
            OPTIONAL MATCH (path)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (path)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
            OPTIONAL MATCH (path)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
            OPTIONAL MATCH (path)-[:INTRODUCED_IN]->(intro:DDVersion)
            RETURN path.id AS id, path.name AS name, path.ids AS ids,
                   path.documentation AS documentation, path.data_type AS data_type,
                   path.physics_domain AS physics_domain, u.id AS units,
                   path.node_type AS node_type,
                   path.lifecycle_status AS lifecycle_status,
                   path.lifecycle_version AS lifecycle_version,
                   path.timebasepath AS timebase,
                   path.path_doc AS structure_reference,
                   path.coordinate1_same_as AS coordinate1,
                   path.coordinate2_same_as AS coordinate2,
                   path.cocos_label_transformation AS cocos_label,
                   path.cocos_transformation_expression AS cocos_expression,
                   path.description AS description,
                   path.keywords AS keywords,
                   path.enrichment_source AS enrichment_source,
                   collect(DISTINCT coord.id) AS coordinates,
                   ident IS NOT NULL AS has_identifier_schema,
                   intro.id AS introduced_after_version
            """,
            path_ids=sorted_ids,
        )

        # Index by path ID for score lookup
        enriched_by_id = {r["id"]: r for r in enriched or []}

        # --- Optional facility cross-references ---
        facility_xrefs: dict[str, dict[str, Any]] = {}
        if facility and sorted_ids:
            facility_xrefs = _get_facility_crossrefs(self._gc, sorted_ids, facility)

        # --- Optional version context ---
        version_ctx: dict[str, dict[str, Any]] = {}
        if include_version_context and sorted_ids:
            version_ctx = _get_version_context(self._gc, sorted_ids)

        hits = []
        physics_domains = set()
        for rank, pid in enumerate(sorted_ids, start=1):
            r = enriched_by_id.get(pid)
            if not r:
                continue
            xref = facility_xrefs.get(pid)
            vctx = version_ctx.get(pid)
            hits.append(
                SearchHit(
                    path=r["id"],
                    ids_name=r["ids"] or "",
                    documentation=r["documentation"] or "",
                    data_type=r["data_type"],
                    units=r["units"] or "",
                    physics_domain=r["physics_domain"],
                    node_type=r["node_type"],
                    coordinates=r["coordinates"] or [],
                    lifecycle_status=r["lifecycle_status"],
                    lifecycle_version=r["lifecycle_version"],
                    timebase=r["timebase"],
                    structure_reference=r["structure_reference"],
                    coordinate1=r["coordinate1"],
                    coordinate2=r["coordinate2"],
                    cocos_label_transformation=r["cocos_label"],
                    cocos_transformation_expression=r["cocos_expression"],
                    description=r.get("description"),
                    keywords=r.get("keywords"),
                    enrichment_source=r.get("enrichment_source"),
                    has_identifier_schema=bool(r["has_identifier_schema"]),
                    introduced_after_version=r["introduced_after_version"],
                    score=scores.get(pid, 0.0),
                    rank=rank,
                    search_mode=mode,
                    facility_xrefs=xref,
                    version_context=vctx,
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
        "ids: Optional IDS prefix to prepend (e.g., ids='equilibrium' with paths='time_slice/profiles_1d/psi'). "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None checks all versions."
    )
    async def check_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> CheckPathsResult:
        """Validate IMAS paths against graph."""
        path_list = _normalize_paths(paths)
        if ids:
            path_list = [
                f"{ids}/{p}" if "/" not in p or not p.startswith(ids) else p
                for p in path_list
            ]

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        results = []
        found = 0
        for path in path_list:
            row = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})
                WHERE true {dd_clause}
                OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
                RETURN p.id AS id, p.ids AS ids, p.data_type AS data_type,
                       u.id AS units
                """,
                path=path,
                **dd_params,
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
                    MATCH (old:IMASNode {id: $path})-[:RENAMED_TO]->(new:IMASNode)
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
        "ids: Optional IDS prefix to prepend. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions. "
        "include_version_history: Include notable version changes (sign_convention, units, etc.)."
    )
    async def fetch_imas_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        dd_version: int | None = None,
        include_version_history: bool = False,
        ctx: Context | None = None,
    ) -> FetchPathsResult:
        """Fetch detailed path information from graph."""
        path_list = _normalize_paths(paths)
        if ids:
            path_list = [
                f"{ids}/{p}" if "/" not in p or not p.startswith(ids) else p
                for p in path_list
            ]

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        nodes = []
        not_found = []
        deprecated = []

        for path in path_list:
            version_clause = ""
            if include_version_history:
                version_clause = """
                OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
                WHERE change.semantic_change_type IN
                      ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
                WITH p, u, cluster_labels, coordinates, ident, iv,
                     collect(DISTINCT {version: change.version,
                                       type: change.semantic_change_type,
                                       summary: change.summary}) AS version_changes
                """
            else:
                version_clause = """
                WITH p, u, cluster_labels, coordinates, ident, iv,
                     [] AS version_changes
                """

            row = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})
                WHERE true {dd_clause}
                OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (p)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
                OPTIONAL MATCH (p)-[:HAS_IDENTIFIER_SCHEMA]->(ident:IdentifierSchema)
                OPTIONAL MATCH (p)-[:INTRODUCED_IN]->(iv:DDVersion)
                WITH p, u,
                     collect(DISTINCT c.label) AS cluster_labels,
                     collect(DISTINCT coord.id) AS coordinates,
                     ident, iv
                {version_clause}
                RETURN p.id AS id, p.name AS name, p.ids AS ids,
                       p.documentation AS documentation, p.data_type AS data_type,
                       p.node_type AS node_type, p.physics_domain AS physics_domain,
                       p.ndim AS ndim,
                       p.path_doc AS structure_path,
                       p.lifecycle_status AS lifecycle_status,
                       p.lifecycle_version AS lifecycle_version,
                       p.cocos_label_transformation AS cocos_label,
                       p.cocos_transformation_expression AS cocos_expression,
                       p.coordinate1_same_as AS coordinate1,
                       p.coordinate2_same_as AS coordinate2,
                       p.timebasepath AS timebase,
                       p.description AS description,
                       p.keywords AS keywords,
                       p.enrichment_source AS enrichment_source,
                       u.id AS units,
                       cluster_labels,
                       coordinates,
                       ident.name AS identifier_schema_name,
                       ident.documentation AS identifier_schema_documentation,
                       ident.options AS identifier_schema_options,
                       iv.id AS introduced_after_version,
                       version_changes
                """,
                path=path,
                **dd_params,
            )
            if row and row[0]["id"]:
                r = row[0]
                # Build identifier schema if present
                ident_schema = None
                if r.get("identifier_schema_name"):
                    options = []
                    if r["identifier_schema_options"]:
                        try:
                            raw_opts = json.loads(r["identifier_schema_options"])
                            from imas_codex.core.data_model import IdentifierOption

                            for opt in raw_opts:
                                if isinstance(opt, dict):
                                    options.append(
                                        IdentifierOption(
                                            name=opt.get("name", ""),
                                            index=opt.get("index", 0),
                                            description=opt.get("description", ""),
                                        )
                                    )
                        except (json.JSONDecodeError, TypeError):
                            pass
                    from imas_codex.core.data_model import IdentifierSchema

                    ident_schema = IdentifierSchema(
                        schema_path=r["identifier_schema_name"],
                        documentation=r.get("identifier_schema_documentation"),
                        options=options,
                    )

                # Build version changes list
                v_changes = None
                if include_version_history:
                    raw_changes = r.get("version_changes") or []
                    # Filter out null entries from OPTIONAL MATCH
                    v_changes = [
                        c for c in raw_changes if c.get("version") is not None
                    ] or None

                node = IdsNode(
                    path=r["id"],
                    ids_name=r["ids"],
                    name=r["name"],
                    documentation=r["documentation"] or "",
                    data_type=r["data_type"],
                    units=r["units"] or "",
                    physics_domain=r["physics_domain"],
                    coordinates=r["coordinates"] or [],
                    coordinate1=r.get("coordinate1"),
                    coordinate2=r.get("coordinate2"),
                    timebase=r.get("timebase"),
                    cluster_labels=[cl for cl in r["cluster_labels"] if cl],
                    identifier_schema=ident_schema,
                    version_changes=v_changes,
                    cocos_label_transformation=r.get("cocos_label"),
                    cocos_transformation_expression=r.get("cocos_expression"),
                    description=r.get("description"),
                    keywords=r.get("keywords"),
                    enrichment_source=r.get("enrichment_source"),
                    introduced_after_version=r.get("introduced_after_version"),
                    lifecycle_status=r.get("lifecycle_status"),
                    lifecycle_version=r.get("lifecycle_version"),
                    structure_reference=r.get("structure_path"),
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

    @measure_performance(include_metrics=True, slow_threshold=0.5)
    @handle_errors(fallback="fetch_suggestions")
    @mcp_tool(
        "Fetch error fields for a data path via HAS_ERROR relationships. "
        "path (required): Exact IMAS data path. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions."
    )
    async def fetch_error_fields(
        self,
        path: str,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Fetch HAS_ERROR-linked fields for an IMAS data path."""
        clean_path = path.strip()
        dd_params: dict[str, Any] = {"path": clean_path}
        dd_clause = _dd_version_clause("d", dd_version, dd_params)

        rows = self._gc.query(
            f"""
            MATCH (d:IMASNode {{id: $path}})
            WHERE true {dd_clause}
            OPTIONAL MATCH (d)-[rel:HAS_ERROR]->(e:IMASNode)
            WITH d, e, rel
            WHERE e IS NOT NULL
            RETURN d.id AS path,
                   collect({{
                       path: e.id,
                       name: e.name,
                       error_type: rel.error_type,
                       documentation: e.documentation,
                       data_type: e.data_type
                   }}) AS error_fields
            """,
            **dd_params,
        )

        if not rows:
            return {
                "path": clean_path,
                "count": 0,
                "error_fields": [],
                "not_found": True,
            }

        error_fields = [field for field in rows[0].get("error_fields", []) if field]
        return {
            "path": clean_path,
            "count": len(error_fields),
            "error_fields": error_fields,
            "not_found": False,
        }


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
        "max_paths: Maximum number of paths to return per query. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions. "
        "response_profile: 'minimal' (default, path IDs only) or 'standard' (includes name, data_type, node_type, documentation, units)."
    )
    async def list_imas_paths(
        self,
        paths: str,
        format: str = "yaml",
        leaf_only: bool = False,
        include_ids_prefix: bool = True,
        max_paths: int | None = None,
        dd_version: int | None = None,
        response_profile: str = "minimal",
        ctx: Context | None = None,
    ) -> ListPathsResult:
        """List paths from graph."""
        queries = paths.strip().split()
        results = []

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

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
                "MATCH (i:IDS {id: $name}) RETURN i.name",
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

            include_metadata = response_profile != "minimal"

            if include_metadata:
                return_clause = (
                    "OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)\n"
                    "                RETURN p.id AS id, p.name AS name, "
                    "p.data_type AS data_type,\n"
                    "                       p.node_type AS node_type, "
                    "p.documentation AS documentation,\n"
                    "                       u.symbol AS units"
                )
            else:
                return_clause = "RETURN p.id AS id"

            path_results = self._gc.query(
                f"""
                MATCH (p:IMASNode)
                WHERE p.id STARTS WITH $prefix
                AND p.node_category = 'data'
                {leaf_filter}
                {dd_clause}
                {return_clause}
                ORDER BY p.id
                {limit_clause}
                """,
                prefix=prefix + ("/" if "/" not in prefix else ""),
                **dd_params,
            )

            # Also include the prefix itself if it's an exact IDS
            if "/" not in prefix:
                path_results = self._gc.query(
                    f"""
                    MATCH (p:IMASNode)
                    WHERE p.ids = $ids_name
                    AND p.node_category = 'data'
                    {leaf_filter}
                    {dd_clause}
                    {return_clause}
                    ORDER BY p.id
                    {limit_clause}
                    """,
                    ids_name=ids_name,
                    **dd_params,
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

            # Build path_details when metadata is available
            details: list[dict[str, Any]] | None = None
            if include_metadata and path_results:
                details = [
                    {
                        "id": r["id"],
                        "name": r.get("name"),
                        "data_type": r.get("data_type"),
                        "node_type": r.get("node_type"),
                        "documentation": r.get("documentation"),
                        "units": r.get("units"),
                    }
                    for r in path_results
                ]

            results.append(
                ListPathsResultItem(
                    query=query,
                    path_count=len(path_ids),
                    truncated_to=truncated,
                    paths=formatted,
                    path_details=details,
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
        "query: Optional filter to narrow results (e.g., 'magnetics' or 'plasma equilibrium'). "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions."
    )
    async def get_imas_overview(
        self,
        query: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> GetOverviewResult:
        """Get overview from graph."""
        import importlib.metadata

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        # Query IDS nodes from graph
        ids_results = self._gc.query(
            f"""
            MATCH (i:IDS)
            OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASNode)
            WHERE p.node_category = 'data' {dd_clause}
            WITH i, count(p) AS path_count
            RETURN i.name AS name,
                   COALESCE(i.description, i.documentation) AS description,
                   i.physics_domain AS physics_domain,
                   i.lifecycle_status AS lifecycle_status,
                   path_count
            ORDER BY path_count DESC
            """,
            **dd_params,
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

        # If query provided, try semantic search via ids_embedding vector index
        semantic_scores: dict[str, float] = {}
        if query:
            try:
                from imas_codex.embeddings.config import EncoderConfig
                from imas_codex.embeddings.encoder import Encoder
                from imas_codex.settings import get_embedding_model

                encoder = Encoder(
                    config=EncoderConfig(
                        model_name=get_embedding_model(),
                        normalize_embeddings=True,
                    )
                )
                query_vec = encoder.embed_texts([query])[0].tolist()
                sem_results = self._gc.query(
                    """
                    CALL db.index.vector.queryNodes(
                        'ids_embedding', $k, $query_vec
                    ) YIELD node, score
                    RETURN node.id AS name, score
                    """,
                    k=20,
                    query_vec=query_vec,
                )
                for r in sem_results or []:
                    semantic_scores[r["name"]] = r["score"]
            except Exception:
                pass  # Vector index may not exist yet

        for r in ids_results or []:
            ids_name = r["name"]
            # Apply query filter: text match OR semantic match
            if query:
                query_lower = query.lower()
                name_match = query_lower in ids_name.lower()
                desc_match = (r["description"] or "").lower().find(query_lower) >= 0
                domain_match = (r["physics_domain"] or "").lower().find(
                    query_lower
                ) >= 0
                semantic_match = ids_name in semantic_scores
                if not (name_match or desc_match or domain_match or semantic_match):
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
        "query: Natural language description (e.g., 'boundary geometry', 'transport coefficients') "
        "or exact IMAS path to find its cluster membership. Optional when ids_filter is provided. "
        "scope: Filter by cluster scope - 'global', 'domain', or 'ids'. "
        "ids_filter: Limit to clusters containing paths from specific IDS. "
        "section_only: If true, only return clusters containing structural sections. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions."
    )
    async def search_imas_clusters(
        self,
        query: str | None = None,
        scope: Literal["global", "domain", "ids"] | None = None,
        ids_filter: str | list[str] | None = None,
        section_only: bool = False,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search clusters using graph vector indexes."""
        normalized_filter = normalize_ids_filter(ids_filter)

        # IDS listing mode: no query, ids_filter provided
        if not query and normalized_filter:
            return self._list_by_ids(
                normalized_filter,
                scope,
                section_only=section_only,
                dd_version=dd_version,
            )

        if not query:
            return {
                "error": "Either query or ids_filter is required.",
                "clusters_found": 0,
                "clusters": [],
            }

        # Detect query type: path lookup vs semantic search
        if "/" in query and " " not in query:
            return self._search_by_path(query, scope, dd_version=dd_version)

        return self._search_by_text(
            query, scope, normalized_filter, dd_version=dd_version
        )

    def _list_by_ids(
        self,
        ids_filter: str | list[str],
        scope: str | None,
        *,
        section_only: bool = False,
        dd_version: int | None = None,
    ) -> dict[str, Any]:
        """List all clusters for specific IDS without a search query."""
        filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
        scope_filter = "AND c.scope = $scope" if scope else ""
        section_filter = (
            "WHERE any(p IN section_paths WHERE p CONTAINS '/')" if section_only else ""
        )
        params: dict[str, Any] = {"ids_filter": filter_list}
        if scope:
            params["scope"] = scope

        dd_clause = _dd_version_clause("p", dd_version, params)

        results = self._gc.query(
            f"""
            MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            WHERE p.ids IN $ids_filter
            {dd_clause}
            {scope_filter}
            WITH c, collect(DISTINCT p.id) AS section_paths,
                 collect(DISTINCT p.ids) AS ids_covered
            {section_filter}
            RETURN c.id AS id, c.label AS label, c.description AS description,
                   c.scope AS scope, c.cross_ids AS cross_ids,
                   c.ids_names AS ids_names, c.similarity_score AS similarity,
                   section_paths AS paths
            ORDER BY size(section_paths) DESC
            """,
            **params,
        )

        clusters = self._format_clusters(results or [])
        return {
            "query": None,
            "query_type": "ids_listing",
            "ids_filter": filter_list,
            "section_only": section_only,
            "clusters_found": len(clusters),
            "clusters": clusters,
        }

    def _search_by_path(
        self, path: str, scope: str | None, *, dd_version: int | None = None
    ) -> dict[str, Any]:
        """Find clusters containing a specific path."""
        scope_filter = "AND c.scope = $scope" if scope else ""
        params: dict[str, Any] = {"path": path}
        if scope:
            params["scope"] = scope

        dd_clause = _dd_version_clause("member", dd_version, params)

        results = self._gc.query(
            f"""
            MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            WHERE true {scope_filter}
            OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
            WHERE true {dd_clause}
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
        *,
        dd_version: int | None = None,
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
                'cluster_embedding', $k, $embedding
            )
            YIELD node AS cluster, score
            WHERE score > 0.3
            {scope_filter}
            {ids_filter_clause}
            OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(cluster)
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

        # Enrich unlabeled clusters with member-derived context
        unlabeled_ids = [
            c["id"] for c in clusters if not c.get("label") and c.get("paths")
        ]
        if unlabeled_ids:
            member_docs = self._gc.query(
                """
                UNWIND $cluster_ids AS cid
                MATCH (c:IMASSemanticCluster {id: cid})<-[:IN_CLUSTER]-(m:IMASNode)
                WITH c, m ORDER BY m.id
                WITH c.id AS cluster_id,
                     collect(DISTINCT m.ids)[..5] AS ids_names,
                     collect(m.documentation)[..5] AS sample_docs,
                     collect(m.id)[..5] AS sample_paths
                RETURN cluster_id, ids_names, sample_docs, sample_paths
                """,
                cluster_ids=unlabeled_ids,
            )
            member_ctx = {r["cluster_id"]: r for r in member_docs or []}
            for cluster in clusters:
                if cluster["id"] in member_ctx:
                    ctx = member_ctx[cluster["id"]]
                    if not cluster.get("label"):
                        # Derive label from common path prefix
                        paths = ctx.get("sample_paths", [])
                        if paths:
                            prefix = _common_path_prefix(paths)
                            cluster["label"] = prefix or ""
                    if not cluster.get("description"):
                        docs = [d for d in (ctx.get("sample_docs") or []) if d]
                        if docs:
                            cluster["description"] = "; ".join(docs[:3])
                    if not cluster.get("ids") and ctx.get("ids_names"):
                        cluster["ids"] = ctx["ids_names"]

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
        "query: Optional filter (e.g., 'coordinate' or 'magnetics'). "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions."
    )
    async def get_imas_identifiers(
        self,
        query: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> GetIdentifiersResult:
        """Get identifier schemas from graph.

        When a query is provided, uses a two-strategy approach:
        1. Vector similarity search on enriched description embeddings
        2. Keyword matching on name, description, keywords, and options

        Results from both strategies are merged (vector matches first).
        """
        if query:
            return self._search_identifiers(query)
        return self._list_all_identifiers()

    def _list_all_identifiers(self) -> GetIdentifiersResult:
        """Return all identifier schemas."""
        results = self._gc.query(
            """
            MATCH (s:IdentifierSchema)
            RETURN s.name AS name,
                   COALESCE(s.description, s.documentation) AS description,
                   s.keywords AS keywords,
                   s.option_count AS option_count, s.options AS options,
                   s.field_count AS field_count, s.source AS source
            ORDER BY s.name
            """
        )
        schemas = [self._format_schema(r) for r in results or []]
        total_options = sum(s["option_count"] for s in schemas)

        return GetIdentifiersResult(
            schemas=schemas,
            paths=[],
            analytics={
                "total_schemas": len(schemas),
                "total_paths": 0,
                "enumeration_space": total_options,
                "query_context": None,
            },
        )

    def _search_identifiers(self, query: str) -> GetIdentifiersResult:
        """Search identifier schemas by vector similarity + keyword match."""
        seen: set[str] = set()
        schemas: list[dict] = []

        # Strategy 1: Vector similarity search
        try:
            from imas_codex.embeddings.config import EncoderConfig
            from imas_codex.embeddings.encoder import Encoder
            from imas_codex.settings import (
                get_embedding_dimension,
                get_embedding_model,
            )

            model_name = get_embedding_model()
            dim = get_embedding_dimension()
            encoder = Encoder(
                config=EncoderConfig(model_name=model_name, dimension=dim)
            )
            query_embedding = encoder.encode([query])[0].tolist()

            vector_results = self._gc.query(
                """
                CALL db.index.vector.queryNodes(
                    'identifier_schema_embedding', $k, $embedding
                ) YIELD node, score
                WHERE score >= $threshold
                RETURN node.name AS name,
                       COALESCE(node.description, node.documentation) AS description,
                       node.keywords AS keywords,
                       node.option_count AS option_count,
                       node.options AS options,
                       node.field_count AS field_count,
                       node.source AS source,
                       score
                """,
                embedding=query_embedding,
                k=10,
                threshold=0.3,
            )
            for r in vector_results or []:
                name = r["name"]
                if name not in seen:
                    seen.add(name)
                    schemas.append(self._format_schema(r))
        except Exception:
            pass  # Vector index may not exist yet

        # Strategy 2: Keyword matching on remaining schemas
        keyword_results = self._gc.query(
            """
            MATCH (s:IdentifierSchema)
            RETURN s.name AS name,
                   COALESCE(s.description, s.documentation) AS description,
                   s.keywords AS keywords,
                   s.option_count AS option_count, s.options AS options,
                   s.field_count AS field_count, s.source AS source
            ORDER BY s.name
            """
        )
        query_lower = query.lower()
        for r in keyword_results or []:
            name = r["name"]
            if name in seen:
                continue
            name_match = query_lower in (name or "").lower()
            desc_match = query_lower in (r["description"] or "").lower()
            opts_match = query_lower in (r["options"] or "").lower()
            kw_match = any(
                query_lower in kw.lower() for kw in (r.get("keywords") or [])
            )
            if name_match or desc_match or opts_match or kw_match:
                seen.add(name)
                schemas.append(self._format_schema(r))

        total_options = sum(s["option_count"] for s in schemas)
        return GetIdentifiersResult(
            schemas=schemas,
            paths=[],
            analytics={
                "total_schemas": len(schemas),
                "total_paths": 0,
                "enumeration_space": total_options,
                "query_context": query,
            },
        )

    @staticmethod
    def _format_schema(r: dict) -> dict:
        """Format a graph row into a schema result dict."""
        options = []
        if r.get("options"):
            try:
                options = json.loads(r["options"])
            except (json.JSONDecodeError, TypeError):
                pass

        option_count = r.get("option_count") or len(options)
        desc = r.get("description") or ""
        return {
            "path": r["name"],
            "schema_path": r.get("source") or "",
            "option_count": option_count,
            "branching_significance": _classify_significance(option_count),
            "options": options,
            "description": desc,
        }


class GraphPathContextTool:
    """Graph-backed path context for cross-IDS relationship discovery."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @mcp_tool(
        "Get structural context for an IMAS path via graph traversal and semantic similarity. "
        "Combines vector embedding similarity, cluster membership, physics coordinate sharing, "
        "unit+domain affinity, and identifier schemas to discover meaningful cross-IDS relationships. "
        "path (required): Exact IMAS path (e.g. 'equilibrium/time_slice/profiles_1d/psi'). "
        "relationship_types: Filter to 'semantic', 'cluster', 'coordinate', 'unit', "
        "'identifier', or 'all' (default). "
        "max_results: Maximum results per section (default 20). "
        "dd_version: Filter by DD major version (e.g., 3 or 4)."
    )
    @handle_errors("get_imas_path_context")
    async def get_imas_path_context(
        self,
        path: str,
        relationship_types: str = "all",
        max_results: int = 20,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Discover cross-IDS relationships for an IMAS path via graph + semantics.

        Strategy:
        1. Semantic neighbors  — vector similarity via imas_node_embedding index
        2. Cluster siblings    — explicit cluster co-membership (different IDS)
        3. Coordinate partners — paths sharing physics-specific coordinates only
                                 (generic coords like '1...N' are filtered out)
        4. Unit companions     — same unit symbol in same physics domain
        5. Identifier links    — shared identifier schema (enumerations)
        """
        sections: dict[str, list[dict[str, Any]]] = {}

        # ── Fetch source node properties upfront ────────────────────────────
        node_rows = self._gc.query(
            "MATCH (p:IMASNode {id: $path}) "
            "RETURN p.embedding AS embedding, p.ids AS ids, "
            "       p.physics_domain AS physics_domain",
            path=path,
        )
        if not node_rows:
            return {
                "path": path,
                "error": f"Path '{path}' not found in IMAS graph",
                "sections": sections,
                "total_connections": 0,
            }

        node = node_rows[0]
        src_ids: str = node.get("ids") or ""
        src_domain: str = node.get("physics_domain") or ""
        embedding: list[float] | None = node.get("embedding")

        # ── 1. Semantic neighbors via vector index ───────────────────────────
        # Primary signal: find paths whose description embedding is closest to
        # this path's embedding. Filtered to different IDS only.
        if relationship_types in ("all", "semantic") and embedding:
            sem_params: dict[str, Any] = {
                "embedding": embedding,
                "src_ids": src_ids,
                "sem_k": min(max_results * 10, 200),
                "sem_limit": max_results,
            }
            dd_clause_sem = _dd_version_clause("sibling", dd_version, sem_params)
            semantic_results = self._gc.query(
                f"""
                CALL db.index.vector.queryNodes('imas_node_embedding', $sem_k, $embedding)
                YIELD node AS sibling, score
                WHERE sibling.ids <> $src_ids
                  AND sibling.node_category = 'data'
                  AND NOT (sibling)-[:DEPRECATED_IN]->(:DDVersion)
                  {dd_clause_sem}
                OPTIONAL MATCH (sibling)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (sibling)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
                RETURN sibling.id AS path, sibling.ids AS ids,
                       coalesce(sibling.description, sibling.documentation) AS description,
                       sibling.data_type AS data_type, u.id AS unit,
                       sibling.physics_domain AS physics_domain,
                       collect(DISTINCT cl.label) AS clusters,
                       round(score, 4) AS score
                ORDER BY score DESC
                LIMIT $sem_limit
                """,
                **sem_params,
            )
            if semantic_results:
                sections["semantic_neighbors"] = semantic_results

        # ── 2. Cluster siblings — explicit cluster co-membership ─────────────
        if relationship_types in ("all", "cluster"):
            cl_params: dict[str, Any] = {"path": path, "cl_limit": max_results * 2}
            dd_clause_cl = _dd_version_clause("sibling", dd_version, cl_params)
            cluster_siblings = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
                      <-[:IN_CLUSTER]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause_cl}
                RETURN cl.label AS cluster, sibling.id AS path,
                       sibling.ids AS ids,
                       coalesce(sibling.description, sibling.documentation) AS doc
                ORDER BY cl.label, sibling.ids
                LIMIT $cl_limit
                """,
                **cl_params,
            )
            if cluster_siblings:
                sections["cluster_siblings"] = cluster_siblings

        # ── 3. Physics coordinate partners (generic coords filtered out) ─────
        # Only match on IMASCoordinateSpec nodes whose id contains '/' —
        # i.e. actual IMAS path-based coordinate expressions like
        # "profiles_1d(itime)/grid/rho_tor_norm". Generic tokens like
        # "1...N" are shared across thousands of nodes and carry no semantic
        # meaning; matching on them produces 13 k+ noise results.
        if relationship_types in ("all", "coordinate"):
            co_params: dict[str, Any] = {"path": path}
            dd_clause_co = _dd_version_clause("sibling", dd_version, co_params)
            coord_partners = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
                WHERE coord.id CONTAINS '/'
                MATCH (coord)<-[:HAS_COORDINATE]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause_co}
                RETURN coord.id AS coordinate, sibling.id AS path,
                       sibling.ids AS ids, sibling.data_type AS data_type,
                       sibling.physics_domain AS physics_domain
                ORDER BY coord.id, sibling.ids
                LIMIT 40
                """,
                **co_params,
            )
            if coord_partners:
                sections["coordinate_partners"] = coord_partners

        # ── 4. Unit companions — same unit, same physics domain ──────────────
        if relationship_types in ("all", "unit") and src_domain:
            u_params: dict[str, Any] = {"path": path, "src_domain": src_domain}
            dd_clause_u = _dd_version_clause("sibling", dd_version, u_params)
            unit_companions = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_UNIT]->(u:Unit)
                      <-[:HAS_UNIT]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids
                  AND sibling.physics_domain = $src_domain {dd_clause_u}
                RETURN u.id AS unit, sibling.id AS path,
                       sibling.ids AS ids,
                       coalesce(sibling.description, sibling.documentation) AS doc
                ORDER BY u.id, sibling.ids
                LIMIT 30
                """,
                **u_params,
            )
            if unit_companions:
                sections["unit_companions"] = unit_companions

        # ── 5. Identifier schema links ───────────────────────────────────────
        if relationship_types in ("all", "identifier"):
            id_params: dict[str, Any] = {"path": path}
            dd_clause_id = _dd_version_clause("sibling", dd_version, id_params)
            ident_links = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
                      <-[:HAS_IDENTIFIER_SCHEMA]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause_id}
                RETURN s.name AS schema, sibling.id AS path,
                       sibling.ids AS ids
                ORDER BY s.name
                LIMIT 30
                """,
                **id_params,
            )
            if ident_links:
                sections["identifier_links"] = ident_links

        return {
            "path": path,
            "relationship_types": relationship_types,
            "sections": sections,
            "total_connections": sum(len(v) for v in sections.values()),
        }


class GraphStructureTool:
    """Graph-backed structural analysis of IMAS IDS."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @mcp_tool(
        "Analyze the hierarchical structure of an IMAS IDS. "
        "Returns depth metrics, leaf/structure ratio, array patterns, "
        "physics domain distribution, coordinate usage, and COCOS-labeled fields. "
        "ids_name (required): IDS name (e.g. 'equilibrium')."
    )
    @handle_errors("analyze_imas_structure")
    async def analyze_imas_structure(
        self,
        ids_name: str,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Analyze the hierarchical structure of an IMAS IDS."""
        dd_params: dict[str, Any] = {"ids_name": ids_name}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        # Basic metrics — single scan using nullIf for leaf counting (Neo4j 2026 compat)
        metrics = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name {dd_clause}
            RETURN count(p) AS total_paths,
                   max(size(split(p.id, '/')) - 1) AS max_depth,
                   avg(size(split(p.id, '/')) - 1) AS avg_depth,
                   count(nullIf(
                       p.data_type IS NULL OR p.data_type IN {_structure_type_list()},
                       true
                   )) AS leaf_count
            """,
            **dd_params,
        )

        # Physics domain distribution
        domains = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name AND p.physics_domain IS NOT NULL {dd_clause}
            RETURN p.physics_domain AS domain, count(p) AS count
            ORDER BY count DESC
            """,
            **dd_params,
        )

        # Data type distribution
        types = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name AND p.data_type IS NOT NULL {dd_clause}
            RETURN p.data_type AS data_type, count(p) AS count
            ORDER BY count DESC
            """,
            **dd_params,
        )

        # Array structures with coordinates
        arrays = self._gc.query(
            f"""
            MATCH (p:IMASNode)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
            WHERE p.ids = $ids_name {dd_clause}
            RETURN p.id AS path, collect(coord.id) AS coordinates
            ORDER BY p.id
            """,
            **dd_params,
        )

        # COCOS-labeled fields
        cocos_fields = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name
              AND p.cocos_label_transformation IS NOT NULL {dd_clause}
            RETURN p.id AS path,
                   p.cocos_label_transformation AS cocos_label
            ORDER BY p.id
            """,
            **dd_params,
        )

        basic = metrics[0] if metrics else {}
        total_paths = basic.get("total_paths", 0)
        leaf_count = basic.get("leaf_count", 0)
        return {
            "ids_name": ids_name,
            "total_paths": total_paths,
            "leaf_count": leaf_count,
            "structure_count": total_paths - leaf_count,
            "max_depth": basic.get("max_depth", 0),
            "avg_depth": round(basic.get("avg_depth", 0), 1),
            "physics_domains": [
                {"domain": d["domain"], "count": d["count"]} for d in domains
            ],
            "data_types": [
                {"type": t["data_type"], "count": t["count"]} for t in types
            ],
            "array_structures": [
                {"path": a["path"], "coordinates": a["coordinates"]} for a in arrays
            ],
            "cocos_fields": [
                {"path": c["path"], "label": c["cocos_label"]} for c in cocos_fields
            ],
        }

    @mcp_tool(
        "Export full IDS structure with documentation, units, and types. "
        "Returns all paths in an IDS with their complete metadata. "
        "ids_name (required): IDS name (e.g. 'equilibrium'). "
        "leaf_only: If true, return only leaf nodes (default false)."
    )
    @handle_errors("export_imas_ids")
    async def export_imas_ids(
        self,
        ids_name: str,
        leaf_only: bool = False,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Export full IDS structure with documentation, units, and types."""
        dd_params: dict[str, Any] = {"ids_name": ids_name}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        leaf_filter = f"AND {_leaf_data_type_clause('p')}" if leaf_only else ""

        paths = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name
            AND p.node_category = 'data'
            {leaf_filter} {dd_clause}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
            OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
            RETURN p.id AS path,
                   p.documentation AS documentation,
                   p.data_type AS data_type,
                   p.units AS units,
                   u.id AS unit_id,
                   p.physics_domain AS physics_domain,
                   p.lifecycle_status AS lifecycle_status,
                   p.cocos_label_transformation AS cocos_label,
                   collect(DISTINCT coord.id) AS coordinates,
                   collect(DISTINCT cl.label) AS clusters
            ORDER BY p.id
            """,
            **dd_params,
        )

        return {
            "ids_name": ids_name,
            "leaf_only": leaf_only,
            "path_count": len(paths),
            "paths": paths,
        }

    @mcp_tool(
        "Export all IMAS paths in a physics domain, grouped by IDS. "
        "domain (required): Physics domain name (e.g. 'magnetics', 'equilibrium'). "
        "ids_filter: Optional IDS name filter."
    )
    @handle_errors("export_imas_domain")
    async def export_imas_domain(
        self,
        domain: str,
        ids_filter: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Export all IMAS paths in a physics domain, grouped by IDS."""
        resolved_domains, resolution = _resolve_physics_domain(self._gc, domain)

        if not resolved_domains:
            return {
                "domain": domain,
                "resolved_domains": [],
                "resolution": "no_match",
                "total_paths": 0,
                "ids_count": 0,
                "by_ids": {},
                "error": f"No physics domain found matching '{domain}'.",
            }

        dd_params: dict[str, Any] = {"domains": resolved_domains}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        ids_clause = ""
        if ids_filter:
            ids_clause = "AND p.ids = $ids_filter"
            dd_params["ids_filter"] = ids_filter

        paths = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.physics_domain IN $domains
            AND p.node_category = 'data'
            {ids_clause} {dd_clause}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS path,
                   p.ids AS ids,
                   p.documentation AS documentation,
                   p.data_type AS data_type,
                   p.units AS units,
                   u.id AS unit_id
            ORDER BY p.ids, p.id
            """,
            **dd_params,
        )

        # Group by IDS
        grouped: dict[str, list[dict[str, Any]]] = {}
        for p in paths:
            ids = p.get("ids", "unknown")
            grouped.setdefault(ids, []).append(p)

        return {
            "domain": domain,
            "resolved_domains": resolved_domains,
            "resolution": resolution,
            "ids_filter": ids_filter,
            "total_paths": len(paths),
            "ids_count": len(grouped),
            "by_ids": grouped,
        }


def _normalize_paths(paths: str | list[str]) -> list[str]:
    """Normalize paths input to a flat list, stripping index annotations."""
    import json

    from imas_codex.core.paths import strip_path_annotations

    if isinstance(paths, list):
        return [strip_path_annotations(p) for p in paths]

    s = paths.strip()
    # Handle JSON array strings from MCP transport
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    strip_path_annotations(str(p).strip())
                    for p in parsed
                    if str(p).strip()
                ]
        except (json.JSONDecodeError, TypeError):
            pass
    raw = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [strip_path_annotations(p) for p in raw]


STRUCTURE_DATA_TYPES = ("STRUCTURE", "STRUCT_ARRAY")


def _structure_type_list() -> str:
    """Return a Cypher list literal of structure data types."""
    quoted = ", ".join(f"'{dtype}'" for dtype in STRUCTURE_DATA_TYPES)
    return f"[{quoted}]"


def _leaf_data_type_clause(alias: str) -> str:
    """Return a Cypher clause that matches non-structure data nodes."""
    return f"{alias}.data_type IS NOT NULL AND NOT ({alias}.data_type IN {_structure_type_list()})"


def _text_search_imas_paths(
    gc: GraphClient,
    query: str,
    limit: int,
    ids_filter: str | list[str] | None,
    *,
    dd_version: int | None = None,
) -> list[dict[str, Any]]:
    """Text-based search on IMAS paths by query string.

    Uses fulltext index for BM25 scoring when available, falls back to
    CONTAINS matching. Filters out generic metadata paths.
    """
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]

    where_parts = ["NOT (p)-[:DEPRECATED_IN]->(:DDVersion)", "p.node_category = 'data'"]
    params: dict[str, Any] = {"query_lower": query_lower, "limit": limit}

    dd_clause = _dd_version_clause("p", dd_version, params)
    if dd_clause:
        where_parts.append(dd_clause.lstrip("AND "))

    if ids_filter is not None:
        filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
        where_parts.append("p.ids IN $ids_filter")
        params["ids_filter"] = filter_list

    where_base = " AND ".join(where_parts)

    # Try fulltext index first (BM25 scoring)
    try:
        ft_where = "WHERE NOT (p)-[:DEPRECATED_IN]->(:DDVersion) AND p.node_category = 'data'"
        ft_params: dict[str, Any] = {"query": query, "limit": limit}
        if ids_filter is not None:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            ft_where += " AND p.ids IN $ids_filter"
            ft_params["ids_filter"] = filter_list

        ft_dd_clause = _dd_version_clause("p", dd_version, ft_params)
        if ft_dd_clause:
            ft_where += f" {ft_dd_clause}"

        ft_cypher = f"""
            CALL db.index.fulltext.queryNodes('imas_node_text', $query)
            YIELD node AS p, score
            {ft_where}
            WITH p, score
            WHERE size(coalesce(p.documentation, '')) > 10
                  OR p.description IS NOT NULL
            RETURN p.id AS id, score
            LIMIT $limit
        """
        ft_results = gc.query(ft_cypher, **ft_params)
        if ft_results:
            # Normalize BM25 scores to 0-1 range
            max_score = max(r["score"] for r in ft_results) if ft_results else 1.0
            normalized = []
            for r in ft_results:
                pid = r["id"]
                raw = r["score"] / max_score if max_score > 0 else 0.0
                normalized.append({"id": pid, "score": max(raw, 0.7)})
            return normalized
    except Exception:
        pass

    # Fallback: CONTAINS matching with scored results
    cypher = f"""
        MATCH (p:IMASNode)
        WHERE {where_base}
          AND (
            toLower(p.documentation) CONTAINS $query_lower
            OR toLower(p.id) CONTAINS $query_lower
            OR toLower(p.name) CONTAINS $query_lower
            OR toLower(coalesce(p.description, '')) CONTAINS $query_lower
            OR any(kw IN coalesce(p.keywords, []) WHERE toLower(kw) CONTAINS $query_lower)
          )
          AND size(coalesce(p.documentation, '')) > 10
        WITH p,
             CASE
            WHEN toLower(p.documentation) CONTAINS $query_lower
                AND {_leaf_data_type_clause('p')}
                 THEN 0.95
               WHEN toLower(p.documentation) CONTAINS $query_lower
                 THEN 0.88
               WHEN toLower(p.name) CONTAINS $query_lower
                AND {_leaf_data_type_clause('p')}
                 THEN 0.93
               WHEN toLower(p.id) CONTAINS $query_lower
                 THEN 0.90
               ELSE 0.85
             END AS base_score
        RETURN p.id AS id, base_score AS score
        ORDER BY base_score DESC, size(p.id) ASC
        LIMIT $limit
    """
    results = gc.query(cypher, **params)

    # Also search individual words for abbreviations like "ip"
    if query_words:
        word_results = []
        for word in query_words[:3]:
            word_cypher = f"""
                MATCH (p:IMASNode)
                WHERE {where_base}
                  AND (toLower(p.name) = $word OR toLower(p.id) CONTAINS $word)
                                    AND {_leaf_data_type_clause('p')}
                  AND size(coalesce(p.documentation, '')) > 10
                RETURN p.id AS id, 0.90 AS score
                LIMIT 10
            """
            word_params = {**params, "word": word}
            word_results.extend(gc.query(word_cypher, **word_params))

        seen = {r["id"]: r["score"] for r in results}
        for r in word_results:
            if r["id"] not in seen or r["score"] > seen[r["id"]]:
                seen[r["id"]] = r["score"]
                results.append(r)

    # Deduplicate and filter generic paths
    final: dict[str, dict[str, Any]] = {}
    for r in results:
        pid = r["id"]
        if pid not in final or r["score"] > final[pid]["score"]:
            final[pid] = r
    return list(final.values())


def _classify_significance(option_count: int) -> str:
    """Classify branching significance based on option count."""
    if option_count >= 20:
        return "CRITICAL"
    elif option_count >= 10:
        return "HIGH"
    elif option_count >= 5:
        return "MODERATE"
    return "MINIMAL"


def _get_facility_crossrefs(
    gc: GraphClient,
    path_ids: list[str],
    facility: str,
) -> dict[str, dict[str, Any]]:
    """Get facility cross-references for IMAS paths.

    Uses both relationship-based traversals (populated by migration/ingestion)
    and property-based fallbacks for comprehensive results:
    - FacilitySignal: IMASMapping traversal via SignalNode
    - WikiChunk: MENTIONS_IMAS relationship or imas_paths_mentioned property
    - CodeChunk: RESOLVES_TO_IMAS_PATH via DataReference or related_ids property
    """
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (ip:IMASNode {id: pid})
        OPTIONAL MATCH (sig:FacilitySignal {facility_id: $facility})
            -[:HAS_DATA_SOURCE_NODE]->(dn:SignalNode)
            -[:MEMBER_OF]->(sg:SignalSource)-[:MAPS_TO_IMAS]->(ip)
        OPTIONAL MATCH (sig2:FacilitySignal)
        WHERE sig2.facility_id = $facility
          AND sig2.physics_domain IS NOT NULL
          AND ip.ids = sig2.physics_domain
        WITH ip,
             collect(DISTINCT sig.id) + collect(DISTINCT sig2.id) AS all_sigs
        OPTIONAL MATCH (wc:WikiChunk {facility_id: $facility})-[:MENTIONS_IMAS]->(ip)
        OPTIONAL MATCH (wc2:WikiChunk)
        WHERE wc2.facility_id = $facility
          AND wc2.imas_paths_mentioned IS NOT NULL
          AND ip.id IN wc2.imas_paths_mentioned
        WITH ip, all_sigs,
             collect(DISTINCT wc.section) + collect(DISTINCT wc2.section) AS all_wiki
        OPTIONAL MATCH (cc:CodeChunk {facility_id: $facility})
            -[:CONTAINS_REF]->(dr:DataReference)-[:RESOLVES_TO_IMAS_PATH]->(ip)
        OPTIONAL MATCH (cc2:CodeChunk)
        WHERE cc2.facility_id = $facility
          AND cc2.related_ids IS NOT NULL
          AND ip.ids IN cc2.related_ids
        RETURN ip.id AS id,
               [x IN all_sigs WHERE x IS NOT NULL] AS facility_signals,
               [x IN all_wiki WHERE x IS NOT NULL] AS wiki_mentions,
               [x IN collect(DISTINCT cc.source_file) +
                    collect(DISTINCT cc2.source_file) WHERE x IS NOT NULL] AS code_files
    """
    results = gc.query(cypher, path_ids=path_ids, facility=facility)
    return {r["id"]: r for r in results}


def _get_version_context(
    gc: GraphClient,
    path_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Get version change context for IMAS paths."""
    cypher = """
        UNWIND $path_ids AS pid
        MATCH (p:IMASNode {id: pid})
        OPTIONAL MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p)
        WHERE change.semantic_change_type IN
              ['sign_convention', 'coordinate_convention', 'units', 'definition_clarification']
        RETURN p.id AS id,
               count(change) AS change_count,
               collect({version: change.version,
                        type: change.semantic_change_type,
                        summary: change.summary})[..5] AS notable_changes
    """
    results = gc.query(cypher, path_ids=path_ids)
    return {r["id"]: r for r in results}


def _common_path_prefix(paths: list[str]) -> str:
    """Find the longest common path prefix from a list of IMAS paths."""
    if not paths:
        return ""
    split_paths = [p.split("/") for p in paths]
    prefix = []
    for segments in zip(*split_paths, strict=False):
        if len(set(segments)) == 1:
            prefix.append(segments[0])
        else:
            break
    return "/".join(prefix)


def _resolve_physics_domain(
    gc: GraphClient, domain_query: str
) -> tuple[list[str], str]:
    """Resolve a user-provided domain query to canonical physics domain names.

    Resolution order:
    1. Exact match on PhysicsDomain enum value
    2. IDS name → its physics_domain from the graph
    3. DomainCategory expansion (all domains in that category)
    4. Substring match on domain names

    Returns:
        (resolved_domains, resolution_method) — list of canonical domain
        names and a string describing how the input was resolved.
    """
    from imas_codex.core.physics_domain import PhysicsDomain

    query = domain_query.strip().lower().replace(" ", "_").replace("-", "_")

    # 1. Exact match on PhysicsDomain enum value
    valid_domains = {d.value for d in PhysicsDomain}
    if query in valid_domains:
        return [query], "exact"

    # 2. IDS name → its physics_domain
    ids_rows = gc.query(
        "MATCH (i:IDS {name: $name}) RETURN i.physics_domain AS domain",
        name=query,
    )
    if ids_rows:
        domains = [r["domain"] for r in ids_rows if r.get("domain")]
        if domains:
            return sorted(set(domains)), f"ids_name:{query}"

    # 3. DomainCategory expansion
    try:
        from pathlib import Path

        from linkml_runtime.utils.schemaview import SchemaView

        schema_path = Path(__file__).parent.parent / "definitions/physics/domains.yaml"
        if schema_path.exists():
            sv = SchemaView(str(schema_path))
            cat_enum = sv.get_enum("DomainCategory")
            if cat_enum and query in cat_enum.permissible_values:
                pd_enum = sv.get_enum("PhysicsDomain")
                if pd_enum:
                    domains = []
                    for pv_name, pv in pd_enum.permissible_values.items():
                        cat = (
                            pv.annotations.get("category", {}).get("value", "")
                            if pv.annotations
                            else ""
                        )
                        if cat == query:
                            domains.append(pv_name)
                    if domains:
                        return sorted(domains), f"category:{query}"
    except Exception:
        pass

    # 4. Substring match on domain names
    matches = [d for d in sorted(valid_domains) if query in d]
    if matches:
        return matches, f"substring:{query}"

    return [], "no_match"
