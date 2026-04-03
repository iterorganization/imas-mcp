"""Graph-backed tool implementations for graph-native MCP server.

Replaces DocumentStore, SemanticSearch, and ClusterSearcher with
GraphClient queries against Neo4j vector indexes and Cypher traversals.
"""

import json
import logging
import re
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
from imas_codex.search.fuzzy_matcher import suggest_correction
from imas_codex.search.search_strategy import SearchHit
from imas_codex.tools.query_analysis import QueryAnalyzer, strip_accessor_suffix
from imas_codex.tools.utils import normalize_ids_filter, validate_query

logger = logging.getLogger(__name__)

# Short physics terms that must not be filtered out of search queries
_PHYSICS_SHORT_TERMS = frozenset(
    {
        "q",
        "ip",
        "b0",
        "te",
        "ne",
        "ti",
        "ni",
        "psi",
        "r",
        "z",
        "phi",
        "j",
        "e",
        "b",
        "v",
        "p",
        "rho",
        "li",
        "wi",
        "we",
    }
)

# Module-level encoder singleton — avoids re-loading the model per query
_encoder: Any = None
_encoder_lock: Any = None


def _get_encoder():
    """Get or create the module-level Encoder singleton."""
    global _encoder, _encoder_lock
    import threading

    if _encoder_lock is None:
        _encoder_lock = threading.Lock()

    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from imas_codex.embeddings.encoder import Encoder

                _encoder = Encoder()
    return _encoder


def warmup_encoder():
    """Pre-warm the encoder by loading the model.

    Call from a background thread at server startup so the first
    search_imas call doesn't pay the cold-start penalty.
    """
    try:
        encoder = _get_encoder()
        encoder.embed_texts(["warmup"])
        logger.info("Encoder warmup complete")
    except Exception as e:
        logger.warning(f"Encoder warmup failed (will retry on first query): {e}")


def resolve_dd_version(dd_version: int | str | None) -> int | None:
    """Resolve flexible dd_version input to integer major version.

    Accepts: int (3, 4), str ("3.39.0", "4", "latest"), or None.
    Returns: int major version or None (no filter).
    """
    if dd_version is None:
        return None
    if isinstance(dd_version, int):
        return dd_version
    if isinstance(dd_version, str):
        v = dd_version.strip().lower()
        if v == "latest":
            return None
        try:
            return int(v.split(".")[0])
        except (ValueError, IndexError):
            return None
    return None


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    text_results: list[dict],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion — rank-based, score-normalization-free.

    Standard RRF from Cormack et al. (2009). Immune to score normalization
    issues because it uses rank positions, not raw scores. The ``k``
    constant (default 60) controls how quickly relevance decays with rank.
    """
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
    for rank, r in enumerate(text_results):
        scores[r["id"]] = scores.get(r["id"], 0.0) + 1.0 / (k + rank + 1)
    return scores


def _apply_cluster_boost(
    gc: "GraphClient",
    scores: dict[str, float],
    top_n: int = 10,
    boost: float = 0.02,
) -> dict[str, float]:
    """Boost paths sharing clusters with top-ranked results."""
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if not top_ids:
        return scores

    all_ids = list(scores.keys())
    result = gc.query(
        """
        UNWIND $top_ids AS tid
        MATCH (top:IMASNode {id: tid})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        WITH collect(DISTINCT c.id) AS top_clusters
        UNWIND $all_ids AS aid
        MATCH (a:IMASNode {id: aid})-[:IN_CLUSTER]->(c2:IMASSemanticCluster)
        WHERE c2.id IN top_clusters
        RETURN aid AS id, count(DISTINCT c2) AS cluster_overlap
        """,
        top_ids=top_ids,
        all_ids=all_ids,
    )

    boosted = dict(scores)
    for row in result:
        pid = row["id"]
        if pid in boosted:
            boosted[pid] += boost * row["cluster_overlap"]
    return boosted


def _apply_hierarchy_boost(
    gc: "GraphClient",
    scores: dict[str, float],
    top_n: int = 10,
    boost: float = 0.02,
) -> dict[str, float]:
    """Boost siblings of top-ranked results via shared HAS_PARENT."""
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if not top_ids:
        return scores

    all_ids = list(scores.keys())
    result = gc.query(
        """
        UNWIND $top_ids AS tid
        MATCH (top:IMASNode {id: tid})-[:HAS_PARENT]->(parent)
            <-[:HAS_PARENT]-(sibling:IMASNode)
        WHERE sibling.id IN $all_ids AND sibling.id <> tid
        RETURN DISTINCT sibling.id AS id
        """,
        top_ids=top_ids,
        all_ids=all_ids,
    )

    boosted = dict(scores)
    for row in result:
        pid = row["id"]
        if pid in boosted:
            boosted[pid] += boost
    return boosted


def _apply_coordinate_boost(
    gc: "GraphClient",
    scores: dict[str, float],
    top_n: int = 5,
    boost: float = 0.01,
) -> dict[str, float]:
    """Boost paths sharing coordinate bases with top results."""
    top_ids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    if not top_ids:
        return scores

    all_ids = list(scores.keys())
    result = gc.query(
        """
        UNWIND $top_ids AS tid
        MATCH (top:IMASNode {id: tid})-[:HAS_COORDINATE]->(coord:IMASNode)
            <-[:HAS_COORDINATE]-(related:IMASNode)
        WHERE related.id IN $all_ids AND related.id <> tid
        RETURN DISTINCT related.id AS id
        """,
        top_ids=top_ids,
        all_ids=all_ids,
    )

    boosted = dict(scores)
    for row in result:
        pid = row["id"]
        if pid in boosted:
            boosted[pid] += boost
    return boosted


def _dd_version_clause(
    alias: str = "p",
    dd_version: int | None = None,
    params: dict[str, Any] | None = None,
) -> str:
    """Return a Cypher WHERE fragment for DD major version filtering.

    When dd_version is None, returns empty string (no filter).
    Otherwise generates a clause ensuring the path was introduced in
    DD version N or earlier, and not deprecated in version N or earlier.
    This returns all paths **active** in the given major version — not
    only paths newly introduced in that version.

    The DDVersion.id is a semver string like "3.42.2" or "4.0.0".
    Major version is extracted via ``toInteger(split(id, '.')[0])``.

    If *params* dict is provided, adds ``dd_major_version`` to it.
    """
    if dd_version is None:
        return ""
    if params is not None:
        params["dd_major_version"] = dd_version
    return (
        f"AND EXISTS {{ "
        f"  MATCH ({alias})-[:INTRODUCED_IN]->(iv:DDVersion) "
        f"  WHERE toInteger(split(iv.id, '.')[0]) <= $dd_major_version "
        f"}} "
        f"AND NOT EXISTS {{ "
        f"  MATCH ({alias})-[:DEPRECATED_IN]->(dv:DDVersion) "
        f"  WHERE toInteger(split(dv.id, '.')[0]) <= $dd_major_version "
        f"}}"
    )


def _path_search(
    gc: GraphClient,
    query: str,
    max_results: int,
    ids_filter: str | list[str] | None,
    *,
    dd_version: int | None = None,
) -> list[dict[str, Any]]:
    """Search by path structure — exact, suffix, and substring matching.

    Used for queries that look like IMAS paths (contain '/').
    Returns results with scores in [0, 1] range.
    """
    query_lower = query.lower().strip()
    params: dict[str, Any] = {"search_query": query_lower, "limit": max_results}
    dd_params: dict[str, Any] = {}
    dd_clause = _dd_version_clause("p", dd_version, dd_params)
    params.update(dd_params)

    ids_clause = ""
    if ids_filter:
        filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
        ids_clause = "AND p.ids IN $ids_filter"
        params["ids_filter"] = filter_list

    results: dict[str, float] = {}

    # Strategy 1: Exact match (highest score)
    exact = gc.query(
        f"""
        MATCH (p:IMASNode)
        WHERE toLower(p.id) = $search_query
          AND p.node_category = 'data'
          {ids_clause} {dd_clause}
        RETURN p.id AS id, 1.0 AS score
        """,
        **params,
    )
    for r in exact or []:
        results[r["id"]] = 1.0

    # Strategy 2: Suffix match — query is a partial path (no IDS prefix)
    suffix = gc.query(
        f"""
        MATCH (p:IMASNode)
        WHERE p.node_category = 'data'
          AND toLower(p.id) ENDS WITH $search_query
          {ids_clause} {dd_clause}
        RETURN p.id AS id, 0.95 AS score
        LIMIT $limit
        """,
        **params,
    )
    for r in suffix or []:
        if r["id"] not in results:
            results[r["id"]] = r["score"]

    # Strategy 3: Contains match
    if len(results) < max_results:
        contains = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.node_category = 'data'
              AND toLower(p.id) CONTAINS $search_query
              {ids_clause} {dd_clause}
            RETURN p.id AS id, 0.80 AS score
            LIMIT $limit
            """,
            **params,
        )
        for r in contains or []:
            if r["id"] not in results:
                results[r["id"]] = r["score"]

    # Strategy 4: Accessor-stripped match
    stripped = strip_accessor_suffix(query_lower)
    if stripped != query_lower and len(results) < max_results:
        params["stripped"] = stripped
        accessor_match = gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.node_category = 'data'
              AND toLower(p.id) CONTAINS $stripped
              {ids_clause} {dd_clause}
            RETURN p.id AS id, 0.75 AS score
            LIMIT $limit
            """,
            **params,
        )
        for r in accessor_match or []:
            if r["id"] not in results:
                results[r["id"]] = r["score"]

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [{"id": pid, "score": score} for pid, score in sorted_results[:max_results]]


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
        dd_version: int | str | None = None,
        facility: str | None = None,
        include_version_context: bool = False,
        ctx: Context | None = None,
    ) -> SearchPathsResult:
        """Search IMAS paths using hybrid vector + text search."""
        dd_version = resolve_dd_version(dd_version)
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

        # Analyze query for path detection and abbreviation expansion
        analyzer = QueryAnalyzer()
        intent = analyzer.analyze(query)

        # Route path-like queries to structural search
        vector_results: list[dict[str, Any]] | None = None
        text_results: list[dict[str, Any]] = []
        if intent.query_type in ("path_exact", "path_partial"):
            path_results = _path_search(
                self._gc, query, max_results, ids_filter, dd_version=dd_version
            )
            if path_results:
                sorted_ids = [r["id"] for r in path_results]
                scores = {r["id"]: r["score"] for r in path_results}
                # Fall through to enrichment below
            else:
                scores = {}
                sorted_ids = []
        else:
            # Expand abbreviations for better recall
            expanded = (
                " ".join(intent.expanded_terms) if intent.expanded_terms else query
            )
            search_query = expanded

            normalized_filter = normalize_ids_filter(ids_filter)
            embedding = self._embed_query(search_query)

            # --- Vector search ---
            filter_clause = ""
            params: dict[str, Any] = {
                "embedding": embedding,
                "k": min(max_results * 5, 500),
                "vector_limit": min(max_results * 5, 500),
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

            # --- Text search ---
            text_results = _text_search_imas_paths(
                self._gc,
                search_query,
                min(max_results * 5, 500),
                normalized_filter,
                dd_version=dd_version,
            )

            # --- Score fusion (Reciprocal Rank Fusion) ---
            # RRF is rank-based and immune to score normalization issues.
            # Normalized to [0, 1] so additive boosts are proportional.
            scores = _reciprocal_rank_fusion(vector_results or [], text_results, k=60)
            if scores:
                max_rrf = max(scores.values())
                if max_rrf > 0:
                    scores = {pid: s / max_rrf for pid, s in scores.items()}

            # --- Path segment boost ---
            # Conservative: these help when query terms legitimately appear
            # in path names, but must not overwhelm embedding scores.
            query_words = [
                w.lower()
                for w in search_query.split()
                if len(w) > 2 or w.lower() in _PHYSICS_SHORT_TERMS
            ]
            if query_words:
                for pid in scores:
                    segments = pid.lower().split("/")
                    match_count = sum(
                        1 for w in query_words if any(w in seg for seg in segments)
                    )
                    if match_count > 0:
                        scores[pid] = round(scores[pid] + 0.03 * match_count, 4)

                    # Exact terminal segment bonus
                    terminal = segments[-1] if segments else ""
                    if any(w == terminal for w in query_words):
                        scores[pid] = round(scores[pid] + 0.08, 4)

                    # IDS name bonus
                    ids_name = segments[0] if segments else ""
                    if any(w == ids_name for w in query_words):
                        scores[pid] = round(scores[pid] + 0.05, 4)

            # --- Abbreviation exact-match boost ---
            # When the query is a known physics abbreviation (e.g. "ip",
            # "q", "b0"), the expanded search ("plasma current ip") pulls
            # in many semantically similar but wrong paths.  Give a strong
            # extra boost to paths whose terminal segment exactly matches a
            # word from the *original* (pre-expansion) query so that
            # .../ip ranks above .../bootstrap_current, etc.
            if intent.is_abbreviation:
                _orig_terms = {w.lower() for w in intent.original_query.split()}
                for pid in scores:
                    terminal = pid.rsplit("/", 1)[-1].lower()
                    if terminal in _orig_terms:
                        scores[pid] = round(scores[pid] + 0.15, 4)

            # --- Graph-native boosts ---
            scores = _apply_cluster_boost(self._gc, scores)
            scores = _apply_hierarchy_boost(self._gc, scores)
            scores = _apply_coordinate_boost(self._gc, scores)

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

        # --- Children preview for structure nodes ---
        children_data: dict[str, dict[str, Any]] = {}
        if sorted_ids:
            children_data = _enrich_children(self._gc, sorted_ids)

        # --- Search channel provenance ---
        provenance: dict[str, dict[str, Any]] = {}
        for vrank, vr in enumerate(vector_results or []):
            provenance[vr["id"]] = {"vector_rank": vrank + 1}
        for trank, tr in enumerate(text_results):
            pid = tr["id"]
            if pid in provenance:
                provenance[pid]["text_rank"] = trank + 1
            else:
                provenance[pid] = {"text_rank": trank + 1}

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
            cd = children_data.get(pid)
            hit_children = cd["children"] if cd else None
            hit_children_total = cd["total"] if cd else None
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
                    children=hit_children,
                    children_total=hit_children_total,
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
                "provenance": provenance,
            },
            query=query,
            search_mode=mode,
            physics_domains=sorted(physics_domains),
        )

    def _embed_query(self, query: str) -> list[float]:
        """Embed query text using the module-level Encoder singleton."""
        encoder = _get_encoder()
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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> CheckPathsResult:
        """Validate IMAS paths against graph."""
        dd_version = resolve_dd_version(dd_version)
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

        # Pre-fetch valid IDS and paths for fuzzy matching (lazy, only on first miss)
        _fuzzy_ids: list[str] | None = None
        _fuzzy_paths: list[str] | None = None

        def _get_fuzzy_data():
            nonlocal _fuzzy_ids, _fuzzy_paths
            if _fuzzy_ids is None:
                _fuzzy_ids = [
                    r["id"]
                    for r in self._gc.query("MATCH (i:IDS) RETURN i.id AS id") or []
                ]
                _fuzzy_paths = [
                    r["id"]
                    for r in self._gc.query(
                        "MATCH (p:IMASNode) WHERE p.node_category = 'data' "
                        "RETURN p.id AS id LIMIT 10000"
                    )
                    or []
                ]
            return _fuzzy_ids, _fuzzy_paths

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
                    new_path = renamed[0]["new_path"]
                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                            renamed_from=[
                                {
                                    "old_path": renamed[0]["old_path"],
                                    "new_path": new_path,
                                }
                            ],
                            migration={"type": "renamed", "target": new_path},
                            suggestion=new_path,
                        )
                    )
                else:
                    # Try fuzzy matching for typo correction
                    suggestion = None
                    try:
                        valid_ids, valid_paths = _get_fuzzy_data()
                        suggestion = suggest_correction(path, valid_ids, valid_paths)
                    except Exception:
                        pass
                    results.append(
                        CheckPathsResultItem(
                            path=path,
                            exists=False,
                            suggestion=suggestion,
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
        dd_version: int | str | None = None,
        include_version_history: bool = False,
        ctx: Context | None = None,
    ) -> FetchPathsResult:
        """Fetch detailed path information from graph."""
        dd_version = resolve_dd_version(dd_version)
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
        dd_version: int | str | None = None,
        response_profile: str = "minimal",
        ctx: Context | None = None,
    ) -> ListPathsResult:
        """List paths from graph."""
        dd_version = resolve_dd_version(dd_version)
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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> GetOverviewResult:
        """Get overview from graph."""
        dd_version = resolve_dd_version(dd_version)
        import importlib.metadata

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        # Query IDS nodes from graph
        ids_results = self._gc.query(
            f"""
            MATCH (i:IDS)
            OPTIONAL MATCH (i)<-[:IN_IDS]-(p:IMASNode)
            WHERE true {dd_clause}
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

        # Build tools list (query_imas_graph and get_dd_graph_schema were removed)
        mcp_tools = [
            "search_imas_paths",
            "check_imas_paths",
            "fetch_imas_paths",
            "list_imas_paths",
            "get_imas_overview",
            "search_imas_clusters",
            "get_imas_identifiers",
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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search clusters using graph vector indexes."""
        dd_version = resolve_dd_version(dd_version)
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
            WITH c, collect(DISTINCT p.id)[..50] AS section_paths,
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
            {scope_filter}
            OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(c)
            WHERE true {dd_clause}
            WITH c, collect(DISTINCT member.id)[..50] AS paths
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
        vector_k = 50 if ids_filter else 10  # More candidates when post-filtering
        params: dict[str, Any] = {"embedding": embedding, "k": vector_k}
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
            WITH cluster, score, collect(DISTINCT member.id)[..50] AS paths
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

        # Re-rank clusters: penalize overly broad, boost label relevance
        query_lower = query.lower()
        for cluster in clusters:
            score = cluster.get("relevance_score", 0.0) or 0.0
            path_count = cluster.get("total_paths", len(cluster.get("paths", [])))
            # Breadth penalty: very large clusters are less specific
            if path_count > 50:
                score *= 0.85
            elif path_count > 30:
                score *= 0.92
            # Label relevance boost: cluster label contains query terms
            label = (cluster.get("label") or "").lower()
            if label and any(term in label for term in query_lower.split()):
                score += 0.08
            cluster["relevance_score"] = round(score, 4)
        clusters.sort(key=lambda c: c.get("relevance_score", 0), reverse=True)

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
        """Embed query text using the module-level Encoder singleton."""
        encoder = _get_encoder()
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
        dd_version: int | str | None = None,
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
        "Get structural context for an IMAS path via graph traversal. "
        "Discovers sibling paths via shared clusters, coordinates, units, "
        "and identifier schemas across IDS boundaries. "
        "path (required): Exact IMAS path (e.g. 'equilibrium/time_slice/profiles_1d/psi'). "
        "relationship_types: Filter to specific types — 'cluster', 'coordinate', "
        "'unit', 'identifier', or 'all' (default)."
    )
    @handle_errors("get_imas_path_context")
    async def get_imas_path_context(
        self,
        path: str,
        relationship_types: str = "all",
        max_results: int = 20,
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Discover cross-IDS relationships for an IMAS path."""
        dd_version = resolve_dd_version(dd_version)
        dd_params: dict[str, Any] = {"path": path}
        dd_clause = _dd_version_clause("sibling", dd_version, dd_params)
        sections: dict[str, list[dict[str, Any]]] = {}

        # Cluster siblings — paths in same cluster but different IDS
        if relationship_types in ("all", "cluster"):
            cluster_siblings = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
                      <-[:IN_CLUSTER]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause}
                RETURN cl.label AS cluster, sibling.id AS path,
                       sibling.ids AS ids, sibling.documentation AS doc
                ORDER BY cl.label, sibling.ids
                LIMIT $limit
                """,
                **dd_params,
                limit=max_results,
            )
            if cluster_siblings:
                sections["cluster_siblings"] = cluster_siblings

        # Coordinate partners — paths sharing coordinate spec
        if relationship_types in ("all", "coordinate"):
            coord_partners = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_COORDINATE]->(coord:IMASNode)
                      <-[:HAS_COORDINATE]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause}
                RETURN coord.id AS coordinate, sibling.id AS path,
                       sibling.ids AS ids, sibling.data_type AS data_type
                ORDER BY coord.id, sibling.ids
                LIMIT $limit
                """,
                **dd_params,
                limit=max_results,
            )
            if coord_partners:
                # Filter out generic coordinates like "1...N"
                filtered = [
                    cp
                    for cp in coord_partners
                    if not (cp.get("coordinate") or "").startswith("1...")
                ]
                if filtered:
                    sections["coordinate_partners"] = filtered

        # Unit companions — paths with same unit in same physics domain
        if relationship_types in ("all", "unit"):
            unit_companions = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_UNIT]->(u:Unit)
                      <-[:HAS_UNIT]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids
                  AND sibling.physics_domain = p.physics_domain {dd_clause}
                RETURN u.id AS unit, sibling.id AS path,
                       sibling.ids AS ids, sibling.documentation AS doc
                ORDER BY u.id, sibling.ids
                LIMIT $limit
                """,
                **dd_params,
                limit=max_results,
            )
            if unit_companions:
                sections["unit_companions"] = unit_companions

        # Identifier schema links
        if relationship_types in ("all", "identifier"):
            ident_links = self._gc.query(
                f"""
                MATCH (p:IMASNode {{id: $path}})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
                      <-[:HAS_IDENTIFIER_SCHEMA]-(sibling:IMASNode)
                WHERE sibling.ids <> p.ids {dd_clause}
                RETURN s.name AS schema, sibling.id AS path,
                       sibling.ids AS ids
                ORDER BY s.name
                LIMIT $limit
                """,
                **dd_params,
                limit=max_results,
            )
            if ident_links:
                sections["identifier_links"] = ident_links

        # Semantic vector search — find paths with similar descriptions
        if relationship_types in ("all", "semantic"):
            try:
                # Get the source node's embedding
                src = self._gc.query(
                    """
                    MATCH (p:IMASNode {id: $path})
                    WHERE p.embedding IS NOT NULL
                    RETURN p.embedding AS emb, p.ids AS ids
                    """,
                    path=path,
                )
                if src and src[0].get("emb"):
                    source_ids = src[0]["ids"]
                    sem_results = self._gc.query(
                        f"""
                        CALL db.index.vector.queryNodes(
                            'imas_node_embedding', $k, $emb
                        ) YIELD node AS sibling, score
                        WHERE sibling.ids <> $src_ids
                          AND sibling.id <> $path {dd_clause}
                        RETURN sibling.id AS path, sibling.ids AS ids,
                               sibling.documentation AS doc, score
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        emb=src[0]["emb"],
                        k=max_results * 2,
                        src_ids=source_ids,
                        limit=max_results,
                        **dd_params,
                    )
                    if sem_results:
                        sections["semantic_neighbors"] = sem_results
            except Exception:
                pass

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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Analyze the hierarchical structure of an IMAS IDS."""
        dd_version = resolve_dd_version(dd_version)
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
        result: dict[str, Any] = {
            "ids_name": ids_name,
            "dd_version": dd_version,
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

        # When version-filtered, add deprecated/renamed context for this IDS
        if dd_version is not None:
            dep_params: dict[str, Any] = {
                "ids_name": ids_name,
                "dd_major_version": dd_version,
            }
            deprecated = self._gc.query(
                """
                MATCH (p:IMASNode)-[:DEPRECATED_IN]->(dv:DDVersion)
                WHERE p.ids = $ids_name
                  AND toInteger(split(dv.id, '.')[0]) <= $dd_major_version
                RETURN count(p) AS count
                """,
                **dep_params,
            )
            renamed = self._gc.query(
                """
                MATCH (old:IMASNode)-[:RENAMED_TO]->(new:IMASNode)
                WHERE old.ids = $ids_name
                RETURN count(old) AS count
                """,
                ids_name=ids_name,
            )
            result["version_context"] = {
                "note": (
                    f"Filtered to paths active in DD v{dd_version}. "
                    f"Counts include paths carried forward from earlier major versions."
                ),
                "deprecated_in_or_before": deprecated[0]["count"] if deprecated else 0,
                "renamed_paths": renamed[0]["count"] if renamed else 0,
            }

        return result

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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Export full IDS structure with documentation, units, and types."""
        dd_version = resolve_dd_version(dd_version)
        dd_params: dict[str, Any] = {"ids_name": ids_name}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        leaf_filter = f"AND {_leaf_data_type_clause('p')}" if leaf_only else ""

        paths = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name {leaf_filter} {dd_clause}
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
        dd_version: int | str | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Export all IMAS paths in a physics domain, grouped by IDS."""
        dd_version = resolve_dd_version(dd_version)
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
            WHERE p.physics_domain IN $domains {ids_clause} {dd_clause}
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


class GraphExplainTool:
    """Graph-backed concept explanations using cluster, COCOS, and identifier data."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "explain_concept"

    @measure_performance(include_metrics=True, slow_threshold=1.0)
    @handle_errors(fallback="explain_error")
    @mcp_tool(
        "Provide detailed explanations of fusion physics concepts and IMAS terminology. "
        "concept (required): The concept to explain (e.g. 'COCOS', 'safety factor', 'bootstrap current'). "
        "detail_level: 'basic', 'intermediate' (default), or 'advanced'."
    )
    async def explain_concept(
        self,
        concept: str,
        detail_level: str = "intermediate",
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Explain a concept using graph-backed data (pure Cypher, no embeddings)."""
        sections: list[dict[str, Any]] = []
        concept_lower = concept.lower().strip()

        # 1. Search semantic clusters by text matching on label/description
        cluster_results = self._gc.query(
            """
            MATCH (cluster:IMASSemanticCluster)
            WHERE toLower(cluster.label) CONTAINS $concept
               OR toLower(cluster.description) CONTAINS $concept
            OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(cluster)
            WITH cluster,
                 CASE WHEN toLower(cluster.label) CONTAINS $concept
                      THEN 0 ELSE 1 END AS rank,
                 collect(DISTINCT member.id)[..15] AS paths
            RETURN cluster.id AS id, cluster.label AS label,
                   cluster.description AS description,
                   cluster.scope AS scope,
                   cluster.ids_names AS ids_names,
                   paths
            ORDER BY rank, cluster.label
            LIMIT 8
            """,
            concept=concept_lower,
        )
        if cluster_results:
            sections.append(
                {
                    "type": "clusters",
                    "title": "Related IMAS Concepts",
                    "clusters": [
                        {
                            "label": r["label"],
                            "description": r["description"],
                            "scope": r["scope"],
                            "ids": r.get("ids_names", []),
                            "example_paths": r["paths"][:5]
                            if detail_level == "basic"
                            else r["paths"][:10],
                        }
                        for r in cluster_results[:5]
                    ],
                }
            )

        # 2. Check COCOS metadata if concept is COCOS-related
        cocos_terms = {
            "cocos",
            "sign convention",
            "coordinate convention",
            "orientation",
            "toroidal",
            "poloidal",
            "flux",
            "psi",
            "q",
        }
        if any(t in concept_lower for t in cocos_terms):
            cocos_results = self._gc.query("""
                MATCH (v:DDVersion)
                WHERE v.cocos IS NOT NULL
                RETURN v.version AS version, v.cocos AS cocos_id
                ORDER BY v.version DESC
                LIMIT 3
            """)
            if cocos_results:
                sections.append(
                    {
                        "type": "cocos",
                        "title": "COCOS (COordinate COnventionS)",
                        "versions": [
                            {"version": r["version"], "cocos_id": r["cocos_id"]}
                            for r in cocos_results
                        ],
                    }
                )

            # Find COCOS-affected paths
            cocos_paths = self._gc.query("""
                MATCH (change:IMASNodeChange)-[:FOR_IMAS_PATH]->(p:IMASNode)
                WHERE change.semantic_change_type = 'sign_convention'
                RETURN DISTINCT p.id AS path, p.ids AS ids,
                       change.summary AS summary
                ORDER BY p.id
                LIMIT 15
            """)
            if cocos_paths:
                sections.append(
                    {
                        "type": "cocos_paths",
                        "title": "COCOS-Affected Paths",
                        "paths": [
                            {
                                "path": r["path"],
                                "ids": r["ids"],
                                "summary": r.get("summary"),
                            }
                            for r in cocos_paths
                        ],
                    }
                )

        # 3. Search identifier schemas for enumeration explanations
        identifier_results = self._gc.query(
            """
            MATCH (schema)-[:HAS_OPTION]->(opt)
            WHERE toLower(schema.id) CONTAINS $concept
               OR toLower(schema.description) CONTAINS $concept
            RETURN DISTINCT schema.id AS schema_id,
                   schema.description AS schema_desc,
                   collect({name: opt.name, index: opt.index,
                           description: opt.description})[..20] AS options
            LIMIT 5
            """,
            concept=concept_lower,
        )
        if identifier_results:
            sections.append(
                {
                    "type": "identifiers",
                    "title": "Identifier Schemas",
                    "schemas": [
                        {
                            "id": r["schema_id"],
                            "description": r["schema_desc"],
                            "options": r["options"][:10]
                            if detail_level == "basic"
                            else r["options"],
                        }
                        for r in identifier_results
                    ],
                }
            )

        # 4. Find relevant IDS descriptions
        ids_results = self._gc.query(
            """
            MATCH (i:IDS)
            WHERE toLower(i.description) CONTAINS $concept
               OR toLower(i.name) CONTAINS $concept
            RETURN i.name AS name, i.description AS description,
                   i.physics_domain AS physics_domain
            ORDER BY i.name
            LIMIT 5
            """,
            concept=concept_lower,
        )
        if ids_results:
            sections.append(
                {
                    "type": "ids",
                    "title": "Related IDSs",
                    "ids_list": [
                        {
                            "name": r["name"],
                            "description": r["description"],
                            "physics_domain": r.get("physics_domain"),
                        }
                        for r in ids_results
                    ],
                }
            )

        # 5. Search path documentation for concept mentions
        doc_results = self._gc.query(
            """
            MATCH (p:IMASNode)
            WHERE p.node_category = 'data'
              AND (toLower(p.documentation) CONTAINS $concept
                   OR toLower(p.id) CONTAINS $concept)
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            RETURN p.id AS path, p.documentation AS documentation,
                   p.data_type AS data_type, u.symbol AS units
            ORDER BY p.id
            LIMIT 5
            """,
            concept=concept_lower,
        )
        if doc_results:
            sections.append(
                {
                    "type": "paths",
                    "title": "Related Data Paths",
                    "paths": [
                        {
                            "path": r["path"],
                            "documentation": r["documentation"],
                            "data_type": r.get("data_type"),
                            "units": r.get("units"),
                        }
                        for r in doc_results
                    ],
                }
            )

        return {
            "concept": concept,
            "detail_level": detail_level,
            "sections": sections,
            "section_count": len(sections),
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


_LUCENE_SPECIAL_RE = re.compile(r'([+\-&|!(){}[\]^"~*?:\\/])')


def _escape_lucene(term: str) -> str:
    """Escape Lucene special characters in a search term."""
    return _LUCENE_SPECIAL_RE.sub(r"\\\1", term)


def _build_lucene_query(query: str) -> str:
    """Build a Lucene query with field boosting and fuzzy matching.

    Uses field boosting to weight description > keywords > name > documentation,
    and adds fuzzy variants for typo tolerance on terms > 3 chars.
    """
    terms = query.split()
    parts = []
    for term in terms:
        escaped = _escape_lucene(term)
        parts.append(
            f"(description:{escaped}^3 OR name:{escaped}^2 "
            f"OR documentation:{escaped} OR keywords:{escaped}^2)"
        )
    base = " AND ".join(parts)
    fuzzy_terms = [f"{_escape_lucene(t)}~1" for t in terms if len(t) > 3]
    if fuzzy_terms:
        fuzzy = " OR ".join(fuzzy_terms)
        return f"({base}) OR ({fuzzy})"
    return base


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
    query_words = [
        w for w in query_lower.split() if len(w) > 2 or w in _PHYSICS_SHORT_TERMS
    ]

    where_parts = ["NOT (p)-[:DEPRECATED_IN]->(:DDVersion)", "p.node_category = 'data'"]
    # Cap CONTAINS fallback to avoid full scans on large graphs
    contains_limit = min(limit, 100)
    params: dict[str, Any] = {"query_lower": query_lower, "limit": contains_limit}

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
        ft_where = (
            "WHERE NOT (p)-[:DEPRECATED_IN]->(:DDVersion) AND p.node_category = 'data'"
        )
        ft_params: dict[str, Any] = {
            "ft_query": query,
            "limit": limit,
        }
        if ids_filter is not None:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            ft_where += " AND p.ids IN $ids_filter"
            ft_params["ids_filter"] = filter_list

        ft_dd_clause = _dd_version_clause("p", dd_version, ft_params)
        if ft_dd_clause:
            ft_where += f" {ft_dd_clause}"

        ft_cypher = f"""
            CALL db.index.fulltext.queryNodes('imas_node_text', $ft_query)
            YIELD node AS p, score
            {ft_where}
            WITH p, score
            WHERE (p.description IS NOT NULL OR size(coalesce(p.documentation, '')) > 0)
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
                normalized.append({"id": pid, "score": raw})
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
          AND (p.description IS NOT NULL OR size(coalesce(p.documentation, '')) > 0)
        WITH p,
             CASE
            WHEN toLower(p.documentation) CONTAINS $query_lower
                AND {_leaf_data_type_clause("p")}
                 THEN 0.95
               WHEN toLower(p.documentation) CONTAINS $query_lower
                 THEN 0.88
               WHEN toLower(p.name) CONTAINS $query_lower
                AND {_leaf_data_type_clause("p")}
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
                                    AND {_leaf_data_type_clause("p")}
                  AND (p.description IS NOT NULL OR size(coalesce(p.documentation, '')) > 0)
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


def _enrich_children(gc: GraphClient, path_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch immediate children for structure nodes.

    Returns a dict keyed by path ID with ``{"children": [...], "total": N}``
    for each STRUCTURE / STRUCT_ARRAY node in *path_ids*.  Non-structure
    nodes are omitted from the result.
    """
    result = gc.query(
        """
        UNWIND $path_ids AS pid
        MATCH (p:IMASNode {id: pid})
        WHERE p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(p)
        WHERE child.data_type IS NOT NULL
            AND NOT (child.name ENDS WITH '_error_upper'
                 OR child.name ENDS WITH '_error_lower'
                 OR child.name ENDS WITH '_error_index')
        OPTIONAL MATCH (child)-[:HAS_UNIT]->(u:Unit)
        WITH pid, child, u
        ORDER BY child.name
        WITH pid, collect(DISTINCT {
            name: child.name,
            data_type: child.data_type,
            unit: u.id,
            description: left(coalesce(child.description, ''), 80)
        })[..10] AS children,
        size(collect(DISTINCT child)) AS total_children
        RETURN pid AS id, children, total_children
        """,
        path_ids=path_ids,
    )

    return {
        row["id"]: {"children": row["children"], "total": row["total_children"]}
        for row in result
    }


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
                        ann = pv.annotations.get("category") if pv.annotations else None
                        cat = ann.value if ann else ""
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
