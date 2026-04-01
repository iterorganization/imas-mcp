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

# Child role classification for concept node children
CHILD_ROLE_MAP: dict[str, str] = {
    # Data containers
    "value": "data",
    "data": "data",
    "values": "data",
    # Time bases
    "time": "time",
    # Geometric coordinates
    "r": "coordinates",
    "z": "coordinates",
    "phi": "coordinates",
    "x": "coordinates",
    "y": "coordinates",
    # Directional components
    "parallel": "components",
    "poloidal": "components",
    "radial": "components",
    "toroidal": "components",
    "diamagnetic": "components",
    # Interpolation
    "coefficients": "interpolation",
    # Grid references
    "grid_index": "grid",
    "grid_subset_index": "grid",
    "index": "grid",
    # Quality indicators
    "validity": "quality",
    "validity_timed": "quality",
    "chi_squared": "quality",
    # Fit results
    "measured": "fit",
    "reconstructed": "fit",
    # Labels
    "label": "metadata",
}


def _classify_child_role(name: str) -> str:
    """Classify a child node into a role category."""
    if name in CHILD_ROLE_MAP:
        return CHILD_ROLE_MAP[name]
    if name.endswith("_coefficients"):
        return "interpolation"
    if name.endswith("_n"):
        return "normalized"
    if any(name.endswith(s) for s in ("_error_upper", "_error_lower", "_error_index")):
        return "error"
    if name.endswith(("_validity", "_validity_timed")):
        return "quality"
    return "other"


def _fetch_and_group_children(
    gc: "GraphClient", concept_ids: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Fetch and group children for structure concept nodes.

    Returns dict mapping parent_id -> list of grouped children.
    """
    if not concept_ids:
        return {}

    children_results = gc.query(
        """
        UNWIND $concept_ids AS pid
        MATCH (concept:IMASNode {id: pid})
        WHERE concept.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        OPTIONAL MATCH (child:IMASNode {node_category: 'data'})-[:HAS_PARENT]->(concept)
        WITH pid, collect({
            name: child.name,
            id: child.id,
            data_type: child.data_type,
            documentation: child.documentation
        }) AS children
        WHERE size(children) > 0 AND children[0].name IS NOT NULL
        RETURN pid AS parent_id, children
        """,
        concept_ids=concept_ids,
    )

    result = {}
    for row in children_results:
        pid = row["parent_id"]
        children = row["children"]
        # Group by role
        grouped: dict[str, list[dict]] = {}
        for child in children:
            if child.get("name") is None:
                continue
            role = _classify_child_role(child["name"])
            if role not in grouped:
                grouped[role] = []
            grouped[role].append(
                {
                    "name": child["name"],
                    "data_type": child.get("data_type"),
                }
            )
        # Sort each group by name
        for role in grouped:
            grouped[role].sort(key=lambda c: c["name"])
        result[pid] = [
            {"role": role, "children": children}
            for role, children in sorted(grouped.items())
        ]

    return result


# Child name synonyms for common accessor terminals
CHILD_SYNONYMS: dict[str, set[str]] = {
    "r": {"radius", "major_radius", "radial"},
    "z": {"height", "vertical", "elevation"},
    "phi": {"toroidal_angle", "azimuthal"},
    "time": {"timebase", "temporal"},
    "measured": {"measurement"},
    "reconstructed": {"reconstruction"},
    "parallel": {"parallel_component"},
    "poloidal": {"poloidal_component"},
    "radial": {"radial_component"},
    "toroidal": {"toroidal_component"},
}


def reciprocal_rank_fusion(
    vector_hits: list[dict], text_hits: list[dict], k: int = 60
) -> dict[str, float]:
    """Combine ranked lists using Reciprocal Rank Fusion.

    Score-scale invariant: uses ranks, not scores. Eliminates the problem
    where BM25 scores (0.7-0.95) drown vector scores (0.82-0.88).
    """
    rrf_scores: dict[str, float] = {}
    for rank, hit in enumerate(
        sorted(vector_hits, key=lambda x: x["score"], reverse=True), start=1
    ):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + 1 / (k + rank)
    for rank, hit in enumerate(
        sorted(text_hits, key=lambda x: x["score"], reverse=True), start=1
    ):
        rrf_scores[hit["id"]] = rrf_scores.get(hit["id"], 0) + 1 / (k + rank)
    return rrf_scores


def heuristic_rerank(
    scores: dict[str, float], query: str, metadata: dict[str, dict] | None = None
) -> dict[str, float]:
    """Zero-cost reranking based on metadata signals.

    Applies small boosts for IDS name match, exact path segment match,
    and well-documented paths.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    boosted = dict(scores)

    for pid in boosted:
        segments = pid.lower().split("/")
        # IDS name appears in query
        if segments and segments[0] in query_lower:
            boosted[pid] += 0.02

        # Exact path segment match
        for seg in segments:
            if seg.replace("_", " ") in query_lower or seg in query_words:
                boosted[pid] += 0.01
                break  # Only one segment boost per path

        # Well-documented paths get a small boost
        if metadata and pid in metadata:
            doc_len = len(metadata[pid].get("documentation", "") or "")
            if doc_len > 100:
                boosted[pid] += 0.01

    return boosted


# Vector confidence gate: below this threshold, vector results are noise
VECTOR_GATE_THRESHOLD = 0.65


def _boost_by_child_match(
    scores: dict[str, float], query: str, gc: "GraphClient"
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Boost concept nodes whose children match query terms.

    Returns updated scores and a map of parent_id -> matched child names.
    """
    query_words = {w.lower() for w in query.split() if len(w) > 2}
    matched_children_map: dict[str, list[str]] = {}
    if not query_words:
        return scores, matched_children_map

    path_ids = list(scores.keys())
    if not path_ids:
        return scores, matched_children_map

    children = gc.query(
        """
        UNWIND $parent_ids AS pid
        MATCH (child:IMASNode)-[:HAS_PARENT]->(parent:IMASNode {id: pid})
        WHERE child.node_category = 'data'
        RETURN pid AS parent_id, collect(child.name) AS child_names
        """,
        parent_ids=path_ids,
    )

    for row in children:
        pid = row["parent_id"]
        child_names = set(row["child_names"])
        # Expand child names with synonyms
        expanded: set[str] = set()
        for cn in child_names:
            expanded.add(cn)
            expanded.update(CHILD_SYNONYMS.get(cn, set()))

        matches = query_words & expanded
        if matches and pid in scores:
            scores[pid] = round(scores[pid] + 0.03 * len(matches), 4)
            # Map back to actual child names (not synonyms)
            actual_matches = []
            for cn in child_names:
                cn_expanded = {cn} | CHILD_SYNONYMS.get(cn, set())
                if cn_expanded & query_words:
                    actual_matches.append(cn)
            if actual_matches:
                matched_children_map[pid] = actual_matches

    return scores, matched_children_map


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


def _deprecated_filter(alias: str, dd_version: int | None) -> str:
    """Return the correct deprecated-path filter for the given version scope.

    When dd_version is None (default, current version): hard-exclude all
    deprecated paths.  When dd_version is specified: omit the hard filter
    entirely — ``_dd_version_clause`` handles version-scoped deprecation.
    This allows paths deprecated in later versions to appear when searching
    older DD versions (e.g., ``ece/channel/t_e`` deprecated in 4.0.0 is
    still valid when searching DD 3.x).
    """
    if dd_version is not None:
        return ""
    return f"NOT ({alias})-[:DEPRECATED_IN]->(:DDVersion)"


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

        # Path query short-circuit: if query contains '/', treat as path lookup
        if "/" in query:
            text_results = _text_search_imas_paths(
                self._gc,
                query,
                min(max_results * 3, 150),
                normalized_filter,
                dd_version=dd_version,
            )
            scores: dict[str, float] = {}
            for r in text_results:
                scores[r["id"]] = round(r["score"], 4)
        else:
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
            dep_filter = _deprecated_filter("path", dd_version)
            dep_clause = f"AND {dep_filter}" if dep_filter else ""

            vector_results = self._gc.query(
                f"""
                CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
                YIELD node AS path, score
                WHERE path.node_category = 'data'
                {dep_clause}
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
                query,
                min(max_results * 3, 150),
                normalized_filter,
                dd_version=dd_version,
            )

            # --- RRF merge with vector confidence gating ---
            v_hits = [
                {"id": r["id"], "score": r["score"]} for r in (vector_results or [])
            ]
            t_hits = [{"id": r["id"], "score": r["score"]} for r in text_results]

            best_vector = max((r["score"] for r in v_hits), default=0.0)
            if best_vector < VECTOR_GATE_THRESHOLD:
                # Low-confidence vector results: use text scores only
                scores: dict[str, float] = {
                    r["id"]: round(r["score"], 4) for r in t_hits
                }
            else:
                scores = {
                    pid: round(s, 6)
                    for pid, s in reciprocal_rank_fusion(v_hits, t_hits, k=60).items()
                }

            # --- Heuristic reranking ---
            scores = heuristic_rerank(scores, query)

        # --- Child name matching boost (accessor query routing) ---
        matched_children_map: dict[str, list[str]] = {}
        if scores:
            scores, matched_children_map = _boost_by_child_match(
                scores, query, self._gc
            )

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

        # --- Child traversal for structure nodes ---
        children_map: dict[str, list[dict[str, Any]]] = {}
        structure_ids = [
            r["id"]
            for r in (enriched or [])
            if r.get("data_type") in ("STRUCTURE", "STRUCT_ARRAY")
        ]
        if structure_ids:
            children_map = _fetch_and_group_children(self._gc, structure_ids)

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
                    children=children_map.get(pid),
                    matched_children=matched_children_map.get(pid),
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
        """Embed query text, expanding physics abbreviations first."""
        expanded = _expand_abbreviations(query)
        encoder = _get_encoder()
        return encoder.embed_texts([expanded])[0].tolist()


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
        """Embed query text, expanding physics abbreviations first."""
        expanded = _expand_abbreviations(query)
        encoder = _get_encoder()
        return encoder.embed_texts([expanded])[0].tolist()


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
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Discover cross-IDS relationships for an IMAS path."""
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
                MATCH (p:IMASNode {{id: $path}})-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
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
                sections["coordinate_partners"] = coord_partners

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


# Physics abbreviation/synonym expansion for BM25 query augmentation
PHYSICS_ABBREVIATIONS: dict[str, str] = {
    "te": "electron temperature t_e",
    "ti": "ion temperature t_i",
    "ne": "electron density n_e",
    "ni": "ion density n_i",
    "ip": "plasma current ip",
    "bt": "toroidal magnetic field b_field_tor",
    "bp": "poloidal magnetic field b_field_pol",
    "vp": "loop voltage v_loop",
    "beta_pol": "poloidal beta beta_pol",
    "beta_tor": "toroidal beta beta_tor",
    "beta_n": "normalized beta beta_normal",
    "bpol": "poloidal beta beta_pol",
    "btor": "toroidal field b_field_tor",
    "q95": "safety factor q_95",
    "q": "safety factor q profile",
    "zeff": "effective charge zeff impurity",
    "z_eff": "effective charge zeff impurity",
    "li": "internal inductance li",
    "wmhd": "stored energy diamagnetic w_mhd",
    "nbi": "neutral beam injection nbi",
    "icrh": "ion cyclotron resonance heating ic_antennas",
    "ecrh": "electron cyclotron resonance heating ec_launchers",
    "ece": "electron cyclotron emission ece",
    "cxrs": "charge exchange spectroscopy charge_exchange",
    "lcfs": "last closed flux surface boundary separatrix",
    "psi": "poloidal flux psi magnetic",
    "rho": "normalized toroidal flux rho_tor_norm",
    "b0": "toroidal magnetic field vacuum b0",
}


def _expand_abbreviations(query: str) -> str:
    """Expand physics abbreviations in query for better BM25 matching."""
    query_lower = query.lower().strip()
    words = query_lower.split()

    # Single-word abbreviation: expand it
    if len(words) == 1 and query_lower in PHYSICS_ABBREVIATIONS:
        return PHYSICS_ABBREVIATIONS[query_lower]

    # Multi-word: expand individual abbreviation words
    expanded = []
    for w in words:
        if w in PHYSICS_ABBREVIATIONS:
            expanded.append(PHYSICS_ABBREVIATIONS[w])
        else:
            expanded.append(w)
    result = " ".join(expanded)
    return result if result != query_lower else query


def _rerank_bm25_results(
    results: list[dict[str, Any]], query: str, gc: "GraphClient"
) -> list[dict[str, Any]]:
    """Post-BM25 re-ranking using documentation and keyword signals.

    BM25 treats all indexed fields equally, so a node named 'current'
    scores higher than one documented as 'Plasma current'. This function
    boosts results where query terms appear in documentation or keywords.
    """
    if not results:
        return results

    query_lower = query.lower()
    query_words = {w.lower() for w in query.split() if len(w) > 1}

    path_ids = [r["id"] for r in results[:100]]
    metadata = gc.query(
        """
        UNWIND $pids AS pid
        MATCH (p:IMASNode {id: pid})
        RETURN p.id AS id, p.documentation AS doc, p.description AS desc,
               p.keywords AS kw, p.name AS name, p.ids AS ids
        """,
        pids=path_ids,
    )
    meta_by_id = {r["id"]: r for r in (metadata or [])}

    for r in results:
        pid = r["id"]
        m = meta_by_id.get(pid)
        if not m:
            continue

        boost = 0.0
        doc = (m.get("doc") or "").lower()
        desc = (m.get("desc") or "").lower()
        kws = [k.lower() for k in (m.get("kw") or [])]
        ids_name = (m.get("ids") or "").lower()

        # Strong boost: query appears as phrase in documentation/description
        if query_lower in doc or query_lower in desc:
            boost += 0.30

        # Medium boost: all query words present in doc+desc+keywords
        combined_text = f"{doc} {desc} {' '.join(kws)}"
        word_matches = sum(1 for w in query_words if w in combined_text)
        if query_words and word_matches == len(query_words):
            boost += 0.15

        # Keyword match boost
        for kw in kws:
            if kw in query_lower or any(w in kw for w in query_words):
                boost += 0.10
                break

        # Canonical IDS boost for concept queries
        canonical_ids = {
            "equilibrium",
            "core_profiles",
            "core_transport",
            "summary",
            "nbi",
            "ece",
            "magnetics",
        }
        if ids_name in canonical_ids:
            boost += 0.05

        r["score"] = r["score"] + boost

    results.sort(key=lambda r: r["score"], reverse=True)
    # Re-normalize to 0-1
    if results:
        max_score = max(r["score"] for r in results)
        if max_score > 0:
            for r in results:
                r["score"] = round(r["score"] / max_score, 4)
    return results


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
    CONTAINS matching. Applies abbreviation expansion and post-BM25
    re-ranking for improved precision.
    """
    # Expand physics abbreviations before search
    expanded_query = _expand_abbreviations(query)
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]

    # Build version-aware deprecated filter
    dep_filter = _deprecated_filter("p", dd_version)
    where_parts = [
        "p.node_category = 'data'",
        "(p.enrichment_source IS NULL OR p.enrichment_source <> 'template')",
    ]
    if dep_filter:
        where_parts.insert(0, dep_filter)
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
        ft_dep = f"AND {dep_filter} " if dep_filter else ""
        ft_where = (
            f"WHERE p.node_category = 'data' {ft_dep}"
            "AND (p.enrichment_source IS NULL OR p.enrichment_source <> 'template')"
        )
        # Use expanded query for BM25 (abbreviations → full terms)
        ft_params: dict[str, Any] = {"search_query": expanded_query, "limit": limit}
        if ids_filter is not None:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            ft_where += " AND p.ids IN $ids_filter"
            ft_params["ids_filter"] = filter_list

        ft_dd_clause = _dd_version_clause("p", dd_version, ft_params)
        if ft_dd_clause:
            ft_where += f" {ft_dd_clause}"

        ft_cypher = f"""
            CALL db.index.fulltext.queryNodes('imas_node_text', $search_query)
            YIELD node AS p, score
            {ft_where}
            WITH p, score
            WHERE size(coalesce(p.documentation, '')) > 10
                  OR p.description IS NOT NULL
            RETURN p.id AS id, score
            ORDER BY score DESC
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
            # Re-rank using documentation/keyword signals
            return _rerank_bm25_results(normalized, query, gc)
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
                AND {_leaf_data_type_clause("p")}
                 THEN 0.80
               WHEN toLower(p.name) CONTAINS $query_lower
                AND {_leaf_data_type_clause("p")}
                 THEN 0.75
               WHEN toLower(p.id) CONTAINS $query_lower
                 THEN 0.70
               WHEN toLower(p.documentation) CONTAINS $query_lower
                 THEN 0.65
               ELSE 0.55
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
                  AND size(coalesce(p.documentation, '')) > 10
                RETURN p.id AS id, 0.70 AS score
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
