"""Graph-backed tool implementations for graph-native MCP server.

Replaces DocumentStore, SemanticSearch, and ClusterSearcher with
GraphClient queries against Neo4j vector indexes and Cypher traversals.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal

from fastmcp import Context

from imas_codex.core.data_model import IdsNode
from imas_codex.core.node_categories import SEARCHABLE_CATEGORIES
from imas_codex.graph.client import GraphClient
from imas_codex.models.constants import SearchMode

if TYPE_CHECKING:
    from imas_codex.search.fuzzy_matcher import PathFuzzyMatcher
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

# Short physics abbreviations mapped to their expanded physics concepts.
# Keys survive word-length filtering (`len(w) > 2` guards).
# Values are used as vector-search queries — the embedding model maps
# "plasma current" correctly even though "ip" would map to "internet protocol".
_PHYSICS_SHORT_TERMS: dict[str, str] = {
    # Currents & fields
    "ip": "plasma current",
    "j": "current density",
    "jt": "toroidal current density",
    "bt": "toroidal magnetic field",
    "bp": "poloidal magnetic field",
    "b0": "vacuum toroidal magnetic field",
    # Kinetic profiles
    "te": "electron temperature",
    "ne": "electron density",
    "ti": "ion temperature",
    "ni": "ion density",
    "nz": "impurity density",
    "ze": "effective charge",
    # Global / geometry
    "li": "internal inductance",
    "q": "safety factor",
    "r0": "magnetic axis major radius",
    # Coils & heating
    "pf": "poloidal field coil",
    "tf": "toroidal field coil",
    "ec": "electron cyclotron",
    "ic": "ion cyclotron",
    "lh": "lower hybrid",
}

# Accessor terminals (data containers, time bases, validity flags) are
# rarely the user's intent for concept queries.  Apply a small penalty
# so the parent concept path ranks higher.
_ACCESSOR_TERMINALS: frozenset[str] = frozenset(
    {"data", "value", "time", "validity", "fit", "coefficients"}
)

# Soft IDS preferences for unqualified concept queries.
# When query terms overlap these keywords and the result lives in the
# preferred IDS, a small boost (3%) helps canonical paths rank first.
# Only active when no explicit ids_filter is provided.
_CONCEPT_IDS_PREFERENCE: dict[str, str] = {
    "temperature": "core_profiles",
    "density": "core_profiles",
    "psi": "equilibrium",
    "current": "equilibrium",
    "boundary": "equilibrium",
    "b_field": "magnetics",
}

# Lucene special characters that must be escaped in user queries
_LUCENE_SPECIAL = frozenset('+-&&||!(){}[]^"~*?:\\/')


def _escape_lucene(term: str) -> str:
    """Escape Lucene special characters in a search term."""
    return "".join(f"\\{c}" if c in _LUCENE_SPECIAL else c for c in term)


def _build_phrase_aware_query(query: str) -> str:
    """Build a Lucene query with adjacent-bigram phrases + OR fallback.

    For multi-word queries, creates bigram phrases from adjacent words
    and combines with individual terms using OR. Lucene naturally boosts
    phrase matches via position-aware scoring.

    Examples:
        "magnetic flux poloidal" → '"magnetic flux" OR "flux poloidal" OR magnetic OR flux OR poloidal'
        "electron temperature" → '"electron temperature" OR electron OR temperature'
        "ip" → 'ip'  (single word unchanged)
    """
    words = query.strip().split()
    if len(words) <= 1:
        return _escape_lucene(query.strip())

    escaped = [_escape_lucene(w) for w in words]
    parts = []

    # Adjacent bigram phrases (highest weight in Lucene scoring)
    for i in range(len(escaped) - 1):
        parts.append(f'"{escaped[i]} {escaped[i + 1]}"')

    # Individual terms as OR fallback
    parts.extend(escaped)

    return " OR ".join(parts)


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


def _resolve_node_categories(node_category: str | None) -> list[str]:
    """Resolve a node_category filter into a list of allowed categories.

    When *node_category* is ``None``, returns the full
    :data:`SEARCHABLE_CATEGORIES` set.  Otherwise, splits on commas and
    intersects with ``SEARCHABLE_CATEGORIES`` to prevent injection of
    non-searchable categories.
    """
    if node_category is None:
        return list(SEARCHABLE_CATEGORIES)
    requested = {c.strip() for c in node_category.split(",") if c.strip()}
    valid = requested & SEARCHABLE_CATEGORIES
    # Fall back to full set if the caller asked for categories that don't exist
    return list(valid) if valid else list(SEARCHABLE_CATEGORIES)


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


class GraphSearchTool:
    """Graph-backed semantic search for IMAS paths."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "search_dd_paths"

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
        "include_summary_ids: Include paths from the 'summary' IDS in results (default: false). "
        "The summary IDS contains aggregate quantities duplicated from primary IDSs. "
        "search_mode: 'auto' (default), 'semantic', 'lexical', or 'hybrid'. "
        "response_profile: 'minimal', 'standard' (default), or 'detailed'."
    )
    async def search_dd_paths(
        self,
        query: str,
        ids_filter: str | list[str] | None = None,
        max_results: int = 50,
        search_mode: str | SearchMode = "auto",
        response_profile: str = "standard",
        dd_version: int | None = None,
        facility: str | None = None,
        include_version_context: bool = False,
        include_summary_ids: bool = False,
        physics_domain: str | None = None,
        lifecycle_filter: str | None = None,
        node_category: str | None = None,
        ctx: Context | None = None,
    ) -> SearchPathsResult:
        """Search IMAS paths using hybrid vector + text search."""
        is_valid, error_message = validate_query(query, "search_dd_paths")
        if not is_valid:
            return SearchPathsResult(
                hits=[],
                summary={"error": "Query cannot be empty.", "query": query or ""},
                query=query or "",
                search_mode=SearchMode.AUTO,
                physics_domains=[],
                error="Query cannot be empty.",
            )

        from imas_codex.core.paths import normalize_imas_path

        # Normalize dot-notation and bracket notation before any analysis
        query = normalize_imas_path(query)

        normalized_filter = normalize_ids_filter(ids_filter)

        # Resolve effective node categories — restrict SEARCHABLE_CATEGORIES
        # when the caller asks for a specific node_category.
        effective_categories = _resolve_node_categories(node_category)

        # Determine whether to exclude summary IDS paths.
        # Only exclude when the user has not explicitly requested summary paths
        # via ids_filter and has not opted in via include_summary_ids=True.
        _filter_list = (
            normalized_filter
            if isinstance(normalized_filter, list)
            else ([normalized_filter] if normalized_filter else [])
        )
        _exclude_summary_ids = not include_summary_ids and "summary" not in _filter_list
        summary_clause = "AND path.ids <> 'summary'" if _exclude_summary_ids else ""

        # Path-like queries (contain "/") should skip expensive vector search
        # and rely on text/path matching instead.
        # Short physics abbreviations (e.g. "ip", "q", "b0") use their expanded
        # form for vector search: the embedding model maps "plasma current"
        # correctly even though "ip" would map to "internet protocol".
        is_path_query = "/" in query and " " not in query
        is_short_physics_term = " " not in query.strip() and (
            len(query.strip()) <= 3 or query.strip().lower() in _PHYSICS_SHORT_TERMS
        )
        if is_path_query:
            embedding = None
        elif is_short_physics_term:
            expansion = _PHYSICS_SHORT_TERMS.get(query.strip().lower())
            embedding = self._embed_query(expansion) if expansion else None
        else:
            embedding = self._embed_query(query)

        # --- Vector search ---
        vec_scores: dict[str, float] = {}
        if embedding is not None:
            filter_clause = ""
            params: dict[str, Any] = {
                "embedding": embedding,
                "k": min(max_results * 5, 500),
                "vector_limit": min(max_results * 3, 150),
                "categories": effective_categories,
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
                  AND path.node_category IN $categories
                {filter_clause}
                {summary_clause}
                {dd_clause}
                RETURN path.id AS id, score
                ORDER BY score DESC
                LIMIT $vector_limit
                """,
                **params,
            )

            for r in vector_results or []:
                pid = r["id"]
                vec_scores[pid] = round(r["score"], 4)

        # --- Text search ---
        text_results = _text_search_dd_paths(
            self._gc,
            query,
            min(max_results * 3, 150),
            normalized_filter,
            dd_version=dd_version,
            exclude_summary=_exclude_summary_ids,
            categories=effective_categories,
        )
        text_scores: dict[str, float] = {}
        for r in text_results:
            text_scores[r["id"]] = round(r["score"], 4)

        # --- Weighted blend (preserves both signals) ---
        # When vector search was skipped (path query or short physics term),
        # text results use full weight (1.0) rather than the reduced TEXT_WEIGHT
        # to avoid artificially penalising the only available signal.
        text_only_mode = embedding is None
        _VEC_WEIGHT = 0.6
        _TEXT_WEIGHT = 0.4
        all_ids = set(vec_scores) | set(text_scores)
        scores: dict[str, float] = {}
        for pid in all_ids:
            vs = vec_scores.get(pid, 0.0)
            ts = text_scores.get(pid, 0.0)
            if text_only_mode:
                scores[pid] = round(ts, 4)
            elif vs > 0 and ts > 0:
                scores[pid] = round(_VEC_WEIGHT * vs + _TEXT_WEIGHT * ts, 4)
            elif vs > 0:
                scores[pid] = round(vs * _VEC_WEIGHT, 4)
            else:
                scores[pid] = round(ts * _TEXT_WEIGHT, 4)

        # --- Path segment tiebreaker ---
        # Multiplicative boost for paths whose segments exactly match query
        # words.  Use exact segment match to avoid substring false positives
        # (e.g. "ip" matching "pipe" or "description").
        query_words = [
            w.lower()
            for w in query.split()
            if len(w) > 2 or w.lower() in _PHYSICS_SHORT_TERMS
        ]
        if query_words:
            for pid in scores:
                segments = pid.lower().split("/")
                match_count = sum(
                    1
                    for w in query_words
                    if any(w == seg for seg in segments)
                    or (len(w) > 3 and any(w in seg for seg in segments))
                )
                if match_count > 0:
                    scores[pid] = round(scores[pid] * (1 + 0.015 * match_count), 4)

        # --- Accessor de-ranking ---
        # Accessor terminals (data containers, time bases, validity flags) are
        # rarely the user's intent for concept queries.  Apply a small penalty
        # so the parent concept path ranks higher.
        for pid in scores:
            terminal = pid.rsplit("/", 1)[-1].lower()
            if terminal in _ACCESSOR_TERMINALS:
                scores[pid] = round(scores[pid] * 0.95, 4)

        # --- IDS preference for unqualified queries ---
        # When no explicit IDS filter is active, give a small boost to results
        # from the canonical IDS for recognised concept keywords.
        if not normalized_filter and query_words:
            matched_ids_prefs: set[str] = set()
            for w in query_words:
                if w in _CONCEPT_IDS_PREFERENCE:
                    matched_ids_prefs.add(_CONCEPT_IDS_PREFERENCE[w])
            if matched_ids_prefs:
                for pid in scores:
                    result_ids = pid.split("/", 1)[0]
                    if result_ids in matched_ids_prefs:
                        scores[pid] = round(scores[pid] * 1.03, 4)

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

        # --- Post-filter by physics_domain and lifecycle_status ---
        if physics_domain:
            hits = [h for h in hits if h.physics_domain == physics_domain]
        if lifecycle_filter:
            hits = [h for h in hits if h.lifecycle_status == lifecycle_filter]

        # --- Expand STRUCTURE hits with leaf children ---
        _STRUCTURE_TYPES = {"structure", "struct_array", "STRUCTURE"}
        structure_hits = [h for h in hits if h.data_type in _STRUCTURE_TYPES][:5]

        if structure_hits:
            parent_ids = [h.path for h in structure_hits]
            child_results = self._gc.query(
                """
                UNWIND $parent_ids AS parent_id
                MATCH (child:IMASNode)
                WHERE child.id STARTS WITH parent_id + '/'
                  AND NOT (child)-[:DEPRECATED_IN]->(:DDVersion)
                  AND child.node_category IN $categories
                  AND child.data_type IS NOT NULL
                  AND NOT (toLower(child.data_type) IN ['structure', 'struct_array'])
                  AND NOT (child.id CONTAINS '_error_')
                WITH parent_id, child
                ORDER BY child.id
                WITH parent_id, collect({
                    id: child.id,
                    name: child.name,
                    data_type: child.data_type,
                    units: child.units
                })[..10] AS children, count(child) AS total
                RETURN parent_id, children, total
                """,
                parent_ids=parent_ids,
                categories=effective_categories,
            )

            children_by_parent: dict[str, tuple[list, int]] = {
                r["parent_id"]: (r["children"], r["total"]) for r in child_results or []
            }

            for hit in structure_hits:
                child_data = children_by_parent.get(hit.path)
                if child_data:
                    children_list, total = child_data
                    hit.children = children_list
                    hit.children_total = total

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
        """Embed query text using the module-level Encoder singleton."""
        encoder = _get_encoder()
        return encoder.embed_texts([query], prompt_name="query")[0].tolist()


class GraphPathTool:
    """Graph-backed path validation and fetching."""

    _fuzzy_cache: dict[int | None, PathFuzzyMatcher] = {}

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    def _get_fuzzy_matcher(self, dd_version: int | None = None) -> PathFuzzyMatcher:
        """Lazy-init a version-scoped PathFuzzyMatcher."""
        if dd_version not in self._fuzzy_cache:
            from imas_codex.search.fuzzy_matcher import PathFuzzyMatcher

            dd_params: dict[str, Any] = {}
            dd_clause = _dd_version_clause("p", dd_version, dd_params)
            rows = self._gc.query(
                f"""
                MATCH (p:IMASNode)
                WHERE p.node_category IN $categories {dd_clause}
                RETURN p.id AS id, p.ids AS ids
                """,
                categories=list(SEARCHABLE_CATEGORIES),
                **dd_params,
            )
            paths = [r["id"] for r in rows] if rows else []
            ids_names = sorted({r["ids"] for r in rows if r["ids"]}) if rows else []
            self._fuzzy_cache[dd_version] = PathFuzzyMatcher(ids_names, paths)
        return self._fuzzy_cache[dd_version]

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
    async def check_dd_paths(
        self,
        paths: str | list[str],
        ids: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> CheckPathsResult:
        """Validate IMAS paths against graph using batch UNWIND."""
        path_list = _normalize_paths(paths)
        if ids:
            path_list = [
                f"{ids}/{p}" if "/" not in p or not p.startswith(ids) else p
                for p in path_list
            ]

            # Validate that the IDS itself exists
            ids_check = self._gc.query(
                "MATCH (i:IDS {id: $ids_name}) RETURN i.id",
                ids_name=ids,
            )
            if not ids_check:
                # IDS not found — try fuzzy suggestion
                ids_suggestions: list[str] = []
                try:
                    matcher = self._get_fuzzy_matcher(dd_version)
                    ids_suggestions = matcher.suggest_ids(ids, max_suggestions=3)
                except Exception:
                    logger.debug("Fuzzy IDS suggestion unavailable", exc_info=True)

                suggestion_text = (
                    f"Unknown IDS '{ids}'. Did you mean: {ids_suggestions[0]}?"
                    if ids_suggestions
                    else f"IDS '{ids}' not found in data dictionary"
                )
                top_suggestion = ids_suggestions[0] if ids_suggestions else None
                return CheckPathsResult(
                    results=[
                        CheckPathsResultItem(
                            path=p,
                            exists=False,
                            suggestion=top_suggestion,
                            suggestions=ids_suggestions or None,
                            error=suggestion_text,
                        )
                        for p in path_list
                    ],
                    summary={
                        "total": len(path_list),
                        "found": 0,
                        "not_found": len(path_list),
                    },
                    error=suggestion_text,
                )

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        # Batch existence + rename check in a single UNWIND query
        rows = self._gc.query(
            f"""
            UNWIND $paths AS check_path
            OPTIONAL MATCH (p:IMASNode {{id: check_path}})
            WHERE true {dd_clause}
            OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (old:IMASNode {{id: check_path}})-[:RENAMED_TO]->(new:IMASNode)
            RETURN check_path,
                   p.id AS id, p.ids AS ids, p.data_type AS data_type,
                   u.id AS units, p.lifecycle_status AS lifecycle_status,
                   old.id AS renamed_from, new.id AS renamed_to
            """,
            paths=path_list,
            **dd_params,
        )

        row_map: dict[str, dict] = {}
        for r in rows or []:
            row_map[r["check_path"]] = r

        results = []
        found = 0
        not_found_paths = []
        for path in path_list:
            r = row_map.get(path)
            if r and r["id"]:
                results.append(
                    CheckPathsResultItem(
                        path=r["id"],
                        exists=True,
                        ids_name=r["ids"],
                        data_type=r["data_type"],
                        units=r["units"] or "",
                        lifecycle_status=r.get("lifecycle_status"),
                    )
                )
                found += 1
            elif r and r["renamed_to"]:
                new_path = r["renamed_to"]
                results.append(
                    CheckPathsResultItem(
                        path=path,
                        exists=False,
                        renamed_from=[
                            {"old_path": r["renamed_from"], "new_path": new_path}
                        ],
                        migration={"type": "renamed", "target": new_path},
                        suggestion=new_path,
                    )
                )
            else:
                not_found_paths.append(path)
                results.append(
                    CheckPathsResultItem(
                        path=path,
                        exists=False,
                    )
                )

        # Fuzzy suggestions for paths not found and not renamed
        if not_found_paths:
            try:
                matcher = self._get_fuzzy_matcher(dd_version)
                for item in results:
                    if (
                        not item.exists
                        and item.path in not_found_paths
                        and not item.suggestion
                    ):
                        suggestions = matcher.suggest_paths(
                            item.path, max_suggestions=3
                        )
                        if suggestions:
                            item.suggestions = suggestions
                            item.suggestion = suggestions[0]
            except Exception:
                logger.debug("Fuzzy matching unavailable", exc_info=True)

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
    async def fetch_dd_paths(
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
                OPTIONAL MATCH (change)-[:IN_VERSION]->(cv:DDVersion)
                WITH p, u, cluster_labels, coordinates, ident, iv,
                     collect(DISTINCT {version: cv.id,
                                       type: change.change_type,
                                       old_value: change.old_value,
                                       new_value: change.new_value}) AS version_changes
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
        return "list_dd_paths"

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
    async def list_dd_paths(
        self,
        paths: str,
        format: str = "yaml",
        leaf_only: bool = False,
        include_ids_prefix: bool = True,
        max_paths: int | None = None,
        dd_version: int | None = None,
        response_profile: str = "minimal",
        physics_domain: str | None = None,
        node_type: str | None = None,
        lifecycle_filter: str | None = None,
        node_category: str | None = None,
        ctx: Context | None = None,
    ) -> ListPathsResult:
        """List paths from graph."""
        from imas_codex.core.paths import normalize_imas_path

        queries = paths.strip().split()
        results = []

        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        for query in queries:
            query = normalize_imas_path(query)
            # Determine if this is an IDS name or a path prefix
            if "/" in query:
                ids_name = query.split("/")[0]
                prefix = query
            else:
                ids_name = query
                prefix = None

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

            # Build filter clauses
            leaf_filter = (
                "AND NOT (p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])"
                if leaf_only
                else ""
            )
            limit_clause = f"LIMIT {max_paths}" if max_paths else ""

            extra_filters = ""
            if physics_domain:
                extra_filters += " AND p.physics_domain = $physics_domain"
                dd_params["physics_domain"] = physics_domain
            if node_type:
                extra_filters += " AND p.node_type = $node_type"
                dd_params["node_type"] = node_type
            if lifecycle_filter:
                extra_filters += " AND p.lifecycle_status = $lifecycle_filter"
                dd_params["lifecycle_filter"] = lifecycle_filter

            include_metadata = response_profile != "minimal"

            if include_metadata:
                return_clause = (
                    "OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)\n"
                    "                RETURN p.id AS id, p.name AS name, "
                    "p.data_type AS data_type,\n"
                    "                       p.node_type AS node_type, "
                    "p.documentation AS documentation,\n"
                    "                       p.lifecycle_status AS lifecycle_status,\n"
                    "                       u.symbol AS units"
                )
            else:
                return_clause = "RETURN p.id AS id"

            # Unified query — use ids= for plain IDS names, STARTS WITH for prefixes
            dd_params["categories"] = _resolve_node_categories(node_category)
            if prefix:
                match_clause = (
                    "WHERE p.id STARTS WITH $prefix AND p.node_category IN $categories"
                )
                dd_params["prefix"] = prefix + ("/" if not prefix.endswith("/") else "")
            else:
                match_clause = (
                    "WHERE p.ids = $ids_name AND p.node_category IN $categories"
                )
                dd_params["ids_name"] = ids_name

            # Count total paths before applying LIMIT
            total_count = None
            if max_paths:
                count_result = self._gc.query(
                    f"""
                    MATCH (p:IMASNode)
                    {match_clause}
                    {leaf_filter}
                    {extra_filters}
                    {dd_clause}
                    RETURN count(p) AS cnt
                    """,
                    **dd_params,
                )
                total_count = count_result[0]["cnt"] if count_result else 0

            path_results = self._gc.query(
                f"""
                MATCH (p:IMASNode)
                {match_clause}
                {leaf_filter}
                {extra_filters}
                {dd_clause}
                {return_clause}
                ORDER BY p.id
                {limit_clause}
                """,
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
                        "lifecycle_status": r.get("lifecycle_status"),
                    }
                    for r in path_results
                ]

            results.append(
                ListPathsResultItem(
                    query=query,
                    path_count=total_count
                    if total_count is not None
                    else len(path_ids),
                    total_paths=total_count,
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
        return "get_dd_catalog"

    @cache_results(ttl=3600)
    @handle_errors(fallback="overview_error")
    @mcp_tool(
        "List all available IDSs (Interface Data Structures) with descriptions "
        "and statistics. Returns every IDS with name, description, path count, "
        "physics domain, and lifecycle status. Use as a starting point to discover "
        "which IDS contains the data you need. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns latest."
    )
    async def get_dd_catalog(
        self,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> GetOverviewResult:
        """Get full catalog of all IDSs from graph."""
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

        for r in ids_results or []:
            ids_name = r["name"]
            all_ids.append(ids_name)
            ids_statistics[ids_name] = {
                "path_count": r["path_count"],
                "description": r["description"] or "",
                "physics_domain": r["physics_domain"] or "",
                "lifecycle_status": r["lifecycle_status"] or "",
            }
            if r["physics_domain"]:
                physics_domains.add(r["physics_domain"])

        # Domain summary: group IDS counts and path counts by physics domain
        domain_summary: dict[str, dict[str, int]] = {}
        for _ids_name, stats in ids_statistics.items():
            dom = stats.get("physics_domain") or "uncategorized"
            entry = domain_summary.setdefault(dom, {"ids_count": 0, "path_count": 0})
            entry["ids_count"] += 1
            entry["path_count"] += stats["path_count"]

        # Lifecycle summary
        lifecycle_counts: dict[str, int] = {}
        for r in ids_results or []:
            lc = r.get("lifecycle_status") or "unknown"
            lifecycle_counts[lc] = lifecycle_counts.get(lc, 0) + 1

        # Build tools list
        mcp_tools = [
            "search_dd_paths",
            "check_dd_paths",
            "fetch_dd_paths",
            "list_dd_paths",
            "get_dd_catalog",
            "search_dd_clusters",
            "get_dd_identifiers",
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
            query=None,
            physics_domains=sorted(physics_domains),
            ids_statistics=ids_statistics,
            mcp_tools=mcp_tools,
            dd_version=current_version,
            mcp_version=version,
            total_leaf_nodes=total_paths,
            domain_summary=domain_summary,
            lifecycle_summary=lifecycle_counts,
            unit_statistics=None,
        )


def _filter_summary_from_clusters(result: dict[str, Any]) -> None:
    """Remove summary IDS paths from cluster member lists in-place."""
    for cluster in result.get("clusters", []):
        paths = cluster.get("paths")
        if paths:
            cluster["paths"] = [p for p in paths if not p.startswith("summary/")]


class GraphClustersTool:
    """Graph-backed cluster search using vector indexes."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "search_dd_clusters"

    @cache_results(ttl=600)
    @handle_errors(fallback="cluster_error")
    @mcp_tool(
        "Search semantic clusters of related IMAS data paths. "
        "query: Natural language description (e.g., 'boundary geometry', 'transport coefficients') "
        "or exact IMAS path to find its cluster membership. Optional when ids_filter is provided. "
        "scope: Filter by cluster scope - 'global', 'domain', or 'ids'. "
        "ids_filter: Limit to clusters containing paths from specific IDS. "
        "section_only: If true, only return clusters containing structural sections. "
        "include_summary_ids: Include paths from the 'summary' IDS in cluster member lists (default: false). "
        "The summary IDS contains aggregate quantities duplicated from primary IDSs. "
        "dd_version: Filter by DD major version (e.g., 3 or 4). None returns all versions."
    )
    async def search_dd_clusters(
        self,
        query: str | None = None,
        scope: Literal["global", "domain", "ids"] | None = None,
        ids_filter: str | list[str] | None = None,
        section_only: bool = False,
        dd_version: int | None = None,
        include_summary_ids: bool = False,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Search clusters using graph vector indexes."""
        normalized_filter = normalize_ids_filter(ids_filter)

        # Determine whether to filter summary paths from cluster member lists.
        _filter_list = (
            normalized_filter
            if isinstance(normalized_filter, list)
            else ([normalized_filter] if normalized_filter else [])
        )
        _exclude_summary_ids = not include_summary_ids and "summary" not in _filter_list

        # IDS listing mode: no query, ids_filter provided
        if not query and normalized_filter:
            result = self._list_by_ids(
                normalized_filter,
                scope,
                section_only=section_only,
                dd_version=dd_version,
            )
            if _exclude_summary_ids:
                _filter_summary_from_clusters(result)
            return result

        if not query:
            return {
                "error": "Either query or ids_filter is required.",
                "clusters_found": 0,
                "clusters": [],
            }

        # Detect query type: path lookup vs semantic search
        if "/" in query and " " not in query:
            result = self._search_by_path(query, scope, dd_version=dd_version)
            if _exclude_summary_ids:
                _filter_summary_from_clusters(result)
            return result

        result = self._search_by_text(
            query, scope, normalized_filter, dd_version=dd_version
        )
        if _exclude_summary_ids:
            _filter_summary_from_clusters(result)
        return result

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
        """Semantic search over cluster embeddings with text fallback for short terms."""
        is_short_physics_term = " " not in query.strip() and (
            len(query.strip()) <= 3 or query.strip().lower() in _PHYSICS_SHORT_TERMS
        )
        expansion = (
            _PHYSICS_SHORT_TERMS.get(query.strip().lower())
            if is_short_physics_term
            else None
        )

        # --- Text-based CONTAINS search (always run as supplement) ---
        text_scope_filter = "AND cluster.scope = $scope" if scope else ""
        text_ids_clause = ""
        text_params: dict[str, Any] = {"text_query": query.lower()}
        if scope:
            text_params["scope"] = scope
        if ids_filter:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            text_ids_clause = (
                "AND any(ids_name IN cluster.ids_names WHERE ids_name IN $ids_filter)"
            )
            text_params["ids_filter"] = filter_list

        text_results: list[dict[str, Any]] = (
            self._gc.query(
                f"""
                MATCH (cluster:IMASSemanticCluster)
                WHERE (toLower(cluster.label) CONTAINS $text_query
                       OR toLower(cluster.description) CONTAINS $text_query)
                {text_scope_filter}
                {text_ids_clause}
                OPTIONAL MATCH (member:IMASNode)-[:IN_CLUSTER]->(cluster)
                WITH cluster, 0.9 AS score, collect(DISTINCT member.id)[..50] AS paths
                RETURN cluster.id AS id, cluster.label AS label,
                       cluster.description AS description,
                       cluster.scope AS scope, cluster.cross_ids AS cross_ids,
                       cluster.ids_names AS ids_names,
                       cluster.similarity_score AS similarity,
                       score AS relevance_score, paths
                ORDER BY size(cluster.label) ASC
                LIMIT 10
                """,
                **text_params,
            )
            or []
        )

        # --- Vector search (use expansion for short physics terms) ---
        vector_results: list[dict[str, Any]] = []
        vector_query = expansion if expansion else query
        embedding = self._embed_query(vector_query)
        if embedding is not None:
            scope_filter = "AND cluster.scope = $scope" if scope else ""
            ids_filter_clause = ""
            vector_k = (
                5 if text_results else 10
            )  # Fewer vector results when text already found matches
            params: dict[str, Any] = {"embedding": embedding, "k": vector_k}
            if scope:
                params["scope"] = scope
            if ids_filter:
                filter_list = (
                    ids_filter if isinstance(ids_filter, list) else [ids_filter]
                )
                ids_filter_clause = "AND any(ids_name IN cluster.ids_names WHERE ids_name IN $ids_filter)"
                params["ids_filter"] = filter_list

            # Calculate adaptive threshold based on query complexity
            query_word_count = len(query.strip().split())
            min_vector_score = 0.35 if query_word_count > 3 else 0.3
            params["min_score"] = min_vector_score

            vector_results = (
                self._gc.query(
                    f"""
                    CALL db.index.vector.queryNodes(
                        'cluster_label_embedding', $k, $embedding
                    )
                    YIELD node AS cluster, score
                    WHERE score > $min_score
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
                or []
            )

        # --- Merge: text results take priority (dedup by cluster id, keep higher score) ---
        seen: dict[str, Any] = {}
        for r in text_results:
            seen[r["id"]] = r
        for r in vector_results:
            existing = seen.get(r["id"])
            if existing is None or (
                r.get("relevance_score", 0) > existing.get("relevance_score", 0)
            ):
                seen[r["id"]] = r

        merged = sorted(
            seen.values(), key=lambda x: x.get("relevance_score", 0), reverse=True
        )
        clusters = self._format_clusters(merged, include_score=True)

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
        return encoder.embed_texts([query], prompt_name="query")[0].tolist()


class GraphIdentifiersTool:
    """Graph-backed identifier schemas."""

    def __init__(self, graph_client: GraphClient):
        self._gc = graph_client

    @property
    def tool_name(self) -> str:
        return "get_dd_identifiers"

    @cache_results(ttl=3600)
    @handle_errors(fallback="identifiers_error")
    @mcp_tool(
        "Get IMAS identifier/enumeration schemas. "
        "These define valid values for typed fields like coordinate systems, "
        "probe types, and grid types. "
        "query: Optional filter (e.g., 'coordinate', 'grid_type', 'magnetics'). "
        "Supports multi-word queries with OR logic: 'coordinate transport'. "
        "Underscores are treated as word separators: 'grid_type' matches schemas "
        "containing 'grid' or 'type'."
    )
    async def get_dd_identifiers(
        self,
        query: str | None = None,
        ctx: Context | None = None,
    ) -> GetIdentifiersResult:
        """Get identifier schemas from graph.

        When a query is provided, uses a two-strategy approach:
        1. Vector similarity search on enriched description embeddings
        2. Keyword matching on name, description, keywords, and options

        Results from both strategies are merged, then ranked by relevance.
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

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """Split query into keywords on underscores, commas, and whitespace."""
        import re

        return [kw for kw in re.split(r"[_\s,]+", query.strip().lower()) if kw]

    @staticmethod
    def _score_keyword_match(
        keywords: list[str],
        original_query: str,
        name: str,
        description: str,
        parsed_options: list[dict],
        schema_keywords: list[str],
    ) -> int:
        """Score a schema against query keywords with tiered relevance.

        Scoring tiers (higher = more relevant):
        - 6: exact original query substring in name
        - 5: normalized query (spaces for underscores) in name
        - 4: all keywords match name
        - 3: keyword in name
        - 2: keyword in description or schema keywords
        - 1: keyword in option names/descriptions
        """
        score = 0
        query_lower = original_query.lower()
        name_lower = name.lower()
        desc_lower = description.lower()

        # Tier 6: exact original query in name
        if query_lower in name_lower:
            score += 6

        # Tier 5: normalized query (underscores → spaces) in name
        normalized = query_lower.replace("_", " ")
        name_spaced = name_lower.replace("_", " ")
        if normalized in name_spaced:
            score += 5

        # Tier 4: all keywords match name
        if all(kw in name_lower for kw in keywords):
            score += 4

        # Per-keyword scoring
        for kw in keywords:
            if kw in name_lower:
                score += 3
            if kw in desc_lower:
                score += 2
            if any(kw in skw.lower() for skw in schema_keywords):
                score += 2
            # Tier 1: keyword in parsed option names/descriptions
            for opt in parsed_options:
                opt_name = (opt.get("name") or "").lower()
                opt_desc = (opt.get("description") or "").lower()
                if kw in opt_name or kw in opt_desc:
                    score += 1
                    break  # one match per keyword is enough

        return score

    def _search_identifiers(self, query: str) -> GetIdentifiersResult:
        """Search identifier schemas by vector similarity + keyword match."""
        keywords = self._tokenize_query(query)
        if not keywords:
            return self._list_all_identifiers()

        seen: set[str] = set()
        scored: list[tuple[dict, int]] = []

        # Strategy 1: Vector similarity search
        try:
            encoder = _get_encoder()
            query_embedding = encoder.embed_texts([query])[0].tolist()

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
                    schema = self._format_schema(r)
                    # Vector matches get a bonus on top of keyword score
                    kw_score = self._score_keyword_match(
                        keywords,
                        query,
                        name,
                        r.get("description") or "",
                        schema.get("options") or [],
                        r.get("keywords") or [],
                    )
                    scored.append((schema, kw_score + 10))
        except Exception as exc:
            # Vector search is optional — log and fall through to keyword matching.
            # Expected failures: EmbeddingBackendError, ConnectionError, missing index.
            logger.debug("Vector search unavailable for identifiers: %s", exc)

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
        for r in keyword_results or []:
            name = r["name"]
            if name in seen:
                continue

            schema = self._format_schema(r)
            kw_score = self._score_keyword_match(
                keywords,
                query,
                name,
                r.get("description") or "",
                schema.get("options") or [],
                r.get("keywords") or [],
            )
            if kw_score > 0:
                seen.add(name)
                scored.append((schema, kw_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        schemas = [s for s, _ in scored]

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
        "Find paths in other IDSs that are related to a given path. "
        "Discovers related paths via shared clusters, coordinates, units, "
        "and identifier schemas across IDS boundaries. "
        "path (required): Exact IMAS path (e.g. 'equilibrium/time_slice/profiles_1d/psi'). "
        "relationship_types: Filter to specific types — 'semantic', 'cluster', 'coordinate', "
        "'unit', 'identifier', or 'all' (default)."
    )
    @handle_errors("find_related_dd_paths")
    async def find_related_dd_paths(
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
        "Analyze the internal structure and organization of a specific IMAS IDS. "
        "Returns metrics (path counts, depth), data type distribution, "
        "physics domains, coordinate arrays, and COCOS/cluster counts. "
        "Use get_dd_cocos_fields or search_dd_clusters for full listings. "
        "ids_name (required): IDS name (e.g. 'equilibrium')."
    )
    @handle_errors("get_ids_summary")
    async def get_ids_summary(
        self,
        ids_name: str,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get a compact structural summary of an IDS."""
        dd_params: dict[str, Any] = {
            "ids_name": ids_name,
            "categories": list(SEARCHABLE_CATEGORIES),
        }
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        # Query 1: IDS metadata + metrics + top-level sections
        combined = self._gc.query(
            f"""
            MATCH (i:IDS {{id: $ids_name}})
            OPTIONAL MATCH (p:IMASNode)
            WHERE p.ids = $ids_name AND p.node_category IN $categories {dd_clause}
            WITH i,
                 count(p) AS total,
                 count(CASE WHEN NOT (p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY'])
                            THEN 1 END) AS leaves,
                 max(size(split(p.id, '/')) - 1) AS max_depth,
                 collect(CASE WHEN size(split(p.id, '/')) = 2
                              THEN p END) AS top_nodes
            RETURN i.name AS name,
                   COALESCE(i.description, i.documentation) AS description,
                   i.physics_domain AS physics_domain,
                   i.lifecycle_status AS lifecycle_status,
                   total, leaves, max_depth,
                   [n IN top_nodes WHERE n IS NOT NULL |
                    {{id: n.id, name: n.name, data_type: n.data_type,
                      doc: left(n.documentation, 80)}}] AS top_sections
            """,
            **dd_params,
        )

        if not combined:
            return {
                "ids_name": ids_name,
                "error": f"IDS '{ids_name}' not found",
            }

        meta = combined[0]

        # Query 2: Cluster count (compact — use search_dd_clusters for full listings)
        cluster_count_result = self._gc.query(
            f"""
            MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            WHERE p.ids = $ids_name {dd_clause}
            RETURN count(DISTINCT c) AS count
            """,
            **dd_params,
        )

        # Query 3: Identifier schemas used in this IDS
        identifiers = self._gc.query(
            f"""
            MATCH (p:IMASNode)-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
            WHERE p.ids = $ids_name {dd_clause}
            WITH s, collect(p.id) AS paths, count(p) AS usage_count
            RETURN s.name AS schema, usage_count,
                   paths[0..3] AS example_paths
            ORDER BY usage_count DESC
            """,
            **dd_params,
        )

        # Query 4: COCOS count only (use get_dd_cocos_fields for full listing)
        cocos_count_result = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name
              AND p.cocos_label_transformation IS NOT NULL {dd_clause}
            RETURN count(p) AS count
            """,
            **dd_params,
        )

        # Query 5: Coordinate array count
        coord_count_result = self._gc.query(
            f"""
            MATCH (p:IMASNode)-[:HAS_COORDINATE]->(coord:IMASCoordinateSpec)
            WHERE p.ids = $ids_name {dd_clause}
            RETURN count(DISTINCT p) AS count
            """,
            **dd_params,
        )

        # Query 6: Data type distribution
        types = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name AND p.data_type IS NOT NULL {dd_clause}
            RETURN p.data_type AS data_type, count(p) AS count
            ORDER BY count DESC
            """,
            **dd_params,
        )

        # Query 7: Lifecycle distribution within this IDS
        lifecycle_dist = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.ids = $ids_name AND p.node_category IN $categories
              AND p.lifecycle_status IS NOT NULL {dd_clause}
            RETURN p.lifecycle_status AS status, count(p) AS count
            ORDER BY count DESC
            """,
            **dd_params,
        )

        total = meta.get("total", 0)
        leaves = meta.get("leaves", 0)
        cluster_count = cluster_count_result[0]["count"] if cluster_count_result else 0
        cocos_count = cocos_count_result[0]["count"] if cocos_count_result else 0
        coord_count = coord_count_result[0]["count"] if coord_count_result else 0

        result: dict[str, Any] = {
            "ids_name": ids_name,
            "description": meta.get("description", ""),
            "physics_domain": meta.get("physics_domain", ""),
            "lifecycle_status": meta.get("lifecycle_status", ""),
            "metrics": {
                "total_paths": total,
                "leaf_count": leaves,
                "structure_count": total - leaves,
                "max_depth": meta.get("max_depth", 0),
            },
            "top_sections": meta.get("top_sections", []),
            "semantic_clusters": cluster_count,
            "identifier_schemas": [
                {
                    "schema": s["schema"],
                    "usage_count": s["usage_count"],
                    "examples": s["example_paths"],
                }
                for s in (identifiers or [])
            ],
            "coordinate_arrays": coord_count,
            "cocos_fields": cocos_count,
            "data_types": {t["data_type"]: t["count"] for t in (types or [])},
            "lifecycle_distribution": {
                r["status"]: r["count"] for r in (lifecycle_dist or [])
            },
        }

        return result

    @handle_errors("get_dd_cocos_fields")
    async def get_dd_cocos_fields(
        self,
        transformation_type: str | None = None,
        ids_filter: str | None = None,
        dd_version: int | None = None,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Get all COCOS-dependent fields grouped by transformation type."""
        dd_params: dict[str, Any] = {}
        dd_clause = _dd_version_clause("p", dd_version, dd_params)

        ids_clause = ""
        if ids_filter:
            ids_clause = "AND p.ids = $ids_filter"
            dd_params["ids_filter"] = ids_filter

        transform_clause = ""
        if transformation_type:
            transform_clause = "AND p.cocos_label_transformation = $transform_type"
            dd_params["transform_type"] = transformation_type

        rows = self._gc.query(
            f"""
            MATCH (p:IMASNode)
            WHERE p.cocos_label_transformation IS NOT NULL
              {ids_clause} {transform_clause} {dd_clause}
            RETURN p.id AS path,
                   p.ids AS ids,
                   p.cocos_label_transformation AS transformation_type,
                   p.data_type AS data_type,
                   p.documentation AS documentation
            ORDER BY p.cocos_label_transformation, p.id
            """,
            **dd_params,
        )

        grouped: dict[str, list[dict]] = {}
        for row in rows:
            tt = row["transformation_type"]
            grouped.setdefault(tt, []).append(
                {
                    "path": row["path"],
                    "ids": row["ids"],
                    "data_type": row["data_type"],
                    "documentation": row["documentation"],
                }
            )

        return {
            "transformation_types": {
                tt: {"count": len(fields), "fields": fields}
                for tt, fields in grouped.items()
            },
            "total_fields": len(rows),
            "filters": {
                "transformation_type": transformation_type,
                "ids_filter": ids_filter,
                "dd_version": dd_version,
            },
        }

    @mcp_tool(
        "Export full IDS structure with documentation, units, and types. "
        "Returns all paths in an IDS with their complete metadata. "
        "ids_name (required): IDS name (e.g. 'equilibrium'). "
        "leaf_only: If true, return only leaf nodes (default false)."
    )
    @handle_errors("export_dd_ids")
    async def export_dd_ids(
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
    @handle_errors("export_dd_domain")
    async def export_dd_domain(
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
    """Normalize paths input to a flat list, handling dots, annotations, and JSON."""
    import json

    from imas_codex.core.paths import normalize_imas_path

    if isinstance(paths, list):
        return [normalize_imas_path(p) for p in paths]

    s = paths.strip()
    # Handle JSON array strings from MCP transport
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [
                    normalize_imas_path(str(p).strip())
                    for p in parsed
                    if str(p).strip()
                ]
        except (json.JSONDecodeError, TypeError):
            pass
    raw = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    return [normalize_imas_path(p) for p in raw]


STRUCTURE_DATA_TYPES = ("STRUCTURE", "STRUCT_ARRAY")


def _structure_type_list() -> str:
    """Return a Cypher list literal of structure data types."""
    quoted = ", ".join(f"'{dtype}'" for dtype in STRUCTURE_DATA_TYPES)
    return f"[{quoted}]"


def _leaf_data_type_clause(alias: str) -> str:
    """Return a Cypher clause that matches non-structure data nodes."""
    return f"{alias}.data_type IS NOT NULL AND NOT ({alias}.data_type IN {_structure_type_list()})"


def _text_search_dd_paths(
    gc: GraphClient,
    query: str,
    limit: int,
    ids_filter: str | list[str] | None,
    *,
    dd_version: int | None = None,
    exclude_summary: bool = False,
    categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Text-based search on IMAS paths by query string.

    Uses fulltext index for BM25 scoring when available, falls back to
    CONTAINS matching. Filters out generic metadata paths.
    """
    _categories = categories if categories is not None else list(SEARCHABLE_CATEGORIES)
    query_lower = query.lower()
    query_words = [
        w
        for w in query_lower.split()
        if len(w) > 2 or w.lower() in _PHYSICS_SHORT_TERMS
    ]

    where_parts = [
        "NOT (p)-[:DEPRECATED_IN]->(:DDVersion)",
        "p.node_category IN $categories",
    ]
    # Cap CONTAINS fallback to avoid full scans on large graphs
    contains_limit = min(limit, 100)
    params: dict[str, Any] = {
        "query_lower": query_lower,
        "limit": contains_limit,
        "categories": _categories,
    }

    dd_clause = _dd_version_clause("p", dd_version, params)
    if dd_clause:
        where_parts.append(dd_clause.lstrip("AND "))

    if ids_filter is not None:
        filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
        where_parts.append("p.ids IN $ids_filter")
        params["ids_filter"] = filter_list

    if exclude_summary:
        where_parts.append("p.ids <> 'summary'")

    where_base = " AND ".join(where_parts)

    # Try fulltext index first (BM25 scoring)
    try:
        ft_where = "WHERE NOT (p)-[:DEPRECATED_IN]->(:DDVersion) AND p.node_category IN $categories"
        ft_params: dict[str, Any] = {
            "query": _build_phrase_aware_query(query),
            "limit": limit,
            "categories": _categories,
        }
        if ids_filter is not None:
            filter_list = ids_filter if isinstance(ids_filter, list) else [ids_filter]
            ft_where += " AND p.ids IN $ids_filter"
            ft_params["ids_filter"] = filter_list

        if exclude_summary:
            ft_where += " AND p.ids <> 'summary'"

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
            # Normalize BM25 scores to 0-1 range preserving relative ranking.
            # No floor — low-ranked BM25 hits should score low so vector
            # results can outrank them in the weighted blend.
            max_score = max(r["score"] for r in ft_results) if ft_results else 1.0
            normalized = []
            for r in ft_results:
                pid = r["id"]
                raw = r["score"] / max_score if max_score > 0 else 0.0
                normalized.append({"id": pid, "score": raw})
            return normalized
    except Exception:
        pass

    # Fallback: CONTAINS matching with scored results.
    # For short queries (<=3 chars), use exact name/segment matching instead
    # of CONTAINS to avoid substring false positives (e.g. "ip" in "description").
    is_short_query = len(query_lower) <= 3 or query_lower in _PHYSICS_SHORT_TERMS
    if is_short_query:
        # Short term: match exact path segments or exact name
        cypher = f"""
            MATCH (p:IMASNode)
            WHERE {where_base}
              AND (
                toLower(p.name) = $query_lower
                OR any(seg IN split(p.id, '/') WHERE toLower(seg) = $query_lower)
                OR any(kw IN coalesce(p.keywords, []) WHERE toLower(kw) = $query_lower)
              )
            WITH p,
                 CASE
                   WHEN toLower(p.name) = $query_lower
                     AND {_leaf_data_type_clause("p")}
                    THEN 0.98
                   WHEN toLower(p.name) = $query_lower
                    THEN 0.95
                   WHEN any(seg IN split(p.id, '/') WHERE toLower(seg) = $query_lower)
                    THEN 0.93
                   ELSE 0.88
                 END AS base_score
            RETURN p.id AS id, base_score AS score
            ORDER BY base_score DESC, size(p.id) ASC
            LIMIT $limit
        """
    else:
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

    # Also search individual words for abbreviations like "ip".
    # Use exact segment matching to avoid substring false positives.
    if query_words:
        word_results = []
        for word in query_words[:3]:
            is_short = len(word) <= 3 or word in _PHYSICS_SHORT_TERMS
            if is_short:
                # Exact segment or name match only
                word_cypher = f"""
                    MATCH (p:IMASNode)
                    WHERE {where_base}
                      AND (
                        toLower(p.name) = $word
                        OR any(seg IN split(p.id, '/') WHERE toLower(seg) = $word)
                      )
                    RETURN p.id AS id, 0.95 AS score
                    LIMIT 15
                """
            else:
                word_cypher = f"""
                    MATCH (p:IMASNode)
                    WHERE {where_base}
                      AND (toLower(p.name) CONTAINS $word
                           OR toLower(p.id) CONTAINS $word)
                      AND {_leaf_data_type_clause("p")}
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
        OPTIONAL MATCH (change)-[:IN_VERSION]->(v:DDVersion)
        RETURN p.id AS id,
               count(change) AS change_count,
               collect({version: v.id,
                        type: change.change_type,
                        old_value: change.old_value,
                        new_value: change.new_value})[..5] AS notable_changes
    """
    results = gc.query(cypher, path_ids=path_ids)
    out = {}
    for r in results:
        # Filter null entries from OPTIONAL MATCH no-match
        raw = r.get("notable_changes") or []
        filtered = [c for c in raw if c.get("version") is not None]
        out[r["id"]] = {
            "change_count": len(filtered),
            "notable_changes": filtered,
        }
    return out


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
    3. Substring match on domain names

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

    # 3. Substring match on domain names
    matches = [d for d in sorted(valid_domains) if query in d]
    if matches:
        return matches, f"substring:{query}"

    return [], "no_match"
