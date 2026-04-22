"""Pure-function hybrid DD search for pipeline reuse.

Extracts the core hybrid vector+text search logic from the MCP tool
``GraphSearchTool.search_dd_paths`` into a standalone function that
accepts a :class:`GraphClient` and returns structured
:class:`SearchHit` rows.

This module is the single source of truth for hybrid DD search logic.
Both the MCP tool and the SN generation pipeline call
:func:`hybrid_dd_search` — the MCP tool adds only a formatting layer.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from imas_codex.core.node_categories import SEARCHABLE_CATEGORIES
from imas_codex.models.constants import SearchMode
from imas_codex.search.search_strategy import SearchHit

if TYPE_CHECKING:
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# ── Physics lookup tables (shared with graph_search.py via import) ───────

# Short physics abbreviations mapped to their expanded physics concepts.
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

# Accessor terminals de-ranked in scoring
_ACCESSOR_TERMINALS: frozenset[str] = frozenset(
    {"data", "value", "time", "validity", "fit", "coefficients"}
)

# Soft IDS preferences for unqualified concept queries
_CONCEPT_IDS_PREFERENCE: dict[str, str] = {
    "temperature": "core_profiles",
    "density": "core_profiles",
    "psi": "equilibrium",
    "current": "equilibrium",
    "boundary": "equilibrium",
    "b_field": "magnetics",
}

# Weight constants for score blending
_VEC_WEIGHT = 0.6
_TEXT_WEIGHT = 0.4


def hybrid_dd_search(
    gc: GraphClient,
    query: str,
    *,
    ids_filter: str | list[str] | None = None,
    facility: str | None = None,
    include_version_context: bool = False,
    include_summary_ids: bool = False,
    physics_domain: str | None = None,
    lifecycle_filter: str | None = None,
    node_category: str | None = None,
    cocos_transformation_type: str | None = None,
    dd_version: int | None = None,
    k: int = 20,
    encoder: Encoder | None = None,
) -> list[SearchHit]:
    """Return structured DD path hits with hybrid search + enrichment.

    Combines vector similarity search and fulltext/keyword search with
    weighted score blending, path-segment tiebreakers, accessor
    de-ranking, and IDS preference boosts.  Results are enriched with
    full metadata, optional facility cross-references, version context,
    and rename lineage.

    This is a **sync** function suitable for calling from any context
    (pipeline workers, CLI tools, notebooks).  The MCP tool wraps this
    with async and adds text formatting.

    Args:
        gc: Active graph client connected to Neo4j.
        query: Natural-language description or physics term.
        ids_filter: Restrict to specific IDS(es).
        facility: Include facility cross-references (e.g. ``"tcv"``).
        include_version_context: Attach DD version change history.
        include_summary_ids: Include paths from 'summary' IDS.
        physics_domain: Post-filter by physics domain.
        lifecycle_filter: Post-filter by lifecycle status.
        node_category: Restrict to specific node categories.
        cocos_transformation_type: Post-filter by COCOS type.
        dd_version: Filter by DD major version (e.g. 3 or 4).
        k: Maximum number of results to return.
        encoder: Optional pre-created encoder; if ``None``, uses the
            module-level singleton from ``graph_search``.

    Returns:
        List of :class:`SearchHit` instances ordered by relevance score.
    """
    from imas_codex.core.paths import normalize_imas_path
    from imas_codex.tools.graph_search import (
        _dd_version_clause,
        _fetch_rename_lineage,
        _get_facility_crossrefs,
        _get_version_context,
        _text_search_dd_paths,
    )
    from imas_codex.tools.utils import normalize_ids_filter

    # ── Normalise inputs ────────────────────────────────────────────
    query = normalize_imas_path(query)
    normalized_filter = normalize_ids_filter(ids_filter)
    effective_categories = _resolve_node_categories(node_category)

    _filter_list = (
        normalized_filter
        if isinstance(normalized_filter, list)
        else ([normalized_filter] if normalized_filter else [])
    )
    _exclude_summary_ids = not include_summary_ids and "summary" not in _filter_list
    summary_clause = "AND path.ids <> 'summary'" if _exclude_summary_ids else ""

    # ── Determine embedding strategy ────────────────────────────────
    is_path_query = "/" in query and " " not in query
    is_short_physics_term = " " not in query.strip() and (
        len(query.strip()) <= 3 or query.strip().lower() in _PHYSICS_SHORT_TERMS
    )
    if is_path_query:
        embedding = None
    elif is_short_physics_term:
        expansion = _PHYSICS_SHORT_TERMS.get(query.strip().lower())
        embedding = _embed(expansion, encoder) if expansion else None
    else:
        embedding = _embed(query, encoder)

    # ── Vector search ───────────────────────────────────────────────
    vec_scores: dict[str, float] = {}
    if embedding is not None:
        filter_clause = ""
        params: dict[str, Any] = {
            "embedding": embedding,
            "k": min(k * 5, 500),
            "vector_limit": min(k * 3, 150),
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

        vector_results = gc.query(
            f"""
            CALL db.index.vector.queryNodes('imas_node_embedding', $k, $embedding)
            YIELD node AS path, score
            WHERE NOT (path)-[:DEPRECATED_IN]->(:DDVersion)
              AND path.node_category IN $categories
              AND (path.node_category <> 'identifier' OR path.description IS NOT NULL)
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

    # ── Text search ─────────────────────────────────────────────────
    text_results = _text_search_dd_paths(
        gc,
        query,
        min(k * 3, 150),
        normalized_filter,
        dd_version=dd_version,
        exclude_summary=_exclude_summary_ids,
        categories=effective_categories,
    )
    text_scores: dict[str, float] = {}
    for r in text_results:
        text_scores[r["id"]] = round(r["score"], 4)

    # ── Weighted blend ──────────────────────────────────────────────
    text_only_mode = embedding is None
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

    # ── Path-segment tiebreaker ─────────────────────────────────────
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

    # ── Accessor de-ranking ─────────────────────────────────────────
    for pid in scores:
        terminal = pid.rsplit("/", 1)[-1].lower()
        if terminal in _ACCESSOR_TERMINALS:
            scores[pid] = round(scores[pid] * 0.95, 4)

    # ── IDS preference for unqualified queries ──────────────────────
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

    # ── Rank and limit ──────────────────────────────────────────────
    sorted_ids = sorted(scores, key=lambda pid: scores[pid], reverse=True)[:k]

    mode = SearchMode.AUTO

    if not sorted_ids:
        return []

    # ── Enrich with full metadata ───────────────────────────────────
    enriched = gc.query(
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
               path.cocos_transformation_type AS cocos_label,
               path.cocos_transformation_expression AS cocos_expression,
               path.description AS description,
               path.keywords AS keywords,
               path.enrichment_source AS enrichment_source,
               collect(DISTINCT coord.id) AS coordinates,
               ident IS NOT NULL AS has_identifier_schema,
               ident.name AS identifier_schema_name,
               ident.description AS identifier_schema_description,
               intro.id AS introduced_after_version
        """,
        path_ids=sorted_ids,
    )

    enriched_by_id = {r["id"]: r for r in enriched or []}

    # ── COCOS transformation filter (post-enrichment) ───────────────
    if cocos_transformation_type:
        enriched_by_id = {
            pid: r
            for pid, r in enriched_by_id.items()
            if r.get("cocos_label") == cocos_transformation_type
        }
        sorted_ids = [pid for pid in sorted_ids if pid in enriched_by_id]

    # ── Optional facility cross-references ──────────────────────────
    facility_xrefs: dict[str, dict[str, Any]] = {}
    if facility and sorted_ids:
        facility_xrefs = _get_facility_crossrefs(gc, sorted_ids, facility)

    # ── Optional version context ────────────────────────────────────
    version_ctx: dict[str, dict[str, Any]] = {}
    if include_version_context and sorted_ids:
        version_ctx = _get_version_context(gc, sorted_ids)

    # ── Rename lineage (always fetched) ─────────────────────────────
    rename_lineage_map = _fetch_rename_lineage(gc, sorted_ids)

    # ── Build SearchHit list ────────────────────────────────────────
    hits: list[SearchHit] = []
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
                cocos_transformation_type=r["cocos_label"],
                cocos_transformation_expression=r["cocos_expression"],
                description=r.get("description"),
                keywords=r.get("keywords"),
                enrichment_source=r.get("enrichment_source"),
                has_identifier_schema=bool(r["has_identifier_schema"]),
                identifier_schema_name=r.get("identifier_schema_name"),
                identifier_schema_description=r.get("identifier_schema_description"),
                introduced_after_version=r["introduced_after_version"],
                score=scores.get(pid, 0.0),
                rank=rank,
                search_mode=mode,
                facility_xrefs=xref,
                version_context=vctx,
                rename_lineage=rename_lineage_map.get(pid),
            )
        )

    # ── Post-filter by physics_domain and lifecycle_status ──────────
    if physics_domain:
        hits = [h for h in hits if h.physics_domain == physics_domain]
    if lifecycle_filter:
        hits = [h for h in hits if h.lifecycle_status == lifecycle_filter]

    # ── Expand STRUCTURE hits with leaf children ────────────────────
    _STRUCTURE_TYPES = {"structure", "struct_array", "STRUCTURE"}
    structure_hits = [h for h in hits if h.data_type in _STRUCTURE_TYPES][:5]

    if structure_hits:
        from imas_codex.tools.graph_search import STRUCTURE_DATA_TYPES

        parent_ids = [h.path for h in structure_hits]
        _sdt = ", ".join(f"'{dtype}'" for dtype in STRUCTURE_DATA_TYPES)
        child_results = gc.query(
            f"""
            UNWIND $parent_ids AS parent_id
            MATCH (child:IMASNode)
            WHERE child.id STARTS WITH parent_id + '/'
              AND NOT (child)-[:DEPRECATED_IN]->(:DDVersion)
              AND child.node_category IN $categories
              AND child.data_type IS NOT NULL
              AND NOT (toLower(child.data_type) IN [{_sdt}])
              AND NOT (child.id CONTAINS '_error_')
            WITH parent_id, child
            ORDER BY child.id
            WITH parent_id, collect({{
                id: child.id,
                name: child.name,
                data_type: child.data_type,
                units: child.units
            }})[..10] AS children, count(child) AS total
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

    return hits


# ── Internal helpers ────────────────────────────────────────────────────


def _embed(text: str, encoder: Encoder | None = None) -> list[float]:
    """Embed text using the provided encoder or the module-level singleton."""
    if encoder is not None:
        return encoder.embed_texts([text], prompt_name="query")[0].tolist()
    # Fall back to the module-level singleton in graph_search
    from imas_codex.tools.graph_search import _get_encoder

    enc = _get_encoder()
    return enc.embed_texts([text], prompt_name="query")[0].tolist()


def _resolve_node_categories(node_category: str | None) -> list[str]:
    """Resolve a node_category filter into a list of allowed categories.

    Mirrors :func:`imas_codex.tools.graph_search._resolve_node_categories`.
    """
    if node_category is None:
        return list(SEARCHABLE_CATEGORIES)
    requested = {c.strip() for c in node_category.split(",") if c.strip()}
    valid = requested & SEARCHABLE_CATEGORIES
    return list(valid) if valid else list(SEARCHABLE_CATEGORIES)
