"""Backing functions for standard-name search & fetch (plan 40 §5).

This module is the **single source of truth** for SN retrieval. MCP tool
wrappers in :mod:`imas_codex.llm.sn_tools` and pipeline workers in
:mod:`imas_codex.standard_names.workers` and the review modules both
import from here. No private duplicates.

Public surface (plan 40 §5.1):

- :func:`search_standard_names` — hybrid (vector + tiered grammar +
  keyword) RRF search. Mirrors :mod:`imas_codex.graph.dd_search`.
- :func:`fetch_standard_names` — fetch full entries by id list.
- :func:`find_related` — bucketed cross-relation discovery.
- :func:`check_names` — existence + Levenshtein suggestions.
- :func:`summarise_family` — family overview keyed on physical_base.

Plus 1-release deprecation aliases for plan 39's pipeline callers:

- :func:`search_similar_names` — wraps :func:`search_standard_names_vector`.
- :func:`search_similar_sns_with_full_docs` — wraps
  :func:`search_standard_names_with_documentation`.

Both emit ``DeprecationWarning`` and delegate to the canonical function.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from imas_codex.standard_names.grammar_query import (
    ALL_TIER_SEGMENTS,
    TIER_WEIGHT,
    filter_by_tier_policy,
    tier_of,
    tokenise_query,
)

logger = logging.getLogger(__name__)

# Shared executor (§5.7) — module-scoped to avoid per-call thread churn.
_STREAM_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sn-search")

#: Default RRF damping constant (Cormack et al. 2009; plan 40 §5.3).
RRF_K: int = 60


# ---------------------------------------------------------------------------
# Encoder helper (lazy, optional)
# ---------------------------------------------------------------------------


def _embed(query: str) -> list[float] | None:
    """Embed *query* into a vector or return ``None`` if the encoder is unavailable."""
    if not query or not query.strip():
        return None
    try:
        from imas_codex.embeddings.config import EncoderConfig
        from imas_codex.embeddings.encoder import Encoder

        encoder = Encoder(EncoderConfig())
        result = encoder.embed_texts([query])[0]
        return result.tolist() if hasattr(result, "tolist") else list(result)
    except Exception:
        logger.debug("Encoder unavailable; vector stream disabled", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Per-stream queries
# ---------------------------------------------------------------------------


def vector_stream(
    embedding: list[float],
    gc: Any,
    *,
    k_candidates: int = 100,
) -> list[dict]:
    """Cosine-similarity ranked SN ids via Neo4j vector index.

    Returns rows ordered by descending score (closest first).
    """
    rows = (
        gc.query(
            """
            CALL db.index.vector.queryNodes(
                'standard_name_desc_embedding', $k, $embedding
            )
            YIELD node AS sn, score
            WHERE sn.id IS NOT NULL
              AND coalesce(sn.validation_status, '') <> 'quarantined'
              AND coalesce(sn.pipeline_status, '') <> 'superseded'
              AND coalesce(sn.name_stage, '') <> 'exhausted'
            RETURN sn.id AS id, score
            ORDER BY score DESC
            """,
            embedding=embedding,
            k=k_candidates,
        )
        or []
    )
    return [dict(r) for r in rows]


def keyword_stream(
    query: str,
    gc: Any,
    *,
    k_candidates: int = 100,
) -> list[dict]:
    """Substring-match ranked SN ids over name + description + documentation.

    Cheap proxy for BM25; Neo4j core does not ship FTS5. Results are not
    score-ordered (pure existence filter); rank order is row order.
    """
    if not query or not query.strip():
        return []
    rows = (
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE toLower(sn.id) CONTAINS toLower($keyword)
               OR toLower(coalesce(sn.description, '')) CONTAINS toLower($keyword)
               OR toLower(coalesce(sn.documentation, '')) CONTAINS toLower($keyword)
            RETURN sn.id AS id
            LIMIT $k
            """,
            keyword=query,
            k=k_candidates,
        )
        or []
    )
    return [dict(r) for r in rows]


def grammar_stream(
    tokens: list[str],
    gc: Any,
    *,
    vector_hits: set[str],
    keyword_hits: set[str],
) -> list[dict]:
    """Tier-aware grammar segment matches (§5.4).

    For each of the 12 segments, finds SNs whose bare-name column matches
    any token in *tokens*. Applies the strict Tier-1/2/3 AND-gate via
    :func:`filter_by_tier_policy`, then ranks the admitted candidates by
    weighted RRF mass summed over their per-segment hits.

    Returns rows ``{id, score}`` ordered by descending RRF mass.
    """
    if not tokens:
        return []

    # Index: by_id[sn] = {tier: [(segment, rank_within_segment, weight), …]}
    by_id: dict[str, dict[int, list[tuple[str, int, float]]]] = {}

    for seg in ALL_TIER_SEGMENTS:
        tier = tier_of(seg)
        if tier == 0:
            continue
        weight = TIER_WEIGHT[tier]
        rows = (
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE sn.{seg} IN $tokens
                  AND coalesce(sn.validation_status, '') <> 'quarantined'
                  AND coalesce(sn.pipeline_status, '') <> 'superseded'
                RETURN sn.id AS id
                """,
                tokens=tokens,
            )
            or []
        )
        for rank, row in enumerate(rows):
            sn_id = row.get("id")
            if not sn_id:
                continue
            by_id.setdefault(sn_id, {}).setdefault(tier, []).append((seg, rank, weight))

    admitted = filter_by_tier_policy(by_id, vector_hits, keyword_hits)

    # Score admitted candidates by sum of weighted RRF contributions.
    scored: list[tuple[str, float]] = []
    for sn_id in admitted:
        mass = 0.0
        for tier_hits in by_id[sn_id].values():
            for _seg, rank, weight in tier_hits:
                mass += weight / (RRF_K + rank + 1)
        scored.append((sn_id, mass))

    scored.sort(key=lambda r: (-r[1], r[0]))
    return [{"id": sn_id, "score": mass} for sn_id, mass in scored]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def rrf_fuse(streams: list[list[dict]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over multiple ranked id-lists.

    Each stream contributes ``1 / (k + rank + 1)`` per id. Tie-break on
    descending fused mass, then ascending id.
    """
    mass: dict[str, float] = {}
    for stream in streams:
        for rank, row in enumerate(stream):
            sn_id = row.get("id") if isinstance(row, dict) else row
            if not sn_id:
                continue
            mass[sn_id] = mass.get(sn_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(mass.items(), key=lambda r: (-r[1], r[0]))
    return fused


# ---------------------------------------------------------------------------
# Public: search_standard_names
# ---------------------------------------------------------------------------


def _enrich_search_rows(
    gc: Any, ranked_ids: list[str], scores: dict[str, float]
) -> list[dict]:
    """Expand ranked SN ids to full result rows with description/unit/etc."""
    if not ranked_ids:
        return []
    rows = (
        gc.query(
            """
            UNWIND $ids AS sn_id
            MATCH (sn:StandardName {id: sn_id})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            RETURN sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   sn.pipeline_status AS pipeline_status,
                   sn.cocos_transformation_type AS cocos_transformation_type,
                   sn.cocos AS cocos,
                   sn.physical_base AS physical_base,
                   sn.subject AS subject
            """,
            ids=ranked_ids,
        )
        or []
    )
    by_id = {r["name"]: dict(r) for r in rows if r.get("name")}
    enriched: list[dict] = []
    for sn_id in ranked_ids:
        row = by_id.get(sn_id)
        if row is None:
            continue
        row["score"] = scores.get(sn_id, 0.0)
        enriched.append(row)
    return enriched


def search_standard_names(
    query: str,
    *,
    k: int = 20,
    mode: Literal["hybrid", "vector"] = "hybrid",
    segment_filters: dict[str, str] | None = None,
    kind: str | None = None,
    pipeline_status: str | None = None,
    cocos_type: str | None = None,
    gc: Any = None,
) -> list[dict]:
    """Hybrid (vector + keyword + tiered grammar) RRF search over StandardNames.

    Args:
        query: Natural-language query text (e.g. ``"electron temperature"``).
        k: Maximum results to return.
        mode: ``"hybrid"`` (default) runs all three streams; ``"vector"``
            skips keyword and grammar streams.
        segment_filters: Optional dict of ``{segment: token}`` — short-circuits
            into a graph-native typed-edge query (legacy behaviour).
        kind / pipeline_status / cocos_type: Post-filter constraints.
        gc: Open :class:`GraphClient`. If ``None``, opens one for the duration
            of the call.

    Returns:
        List of result dicts with keys ``name, description, documentation,
        kind, unit, pipeline_status, cocos_transformation_type, cocos,
        physical_base, subject, score``.
    """
    if not query or not query.strip():
        return []

    own_gc = False
    if gc is None:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        own_gc = True
    try:
        # ---- Segment-filter shortcut --------------------------------------
        if segment_filters:
            return _segment_filter_search(
                gc, query, k, segment_filters, kind, pipeline_status, cocos_type
            )

        # ---- Three-stream (or vector-only) RRF ----------------------------
        embedding = _embed(query)
        v_rows: list[dict] = []
        if embedding is not None:
            v_rows = vector_stream(embedding, gc, k_candidates=5 * k)

        if mode == "vector":
            ranked = [(r["id"], r.get("score", 0.0)) for r in v_rows]
        else:
            k_rows = keyword_stream(query, gc, k_candidates=5 * k)
            tokens = tokenise_query(query)
            v_set = {r["id"] for r in v_rows if r.get("id")}
            k_set = {r["id"] for r in k_rows if r.get("id")}
            g_rows = grammar_stream(tokens, gc, vector_hits=v_set, keyword_hits=k_set)
            fused = rrf_fuse([v_rows, k_rows, g_rows], k=RRF_K)
            ranked = fused

        scores = dict(ranked)
        ranked_ids = [sn_id for sn_id, _ in ranked[: 5 * k]]
        rows = _enrich_search_rows(gc, ranked_ids, scores)

        # Post-filter
        if kind:
            rows = [r for r in rows if (r.get("kind") or "").lower() == kind.lower()]
        if pipeline_status:
            rows = [
                r
                for r in rows
                if (r.get("pipeline_status") or "").lower() == pipeline_status.lower()
            ]
        if cocos_type:
            rows = [
                r
                for r in rows
                if (r.get("cocos_transformation_type") or "") == cocos_type
            ]

        return rows[:k]
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


#: Public segments accepted by ``segment_filters``. Bare-name columns are
#: the post-Phase-1 source of truth; typed edges may be unpopulated for
#: open-vocab segments (``physical_base``, ``subject``).
_SEGMENT_FILTER_COLUMNS: tuple[str, ...] = (
    "physical_base",
    "subject",
    "transformation",
    "component",
    "coordinate",
    "process",
    "position",
    "region",
    "device",
    "geometric_base",
    "object",
    "geometry",
)


def _segment_filter_search(
    gc: Any,
    query: str,
    k: int,
    segment_filters: dict[str, str],
    kind: str | None,
    pipeline_status: str | None,
    cocos_type: str | None,
) -> list[dict]:
    """Bare-name column segment-filter search (Plan 40 §5).

    Pre-v3.2 implementations matched typed grammar edges
    (``(sn)-[:HAS_PHYSICAL_BASE]->(:GrammarToken {value: ...})``).
    Open-vocabulary segments (``physical_base``, ``subject``) never have
    typed edges populated, so the typed-edge path silently returned []
    even when the bare-name column was set. We now match the
    ``sn.<segment>`` column directly, which is the post-Phase-1 source of
    truth.
    """
    params: dict[str, Any] = {"k": k}
    where: list[str] = []
    for i, (seg, val) in enumerate(segment_filters.items()):
        if seg not in _SEGMENT_FILTER_COLUMNS:
            continue
        param_key = f"seg_{i}"
        params[param_key] = val
        where.append(f"sn.{seg} = ${param_key}")
    if not where:
        return []
    cypher = (
        "MATCH (sn:StandardName)\nWHERE "
        + " AND ".join(where)
        + """
OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
RETURN sn.id AS name,
       sn.description AS description,
       sn.documentation AS documentation,
       sn.kind AS kind,
       coalesce(u.id, sn.unit) AS unit,
       sn.pipeline_status AS pipeline_status,
       sn.cocos_transformation_type AS cocos_transformation_type,
       sn.cocos AS cocos,
       sn.physical_base AS physical_base,
       sn.subject AS subject,
       1.0 AS score
LIMIT $k
"""
    )
    rows = [dict(r) for r in (gc.query(cypher, **params) or [])]
    if kind:
        rows = [r for r in rows if (r.get("kind") or "").lower() == kind.lower()]
    if pipeline_status:
        rows = [
            r
            for r in rows
            if (r.get("pipeline_status") or "").lower() == pipeline_status.lower()
        ]
    if cocos_type:
        rows = [
            r for r in rows if (r.get("cocos_transformation_type") or "") == cocos_type
        ]
    return rows


# ---------------------------------------------------------------------------
# Public: fetch_standard_names
# ---------------------------------------------------------------------------


def fetch_standard_names(
    names: list[str],
    *,
    include_grammar: bool = True,
    include_neighbours: bool = False,
    include_documentation: bool = True,
    return_fields: list[str] | None = None,
    gc: Any = None,
) -> list[dict]:
    """Fetch full entries for known standard names.

    Args:
        names: List of StandardName.id values.
        include_grammar: Project per-segment columns + typed-edge tokens
            into the result rows. Default True.
        include_neighbours: One-hop expand (predecessors, successors,
            refined-from). Default False.
        include_documentation: Project description + documentation
            (large text). Default True.
        return_fields: Whitelist of node properties to return. None →
            schema default set.
        gc: Open GraphClient.

    Returns:
        List of result dicts, one per matched name.
    """
    if not names:
        return []

    own_gc = False
    if gc is None:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        own_gc = True
    try:
        # Build select clause based on include_* flags
        select = [
            "sn.id AS name",
            "sn.kind AS kind",
            "coalesce(u.id, sn.unit) AS unit",
            "sn.pipeline_status AS pipeline_status",
            "sn.cocos_transformation_type AS cocos_transformation_type",
            "sn.cocos AS cocos",
            "sn.dd_version AS dd_version",
            "sn.links AS links",
            "sn.source_paths AS source_paths",
            "sn.constraints AS constraints",
            "sn.validity_domain AS validity_domain",
            "sn.confidence AS confidence",
            "sn.model AS model",
            "sn.description AS description",
        ]
        if include_documentation:
            select.append("sn.documentation AS documentation")
        if include_grammar:
            select.extend(
                [
                    "sn.physical_base AS physical_base",
                    "sn.subject AS subject",
                    "sn.transformation AS transformation",
                    "sn.component AS component",
                    "sn.coordinate AS coordinate",
                    "sn.position AS position",
                    "sn.process AS process",
                    "sn.region AS region",
                    "sn.device AS device",
                    "sn.geometric_base AS geometric_base",
                    "sn.object AS object",
                    "sn.geometry AS geometry",
                    "sn.grammar_parse_fallback AS grammar_parse_fallback",
                ]
            )

        cypher = f"""
            UNWIND $names AS name_id
            MATCH (sn:StandardName {{id: name_id}})
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
            OPTIONAL MATCH (src)-[:IN_IDS]->(ids:IDS)
            RETURN {", ".join(select)},
                   collect(DISTINCT src.id) AS source_ids,
                   collect(DISTINCT ids.id) AS source_ids_names
            """
        rows = [dict(r) for r in (gc.query(cypher, names=names) or [])]

        if include_neighbours and rows:
            neighbours = _fetch_neighbours(gc, [r["name"] for r in rows])
            for r in rows:
                r["neighbours"] = neighbours.get(r["name"], {})

        if return_fields is not None:
            allowed = set(return_fields)
            rows = [{k: v for k, v in r.items() if k in allowed} for r in rows]

        return rows
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


def _fetch_neighbours(gc: Any, names: list[str]) -> dict[str, dict[str, list[str]]]:
    """One-hop neighbour fetch — predecessors / successors / refined-from."""
    if not names:
        return {}
    rows = (
        gc.query(
            """
            UNWIND $names AS sn_id
            MATCH (sn:StandardName {id: sn_id})
            OPTIONAL MATCH (sn)-[:HAS_PREDECESSOR]->(p:StandardName)
            OPTIONAL MATCH (sn)-[:HAS_SUCCESSOR]->(s:StandardName)
            OPTIONAL MATCH (sn)-[:REFINED_FROM]->(rf:StandardName)
            RETURN sn.id AS name,
                   collect(DISTINCT p.id) AS predecessors,
                   collect(DISTINCT s.id) AS successors,
                   collect(DISTINCT rf.id) AS refined_from
            """,
            names=names,
        )
        or []
    )
    return {
        r["name"]: {
            "predecessors": [x for x in (r.get("predecessors") or []) if x],
            "successors": [x for x in (r.get("successors") or []) if x],
            "refined_from": [x for x in (r.get("refined_from") or []) if x],
        }
        for r in rows
    }


# ---------------------------------------------------------------------------
# Public: find_related
# ---------------------------------------------------------------------------

#: Plan 40 §7.2.3 — deterministic bucket order regardless of which contain hits.
RELATED_BUCKET_ORDER: tuple[str, ...] = (
    "Grammar Family",
    "Subject Companions",
    "Unit Companions",
    "COCOS Companions",
    "Cluster Siblings",
    "Predecessors",
    "Successors",
    "Refined-From",
    "Source Paths",
    "Source Signals",
)


def find_related(
    name: str,
    *,
    relationship_types: str = "all",
    max_results: int = 20,
    gc: Any = None,
) -> dict[str, list[dict]]:
    """Bucketed cross-relationship discovery for a StandardName.

    Args:
        name: StandardName.id to centre the discovery on.
        relationship_types: ``"all"|"grammar"|"unit"|"cocos"|"cluster"|"lineage"|"source"``.
        max_results: Cap per bucket.
        gc: Open GraphClient.

    Returns:
        Dict bucket-name → list of related-name dicts. Empty buckets are
        suppressed entirely (not returned). Buckets are returned in the
        deterministic order defined by :data:`RELATED_BUCKET_ORDER`.
    """
    own_gc = False
    if gc is None:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        own_gc = True
    try:
        wanted = _resolve_related_types(relationship_types)
        out: dict[str, list[dict]] = {}

        if "grammar" in wanted:
            rows = (
                gc.query(
                    """
                    MATCH (anchor:StandardName {id: $name})
                    WHERE anchor.physical_base IS NOT NULL
                    MATCH (sn:StandardName)
                    WHERE sn.physical_base = anchor.physical_base
                      AND sn.id <> anchor.id
                    RETURN sn.id AS name, sn.description AS description
                    ORDER BY sn.id
                    LIMIT $k
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            if rows:
                out["Grammar Family"] = [dict(r) for r in rows]

            rows = (
                gc.query(
                    """
                    MATCH (anchor:StandardName {id: $name})
                    WHERE anchor.subject IS NOT NULL
                    MATCH (sn:StandardName)
                    WHERE sn.subject = anchor.subject
                      AND sn.id <> anchor.id
                    RETURN sn.id AS name, sn.description AS description
                    ORDER BY sn.id
                    LIMIT $k
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            if rows:
                out["Subject Companions"] = [dict(r) for r in rows]

        if "unit" in wanted:
            rows = (
                gc.query(
                    """
                    MATCH (anchor:StandardName {id: $name})-[:HAS_UNIT]->(u:Unit)
                    MATCH (sn:StandardName)-[:HAS_UNIT]->(u)
                    WHERE sn.id <> anchor.id
                    RETURN sn.id AS name, sn.description AS description
                    ORDER BY sn.id
                    LIMIT $k
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            if rows:
                out["Unit Companions"] = [dict(r) for r in rows]

        if "cocos" in wanted:
            rows = (
                gc.query(
                    """
                    MATCH (anchor:StandardName {id: $name})
                    WHERE anchor.cocos_transformation_type IS NOT NULL
                    MATCH (sn:StandardName)
                    WHERE sn.cocos_transformation_type = anchor.cocos_transformation_type
                      AND sn.id <> anchor.id
                    RETURN sn.id AS name, sn.description AS description
                    ORDER BY sn.id
                    LIMIT $k
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            if rows:
                out["COCOS Companions"] = [dict(r) for r in rows]

        if "cluster" in wanted:
            rows = (
                gc.query(
                    """
                    MATCH (anchor:StandardName {id: $name})-[:IN_CLUSTER]->(c)
                    MATCH (sn:StandardName)-[:IN_CLUSTER]->(c)
                    WHERE sn.id <> anchor.id
                    RETURN sn.id AS name, sn.description AS description
                    ORDER BY sn.id
                    LIMIT $k
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            if rows:
                out["Cluster Siblings"] = [dict(r) for r in rows]

        if "lineage" in wanted:
            for rel, bucket in (
                ("HAS_PREDECESSOR", "Predecessors"),
                ("HAS_SUCCESSOR", "Successors"),
                ("REFINED_FROM", "Refined-From"),
            ):
                rows = (
                    gc.query(
                        f"""
                        MATCH (anchor:StandardName {{id: $name}})-[:{rel}]->(sn:StandardName)
                        RETURN sn.id AS name, sn.description AS description
                        ORDER BY sn.id
                        LIMIT $k
                        """,
                        name=name,
                        k=max_results,
                    )
                    or []
                )
                if rows:
                    out[bucket] = [dict(r) for r in rows]

        if "source" in wanted:
            rows = (
                gc.query(
                    """
                    MATCH (n)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: $name})
                    WITH n, sn LIMIT $k
                    OPTIONAL MATCH (n)-[:IN_IDS]->(ids:IDS)
                    RETURN n.id AS name, ids.id AS description
                    """,
                    name=name,
                    k=max_results,
                )
                or []
            )
            paths = [
                {"name": r["name"], "description": r.get("description") or ""}
                for r in rows
                if r.get("name") and "/" in str(r["name"])
            ]
            signals = [
                {"name": r["name"], "description": r.get("description") or ""}
                for r in rows
                if r.get("name") and "/" not in str(r["name"])
            ]
            if paths:
                out["Source Paths"] = paths
            if signals:
                out["Source Signals"] = signals

        # Re-order via RELATED_BUCKET_ORDER
        ordered: dict[str, list[dict]] = {}
        for bucket in RELATED_BUCKET_ORDER:
            if bucket in out:
                ordered[bucket] = out[bucket]
        return ordered
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


def _resolve_related_types(relationship_types: str) -> set[str]:
    """Map the public ``relationship_types`` arg to internal bucket-group set."""
    rt = (relationship_types or "all").strip().lower()
    if rt in ("all", "*", ""):
        return {"grammar", "unit", "cocos", "cluster", "lineage", "source"}
    return {p.strip() for p in rt.split(",") if p.strip()}


# ---------------------------------------------------------------------------
# Public: check_names
# ---------------------------------------------------------------------------


def check_names(
    names: list[str],
    *,
    gc: Any = None,
    suggestion_pool_limit: int = 5000,
) -> list[dict]:
    """Validate that names exist in the StandardName catalogue.

    For each input, returns ``{name, exists, suggestion, reason}`` where
    ``suggestion`` is the closest-Levenshtein candidate when
    ``exists=False``. Grammar-share is a tiebreaker only (plan §7.2.6).
    """
    if not names:
        return []

    own_gc = False
    if gc is None:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        own_gc = True
    try:
        all_rows = (
            gc.query(
                """
                MATCH (sn:StandardName)
                RETURN sn.id AS id, sn.physical_base AS physical_base
                LIMIT $cap
                """,
                cap=suggestion_pool_limit,
            )
            or []
        )
        catalog = {r["id"]: r.get("physical_base") for r in all_rows if r.get("id")}

        results: list[dict] = []
        for raw in names:
            n = (raw or "").strip()
            if not n:
                continue
            if n in catalog:
                results.append(
                    {"name": n, "exists": True, "suggestion": "", "reason": ""}
                )
                continue
            best = _levenshtein_suggest(n, catalog)
            results.append(
                {
                    "name": n,
                    "exists": False,
                    "suggestion": best["suggestion"],
                    "reason": best["reason"],
                }
            )
        return results
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein distance — small enough for catalogue scale."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def _levenshtein_suggest(query: str, catalog: dict[str, str | None]) -> dict[str, str]:
    """Return ``{suggestion, reason}`` per the §7.2.6 tiebreak rule."""
    if not catalog:
        return {"suggestion": "", "reason": "no_catalog"}
    distances = sorted(
        ((cand, _levenshtein(query, cand)) for cand in catalog),
        key=lambda r: (r[1], r[0]),
    )[:5]
    if not distances:
        return {"suggestion": "", "reason": "no_catalog"}
    smallest = distances[0][1]
    tied = [c for c, d in distances if d == smallest]
    if len(tied) == 1:
        return {"suggestion": tied[0], "reason": "levenshtein"}

    # Ties → grammar tiebreak (parse query, prefer same physical_base)
    try:
        from imas_standard_names.grammar import parse_standard_name

        parsed = parse_standard_name(query)
        q_pb = getattr(parsed, "physical_base", None)
        q_pb = getattr(q_pb, "value", q_pb)
    except Exception:
        q_pb = None

    if q_pb:
        same_pb = [c for c in tied if catalog.get(c) == q_pb]
        if same_pb:
            return {
                "suggestion": sorted(same_pb)[0],
                "reason": "levenshtein+grammar_tiebreak",
            }
    return {"suggestion": sorted(tied)[0], "reason": "levenshtein+lex_tiebreak"}


# ---------------------------------------------------------------------------
# Public: summarise_family
# ---------------------------------------------------------------------------


def summarise_family(
    physical_base: str,
    *,
    gc: Any = None,
) -> dict[str, Any]:
    """Family overview keyed on ``physical_base`` (plan §7.2.7).

    Returns:
        Dict with keys: ``physical_base, count, segment_distinct,
        unit_distinct, cocos_distinct, physics_domain_distinct,
        sample_names, lineage`` (plan 40 §7.2.7 lineage subsection).
    """
    own_gc = False
    if gc is None:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
        own_gc = True
    try:
        rows = (
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.physical_base = $pb
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS name,
                       sn.subject AS subject,
                       sn.transformation AS transformation,
                       sn.component AS component,
                       sn.position AS position,
                       sn.process AS process,
                       sn.geometric_base AS geometric_base,
                       sn.physics_domain AS physics_domain,
                       sn.cocos_transformation_type AS cocos_transformation_type,
                       coalesce(u.id, sn.unit) AS unit
                ORDER BY sn.id
                """,
                pb=physical_base,
            )
            or []
        )
        rows = [dict(r) for r in rows]

        def _distinct(field: str) -> list[str]:
            return sorted(
                {
                    str(r[field])
                    for r in rows
                    if r.get(field) is not None and r.get(field) != ""
                }
            )

        result = {
            "physical_base": physical_base,
            "count": len(rows),
            "segment_distinct": {
                "subject": _distinct("subject"),
                "transformation": _distinct("transformation"),
                "component": _distinct("component"),
                "position": _distinct("position"),
                "process": _distinct("process"),
                "geometric_base": _distinct("geometric_base"),
            },
            "unit_distinct": _distinct("unit"),
            "cocos_distinct": _distinct("cocos_transformation_type"),
            "physics_domain_distinct": _distinct("physics_domain"),
            "sample_names": [r["name"] for r in rows[:5]],
            "lineage": _lineage_counts(gc, physical_base),
        }
        return result
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


def _lineage_counts(gc: Any, physical_base: str) -> dict[str, int]:
    """Plan §7.2.7 lineage subsection — counts + max chain depth per relation."""
    out: dict[str, int] = {
        "predecessors_count": 0,
        "predecessors_max_depth": 0,
        "successors_count": 0,
        "successors_max_depth": 0,
        "refined_from_count": 0,
        "refined_from_max_depth": 0,
        "total_edges": 0,
    }
    for rel, count_key, depth_key in (
        ("HAS_PREDECESSOR", "predecessors_count", "predecessors_max_depth"),
        ("HAS_SUCCESSOR", "successors_count", "successors_max_depth"),
        ("REFINED_FROM", "refined_from_count", "refined_from_max_depth"),
    ):
        try:
            rows = (
                gc.query(
                    f"""
                    MATCH (sn:StandardName)
                    WHERE sn.physical_base = $pb
                    OPTIONAL MATCH (sn)-[:{rel}]->(:StandardName)
                    WITH sn, count(*) AS edges
                    RETURN sum(CASE WHEN edges > 0 THEN 1 ELSE 0 END) AS c,
                           sum(edges) AS total
                    """,
                    pb=physical_base,
                )
                or []
            )
            if rows:
                out[count_key] = int(rows[0].get("c") or 0)
                out["total_edges"] += int(rows[0].get("total") or 0)
            depth_rows = (
                gc.query(
                    f"""
                    MATCH (sn:StandardName) WHERE sn.physical_base = $pb
                    OPTIONAL MATCH path = (sn)-[:{rel}*1..10]->(:StandardName)
                    RETURN coalesce(max(length(path)), 0) AS d
                    """,
                    pb=physical_base,
                )
                or []
            )
            if depth_rows:
                out[depth_key] = int(depth_rows[0].get("d") or 0)
        except Exception:
            logger.debug("lineage query failed for %s", rel, exc_info=True)
    return out


# ---------------------------------------------------------------------------
# Plan-39 worker entry points (canonical names)
# ---------------------------------------------------------------------------


def search_standard_names_vector(
    query: str,
    k: int = 5,
    *,
    gc: Any = None,
    include_superseded: bool = False,
) -> list[dict[str, Any]]:
    """Pure-vector SN search returning the legacy result schema.

    Replaces :func:`search_similar_names`. Uses the encoder + Neo4j
    vector index, filters out problematic lifecycle states, and returns
    ``[{id, description, kind, unit, score}, …]``.

    Parameters
    ----------
    gc:
        Optional pre-opened :class:`GraphClient`. When provided it is
        re-used (no new session opened, no close); when ``None`` the
        function opens its own short-lived session. Plan 39 §3.6 (a):
        the catalog runners pass the refine-cycle's ``gc`` so a single
        refine cycle never instantiates more than one ``GraphClient``.
    include_superseded:
        When ``True``, drop the ``pipeline_status='superseded'``
        exclusion. Used by cycle-2 refine fan-out to surface the
        just-superseded cycle-1 name as a comparator. Default ``False``
        preserves existing behaviour for all other callers.
    """
    if not query or not query.strip():
        return []

    own_gc = False
    _gc_ctx: Any = None
    if gc is None:
        try:
            from imas_codex.graph.client import GraphClient

            _gc_ctx = GraphClient()
            gc = _gc_ctx.__enter__() if hasattr(_gc_ctx, "__enter__") else _gc_ctx
            own_gc = True
        except Exception:
            logger.debug("GraphClient unavailable", exc_info=True)
            return []

    try:
        embedding = _embed(query)
        if embedding is None:
            return []
        # Build the lifecycle WHERE clause; superseded is conditionally dropped.
        lifecycle_clauses = [
            "sn.id IS NOT NULL",
            "coalesce(sn.validation_status, '') <> 'quarantined'",
            "coalesce(sn.name_stage, '') <> 'exhausted'",
        ]
        if not include_superseded:
            lifecycle_clauses.insert(
                2, "coalesce(sn.pipeline_status, '') <> 'superseded'"
            )
        where_clause = " AND ".join(lifecycle_clauses)
        rows = (
            gc.query(
                f"""
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $k, $embedding
                )
                YIELD node AS sn, score
                WHERE {where_clause}
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS id,
                       sn.description AS description,
                       sn.kind AS kind,
                       coalesce(u.id, sn.unit) AS unit,
                       score
                ORDER BY score DESC
                """,
                embedding=embedding,
                k=k + 10,
            )
            or []
        )
        return [dict(r) for r in rows][:k]
    except Exception:
        logger.debug("Vector SN search failed", exc_info=True)
        return []
    finally:
        if own_gc:
            try:
                if hasattr(_gc_ctx, "__exit__"):
                    _gc_ctx.__exit__(None, None, None)
                else:
                    gc.close()
            except Exception:
                pass


def search_standard_names_with_documentation(
    description_query: str,
    k: int = 5,
    exclude_ids: list[str] | None = None,
    *,
    gc: Any = None,
) -> list[dict[str, Any]]:
    """Vector SN search returning richer records (description + documentation)."""
    if not description_query or not description_query.strip():
        return []
    exclude_set = set(exclude_ids) if exclude_ids else set()
    own_gc = False
    if gc is None:
        try:
            from imas_codex.graph.client import GraphClient

            gc = GraphClient()
            own_gc = True
        except Exception:
            logger.debug("GraphClient unavailable", exc_info=True)
            return []
    try:
        embedding = _embed(description_query)
        if embedding is None:
            return []
        rows = (
            gc.query(
                """
                CALL db.index.vector.queryNodes(
                    'standard_name_desc_embedding', $k, $embedding
                )
                YIELD node AS sn, score
                WHERE sn.id IS NOT NULL
                  AND sn.validation_status = 'valid'
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS name,
                       sn.description AS description,
                       sn.documentation AS documentation,
                       coalesce(u.id, sn.unit) AS unit,
                       score
                ORDER BY score DESC
                """,
                embedding=embedding,
                k=k + len(exclude_set) + 5,
            )
            or []
        )
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            nm = d.get("name") or ""
            if nm in exclude_set:
                continue
            out.append(
                {
                    "name": nm,
                    "description": d.get("description") or "",
                    "documentation": d.get("documentation") or "",
                    "unit": d.get("unit") or "1",
                }
            )
            if len(out) >= k:
                break
        return out
    except Exception:
        logger.debug("Doc-rich SN search failed", exc_info=True)
        return []
    finally:
        if own_gc:
            try:
                gc.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Deprecation aliases (plan §15.1 — alias-bridge window, 1 release)
# ---------------------------------------------------------------------------


def search_similar_names(query: str, k: int = 5) -> list[dict[str, Any]]:
    """DEPRECATED: use :func:`search_standard_names_vector`.

    Plan 40 §15.1 alias-bridge window. Wrapper emits ``DeprecationWarning``
    on first call and delegates to the canonical name. Removed in the
    release after Phase 4.
    """
    warnings.warn(
        "search_similar_names is deprecated; use search_standard_names_vector.",
        DeprecationWarning,
        stacklevel=2,
    )
    return search_standard_names_vector(query, k=k)


def search_similar_sns_with_full_docs(
    description_query: str,
    k: int = 5,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """DEPRECATED: use :func:`search_standard_names_with_documentation`."""
    warnings.warn(
        "search_similar_sns_with_full_docs is deprecated; use "
        "search_standard_names_with_documentation.",
        DeprecationWarning,
        stacklevel=2,
    )
    return search_standard_names_with_documentation(
        description_query, k=k, exclude_ids=exclude_ids
    )


# ---------------------------------------------------------------------------
# Async stream wrapper (plan §5.7) — exported for advanced callers
# ---------------------------------------------------------------------------


async def _run_streams_concurrent(
    query: str, embedding: list[float] | None, gc: Any, *, k_candidates: int = 100
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run vector / keyword / grammar streams concurrently in the shared pool."""
    loop = asyncio.get_running_loop()
    tokens = tokenise_query(query)

    def _v() -> list[dict]:
        return (
            vector_stream(embedding, gc, k_candidates=k_candidates) if embedding else []
        )

    def _k() -> list[dict]:
        return keyword_stream(query, gc, k_candidates=k_candidates)

    # Grammar needs vector/keyword sets — run vector + keyword first, then grammar
    v_task = loop.run_in_executor(_STREAM_POOL, _v)
    k_task = loop.run_in_executor(_STREAM_POOL, _k)
    v_rows, k_rows = await asyncio.gather(v_task, k_task)
    v_set = {r["id"] for r in v_rows if r.get("id")}
    k_set = {r["id"] for r in k_rows if r.get("id")}

    def _g() -> list[dict]:
        return grammar_stream(tokens, gc, vector_hits=v_set, keyword_hits=k_set)

    g_rows = await loop.run_in_executor(_STREAM_POOL, _g)
    return v_rows, k_rows, g_rows


__all__ = [
    "search_standard_names",
    "fetch_standard_names",
    "find_related",
    "check_names",
    "summarise_family",
    "search_standard_names_vector",
    "search_standard_names_with_documentation",
    "search_similar_names",  # deprecated alias
    "search_similar_sns_with_full_docs",  # deprecated alias
    "vector_stream",
    "keyword_stream",
    "grammar_stream",
    "rrf_fuse",
    "RELATED_BUCKET_ORDER",
    "RRF_K",
]
