"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.defaults import (
    DEFAULT_MIN_SCORE,
    DEFAULT_REFINE_ROTATIONS,
)

logger = logging.getLogger(__name__)


def _ensure_json(value: Any) -> str | None:
    """Ensure a value is a JSON string, not a raw dict/list (Neo4j rejects Maps)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _ensure_list(value: Any) -> list[str]:
    """Coerce *value* to a list of strings (for multi-valued fields like physics_domain).

    - ``None`` / empty string → ``[]``
    - scalar string → ``[value]``
    - list → passed through
    """
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _parse_grammar_vnext(name: str) -> dict[str, str | None]:
    """Parse ``name`` with the vNext ISN grammar API.

    Returns a dict with ``grammar_parse_version`` (ISN package version string)
    and ``validation_diagnostics_json`` (JSON array of diagnostic objects).
    The parse version is always set when ISN is available; diagnostics default
    to ``"[]"`` on parse failure so the field is never ``null`` after the first
    successful stamp.
    """
    try:
        import dataclasses

        import imas_standard_names
        from imas_standard_names.grammar.parser import ParseError, parse

        version: str = imas_standard_names.__version__
    except ImportError:
        return {"grammar_parse_version": None, "validation_diagnostics_json": None}

    try:
        result = parse(name)
        diags = json.dumps([dataclasses.asdict(d) for d in result.diagnostics])
    except ParseError:
        logger.debug(
            "vNext grammar parse rejected '%s' — storing empty diagnostics", name
        )
        diags = "[]"
    except Exception:
        logger.debug(
            "vNext grammar parse failed for '%s' — storing empty diagnostics", name
        )
        diags = "[]"

    return {"grammar_parse_version": version, "validation_diagnostics_json": diags}


def _compute_link_status(links: list[str] | None) -> str | None:
    """Determine link resolution status from link prefixes.

    Returns 'resolved' if all links are ``name:`` or URL prefixed,
    'unresolved' if any link has ``dd:`` prefix (pending resolution),
    or None if no links exist.
    """
    if not links:
        return None
    has_unresolved = any(link.startswith("dd:") for link in links)
    return "unresolved" if has_unresolved else "resolved"


# =============================================================================
# Pipeline cost model — bucket queries & historical CPI
# =============================================================================

# Module-level TTL cache for pipeline buckets.
_pipeline_buckets_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}

# Module-level cache for historical CPI (once per process).
_historical_cpi_cache: dict[str, Any] = {
    "value": None,
    "ts": 0.0,
}


def query_pipeline_buckets(
    threshold: float = 0.65,
    cap: int = 3,
    cache_ttl: float = 3.0,
) -> Any:
    """Query the 6 disjoint pipeline buckets from the graph.

    Uses a module-level time-based cache (TTL = *cache_ttl* seconds).

    Bucket definitions (disjoint by construction):

    - **A**: StandardNameSource nodes with ``status='extracted'``.
    - **B**: StandardName with ``name_stage='drafted'`` and no
      ``reviewer_score_name``.
    - **C**: StandardName reviewed/refining, below *threshold*, under
      *cap* → definite refine + re-review.
    - **D**: StandardName accepted, docs not yet drafted.
    - **E**: StandardName with ``docs_stage='drafted'``, no
      ``reviewer_score_docs``.
    - **F**: StandardName docs reviewed/refining, below *threshold*,
      under *cap* → definite docs refine.

    Args:
        threshold: Score threshold below which a name/docs enters
            the refine pool (default 0.65).
        cap: Maximum refine rotations before exhaustion (default 3).
        cache_ttl: Cache time-to-live in seconds (default 3.0).

    Returns:
        :class:`PipelineBuckets` with the 6 counts.
    """
    import time as _time

    from imas_codex.standard_names.cost_model import PipelineBuckets

    now = _time.time()
    if (
        _pipeline_buckets_cache["value"] is not None
        and now - _pipeline_buckets_cache["ts"] < cache_ttl
    ):
        return _pipeline_buckets_cache["value"]

    with GraphClient() as gc:
        # Bucket A: pre-draft sources
        r_a = gc.query(
            "MATCH (sns:StandardNameSource {status: 'extracted'}) "
            "RETURN count(*) AS cnt"
        )
        a = r_a[0]["cnt"] if r_a else 0

        # Bucket B: drafted, not yet name-reviewed
        r_b = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage = 'drafted' "
            "AND sn.reviewer_score_name IS NULL "
            "RETURN count(*) AS cnt"
        )
        b = r_b[0]["cnt"] if r_b else 0

        # Bucket C: reviewed/refining, below threshold, under cap
        r_c = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage IN ['reviewed', 'refining'] "
            "AND coalesce(sn.reviewer_score_name, 0.0) < $threshold "
            "AND coalesce(sn.regen_count, 0) < $cap "
            "RETURN count(*) AS cnt",
            threshold=threshold,
            cap=cap,
        )
        c = r_c[0]["cnt"] if r_c else 0

        # Bucket D: name accepted, docs not yet drafted
        r_d = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.name_stage = 'accepted' "
            "AND (sn.docs_stage IS NULL OR sn.docs_stage = 'pending') "
            "RETURN count(*) AS cnt"
        )
        d = r_d[0]["cnt"] if r_d else 0

        # Bucket E: docs drafted, not yet reviewed
        r_e = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.docs_stage = 'drafted' "
            "AND sn.reviewer_score_docs IS NULL "
            "RETURN count(*) AS cnt"
        )
        e = r_e[0]["cnt"] if r_e else 0

        # Bucket F: docs reviewed/refining, below threshold, under cap
        # Schema note: plan.md referenced refine_docs_count but actual
        # schema field is docs_chain_length.
        r_f = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.docs_stage IN ['reviewed', 'refining'] "
            "AND coalesce(sn.reviewer_score_docs, 0.0) < $threshold "
            "AND coalesce(sn.docs_chain_length, 0) < $cap "
            "RETURN count(*) AS cnt",
            threshold=threshold,
            cap=cap,
        )
        f = r_f[0]["cnt"] if r_f else 0

    result = PipelineBuckets(
        a_sources=a,
        b_drafted_unreviewed=b,
        c_refine_pending=c,
        d_accepted_no_docs=d,
        e_docs_unreviewed=e,
        f_refine_docs_pending=f,
    )
    _pipeline_buckets_cache["value"] = result
    _pipeline_buckets_cache["ts"] = now
    return result


def query_historical_cpi(model: str | None = None) -> dict[str, float]:
    """Average per-phase LLM cost across past StandardName nodes.

    Returns dict keyed by pool name (``generate_name``, ``review_name``,
    ``refine_name``, ``generate_docs``, ``review_docs``, ``refine_docs``).
    Missing keys = no historical data for that pool.

    Cached for the lifetime of the process (historical data is static
    within a run).

    Args:
        model: Optional LLM model filter. If provided, restricts to
            nodes composed with that model.
    """
    import time as _time

    # Process-lifetime cache (historical data is static within a run)
    if _historical_cpi_cache["value"] is not None:
        return _historical_cpi_cache["value"]

    result: dict[str, float] = {}
    with GraphClient() as gc:
        # generate_name: avg compose cost per name
        rows = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.llm_cost_compose IS NOT NULL "
            "AND sn.llm_cost_compose > 0 "
            "AND coalesce(sn.compose_count, 0) > 0 "
            "RETURN avg(sn.llm_cost_compose / sn.compose_count) AS cpi, "
            "count(*) AS n"
        )
        if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
            result["generate_name"] = float(rows[0]["cpi"])

        # review_name: avg review cost for name-reviewed nodes
        rows = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.llm_cost_review IS NOT NULL "
            "AND sn.llm_cost_review > 0 "
            "AND sn.reviewer_score_name IS NOT NULL "
            "RETURN avg(sn.llm_cost_review) AS cpi, "
            "count(*) AS n"
        )
        if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
            result["review_name"] = float(rows[0]["cpi"])

        # refine_name: avg regen cost per refine cycle
        rows = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.llm_cost_regen IS NOT NULL "
            "AND sn.llm_cost_regen > 0 "
            "AND coalesce(sn.regen_count, 0) > 0 "
            "RETURN avg(sn.llm_cost_regen / sn.regen_count) AS cpi, "
            "count(*) AS n"
        )
        if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
            result["refine_name"] = float(rows[0]["cpi"])

        # generate_docs: avg docs compose cost
        rows = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.llm_cost_docs IS NOT NULL "
            "AND sn.llm_cost_docs > 0 "
            "AND sn.docs_stage IS NOT NULL "
            "RETURN avg(sn.llm_cost_docs) AS cpi, "
            "count(*) AS n"
        )
        if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
            result["generate_docs"] = float(rows[0]["cpi"])

        # review_docs: use llm_cost_review for docs-reviewed nodes as proxy
        rows = gc.query(
            "MATCH (sn:StandardName) "
            "WHERE sn.llm_cost_review IS NOT NULL "
            "AND sn.llm_cost_review > 0 "
            "AND sn.reviewer_score_docs IS NOT NULL "
            "RETURN avg(sn.llm_cost_review) AS cpi, "
            "count(*) AS n"
        )
        if rows and rows[0]["n"] > 0 and rows[0]["cpi"] is not None:
            result["review_docs"] = float(rows[0]["cpi"])

        # refine_docs: no dedicated cost field — will fall to sibling

    _historical_cpi_cache["value"] = result
    _historical_cpi_cache["ts"] = _time.time()
    return result


# =============================================================================
# Read helpers — extraction candidates
# =============================================================================


def get_extraction_candidates_dd(
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Query IMASNode paths grouped by semantic cluster.

    Returns dynamic leaf nodes that have been enriched (status=embedded),
    optionally filtered by IDS or physics domain.
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {"limit": limit}
        where_clauses = [
            "n.node_type IN ['dynamic', 'constant']",
            "n.description IS NOT NULL",
            "n.description <> ''",
        ]

        if ids_filter:
            where_clauses.append("ids.id = $ids_filter")
            params["ids_filter"] = ids_filter
        if domain_filter:
            where_clauses.append("n.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where = " AND ".join(where_clauses)
        results = gc.query(
            f"""
            MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
            WHERE {where}
            WITH n, ids
            OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            RETURN n.id AS path, n.description AS description,
                   n.unit AS unit, n.data_type AS data_type,
                   ids.id AS ids_name, c.label AS cluster_label
            ORDER BY ids.id, n.id
            LIMIT $limit
            """,
            **params,
        )
        return list(results)


def get_extraction_candidates_signals(
    facility: str,
    domain_filter: str | None = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Query FacilitySignal nodes for a given facility.

    Returns signals that have been enriched, optionally filtered by
    physics domain.
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {"facility": facility, "limit": limit}
        where_clauses = ["s.status = 'enriched'"]

        if domain_filter:
            where_clauses.append("s.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where = " AND ".join(where_clauses)
        results = gc.query(
            f"""
            MATCH (s:FacilitySignal)-[:AT_FACILITY]->(f:Facility {{id: $facility}})
            WHERE {where}
            WITH s
            OPTIONAL MATCH (s)-[:MAPS_TO]->(m:IMASNode)
            RETURN s.id AS signal_id, s.description AS description,
                   s.physics_domain AS physics_domain,
                   s.units AS units,
                   m.id AS imas_path
            ORDER BY s.id
            LIMIT $limit
            """,
            **params,
        )
        return list(results)


# =============================================================================
# Deduplication
# =============================================================================


def get_existing_standard_names() -> set[str]:
    """Return the set of existing StandardName node IDs for deduplication."""
    with GraphClient() as gc:
        results = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}


def get_named_source_ids() -> set[str]:
    """Return source IDs already linked via HAS_STANDARD_NAME.

    Used for resumability: extract skips sources that already have
    a standard name unless --force is specified.
    """
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN DISTINCT src.id AS source_id
        """)
        return {r["source_id"] for r in results}


def get_source_name_mapping(*, rich: bool = False) -> dict[str, dict]:
    """Return mapping of source_id → previous standard name details.

    Used by extract_worker in --force mode to inject per-path
    previous_name context so the LLM can improve on prior names.

    Args:
        rich: If True, return full SN metadata including documentation,
            links, and all linked DD paths. Used by --paths mode.

    Returns:
        Dict mapping source entity ID to dict with keys:
        name, description, kind, pipeline_status (and more if rich=True).
        If a source has multiple names, prefers the accepted one.
    """
    if rich:
        return _get_rich_source_name_mapping()

    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN src.id AS source_id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.kind AS kind,
                   sn.pipeline_status AS pipeline_status
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            # If multiple names exist for same source, prefer accepted
            if sid not in mapping or r.get("pipeline_status") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "kind": r.get("kind"),
                    "pipeline_status": r.get("pipeline_status"),
                }
        return mapping


def _get_rich_source_name_mapping() -> dict[str, dict]:
    """Full SN metadata with documentation + all linked DD paths."""
    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (other_src)-[:HAS_STANDARD_NAME]->(sn)
            WHERE other_src <> src
            RETURN src.id AS source_id,
                   sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   sn.links AS links,
                   sn.pipeline_status AS pipeline_status,
                   sn.reviewer_score_name AS reviewer_score,
                   sn.review_tier AS review_tier,
                   sn.validation_issues AS validation_issues,
                   u.id AS unit,
                   collect(DISTINCT other_src.id) AS linked_dd_paths
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            if sid not in mapping or r.get("pipeline_status") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "documentation": r.get("documentation"),
                    "kind": r.get("kind"),
                    "links": r.get("links"),
                    "pipeline_status": r.get("pipeline_status"),
                    "reviewer_score": r.get("reviewer_score"),
                    "review_tier": r.get("review_tier"),
                    "validation_issues": r.get("validation_issues"),
                    "unit": r.get("unit"),
                    "linked_dd_paths": [
                        p for p in (r.get("linked_dd_paths") or []) if p != sid
                    ],
                }
        return mapping


# =============================================================================
# Write helpers
# =============================================================================


def fetch_low_score_sources(
    *,
    min_score: float | None = None,
    domain: str | None = None,
    ids: str | None = None,
    limit: int | None = None,
    source_type: str = "dd",
) -> list[dict[str, Any]]:
    """Enumerate sources whose linked StandardName has ``reviewer_score_name < min_score``.

    Walks ``(:IMASNode|:FacilitySignal)-[:HAS_STANDARD_NAME]->(:StandardName)``
    and returns the originating source IDs along with the reviewer feedback
    needed to prompt a targeted regeneration. Used by the extract worker's
    regen mode (``sn run --min-score F``) to re-queue the exact sources
    whose names scored below the threshold.

    Only reviewed names are considered (``reviewer_score_name IS NOT NULL``), so
    unreviewed names are never pulled into regen. Duplicates per source_id
    are collapsed, keeping the lowest-scoring entry ("worst critique wins").

    Args:
        min_score: Reviewer-score threshold (0-1). Names with
            ``reviewer_score_name < min_score`` are returned. Required for any
            results; None short-circuits to an empty list.
        domain: Optional physics-domain filter applied to the StandardName.
        ids: Optional IDS-name filter (DD source only).
        limit: Optional cap on rows returned (ordered by worst score first).
        source_type: ``"dd"`` or ``"signals"``.

    Returns:
        List of dicts with ``source_id``, ``source_type``, ``review_feedback``.
    """
    if min_score is None:
        return []

    if source_type == "dd":
        match_clause = "MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
    elif source_type == "signals":
        match_clause = (
            "MATCH (src:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)"
        )
    else:
        raise ValueError(f"source_type must be 'dd' or 'signals', got {source_type!r}")

    where_clauses = [
        "sn.reviewer_score_name IS NOT NULL",
        "sn.reviewer_score_name < $min_score",
        "coalesce(sn.regen_count, 0) < 1",
    ]
    params: dict[str, Any] = {"min_score": float(min_score)}

    if domain:
        where_clauses.append("$domain IN sn.physics_domain")
        params["domain"] = domain

    if ids and source_type == "dd":
        match_clause += "\n            MATCH (src)-[:IN_IDS]->(ids_node:IDS)"
        where_clauses.append("ids_node.id = $ids")
        params["ids"] = ids

    query = f"""
        {match_clause}
        WHERE {" AND ".join(where_clauses)}
        RETURN src.id AS source_id,
               sn.id AS previous_name,
               sn.description AS previous_description,
               sn.documentation AS previous_documentation,
               sn.reviewer_score_name AS reviewer_score,
               sn.review_tier AS review_tier,
               sn.reviewer_comments_name AS reviewer_comments,
               sn.reviewer_scores_name AS reviewer_scores_json,
               sn.validation_status AS validation_status
        ORDER BY coalesce(sn.reviewer_score_name, 1.0) ASC, src.id ASC
    """
    if limit is not None and limit > 0:
        query += "\n        LIMIT $limit"
        params["limit"] = int(limit)

    with GraphClient() as gc:
        rows = list(gc.query(query, **params))

    # Collapse duplicates: keep the worst-scoring feedback per source_id.
    by_source: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores: dict[str, Any] | None = None
        if scores_json:
            try:
                scores = json.loads(scores_json)
            except (TypeError, ValueError):
                scores = None
        feedback = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores,
            "validation_status": row.get("validation_status"),
        }
        existing = by_source.get(sid)
        if existing is None:
            by_source[sid] = feedback
            order.append(sid)
        else:
            prev = existing.get("reviewer_score")
            cur = feedback.get("reviewer_score")
            if prev is None or (cur is not None and cur < prev):
                by_source[sid] = feedback

    return [
        {
            "source_id": sid,
            "source_type": source_type,
            "review_feedback": by_source[sid],
        }
        for sid in order
    ]


def fetch_review_feedback_for_sources(
    source_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch prior reviewer feedback for a batch of StandardNameSource ids.

    Used by the compose worker when a regeneration run is invoked with
    ``--min-score F``. Each returned dict carries enough context
    to let the LLM understand what the previous reviewer objected to and
    adjust the new candidate accordingly.

    Args:
        source_ids: Iterable of source node ids (e.g. ``dd:equilibrium/...``
            or ``signals:tcv:...``). ``None`` or empty input returns ``{}``
            without hitting the graph.

    Returns:
        ``{source_id: feedback_dict}`` where feedback_dict has keys:

        - ``previous_name`` (str | None): prior standard-name id
        - ``previous_description`` (str | None)
        - ``previous_documentation`` (str | None)
        - ``reviewer_score`` (float | None): composite 0–1 score
        - ``review_tier`` (str | None): ``outstanding|good|inadequate|poor``
        - ``reviewer_comments`` (str | None): free-form reviewer critique
        - ``reviewer_scores`` (dict | None): parsed name-axis dimensional scores
        - ``validation_status`` (str | None): graph lifecycle state at
          fetch time

        Only sources that currently link to a StandardName with a
        non-null ``reviewer_score_name`` are returned — entries without prior
        review data are silently omitted (the caller can treat this as a
        cold-start and skip feedback injection).
    """
    if not source_ids:
        return {}

    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS source_id
            MATCH (src {id: source_id})-[:HAS_STANDARD_NAME]->(sn:StandardName)
            WHERE sn.reviewer_score_name IS NOT NULL
            RETURN source_id AS source_id,
                   sn.id AS previous_name,
                   sn.description AS previous_description,
                   sn.documentation AS previous_documentation,
                   sn.reviewer_score_name AS reviewer_score,
                   sn.review_tier AS review_tier,
                   sn.reviewer_comments_name AS reviewer_comments,
                   sn.reviewer_scores_name AS reviewer_scores_json,
                   sn.reviewer_suggested_name AS reviewer_suggested_name,
                   sn.reviewer_suggestion_justification_name
                       AS reviewer_suggestion_justification,
                   sn.validation_status AS validation_status
            """,
            ids=ids,
        )

    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("source_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        # If a source_id has multiple linked SNs, prefer the one with the
        # lowest reviewer_score (the one most in need of revision).
        new_entry = {
            "previous_name": row.get("previous_name"),
            "previous_description": row.get("previous_description"),
            "previous_documentation": row.get("previous_documentation"),
            "reviewer_score": row.get("reviewer_score"),
            "review_tier": row.get("review_tier"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "reviewer_suggested_name": row.get("reviewer_suggested_name") or None,
            "reviewer_suggestion_justification": (
                row.get("reviewer_suggestion_justification") or None
            ),
            "validation_status": row.get("validation_status"),
        }
        existing = mapping.get(sid)
        if existing is None:
            mapping[sid] = new_entry
            continue
        prev = existing.get("reviewer_score")
        cur = new_entry.get("reviewer_score")
        if prev is None or (cur is not None and cur < prev):
            mapping[sid] = new_entry
    return mapping


def fetch_reviewer_history_for_sources(
    source_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch full StandardNameReview-node history per source for compose prompt enrichment.

    For each source that currently links to a StandardName with ≥1
    StandardNameReview node, returns:

    - ``latest``: the most recent StandardNameReview (score, comment ≤800 chars, model).
    - ``prior_themes``: n-gram theme extraction from older StandardNameReview comments,
      rendered as ``[{theme, count, example}]``.

    This complements :func:`fetch_review_feedback_for_sources` which injects
    only the denormalised aggregates from the StandardName node itself.  The
    history variant drills into *all* StandardNameReview nodes for richer compose context.

    Args:
        source_ids: Iterable of source-node ids (e.g. ``dd:equilibrium/...``).
            ``None`` or empty input returns ``{}`` without hitting the graph.

    Returns:
        ``{source_id: history_dict}`` where history_dict has keys:

        - ``latest`` (dict): ``{score, comment, model}`` of newest review.
        - ``prior_themes`` (list[dict]): theme summaries from older reviews,
          each ``{theme, count, example}``.  Empty when only one review exists.
    """
    if not source_ids:
        return {}

    ids = sorted({sid for sid in source_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS source_id
            MATCH (src {id: source_id})-[:HAS_STANDARD_NAME]->(sn:StandardName)
                  -[:HAS_REVIEW]->(r:StandardNameReview)
            RETURN source_id,
                   sn.id AS sn_id,
                   r.score AS score,
                   r.comments AS full_comment,
                   r.model AS model,
                   r.reviewed_at AS ts
            ORDER BY source_id, ts DESC
            """,
            ids=ids,
        )

    # Group rows by source_id
    from collections import defaultdict

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = row.get("source_id")
        if sid:
            by_source[sid].append(row)

    mapping: dict[str, dict[str, Any]] = {}
    for sid, reviews in by_source.items():
        if not reviews:
            continue

        # Latest review
        latest = reviews[0]
        comment_text = latest.get("full_comment") or ""
        history: dict[str, Any] = {
            "latest": {
                "score": latest.get("score"),
                "comment": comment_text[:800],
                "model": latest.get("model"),
            },
            "prior_themes": [],
        }

        # Extract themes from older reviews (N-1)
        if len(reviews) > 1:
            prior_comments = [
                r.get("full_comment") or ""
                for r in reviews[1:]
                if r.get("full_comment")
            ]
            if prior_comments:
                from imas_codex.standard_names.review.themes import (
                    _extract_themes_from_texts,
                )

                themes = _extract_themes_from_texts(prior_comments, top_n=5)
                # Build theme entries with count and example
                for theme in themes:
                    # Find matching example comment (first containing theme words)
                    example = ""
                    theme_words = set(theme.lower().split())
                    for c in prior_comments:
                        c_lower = c.lower()
                        if any(w in c_lower for w in theme_words):
                            example = c[:160]
                            break
                    history["prior_themes"].append(
                        {"theme": theme, "count": 1, "example": example}
                    )

        mapping[sid] = history

    return mapping


def fetch_docs_review_feedback_for_sns(
    sn_ids: list[str] | set[str] | None,
) -> dict[str, dict[str, Any]]:
    """Fetch prior docs-axis reviewer feedback keyed by StandardName id.

    Used by the enrich contextualise worker when re-enriching SNs that
    already have a docs-axis review on file.  Lets the LLM target the
    specific weaknesses (description quality, documentation_quality,
    completeness, physics_accuracy) the reviewer flagged previously
    instead of running blind.

    This is the docs-axis analogue of
    :func:`fetch_review_feedback_for_sources` — it keys on SN id (since
    enrich operates on SN nodes directly, not on sources) and pulls the
    ``*_docs`` reviewer fields plus ``validation_issues``.

    Args:
        sn_ids: Iterable of StandardName ids. ``None`` or empty returns
            ``{}`` without hitting the graph.

    Returns:
        ``{sn_id: feedback_dict}`` where feedback_dict has keys:

        - ``reviewer_score`` (float | None): docs-axis 0–1 score.
        - ``reviewer_comments`` (str | None): free-form docs critique.
        - ``reviewer_scores`` (dict | None): parsed docs-axis dimensional
          scores (description_quality / documentation_quality /
          completeness / physics_accuracy).
        - ``reviewer_verdict`` (str | None): accept / revise / reject.
        - ``validation_issues`` (list[str] | None): tagged ISN validation
          issue strings, when present.

        Only SNs with ``reviewer_score_docs IS NOT NULL`` are returned —
        cold-start enrichments are silently omitted.
    """
    if not sn_ids:
        return {}

    ids = sorted({sid for sid in sn_ids if sid})
    if not ids:
        return {}

    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sn_id
            MATCH (sn:StandardName {id: sn_id})
            WHERE sn.reviewer_score_docs IS NOT NULL
            RETURN sn_id AS sn_id,
                   sn.reviewer_score_docs AS reviewer_score,
                   sn.reviewer_comments_docs AS reviewer_comments,
                   sn.reviewer_scores_docs AS reviewer_scores_json,
                   sn.reviewer_verdict_docs AS reviewer_verdict,
                   sn.validation_issues AS validation_issues
            """,
            ids=ids,
        )

    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        sid = row.get("sn_id")
        if not sid:
            continue
        scores_json = row.get("reviewer_scores_json")
        scores_dict: dict[str, Any] | None = None
        if scores_json:
            try:
                scores_dict = json.loads(scores_json)
            except (TypeError, ValueError):
                scores_dict = None
        issues = row.get("validation_issues")
        if isinstance(issues, str):
            issues = [issues]
        elif issues is None:
            issues = None
        mapping[sid] = {
            "reviewer_score": row.get("reviewer_score"),
            "reviewer_comments": row.get("reviewer_comments"),
            "reviewer_scores": scores_dict,
            "reviewer_verdict": row.get("reviewer_verdict"),
            "validation_issues": issues if issues else None,
        }
    return mapping


def _write_standard_name_edges(gc: Any, names: list[dict[str, Any]]) -> None:
    """Emit all structural edges for a batch of StandardName nodes.

    Called as a tail pass **after** all nodes in the batch have been
    MERGEd.  Forward-reference targets are MERGEd as bare placeholder
    ``StandardName`` nodes so the edge can be created immediately; their
    full properties arrive in the same batch, a later batch, or via
    catalog import.

    Handles the following edge types:

    - ``HAS_ARGUMENT``: derived from the ISN grammar parser (one layer
      per call: unary prefix/postfix, binary, or projection).
    - ``HAS_ERROR``: uncertainty siblings — direction inverted relative to
      the derivation (inner → uncertainty form).
    - ``HAS_PREDECESSOR``: from ``predecessor`` or ``deprecates`` field.
    - ``HAS_SUCCESSOR``: from ``successor`` or ``superseded_by`` field.
    - ``IN_CLUSTER``: from ``primary_cluster_id`` field.
    - ``HAS_PHYSICS_DOMAIN``: from ``physics_domain`` field.

    Parameters
    ----------
    gc:
        Active ``GraphClient`` context (already open — do not open a new
        one inside this function).
    names:
        List of name dicts, each containing at minimum ``id``.  All other
        fields are optional; missing fields produce no edges.
    """
    from imas_codex.standard_names.derivation import derive_edges

    ha_batch: list[dict[str, Any]] = []  # HAS_ARGUMENT
    he_batch: list[dict[str, Any]] = []  # HAS_ERROR

    for n in names:
        name_id = n.get("id")
        if not name_id:
            continue
        for edge in derive_edges(name_id):
            if edge.edge_type == "HAS_ARGUMENT":
                ha_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "operator": edge.props.get("operator"),
                        "operator_kind": edge.props.get("operator_kind"),
                        "role": edge.props.get("role"),
                        "separator": edge.props.get("separator"),
                        "axis": edge.props.get("axis"),
                        "shape": edge.props.get("shape"),
                    }
                )
            elif edge.edge_type == "HAS_ERROR":
                he_batch.append(
                    {
                        "from_name": edge.from_name,
                        "to_name": edge.to_name,
                        "error_type": edge.props.get("error_type"),
                    }
                )

    if ha_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[r:HAS_ARGUMENT]->(tgt)
            SET r.operator      = b.operator,
                r.operator_kind = b.operator_kind,
                r.role          = b.role,
                r.separator     = b.separator,
                r.axis          = b.axis,
                r.shape         = b.shape
            """,
            batch=ha_batch,
        )

    if he_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[r:HAS_ERROR]->(tgt)
            SET r.error_type = b.error_type
            """,
            batch=he_batch,
        )

    # --- HAS_PREDECESSOR / HAS_SUCCESSOR ---
    # Support both 'predecessor'/'successor' (pipeline) and
    # 'deprecates'/'superseded_by' (catalog import).
    pred_batch: list[dict[str, str]] = []
    succ_batch: list[dict[str, str]] = []
    for n in names:
        name_id = n.get("id")
        if not name_id:
            continue
        predecessor = n.get("predecessor") or n.get("deprecates")
        if predecessor:
            pred_batch.append({"from_name": name_id, "to_name": predecessor})
        successor = n.get("successor") or n.get("superseded_by")
        if successor:
            succ_batch.append({"from_name": name_id, "to_name": successor})

    if pred_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[:HAS_PREDECESSOR]->(tgt)
            """,
            batch=pred_batch,
        )

    if succ_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (src:StandardName {id: b.from_name})
            MERGE (tgt:StandardName {id: b.to_name})
            MERGE (src)-[:HAS_SUCCESSOR]->(tgt)
            """,
            batch=succ_batch,
        )

    # --- IN_CLUSTER ---
    cluster_batch = [
        {"sn_id": n["id"], "cluster_id": n["primary_cluster_id"]}
        for n in names
        if n.get("id") and n.get("primary_cluster_id")
    ]
    if cluster_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.sn_id})
            MERGE (c:IMASSemanticCluster {id: b.cluster_id})
            MERGE (sn)-[:IN_CLUSTER]->(c)
            """,
            batch=cluster_batch,
        )

    # --- HAS_PHYSICS_DOMAIN (multi-valued) ---
    domain_batch = []
    for n in names:
        if not n.get("id"):
            continue
        domains = _ensure_list(n.get("physics_domain"))
        for d in domains:
            domain_batch.append({"sn_id": n["id"], "domain_id": d})
    if domain_batch:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.sn_id})
            MERGE (d:PhysicsDomain {id: b.domain_id})
            MERGE (sn)-[:HAS_PHYSICS_DOMAIN]->(d)
            """,
            batch=domain_batch,
        )


def write_standard_names(
    names: list[dict[str, Any]],
    *,
    override: bool = False,
) -> int:
    """MERGE StandardName nodes with HAS_STANDARD_NAME relationships.

    Relationship direction: entity → concept
      (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
      (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)

    Each dict in *names* must have at least:
      - ``id``: the composed standard name string
      - ``source_types``: ["dd"] or ["signals"] etc.
      - ``source_id``: the originating path / signal ID

    Optional fields: ``unit``, ``description``,
    ``documentation``, ``kind``, ``links``, ``source_paths``,
    ``validity_domain``, ``constraints``, ``model``, ``pipeline_status``,
    ``generated_at``,
    ``review_tier``,
    ``vocab_gap_detail``, ``validation_issues``,
    ``validation_layer_summary``, ``cocos_transformation_type``, ``dd_version``,
    ``review_input_hash``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection — write protected fields
        even on catalog-edited names.

    Performs conflict detection on ``unit``: if a StandardName already exists
    with a different unit value, that entry is skipped (not written)
    and a warning is logged.

    Returns the number of nodes written.
    """
    if not names:
        return 0

    # Pipeline protection — strip catalog-owned fields from catalog_edit items
    from imas_codex.standard_names.protection import filter_protected

    names, skipped = filter_protected(names, override=override)
    if skipped:
        logger.warning(
            "write_standard_names: stripped protected fields from %d catalog-edited name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not names:
        return 0

    # Guard: warn when cocos_transformation_type is set but cocos integer is missing
    for n in names:
        if n.get("cocos_transformation_type") and n.get("cocos") is None:
            logger.warning(
                "StandardName '%s' has cocos_transformation_type='%s' but no cocos "
                "integer — HAS_COCOS edge will not be created",
                n["id"],
                n["cocos_transformation_type"],
            )

    with GraphClient() as gc:
        # Conflict-detect on unit — same name with different unit is a data
        # integrity error.  Filter out conflicting entries rather than raising
        # so that non-conflicting entries can still proceed.
        unit_check_batch = [
            {"id": n["id"], "unit": n.get("unit")} for n in names if n.get("unit")
        ]
        if unit_check_batch:
            unit_conflicts = list(
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.id})
                    WHERE sn.unit IS NOT NULL AND b.unit IS NOT NULL
                      AND sn.unit <> b.unit
                    RETURN sn.id AS name,
                           sn.unit AS existing_unit,
                           b.unit AS incoming_unit
                    """,
                    batch=unit_check_batch,
                )
                or []
            )
            if unit_conflicts:
                conflict_details = "; ".join(
                    f"{c['name']}: {c['existing_unit']} vs {c['incoming_unit']}"
                    for c in unit_conflicts
                )
                logger.warning("Unit conflicts detected: %s", conflict_details)
                conflicting_ids = {c["name"] for c in unit_conflicts}
                names = [n for n in names if n["id"] not in conflicting_ids]
                if not names:
                    logger.warning("All entries had unit conflicts — nothing to write")
                    return 0

        # MERGE StandardName nodes with provenance — coalesce to preserve existing data
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            SET sn.source_types = coalesce(b.source_types, sn.source_types),
                sn.description = coalesce(b.description, sn.description),
                sn.documentation = coalesce(b.documentation, sn.documentation),
                sn.kind = coalesce(b.kind, sn.kind),
                sn.links = coalesce(b.links, sn.links),
                sn.source_paths = coalesce(b.source_paths, sn.source_paths),
                sn.validity_domain = coalesce(b.validity_domain, sn.validity_domain),
                sn.constraints = coalesce(b.constraints, sn.constraints),
                sn.unit = coalesce(b.unit, sn.unit),
                sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
                sn.cocos = coalesce(b.cocos, sn.cocos),
                sn.dd_version = coalesce(b.dd_version, sn.dd_version),
                sn.model = coalesce(b.model, sn.model),
                sn.pipeline_status = coalesce(b.pipeline_status, sn.pipeline_status),
                sn.generated_at = coalesce(b.generated_at, sn.generated_at),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.vocab_gap_detail = coalesce(b.vocab_gap_detail, sn.vocab_gap_detail),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
                sn.validation_layer_summary = coalesce(b.validation_layer_summary, sn.validation_layer_summary),
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = coalesce(b.embedded_at, sn.embedded_at),
                sn.grammar_parse_version = coalesce(b.grammar_parse_version, sn.grammar_parse_version),
                sn.validation_diagnostics_json = coalesce(b.validation_diagnostics_json, sn.validation_diagnostics_json),
                sn.llm_cost_regen = CASE WHEN sn.compose_count IS NOT NULL
                                             AND sn.compose_count > 0
                                             AND b.llm_cost IS NOT NULL
                                        THEN coalesce(sn.llm_cost_regen, 0.0) + b.llm_cost
                                        ELSE sn.llm_cost_regen END,
                sn.llm_cost_compose = CASE WHEN b.llm_cost IS NOT NULL
                                      THEN coalesce(sn.llm_cost_compose, 0.0) + b.llm_cost
                                      ELSE sn.llm_cost_compose END,
                sn.compose_count = CASE WHEN b.llm_cost IS NOT NULL
                                   THEN coalesce(sn.compose_count, 0) + 1
                                   ELSE sn.compose_count END,
                sn.regen_count = CASE WHEN b.regen_increment = true
                                 THEN coalesce(sn.regen_count, 0) + 1
                                 ELSE sn.regen_count END,
                sn.llm_cost = CASE WHEN b.llm_cost IS NOT NULL
                              THEN coalesce(sn.llm_cost, 0.0) + b.llm_cost
                              ELSE sn.llm_cost END,
                sn.llm_model = coalesce(b.llm_model, sn.llm_model),
                sn.llm_service = coalesce(b.llm_service, sn.llm_service),
                sn.llm_at = coalesce(b.llm_at, sn.llm_at),
                sn.llm_tokens_in = coalesce(b.llm_tokens_in, sn.llm_tokens_in),
                sn.llm_tokens_out = coalesce(b.llm_tokens_out, sn.llm_tokens_out),
                sn.llm_tokens_cached_read = coalesce(b.llm_tokens_cached_read, sn.llm_tokens_cached_read),
                sn.llm_tokens_cached_write = coalesce(b.llm_tokens_cached_write, sn.llm_tokens_cached_write),
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_types": n.get("source_types") or None,
                    "description": n.get("description"),
                    "documentation": n.get("documentation"),
                    "kind": n.get("kind"),
                    "links": n.get("links") or None,
                    "source_paths": n.get("source_paths") or None,
                    "validity_domain": n.get("validity_domain"),
                    "constraints": n.get("constraints") or None,
                    "unit": n.get("unit"),
                    "physics_domain": _ensure_list(n.get("physics_domain")),
                    "cocos_transformation_type": n.get("cocos_transformation_type"),
                    "cocos": n.get("cocos"),
                    "dd_version": n.get("dd_version"),
                    "model": n.get("model"),
                    "pipeline_status": n.get("pipeline_status")
                    or n.get("review_status"),
                    "generated_at": n.get("generated_at"),
                    "review_tier": n.get("review_tier"),
                    "vocab_gap_detail": _ensure_json(n.get("vocab_gap_detail")),
                    "validation_issues": n.get("validation_issues") or None,
                    "validation_layer_summary": _ensure_json(
                        n.get("validation_layer_summary")
                    ),
                    "validation_status": n.get("validation_status"),
                    "link_status": _compute_link_status(n.get("links")),
                    "review_input_hash": n.get("review_input_hash"),
                    "embedding": n.get("embedding"),
                    "embedded_at": n.get("embedded_at"),
                    "llm_cost": n.get("llm_cost"),
                    "llm_model": n.get("llm_model"),
                    "llm_service": n.get("llm_service"),
                    "llm_at": n.get("llm_at"),
                    "llm_tokens_in": n.get("llm_tokens_in"),
                    "llm_tokens_out": n.get("llm_tokens_out"),
                    "llm_tokens_cached_read": n.get("llm_tokens_cached_read"),
                    "llm_tokens_cached_write": n.get("llm_tokens_cached_write"),
                    "regen_increment": n.get("regen_increment"),
                    **_parse_grammar_vnext(n["id"]),
                }
                for n in names
            ],
        )

        # Append-with-dedupe physics_domain (list property, never overwrite)
        pd_batch = [
            {"id": n["id"], "physics_domain": _ensure_list(n.get("physics_domain"))}
            for n in names
            if _ensure_list(n.get("physics_domain"))
        ]
        if pd_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MERGE (sn:StandardName {id: b.id})
                WITH sn, b,
                     coalesce(sn.physics_domain, []) AS existing,
                     coalesce(b.physics_domain, []) AS incoming
                WITH sn, existing,
                     [d IN incoming WHERE d IS NOT NULL
                      AND NOT (d IN existing) | d] AS new_domains
                SET sn.physics_domain = existing + new_domains
                """,
                batch=pd_batch,
            )

        # Create HAS_STANDARD_NAME relationships: entity → concept
        dd_names = [n for n in names if "dd" in (n.get("source_types") or [])]
        signal_names = [n for n in names if "signals" in (n.get("source_types") or [])]

        if dd_names:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in dd_names
                    if n.get("source_id")
                ],
            )
        if signal_names:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (src:FacilitySignal {id: b.source_id})
                MERGE (src)-[:HAS_STANDARD_NAME]->(sn)
                """,
                batch=[
                    {"id": n["id"], "source_id": n["source_id"]}
                    for n in signal_names
                    if n.get("source_id")
                ],
            )

        # Create HAS_UNIT relationships: StandardName → Unit
        units_batch = [
            {"id": n["id"], "unit": n["unit"]} for n in names if n.get("unit")
        ]
        if units_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MERGE (u:Unit {id: b.unit})
                SET u.symbol = coalesce(u.symbol, b.unit)
                MERGE (sn)-[:HAS_UNIT]->(u)
                """,
                batch=units_batch,
            )

        # Create HAS_COCOS relationships: StandardName → COCOS
        # Use MATCH (not MERGE) — COCOS singleton nodes already exist.
        cocos_batch = [
            {"id": n["id"], "cocos": n["cocos"]}
            for n in names
            if n.get("cocos") is not None
        ]
        if cocos_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (sn:StandardName {id: b.id})
                MATCH (c:COCOS {id: b.cocos})
                MERGE (sn)-[:HAS_COCOS]->(c)
                """,
                batch=cocos_batch,
            )

        # Create HAS_SEGMENT relationships: StandardName → GrammarToken
        token_miss_gaps = _write_segment_edges(gc, [n["id"] for n in names])

        # Emit structural edges: HAS_ARGUMENT, HAS_ERROR, HAS_PREDECESSOR,
        # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN.
        # Tail pass — all nodes in this batch exist before edges are written.
        _write_standard_name_edges(gc, names)

    # Persist token-miss gaps as VocabGap nodes (outside gc context —
    # write_vocab_gaps opens its own GraphClient)
    if token_miss_gaps:
        # Build sn_id → source mapping from the names list
        sn_source_map: dict[str, tuple[str, str]] = {}
        for n in names:
            sn_id = n["id"]
            source_id = n.get("source_id")
            source_type = "dd" if "dd" in (n.get("source_types") or []) else "signals"
            if source_id and sn_id not in sn_source_map:
                sn_source_map[sn_id] = (source_id, source_type)

        # Group gaps by source_type for write_vocab_gaps
        dd_gap_dicts: list[dict[str, str]] = []
        signal_gap_dicts: list[dict[str, str]] = []
        for gap in token_miss_gaps:
            mapping = sn_source_map.get(gap["sn_id"])
            if not mapping:
                continue
            source_id, source_type = mapping
            gap_dict = {
                "source_id": source_id,
                "segment": gap["segment"],
                "needed_token": gap["needed_token"],
                "reason": f"Token-miss during grammar edge writing for '{gap['sn_id']}'",
            }
            if source_type == "dd":
                dd_gap_dicts.append(gap_dict)
            else:
                signal_gap_dicts.append(gap_dict)

        if dd_gap_dicts:
            write_vocab_gaps(dd_gap_dicts, source_type="dd")
        if signal_gap_dicts:
            write_vocab_gaps(signal_gap_dicts, source_type="signals")

    # Sweep skeleton placeholders created by relationship-side MERGE on
    # uncomposed targets (HAS_ARGUMENT, HAS_ERROR, HAS_PREDECESSOR,
    # HAS_SUCCESSOR, IN_CLUSTER, HAS_PHYSICS_DOMAIN). A real StandardName
    # always has at least a created_at OR generated_at timestamp; pure
    # skeletons (id-only) are detached and deleted.
    # Opens its own GraphClient — the surrounding `with` block has already
    # closed; write_vocab_gaps above follows the same pattern.
    with GraphClient() as gc:
        swept = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.created_at IS NULL
              AND sn.generated_at IS NULL
              AND sn.validation_status IS NULL
              AND sn.unit IS NULL
              AND sn.kind IS NULL
            DETACH DELETE sn
            RETURN count(sn) AS swept
            """
        )
    swept_count = (swept[0]["swept"] if swept else 0) if swept else 0
    if swept_count:
        logger.info("Swept %d skeleton StandardName placeholder(s)", swept_count)

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    return written


def write_reviews(records: list[dict[str, Any]], *, skip_cost: bool = False) -> int:
    """MERGE ``StandardNameReview`` nodes and ``HAS_REVIEW`` edges from StandardName.

    Each record must contain:

    - ``id`` (str) — composite key
      ``{standard_name_id}:{axis}:{review_group_id}:{cycle_index}``
    - ``standard_name_id`` (str) — parent StandardName
    - ``model`` (str), ``model_family`` (str), ``is_canonical`` (bool)
    - ``score`` (float 0-1), ``scores_json`` (str), ``tier`` (str)
    - ``reviewed_at`` (str ISO 8601)

    RD-quorum fields (required for new-style reviews):
    - ``review_axis`` (str) — "names" or "docs"
    - ``cycle_index`` (int) — 0, 1, or 2
    - ``review_group_id`` (str) — UUID
    - ``resolution_role`` (str) — "primary", "secondary", or "escalator"
    - ``resolution_method`` (str | None)

    Optional: ``comments`` (str), ``llm_cost`` (float),
    ``llm_tokens_in`` (int), ``llm_tokens_out`` (int),
    ``llm_model`` (str), ``llm_at`` (str), ``llm_service`` (str).

    MERGE-by-``id`` semantics make re-runs idempotent when the same
    model reviews the same name at the same timestamp.

    Parameters
    ----------
    records:
        Review record dicts.
    skip_cost:
        When ``True``, skip accumulating ``llm_cost_review`` on
        StandardName nodes.  Use when StandardNameReview records were
        already persisted inline (crash-safety path) to avoid
        double-counting.

    Returns the number of StandardNameReview records written.
    """
    if not records:
        return 0
    # Guard: must attach to an existing StandardName.
    valid = [r for r in records if r.get("id") and r.get("standard_name_id")]
    if not valid:
        return 0
    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (r:StandardNameReview {id: b.id})
            SET r.standard_name_id = b.standard_name_id,
                r.model = b.model,
                r.reviewer_model = b.reviewer_model,
                r.model_family = b.model_family,
                r.is_canonical = b.is_canonical,
                r.score = b.score,
                r.scores_json = b.scores_json,
                r.tier = b.tier,
                r.verdict = b.verdict,
                r.comments = b.comments,
                r.comments_per_dim_json = b.comments_per_dim_json,
                r.suggested_name = b.suggested_name,
                r.suggestion_justification = b.suggestion_justification,
                r.reviewed_at = b.reviewed_at,
                r.review_axis = b.review_axis,
                r.cycle_index = b.cycle_index,
                r.review_group_id = b.review_group_id,
                r.resolution_role = b.resolution_role,
                r.resolution_method = b.resolution_method,
                r.llm_model = b.llm_model,
                r.llm_cost = b.llm_cost,
                r.llm_tokens_in = b.llm_tokens_in,
                r.llm_tokens_out = b.llm_tokens_out,
                r.llm_tokens_cached_read = b.llm_tokens_cached_read,
                r.llm_tokens_cached_write = b.llm_tokens_cached_write,
                r.llm_at = b.llm_at,
                r.llm_service = b.llm_service
            WITH r, b
            MATCH (sn:StandardName {id: b.standard_name_id})
            MERGE (sn)-[:HAS_REVIEW]->(r)
            """,
            batch=[
                {
                    "id": r["id"],
                    "standard_name_id": r["standard_name_id"],
                    "model": r.get("model") or "",
                    # reviewer_model is the consumer-facing alias; fall back to model
                    "reviewer_model": r.get("reviewer_model") or r.get("model") or "",
                    "model_family": r.get("model_family") or "other",
                    "is_canonical": bool(r.get("is_canonical", False)),
                    "score": float(r.get("score") or 0.0),
                    "scores_json": _ensure_json(r.get("scores_json") or "{}"),
                    "tier": r.get("tier") or "unknown",
                    # verdict is the accept/reject/revise decision from the LLM
                    "verdict": r.get("verdict") or "",
                    "comments": r.get("comments") or "",
                    "comments_per_dim_json": _ensure_json(
                        r.get("comments_per_dim_json")
                    ),
                    "suggested_name": r.get("suggested_name") or "",
                    "suggestion_justification": r.get("suggestion_justification") or "",
                    "reviewed_at": r.get("reviewed_at"),
                    "review_axis": r.get("review_axis"),
                    "cycle_index": r.get("cycle_index"),
                    "review_group_id": r.get("review_group_id"),
                    "resolution_role": r.get("resolution_role"),
                    "resolution_method": r.get("resolution_method"),
                    "llm_model": r.get("llm_model"),
                    "llm_cost": r.get("llm_cost"),
                    "llm_tokens_in": r.get("llm_tokens_in"),
                    "llm_tokens_out": r.get("llm_tokens_out"),
                    "llm_tokens_cached_read": r.get("llm_tokens_cached_read"),
                    "llm_tokens_cached_write": r.get("llm_tokens_cached_write"),
                    "llm_at": r.get("llm_at"),
                    "llm_service": r.get("llm_service"),
                }
                for r in valid
            ],
        )

        # --- Accumulate review cost on StandardName ---
        # Build per-SN cost totals from the batch (sum review costs per name).
        # Skipped when skip_cost=True (records already persisted inline).
        if not skip_cost:
            sn_cost_map: dict[str, float] = {}
            for r in valid:
                sn_id = r.get("standard_name_id")
                cost = r.get("llm_cost")
                if sn_id and cost:
                    sn_cost_map[sn_id] = sn_cost_map.get(sn_id, 0.0) + cost
            if sn_cost_map:
                gc.query(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.llm_cost_review = coalesce(sn.llm_cost_review, 0.0) + b.cost,
                        sn.llm_cost = coalesce(sn.llm_cost, 0.0) + b.cost
                    """,
                    batch=[
                        {"sn_id": sn_id, "cost": cost}
                        for sn_id, cost in sn_cost_map.items()
                    ],
                )

    logger.info("Wrote %d StandardNameReview nodes", len(valid))
    return len(valid)


def update_review_aggregates(
    standard_name_ids: list[str],
    *,
    threshold: float = 0.2,
) -> int:
    """Recompute per-StandardName aggregates from attached StandardNameReview nodes.

    **Winning-group selection**: identifies the most recent review group
    whose ``resolution_method`` is one of ``quorum_consensus``,
    ``authoritative_escalation``, or ``single_review`` (excluding
    ``retry_item`` and ``max_cycles_reached``). Mirrors that group's
    final scores onto the SN aggregates.

    Also sets ``review_count``, ``review_mean_score``, and
    ``review_disagreement`` across all attached StandardNameReview nodes.

    Returns the number of StandardName nodes updated.
    """
    if not standard_name_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
            WITH sn, count(r) AS n, avg(r.score) AS mean,
                 CASE WHEN count(r) > 1 THEN max(r.score) - min(r.score) ELSE 0.0 END AS spread
            SET sn.review_count = n,
                sn.review_mean_score = CASE WHEN n > 0 THEN mean ELSE null END,
                sn.review_disagreement = (n > 1 AND spread >= $threshold)
            RETURN sn.id AS id
            """,
            ids=standard_name_ids,
            threshold=float(threshold),
        )
        return len(list(rows or []))


def write_name_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write name-axis review scores to StandardName nodes.

    Writes ``reviewer_score_name``, ``reviewed_name_at``, and all
    name-axis rubric fields. Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_name`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_name`` on the
        graph.
    stats:
        Optional mutable dict to accumulate write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.reviewer_score_name = b.reviewer_score_name,
                sn.reviewed_name_at = b.reviewed_name_at,
                sn.reviewer_scores_name = coalesce(b.reviewer_scores_name, sn.reviewer_scores_name),
                sn.reviewer_comments_name = coalesce(b.reviewer_comments_name, sn.reviewer_comments_name),
                sn.reviewer_comments_per_dim_name = coalesce(b.reviewer_comments_per_dim_name, sn.reviewer_comments_per_dim_name),
                sn.reviewer_verdict_name = coalesce(b.reviewer_verdict_name, sn.reviewer_verdict_name),
                sn.reviewer_model_name = coalesce(b.reviewer_model_name, sn.reviewer_model_name),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.review_input_hash = b.review_input_hash,
                sn.reviewer_suggested_name = coalesce(nullIf(b.reviewer_suggested_name, ''), sn.reviewer_suggested_name),
                sn.reviewer_suggestion_justification_name = coalesce(nullIf(b.reviewer_suggestion_justification_name, ''), sn.reviewer_suggestion_justification_name),
                sn.llm_cost_review = coalesce(sn.llm_cost_review, 0.0) + coalesce(b.llm_cost_review, 0.0),
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review, 0.0)
            """,
            batch=[
                {
                    "id": e.get("_original_id") or e["id"],
                    "reviewer_score_name": e.get("reviewer_score"),
                    "reviewed_name_at": e.get("reviewed_at"),
                    "reviewer_scores_name": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_name": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_name": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_verdict_name": e.get("reviewer_verdict"),
                    "reviewer_model_name": e.get("reviewer_model"),
                    "review_tier": e.get("review_tier"),
                    "review_input_hash": e.get("review_input_hash"),
                    "reviewer_suggested_name": e.get("_suggested_name") or "",
                    "reviewer_suggestion_justification_name": e.get(
                        "_suggestion_justification"
                    )
                    or "",
                    "llm_cost_review": e.get("llm_cost") or 0.0,
                }
                for e in entries
            ],
        )

    written = len(entries)
    logger.info("write_name_review_results: wrote %d names", written)
    return written


def write_docs_review_results(
    entries: list[dict[str, Any]],
    *,
    stats: dict[str, Any] | None = None,
) -> int:
    """Write docs-axis review scores to StandardName nodes.

    Gate: entries whose ``reviewed_name_at IS NULL`` in the graph are
    skipped (logged as ERROR, counted in
    ``stats['docs_skipped_missing_name']``).

    Does **not** touch any shared aggregate slots
    (those have been removed from the schema).

    The in-memory ``reviewer_score`` dict key is mapped to the graph
    property ``reviewer_score_docs`` here — there is no generic
    ``reviewer_score`` graph property.

    Parameters
    ----------
    entries:
        Dicts with at least ``id`` and ``reviewer_score`` keys.
        The ``reviewer_score`` key is the in-memory generic name used
        by the pipeline; it maps to ``sn.reviewer_score_docs`` on the
        graph.
    stats:
        Optional mutable dict to accumulate skip/write counters.

    Returns
    -------
    int
        Number of StandardName nodes written.
    """
    if not entries:
        return 0
    if stats is None:
        stats = {}

    # Gate check: query graph for reviewed_name_at on target names
    entry_ids = [e["id"] for e in entries if e.get("id")]
    if not entry_ids:
        return 0
    with GraphClient() as gc:
        rows = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (sn:StandardName {id: sid})
            RETURN sn.id AS id, sn.reviewed_name_at AS reviewed_name_at
            """,
            ids=entry_ids,
        )
        gated: dict[str, bool] = {}
        for r in rows or []:
            gated[r["id"]] = r.get("reviewed_name_at") is not None

    passed: list[dict[str, Any]] = []
    skipped = 0
    for e in entries:
        eid = e.get("id", "")
        if not gated.get(eid, False):
            logger.error(
                "write_docs_review_results: skipping %r — reviewed_name_at IS NULL",
                eid,
            )
            skipped += 1
            continue
        passed.append(e)
    stats["docs_skipped_missing_name"] = (
        stats.get("docs_skipped_missing_name", 0) + skipped
    )
    if not passed:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.reviewer_score_docs = b.reviewer_score_docs,
                sn.reviewed_docs_at = b.reviewed_docs_at,
                sn.reviewer_scores_docs = coalesce(b.reviewer_scores_docs, sn.reviewer_scores_docs),
                sn.reviewer_comments_docs = coalesce(b.reviewer_comments_docs, sn.reviewer_comments_docs),
                sn.reviewer_comments_per_dim_docs = coalesce(b.reviewer_comments_per_dim_docs, sn.reviewer_comments_per_dim_docs),
                sn.reviewer_verdict_docs = coalesce(b.reviewer_verdict_docs, sn.reviewer_verdict_docs),
                sn.reviewer_model_docs = coalesce(b.reviewer_model_docs, sn.reviewer_model_docs),
                sn.review_input_hash = b.review_input_hash,
                sn.llm_cost_review = coalesce(sn.llm_cost_review, 0.0) + coalesce(b.llm_cost_review, 0.0),
                sn.llm_cost = coalesce(sn.llm_cost, 0.0) + coalesce(b.llm_cost_review, 0.0)
            """,
            batch=[
                {
                    "id": e["id"],
                    "reviewer_score_docs": e.get("reviewer_score"),
                    "reviewed_docs_at": e.get("reviewed_at"),
                    "reviewer_scores_docs": _ensure_json(e.get("reviewer_scores")),
                    "reviewer_comments_docs": e.get("reviewer_comments"),
                    "reviewer_comments_per_dim_docs": _ensure_json(
                        e.get("reviewer_comments_per_dim")
                    ),
                    "reviewer_verdict_docs": e.get("reviewer_verdict"),
                    "reviewer_model_docs": e.get("reviewer_model"),
                    "review_input_hash": e.get("review_input_hash"),
                    "llm_cost_review": e.get("llm_cost") or 0.0,
                }
                for e in passed
            ],
        )

    written = len(passed)
    logger.info("write_docs_review_results: wrote %d names", written)
    return written


def _resolve_grammar_token_version(gc: GraphClient, isn_version: str) -> str | None:
    """Find the best GrammarToken version for HAS_SEGMENT resolution.

    Prefers exact match with the ISN runtime version.  When no tokens
    exist for that version (e.g. ISN was upgraded but
    ``imas-codex sn sync-grammar`` was not re-run), falls back
    to the latest available version so that token-miss detection does
    not produce false-positive VocabGap nodes.

    Returns ``None`` when no GrammarToken nodes exist at all.
    """
    # Fast path: check exact version
    rows = list(
        gc.query(
            "MATCH (t:GrammarToken {version: $v}) RETURN t.version LIMIT 1",
            v=isn_version,
        )
        or []
    )
    if rows:
        return isn_version

    # Fallback: latest available version
    rows = list(
        gc.query(
            "MATCH (t:GrammarToken) "
            "RETURN DISTINCT t.version AS v ORDER BY v DESC LIMIT 1"
        )
        or []
    )
    if rows:
        fallback = rows[0]["v"]
        logger.warning(
            "No GrammarToken nodes for ISN %s — falling back to %s. "
            "Run `imas-codex sn sync-grammar` to update.",
            isn_version,
            fallback,
        )
        return fallback

    return None


def _write_segment_edges(gc: GraphClient, name_ids: list[str]) -> list[dict[str, str]]:
    """Write HAS_SEGMENT edges from StandardName nodes to GrammarToken nodes.

    For each name, parses via ISN grammar, resolves segment tokens, and
    MERGEs ``(sn:StandardName)-[:HAS_SEGMENT {position, segment}]->(t:GrammarToken)``.

    Idempotent: existing HAS_SEGMENT edges are deleted before re-writing.
    Parse failures are logged and skipped — the SN node remains intact.
    Token-miss (vocabulary drift) is detected, warned, and returned.

    The GrammarToken version used for resolution is the installed ISN
    version when available, otherwise the latest synced version (see
    :func:`_resolve_grammar_token_version`).

    Args:
        gc: Open GraphClient session (caller manages the context manager).
        name_ids: List of StandardName.id values to process.

    Returns:
        List of detected token-miss gaps, each a dict with keys:
        ``sn_id``, ``segment``, ``needed_token``.
    """
    all_gaps: list[dict[str, str]] = []

    try:
        from imas_standard_names import __version__ as isn_version
        from imas_standard_names.grammar import parse_standard_name
        from imas_standard_names.graph.spec import segment_edge_specs
    except ImportError:
        logger.debug("ISN grammar not available — skipping HAS_SEGMENT edges")
        return all_gaps

    # Resolve the best available GrammarToken version
    token_version = _resolve_grammar_token_version(gc, isn_version)
    if token_version is None:
        logger.debug("No GrammarToken nodes in graph — skipping HAS_SEGMENT edges")
        return all_gaps

    for sn_id in name_ids:
        try:
            parsed = parse_standard_name(sn_id)
            edge_specs = segment_edge_specs(parsed)
        except Exception:
            logger.warning(
                "Grammar parse failed for '%s' — skipping HAS_SEGMENT edges",
                sn_id,
                exc_info=True,
            )
            continue

        if not edge_specs:
            continue

        # Idempotent: delete old HAS_SEGMENT and all 10 typed segment edges
        # before re-writing so repeated calls leave exactly one edge per type.
        gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})-[r:HAS_SEGMENT|HAS_PHYSICAL_BASE|HAS_SUBJECT|HAS_TRANSFORMATION|HAS_COMPONENT|HAS_COORDINATE|HAS_PROCESS|HAS_POSITION|HAS_REGION|HAS_DEVICE|HAS_GEOMETRIC_BASE]->(:GrammarToken)
            DELETE r
            """,
            sn_id=sn_id,
        )

        # Write new HAS_SEGMENT edges via OPTIONAL MATCH to detect token-miss
        edges_param = [
            {
                "position": s.position,
                "segment": s.segment,
                "token": s.token,
            }
            for s in edge_specs
        ]

        results = list(
            gc.query(
                """
                MATCH (sn:StandardName {id: $sn_id})
                UNWIND $edges AS edge
                OPTIONAL MATCH (t:GrammarToken {
                    value: edge.token,
                    segment: edge.segment,
                    version: $token_version
                })
                WITH sn, edge, t
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (sn)-[r:HAS_SEGMENT]->(t)
                    SET r.position = edge.position,
                        r.segment = edge.segment
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'physical_base' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_PHYSICAL_BASE]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'subject' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_SUBJECT]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'transformation' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_TRANSFORMATION]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'component' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_COMPONENT]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'coordinate' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_COORDINATE]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'process' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_PROCESS]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'position' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_POSITION]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'region' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_REGION]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'device' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_DEVICE]->(t)
                )
                FOREACH (_ IN CASE WHEN t IS NOT NULL AND edge.segment = 'geometric_base' THEN [1] ELSE [] END |
                    MERGE (sn)-[:HAS_GEOMETRIC_BASE]->(t)
                )
                RETURN edge.token AS token,
                       edge.segment AS segment,
                       t IS NOT NULL AS matched
                """,
                sn_id=sn_id,
                edges=edges_param,
                token_version=token_version,
            )
            or []
        )

        # Detect token-miss (vocabulary gaps)
        missing = [
            f"{r['segment']}:{r['token']}"
            for r in results
            if not r.get("matched", True)
        ]
        if missing:
            logger.warning(
                "Token-miss for '%s': %s (ISN %s, tokens %s) — vocab gap",
                sn_id,
                ", ".join(missing),
                isn_version,
                token_version,
            )
            for r in results:
                if not r.get("matched", True):
                    all_gaps.append(
                        {
                            "sn_id": sn_id,
                            "segment": r["segment"],
                            "needed_token": r["token"],
                        }
                    )

    return all_gaps


# =============================================================================
# Immediate-persist helpers — graph-state-machine generate_name
# =============================================================================

_GRAMMAR_FIELDS = (
    "physical_base",
    "subject",
    "component",
    "coordinate",
    "position",
    "process",
    "geometric_base",
    "object",
)

#: Units for which scalar quantities can safely default to ``one_like``
#: COCOS transformation type — these are sign-invariant under all COCOS
#: conventions.  Do not include ``Wb``, ``T``, ``A``, ``V.s``, ``T.m``
#: or any unit that may carry a COCOS-dependent sign.
SAFE_SCALAR_COCOS_UNITS: frozenset[str] = frozenset(
    {
        "1",
        "m",
        "m^2",
        "m^3",
        "eV",
        "Pa",
        "kg.m^-3",
        "s",
        "s^-1",
        "Hz",
        "m^-3",
        "m.s^-1",
        "A.m^-2",
    }
)


def persist_generated_name_batch(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
) -> int:
    """Persist a single generate-name batch immediately to graph.

    Called from within ``_compose_batch_core`` after LLM success.
    Enriches candidates with provenance metadata, embeds the standard-name
    string, and extracts grammar fields before writing.

    Embedding uses the standard-name string (``id``) — not the description,
    which is added later by the enrich pipeline.  If embedding fails for a
    candidate, it is quarantined (``validation_status='quarantined'``,
    ``validation_issues=['embedding_failed']``) so downstream consumers
    know the vector is missing.

    After writing the StandardName nodes via :func:`write_standard_names`,
    atomically transitions each new SN to ``name_stage='drafted'`` and
    ``docs_stage='pending'`` (chain lengths = 0) and clears the
    ``StandardNameSource`` claim in a **single** Neo4j transaction — so that
    either all stage/claim updates land or none do.

    Returns the number of nodes written.
    """
    from datetime import UTC, datetime

    if not candidates:
        return 0

    now = datetime.now(UTC).isoformat()
    from imas_codex.standard_names.kind_derivation import derive_kind

    for entry in candidates:
        entry.setdefault("model", compose_model)
        entry.setdefault("pipeline_status", "named")
        entry.setdefault("validation_status", "pending")
        entry.setdefault("generated_at", now)
        # Strip private markers used only during in-batch attribution.
        entry.pop("_from_error_sibling", None)
        # D5/P0.3: derive kind from name structure (overrides LLM default).
        name = entry.get("id") or ""
        if name:
            entry["kind"] = derive_kind(name)

        # Default cocos_transformation_type to "one_like" for safe scalars
        # when the extractor / DD node did not already annotate one.
        if (
            not entry.get("cocos_transformation_type")
            and entry.get("kind") == "scalar"
            and (entry.get("unit") or "") in SAFE_SCALAR_COCOS_UNITS
        ):
            entry["cocos_transformation_type"] = "one_like"

    # --- Batch-embed standard-name strings ---
    # Embed the name (id) field in a single batch call for efficiency.
    # The id IS the standard-name string (e.g. "electron_temperature").
    try:
        from imas_codex.embeddings.description import embed_descriptions_batch

        embed_descriptions_batch(candidates, text_field="id")
    except Exception:
        logger.warning(
            "Embedding server unavailable — all %d candidates quarantined",
            len(candidates),
            exc_info=True,
        )
        # Total failure — mark all as quarantined
        for entry in candidates:
            entry["embedding"] = None

    # Set embedded_at for successful embeddings; quarantine failures
    for entry in candidates:
        if entry.get("embedding"):
            entry["embedded_at"] = now
        else:
            entry["validation_status"] = "quarantined"
            existing = entry.get("validation_issues") or []
            if "embedding_failed" not in existing:
                existing = list(existing) + ["embedding_failed"]
            entry["validation_issues"] = existing

    written = write_standard_names(candidates)

    # --- Atomically transition stage + clear source claim ---
    # Build the batch excluding error-sibling candidates (no source node).
    finalize_batch = [
        {
            "sn_id": entry["id"],
            "sns_id": entry.get("source_id"),
            "model": compose_model,
        }
        for entry in candidates
        if entry.get("id") and entry.get("model") != "deterministic:dd_error_modifier"
    ]
    if finalize_batch:
        _finalize_generated_name_stage(finalize_batch)

    return written


@retry_on_deadlock()
def _finalize_generated_name_stage(
    batch: list[dict[str, Any]],
) -> None:
    """Set stage fields on new SNs and clear source claims — single transaction.

    Each item in *batch* must have ``sn_id`` (StandardName id), optionally
    ``sns_id`` (StandardNameSource id), and ``model``.

    In one transaction:
    - ``name_stage = 'drafted'``, ``chain_length = 0``
    - ``docs_stage = 'pending'``, ``docs_chain_length = 0``
    - ``generated_at = datetime()``, ``model = <model>``
    - Source: ``claim_token = null``, ``claimed_at = null``,
      ``status = 'composed'``, ``composed_at = datetime()``,
      ``produced_sn_id = sn.id``
    - Edge: ``(sns)-[:PRODUCED_NAME]->(sn)``
    """
    if not batch:
        return

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                tx.run(
                    """
                    UNWIND $batch AS b
                    MATCH (sn:StandardName {id: b.sn_id})
                    SET sn.name_stage       = 'drafted',
                        sn.chain_length     = 0,
                        sn.docs_stage       = 'pending',
                        sn.docs_chain_length = 0,
                        sn.generated_at     = datetime(),
                        sn.model            = b.model
                    WITH sn, b
                    WHERE b.sns_id IS NOT NULL
                    MATCH (sns:StandardNameSource {id: b.sns_id})
                    SET sns.claim_token  = null,
                        sns.claimed_at   = null,
                        sns.status       = 'composed',
                        sns.composed_at  = datetime(),
                        sns.produced_sn_id = sn.id
                    MERGE (sns)-[:PRODUCED_NAME]->(sn)
                    """,
                    batch=batch,
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_finalize_generated_name_stage: finalized %d SNs",
        len(batch),
    )


@retry_on_deadlock()
def persist_enriched_batch(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
) -> int:
    """Persist enriched StandardName data and REFERENCES relationships.

    Called by the enrich pipeline PERSIST worker after validation and
    embedding.  Each item dict must have at minimum ``id`` and
    ``enriched_description``.  Optional: ``enriched_documentation``,
    ``enriched_links``, ``embedding``, ``llm_model``,
    ``llm_cost``, ``enrich_tokens``, ``validation_status``,
    ``validation_issues``.

    The Cypher MERGE preserves existing values for identity fields
    (``unit``, ``physics_domain``, ``cocos``, ``cocos_transformation_type``,
    ``kind``, ``source_paths``) via ``coalesce(b.field, sn.field)``
    semantics.

    After node updates, REFERENCES relationships are created for each
    link in ``enriched_links`` whose target StandardName exists.

    Enrichment claims (``enrich_claimed_at``, ``enrich_claim_token``)
    are released on all processed nodes.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.

    Returns the number of nodes written.
    """
    if not items:
        return 0

    # Pipeline protection
    from imas_codex.standard_names.protection import filter_protected

    items, skipped = filter_protected(items, override=override)
    if skipped:
        logger.warning(
            "persist_enriched_batch: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not items:
        return 0

    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()

    # --- Build UNWIND batch with safe types ---
    batch = []
    for item in items:
        batch.append(
            {
                "id": item["id"],
                "description": item.get("enriched_description"),
                "documentation": item.get("enriched_documentation"),
                "embedding": item.get("embedding"),
                "pipeline_status": "enriched",
                "enriched_at": now,
                "llm_model": item.get("llm_model") or item.get("enrich_model"),
                "llm_cost": item.get("llm_cost") or item.get("enrich_cost_usd"),
                "llm_service": item.get("llm_service"),
                "enrich_tokens": item.get("enrich_tokens"),
                "validation_status": item.get("validation_status"),
                "validation_issues": item.get("validation_issues") or None,
                "kind": item.get("kind"),
            }
        )

    with GraphClient() as gc:
        # MERGE StandardName nodes — overwrite enrichment fields,
        # preserve identity fields via coalesce.
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            SET sn.description = coalesce(b.description, sn.description),
                sn.documentation = coalesce(b.documentation, sn.documentation),
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = CASE WHEN b.embedding IS NOT NULL
                                      THEN datetime() ELSE sn.embedded_at END,
                sn.pipeline_status = b.pipeline_status,
                sn.enriched_at = datetime(b.enriched_at),
                sn.llm_model = coalesce(b.llm_model, sn.llm_model),
                sn.llm_cost_enrich = CASE WHEN b.llm_cost IS NOT NULL
                                     THEN coalesce(sn.llm_cost_enrich, 0.0) + b.llm_cost
                                     ELSE sn.llm_cost_enrich END,
                sn.llm_cost = CASE WHEN b.llm_cost IS NOT NULL
                              THEN coalesce(sn.llm_cost, 0.0) + b.llm_cost
                              ELSE sn.llm_cost END,
                sn.llm_service = coalesce(b.llm_service, sn.llm_service),
                sn.enrich_tokens = coalesce(b.enrich_tokens, sn.enrich_tokens),
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
                sn.kind = coalesce(b.kind, sn.kind),
                sn.enrich_claimed_at = null,
                sn.enrich_claim_token = null
            """,
            batch=batch,
        )

        # --- REFERENCES relationships for enriched links ---
        link_batch = []
        for item in items:
            links = item.get("enriched_links") or []
            for link in links:
                link_batch.append({"source_id": item["id"], "target_id": link})

        if link_batch:
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (source:StandardName {id: b.source_id})
                MATCH (target:StandardName {id: b.target_id})
                MERGE (source)-[:REFERENCES]->(target)
                """,
                batch=link_batch,
            )

    written = len(batch)
    logger.info("Persisted %d enriched StandardNames", written)
    return written


def write_vocab_gaps(
    gaps: list[dict[str, str]],
    source_type: str = "dd",
    *,
    skip_segment_filter: bool = False,
) -> int:
    """Persist VocabGap nodes and HAS_STANDARD_NAME_VOCAB_GAP relationships.

    Each gap dict has: source_id, segment, needed_token, reason.

    Deduplicates VocabGap nodes by id (vocab_gap:{segment}:{needed_token}).
    Creates HAS_STANDARD_NAME_VOCAB_GAP relationships from source entities with
    per-source reason as a relationship property.

    Args:
        gaps: List of gap dicts.
        source_type: Source type ('dd' or 'signals').
        skip_segment_filter: When ``True``, bypass the open-segment filter.
            Used by auto-VocabGap detection for ``physical_base`` — the
            segment is open (any token is valid) but we still want to
            track novel tokens for ISN review.

    Returns the number of VocabGap nodes written.
    """
    if not gaps:
        return 0

    from datetime import UTC, datetime

    if not skip_segment_filter:
        from imas_codex.standard_names.segments import filter_closed_segment_gaps

        # Drop gaps reported against open-vocabulary segments (e.g. physical_base)
        # and pseudo segments (grammar_ambiguity) — they are not missing tokens.
        gaps, dropped = filter_closed_segment_gaps(gaps)
        if dropped:
            from collections import Counter

            drop_hist = Counter(g.get("segment") for g in dropped)
            logger.info(
                "write_vocab_gaps: skipped %d gaps on open/pseudo segments (%s)",
                len(dropped),
                ", ".join(f"{seg}={n}" for seg, n in drop_hist.most_common()),
            )
        if not gaps:
            return 0

    now = datetime.now(UTC).isoformat()

    # Build deduplicated gap nodes and relationship batch
    gap_nodes: dict[str, dict] = {}
    rel_batch: list[dict] = []

    for g in gaps:
        segment = g["segment"]
        needed_token = g["needed_token"]
        gap_id = f"vocab_gap:{segment}:{needed_token}"

        if gap_id not in gap_nodes:
            gap_nodes[gap_id] = {
                "id": gap_id,
                "segment": segment,
                "needed_token": needed_token,
                "example_count": 0,
            }
        gap_nodes[gap_id]["example_count"] += 1

        rel_batch.append(
            {
                "gap_id": gap_id,
                "source_id": g["source_id"],
                "reason": g.get("reason", ""),
                "observed_at": now,
            }
        )

    with GraphClient() as gc:
        # MERGE VocabGap nodes — increment count, update timestamps
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (vg:VocabGap {id: b.id})
            SET vg.segment = b.segment,
                vg.needed_token = b.needed_token,
                vg.example_count = coalesce(vg.example_count, 0) + b.example_count,
                vg.first_seen_at = coalesce(vg.first_seen_at, datetime()),
                vg.last_seen_at = datetime()
            """,
            batch=list(gap_nodes.values()),
        )

        # Create HAS_STANDARD_NAME_VOCAB_GAP relationships from source entities
        if source_type == "dd":
            # DD sources (IMASNode)
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (vg:VocabGap {id: b.gap_id})
                MATCH (src:IMASNode {id: b.source_id})
                MERGE (src)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
                SET r.reason = b.reason,
                    r.observed_at = datetime(b.observed_at)
                """,
                batch=rel_batch,
            )
        else:
            # Signal sources (FacilitySignal)
            gc.query(
                """
                UNWIND $batch AS b
                MATCH (vg:VocabGap {id: b.gap_id})
                MATCH (src:FacilitySignal {id: b.source_id})
                MERGE (src)-[r:HAS_STANDARD_NAME_VOCAB_GAP]->(vg)
                SET r.reason = b.reason,
                    r.observed_at = datetime(b.observed_at)
                """,
                batch=rel_batch,
            )

    written = len(gap_nodes)
    logger.info("Wrote %d VocabGap nodes from %d gap reports", written, len(gaps))
    return written


# =============================================================================
# Claim/mark/release — graph-state-machine workers
#
# Follows the battle-tested pattern from discovery/code/graph_ops.py:
#   1. claim: ORDER BY rand(), SET claimed_at + claim_token
#   2. verify: re-query by claim_token (prevents double-claim)
#   3. process (caller)
#   4. mark: SET result fields + clear claimed_at/claim_token (token-verified)
#   5. release (on error): clear claimed_at/claim_token (token-verified)
# =============================================================================

_CLAIM_TIMEOUT = "PT300S"  # 5 minutes — matches DEFAULT_CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_names_for_validation(limit: int = 50) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim unvalidated StandardNames for ISN validation.

    Returns ``(token, items)`` where *token* must be passed to
    ``mark_names_validated`` or ``release_validation_claims``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        # Step 1: claim with random ordering and unique token
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.pipeline_status IN ['named', 'drafted']
              AND sn.generated_at IS NOT NULL
              AND sn.validated_at IS NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        # Step 2: verify — only our token
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.object AS object,
                   sn.physics_domain AS physics_domain,
                   collect(DISTINCT src.id) AS source_ids
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def mark_names_validated(
    token: str,
    results: list[dict[str, Any]],
) -> int:
    """Write validation results and release claims atomically.

    Each result dict must have ``id``, ``validation_issues`` (list[str]),
    ``validation_layer_summary`` (JSON string), and ``validation_status``
    (``"valid"`` or ``"quarantined"``).
    Token-verified: only updates nodes still claimed by this token.
    """
    if not results:
        return 0
    batch = []
    for r in results:
        batch.append(
            {
                "id": r["id"],
                "issues": r.get("validation_issues") or [],
                "summary": _ensure_json(r.get("validation_layer_summary")),
                "validation_status": r.get("validation_status", "valid"),
            }
        )
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.validated_at = datetime(),
                sn.validation_issues = b.issues,
                sn.validation_layer_summary = b.summary,
                sn.validation_status = b.validation_status,
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=batch,
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_validation_claims(token: str) -> int:
    """Release validation claims on error. Token-verified."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


@retry_on_deadlock()
def claim_names_for_embedding(limit: int = 100) -> tuple[str, list[dict[str, Any]]]:
    """Atomically claim validated StandardNames needing embedding.

    Returns ``(token, items)`` with ``id`` and ``description``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.pipeline_status IN ['named', 'drafted', 'published', 'accepted']
              AND sn.validated_at IS NOT NULL
              AND sn.embedding IS NULL
              AND sn.description IS NOT NULL
              AND (sn.claimed_at IS NULL
                   OR sn.claimed_at < datetime() - duration($timeout))
            WITH sn ORDER BY rand() LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            token=token,
            timeout=_CLAIM_TIMEOUT,
        )
        results = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            RETURN sn.id AS id, sn.description AS description
            """,
            token=token,
        )
        return token, [dict(r) for r in results]


def mark_names_embedded(
    token: str,
    embed_batch: list[dict[str, Any]],
) -> int:
    """Write embeddings and release claims. Token-verified.

    Each item in *embed_batch* must have ``id`` and ``embedding``.
    """
    if not embed_batch:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id, claim_token: $token})
            SET sn.embedding = b.embedding,
                sn.embedded_at = datetime(),
                sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS marked
            """,
            batch=[
                {"id": e["id"], "embedding": e["embedding"]}
                for e in embed_batch
                if e.get("embedding")
            ],
            token=token,
        )
        return result[0]["marked"] if result else 0


def release_embedding_claims(token: str) -> int:
    """Release embedding claims on error. Token-verified."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null, sn.claim_token = null
            RETURN count(sn) AS released
            """,
            token=token,
        )
        return result[0]["released"] if result else 0


def get_validated_names(
    ids_filter: str | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Query all validated StandardNames for consolidation analysis.

    Read-only — no claims needed since consolidation is a batch analysis.
    Returns drafted names that have ``validated_at`` set and
    ``validation_status`` = ``'valid'``.
    """
    where_parts = [
        "sn.pipeline_status IN ['named', 'drafted']",
        "sn.validated_at IS NOT NULL",
        "sn.validation_status = 'valid'",
    ]
    params: dict[str, Any] = {"limit": limit}

    if ids_filter:
        where_parts.append("ANY(p IN sn.source_paths WHERE p STARTS WITH $ids_prefix)")
        from imas_codex.standard_names.source_paths import ids_prefix_for_source_paths

        params["ids_prefix"] = ids_prefix_for_source_paths(ids_filter)

    where_clause = " AND ".join(where_parts)

    with GraphClient() as gc:
        results = gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {where_clause}
            OPTIONAL MATCH (sn)<-[:HAS_STANDARD_NAME]-(src)
            WITH sn, collect(DISTINCT src.id) AS source_ids
            RETURN sn.id AS id, sn.description AS description,
                   sn.documentation AS documentation, sn.kind AS kind,
                   sn.unit AS unit, sn.links AS links,
                   sn.source_paths AS source_paths,
                   source_ids
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in results]


def mark_names_consolidated(name_ids: list[str]) -> int:
    """Mark names as consolidated (approved by cross-batch analysis)."""
    if not name_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.consolidated_at = datetime()
            RETURN count(sn) AS marked
            """,
            ids=name_ids,
        )
        return result[0]["marked"] if result else 0


# =============================================================================
# Read helpers — publish (validated standard names)
# =============================================================================


def get_validated_standard_names(
    ids_filter: str | None = None,
    pipeline_status: str = "drafted",
) -> list[dict[str, Any]]:
    """Read validated StandardName nodes and their provenance.

    Queries StandardName nodes with the given ``pipeline_status``, joining
    through ``HAS_STANDARD_NAME`` to find source entities and their parent IDS,
    and through ``HAS_UNIT`` to find the unit node.  Uses ``collect()``
    to avoid row duplication when a name has multiple sources (takes the first).

    Parameters
    ----------
    ids_filter:
        Restrict to names derived from a specific IDS (matched via
        ``IMASNode -[:HAS_STANDARD_NAME]-> StandardName`` and
        ``IMASNode -[:IN_IDS]-> IDS``).
    pipeline_status:
        Filter by ``pipeline_status`` property (default ``"drafted"``).

    Returns
    -------
    list of dicts with keys: name, description, documentation, kind,
    unit, links, dd_paths, constraints, validity_domain,
    model, source, source_path, ids_name, source_ids_names.
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {
            "pipeline_status": pipeline_status,
        }

        # Collect source info — use HAS_STANDARD_NAME (entity → concept)
        cypher = """
            MATCH (sn:StandardName)
            WHERE sn.pipeline_status = $pipeline_status
            AND sn.validation_status = 'valid'
            OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
            OPTIONAL MATCH (src)-[:IN_IDS]->(ids:IDS)
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            WITH sn,
                 collect(DISTINCT src.id)[0] AS first_source,
                 collect(DISTINCT ids.id)[0] AS first_ids,
                 collect(DISTINCT ids.id) AS all_ids,
                 u
        """

        if ids_filter:
            # Re-check: at least one HAS_STANDARD_NAME source must be in the target IDS
            cypher += """
            WITH sn, first_source, first_ids, all_ids, u
            WHERE first_ids = $ids_filter
            """
            params["ids_filter"] = ids_filter

        cypher += """
            RETURN sn.id AS name,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.constraints AS constraints,
                   sn.validity_domain AS validity_domain,
                   sn.model AS model,
                   sn.source_types AS source_types,
                   first_source AS source_path,
                   first_ids AS ids_name,
                   all_ids AS source_ids_names,
                   sn.cocos_transformation_type AS cocos_transformation_type,
                   sn.cocos AS cocos,
                   sn.dd_version AS dd_version
            ORDER BY sn.id
        """

        results = gc.query(cypher, **params)
        logger.info(
            "Read %d validated standard names (ids_filter=%s, pipeline_status=%s)",
            len(results),
            ids_filter,
            pipeline_status,
        )
        return list(results)


def reset_standard_names(
    *,
    from_status: str = "drafted",
    to_status: str | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
) -> int:
    """Reset StandardName nodes to allow re-processing.

    Clears transient fields (embedding, embedded_at, model, generated_at)
    and removes HAS_STANDARD_NAME, HAS_UNIT, and
    HAS_COCOS relationships for matching nodes.

    Parameters
    ----------
    from_status:
        Only reset nodes with this ``pipeline_status`` (default ``"drafted"``).
    to_status:
        Target ``pipeline_status`` after reset.  ``None`` (default) clears fields
        only without changing the status.
    source_filter:
        Restrict to nodes with ``source`` equal to ``"dd"`` or ``"signals"``.
    ids_filter:
        Restrict to nodes whose HAS_STANDARD_NAME source path starts with this
        IDS name (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn``).
    dry_run:
        Return the count of matching nodes without modifying anything.
    since:
        Only reset names with ``generated_at >= this`` ISO timestamp.
    before:
        Only reset names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only reset names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only reset names with ``review_tier`` in this list.
    validation_status:
        Only reset names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes reset (or that would be reset in dry-run mode).
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {"from_status": from_status}
        where_clauses = ["sn.pipeline_status = $from_status"]

        if source_filter:
            where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        where = " AND ".join(where_clauses)

        if ids_filter:
            # Match through HAS_STANDARD_NAME to an IMASNode whose id starts with
            # the given IDS name (ids_filter + "/")
            params["ids_prefix"] = ids_filter + "/"
            count_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND src.id STARTS WITH $ids_prefix
                RETURN count(DISTINCT sn) AS n
            """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "reset_standard_names: %d nodes match (from_status=%s, source=%s, ids=%s)",
            count,
            from_status,
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        if ids_filter:
            # Collect matching SN ids first, then operate on them
            collect_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {where}
                AND src.id STARTS WITH $ids_prefix
                RETURN DISTINCT sn.id AS sn_id
            """
            rows = gc.query(collect_cypher, **params)
            sn_ids = [r["sn_id"] for r in rows]
            reset_params: dict[str, Any] = {"sn_ids": sn_ids}
            node_match = "MATCH (sn:StandardName) WHERE sn.id IN $sn_ids"
        else:
            reset_params = dict(params)
            if ids_filter:
                reset_params["ids_prefix"] = ids_filter + "/"
            node_match = f"MATCH (sn:StandardName) WHERE {where}"

        # Remove HAS_STANDARD_NAME, HAS_UNIT, and HAS_COCOS relationships
        gc.query(
            f"""
            {node_match}
            OPTIONAL MATCH (src)-[r:HAS_STANDARD_NAME]->(sn)
            DELETE r
            """,
            **reset_params,
        )
        gc.query(
            f"""
            {node_match}
            OPTIONAL MATCH (sn)-[r:HAS_UNIT]->(u)
            DELETE r
            """,
            **reset_params,
        )
        gc.query(
            f"""
            {node_match}
            OPTIONAL MATCH (sn)-[r:HAS_COCOS]->(c)
            DELETE r
            """,
            **reset_params,
        )

        # Clear transient fields, optionally set new status
        if to_status is not None:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null, "
                "sn.pipeline_status = $to_status"
            )
            reset_params["to_status"] = to_status
        else:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null"
            )

        gc.query(
            f"""
            {node_match}
            SET {set_clause}
            """,
            **reset_params,
        )

    logger.info("Reset %d StandardName nodes", count)
    return count


def clear_standard_names(
    *,
    status_filter: list[str] | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    include_accepted: bool = False,
    dry_run: bool = False,
    since: str | None = None,
    before: str | None = None,
    below_score: float | None = None,
    tiers: list[str] | None = None,
    validation_status: str | None = None,
) -> int:
    """Delete StandardName nodes and their relationships.

    Safety model (relationship-first):

    1. If ``ids_filter`` or ``source_filter`` is set, delete matching
       ``HAS_STANDARD_NAME`` relationships first.
    2. Then delete ``StandardName`` nodes that have zero remaining
       ``HAS_STANDARD_NAME`` edges.

    By default only nodes with ``pipeline_status = 'drafted'`` are deleted.
    Accepted names require ``include_accepted=True``.

    Parameters
    ----------
    status_filter:
        List of ``pipeline_status`` values to delete (default ``["drafted"]``).
    source_filter:
        Restrict to nodes with ``source`` equal to ``"dd"`` or ``"signals"``.
    ids_filter:
        Delete only names linked to an IMASNode whose id starts with this IDS
        name.  Relationships are removed first; nodes become orphans and are
        then deleted.
    include_accepted:
        When ``True``, ``"accepted"`` names are eligible for deletion even if
        not listed in ``status_filter``.
    dry_run:
        Return the count of nodes that would be deleted without modifying
        anything.
    since:
        Only clear names with ``generated_at >= this`` ISO timestamp.
    before:
        Only clear names with ``generated_at < this`` ISO timestamp.
    below_score:
        Only clear names with ``reviewer_score < this`` value (0.0–1.0).
    tiers:
        Only clear names with ``review_tier`` in this list.
    validation_status:
        Only clear names with this ``validation_status`` value.

    Returns
    -------
    Number of nodes deleted (or that would be deleted in dry-run mode).
    """
    all_statuses = status_filter is None
    effective_statuses = list(status_filter) if status_filter else []
    if status_filter is not None:
        if include_accepted and "accepted" not in effective_statuses:
            effective_statuses.append("accepted")
        elif not include_accepted and "accepted" in effective_statuses:
            effective_statuses.remove("accepted")

    with GraphClient() as gc:
        params: dict[str, Any] = {}
        sn_where_clauses: list[str] = []
        if all_statuses:
            if not include_accepted:
                sn_where_clauses.append("sn.pipeline_status <> 'accepted'")
        else:
            params["statuses"] = effective_statuses
            sn_where_clauses.append("sn.pipeline_status IN $statuses")

        if source_filter:
            sn_where_clauses.append("$source_filter IN sn.source_types")
            params["source_filter"] = source_filter

        if since:
            sn_where_clauses.append("sn.generated_at >= datetime($since)")
            params["since"] = since
        if before:
            sn_where_clauses.append("sn.generated_at < datetime($before)")
            params["before"] = before
        if below_score is not None:
            sn_where_clauses.append("sn.reviewer_score_name < $below_score")
            params["below_score"] = below_score
        if tiers:
            sn_where_clauses.append("sn.review_tier IN $tiers")
            params["tiers"] = tiers
        if validation_status:
            sn_where_clauses.append("sn.validation_status = $validation_status")
            params["validation_status"] = validation_status

        sn_where = " AND ".join(sn_where_clauses) if sn_where_clauses else "true"

        if ids_filter:
            params["ids_prefix"] = ids_filter + "/"
            count_cypher = f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {sn_where}
                AND src.id STARTS WITH $ids_prefix
                RETURN count(DISTINCT sn) AS n
            """
        else:
            count_cypher = f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                RETURN count(sn) AS n
            """

        result = gc.query(count_cypher, **params)
        count = result[0]["n"] if result else 0
        logger.info(
            "clear_standard_names: %d nodes match (statuses=%s, source=%s, ids=%s)",
            count,
            effective_statuses,
            source_filter,
            ids_filter,
        )

        if dry_run or count == 0:
            return count

        # Step A: delete StandardNameReview nodes attached to in-scope StandardName nodes
        # (HAS_REVIEW edge goes StandardName -> StandardNameReview). DETACH DELETE on the
        # StandardName alone orphans the StandardNameReview node; we must delete it explicitly.
        if ids_filter:
            gc.query(
                f"""
                MATCH (src:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
                WHERE {sn_where}
                AND src.id STARTS WITH $ids_prefix
                DETACH DELETE r
                """,
                **params,
            )
        else:
            gc.query(
                f"""
                MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
                WHERE {sn_where}
                DETACH DELETE r
                """,
                **params,
            )

        # Step B: delete StandardName nodes and their remaining edges
        if ids_filter:
            gc.query(
                f"""
                MATCH (src:IMASNode)-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {sn_where}
                AND src.id STARTS WITH $ids_prefix
                DELETE r
                """,
                **params,
            )
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                AND NOT EXISTS {{ MATCH ()-[:HAS_STANDARD_NAME]->(sn) }}
                DETACH DELETE sn
                """,
                **params,
            )
        else:
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                DETACH DELETE sn
                """,
                **params,
            )

        # Step C: sweep up any fully-orphaned StandardNameReview nodes left by prior clears
        # (pre-p39 runs detached StandardName without deleting StandardNameReview).
        orphan_result = gc.query(
            """
            MATCH (r:StandardNameReview)
            WHERE NOT EXISTS { MATCH (:StandardName)-[:HAS_REVIEW]->(r) }
            WITH r LIMIT 10000
            DETACH DELETE r
            RETURN count(r) AS n
            """
        )
        orphan_count = orphan_result[0]["n"] if orphan_result else 0
        if orphan_count:
            logger.info("Swept %d orphaned StandardNameReview nodes", orphan_count)

        # Step D: delete all LLMCost rows — they represent cost for pipeline
        # runs whose StandardName output is now being cleared. Leaving them
        # behind would accumulate stale cost-ledger rows across reset cycles.
        gc.query("MATCH (c:LLMCost) DETACH DELETE c")

    logger.info("Deleted %d StandardName nodes", count)
    return count


def clear_sn_subsystem(
    *,
    dry_run: bool = False,
) -> dict[str, int]:
    """Wipe every Standard Name the pipeline has produced.

    Deletes the six labels owned by the SN pipeline output:

    * ``StandardName`` — the generated names
    * ``StandardNameReview`` — RD-quorum review records
    * ``StandardNameSource`` — per-path extraction tracking
    * ``VocabGap`` — grammar vocabulary gap reports
    * ``SNRun`` — run audit / rotation memory
    * ``LLMCost`` — LLM call cost ledger rows

    **Grammar nodes** (``GrammarToken``, ``GrammarSegment``,
    ``GrammarTemplate``, ``ISNGrammarVersion``) are ISN-authoritative
    reference data and are never touched. They stay in the graph so the
    vocabulary is immediately available for the next ``sn run``. Use
    ``sn sync-grammar`` to re-sync the grammar from a new ISN release.

    Parameters
    ----------
    dry_run:
        Count matching nodes without modifying the graph.

    Returns
    -------
    Dict mapping node label to deleted count. In dry-run mode,
    values are the current counts that would be deleted.
    """
    counts: dict[str, int] = {}
    labels = (
        "StandardName",
        "StandardNameReview",
        "StandardNameSource",
        "VocabGap",
        "SNRun",
        "LLMCost",
    )

    with GraphClient() as gc:

        def _count(label: str) -> int:
            r = gc.query(f"MATCH (n:{label}) RETURN count(n) AS n")
            return r[0]["n"] if r else 0

        for label in labels:
            counts[label] = _count(label)

        if dry_run:
            return counts

        # Delete order is significant: StandardNameReview BEFORE StandardName so
        # orphan StandardNameReview nodes can't linger if HAS_STANDARD_NAME edges
        # are missing (pre-p39 bug). DETACH DELETE handles remaining
        # edges on each pass. At SN-pipeline scale (~thousands of
        # nodes total) a single DETACH DELETE per label is sub-second.
        gc.query("MATCH (r:StandardNameReview) DETACH DELETE r")
        gc.query("MATCH (sn:StandardName) DETACH DELETE sn")
        gc.query("MATCH (s:StandardNameSource) DETACH DELETE s")
        gc.query("MATCH (v:VocabGap) DETACH DELETE v")
        gc.query("MATCH (rr:SNRun) DETACH DELETE rr")
        gc.query("MATCH (c:LLMCost) DETACH DELETE c")

    total = sum(counts.values())
    logger.info("clear_sn_subsystem: deleted %d nodes (%s)", total, counts)

    return counts


def update_review_status(names: list[str], status: str = "published") -> int:
    """Update pipeline_status for a batch of StandardName nodes.

    Parameters
    ----------
    names:
        List of StandardName node IDs (``sn.id``) to update.
    status:
        New ``pipeline_status`` value (default ``"published"``).

    Returns
    -------
    Number of nodes updated.
    """
    if not names:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $names AS name
            MATCH (sn:StandardName {id: name})
            SET sn.pipeline_status = $status
            RETURN count(sn) AS updated
            """,
            names=names,
            status=status,
        )
        count = result[0]["updated"] if result else 0
        logger.info("Updated pipeline_status to '%s' for %d names", status, count)
        return count


# =============================================================================
# Link resolution
# =============================================================================

_MAX_LINK_RETRIES = 5


def claim_unresolved_links(limit: int = 20) -> list[dict[str, Any]]:
    """Claim StandardName nodes with unresolved links for resolution.

    Uses age-weighted random selection: nodes not checked recently have
    higher priority, preventing spin on temporarily unresolvable links.

    Returns list of dicts with ``id``, ``links``, ``link_retry_count``.
    """
    import uuid

    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.link_status = 'unresolved'
              AND sn.claimed_at IS NULL
              AND coalesce(sn.link_retry_count, 0) < $max_retries
            WITH sn,
                 duration.between(
                     coalesce(sn.link_checked_at, datetime('2020-01-01')),
                     datetime()
                 ).minutes + 1.0 AS age_minutes
            ORDER BY rand() * age_minutes DESC
            LIMIT $limit
            SET sn.claimed_at = datetime(), sn.claim_token = $token
            """,
            limit=limit,
            max_retries=_MAX_LINK_RETRIES,
            token=token,
        )
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName {claim_token: $token})
                RETURN sn.id AS id, sn.links AS links,
                       coalesce(sn.link_retry_count, 0) AS retry_count
                """,
                token=token,
            )
        )
    return [dict(r) for r in rows]


def resolve_links_batch(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
    override_names: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve dd: links to name: links for a batch of names.

    For each ``dd:path`` link, checks if a StandardName exists that was
    generated from that path. If found, replaces with ``name:sn_id``.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.
    override_names:
        Selective override — set of name IDs that should bypass protection
        even if they have ``origin='catalog_edit'``.
    """
    # Pipeline protection — links is a protected field
    from imas_codex.standard_names.protection import filter_protected

    items, skipped = filter_protected(
        items, override=override, override_names=override_names
    )
    if skipped:
        logger.warning(
            "resolve_links_batch: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )

    resolved_count = 0
    still_unresolved = 0
    failed_count = 0

    with GraphClient() as gc:
        # Build lookup: dd_path -> standard_name_id
        all_dd_paths = set()
        for item in items:
            for link in item.get("links") or []:
                if link.startswith("dd:"):
                    all_dd_paths.add(link[3:])

        path_to_name: dict[str, str] = {}
        if all_dd_paths:
            rows = gc.query(
                """
                UNWIND $paths AS path
                MATCH (n:IMASNode {id: path})-[:HAS_STANDARD_NAME]->(sn:StandardName)
                RETURN path AS dd_path, sn.id AS sn_id
                """,
                paths=list(all_dd_paths),
            )
            for r in rows:
                path_to_name[r["dd_path"]] = r["sn_id"]

        for item in items:
            links = list(item.get("links") or [])
            new_links = []
            any_unresolved = False

            for link in links:
                if link.startswith("dd:"):
                    dd_path = link[3:]
                    if dd_path in path_to_name:
                        new_links.append(f"name:{path_to_name[dd_path]}")
                    else:
                        new_links.append(link)
                        any_unresolved = True
                else:
                    new_links.append(link)

            retry_count = item.get("retry_count", 0) + 1

            if any_unresolved and retry_count >= _MAX_LINK_RETRIES:
                status = "failed"
                failed_count += 1
            elif any_unresolved:
                status = "unresolved"
                still_unresolved += 1
            else:
                status = "resolved"
                resolved_count += 1

            gc.query(
                """
                MATCH (sn:StandardName {id: $id})
                SET sn.links = $links,
                    sn.link_status = $status,
                    sn.link_checked_at = datetime(),
                    sn.link_retry_count = $retry_count,
                    sn.claimed_at = null,
                    sn.claim_token = null
                """,
                id=item["id"],
                links=new_links,
                status=status,
                retry_count=retry_count,
            )

    return {
        "resolved": resolved_count,
        "unresolved": still_unresolved,
        "failed": failed_count,
    }


# =============================================================================
# Enrichment helpers — documentation iteration (Phase 3D)
# =============================================================================


def get_enrichment_candidates(
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    status_filter: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Get StandardName nodes that need documentation enrichment.

    Returns dicts with: id, description, documentation, kind, unit,
    physics_domain, pipeline_status, plus all linked DD paths aggregated with
    their documentation.
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {}
        where_clauses: list[str] = []

        if ids_filter:
            where_clauses.append(
                "EXISTS { MATCH (src)-[:HAS_STANDARD_NAME]->(sn) "
                "MATCH (src)-[:IN_IDS]->(ids:IDS {id: $ids_filter}) }"
            )
            params["ids_filter"] = ids_filter
        if domain_filter:
            where_clauses.append("$domain_filter IN sn.physics_domain")
            params["domain_filter"] = domain_filter
        if status_filter:
            where_clauses.append("sn.pipeline_status = $status_filter")
            params["status_filter"] = status_filter

        where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        limit_clause = ""
        if limit:
            limit_clause = "LIMIT $limit"
            params["limit"] = limit

        results = gc.query(
            f"""
            MATCH (sn:StandardName)
            {where}
            OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
            OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
            WHERE src:IMASNode OR src:FacilitySignal
            WITH sn, u,
                 collect(DISTINCT {{
                     path: src.id,
                     description: src.description,
                     documentation: src.documentation
                 }}) AS dd_paths
            RETURN sn.id AS id,
                   sn.description AS description,
                   sn.documentation AS documentation,
                   sn.kind AS kind,
                   coalesce(u.id, sn.unit) AS unit,
                   sn.links AS links,
                   sn.validity_domain AS validity_domain,
                   sn.constraints AS constraints,
                   sn.physics_domain AS physics_domain,
                   sn.pipeline_status AS pipeline_status,
                   dd_paths
            ORDER BY sn.id
            {limit_clause}
            """,
            **params,
        )

        candidates = []
        for r in results or []:
            row = dict(r)
            # Filter out null-path entries from the OPTIONAL MATCH
            dd_paths = row.get("dd_paths") or []
            row["dd_paths"] = [p for p in dd_paths if p.get("path") is not None]
            candidates.append(row)

        logger.info("Found %d enrichment candidates", len(candidates))
        return candidates


def write_enrichment_results(
    results: list[dict[str, Any]],
    *,
    override: bool = False,
) -> int:
    """Write enrichment results back to graph.

    Only updates doc fields: description, documentation, links,
    validity_domain, constraints. Clears review_input_hash to invalidate
    stale reviews.

    Does NOT touch: id, kind, unit, model, grammar_parse_version, validation_diagnostics_json, etc.

    Parameters
    ----------
    override:
        When ``True``, bypass pipeline protection on catalog-edited names.

    Returns the number of nodes updated.
    """
    if not results:
        return 0

    # Pipeline protection
    from imas_codex.standard_names.protection import filter_protected

    results, skipped = filter_protected(results, override=override)
    if skipped:
        logger.warning(
            "write_enrichment_results: stripped protected fields from %d name(s): %s",
            len(skipped),
            ", ".join(skipped[:5]),
        )
    if not results:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.description = b.description,
                sn.documentation = b.documentation,
                sn.links = b.links,
                sn.validity_domain = b.validity_domain,
                sn.constraints = b.constraints,
                sn.link_status = b.link_status,
                sn.enriched_at = datetime(),
                sn.review_input_hash = null
            """,
            batch=[
                {
                    "id": r["id"],
                    "description": r.get("description") or "",
                    "documentation": r.get("documentation") or "",
                    "links": r.get("links") or None,
                    "validity_domain": r.get("validity_domain"),
                    "constraints": r.get("constraints") or None,
                    "link_status": _compute_link_status(r.get("links")),
                }
                for r in results
            ],
        )

    logger.info("Enriched %d StandardName nodes", len(results))
    return len(results)


# =============================================================================
# StandardNameSource CRUD
# =============================================================================

_VALID_PIPELINE_SOURCE_TYPES = {"dd", "signals"}


def merge_standard_name_sources(
    sources: list[dict],
    *,
    force: bool = False,
) -> int:
    """Batch MERGE StandardNameSource nodes.

    Each source dict must have: id, source_type, source_id, batch_key, status.
    Optional: description, dd_path (for DD sources), signal (for signal sources).

    On CREATE: sets all fields.
    On MATCH (existing node):
      - If force=True: resets to extracted, clears attempt_count/last_error/failed_at.
      - If status is 'stale': requeues to extracted.
      - Otherwise: preserves existing status (skip already-processed sources).

    Rejects source_type values not in {'dd', 'signals'} with ValueError.
    Returns count of nodes created or updated.
    """
    if not sources:
        return 0

    invalid = {s.get("source_type") for s in sources} - _VALID_PIPELINE_SOURCE_TYPES
    if invalid:
        raise ValueError(
            f"Invalid source_type(s) for pipeline: {invalid}. "
            f"Only {_VALID_PIPELINE_SOURCE_TYPES} are valid."
        )

    # Birth-invariant: a 'dd' source with no dd_path cannot have a FROM_DD_PATH
    # edge and would be an orphan from the moment it is written.  Drop these
    # before they reach the graph and log a WARNING so the caller knows.
    valid_sources: list[dict] = []
    for s in sources:
        if s.get("source_type") == "dd" and not s.get("dd_path"):
            logger.warning(
                "merge_standard_name_sources: skipping dd source %r — "
                "no dd_path supplied; writing would create an orphan "
                "(birth invariant violation).",
                s.get("id"),
            )
            continue
        valid_sources.append(s)
    sources = valid_sources
    if not sources:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.batch_key = src.batch_key,
                sns.status = src.status,
                sns.description = src.description,
                sns.attempt_count = 0
            ON MATCH SET
                sns.batch_key = src.batch_key,
                sns.description = coalesce(src.description, sns.description),
                sns.status = CASE
                    WHEN $force THEN 'extracted'
                    WHEN sns.status = 'stale' THEN 'extracted'
                    ELSE sns.status
                END,
                sns.attempt_count = CASE
                    WHEN $force THEN 0
                    ELSE sns.attempt_count
                END,
                sns.last_error = CASE
                    WHEN $force THEN null
                    ELSE sns.last_error
                END,
                sns.failed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.failed_at
                END,
                sns.claimed_at = CASE
                    WHEN $force THEN null
                    ELSE sns.claimed_at
                END,
                sns.claim_token = CASE
                    WHEN $force THEN null
                    ELSE sns.claim_token
                END
            WITH sns, src
            // Create typed relationships to source entities
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
            force=force,
        )
        return result[0]["affected"] if result else 0


@retry_on_deadlock()
def claim_standard_name_source_batch(
    batch_key: str,
    *,
    limit: int = 50,
    timeout_minutes: int = 30,
) -> tuple[str, list[dict]]:
    """Atomic full-batch claim of StandardNameSource nodes by batch_key.

    Claims up to ``limit`` extracted sources with matching batch_key.
    Uses two-step token verification to prevent double-claiming.
    Reclaims sources with stale claims (older than timeout_minutes).

    Returns (claim_token, claimed_sources) where each source dict has
    id, source_id, source_type, batch_key, description.
    """
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        # Step 1: Claim
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.batch_key = $batch_key
              AND (
                (sns.status = 'extracted' AND sns.claimed_at IS NULL)
                OR (sns.claimed_at IS NOT NULL
                    AND sns.claimed_at < datetime() - duration({minutes: $timeout}))
              )
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token
            """,
            batch_key=batch_key,
            limit=limit,
            timeout=timeout_minutes,
            token=token,
        )
        # Step 2: Verify
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description
                """,
                token=token,
            )
        )
    return token, claimed


def fetch_claimed_source_metadata(token: str) -> list[dict]:
    """Fetch full metadata for claimed sources, joining source entities.

    For DD sources: joins IMASNode for documentation, unit, cluster info.
    For signal sources: joins FacilitySignal for description, unit, diagnostic.

    Returns list of dicts with source + joined metadata.
    """
    with GraphClient() as gc:
        return list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                OPTIONAL MATCH (sns)-[:FROM_DD_PATH]->(imas:IMASNode)
                OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)
                OPTIONAL MATCH (imas)<-[:CONTAINS_PATH]-(c:SemanticCluster)
                OPTIONAL MATCH (sns)-[:FROM_SIGNAL]->(sig:FacilitySignal)
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description,
                       imas.id AS dd_path,
                       imas.documentation AS dd_documentation,
                       u.id AS unit,
                       c.label AS cluster_label,
                       c.scope AS cluster_scope,
                       sig.id AS signal_id,
                       sig.description AS signal_description
                """,
                token=token,
            )
        )


def mark_sources_composed(
    token: str,
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as composed and link to the produced StandardName.

    Token-verified: only updates sources matching the claim_token.
    Creates PRODUCED_NAME relationship to the StandardName.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            MATCH (sn:StandardName {id: $sn_id})
            SET sns.status = 'composed',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = sn.id
            MERGE (sns)-[:PRODUCED_NAME]->(sn)
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


def mark_sources_attached(
    token: str,
    source_ids: list[str],
    standard_name_id: str,
) -> int:
    """Mark sources as auto-attached to an existing StandardName.

    Used when a source matches an existing name without needing LLM composition.
    Token-verified. Creates PRODUCED_NAME relationship.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            MATCH (sn:StandardName {id: $sn_id})
            SET sns.status = 'attached',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.produced_sn_id = sn.id
            MERGE (sns)-[:PRODUCED_NAME]->(sn)
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            sn_id=standard_name_id,
        )
        return result[0]["affected"] if result else 0


def mark_sources_vocab_gap(
    token: str,
    source_ids: list[str],
) -> int:
    """Mark sources as blocked by missing grammar vocabulary.

    Token-verified. Clears claim but preserves attempt_count.
    Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            SET sns.status = 'vocab_gap',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
        )
        return result[0]["affected"] if result else 0


def mark_sources_failed(
    token: str,
    source_ids: list[str],
    error: str,
    *,
    max_attempts: int = 3,
) -> int:
    """Mark sources as failed with durable retry.

    Increments attempt_count. If below max_attempts, returns to 'extracted'
    for retry. At max_attempts, transitions to terminal 'failed' status.
    Token-verified. Returns count of updated sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid, claim_token: $token})
            SET sns.attempt_count = coalesce(sns.attempt_count, 0) + 1,
                sns.last_error = $error,
                sns.claimed_at = null,
                sns.claim_token = null,
                sns.status = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN 'failed'
                    ELSE 'extracted'
                END,
                sns.failed_at = CASE
                    WHEN coalesce(sns.attempt_count, 0) + 1 >= $max_attempts
                    THEN datetime()
                    ELSE sns.failed_at
                END
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
            token=token,
            error=error,
            max_attempts=max_attempts,
        )
        return result[0]["affected"] if result else 0


def mark_sources_stale(source_ids: list[str]) -> int:
    """Mark sources as stale (source entity no longer exists).

    Not token-verified — can be called from reconciliation outside claim context.
    Returns count of updated sources.
    """
    if not source_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $source_ids AS sid
            MATCH (sns:StandardNameSource {id: sid})
            SET sns.status = 'stale',
                sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            source_ids=source_ids,
        )
        return result[0]["affected"] if result else 0


def write_skipped_sources(records: list[dict]) -> int:
    """Record DD paths that cannot be resolved to a standard name source.

    Each record must have: source_type, source_id, skip_reason,
    skip_reason_detail. Optional: description, dd_path (auto-derived from
    source_id for DD sources), signal (for signal sources), status
    (defaults to ``'skipped'`` for backward compatibility; may be
    ``'not_physical_quantity'`` for configuration metadata).

    The ``id`` is derived as ``{source_type}:{source_id}`` (matches the
    existing ``merge_standard_name_sources`` key convention).

    Idempotent — subsequent writes for the same id refresh skip_reason/
    skip_reason_detail but do not re-enqueue for composition.

    Returns count of nodes written.
    """
    if not records:
        return 0

    sources = []
    for r in records:
        source_type = r["source_type"]
        source_id = r["source_id"]
        sources.append(
            {
                "id": f"{source_type}:{source_id}",
                "source_type": source_type,
                "source_id": source_id,
                "skip_reason": r["skip_reason"],
                "skip_reason_detail": r.get("skip_reason_detail", ""),
                "description": r.get("description", ""),
                "status": r.get("status", "skipped"),
                "dd_path": r.get("dd_path")
                or (source_id if source_type == "dd" else None),
                "signal": r.get("signal")
                or (source_id if source_type == "signals" else None),
            }
        )

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sources AS src
            MERGE (sns:StandardNameSource {id: src.id})
            ON CREATE SET
                sns.source_type = src.source_type,
                sns.source_id = src.source_id,
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = src.description,
                sns.attempt_count = 0
            ON MATCH SET
                sns.status = src.status,
                sns.skip_reason = src.skip_reason,
                sns.skip_reason_detail = src.skip_reason_detail,
                sns.description = coalesce(src.description, sns.description),
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns, src
            FOREACH (_ IN CASE WHEN src.source_type = 'dd' AND src.dd_path IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (imas:IMASNode {id: src.dd_path})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
            )
            FOREACH (_ IN CASE WHEN src.source_type = 'signals' AND src.signal IS NOT NULL
                          THEN [1] ELSE [] END |
                MERGE (sig:FacilitySignal {id: src.signal})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
            )
            RETURN count(sns) AS affected
            """,
            sources=sources,
        )
        return result[0]["affected"] if result else 0


def list_skipped_sources(
    limit: int = 100,
    reason: str | None = None,
) -> list[dict]:
    """Query skipped/not_physical_quantity StandardNameSource records.

    Returns a list of dicts with keys: id, source_type, source_id,
    skip_reason, skip_reason_detail, description.

    Args:
        limit: Maximum rows to return.
        reason: Optional skip_reason filter.
    """
    where = "sns.status IN ['skipped', 'not_physical_quantity']"
    params: dict = {"limit": limit}
    if reason is not None:
        where += " AND sns.skip_reason = $reason"
        params["reason"] = reason

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN sns.id AS id,
                   sns.source_type AS source_type,
                   sns.source_id AS source_id,
                   sns.skip_reason AS skip_reason,
                   sns.skip_reason_detail AS skip_reason_detail,
                   sns.description AS description
            ORDER BY sns.skip_reason, sns.id
            LIMIT $limit
            """,
            **params,
        )
        return [dict(r) for r in result]


def get_skipped_source_counts(
    source_type: str | None = None,
) -> dict[str, int]:
    """Return counts of skipped StandardNameSource records grouped by skip_reason."""
    where = "sns.status = 'skipped'"
    params: dict = {}
    if source_type is not None:
        where += " AND sns.source_type = $source_type"
        params["source_type"] = source_type

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource)
            WHERE {where}
            RETURN coalesce(sns.skip_reason, 'unknown') AS skip_reason,
                   count(sns) AS cnt
            ORDER BY cnt DESC
            """,
            **params,
        )
        return {r["skip_reason"]: r["cnt"] for r in result}


def release_standard_name_source_claims(token: str) -> int:
    """Release all claims held by this token without changing status.

    Used for error recovery — release claims so other workers can pick them up.
    Returns count of released sources.
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource {claim_token: $token})
            SET sns.claimed_at = null,
                sns.claim_token = null
            RETURN count(sns) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


# =============================================================================
# Polling-based work claiming — compose and review
# =============================================================================

_CLAIM_TIMEOUT_SECONDS = 300  # 5 minutes — matches DEFAULT_CLAIM_TIMEOUT_SECONDS


@retry_on_deadlock()
def claim_compose_sources(
    *,
    limit: int = 15,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[dict]]:
    """Claim extracted StandardNameSource nodes for composition (polling).

    Unlike :func:`claim_standard_name_source_batch` this does NOT filter by
    ``batch_key`` — it claims any extracted source. Workers use this in a
    polling loop to pick up the next available batch of work regardless of
    which batch grouping the extract phase produced.

    Uses the standard ``ORDER BY rand()`` + ``claim_token`` two-step verify
    pattern from the discovery pipelines.

    Returns ``(token, claimed_sources)`` where each source dict has keys
    ``id``, ``source_id``, ``source_type``, ``batch_key``, ``description``.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with GraphClient() as gc:
        # Step 1: Atomically claim with random ordering + unique token
        gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            WITH sns ORDER BY rand() LIMIT $limit
            SET sns.claimed_at = datetime(),
                sns.claim_token = $token
            """,
            limit=limit,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        claimed = list(
            gc.query(
                """
                MATCH (sns:StandardNameSource {claim_token: $token})
                RETURN sns.id AS id,
                       sns.source_id AS source_id,
                       sns.source_type AS source_type,
                       sns.batch_key AS batch_key,
                       sns.description AS description
                """,
                token=token,
            )
        )

    logger.debug(
        "claim_compose_sources: requested %d, won %d (token=%s)",
        limit,
        len(claimed),
        token[:8],
    )
    return token, [dict(r) for r in claimed]


def count_eligible_compose_sources(
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> int:
    """Count StandardNameSource nodes eligible for composition.

    Returns the number of extracted, unclaimed (or stale-claimed) sources.
    Used by polling workers for drain detection — when this returns 0 and
    no active leases remain, the compose phase is complete.
    """
    cutoff = f"PT{timeout_seconds}S"
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sns:StandardNameSource)
            WHERE sns.status = 'extracted'
              AND (sns.claimed_at IS NULL
                   OR sns.claimed_at < datetime() - duration($cutoff))
            RETURN count(sns) AS cnt
            """,
            cutoff=cutoff,
        )
        return result[0]["cnt"] if result else 0


@retry_on_deadlock()
def claim_review_names(
    name_ids: list[str],
    *,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> tuple[str, list[str]]:
    """Claim specific StandardName nodes for review scoring.

    Only claims names from *name_ids* that are still eligible — not already
    claimed by another worker (unless the claim is stale).

    Uses the same ``claim_token`` two-step verify pattern as the compose
    claim functions.

    Returns ``(token, actually_claimed_ids)``.
    """
    if not name_ids:
        return "", []

    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    with GraphClient() as gc:
        # Step 1: Atomically claim unclaimed/stale-claimed names
        gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            WHERE sn.claimed_at IS NULL
               OR sn.claimed_at < datetime() - duration($cutoff)
            SET sn.claimed_at = datetime(),
                sn.claim_token = $token
            """,
            ids=name_ids,
            token=token,
            cutoff=cutoff,
        )
        # Step 2: Verify — only our token
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            RETURN sn.id AS id
            """,
            token=token,
        )
        claimed = [r["id"] for r in result] if result else []

    logger.debug(
        "claim_review_names: requested %d, won %d (token=%s)",
        len(name_ids),
        len(claimed),
        token[:8],
    )
    return token, claimed


def release_review_claims(token: str) -> int:
    """Release all StandardName claims held by this token.

    Clears ``claimed_at`` and ``claim_token`` without changing any other
    fields.  Used for error recovery — released names become eligible for
    other workers.

    Returns count of released names.
    """
    if not token:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {claim_token: $token})
            SET sn.claimed_at = null,
                sn.claim_token = null
            RETURN count(sn) AS affected
            """,
            token=token,
        )
        return result[0]["affected"] if result else 0


def reconcile_standard_name_sources(source_type: str = "dd") -> dict:
    """Post-rebuild reconciliation of StandardNameSource nodes.

    1. Re-links sources to DD paths/signals that still exist
    2. Marks sources as stale if their upstream entity is gone
    3. Revives stale sources if their entity reappears

    Returns dict with counts: {relinked, stale_marked, revived}.
    """
    with GraphClient() as gc:
        if source_type == "dd":
            # Find stale: sources whose DD path no longer exists
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'dd'})
                    WHERE NOT (sns)-[:FROM_DD_PATH]->(:IMASNode)
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = mark_sources_stale(stale_ids)

            # Revive: stale sources whose DD path now exists again
            revived = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'dd', status: 'stale'})
                MATCH (imas:IMASNode {id: sns.source_id})
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            # Re-link: ensure FROM_DD_PATH exists for non-stale sources
            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'dd'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_DD_PATH]->()
                MATCH (imas:IMASNode {id: sns.source_id})
                MERGE (sns)-[:FROM_DD_PATH]->(imas)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0
        else:
            # Signal reconciliation (same pattern, different relationships)
            stale = list(
                gc.query(
                    """
                    MATCH (sns:StandardNameSource {source_type: 'signals'})
                    WHERE NOT (sns)-[:FROM_SIGNAL]->(:FacilitySignal)
                    AND NOT (sns.status = 'stale')
                    RETURN sns.id AS id
                    """
                )
            )
            stale_ids = [r["id"] for r in stale]
            stale_count = mark_sources_stale(stale_ids)

            revived = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'signals', status: 'stale'})
                MATCH (sig:FacilitySignal {id: sns.source_id})
                SET sns.status = 'extracted',
                    sns.claimed_at = null,
                    sns.claim_token = null
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            revived_count = revived[0]["count"] if revived else 0

            relinked = gc.query(
                """
                MATCH (sns:StandardNameSource {source_type: 'signals'})
                WHERE NOT (sns.status = 'stale')
                  AND NOT (sns)-[:FROM_SIGNAL]->()
                MATCH (sig:FacilitySignal {id: sns.source_id})
                MERGE (sns)-[:FROM_SIGNAL]->(sig)
                RETURN count(sns) AS count
                """
            )
            relinked_count = relinked[0]["count"] if relinked else 0

    return {
        "stale_marked": stale_count,
        "revived": revived_count,
        "relinked": relinked_count,
    }


def reconcile_error_siblings() -> dict[str, int]:
    """Detect and mark stale error-sibling StandardNames.

    An error-sibling StandardName is identified by
    ``model='deterministic:dd_error_modifier'``. It is orphaned when
    no parent StandardName exists that the sibling's uncertainty
    operator wraps. Detection: strip the ``upper_uncertainty_of_`` /
    ``lower_uncertainty_of_`` / ``uncertainty_index_of_`` prefix and
    check whether the resulting parent name still has a StandardName
    node in the graph.

    Orphans are marked ``pipeline_status='skipped'`` to prevent them
    from appearing in downstream phases.

    Returns dict with counts: {stale_marked}.
    """
    from imas_codex.standard_names.error_siblings import ERROR_SUFFIX_TO_OPERATOR

    # Build the list of operator prefixes to strip
    prefixes = [f"{op}_of_" for op in ERROR_SUFFIX_TO_OPERATOR.values()]

    with GraphClient() as gc:
        # Find all error-sibling names
        rows = list(
            gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.model = 'deterministic:dd_error_modifier'
                  AND sn.pipeline_status <> 'skipped'
                RETURN sn.id AS id
                """
            )
            or []
        )

        orphan_ids: list[str] = []
        for row in rows:
            sn_id = row["id"]
            parent_name = None
            for prefix in prefixes:
                if sn_id.startswith(prefix):
                    parent_name = sn_id[len(prefix) :]
                    break

            if parent_name is None:
                # Can't determine parent — skip
                continue

            # Check if parent StandardName exists
            parent_check = list(
                gc.query(
                    "MATCH (p:StandardName {id: $pid}) RETURN p.id LIMIT 1",
                    pid=parent_name,
                )
                or []
            )
            if not parent_check:
                orphan_ids.append(sn_id)

        stale_count = 0
        if orphan_ids:
            gc.query(
                """
                UNWIND $ids AS sid
                MATCH (sn:StandardName {id: sid})
                SET sn.pipeline_status = 'skipped'
                """,
                ids=orphan_ids,
            )
            stale_count = len(orphan_ids)
            logger.info(
                "Reconciled %d orphaned error-sibling StandardNames → skipped",
                stale_count,
            )

    return {"stale_marked": stale_count}


def get_standard_name_source_stats(
    source_type: str | None = None,
) -> dict[str, int]:
    """Get StandardNameSource status counts.

    Returns dict mapping status → count. Optionally filtered by source_type.
    """
    with GraphClient() as gc:
        where = ""
        params: dict = {}
        if source_type:
            where = "WHERE sns.source_type = $source_type"
            params["source_type"] = source_type
        result = gc.query(
            f"""
            MATCH (sns:StandardNameSource) {where}
            RETURN sns.status AS status, count(sns) AS count
            """,
            **params,
        )
        return {r["status"]: r["count"] for r in result}


def write_run_provenance(
    name_ids: list[str],
    run_id: str,
    turn_number: int = 1,
) -> int:
    """Stamp run provenance fields on StandardName nodes.

    Sets ``last_run_id``, ``last_run_at``, and ``last_turn_number`` on every
    name touched by the current ``sn run`` invocation.

    Args:
        name_ids: StandardName ids to stamp.
        run_id: UUID string identifying this ``sn run`` invocation.
        turn_number: Turn number supplied via ``--turn-number``.

    Returns:
        Number of nodes updated.
    """
    if not name_ids:
        return 0

    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (sn:StandardName {id: nid})
            SET sn.last_run_id = $rid,
                sn.last_run_at = datetime($ts),
                sn.last_turn_number = $tn
            RETURN count(sn) AS n
            """,
            ids=name_ids,
            rid=run_id,
            ts=now,
            tn=turn_number,
        )
        return result[0]["n"] if result else 0


# =============================================================================
# Review comment export (Phase F — anti-pattern feedback loop)
# =============================================================================


def export_review_comments(
    output_path: str | Path,
    *,
    domain: str | None = None,
) -> int:
    """Dump StandardNameReview node comment data to a JSONL file before ``sn clear``.

    Queries all ``StandardNameReview`` nodes (optionally filtered by the parent
    ``StandardName.physics_domain``) and writes one JSON record per
    line to *output_path*.  Each record contains:

    * ``name`` — ``StandardName.id``
    * ``domain`` — ``StandardName.physics_domain``
    * ``reviewer_model`` — the model that produced the review
    * ``score`` — numeric score (0–1)
    * ``verdict`` — accept / reject / revise
    * ``comments_per_dim`` — parsed dict of per-dimension comments
    * ``comments`` — full free-text comment string
    * ``review_axis`` — "names" or "docs"
    * ``generated_at`` — ``StandardName.generated_at`` ISO string
    * ``reviewed_at`` — ``StandardNameReview.reviewed_at`` ISO string

    Parameters
    ----------
    output_path:
        Destination file path.  Parent directories are created if absent.
    domain:
        When provided, restrict to reviews on names with this
        ``physics_domain``.

    Returns
    -------
    Number of StandardNameReview records written (0 if none found).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {}
    where_clauses: list[str] = []
    if domain:
        where_clauses.append("$domain IN sn.physics_domain")
        params["domain"] = domain

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    cypher = f"""
        MATCH (sn:StandardName)-[:HAS_REVIEW]->(r:StandardNameReview)
        {where}
        RETURN sn.id AS name,
               sn.physics_domain AS domain,
               r.reviewer_model AS reviewer_model,
               r.score AS score,
               r.verdict AS verdict,
               r.comments_per_dim_json AS comments_per_dim_json,
               r.comments AS comments,
               r.review_axis AS review_axis,
               sn.generated_at AS generated_at,
               r.reviewed_at AS reviewed_at
        ORDER BY sn.id, r.reviewed_at
    """

    with GraphClient() as gc:
        rows = gc.query(cypher, **params)

    if not rows:
        logger.info(
            "export_review_comments: no StandardNameReview nodes found (domain=%s)",
            domain,
        )
        return 0

    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            # Parse comments_per_dim_json if it's a string
            cpd_raw = row.get("comments_per_dim_json")
            if isinstance(cpd_raw, str):
                try:
                    cpd = json.loads(cpd_raw)
                except (json.JSONDecodeError, ValueError):
                    cpd = {}
            elif isinstance(cpd_raw, dict):
                cpd = cpd_raw
            else:
                cpd = {}

            record: dict[str, Any] = {
                "name": row.get("name"),
                "domain": row.get("domain"),
                "reviewer_model": row.get("reviewer_model"),
                "score": row.get("score"),
                "verdict": row.get("verdict"),
                "comments_per_dim": cpd,
                "comments": row.get("comments"),
                "review_axis": row.get("review_axis"),
                "generated_at": str(row["generated_at"])
                if row.get("generated_at")
                else None,
                "reviewed_at": str(row["reviewed_at"])
                if row.get("reviewed_at")
                else None,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    logger.info(
        "export_review_comments: wrote %d records to %s (domain=%s)",
        count,
        output_path,
        domain,
    )
    return count


# =============================================================================
# Graph-backed LLM cost tracking API  (Phase 2)
# =============================================================================

# Phase → StandardName.llm_cost_<suffix> field mapping.
_PHASE_TO_SN_COST_FIELD: dict[str, str] = {
    "generate": "llm_cost_compose",
    "enrich": "llm_cost_enrich",
    "review_names": "llm_cost_review",
    "review_docs": "llm_cost_docs",
    "regen": "llm_cost_regen",
}


def create_sn_run_open(
    run_id: str,
    *,
    started_at: Any,
    cost_limit: float,
    turn_number: int = 1,
    min_score: float | None = None,
) -> None:
    """Pre-create an ``SNRun`` node with ``status='started'``.

    Called at the START of ``run_sn_loop`` so that ``(LLMCost)-[:FOR_RUN]->
    (SNRun)`` edges have a target from the first LLM call onward.

    Uses MERGE so repeated calls (e.g. after a retry) are safe.
    """
    from imas_codex.graph.models import SNRun

    run = SNRun(
        id=run_id,
        started_at=started_at,
        cost_limit=round(cost_limit, 6),
        cost_spent=0.0,
        turn_number=turn_number,
        min_score=min_score,
        status="started",
        cost_is_exact=True,
    )
    props = run.model_dump(mode="json")
    try:
        with GraphClient() as gc:
            gc.create_nodes("SNRun", [props])
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to pre-create SNRun %s: %s", run_id, exc)


def finalize_sn_run(
    run_id: str,
    *,
    status: str,
    cost_spent: float,
    cost_is_exact: bool = True,
    ended_at: Any,
    **summary_fields: Any,
) -> None:
    """Update an existing ``SNRun`` node at run end.

    Uses ``MATCH + SET`` (not CREATE) — the node must already exist
    (created by :func:`create_sn_run_open`).

    ``summary_fields`` may contain any other ``SNRun`` property such as
    ``domains_touched``, ``stop_reason``, ``pipeline_hash``,
    ``names_composed``, ``names_enriched``, etc.
    """
    set_clauses = [
        "rr.status = $status",
        "rr.cost_spent = $cost_spent",
        "rr.cost_is_exact = $cost_is_exact",
        "rr.ended_at = datetime($ended_at)",
    ]
    params: dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "cost_spent": round(cost_spent, 6),
        "cost_is_exact": cost_is_exact,
        "ended_at": ended_at if isinstance(ended_at, str) else ended_at.isoformat(),
    }

    for key, value in summary_fields.items():
        set_clauses.append(f"rr.{key} = ${key}")
        params[key] = value

    cypher = (
        "MATCH (rr:SNRun {id: $run_id}) "
        "SET " + ", ".join(set_clauses) + " "
        "RETURN rr.id AS id"
    )
    try:
        with GraphClient() as gc:
            result = gc.query(cypher, **params)
            if not result:
                logger.warning("finalize_sn_run: no SNRun found with id=%s", run_id)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to finalize SNRun %s: %s", run_id, exc)


@retry_on_deadlock()
def record_llm_cost(
    *,
    run_id: str,
    phase: str,
    cycle: str | None = None,
    sn_ids: list[str] | None = None,
    model: str,
    cost: float,
    tokens_in: int,
    tokens_out: int,
    tokens_cached_read: int = 0,
    tokens_cached_write: int = 0,
    service: str = "standard-names",
    batch_id: str | None = None,
    overspend: float = 0.0,
    llm_at: Any | None = None,
) -> str:
    """Write an atomic ``LLMCost`` node and ``FOR_RUN`` edge.

    **Idempotency contract**: ``id`` is a deterministic UUID-5 over
    ``(run_id, phase, batch_id, model, llm_at_iso, cost, tokens_in,
    tokens_out)``.  The node is written with ``CREATE`` (not MERGE).
    If a uniqueness constraint violation fires (duplicate id), the
    exception is swallowed — the previous write already succeeded.

    Returns:
        The deterministic ``id`` string.
    """
    from datetime import UTC, datetime

    if llm_at is None:
        llm_at = datetime.now(UTC)
    llm_at_iso = llm_at.isoformat() if not isinstance(llm_at, str) else llm_at

    # Deterministic id — uuid5 over immutable call identity
    id_seed = (
        f"{run_id}|{phase}|{batch_id}|{model}"
        f"|{llm_at_iso}|{cost}|{tokens_in}|{tokens_out}"
    )
    spend_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_seed))

    sn_ids_clean = list(sn_ids) if sn_ids else []

    cypher = """
        CREATE (c:LLMCost {
            id: $id,
            run_id: $run_id,
            phase: $phase,
            cycle: $cycle,
            sn_ids: $sn_ids,
            batch_id: $batch_id,
            overspend: $overspend,
            llm_model: $llm_model,
            llm_cost: $llm_cost,
            llm_tokens_in: $llm_tokens_in,
            llm_tokens_out: $llm_tokens_out,
            llm_tokens_cached_read: $llm_tokens_cached_read,
            llm_tokens_cached_write: $llm_tokens_cached_write,
            llm_service: $llm_service,
            llm_at: datetime($llm_at),
            for_run: $run_id
        })
        WITH c
        MATCH (rr:SNRun {id: $run_id})
        MERGE (c)-[:FOR_RUN]->(rr)
    """
    params = {
        "id": spend_id,
        "run_id": run_id,
        "phase": phase,
        "cycle": cycle,
        "sn_ids": sn_ids_clean,
        "batch_id": batch_id,
        "overspend": round(overspend, 6),
        "llm_model": model,
        "llm_cost": round(cost, 6),
        "llm_tokens_in": tokens_in,
        "llm_tokens_out": tokens_out,
        "llm_tokens_cached_read": tokens_cached_read,
        "llm_tokens_cached_write": tokens_cached_write,
        "llm_service": service,
        "llm_at": llm_at_iso,
    }

    try:
        with GraphClient() as gc:
            gc.query(cypher, **params)
    except Exception as exc:
        # Swallow constraint violation (idempotent duplicate) —
        # the original write already succeeded.
        from neo4j.exceptions import ConstraintError

        if isinstance(exc, ConstraintError):
            logger.debug(
                "record_llm_cost: duplicate id=%s (idempotent), skipping",
                spend_id,
            )
        else:
            raise

    return spend_id


def aggregate_spend_for_run(run_id: str) -> float:
    """Return total LLM cost for a run by summing ``LLMCost`` nodes."""
    with GraphClient() as gc:
        result = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) "
            "RETURN coalesce(sum(c.llm_cost), 0.0) AS total",
            run_id=run_id,
        )
        return float(result[0]["total"]) if result else 0.0


def aggregate_spend_per_phase(run_id: str) -> dict[str, float]:
    """Return ``{phase: total_cost}`` for a run."""
    with GraphClient() as gc:
        rows = gc.query(
            "MATCH (c:LLMCost {run_id: $run_id}) "
            "RETURN c.phase AS phase, sum(c.llm_cost) AS total "
            "ORDER BY phase",
            run_id=run_id,
        )
        return {r["phase"]: float(r["total"]) for r in rows}


def aggregate_spend_per_name(run_id: str) -> dict[str, float]:
    """Return ``{sn_id: apportioned_cost}`` for a run.

    Per-name cost share is ``llm_cost / size(sn_ids)`` — each name
    in the ``sn_ids`` list gets an equal share.  Rows with an empty
    ``sn_ids`` list are skipped (e.g. L7 audit calls with no names).
    """
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:LLMCost {run_id: $run_id})
            WHERE size(c.sn_ids) > 0
            UNWIND c.sn_ids AS sn_id
            RETURN sn_id, sum(c.llm_cost / size(c.sn_ids)) AS apportioned
            """,
            run_id=run_id,
        )
        return {r["sn_id"]: float(r["apportioned"]) for r in rows}


def update_sn_per_phase_costs(run_id: str) -> int:
    """Push aggregated per-name costs into ``StandardName.llm_cost_*`` fields.

    For each ``LLMCost`` row in the run, apportions ``llm_cost / size(sn_ids)``
    to each name, grouped by phase.  Then writes the per-phase totals and the
    overall ``llm_cost`` onto the ``StandardName`` node.

    Returns:
        Number of ``StandardName`` nodes updated.
    """
    # Build per-(name, phase) apportionment
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (c:LLMCost {run_id: $run_id})
            WHERE size(c.sn_ids) > 0
            UNWIND c.sn_ids AS sn_id
            RETURN sn_id, c.phase AS phase,
                   sum(c.llm_cost / size(c.sn_ids)) AS apportioned
            """,
            run_id=run_id,
        )

    if not rows:
        return 0

    # Aggregate: {sn_id: {field: cost, ...}}
    per_name: dict[str, dict[str, float]] = {}
    for r in rows:
        sn_id = r["sn_id"]
        phase = r["phase"]
        cost = float(r["apportioned"])
        if sn_id not in per_name:
            per_name[sn_id] = {}
        field = _PHASE_TO_SN_COST_FIELD.get(phase)
        if field:
            per_name[sn_id][field] = per_name[sn_id].get(field, 0.0) + cost

    # Write back — batch all names in a single Cypher per phase-field
    updated_ids: set[str] = set()

    with GraphClient() as gc:
        for sn_id, fields in per_name.items():
            total_cost = sum(fields.values())
            set_parts = ["sn.llm_cost = $total"]
            params: dict[str, Any] = {
                "sn_id": sn_id,
                "total": round(total_cost, 6),
            }
            for field_name, field_cost in fields.items():
                set_parts.append(f"sn.{field_name} = ${field_name}")
                params[field_name] = round(field_cost, 6)

            result = gc.query(
                "MATCH (sn:StandardName {id: $sn_id}) "
                "SET " + ", ".join(set_parts) + " "
                "RETURN sn.id AS id",
                **params,
            )
            if result:
                updated_ids.add(sn_id)

    return len(updated_ids)


# =============================================================================
# Seed-and-expand claims  (Phase 8 worker pools)
#
# Each function follows the H4 pattern from plan.md Phase 8:
#   1. Seed: pick one random eligible row, SET claimed_at + claim_token
#   2. Expand: claim up to batch_size−1 more with same (cluster_id, unit);
#      fallback: (physics_domain, unit) when cluster_id is absent
#   3. Read-back: query by claim_token — only our token's rows returned
#
# Compose targets StandardNameSource; the other four target StandardName.
# =============================================================================

_DEFAULT_SEED_EXPAND_BATCH = 15


def _claim_sn_atomic(
    *,
    eligibility_where: str,
    query_params: dict[str, Any],
    batch_size: int,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
    extra_return_fields: str = "",
    stage_field: str | None = None,
    to_stage: str | None = None,
) -> list[dict[str, Any]]:
    """Single-transaction stage-aware claim for StandardName nodes.

    All steps — seed, expand, and read-back — execute inside **one** Neo4j
    transaction.  If any step fails, the entire transaction rolls back and
    no partial claim state leaks into the graph.

    Callers should wrap this with ``@retry_on_deadlock()`` so that a
    ``TransientError`` (deadlock) causes the whole function — including
    token generation — to retry cleanly.

    Parameters
    ----------
    eligibility_where:
        Cypher WHERE fragment applied to ``sn`` (must NOT include ``WHERE``
        keyword itself).  May reference parameters in *query_params* via
        ``$name`` syntax.
    query_params:
        Extra Cypher parameters consumed by *eligibility_where*.
    batch_size:
        Maximum batch size (seed + expand).
    timeout_seconds:
        Stale-claim recovery window.
    extra_return_fields:
        Additional ``RETURN`` columns appended after the common set.
        Must start with a comma if non-empty, e.g.
        ``", sn.enriched_at AS enriched_at"``.
    stage_field:
        Optional stage property to transition at claim time
        (``'name_stage'`` or ``'docs_stage'``).  When *None* (default),
        no stage transition occurs — only ``claim_token`` and
        ``claimed_at`` are written.
    to_stage:
        Target stage value for the transition (e.g. ``'refining'``).
        Ignored when *stage_field* is *None*.

    Returns
    -------
    list[dict]
        Claimed items as plain dicts.  Empty list when no eligible seed
        exists.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"
    params: dict[str, Any] = {**query_params, "token": token, "cutoff": cutoff}

    # Build optional SET clause for atomic stage transition.
    stage_set = ""
    if stage_field and to_stage:
        stage_set = f", sn.{stage_field} = $to_stage"
        params["to_stage"] = to_stage

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                # ── Step 1: Seed ─────────────────────────────────
                seed_result = list(
                    tx.run(
                        f"""
                        MATCH (sn:StandardName)
                        WHERE {eligibility_where}
                          AND (sn.claimed_at IS NULL
                               OR sn.claimed_at < datetime()
                                    - duration($cutoff))
                        WITH sn ORDER BY rand() LIMIT 1
                        SET sn.claimed_at = datetime(),
                            sn.claim_token = $token
                            {stage_set}
                        WITH sn
                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        RETURN c.id AS _cluster_id, u.id AS _unit,
                               sn.physics_domain AS _physics_domain
                        """,
                        **params,
                    )
                )

                if not seed_result:
                    tx.close()
                    return []

                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                physics_domain = seed.get("_physics_domain")
                expand_limit = batch_size - 1

                # ── Step 2: Expand ───────────────────────────────
                if expand_limit > 0:
                    expand_params: dict[str, Any] = {
                        **params,
                        "expand_limit": expand_limit,
                    }
                    if cluster_id is not None and unit is not None:
                        expand_params.update(cluster_id=cluster_id, unit=unit)
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                            MATCH (sn)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            MATCH (sn)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif cluster_id is not None:
                        expand_params["cluster_id"] = cluster_id
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                            MATCH (sn)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif physics_domain is not None and unit is not None:
                        expand_params.update(fallback_domain=physics_domain, unit=unit)
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              AND $fallback_domain IN sn.physics_domain
                            MATCH (sn)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    elif physics_domain is not None:
                        expand_params["fallback_domain"] = physics_domain
                        tx.run(
                            f"""
                            MATCH (sn:StandardName)
                            WHERE {eligibility_where}
                              AND sn.claimed_at IS NULL
                              AND $fallback_domain IN sn.physics_domain
                            WITH sn LIMIT $expand_limit
                            SET sn.claimed_at = datetime(),
                                sn.claim_token = $token
                                {stage_set}
                            """,
                            **expand_params,
                        )
                    # else: no grouping key — seed-only batch

                # ── Step 3: Read-back by token ───────────────────
                results = list(
                    tx.run(
                        f"""
                        MATCH (sn:StandardName {{claim_token: $token}})
                        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                        OPTIONAL MATCH (sn)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        RETURN sn.id AS id,
                               sn.description AS description,
                               sn.documentation AS documentation,
                               sn.kind AS kind,
                               coalesce(u.id, sn.unit) AS unit,
                               c.id AS cluster_id,
                               sn.physics_domain AS physics_domain,
                               sn.validation_status
                                   AS validation_status,
                               sn.claim_token AS claim_token
                               {extra_return_fields}
                        """,
                        token=token,
                    )
                )

                items = [dict(r) for r in results]
                tx.commit()

            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "_claim_sn_atomic: claimed %d (token=%s)",
        len(items),
        token[:8],
    )
    return items


# -- generate_name (StandardNameSource) -------------------------------------


@retry_on_deadlock()
def claim_generate_name_seed_and_expand(
    facility: str | None = None,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardNameSource nodes (status='extracted') for name generation.

    Seed-and-expand: one random seed is claimed, then up to
    ``batch_size - 1`` additional sources sharing the same
    ``(cluster_id, unit)`` (via ``FROM_DD_PATH``→``IN_CLUSTER`` /
    ``HAS_UNIT``).  When no cluster exists the fallback key is
    ``(physics_domain, unit)``.

    All three steps (seed, expand, read-back) execute inside a **single**
    Neo4j transaction so that no partial claim state leaks on deadlock
    retry.

    Parameters
    ----------
    facility:
        Optional facility id to restrict claims to signal-backed sources
        from that facility.  ``None`` means all sources.
    batch_size:
        Maximum batch size including the seed.
    timeout_seconds:
        Stale-claim recovery window.

    Returns
    -------
    list[dict]
        Claimed sources as dicts with keys ``id``, ``source_id``,
        ``source_type``, ``batch_key``, ``description``.
    """
    token = str(uuid.uuid4())
    cutoff = f"PT{timeout_seconds}S"

    # Optional facility filter (only applies to signal sources).
    facility_where = ""
    extra_params: dict[str, Any] = {}
    if facility:
        facility_where = (
            "AND (sns.source_type = 'dd' OR EXISTS {"
            "  MATCH (sns)-[:FROM_SIGNAL]->(:FacilitySignal)"
            "    -[:AT_FACILITY]->(:Facility {id: $facility})"
            "})"
        )
        extra_params["facility"] = facility

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                # ── Step 1: Seed ─────────────────────────────────
                seed_result = list(
                    tx.run(
                        f"""
                        MATCH (sns:StandardNameSource)
                        WHERE sns.status = 'extracted'
                          AND (sns.claimed_at IS NULL
                               OR sns.claimed_at < datetime()
                                    - duration($cutoff))
                          {facility_where}
                        WITH sns ORDER BY rand() LIMIT 1
                        SET sns.claimed_at = datetime(),
                            sns.claim_token = $token
                        WITH sns
                        OPTIONAL MATCH (sns)-[:FROM_DD_PATH]
                            ->(imas:IMASNode)
                        OPTIONAL MATCH (imas)-[:IN_CLUSTER]
                            ->(c:IMASSemanticCluster)
                        OPTIONAL MATCH (imas)-[:HAS_UNIT]->(u:Unit)
                        OPTIONAL MATCH (sns)-[:FROM_SIGNAL]
                            ->(sig:FacilitySignal)
                        RETURN c.id AS _cluster_id,
                               CASE WHEN u IS NOT NULL THEN u.id
                                    WHEN sig IS NOT NULL
                                    THEN sig.unit
                                    ELSE null END AS _unit,
                               CASE WHEN imas IS NOT NULL
                                    THEN imas.physics_domain
                                    WHEN sig IS NOT NULL
                                    THEN sig.physics_domain
                                    ELSE null END
                                        AS _physics_domain,
                               sns.batch_key AS _batch_key
                        """,
                        token=token,
                        cutoff=cutoff,
                        **extra_params,
                    )
                )

                if not seed_result:
                    tx.close()
                    return []

                seed = dict(seed_result[0])
                cluster_id = seed.get("_cluster_id")
                unit = seed.get("_unit")
                physics_domain = seed.get("_physics_domain")
                batch_key = seed.get("_batch_key")
                expand_limit = batch_size - 1

                # ── Step 2: Expand ───────────────────────────────
                if expand_limit > 0:
                    expanded = False
                    if cluster_id is not None and unit is not None:
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND sns.claimed_at IS NULL
                              {facility_where}
                            MATCH (sns)-[:FROM_DD_PATH]
                                ->(imas:IMASNode)
                            MATCH (imas)-[:IN_CLUSTER]
                                ->(:IMASSemanticCluster
                                    {{id: $cluster_id}})
                            MATCH (imas)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token
                            """,
                            token=token,
                            cluster_id=cluster_id,
                            unit=unit,
                            expand_limit=expand_limit,
                            **extra_params,
                        )
                        expanded = True
                    elif physics_domain is not None and unit is not None:
                        # IMASNode.physics_domain is a scalar String;
                        # use = not IN.
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND sns.claimed_at IS NULL
                              {facility_where}
                            MATCH (sns)-[:FROM_DD_PATH]
                                ->(imas:IMASNode)
                            WHERE imas.physics_domain
                                = $fallback_domain
                            MATCH (imas)-[:HAS_UNIT]
                                ->(:Unit {{id: $unit}})
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token
                            """,
                            token=token,
                            fallback_domain=physics_domain,
                            unit=unit,
                            expand_limit=expand_limit,
                            **extra_params,
                        )
                        expanded = True

                    if not expanded and batch_key:
                        # Last resort: group by batch_key.
                        tx.run(
                            f"""
                            MATCH (sns:StandardNameSource)
                            WHERE sns.status = 'extracted'
                              AND sns.claimed_at IS NULL
                              AND sns.batch_key = $batch_key
                              {facility_where}
                            WITH sns LIMIT $expand_limit
                            SET sns.claimed_at = datetime(),
                                sns.claim_token = $token
                            """,
                            token=token,
                            batch_key=batch_key,
                            expand_limit=expand_limit,
                            **extra_params,
                        )

                # ── Step 3: Read-back ────────────────────────────
                results = list(
                    tx.run(
                        """
                        MATCH (sns:StandardNameSource
                               {claim_token: $token})
                        RETURN sns.id AS id,
                               sns.source_id AS source_id,
                               sns.source_type AS source_type,
                               sns.batch_key AS batch_key,
                               sns.description AS description,
                               sns.claim_token AS claim_token
                        """,
                        token=token,
                    )
                )

                items = [dict(r) for r in results]
                tx.commit()

            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    logger.debug(
        "claim_generate_name_seed_and_expand: claimed %d (token=%s)",
        len(items),
        token[:8],
    )
    return items


# -- enrich (StandardName, enriched_at IS NULL) -------------------------------


@retry_on_deadlock()
def claim_enrich_seed_and_expand(
    min_score_threshold: float = 0.0,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes that lack documentation enrichment.

    Eligibility: ``validation_status = 'valid'`` AND ``enriched_at IS NULL``.

    Returns claimed items as dicts.
    """
    where = "sn.validation_status = 'valid' AND sn.enriched_at IS NULL"
    params: dict[str, Any] = {}

    return _claim_sn_atomic(
        eligibility_where=where,
        query_params=params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=", sn.enriched_at AS enriched_at",
    )


# -- review names (StandardName, reviewed_name_at IS NULL) --------------------


@retry_on_deadlock()
def claim_review_names_seed_and_expand(
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    min_score: float = DEFAULT_MIN_SCORE,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name-axis review scoring.

    Eligibility: ``reviewed_name_at IS NULL`` AND
    ``validation_status = 'valid'``.

    B3 exclusivity: names with ``reviewer_score_name < min_score`` are
    reserved for the regen pool and excluded here via
    ``coalesce(sn.reviewer_score_name, 1.0) >= $min_score``.
    """
    where = (
        "sn.reviewed_name_at IS NULL"
        " AND sn.validation_status = 'valid'"
        " AND coalesce(sn.reviewer_score_name, 1.0) >= $min_score"
    )
    return _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewed_name_at AS reviewed_name_at"
        ),
    )


# -- review_name (Phase 8.1: name_stage='drafted', claim only) ----------------


@retry_on_deadlock()
def claim_review_name_seed_and_expand(
    facility: str | None = None,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name review (Phase 8.1 stage machine).

    Eligibility: ``name_stage = 'drafted'`` AND ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_name` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``chain_length``,
    ``claim_token``.
    """
    where = "sn.name_stage = 'drafted'"
    if facility is not None:
        where += " AND sn.facility = $facility"
        query_params: dict[str, Any] = {"facility": facility}
    else:
        query_params = {}
    return _claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.name AS name"
            ", sn.tags AS tags"
            ", coalesce(sn.chain_length, 0) AS chain_length"
            ", sn.name_stage AS name_stage"
        ),
    )


# -- persist_reviewed_name (Phase 8.1: write review + stage transition) -------


@retry_on_deadlock()
def persist_reviewed_name(
    *,
    sn_id: str,
    claim_token: str,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    verdict: str,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
) -> str:
    """Persist name-review results and transition ``name_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``verdict == 'accept'`` AND ``score >= min_score``
       - ``'exhausted'`` if ``chain_length >= rotation_cap - 1``
         AND ``score < min_score`` (cap reached, no further refine)
       - ``'reviewed'`` otherwise (eligible for refine_name pickup)
    3. SET reviewer fields, ``name_stage``, clear claim state.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_name``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_name``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_name``).
    verdict:
        LLM verdict string (``'accept'``, ``'reject'``, or ``'revise'``).
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_name_seed_and_expand`).

    Returns
    -------
    str
        The new ``name_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    # Read current chain_length — needed for exhaustion check.
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.claim_token = $token
            RETURN coalesce(sn.chain_length, 0) AS chain_length
            """,
            id=sn_id,
            token=claim_token,
        )

    if not rows:
        logger.debug(
            "persist_reviewed_name: token mismatch for %s (token=%s) — no-op",
            sn_id,
            claim_token[:8],
        )
        return ""

    chain_length: int = int(rows[0]["chain_length"])

    # ── Stage decision ────────────────────────────────────────────────
    if verdict == "accept" and score >= min_score:
        target_stage = "accepted"
    elif chain_length >= rotation_cap - 1 and score < min_score:
        target_stage = "exhausted"
    else:
        target_stage = "reviewed"

    scores_json = _json.dumps(scores) if scores is not None else None
    comments_per_dim_json = (
        _json.dumps(comments_per_dim) if comments_per_dim is not None else None
    )

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.claim_token = $token
            SET sn.reviewer_score_name        = $score,
                sn.reviewer_scores_name       = $scores_json,
                sn.reviewer_comments_name     = $comments,
                sn.reviewer_comments_per_dim_name = $comments_per_dim_json,
                sn.reviewer_verdict_name      = $verdict,
                sn.reviewer_model_name        = $model,
                sn.reviewed_name_at           = datetime(),
                sn.name_stage                 = $target_stage,
                sn.claim_token                = null,
                sn.claimed_at                 = null
            """,
            id=sn_id,
            token=claim_token,
            score=score,
            scores_json=scores_json,
            comments=comments,
            comments_per_dim_json=comments_per_dim_json,
            verdict=verdict,
            model=model,
            target_stage=target_stage,
        )

    logger.info(
        "persist_reviewed_name: %s → name_stage=%s (score=%.3f, chain=%d/%d)",
        sn_id,
        target_stage,
        score,
        chain_length,
        rotation_cap,
    )
    return target_stage


# -- review_docs (Phase 8.1: docs_stage='drafted', claim only) ----------------


@retry_on_deadlock()
def claim_review_docs_seed_and_expand(
    facility: str | None = None,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs review (Phase 8.1 stage machine).

    Eligibility: ``docs_stage = 'drafted'`` AND ``claimed_at IS NULL``.

    Does NOT transition stage at claim time — stage remains ``'drafted'``
    until :func:`persist_reviewed_docs` writes the final outcome.  Only
    ``claim_token`` and ``claimed_at`` are set so the orphan sweep can
    recover stuck claims.

    Returns claimed items as dicts with keys:
    ``id``, ``name``, ``description``, ``documentation``, ``kind``,
    ``unit``, ``tags``, ``physics_domain``, ``docs_chain_length``,
    ``claim_token``.
    """
    where = "sn.docs_stage = 'drafted'"
    if facility is not None:
        where += " AND sn.facility = $facility"
        query_params: dict[str, Any] = {"facility": facility}
    else:
        query_params = {}
    return _claim_sn_atomic(
        eligibility_where=where,
        query_params=query_params,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.name AS name"
            ", sn.tags AS tags"
            ", coalesce(sn.docs_chain_length, 0) AS docs_chain_length"
            ", sn.docs_stage AS docs_stage"
        ),
    )


# -- persist_reviewed_docs (Phase 8.1: write review + docs_stage transition) --


@retry_on_deadlock()
def persist_reviewed_docs(
    *,
    sn_id: str,
    claim_token: str,
    score: float,
    scores: dict[str, Any] | None = None,
    comments: str | None = None,
    comments_per_dim: dict[str, Any] | None = None,
    verdict: str,
    model: str,
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
) -> str:
    """Persist docs-review results and transition ``docs_stage``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. Compute target stage:
       - ``'accepted'`` if ``verdict == 'accept'`` AND ``score >= min_score``
       - ``'exhausted'`` if ``docs_chain_length >= rotation_cap - 1``
         AND ``score < min_score`` (cap reached, no further refine)
       - ``'reviewed'`` otherwise (eligible for refine_docs pickup)
    3. SET reviewer_docs fields, ``docs_stage``, clear claim state.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    score:
        Normalised review score (0-1).
    scores:
        Per-dimension sub-scores dict (written as JSON to
        ``reviewer_scores_docs``).
    comments:
        Free-text reviewer comments (written to ``reviewer_comments_docs``).
    comments_per_dim:
        Per-dimension comments dict (written as JSON to
        ``reviewer_comments_per_dim_docs``).
    verdict:
        LLM verdict string (``'accept'``, ``'reject'``, or ``'revise'``).
    model:
        LLM model slug used for this review.
    min_score:
        Acceptance threshold.
    rotation_cap:
        Maximum chain depth before exhaustion (same value used by
        :func:`claim_refine_docs_seed_and_expand`).

    Returns
    -------
    str
        The new ``docs_stage`` value (``'accepted'``, ``'reviewed'``, or
        ``'exhausted'``).  Returns ``''`` when the token did not match
        (no-op).
    """
    import json as _json

    # Read current docs_chain_length — needed for exhaustion check.
    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.claim_token = $token
            RETURN coalesce(sn.docs_chain_length, 0) AS docs_chain_length
            """,
            id=sn_id,
            token=claim_token,
        )

    if not rows:
        logger.debug(
            "persist_reviewed_docs: token mismatch for %s (token=%s) — no-op",
            sn_id,
            claim_token[:8],
        )
        return ""

    docs_chain_length: int = int(rows[0]["docs_chain_length"])

    # ── Stage decision ────────────────────────────────────────────────
    if verdict == "accept" and score >= min_score:
        target_stage = "accepted"
    elif docs_chain_length >= rotation_cap - 1 and score < min_score:
        target_stage = "exhausted"
    else:
        target_stage = "reviewed"

    scores_json = _json.dumps(scores) if scores is not None else None
    comments_per_dim_json = (
        _json.dumps(comments_per_dim) if comments_per_dim is not None else None
    )

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (sn:StandardName {id: $id})
            WHERE sn.claim_token = $token
            SET sn.reviewer_score_docs        = $score,
                sn.reviewer_scores_docs       = $scores_json,
                sn.reviewer_comments_docs     = $comments,
                sn.reviewer_comments_per_dim_docs = $comments_per_dim_json,
                sn.reviewer_verdict_docs      = $verdict,
                sn.reviewer_model_docs        = $model,
                sn.reviewed_docs_at           = datetime(),
                sn.docs_stage                 = $target_stage,
                sn.claim_token                = null,
                sn.claimed_at                 = null
            """,
            id=sn_id,
            token=claim_token,
            score=score,
            scores_json=scores_json,
            comments=comments,
            comments_per_dim_json=comments_per_dim_json,
            verdict=verdict,
            model=model,
            target_stage=target_stage,
        )

    logger.info(
        "persist_reviewed_docs: %s → docs_stage=%s (score=%.3f, chain=%d/%d)",
        sn_id,
        target_stage,
        score,
        docs_chain_length,
        rotation_cap,
    )
    return target_stage


# -- refine_name (StandardName, reviewed + low score + chain < cap) -----------


@retry_on_deadlock()
def claim_refine_name_seed_and_expand(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for name refinement (Option B chain creation).

    Eligibility: ``name_stage = 'reviewed'`` AND
    ``reviewer_score_name < min_score`` AND ``chain_length < rotation_cap``.

    The claim atomically transitions ``name_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with REFINED_FROM chain history
    via :func:`~imas_codex.standard_names.chain_history.name_chain_history`.

    Returns claimed items as dicts with chain_history appended.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    where = (
        "sn.name_stage = 'reviewed'"
        " AND sn.reviewer_score_name IS NOT NULL"
        " AND sn.reviewer_score_name < $min_score"
        " AND coalesce(sn.chain_length, 0) < $rotation_cap"
    )
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_verdict_name AS reviewer_verdict_name"
            ", sn.reviewer_comments_per_dim_name"
            "     AS reviewer_comments_per_dim_name"
            ", sn.chain_length AS chain_length"
            ", sn.name_stage AS name_stage"
            ", sn.source_paths AS source_paths"
            ", sn.tags AS tags"
        ),
        stage_field="name_stage",
        to_stage="refining",
    )

    # Enrich each claimed item with its REFINED_FROM chain history.
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])

    return items


# =============================================================================
# Persist — refine_name (Option B: new node + REFINED_FROM + edge migration)
# =============================================================================


@retry_on_deadlock()
def persist_refined_name(
    *,
    old_name: str,
    new_name: str,
    description: str,
    kind: str = "scalar",
    unit: str | None = None,
    physics_domain: list[str] | None = None,
    tags: list[str] | None = None,
    old_chain_length: int = 0,
    model: str = "unknown",
    grammar_fields: dict[str, str] | None = None,
    reason: str = "",
    escalated: bool = False,
) -> dict[str, str]:
    """Persist a refined StandardName as a NEW node with source-edge migration.

    This is the **Option B** persist: since ``StandardName.id`` IS the name
    string, refining a name produces a new node identity.  In a single
    transaction:

    1. MERGE new StandardName with ``name_stage='drafted'``,
       ``chain_length = old_chain_length + 1``.
    2. Create ``(new)-[:REFINED_FROM]->(old)`` edge.
    3. Mark old SN as ``name_stage='superseded'``, clear its claim.
    4. Migrate ``PRODUCED_NAME`` edges from StandardNameSource to new SN.
    5. Migrate ``HAS_STANDARD_NAME`` edges from IMASNode/FacilitySignal to
       new SN.

    Returns ``{"new_name": <new_id>, "old_name": <old_id>}``.
    """
    import json as _json

    new_chain_length = old_chain_length + 1
    grammar_json = _json.dumps(grammar_fields) if grammar_fields else None

    escalation_set = ""
    if escalated:
        escalation_set = ", new.refine_name_escalated_at = datetime()"

    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                result = list(
                    tx.run(
                        f"""
                        // 1. Create (or match) new SN with new id
                        MERGE (new:StandardName {{id: $new_name}})
                        ON CREATE SET
                          new.name_stage        = 'drafted',
                          new.docs_stage        = 'pending',
                          new.chain_length      = $new_chain_length,
                          new.docs_chain_length = 0,
                          new.description       = $description,
                          new.kind              = $kind,
                          new.unit              = $unit,
                          new.physics_domain    = $physics_domain,
                          new.tags              = $tags,
                          new.model             = $model,
                          new.generated_at      = datetime(),
                          new.grammar_fields    = $grammar_json,
                          new.regen_reason      = $reason
                          {escalation_set}

                        // 2. Link to predecessor
                        WITH new
                        MATCH (old:StandardName {{id: $old_name}})
                        MERGE (new)-[:REFINED_FROM]->(old)

                        // 3. Mark old as superseded, clear claim
                        SET old.name_stage  = 'superseded',
                            old.claim_token = null,
                            old.claimed_at  = null

                        // 4. Migrate PRODUCED_NAME edges
                        WITH new, old
                        OPTIONAL MATCH (src:StandardNameSource)-[r:PRODUCED_NAME]->(old)
                        WITH new, old,
                             collect(src) AS pn_sources,
                             collect(r)   AS pn_rels
                        FOREACH (rel IN pn_rels | DELETE rel)
                        WITH new, old, pn_sources
                        FOREACH (s IN pn_sources | MERGE (s)-[:PRODUCED_NAME]->(new))

                        // 5. Migrate HAS_STANDARD_NAME edges
                        WITH DISTINCT new, old
                        OPTIONAL MATCH (n)-[r2:HAS_STANDARD_NAME]->(old)
                        WITH new, old,
                             collect(n)  AS hsn_nodes,
                             collect(r2) AS hsn_rels
                        FOREACH (rel IN hsn_rels | DELETE rel)
                        WITH new, old, hsn_nodes
                        FOREACH (n IN hsn_nodes | MERGE (n)-[:HAS_STANDARD_NAME]->(new))

                        WITH DISTINCT new, old
                        RETURN new.id AS new_name, old.id AS old_name
                        """,
                        new_name=new_name,
                        old_name=old_name,
                        new_chain_length=new_chain_length,
                        description=description,
                        kind=kind,
                        unit=unit,
                        physics_domain=physics_domain or [],
                        tags=tags or [],
                        model=model,
                        grammar_json=grammar_json,
                        reason=reason,
                    )
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if result:
        row = dict(result[0])
        logger.debug(
            "persist_refined_name: %s → %s (chain_length=%d)",
            old_name,
            new_name,
            new_chain_length,
        )
        return row
    return {"new_name": new_name, "old_name": old_name}


# =============================================================================
# Release helpers — seed-and-expand pools
# =============================================================================
#
# These are called by pool_loop when process() raises an exception so that
# claimed items are unlocked and become eligible for other workers.
#
# Every helper verifies ``claim_token`` (and ``expected_stage`` where
# applicable) in the WHERE clause before clearing claim state.  This
# prevents a late-arriving release from clobbering a fresh re-claim that
# was issued after the orphan sweep cleared the stale token.
#
# Return value: count of nodes actually released.  A return value less than
# len(ids) indicates concurrent intervention (orphan sweep or another
# worker) — callers may ignore this; it is logged at DEBUG.
#


@retry_on_deadlock()
def release_generate_name_claims(
    *,
    source_ids: list[str],
    claim_token: str,
) -> int:
    """Release StandardNameSource claims IFF ``claim_token`` matches.

    Clears ``claimed_at`` and ``claim_token`` so items become eligible for
    re-claim.  Used by the generate_name pool's error-recovery path.

    Returns the count of sources actually released.  A count less than
    ``len(source_ids)`` means the orphan sweep (or another worker) already
    cleared the stale claim — the caller can safely ignore this.
    """
    if not source_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (n:StandardNameSource {id: sid})
            WHERE n.claim_token = $token
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=source_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(source_ids):
        logger.debug(
            "release_generate_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(source_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_name_failed_claims(
    *,
    source_ids: list[str],
    claim_token: str,
) -> int:
    """Release StandardNameSource claims on worker failure IFF token matches.

    Identical to :func:`release_generate_name_claims`; provided as the
    symmetric "failed" variant so callers can be explicit about intent.

    Returns the count of sources actually released.
    """
    return release_generate_name_claims(
        source_ids=source_ids,
        claim_token=claim_token,
    )


@retry_on_deadlock()
def release_enrich_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName enrichment claims IFF token (and stage) match.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token that was set at claim time; nodes with a different token are
        left untouched.
    expected_stage:
        If provided, also verify ``name_stage = $expected_stage`` before
        clearing.  Pass ``None`` (default) to skip stage verification.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_enrich_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName enrichment claims on worker failure.

    Clears claim state and optionally reverts ``name_stage`` to *to_stage*
    when the worker processing failed and the item should be retried.

    Parameters
    ----------
    sn_ids:
        StandardName node ids to release.
    claim_token:
        Token set at claim time.
    from_stage:
        If provided, verify ``name_stage = $from_stage`` before acting.
    to_stage:
        If provided (and *from_stage* matches), revert ``name_stage`` to
        this value so the item is eligible for a fresh claim.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_enrich_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName review-names claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_names_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName review-names claims on worker failure.

    Clears claim state and optionally reverts ``name_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_names_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName review-docs claims IFF token (and stage) match.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_review_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    from_stage: str | None = None,
    to_stage: str | None = None,
) -> int:
    """Release StandardName review-docs claims on worker failure.

    Clears claim state and optionally reverts ``name_stage``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_where = "AND n.name_stage = $from_stage" if from_stage is not None else ""
    stage_set = "n.name_stage = $to_stage," if to_stage is not None else ""
    params: dict[str, Any] = {"ids": sn_ids, "token": claim_token}
    if from_stage is not None:
        params["from_stage"] = from_stage
    if to_stage is not None:
        params["to_stage"] = to_stage
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_where}
            SET {stage_set}
                n.claimed_at = null,
                n.claim_token = null
            RETURN count(n) AS released
            """,
            **params,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_review_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_name_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
    expected_stage: str | None = None,
) -> int:
    """Release StandardName refine-name claims IFF token (and stage) match.

    Clears ``claimed_at`` and ``claim_token`` and reverts
    ``name_stage`` from ``'refining'`` back to ``'reviewed'`` when
    the node is still in the refining stage.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    stage_clause = (
        "AND n.name_stage = $expected_stage" if expected_stage is not None else ""
    )
    extra: dict[str, Any] = (
        {"expected_stage": expected_stage} if expected_stage is not None else {}
    )
    with GraphClient() as gc:
        result = gc.query(
            f"""
            UNWIND $ids AS sid
            MATCH (n:StandardName {{id: sid}})
            WHERE n.claim_token = $token
              {stage_clause}
            SET n.claimed_at = null,
                n.claim_token = null,
                n.name_stage = CASE
                    WHEN n.name_stage = 'refining' THEN 'reviewed'
                    ELSE n.name_stage
                END
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
            **extra,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_name_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_name_failed_claims(
    *,
    sn_ids: list[str],
    token: str,
) -> int:
    """Release refine-name claims after LLM or processing failure.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND name_stage = 'refining'``.  This prevents
    late-release from clobbering an SN that was already swept by orphan
    recovery or successfully persisted.

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.name_stage = 'refining'
            SET sn.name_stage = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=token,
        )
    return result[0]["released"] if result else 0


# =============================================================================
# generate_docs — claim / persist / release
# =============================================================================
# Stage gate: name_stage='accepted' AND docs_stage='pending'
# The claim does NOT transition docs_stage (persist_generated_docs does that).


@retry_on_deadlock()
def claim_generate_docs_seed_and_expand(
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes ready for generate_docs.

    Eligibility: ``name_stage = 'accepted'`` AND ``docs_stage = 'pending'``
    AND ``claimed_at IS NULL`` (or stale).

    The claim does NOT transition ``docs_stage`` — that happens in
    :func:`persist_generated_docs` so that a failed worker leaves the node
    cleanly at ``'pending'``.

    Each claimed item is enriched with REFINED_FROM chain history via
    :func:`~imas_codex.standard_names.chain_history.name_chain_history`
    and the name-review feedback fields so the LLM understands why this
    name was accepted.

    Returns claimed items as dicts.
    """
    from imas_codex.standard_names.chain_history import name_chain_history

    where = "sn.name_stage = 'accepted' AND sn.docs_stage = 'pending'"

    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            # description, kind, physics_domain already in base readback —
            # listing them again would raise Neo4j 42N38 (duplicate return
            # item name).  Only add fields not present in _claim_sn_atomic's
            # fixed RETURN list.
            ", sn.tags AS tags"
            ", sn.reviewer_score_name AS reviewer_score_name"
            ", sn.reviewer_comments_name AS reviewer_comments_name"
            ", sn.reviewer_verdict_name AS reviewer_verdict_name"
            ", sn.chain_length AS chain_length"
            ", sn.docs_stage AS docs_stage"
            ", sn.name_stage AS name_stage"
        ),
        # stage_field=None → claim only, no stage transition
    )

    # Enrich each claimed item with REFINED_FROM chain history.
    for item in items:
        item["chain_history"] = name_chain_history(item["id"])

    logger.debug(
        "claim_generate_docs_seed_and_expand: claimed %d",
        len(items),
    )
    return items


@retry_on_deadlock()
def persist_generated_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
) -> str:
    """Persist generate_docs results and transition ``docs_stage`` to ``'drafted'``.

    Single-transaction write:

    1. Verify ``claim_token`` matches the stored token.
    2. SET ``description``, ``documentation``, ``docs_stage = 'drafted'``,
       ``docs_chain_length = 0``, ``docs_model``, ``docs_generated_at``,
       clear ``claim_token`` and ``claimed_at``.

    Parameters
    ----------
    sn_id:
        StandardName node id.
    claim_token:
        Token written at claim time — verified before any write.
    description:
        Short description (1–3 sentences) from the LLM.
    documentation:
        Rich markdown documentation from the LLM.
    model:
        LLM model identifier used for generation.

    Returns the new ``docs_stage`` value (``'drafted'``).
    Raises :exc:`ValueError` if token verification fails (no matching node).
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            WHERE sn.claim_token = $token
            SET sn.description      = $description,
                sn.documentation    = $documentation,
                sn.docs_stage       = 'drafted',
                sn.docs_chain_length = 0,
                sn.docs_model       = $model,
                sn.docs_generated_at = datetime(),
                sn.claim_token      = null,
                sn.claimed_at       = null
            RETURN sn.docs_stage AS docs_stage
            """,
            sn_id=sn_id,
            token=claim_token,
            description=description,
            documentation=documentation,
            model=model,
        )
    if not result:
        raise ValueError(
            f"persist_generated_docs: token mismatch or node not found for {sn_id!r}"
        )
    new_stage: str = result[0]["docs_stage"]
    logger.debug("persist_generated_docs: %s → docs_stage=%s", sn_id, new_stage)
    return new_stage


@retry_on_deadlock()
def release_generate_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release generate_docs claims IFF token matches.

    Called after successful persist to clear any nodes whose claim state
    was not already cleared inside :func:`persist_generated_docs`
    (e.g., skipped items in a multi-item batch).

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS sid
            MATCH (n:StandardName {id: sid})
            WHERE n.claim_token = $token
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """,
            ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_generate_docs_claims: %d/%d released (token=%s) — "
            "remainder already cleared or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_generate_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release generate_docs claims after LLM or processing failure.

    Token-verified: only clears ``claim_token`` and ``claimed_at`` on nodes
    where ``claim_token = $token``.  Does NOT change ``docs_stage`` (it
    stays at ``'pending'`` since the claim never transitioned it).

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
            SET sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_generate_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


# =============================================================================
# refine_docs — claim / persist / release (DocsRevision snapshot architecture)
# =============================================================================
# Stage gate: docs_stage='reviewed' AND reviewer_score_docs < min_score
#             AND docs_chain_length < rotation_cap AND verdict != 'accept'
# Fundamentally different from refine_name: docs refine is IN-PLACE on the
# existing SN node.  The OLD docs are snapshotted into a DocsRevision node
# linked via DOCS_REVISION_OF before the SN is updated.


@retry_on_deadlock()
def claim_refine_docs_seed_and_expand(
    min_score: float = DEFAULT_MIN_SCORE,
    rotation_cap: int = DEFAULT_REFINE_ROTATIONS,
    batch_size: int = _DEFAULT_SEED_EXPAND_BATCH,
    timeout_seconds: int = _CLAIM_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Claim StandardName nodes for docs refinement.

    Eligibility: ``docs_stage = 'reviewed'`` AND
    ``reviewer_score_docs < min_score`` AND
    ``docs_chain_length < rotation_cap`` AND
    ``reviewer_verdict_docs != 'accept'``.

    The claim atomically transitions ``docs_stage`` from ``'reviewed'``
    to ``'refining'`` via :func:`_claim_sn_atomic`.

    After claiming, each item is enriched with DOCS_REVISION_OF chain
    history via :func:`~imas_codex.standard_names.chain_history.docs_chain_history`.

    Returns claimed items as dicts with docs_chain_history appended.
    """
    from imas_codex.standard_names.chain_history import docs_chain_history

    where = (
        "sn.docs_stage = 'reviewed'"
        " AND sn.reviewer_score_docs IS NOT NULL"
        " AND sn.reviewer_score_docs < $min_score"
        " AND coalesce(sn.docs_chain_length, 0) < $rotation_cap"
        " AND coalesce(sn.reviewer_verdict_docs, 'revise') <> 'accept'"
    )
    items = _claim_sn_atomic(
        eligibility_where=where,
        query_params={"min_score": min_score, "rotation_cap": rotation_cap},
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        extra_return_fields=(
            # description, documentation, kind, physics_domain already in base
            # readback — listing them again raises Neo4j 42N38 (duplicate
            # return item name).  Only add fields not present in
            # _claim_sn_atomic's fixed RETURN list.
            ", sn.tags AS tags"
            ", sn.docs_stage AS docs_stage"
            ", sn.docs_chain_length AS docs_chain_length"
            ", sn.docs_model AS docs_model"
            ", sn.docs_generated_at AS docs_generated_at"
            ", sn.reviewer_score_docs AS reviewer_score_docs"
            ", sn.reviewer_comments_per_dim_docs"
            "     AS reviewer_comments_per_dim_docs"
            ", sn.reviewer_comments_docs AS reviewer_comments_docs"
            ", sn.reviewer_verdict_docs AS reviewer_verdict_docs"
        ),
        stage_field="docs_stage",
        to_stage="refining",
    )

    # Enrich each claimed item with its DOCS_REVISION_OF chain history.
    for item in items:
        item["docs_chain_history"] = docs_chain_history(item["id"], limit=5)

    logger.debug(
        "claim_refine_docs_seed_and_expand: claimed %d",
        len(items),
    )
    return items


# =============================================================================
# Persist — refine_docs (snapshot current docs → DocsRevision, update SN)
# =============================================================================


@retry_on_deadlock()
def persist_refined_docs(
    *,
    sn_id: str,
    claim_token: str,
    description: str,
    documentation: str,
    model: str,
    current_description: str,
    current_documentation: str,
    current_model: str | None = None,
    current_generated_at: str | None = None,
    reviewer_score_to_snapshot: float | None = None,
    reviewer_comments_to_snapshot: str | None = None,
    reviewer_comments_per_dim_to_snapshot: str | None = None,
    reviewer_verdict_to_snapshot: str | None = None,
) -> dict[str, Any]:
    """Persist a refined docs revision with DocsRevision snapshot.

    Single-transaction:

    1. Verify ``claim_token`` + ``docs_stage = 'refining'``.
    2. CREATE ``DocsRevision`` snapshot of CURRENT state (before this refine):
       ``id = "{sn_id}#rev-{docs_chain_length}"`` (deterministic key).
    3. CREATE ``(sn)-[:DOCS_REVISION_OF]->(rev)`` edge.
    4. SET new description/documentation on SN, advance chain, clear claim.
    5. Clear ``reviewer_*_docs`` fields on the SN (new docs need fresh review).

    Returns ``{"docs_chain_length": <new>, "revision_id": <id>}``.
    Returns ``{"docs_chain_length": -1, "revision_id": ""}`` on token/stage
    mismatch (no-op).
    """
    with GraphClient() as gc:
        with gc.session() as session:
            tx = session.begin_transaction()
            try:
                result = list(
                    tx.run(
                        """
                        // 1. Match + verify
                        MATCH (sn:StandardName {id: $sn_id})
                        WHERE sn.claim_token = $token
                          AND sn.docs_stage = 'refining'
                        WITH sn, coalesce(sn.docs_chain_length, 0) AS cur_chain

                        // 2. Create DocsRevision snapshot (deterministic id)
                        WITH sn, cur_chain,
                             $sn_id + '#rev-' + toString(cur_chain) AS rev_id
                        MERGE (rev:DocsRevision {id: rev_id})
                        ON CREATE SET
                          rev.sn_id                          = $sn_id,
                          rev.revision_number                = cur_chain,
                          rev.description                    = $cur_desc,
                          rev.documentation                  = $cur_doc,
                          rev.model                          = $cur_model,
                          rev.generated_at                   = $cur_gen_at,
                          rev.reviewer_score_docs            = $snap_score,
                          rev.reviewer_comments_docs         = $snap_comments,
                          rev.reviewer_comments_per_dim_docs = $snap_comments_dim,
                          rev.reviewer_verdict_docs          = $snap_verdict,
                          rev.created_at                     = datetime()

                        // 3. Link SN → revision
                        WITH sn, rev, cur_chain
                        MERGE (sn)-[:DOCS_REVISION_OF]->(rev)

                        // 4. Update SN with new docs + advance chain
                        WITH sn, rev, cur_chain
                        SET sn.description       = $new_desc,
                            sn.documentation     = $new_doc,
                            sn.docs_stage        = 'drafted',
                            sn.docs_chain_length = cur_chain + 1,
                            sn.docs_model        = $model,
                            sn.docs_generated_at = datetime(),
                            sn.claim_token       = null,
                            sn.claimed_at        = null,
                            // 5. Clear reviewer_*_docs — new docs need fresh review
                            sn.reviewer_score_docs            = null,
                            sn.reviewer_scores_docs           = null,
                            sn.reviewer_comments_per_dim_docs = null,
                            sn.reviewer_comments_docs         = null,
                            sn.reviewer_verdict_docs          = null,
                            sn.reviewer_model_docs            = null,
                            sn.reviewed_docs_at               = null

                        RETURN cur_chain + 1 AS docs_chain_length,
                               rev.id        AS revision_id
                        """,
                        sn_id=sn_id,
                        token=claim_token,
                        cur_desc=current_description or "",
                        cur_doc=current_documentation or "",
                        cur_model=current_model,
                        cur_gen_at=current_generated_at,
                        snap_score=reviewer_score_to_snapshot,
                        snap_comments=reviewer_comments_to_snapshot,
                        snap_comments_dim=reviewer_comments_per_dim_to_snapshot,
                        snap_verdict=reviewer_verdict_to_snapshot,
                        new_desc=description,
                        new_doc=documentation,
                        model=model,
                    )
                )
                tx.commit()
            except BaseException:
                if tx.closed is False:
                    tx.close()
                raise

    if result:
        row = dict(result[0])
        logger.debug(
            "persist_refined_docs: %s (chain_length=%d, rev=%s)",
            sn_id,
            row["docs_chain_length"],
            row["revision_id"],
        )
        return row
    logger.debug(
        "persist_refined_docs: no-op for %s (token/stage mismatch)",
        sn_id,
    )
    return {"docs_chain_length": -1, "revision_id": ""}


@retry_on_deadlock()
def release_refine_docs_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release refine-docs claims, reverting ``docs_stage`` to ``'reviewed'``.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.

    Returns the count of SNs actually released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.docs_stage  = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_docs_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_refine_docs_failed_claims(
    *,
    sn_ids: list[str],
    claim_token: str,
) -> int:
    """Release refine-docs claims after LLM or processing failure.

    Token-and-stage verified: only reverts nodes where
    ``claim_token = $token AND docs_stage = 'refining'``.  Prevents
    late-release from clobbering an SN already swept by orphan recovery.

    Returns the count of nodes released.
    """
    if not sn_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $sn_ids AS sid
            MATCH (sn:StandardName {id: sid})
            WHERE sn.claim_token = $token
              AND sn.docs_stage = 'refining'
            SET sn.docs_stage  = 'reviewed',
                sn.claim_token = null,
                sn.claimed_at  = null
            RETURN count(sn) AS released
            """,
            sn_ids=sn_ids,
            token=claim_token,
        )
    released: int = result[0]["released"] if result else 0
    if released < len(sn_ids):
        logger.debug(
            "release_refine_docs_failed_claims: %d/%d released (token=%s) — "
            "remainder already swept or re-claimed",
            released,
            len(sn_ids),
            claim_token[:8],
        )
    return released


@retry_on_deadlock()
def release_all_orphan_claims() -> dict[str, int]:
    """Release all claimed-but-unreleased StandardName and StandardNameSource nodes.

    Called from the ``run_sn_pools`` finally block after clean shutdown so
    that any batch still marked as claimed at process-exit is unlocked.
    Per-batch release already happens inside each process() try/finally, but
    batches in flight at the 60s grace-period timeout leave ``claimed_at``
    set permanently.  This sweep clears them unconditionally — it is safe
    because the run is over and no other process can be competing for the
    same tokens.

    Returns a dict with ``"sn"`` and ``"sns"`` keys showing the counts
    of released nodes.
    """
    with GraphClient() as gc:
        sn_result = gc.query(
            """
            MATCH (n:StandardName)
            WHERE n.claimed_at IS NOT NULL
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """
        )
        sns_result = gc.query(
            """
            MATCH (n:StandardNameSource)
            WHERE n.claimed_at IS NOT NULL
            SET n.claimed_at = null, n.claim_token = null
            RETURN count(n) AS released
            """
        )
    sn_count = sn_result[0]["released"] if sn_result else 0
    sns_count = sns_result[0]["released"] if sns_result else 0
    return {"sn": sn_count, "sns": sns_count}


# ═══════════════════════════════════════════════════════════════════════
# Pool pending counts — single round-trip query for all 6 pools
# ═══════════════════════════════════════════════════════════════════════


def pool_pending_counts(
    *,
    min_score: float = 0.75,
    rotation_cap: int = 3,
) -> dict[str, int]:
    """Return pending-work counts for all six worker pools in one query.

    The six predicates mirror the eligibility criteria in the
    corresponding ``claim_*_seed_and_expand`` functions but do NOT
    filter on ``claimed_at`` — the counts reflect total eligible work,
    including items currently being processed.  This is correct for
    both throttle decisions (total queue depth matters) and display
    (users want to see total pending, not just unclaimed).

    Returns a dict keyed by pool name::

        {
            "generate_name": int,
            "review_name": int,
            "refine_name": int,
            "generate_docs": int,
            "review_docs": int,
            "refine_docs": int,
        }
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
    CALL { MATCH (s:StandardNameSource {status: 'extracted'})
           RETURN count(s) AS generate_name }
    CALL { MATCH (sn:StandardName {name_stage: 'drafted'})
           RETURN count(sn) AS review_name }
    CALL { MATCH (sn:StandardName {name_stage: 'reviewed'})
           WHERE sn.reviewer_score_name < $min_score
             AND coalesce(sn.chain_length, 0) < $rotation_cap
           RETURN count(sn) AS refine_name }
    CALL { MATCH (sn:StandardName {name_stage: 'accepted', docs_stage: 'pending'})
           RETURN count(sn) AS generate_docs }
    CALL { MATCH (sn:StandardName {docs_stage: 'drafted'})
           RETURN count(sn) AS review_docs }
    CALL { MATCH (sn:StandardName {docs_stage: 'reviewed'})
           WHERE sn.reviewer_score_docs < $min_score
             AND coalesce(sn.docs_chain_length, 0) < $rotation_cap
           RETURN count(sn) AS refine_docs }
    RETURN generate_name, review_name, refine_name,
           generate_docs, review_docs, refine_docs
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, min_score=min_score, rotation_cap=rotation_cap))

    if not rows:
        return {
            "generate_name": 0,
            "review_name": 0,
            "refine_name": 0,
            "generate_docs": 0,
            "review_docs": 0,
            "refine_docs": 0,
        }
    r = rows[0]
    return {
        "generate_name": int(r.get("generate_name", 0)),
        "review_name": int(r.get("review_name", 0)),
        "refine_name": int(r.get("refine_name", 0)),
        "generate_docs": int(r.get("generate_docs", 0)),
        "review_docs": int(r.get("review_docs", 0)),
        "refine_docs": int(r.get("refine_docs", 0)),
    }
