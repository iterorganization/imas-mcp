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
from typing import Any

from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


def _ensure_json(value: Any) -> str | None:
    """Ensure a value is a JSON string, not a raw dict/list (Neo4j rejects Maps)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value)


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
            "n.node_type = 'dynamic'",
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
            tags, links, and all linked DD paths. Used by --paths mode.

    Returns:
        Dict mapping source entity ID to dict with keys:
        name, description, kind, review_status (and more if rich=True).
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
                   sn.review_status AS review_status
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            # If multiple names exist for same source, prefer accepted
            if sid not in mapping or r.get("review_status") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "kind": r.get("kind"),
                    "review_status": r.get("review_status"),
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
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.review_status AS review_status,
                   sn.reviewer_score AS reviewer_score,
                   sn.review_tier AS review_tier,
                   sn.validation_issues AS validation_issues,
                   u.id AS unit,
                   collect(DISTINCT other_src.id) AS linked_dd_paths
        """)
        mapping: dict[str, dict] = {}
        for r in results:
            sid = r["source_id"]
            if sid not in mapping or r.get("review_status") == "accepted":
                mapping[sid] = {
                    "name": r["name"],
                    "description": r.get("description"),
                    "documentation": r.get("documentation"),
                    "kind": r.get("kind"),
                    "tags": r.get("tags"),
                    "links": r.get("links"),
                    "review_status": r.get("review_status"),
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


def write_standard_names(names: list[dict[str, Any]]) -> int:
    """MERGE StandardName nodes with HAS_STANDARD_NAME relationships.

    Relationship direction: entity → concept
      (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
      (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)

    Each dict in *names* must have at least:
      - ``id``: the composed standard name string
      - ``source_type``: "dd" or "signal"
      - ``source_id``: the originating path / signal ID

    Optional fields: ``physical_base``, ``subject``, ``component``,
    ``coordinate``, ``position``, ``process``, ``unit``, ``description``,
    ``documentation``, ``kind``, ``tags``, ``links``, ``imas_paths``,
    ``validity_domain``, ``constraints``, ``model``, ``review_status``,
    ``generated_at``, ``confidence``, ``reviewer_model``, ``reviewer_score``,
    ``reviewer_scores``, ``reviewer_comments``, ``reviewed_at``,
    ``review_tier``, ``vocab_gap_detail``, ``validation_issues``,
    ``validation_layer_summary``, ``cocos_transformation_type``, ``dd_version``,
    ``review_input_hash``.

    Performs conflict detection on ``unit``: if a StandardName already exists
    with a different unit value, that entry is skipped (not written)
    and a warning is logged.

    Returns the number of nodes written.
    """
    if not names:
        return 0

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
            SET sn.source_type = coalesce(b.source_type, sn.source_type),
                sn.physical_base = coalesce(b.physical_base, sn.physical_base),
                sn.subject = coalesce(b.subject, sn.subject),
                sn.component = coalesce(b.component, sn.component),
                sn.coordinate = coalesce(b.coordinate, sn.coordinate),
                sn.position = coalesce(b.position, sn.position),
                sn.process = coalesce(b.process, sn.process),
                sn.description = coalesce(b.description, sn.description),
                sn.documentation = coalesce(b.documentation, sn.documentation),
                sn.kind = coalesce(b.kind, sn.kind),
                sn.tags = coalesce(b.tags, sn.tags),
                sn.links = coalesce(b.links, sn.links),
                sn.imas_paths = coalesce(b.imas_paths, sn.imas_paths),
                sn.validity_domain = coalesce(b.validity_domain, sn.validity_domain),
                sn.constraints = coalesce(b.constraints, sn.constraints),
                sn.unit = coalesce(b.unit, sn.unit),
                sn.physics_domain = coalesce(b.physics_domain, sn.physics_domain),
                sn.cocos_transformation_type = coalesce(b.cocos_transformation_type, sn.cocos_transformation_type),
                sn.cocos = coalesce(b.cocos, sn.cocos),
                sn.dd_version = coalesce(b.dd_version, sn.dd_version),
                sn.model = coalesce(b.model, sn.model),
                sn.review_status = coalesce(b.review_status, sn.review_status),
                sn.generated_at = coalesce(b.generated_at, sn.generated_at),
                sn.confidence = coalesce(b.confidence, sn.confidence),
                sn.reviewer_model = coalesce(b.reviewer_model, sn.reviewer_model),
                sn.reviewer_score = coalesce(b.reviewer_score, sn.reviewer_score),
                sn.reviewer_scores = coalesce(b.reviewer_scores, sn.reviewer_scores),
                sn.reviewer_comments = coalesce(b.reviewer_comments, sn.reviewer_comments),
                sn.reviewed_at = coalesce(b.reviewed_at, sn.reviewed_at),
                sn.review_tier = coalesce(b.review_tier, sn.review_tier),
                sn.vocab_gap_detail = coalesce(b.vocab_gap_detail, sn.vocab_gap_detail),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
                sn.validation_layer_summary = coalesce(b.validation_layer_summary, sn.validation_layer_summary),
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_type": n.get("source_type") or None,
                    "physical_base": n.get("physical_base"),
                    "subject": n.get("subject"),
                    "component": n.get("component"),
                    "coordinate": n.get("coordinate"),
                    "position": n.get("position"),
                    "process": n.get("process"),
                    "description": n.get("description"),
                    "documentation": n.get("documentation"),
                    "kind": n.get("kind"),
                    "tags": n.get("tags") or None,
                    "links": n.get("links") or None,
                    "imas_paths": n.get("imas_paths") or None,
                    "validity_domain": n.get("validity_domain"),
                    "constraints": n.get("constraints") or None,
                    "unit": n.get("unit"),
                    "physics_domain": n.get("physics_domain"),
                    "cocos_transformation_type": n.get("cocos_transformation_type"),
                    "cocos": n.get("cocos"),
                    "dd_version": n.get("dd_version"),
                    "model": n.get("model"),
                    "review_status": n.get("review_status"),
                    "generated_at": n.get("generated_at"),
                    "confidence": n.get("confidence"),
                    "reviewer_model": n.get("reviewer_model"),
                    "reviewer_score": n.get("reviewer_score"),
                    "reviewer_scores": _ensure_json(n.get("reviewer_scores")),
                    "reviewer_comments": n.get("reviewer_comments"),
                    "reviewed_at": n.get("reviewed_at"),
                    "review_tier": n.get("review_tier"),
                    "vocab_gap_detail": _ensure_json(n.get("vocab_gap_detail")),
                    "validation_issues": n.get("validation_issues") or None,
                    "validation_layer_summary": _ensure_json(
                        n.get("validation_layer_summary")
                    ),
                    "link_status": _compute_link_status(n.get("links")),
                    "review_input_hash": n.get("review_input_hash"),
                }
                for n in names
            ],
        )

        # Create HAS_STANDARD_NAME relationships: entity → concept
        dd_names = [n for n in names if n.get("source_type") == "dd"]
        signal_names = [n for n in names if n.get("source_type") == "signal"]

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

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    return written


# =============================================================================
# Read helpers — publish (validated standard names)
# =============================================================================


def get_validated_standard_names(
    ids_filter: str | None = None,
    confidence_min: float = 0.0,
    review_status: str = "drafted",
) -> list[dict[str, Any]]:
    """Read validated StandardName nodes and their provenance.

    Queries StandardName nodes with the given ``review_status``, joining
    through ``HAS_STANDARD_NAME`` to find source entities and their parent IDS,
    and through ``HAS_UNIT`` to find the unit node.  Uses ``collect()``
    to avoid row duplication when a name has multiple sources (takes the first).

    Parameters
    ----------
    ids_filter:
        Restrict to names derived from a specific IDS (matched via
        ``IMASNode -[:HAS_STANDARD_NAME]-> StandardName`` and
        ``IMASNode -[:IN_IDS]-> IDS``).
    confidence_min:
        Minimum confidence threshold.  Nodes without a ``confidence``
        property are treated as 1.0 (grammar-validated).
    review_status:
        Filter by ``review_status`` property (default ``"drafted"``).

    Returns
    -------
    list of dicts with keys: name, description, documentation, kind,
    unit, tags, links, ids_paths, constraints, validity_domain,
    confidence, model, source, source_path, ids_name, physical_base,
    subject, component, coordinate, position, process, source_ids_names.
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {
            "confidence_min": confidence_min,
            "review_status": review_status,
        }

        # Collect source info — use HAS_STANDARD_NAME (entity → concept)
        cypher = """
            MATCH (sn:StandardName)
            WHERE sn.review_status = $review_status
            AND coalesce(sn.confidence, 1.0) >= $confidence_min
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
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.ids_paths AS ids_paths,
                   sn.constraints AS constraints,
                   sn.validity_domain AS validity_domain,
                   coalesce(sn.confidence, 1.0) AS confidence,
                   sn.model AS model,
                   sn.source_type AS source,
                   coalesce(sn.source_path, first_source) AS source_path,
                   first_ids AS ids_name,
                   sn.physical_base AS physical_base,
                   sn.subject AS subject,
                   sn.component AS component,
                   sn.coordinate AS coordinate,
                   sn.position AS position,
                   sn.process AS process,
                   all_ids AS source_ids_names,
                   sn.cocos_transformation_type AS cocos_transformation_type,
                   sn.cocos AS cocos,
                   sn.dd_version AS dd_version
            ORDER BY sn.id
        """

        results = gc.query(cypher, **params)
        logger.info(
            "Read %d validated standard names (ids_filter=%s, confidence_min=%.2f, review_status=%s)",
            len(results),
            ids_filter,
            confidence_min,
            review_status,
        )
        return list(results)


def reset_standard_names(
    *,
    from_status: str = "drafted",
    to_status: str | None = None,
    source_filter: str | None = None,
    ids_filter: str | None = None,
    dry_run: bool = False,
) -> int:
    """Reset StandardName nodes to allow re-processing.

    Clears transient fields (embedding, embedded_at, model, generated_at,
    confidence) and removes HAS_STANDARD_NAME, HAS_UNIT, and
    HAS_COCOS relationships for matching nodes.

    Parameters
    ----------
    from_status:
        Only reset nodes with this ``review_status`` (default ``"drafted"``).
    to_status:
        Target ``review_status`` after reset.  ``None`` (default) clears fields
        only without changing the status.
    source_filter:
        Restrict to nodes with ``source`` equal to ``"dd"`` or ``"signals"``.
    ids_filter:
        Restrict to nodes whose HAS_STANDARD_NAME source path starts with this
        IDS name (matched via ``IMASNode -[:HAS_STANDARD_NAME]-> sn``).
    dry_run:
        Return the count of matching nodes without modifying anything.

    Returns
    -------
    Number of nodes reset (or that would be reset in dry-run mode).
    """
    with GraphClient() as gc:
        params: dict[str, Any] = {"from_status": from_status}
        where_clauses = ["sn.review_status = $from_status"]

        if source_filter:
            where_clauses.append("sn.source_type = $source_filter")
            params["source_filter"] = source_filter

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
                "sn.generated_at = null, sn.confidence = null, "
                "sn.cocos_transformation_type = null, sn.cocos = null, sn.dd_version = null, "
                "sn.review_status = $to_status"
            )
            reset_params["to_status"] = to_status
        else:
            set_clause = (
                "sn.embedding = null, sn.embedded_at = null, sn.model = null, "
                "sn.generated_at = null, sn.confidence = null, "
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
) -> int:
    """Delete StandardName nodes and their relationships.

    Safety model (relationship-first):

    1. If ``ids_filter`` or ``source_filter`` is set, delete matching
       ``HAS_STANDARD_NAME`` relationships first.
    2. Then delete ``StandardName`` nodes that have zero remaining
       ``HAS_STANDARD_NAME`` edges.

    By default only nodes with ``review_status = 'drafted'`` are deleted.
    Accepted names require ``include_accepted=True``.

    Parameters
    ----------
    status_filter:
        List of ``review_status`` values to delete (default ``["drafted"]``).
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

    Returns
    -------
    Number of nodes deleted (or that would be deleted in dry-run mode).
    """
    if status_filter is None:
        status_filter = ["drafted"]

    effective_statuses = list(status_filter)
    if include_accepted and "accepted" not in effective_statuses:
        effective_statuses.append("accepted")
    elif not include_accepted and "accepted" in effective_statuses:
        effective_statuses.remove("accepted")

    with GraphClient() as gc:
        params: dict[str, Any] = {"statuses": effective_statuses}
        sn_where_clauses = ["sn.review_status IN $statuses"]

        if source_filter:
            sn_where_clauses.append("sn.source_type = $source_filter")
            params["source_filter"] = source_filter

        sn_where = " AND ".join(sn_where_clauses)

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

        if ids_filter:
            # Step 1: remove HAS_STANDARD_NAME relationships for matching scope
            gc.query(
                f"""
                MATCH (src:IMASNode)-[r:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE {sn_where}
                AND src.id STARTS WITH $ids_prefix
                DELETE r
                """,
                **params,
            )
            # Step 2: delete nodes that are now orphans (no remaining edges)
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
            # No scoping — detach-delete all matching nodes (removes all rels)
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {sn_where}
                DETACH DELETE sn
                """,
                **params,
            )

    logger.info("Deleted %d StandardName nodes", count)
    return count


def update_review_status(names: list[str], status: str = "published") -> int:
    """Update review_status for a batch of StandardName nodes.

    Parameters
    ----------
    names:
        List of StandardName node IDs (``sn.id``) to update.
    status:
        New ``review_status`` value (default ``"published"``).

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
            SET sn.review_status = $status
            RETURN count(sn) AS updated
            """,
            names=names,
            status=status,
        )
        count = result[0]["updated"] if result else 0
        logger.info("Updated review_status to '%s' for %d names", status, count)
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


def resolve_links_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve dd: links to name: links for a batch of names.

    For each ``dd:path`` link, checks if a StandardName exists that was
    generated from that path. If found, replaces with ``name:sn_id``.
    """
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
