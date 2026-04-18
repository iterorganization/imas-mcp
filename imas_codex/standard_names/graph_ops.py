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
from typing import Any

from imas_codex.discovery.base.claims import retry_on_deadlock
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
      - ``source_types``: ["dd"] or ["signals"] etc.
      - ``source_id``: the originating path / signal ID

    Optional fields: ``physical_base``, ``subject``, ``component``,
    ``coordinate``, ``position``, ``process``, ``unit``, ``description``,
    ``documentation``, ``kind``, ``tags``, ``links``, ``source_paths``,
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
            SET sn.source_types = coalesce(b.source_types, sn.source_types),
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
                sn.source_paths = coalesce(b.source_paths, sn.source_paths),
                sn.validity_domain = coalesce(b.validity_domain, sn.validity_domain),
                sn.constraints = coalesce(b.constraints, sn.constraints),
                sn.unit = coalesce(b.unit, sn.unit),
                sn.physics_domain = coalesce(nullIf(b.physics_domain, ''), sn.physics_domain),
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
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.link_status = coalesce(b.link_status, sn.link_status),
                sn.review_input_hash = b.review_input_hash,
                sn.embedding = coalesce(b.embedding, sn.embedding),
                sn.embedded_at = coalesce(b.embedded_at, sn.embedded_at),
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_types": n.get("source_types") or None,
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
                    "source_paths": n.get("source_paths") or None,
                    "validity_domain": n.get("validity_domain"),
                    "constraints": n.get("constraints") or None,
                    "unit": n.get("unit"),
                    "physics_domain": n.get("physics_domain") or None,
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
                    "validation_status": n.get("validation_status"),
                    "link_status": _compute_link_status(n.get("links")),
                    "review_input_hash": n.get("review_input_hash"),
                    "embedding": n.get("embedding"),
                    "embedded_at": n.get("embedded_at"),
                }
                for n in names
            ],
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
        _write_segment_edges(gc, [n["id"] for n in names])

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    return written


def _write_segment_edges(gc: GraphClient, name_ids: list[str]) -> None:
    """Write HAS_SEGMENT edges from StandardName nodes to GrammarToken nodes.

    For each name, parses via ISN grammar, resolves segment tokens, and
    MERGEs ``(sn:StandardName)-[:HAS_SEGMENT {position, segment}]->(t:GrammarToken)``.

    Idempotent: existing HAS_SEGMENT edges are deleted before re-writing.
    Parse failures are logged and skipped — the SN node remains intact.
    Token-miss (vocabulary drift) is detected and warned.

    Args:
        gc: Open GraphClient session (caller manages the context manager).
        name_ids: List of StandardName.id values to process.
    """
    try:
        from imas_standard_names import __version__ as isn_version
        from imas_standard_names.grammar import parse_standard_name
        from imas_standard_names.graph.spec import segment_edge_specs
    except ImportError:
        logger.debug("ISN grammar not available — skipping HAS_SEGMENT edges")
        return

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

        # Idempotent: delete old HAS_SEGMENT edges before re-writing
        gc.query(
            """
            MATCH (sn:StandardName {id: $sn_id})-[r:HAS_SEGMENT]->(:GrammarToken)
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
                    version: $isn_version
                })
                WITH sn, edge, t
                FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (sn)-[r:HAS_SEGMENT]->(t)
                    SET r.position = edge.position,
                        r.segment = edge.segment
                )
                RETURN edge.token AS token,
                       edge.segment AS segment,
                       t IS NOT NULL AS matched
                """,
                sn_id=sn_id,
                edges=edges_param,
                isn_version=isn_version,
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
                "Token-miss for '%s': %s (ISN %s) — vocab gap",
                sn_id,
                ", ".join(missing),
                isn_version,
            )


# =============================================================================
# Immediate-persist helpers — graph-state-machine compose
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


def persist_composed_batch(
    candidates: list[dict[str, Any]],
    *,
    compose_model: str,
    dd_version: str | None = None,
    cocos_version: int | None = None,
) -> int:
    """Persist a single compose batch immediately to graph.

    Called from within ``_compose_batch`` after LLM success.
    Enriches candidates with provenance metadata, embeds the standard-name
    string, and extracts grammar fields before writing.

    Embedding uses the standard-name string (``id``) — not the description,
    which is added later by the enrich pipeline.  If embedding fails for a
    candidate, it is quarantined (``validation_status='quarantined'``,
    ``validation_issues=['embedding_failed']``) so downstream consumers
    know the vector is missing.

    Returns the number of nodes written.
    """
    from datetime import UTC, datetime

    if not candidates:
        return 0

    now = datetime.now(UTC).isoformat()
    for entry in candidates:
        entry.setdefault("model", compose_model)
        entry.setdefault("review_status", "named")
        entry.setdefault("validation_status", "pending")
        entry.setdefault("generated_at", now)
        # Extract grammar fields into top-level properties for graph
        fields = entry.get("fields", {})
        for field_name in _GRAMMAR_FIELDS:
            if field_name in fields and field_name not in entry:
                entry[field_name] = fields[field_name]

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

    return write_standard_names(candidates)


@retry_on_deadlock()
def persist_enriched_batch(items: list[dict[str, Any]]) -> int:
    """Persist enriched StandardName data and REFERENCES relationships.

    Called by the enrich pipeline PERSIST worker after validation and
    embedding.  Each item dict must have at minimum ``id`` and
    ``enriched_description``.  Optional: ``enriched_documentation``,
    ``enriched_links``, ``enriched_tags``, ``embedding``, ``enrich_model``,
    ``enrich_cost_usd``, ``enrich_tokens``, ``validation_status``,
    ``validation_issues``.

    The Cypher MERGE preserves existing values for identity fields
    (``unit``, ``physics_domain``, ``cocos``, ``cocos_transformation_type``,
    ``kind``, ``source_paths``) via ``coalesce(b.field, sn.field)``
    semantics.  Tags are extended (union, deduplicated).

    After node updates, REFERENCES relationships are created for each
    link in ``enriched_links`` whose target StandardName exists.

    Enrichment claims (``enrich_claimed_at``, ``enrich_claim_token``)
    are released on all processed nodes.

    Returns the number of nodes written.
    """
    if not items:
        return 0

    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat()

    # --- Build UNWIND batch with safe types ---
    batch = []
    for item in items:
        enriched_tags = list(item.get("enriched_tags") or [])
        existing_tags = list(item.get("tags") or [])
        # Union of existing + enriched tags (deduplicated, order-preserved)
        merged_tags = list(dict.fromkeys(existing_tags + enriched_tags)) or None

        batch.append(
            {
                "id": item["id"],
                "description": item.get("enriched_description"),
                "documentation": item.get("enriched_documentation"),
                "embedding": item.get("embedding"),
                "tags": merged_tags,
                "review_status": "enriched",
                "enriched_at": now,
                "enrich_model": item.get("enrich_model"),
                "enrich_cost_usd": item.get("enrich_cost_usd"),
                "enrich_tokens": item.get("enrich_tokens"),
                "validation_status": item.get("validation_status"),
                "validation_issues": item.get("validation_issues") or None,
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
                sn.tags = coalesce(b.tags, sn.tags),
                sn.review_status = b.review_status,
                sn.enriched_at = datetime(b.enriched_at),
                sn.enrich_model = coalesce(b.enrich_model, sn.enrich_model),
                sn.enrich_cost_usd = coalesce(b.enrich_cost_usd, sn.enrich_cost_usd),
                sn.enrich_tokens = coalesce(b.enrich_tokens, sn.enrich_tokens),
                sn.validation_status = coalesce(b.validation_status, sn.validation_status),
                sn.validation_issues = coalesce(b.validation_issues, sn.validation_issues),
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
) -> int:
    """Persist VocabGap nodes and HAS_STANDARD_NAME_VOCAB_GAP relationships.

    Each gap dict has: source_id, segment, needed_token, reason.

    Deduplicates VocabGap nodes by id (vocab_gap:{segment}:{needed_token}).
    Creates HAS_STANDARD_NAME_VOCAB_GAP relationships from source entities with
    per-source reason as a relationship property.

    Returns the number of VocabGap nodes written.
    """
    if not gaps:
        return 0

    from datetime import UTC, datetime

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
            WHERE sn.review_status IN ['named', 'drafted']
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
                   sn.unit AS unit, sn.tags AS tags, sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.physical_base AS physical_base,
                   sn.subject AS subject,
                   sn.component AS component,
                   sn.coordinate AS coordinate,
                   sn.position AS position,
                   sn.process AS process,
                   sn.geometric_base AS geometric_base,
                   sn.object AS object,
                   sn.confidence AS confidence,
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
            WHERE sn.review_status IN ['named', 'drafted', 'published', 'accepted']
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
        "sn.review_status IN ['named', 'drafted']",
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
                   sn.unit AS unit, sn.tags AS tags, sn.links AS links,
                   sn.source_paths AS source_paths, sn.confidence AS confidence,
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
    unit, tags, links, dd_paths, constraints, validity_domain,
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
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.source_paths AS source_paths,
                   sn.constraints AS constraints,
                   sn.validity_domain AS validity_domain,
                   coalesce(sn.confidence, 1.0) AS confidence,
                   sn.model AS model,
                   sn.source_types AS source_types,
                   first_source AS source_path,
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
            where_clauses.append("$source_filter IN sn.source_types")
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
            sn_where_clauses.append("$source_filter IN sn.source_types")
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

    Returns dicts with: id, description, documentation, kind, unit, tags,
    physical_base, subject, component, coordinate, position, process,
    plus all linked DD paths aggregated with their documentation.
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
            where_clauses.append("sn.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter
        if status_filter:
            where_clauses.append("sn.review_status = $status_filter")
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
                   sn.tags AS tags,
                   sn.links AS links,
                   sn.validity_domain AS validity_domain,
                   sn.constraints AS constraints,
                   sn.physical_base AS physical_base,
                   sn.subject AS subject,
                   sn.component AS component,
                   sn.coordinate AS coordinate,
                   sn.position AS position,
                   sn.process AS process,
                   sn.geometric_base AS geometric_base,
                   sn.physics_domain AS physics_domain,
                   sn.review_status AS review_status,
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


def write_enrichment_results(results: list[dict[str, Any]]) -> int:
    """Write enrichment results back to graph.

    Only updates doc fields: description, documentation, tags, links,
    validity_domain, constraints. Clears review_input_hash to invalidate
    stale reviews.

    Does NOT touch: id, physical_base, subject, component, coordinate,
    position, process, kind, unit, model, etc.

    Returns the number of nodes updated.
    """
    if not results:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $batch AS b
            MATCH (sn:StandardName {id: b.id})
            SET sn.description = b.description,
                sn.documentation = b.documentation,
                sn.tags = b.tags,
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
                    "tags": r.get("tags") or None,
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
            SET sns.status = 'composed',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns
            MATCH (sn:StandardName {id: $sn_id})
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
            SET sns.status = 'attached',
                sns.composed_at = datetime(),
                sns.claimed_at = null,
                sns.claim_token = null
            WITH sns
            MATCH (sn:StandardName {id: $sn_id})
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
