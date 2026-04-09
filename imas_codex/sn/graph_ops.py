"""Graph operations for the standard-name pipeline.

Provides read/write helpers that query or mutate StandardName nodes and
their HAS_STANDARD_NAME relationships in the Neo4j knowledge graph.

Relationship direction: entity → concept
  (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
  (:FacilitySignal)-[:HAS_STANDARD_NAME]->(sn:StandardName)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
    from imas_codex.graph.client import GraphClient

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
                   n.units AS units, n.data_type AS data_type,
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
    from imas_codex.graph.client import GraphClient

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
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        results = gc.query("MATCH (sn:StandardName) RETURN sn.id AS id")
        return {r["id"] for r in results}


def get_named_source_ids() -> set[str]:
    """Return source IDs already linked via HAS_STANDARD_NAME.

    Used for resumability: extract skips sources that already have
    a standard name unless --force is specified.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        results = gc.query("""
            MATCH (src)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            RETURN DISTINCT src.id AS source_id
        """)
        return {r["source_id"] for r in results}


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
    ``coordinate``, ``position``, ``units``, ``description``,
    ``model``, ``review_status``, ``generated_at``, ``confidence``.

    Returns the number of nodes written.
    """
    from imas_codex.graph.client import GraphClient

    if not names:
        return 0

    with GraphClient() as gc:
        # MERGE StandardName nodes with provenance
        gc.query(
            """
            UNWIND $batch AS b
            MERGE (sn:StandardName {id: b.id})
            SET sn.source_type = b.source_type,
                sn.physical_base = b.physical_base,
                sn.subject = b.subject,
                sn.component = b.component,
                sn.coordinate = b.coordinate,
                sn.position = b.position,
                sn.units = b.units,
                sn.description = b.description,
                sn.model = b.model,
                sn.review_status = b.review_status,
                sn.generated_at = b.generated_at,
                sn.confidence = b.confidence,
                sn.created_at = coalesce(sn.created_at, datetime())
            """,
            batch=[
                {
                    "id": n["id"],
                    "source_type": n.get("source_type", ""),
                    "physical_base": n.get("physical_base"),
                    "subject": n.get("subject"),
                    "component": n.get("component"),
                    "coordinate": n.get("coordinate"),
                    "position": n.get("position"),
                    "units": n.get("units"),
                    "description": n.get("description"),
                    "model": n.get("model"),
                    "review_status": n.get("review_status"),
                    "generated_at": n.get("generated_at"),
                    "confidence": n.get("confidence"),
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

    written = len(names)
    logger.info("Wrote %d StandardName nodes", written)
    return written


# =============================================================================
# Read helpers — publish (validated standard names)
# =============================================================================


def get_validated_standard_names(
    ids_filter: str | None = None,
    confidence_min: float = 0.0,
) -> list[dict[str, Any]]:
    """Read validated StandardName nodes and their provenance.

    Queries all StandardName nodes, joining through ``HAS_STANDARD_NAME``
    to find source entities and their parent IDS.  Uses ``collect()``
    to avoid row duplication when a name has multiple sources (takes
    the first source).

    Parameters
    ----------
    ids_filter:
        Restrict to names derived from a specific IDS (matched via
        ``IMASNode -[:HAS_STANDARD_NAME]-> StandardName`` and
        ``IMASNode -[:IN_IDS]-> IDS``).
    confidence_min:
        Minimum confidence threshold.  Nodes without a ``confidence``
        property are treated as 1.0 (grammar-validated).

    Returns
    -------
    list of dicts with keys: name, description, source, source_path,
    canonical_units, confidence, ids_name.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        params: dict[str, Any] = {"confidence_min": confidence_min}

        # Collect source info — use HAS_STANDARD_NAME (entity → concept)
        cypher = """
            MATCH (sn:StandardName)
            WHERE coalesce(sn.confidence, 1.0) >= $confidence_min
            OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME]->(sn)
            OPTIONAL MATCH (src)-[:IN_IDS]->(ids:IDS)
            WITH sn,
                 collect(DISTINCT src.id)[0] AS first_source,
                 collect(DISTINCT ids.id)[0] AS first_ids
        """

        if ids_filter:
            # Re-check: at least one HAS_STANDARD_NAME source must be in the target IDS
            cypher += """
            WITH sn, first_source, first_ids
            WHERE first_ids = $ids_filter
            """
            params["ids_filter"] = ids_filter

        cypher += """
            RETURN sn.id AS name,
                   sn.description AS description,
                   coalesce(sn.source, sn.source_type) AS source,
                   coalesce(sn.source_path, first_source) AS source_path,
                   coalesce(sn.canonical_units, sn.units) AS canonical_units,
                   coalesce(sn.confidence, 1.0) AS confidence,
                   first_ids AS ids_name
            ORDER BY sn.id
        """

        results = gc.query(cypher, **params)
        logger.info(
            "Read %d validated standard names (ids_filter=%s, confidence_min=%.2f)",
            len(results),
            ids_filter,
            confidence_min,
        )
        return list(results)
