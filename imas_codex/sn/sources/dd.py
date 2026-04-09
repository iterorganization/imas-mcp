"""DD source: extract standard name candidates from IMAS Data Dictionary paths."""

from __future__ import annotations

import logging

from imas_codex.sn.sources.base import ExtractionBatch

logger = logging.getLogger(__name__)


def extract_dd_candidates(
    *,
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
    existing_names: set[str] | None = None,
) -> list[ExtractionBatch]:
    """Extract candidate quantities from IMAS DD paths.

    Queries IMASNode paths from the graph, groups by IDS and semantic cluster,
    and returns batches ready for LLM composition.

    Args:
        ids_filter: Restrict to specific IDS (e.g., "equilibrium")
        domain_filter: Restrict to physics domain
        limit: Max paths to extract
        existing_names: Known standard names for dedup awareness

    Returns:
        List of ExtractionBatch objects grouped by IDS
    """
    from imas_codex.graph.client import GraphClient

    if existing_names is None:
        existing_names = set()

    with GraphClient() as gc:
        # Query leaf data paths with descriptions
        params: dict = {"limit": limit}
        where_parts = [
            "n.node_type = 'dynamic'",
            "n.description IS NOT NULL",
            "n.description <> ''",
        ]
        if ids_filter:
            where_parts.append("ids.id = $ids_filter")
            params["ids_filter"] = ids_filter
        if domain_filter:
            where_parts.append("n.physics_domain = $domain_filter")
            params["domain_filter"] = domain_filter

        where_clause = " AND ".join(where_parts)

        results = list(
            gc.query(
                f"""
            MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
            WHERE {where_clause}
            WITH n, ids
            OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
            RETURN n.id AS path, n.description AS description,
                   n.units AS units, n.data_type AS data_type,
                   n.physics_domain AS physics_domain,
                   ids.id AS ids_name,
                   c.label AS cluster_label, c.id AS cluster_id
            ORDER BY ids.id, n.id
            LIMIT $limit
        """,
                **params,
            )
        )

    if not results:
        logger.info("No DD paths found matching filters")
        return []

    # Group by IDS name for coherent batches
    groups: dict[str, list[dict]] = {}
    for row in results:
        ids_name = row["ids_name"]
        groups.setdefault(ids_name, []).append(dict(row))

    batches = []
    for ids_name, items in groups.items():
        # Build context summary for the LLM
        cluster_labels = sorted(
            {i["cluster_label"] for i in items if i.get("cluster_label")}
        )
        context = f"IDS: {ids_name}"
        if cluster_labels:
            context += f"\nSemantic clusters: {', '.join(cluster_labels[:10])}"
        context += f"\n{len(items)} data paths with physics quantities"

        batches.append(
            ExtractionBatch(
                source="dd",
                group_key=ids_name,
                items=items,
                context=context,
                existing_names=existing_names,
            )
        )

    logger.info("Extracted %d batches from %d DD paths", len(batches), len(results))
    return batches
