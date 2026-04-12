"""DD source: extract standard name candidates from IMAS Data Dictionary paths.

Surfaces rich graph context for each path: authoritative unit from HAS_UNIT,
LLM-enriched descriptions, cluster siblings, coordinates, parent structure,
and keywords. Groups by (IDS × cluster × unit) for unit-safe batching.
"""

from __future__ import annotations

import logging

from imas_codex.sn.sources.base import ExtractionBatch

logger = logging.getLogger(__name__)

# Enriched extraction query — single Cypher surfacing all context
_ENRICHED_QUERY = """
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE {where_clause}
WITH n, ids
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
OPTIONAL MATCH (n)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
OPTIONAL MATCH (n)-[:HAS_PARENT]->(parent:IMASNode)
OPTIONAL MATCH (n)-[:HAS_COORDINATE]->(coord:IMASNode)
OPTIONAL MATCH (coord)-[:HAS_UNIT]->(cu:Unit)
RETURN n.id AS path,
       n.description AS description,
       n.documentation AS documentation,
       n.unit AS unit,
       u.id AS unit_from_rel,
       n.data_type AS data_type,
       n.physics_domain AS physics_domain,
       n.keywords AS keywords,
       n.node_category AS node_category,
       n.ndim AS ndim,
       n.lifecycle_status AS lifecycle_status,
       ids.id AS ids_name,
       c.label AS cluster_label,
       c.id AS cluster_id,
       c.description AS cluster_description,
       parent.id AS parent_path,
       parent.description AS parent_description,
       parent.data_type AS parent_type,
       coord.id AS coord_path,
       coord.description AS coord_description,
       cu.id AS coord_unit
ORDER BY ids.id, n.id
LIMIT $limit
"""

# Cluster siblings query — paths sharing the same cluster
_SIBLINGS_QUERY = """
MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(sibling:IMASNode)
WHERE c.id IN $cluster_ids AND sibling.id <> $exclude_path
WITH c.id AS cluster_id, sibling.id AS sibling_path,
     sibling.description AS sibling_desc
OPTIONAL MATCH (sibling)-[:HAS_UNIT]->(su:Unit)
RETURN cluster_id, sibling_path,
       su.id AS sibling_unit, sibling_desc
ORDER BY cluster_id, sibling_path
"""


def extract_dd_candidates(
    *,
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
    existing_names: set[str] | None = None,
) -> list[ExtractionBatch]:
    """Extract candidate quantities from IMAS DD paths with enriched context.

    Queries IMASNode paths from the graph with full context: authoritative unit,
    cluster siblings, coordinates, parent structure, and LLM-enriched descriptions.
    Groups by (IDS × cluster × unit) for unit-safe batching.

    Args:
        ids_filter: Restrict to specific IDS (e.g., "equilibrium")
        domain_filter: Restrict to physics domain
        limit: Max paths to extract
        existing_names: Known standard names for dedup awareness

    Returns:
        List of ExtractionBatch objects grouped by (IDS, cluster, unit)
    """
    from imas_codex.graph.client import GraphClient

    if existing_names is None:
        existing_names = set()

    with GraphClient() as gc:
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
        query = _ENRICHED_QUERY.format(where_clause=where_clause)

        results = list(gc.query(query, **params))

        if not results:
            logger.info("No DD paths found matching filters")
            return []

        # Resolve authoritative unit: prefer HAS_UNIT relationship, fall back to node property
        for row in results:
            row["unit"] = row.get("unit_from_rel") or row.get("unit") or None

        # Collect cluster IDs for sibling lookup
        cluster_ids = {r["cluster_id"] for r in results if r.get("cluster_id")}
        siblings_by_cluster: dict[str, list[dict]] = {}
        if cluster_ids:
            path_set = {r["path"] for r in results}
            sib_results = list(
                gc.query(
                    _SIBLINGS_QUERY,
                    cluster_ids=sorted(cluster_ids),
                    exclude_path="",  # we filter client-side
                )
            )
            for sr in sib_results:
                cid = sr["cluster_id"]
                if sr["sibling_path"] not in path_set:
                    siblings_by_cluster.setdefault(cid, []).append(
                        {
                            "path": sr["sibling_path"],
                            "unit": sr.get("sibling_unit"),
                            "description": sr.get("sibling_desc", ""),
                        }
                    )

    # Attach siblings to each result
    for row in results:
        cid = row.get("cluster_id")
        if cid and cid in siblings_by_cluster:
            row["cluster_siblings"] = siblings_by_cluster[cid][:10]
        else:
            row["cluster_siblings"] = []

    # Group by (IDS × cluster × unit) for unit-safe batching
    groups: dict[str, list[dict]] = {}
    for row in results:
        ids_name = row["ids_name"]
        cluster = row.get("cluster_label") or "unclustered"
        unit = row.get("unit") or "dimensionless"
        group_key = f"{ids_name}/{cluster}/{unit}"
        groups.setdefault(group_key, []).append(dict(row))

    batches = []
    for group_key, items in groups.items():
        ids_name = items[0]["ids_name"]
        cluster_label = items[0].get("cluster_label") or "unclustered"
        unit = items[0].get("unit") or "dimensionless"

        # Build rich context summary
        context_parts = [f"IDS: {ids_name}"]
        context_parts.append(f"Cluster: {cluster_label}")
        context_parts.append(f"Authoritative unit: {unit}")
        context_parts.append(f"{len(items)} paths sharing this concept")

        # Include sibling summary
        all_siblings = items[0].get("cluster_siblings", [])
        if all_siblings:
            sib_strs = [
                f"  {s['path']} ({s.get('unit', '?')})" for s in all_siblings[:5]
            ]
            context_parts.append("Cross-IDS siblings:\n" + "\n".join(sib_strs))

        batches.append(
            ExtractionBatch(
                source="dd",
                group_key=group_key,
                items=items,
                context="\n".join(context_parts),
                existing_names=existing_names,
            )
        )

    logger.info(
        "Extracted %d batches from %d DD paths (%d clusters)",
        len(batches),
        len(results),
        len(cluster_ids),
    )
    return batches
