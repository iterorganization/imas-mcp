"""DD source: extract standard name candidates from IMAS Data Dictionary paths.

Surfaces rich graph context for each path: authoritative unit from HAS_UNIT,
LLM-enriched descriptions, cluster siblings, coordinates, parent structure,
and keywords. Passes paths through enrichment layer for classification,
primary cluster selection, and global (cluster × unit) batching.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from imas_codex.standard_names.sources.base import ExtractionBatch

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
       cu.id AS coord_unit,
       n.cocos_label_transformation AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression
ORDER BY ids.id, n.id
LIMIT $limit
"""

# Cluster siblings query — paths sharing the same cluster
_SIBLINGS_QUERY = """
MATCH (c:IMASSemanticCluster)<-[:IN_CLUSTER]-(sibling:IMASNode)
WHERE c.id IN $cluster_ids AND NOT (sibling.id IN $exclude_paths)
WITH c.id AS cluster_id, sibling
OPTIONAL MATCH (sibling)-[:HAS_UNIT]->(su:Unit)
RETURN cluster_id, sibling.id AS sibling_path,
       su.id AS sibling_unit, sibling.description AS sibling_desc
ORDER BY cluster_id, sibling_path
"""


def extract_dd_candidates(
    *,
    ids_filter: str | None = None,
    domain_filter: str | None = None,
    limit: int = 500,
    existing_names: set[str] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> list[ExtractionBatch]:
    """Extract candidate quantities from IMAS DD paths with enriched context.

    Queries IMASNode paths from the graph with full context: authoritative unit,
    cluster siblings, coordinates, parent structure, and LLM-enriched descriptions.
    Classifies paths (quantity vs metadata vs skip), selects primary cluster per
    path, and groups **globally** by (cluster × unit) for unit-safe batching.

    Args:
        ids_filter: Restrict to specific IDS (e.g., "equilibrium")
        domain_filter: Restrict to physics domain
        limit: Max paths to extract
        existing_names: Known standard names for dedup awareness
        on_status: Optional callback ``(text: str) -> None`` for progress updates

    Returns:
        List of ExtractionBatch objects grouped by (primary_cluster, unit)
    """
    from imas_codex.graph.client import GraphClient

    def _status(text: str) -> None:
        if on_status:
            on_status(text)

    if existing_names is None:
        existing_names = set()

    _status("querying graph…")

    with GraphClient() as gc:
        # Resolve extraction DD version and COCOS convention
        dv_row = next(
            iter(
                gc.query("""
                MATCH (dv:DDVersion {is_current: true})
                OPTIONAL MATCH (dv)-[:HAS_COCOS]->(c:COCOS)
                RETURN dv.id AS dd_version, dv.cocos AS cocos_version,
                       properties(c) AS cocos_params
            """)
            ),
            None,
        )
        extraction_dd_version = dv_row["dd_version"] if dv_row else None
        cocos_version = dv_row["cocos_version"] if dv_row else None
        cocos_params = dv_row["cocos_params"] if dv_row else None

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

        _status(f"found {len(results)} paths, resolving units…")

        # Resolve authoritative unit: prefer HAS_UNIT relationship, fall back to node property
        for row in results:
            row["unit"] = row.get("unit_from_rel") or row.get("unit") or None

        # Collect cluster IDs for sibling lookup
        cluster_ids = {r["cluster_id"] for r in results if r.get("cluster_id")}
        siblings_by_cluster: dict[str, list[dict]] = {}
        if cluster_ids:
            _status(f"fetching siblings for {len(cluster_ids)} clusters…")
            path_set = {r["path"] for r in results}
            sib_results = list(
                gc.query(
                    _SIBLINGS_QUERY,
                    cluster_ids=sorted(cluster_ids),
                    exclude_paths=sorted(path_set),
                )
            )
            for sr in sib_results:
                cid = sr["cluster_id"]
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

    # --- Enrichment layer: classify, deduplicate, select primary cluster ----
    _status(f"classifying {len(results)} paths…")
    from imas_codex.standard_names.enrichment import (
        enrich_paths,
        group_by_concept_and_unit,
    )

    enriched = enrich_paths(results)

    if not enriched:
        logger.info("No quantity paths after classification")
        return []

    # Group GLOBALLY by (primary_cluster × unit) — same concept across IDSs
    # gets batched together for coherent naming.
    _status(f"grouping {len(enriched)} quantities into batches…")
    batches = group_by_concept_and_unit(
        enriched,
        max_batch_size=25,
        existing_names=existing_names,
    )

    logger.info(
        "Extracted %d batches from %d DD paths (%d clusters)",
        len(batches),
        len(results),
        len(cluster_ids),
    )

    # Propagate COCOS metadata to batches
    for batch in batches:
        batch.extraction_dd_version = extraction_dd_version
        batch.cocos_version = cocos_version
        batch.cocos_params = cocos_params

    return batches


# Targeted extraction query — single path with full context
_TARGETED_PATH_QUERY = """
MATCH (n:IMASNode {id: $path})-[:IN_IDS]->(ids:IDS)
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
       n.cocos_label_transformation AS cocos_label,
       n.cocos_transformation_expression AS cocos_expression,
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
"""


def extract_specific_paths(
    paths: list[str],
    *,
    existing_names: set[str] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> list[ExtractionBatch]:
    """Extract specific DD paths with full context — bypasses classifier.

    Used for targeted debugging via ``--paths`` CLI flag. Each path becomes
    its own batch (serial processing) with rich context including the
    previous StandardName metadata and all DD paths linked to that name.

    Args:
        paths: Explicit list of DD path IDs to process
        existing_names: Known standard names for dedup awareness in compose
        on_status: Optional progress callback

    Returns:
        List of ExtractionBatch objects — one per path for maximum context
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.enrichment import (
        build_batch_context,
        select_primary_cluster,
    )

    if existing_names is None:
        existing_names = set()

    def _status(text: str) -> None:
        if on_status:
            on_status(text)

    _status(f"fetching {len(paths)} targeted paths…")

    results: list[dict] = []
    with GraphClient() as gc:
        # Resolve extraction DD version and COCOS convention
        dv_row = next(
            iter(
                gc.query("""
                MATCH (dv:DDVersion {is_current: true})
                OPTIONAL MATCH (dv)-[:HAS_COCOS]->(c:COCOS)
                RETURN dv.id AS dd_version, dv.cocos AS cocos_version,
                       properties(c) AS cocos_params
            """)
            ),
            None,
        )
        extraction_dd_version = dv_row["dd_version"] if dv_row else None
        cocos_version = dv_row["cocos_version"] if dv_row else None
        cocos_params = dv_row["cocos_params"] if dv_row else None

        for path in paths:
            rows = list(gc.query(_TARGETED_PATH_QUERY, path=path))
            if rows:
                results.extend(rows)
            else:
                logger.warning("Path not found in graph: %s", path)

    if not results:
        logger.info("No targeted paths found in graph")
        return []

    # Resolve units + deduplicate rows (multi-cluster → multiple rows per path)
    path_base: dict[str, dict] = {}
    path_clusters: dict[str, list[dict]] = {}

    for row in results:
        p = row.get("path", "")
        row["unit"] = row.get("unit_from_rel") or row.get("unit") or None
        row["cluster_siblings"] = []

        if p not in path_base:
            path_base[p] = dict(row)
            path_clusters[p] = []

        cid = row.get("cluster_id")
        if cid:
            existing_cids = {c["cluster_id"] for c in path_clusters[p]}
            if cid not in existing_cids:
                path_clusters[p].append(
                    {
                        "cluster_id": cid,
                        "cluster_label": row.get("cluster_label") or "",
                        "cluster_description": row.get("cluster_description") or "",
                    }
                )

    # Select primary cluster + attach enrichment
    enriched: list[dict] = []
    for path, base_row in path_base.items():
        clusters = path_clusters.get(path, [])
        primary = select_primary_cluster(clusters)
        base_row["primary_cluster_id"] = primary["cluster_id"] if primary else None
        base_row["primary_cluster_label"] = (
            primary["cluster_label"] if primary else None
        )
        base_row["primary_cluster_description"] = (
            primary["cluster_description"] if primary else None
        )
        base_row["all_clusters"] = clusters
        enriched.append(base_row)

    _status(f"creating {len(enriched)} single-path batches…")

    # One batch per path for maximum context depth
    batches: list[ExtractionBatch] = []
    for item in enriched:
        ids_name = item.get("ids_name", "unknown")
        context = build_batch_context([item], ids_name)
        batches.append(
            ExtractionBatch(
                source="dd",
                group_key=ids_name,
                items=[item],
                context=context,
                existing_names=existing_names,
            )
        )

    logger.info(
        "Extracted %d single-path batches from %d targeted paths",
        len(batches),
        len(enriched),
    )

    # Propagate COCOS metadata to batches
    for batch in batches:
        batch.extraction_dd_version = extraction_dd_version
        batch.cocos_version = cocos_version
        batch.cocos_params = cocos_params

    return batches
