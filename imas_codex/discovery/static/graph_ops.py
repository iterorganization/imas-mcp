"""Graph operations for static tree discovery claim coordination.

Provides seed, claim, mark, and query functions for the static discovery
pipeline. TreeModelVersion nodes use status + claimed_at for extraction
coordination. TreeNode enrichment uses enrichment_status.
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.discovery.base.claims import (
    DEFAULT_CLAIM_TIMEOUT_SECONDS,
    reset_stale_claims,
)
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

CLAIM_TIMEOUT_SECONDS = DEFAULT_CLAIM_TIMEOUT_SECONDS  # 300s


# ---------------------------------------------------------------------------
# Seed — create pending TreeModelVersion nodes from config
# ---------------------------------------------------------------------------


def seed_versions(
    facility: str,
    tree_name: str,
    ver_list: list[int],
    version_config: list[dict] | None = None,
) -> int:
    """Create TreeModelVersion nodes with status=discovered for each version.

    Idempotent: only creates nodes that don't already exist. Existing nodes
    are not modified (preserves status/claimed_at from prior runs).

    Args:
        facility: Facility identifier
        tree_name: Static tree name
        ver_list: Version numbers to seed
        version_config: Optional version configs with first_shot info

    Returns:
        Number of new versions seeded
    """
    if not ver_list:
        return 0

    # Build first_shot + description lookups
    first_shot_map: dict[int, int] = {}
    description_map: dict[int, str] = {}
    if version_config:
        for vc in version_config:
            v = vc["version"]
            if "first_shot" in vc:
                first_shot_map[v] = vc["first_shot"]
            if "description" in vc:
                description_map[v] = vc["description"]

    records = []
    for ver in ver_list:
        epoch_id = f"{facility}:{tree_name}:v{ver}"
        records.append(
            {
                "id": epoch_id,
                "facility_id": facility,
                "tree_name": tree_name,
                "version": ver,
                "first_shot": first_shot_map.get(ver, ver),
                "description": description_map.get(ver, ""),
                "status": "discovered",
            }
        )

    with GraphClient() as gc:
        gc.ensure_facility(facility)
        result = gc.query(
            """
            UNWIND $records AS rec
            MERGE (v:TreeModelVersion {id: rec.id})
            ON CREATE SET
                v.facility_id = rec.facility_id,
                v.tree_name = rec.tree_name,
                v.version = rec.version,
                v.first_shot = rec.first_shot,
                v.description = rec.description,
                v.status = rec.status
            WITH v, rec
            WHERE v.status = rec.status
            MATCH (f:Facility {id: rec.facility_id})
            MERGE (v)-[:AT_FACILITY]->(f)
            RETURN count(CASE WHEN v.status = 'discovered' THEN 1 END) AS seeded
            """,
            records=records,
        )
        return result[0]["seeded"] if result else 0


# ---------------------------------------------------------------------------
# Claim — atomic claim of pending TreeModelVersion for extraction
# ---------------------------------------------------------------------------


def claim_version_for_extraction(
    facility: str,
    tree_name: str,
) -> dict[str, Any] | None:
    """Atomically claim a pending TreeModelVersion for extraction.

    Claims the lowest-numbered unclaimed version with status=discovered.
    Sets claimed_at = datetime() for stale-claim recovery.

    Returns:
        Dict with id, version, first_shot or None if no work available
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:TreeModelVersion)
            WHERE v.facility_id = $facility
              AND v.tree_name = $tree_name
              AND v.status = 'discovered'
              AND v.claimed_at IS NULL
            WITH v ORDER BY v.version ASC LIMIT 1
            SET v.claimed_at = datetime()
            RETURN v.id AS id, v.version AS version,
                   v.first_shot AS first_shot
            """,
            facility=facility,
            tree_name=tree_name,
        )
        if result:
            return dict(result[0])
        return None


def mark_version_extracted(
    version_id: str,
    node_count: int,
) -> None:
    """Mark a TreeModelVersion as extracted (status=ingested, clear claim)."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:TreeModelVersion {id: $id})
            SET v.status = 'ingested',
                v.claimed_at = null,
                v.node_count = $node_count,
                v.discovery_date = datetime()
            """,
            id=version_id,
            node_count=node_count,
        )


def release_version_claim(version_id: str) -> None:
    """Release claim on a TreeModelVersion (on error)."""
    with GraphClient() as gc:
        gc.query(
            """
            MATCH (v:TreeModelVersion {id: $id})
            SET v.claimed_at = null
            """,
            id=version_id,
        )


# ---------------------------------------------------------------------------
# Enrichment context — tree hierarchy for LLM prompts
# ---------------------------------------------------------------------------


def fetch_enrichment_context(
    facility: str,
    tree_name: str,
    node_paths: list[str],
) -> dict[str, dict[str, Any]]:
    """Fetch tree hierarchy context for nodes being enriched.

    For each node, queries the graph for its parent STRUCTURE node
    and sibling value nodes. This context is injected into the LLM
    prompt so descriptions are informed by the surrounding tree.

    Args:
        facility: Facility identifier
        tree_name: Static tree name
        node_paths: Paths of nodes being enriched

    Returns:
        Dict mapping node path to context dict with:
        - parent_path: Parent STRUCTURE node path
        - parent_tags: Tags on the parent node
        - siblings: List of sibling dicts (path, node_type, tags, units)
    """
    if not node_paths:
        return {}

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $paths AS node_path
            MATCH (n:TreeNode {path: node_path, facility_id: $facility})
            WHERE n.tree_name = $tree_name AND n.is_static = true
            OPTIONAL MATCH (parent:TreeNode {
                path: n.parent_path, facility_id: $facility
            })
            WHERE parent.tree_name = $tree_name
            OPTIONAL MATCH (parent)-[:HAS_NODE]->(sibling:TreeNode)
            WHERE sibling.tree_name = $tree_name
              AND sibling.path <> n.path
              AND sibling.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
            WITH n, parent, collect(DISTINCT {
                path: sibling.path,
                node_type: sibling.node_type,
                tags: sibling.tags,
                units: sibling.units
            })[0..20] AS siblings
            RETURN n.path AS path,
                   parent.path AS parent_path,
                   parent.tags AS parent_tags,
                   parent.node_type AS parent_type,
                   siblings
            """,
            paths=node_paths,
            facility=facility,
            tree_name=tree_name,
        )
        context: dict[str, dict[str, Any]] = {}
        for r in result:
            # Filter out null-path siblings from OPTIONAL MATCH
            siblings = [s for s in (r["siblings"] or []) if s.get("path") is not None]
            context[r["path"]] = {
                "parent_path": r["parent_path"],
                "parent_tags": r["parent_tags"],
                "parent_type": r["parent_type"],
                "siblings": siblings,
            }
        return context


# ---------------------------------------------------------------------------
# Enrichment claiming — TreeNode nodes
# ---------------------------------------------------------------------------


def claim_nodes_for_enrichment(
    facility: str,
    tree_name: str,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Claim TreeNodes needing enrichment.

    Claims enrichable nodes (NUMERIC, SIGNAL, AXIS, TEXT) without
    descriptions and not currently claimed.

    Returns:
        List of dicts with path, node_type, tags, units
    """
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (n:TreeNode)
            WHERE n.facility_id = $facility
              AND n.tree_name = $tree_name
              AND n.is_static = true
              AND n.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
              AND (n.description IS NULL OR n.description = '')
              AND n.claimed_at IS NULL
            WITH n ORDER BY n.path LIMIT $limit
            SET n.claimed_at = datetime()
            RETURN n.path AS path, n.node_type AS node_type,
                   n.tags AS tags, n.units AS units,
                   n.id AS id
            """,
            facility=facility,
            tree_name=tree_name,
            limit=limit,
        )
        return [dict(r) for r in result]


def mark_nodes_enriched(
    node_ids: list[str],
    descriptions: dict[str, str],
    metadata: dict[str, dict] | None = None,
) -> int:
    """Mark TreeNodes as enriched with descriptions."""
    updates = []
    for nid in node_ids:
        update: dict[str, Any] = {
            "id": nid,
            "description": descriptions.get(nid, ""),
        }
        if metadata and nid in metadata:
            m = metadata[nid]
            if m.get("keywords"):
                update["keywords"] = m["keywords"]
            if m.get("category"):
                update["category"] = m["category"]
        updates.append(update)

    if not updates:
        return 0

    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $updates AS u
            MATCH (n:TreeNode {id: u.id})
            SET n.description = u.description,
                n.keywords = u.keywords,
                n.category = u.category,
                n.enrichment_status = 'enriched',
                n.claimed_at = null
            RETURN count(n) AS updated
            """,
            updates=updates,
        )
        return result[0]["updated"] if result else 0


def release_node_claims(node_ids: list[str]) -> int:
    """Release claims on TreeNodes (on error)."""
    if not node_ids:
        return 0
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $ids AS nid
            MATCH (n:TreeNode {id: nid})
            SET n.claimed_at = null
            RETURN count(n) AS released
            """,
            ids=node_ids,
        )
        return result[0]["released"] if result else 0


# ---------------------------------------------------------------------------
# Has-work queries (for PipelinePhase completion detection)
# ---------------------------------------------------------------------------


def has_pending_extract_work(facility: str, tree_name: str) -> bool:
    """Check if any TreeModelVersions need extraction (discovered or claimed)."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (v:TreeModelVersion)
            WHERE v.facility_id = $facility
              AND v.tree_name = $tree_name
              AND v.status = 'discovered'
            RETURN count(v) > 0 AS has_work
            """,
            facility=facility,
            tree_name=tree_name,
        )
        return result[0]["has_work"] if result else False


def has_pending_enrich_work(facility: str, tree_name: str) -> bool:
    """Check if any TreeNodes need enrichment."""
    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (n:TreeNode)
            WHERE n.facility_id = $facility
              AND n.tree_name = $tree_name
              AND n.is_static = true
              AND n.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
              AND (n.description IS NULL OR n.description = '')
            RETURN count(n) > 0 AS has_work
            """,
            facility=facility,
            tree_name=tree_name,
        )
        return result[0]["has_work"] if result else False


def has_pending_ingest_work(facility: str, tree_name: str) -> bool:
    """Check if there are extracted versions not yet fully ingested.

    This is always False since we ingest inline after extraction.
    Kept for PipelinePhase interface consistency.
    """
    return False


# ---------------------------------------------------------------------------
# Startup — orphan recovery
# ---------------------------------------------------------------------------


def reset_orphaned_static_claims(
    facility: str,
    tree_name: str,
    *,
    silent: bool = False,
) -> dict[str, int]:
    """Release stale claims for static tree discovery."""
    version_reset = reset_stale_claims(
        "TreeModelVersion",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )
    node_reset = reset_stale_claims(
        "TreeNode",
        facility,
        timeout_seconds=CLAIM_TIMEOUT_SECONDS,
        silent=silent,
    )
    total = version_reset + node_reset
    if total and not silent:
        logger.info(
            "Released %d orphaned static claims (%d versions, %d nodes)",
            total,
            version_reset,
            node_reset,
        )
    return {"version_reset": version_reset, "node_reset": node_reset}


# ---------------------------------------------------------------------------
# Stats query (for progress display refresh)
# ---------------------------------------------------------------------------


def get_static_discovery_stats(
    facility: str,
    tree_name: str,
) -> dict[str, int | float]:
    """Get static discovery statistics from graph for progress display."""
    with GraphClient() as gc:
        # Version status counts
        ver_result = gc.query(
            """
            MATCH (v:TreeModelVersion)
            WHERE v.facility_id = $facility AND v.tree_name = $tree_name
            RETURN v.status AS status, count(v) AS cnt,
                   sum(coalesce(v.node_count, 0)) AS nodes
            """,
            facility=facility,
            tree_name=tree_name,
        )

        stats: dict[str, int | float] = {
            "versions_total": 0,
            "versions_discovered": 0,
            "versions_ingested": 0,
            "versions_claimed": 0,
            "nodes_total": 0,
        }

        for r in ver_result:
            cnt = r["cnt"]
            status = r["status"]
            stats["versions_total"] += cnt
            if status == "discovered":
                stats["versions_discovered"] += cnt
            elif status == "ingested":
                stats["versions_ingested"] += cnt
                stats["nodes_total"] += r["nodes"]

        # Claimed versions (in-progress extraction)
        claimed = gc.query(
            """
            MATCH (v:TreeModelVersion)
            WHERE v.facility_id = $facility AND v.tree_name = $tree_name
              AND v.claimed_at IS NOT NULL
            RETURN count(v) AS cnt
            """,
            facility=facility,
            tree_name=tree_name,
        )
        stats["versions_claimed"] = claimed[0]["cnt"] if claimed else 0

        # Node enrichment stats
        node_result = gc.query(
            """
            MATCH (n:TreeNode)
            WHERE n.facility_id = $facility AND n.tree_name = $tree_name
              AND n.is_static = true
            RETURN
                count(n) AS total,
                sum(CASE WHEN n.description IS NOT NULL AND n.description <> ''
                    THEN 1 ELSE 0 END) AS enriched,
                sum(CASE WHEN n.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
                    THEN 1 ELSE 0 END) AS enrichable,
                sum(CASE WHEN n.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
                         AND (n.description IS NULL OR n.description = '')
                    THEN 1 ELSE 0 END) AS pending_enrich
            """,
            facility=facility,
            tree_name=tree_name,
        )

        if node_result:
            r = node_result[0]
            stats["nodes_graph"] = r["total"]
            stats["nodes_enriched"] = r["enriched"]
            stats["nodes_enrichable"] = r["enrichable"]
            stats["pending_enrich"] = r["pending_enrich"]
        else:
            stats["nodes_graph"] = 0
            stats["nodes_enriched"] = 0
            stats["nodes_enrichable"] = 0
            stats["pending_enrich"] = 0

        return stats


# ---------------------------------------------------------------------------
# Clear — delete all static discovery data for a facility
# ---------------------------------------------------------------------------


def clear_facility_static(
    facility: str,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Clear all static tree discovery data for a facility.

    Deletes TreeNode nodes (is_static=true) and their TreeModelVersion
    nodes in batches.

    Args:
        facility: Facility ID
        batch_size: Nodes to delete per batch

    Returns:
        Dict with counts: nodes_deleted, versions_deleted
    """
    results = {
        "nodes_deleted": 0,
        "versions_deleted": 0,
    }

    with GraphClient() as gc:
        # Delete static TreeNode nodes in batches
        while True:
            result = gc.query(
                """
                MATCH (n:TreeNode {facility_id: $facility})
                WHERE n.is_static = true
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["nodes_deleted"] += deleted
            if deleted < batch_size:
                break

        # Delete static TreeModelVersion nodes
        # Static versions have status property (set by static discovery pipeline)
        while True:
            result = gc.query(
                """
                MATCH (v:TreeModelVersion {facility_id: $facility})
                WHERE v.status IS NOT NULL
                WITH v LIMIT $batch_size
                DETACH DELETE v
                RETURN count(v) AS deleted
                """,
                facility=facility,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            results["versions_deleted"] += deleted
            if deleted < batch_size:
                break

    logger.info(
        "Cleared static data for %s: %d nodes, %d versions",
        facility,
        results["nodes_deleted"],
        results["versions_deleted"],
    )
    return results


# ---------------------------------------------------------------------------
# Summary stats — facility-level stats across all static trees
# ---------------------------------------------------------------------------


def get_static_summary_stats(facility: str) -> dict[str, int]:
    """Get static discovery statistics across all trees for a facility.

    Unlike get_static_discovery_stats which requires a tree_name, this
    returns aggregate stats suitable for status/clear commands.
    """
    with GraphClient() as gc:
        ver_result = gc.query(
            """
            MATCH (v:TreeModelVersion {facility_id: $facility})
            WHERE v.status IS NOT NULL
            RETURN v.status AS status, count(v) AS cnt,
                   sum(coalesce(v.node_count, 0)) AS nodes
            """,
            facility=facility,
        )

        stats: dict[str, int] = {
            "versions_total": 0,
            "versions_discovered": 0,
            "versions_ingested": 0,
            "nodes_total": 0,
        }

        for r in ver_result:
            cnt = r["cnt"]
            status = r["status"]
            stats["versions_total"] += cnt
            if status == "discovered":
                stats["versions_discovered"] += cnt
            elif status == "ingested":
                stats["versions_ingested"] += cnt
                stats["nodes_total"] += r["nodes"]

        # Node enrichment stats
        node_result = gc.query(
            """
            MATCH (n:TreeNode {facility_id: $facility})
            WHERE n.is_static = true
            RETURN
                count(n) AS total,
                sum(CASE WHEN n.description IS NOT NULL AND n.description <> ''
                    THEN 1 ELSE 0 END) AS enriched
            """,
            facility=facility,
        )

        if node_result:
            stats["nodes_graph"] = node_result[0]["total"]
            stats["nodes_enriched"] = node_result[0]["enriched"]
        else:
            stats["nodes_graph"] = 0
            stats["nodes_enriched"] = 0

        return stats
