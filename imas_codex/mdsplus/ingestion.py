"""MDSplus tree structure ingestion to Neo4j.

Creates TreeModelVersion and TreeNode entities with proper relationships.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def normalize_mdsplus_path(path: str) -> str:
    """Normalize an MDSplus path to canonical form.

    - Single backslash prefix
    - Uppercase tree and node names
    - Consistent :: separator

    Args:
        path: Raw MDSplus path

    Returns:
        Normalized path like \\RESULTS::TOP.NODE
    """
    # Remove all leading backslashes
    path = path.lstrip("\\")
    # Uppercase the entire path
    path = path.upper()
    # Ensure single backslash prefix
    return f"\\{path}"


def ingest_epochs(
    client: "GraphClient",
    epochs: list[dict],
    dry_run: bool = False,
    delete_orphans: bool = True,
) -> int:
    """Ingest TreeModelVersion nodes to the graph.

    Args:
        client: Neo4j GraphClient
        epochs: List of epoch dicts from discover_epochs()
        dry_run: If True, log but don't write
        delete_orphans: If True, delete epochs not in the new set (for full re-scans)

    Returns:
        Number of epochs ingested
    """
    if not epochs:
        logger.warning("No epochs to ingest")
        return 0

    facility = epochs[0]["facility_id"]
    tree_name = epochs[0]["tree_name"]
    epoch_ids = [e["id"] for e in epochs]

    if dry_run:
        for epoch in epochs:
            logger.info(
                f"[DRY RUN] Would create TreeModelVersion: {epoch['id']} "
                f"(v{epoch['version']}, shots {epoch['first_shot']}-{epoch.get('last_shot', 'current')})"
            )
        return len(epochs)

    # Delete orphaned epochs (from buggy fingerprinting) before ingesting new ones
    if delete_orphans:
        orphan_result = client.query(
            """
            MATCH (v:TreeModelVersion {facility_id: $facility, tree_name: $tree})
            WHERE NOT v.id IN $epoch_ids
            WITH v, v.id AS orphan_id
            DETACH DELETE v
            RETURN count(orphan_id) AS deleted
            """,
            facility=facility,
            tree=tree_name,
            epoch_ids=epoch_ids,
        )
        deleted = orphan_result[0]["deleted"] if orphan_result else 0
        if deleted > 0:
            logger.info(f"Deleted {deleted} orphaned epochs for {tree_name}")

    # Prepare records (remove paths used for super tree, not stored)
    records = []
    for epoch in epochs:
        record = {
            k: v for k, v in epoch.items() if k not in ("added_paths", "removed_paths")
        }
        record["discovery_date"] = datetime.now().isoformat()
        records.append(record)

    # Batch insert with UNWIND
    client.query(
        """
        UNWIND $epochs AS epoch
        MERGE (v:TreeModelVersion {id: epoch.id})
        SET v += epoch
        WITH v, epoch
        MATCH (f:Facility {id: epoch.facility_id})
        MERGE (v)-[:FACILITY_ID]->(f)
    """,
        epochs=records,
    )

    # Create predecessor relationships
    client.query("""
        MATCH (v:TreeModelVersion)
        WHERE v.predecessor IS NOT NULL
        WITH v
        MATCH (pred:TreeModelVersion {id: v.predecessor})
        MERGE (v)-[:SUCCEEDS]->(pred)
    """)

    logger.info(f"Ingested {len(epochs)} TreeModelVersion nodes")
    return len(epochs)


def ingest_super_tree(
    client: "GraphClient",
    facility: str,
    tree_name: str,
    epochs: list[dict],
    structures: dict[int, tuple[int, frozenset[str]]],
    dry_run: bool = False,
) -> int:
    """Build the "super tree" - all TreeNodes with applicability ranges.

    Creates TreeNodes that span their full validity range (first_shot to last_shot)
    rather than per-shot nodes. Links nodes to their introducing TreeModelVersion.

    Args:
        client: Neo4j GraphClient
        facility: Facility ID (e.g., "epfl")
        tree_name: MDSplus tree name (e.g., "results")
        epochs: Epoch list from discover_epochs()
        structures: Structure dict from discover_epochs()
        dry_run: If True, log but don't write

    Returns:
        Number of TreeNodes created
    """
    # Build path -> (first_shot, last_shot, introduced_version) map
    path_ranges: dict[str, dict] = {}

    # Sort epochs by version
    sorted_epochs = sorted(epochs, key=lambda e: e["version"])

    for epoch in sorted_epochs:
        version_id = epoch["id"]
        first_shot = epoch["first_shot"]

        # Paths added in this epoch
        added_paths = epoch.get("added_paths", [])
        for path in added_paths:
            if path not in path_ranges:
                path_ranges[path] = {
                    "first_shot": first_shot,
                    "last_shot": None,  # Still present
                    "introduced_version": version_id,
                    "removed_version": None,
                }

        # Paths removed in this epoch
        removed_paths = epoch.get("removed_paths", [])
        for path in removed_paths:
            if path in path_ranges and path_ranges[path]["last_shot"] is None:
                path_ranges[path]["last_shot"] = first_shot - 1
                path_ranges[path]["removed_version"] = version_id

    # Create TreeNode records
    nodes = []
    for path, ranges in path_ranges.items():
        parent_path = _compute_parent_path(path)
        node_id = f"{facility}:{tree_name}:{path}"

        nodes.append(
            {
                "id": node_id,
                "path": path,
                "tree_name": tree_name,
                "facility_id": facility,
                "parent_path": parent_path,
                "first_shot": ranges["first_shot"],
                "last_shot": ranges["last_shot"],
                "introduced_version": ranges["introduced_version"],
                "removed_version": ranges["removed_version"],
                "node_type": "STRUCTURE",  # Default, can be enhanced later
                "units": "dimensionless",
            }
        )

    if dry_run:
        logger.info(f"[DRY RUN] Would create {len(nodes)} TreeNode records")
        for n in nodes[:5]:
            logger.info(
                f"  {n['path']}: shots {n['first_shot']}-{n['last_shot'] or 'present'}"
            )
        if len(nodes) > 5:
            logger.info(f"  ... and {len(nodes) - 5} more")
        return len(nodes)

    # Batch insert TreeNodes - use (path, facility_id) to match existing constraint
    batch_size = 500
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        client.query(
            """
            UNWIND $nodes AS node
            MERGE (n:TreeNode {path: node.path, facility_id: node.facility_id})
            SET n.id = node.id,
                n.tree_name = node.tree_name,
                n.parent_path = node.parent_path,
                n.first_shot = node.first_shot,
                n.last_shot = node.last_shot,
                n.introduced_version = node.introduced_version,
                n.removed_version = node.removed_version,
                n.node_type = COALESCE(n.node_type, node.node_type),
                n.units = COALESCE(n.units, node.units)
        """,
            nodes=batch,
        )
        logger.debug(f"Inserted batch {i // batch_size + 1}")

    # Create FACILITY_ID relationships
    client.query("""
        MATCH (n:TreeNode)
        WHERE n.facility_id IS NOT NULL
        WITH n
        MATCH (f:Facility {id: n.facility_id})
        MERGE (n)-[:FACILITY_ID]->(f)
    """)

    # Create INTRODUCED_IN relationships
    client.query("""
        MATCH (n:TreeNode)
        WHERE n.introduced_version IS NOT NULL
        WITH n
        MATCH (v:TreeModelVersion {id: n.introduced_version})
        MERGE (n)-[:INTRODUCED_IN]->(v)
    """)

    # Create REMOVED_IN relationships
    client.query("""
        MATCH (n:TreeNode)
        WHERE n.removed_version IS NOT NULL
        WITH n
        MATCH (v:TreeModelVersion {id: n.removed_version})
        MERGE (n)-[:REMOVED_IN]->(v)
    """)

    # Create parent-child relationships (HAS_NODE) - use indexed lookup
    client.query(
        """
        MATCH (child:TreeNode)
        WHERE child.tree_name = $tree_name
          AND child.facility_id = $facility
          AND child.parent_path IS NOT NULL
        WITH child
        MATCH (parent:TreeNode {
            path: child.parent_path,
            facility_id: $facility
        })
        WHERE parent.tree_name = $tree_name
        MERGE (parent)-[:HAS_NODE]->(child)
    """,
        tree_name=tree_name,
        facility=facility,
    )

    logger.info(f"Ingested {len(nodes)} TreeNode records with relationships")
    return len(nodes)


def _compute_parent_path(path: str) -> str | None:
    """Compute the parent path for a TreeNode.

    Examples:
        \\TOP.SUB.LEAF -> \\TOP.SUB
        \\RESULTS::TOP.DIAGNOSTICS.SOFT_X -> \\RESULTS::TOP.DIAGNOSTICS
        \\RESULTS::TOP -> None (top-level)
    """
    if "::" in path:
        # Has subtree: \\RESULTS::TOP.SUB.LEAF
        tree_part, node_part = path.split("::", 1)
        if "." not in node_part:
            return None  # Top-level node
        parent_node = ".".join(node_part.rsplit(".", 1)[:-1])
        return f"{tree_part}::{parent_node}"
    else:
        # No subtree: \\TOP.SUB.LEAF
        if "." not in path:
            return None  # Top-level
        return ".".join(path.rsplit(".", 1)[:-1])


def enrich_node_metadata(
    client: "GraphClient",
    facility: str,
    tree_name: str,
    shot: int,
    dry_run: bool = False,
) -> int:
    """Enrich TreeNodes with metadata (node_type, units, description).

    Uses the tree structure at a specific shot to fill in metadata.
    Should be called after ingest_super_tree().

    Args:
        client: Neo4j GraphClient
        facility: Facility ID
        tree_name: MDSplus tree name
        shot: Shot to query for metadata
        dry_run: If True, log but don't write

    Returns:
        Number of nodes updated
    """
    from .discovery import TreeDiscovery

    discovery = TreeDiscovery(facility=facility)

    # Get paths that need metadata
    result = client.query(
        """
        MATCH (n:TreeNode {tree_name: $tree, facility_id: $facility})
        WHERE n.node_type = 'STRUCTURE' OR n.node_type IS NULL
        RETURN n.path AS path
    """,
        tree=tree_name,
        facility=facility,
    )

    paths = [r["path"] for r in result]
    if not paths:
        logger.info("No nodes need metadata enrichment")
        return 0

    logger.info(f"Enriching metadata for {len(paths)} nodes at shot {shot}")

    if dry_run:
        logger.info(f"[DRY RUN] Would query metadata for {len(paths)} paths")
        return len(paths)

    # Get node details from MDSplus
    details = discovery.get_node_details(tree_name, shot, paths[:1000])  # Limit batch

    if not details:
        logger.warning("No details returned from MDSplus")
        return 0

    # Update nodes with metadata
    client.query(
        """
        UNWIND $details AS d
        MATCH (n:TreeNode {
            path: d.path,
            tree_name: $tree,
            facility_id: $facility
        })
        SET n.node_type = d.node_type,
            n.units = d.units,
            n.description = d.description
    """,
        details=details,
        tree=tree_name,
        facility=facility,
    )

    logger.info(f"Enriched {len(details)} nodes with metadata")
    return len(details)


def merge_legacy_metadata(
    client: "GraphClient",
    facility: str,
    tree_name: str,
    dry_run: bool = False,
) -> dict:
    """Merge metadata from legacy nodes (without epoch info) into epoch-aware nodes.

    Legacy nodes may have valuable description, physics_domain, or units fields
    that should be preserved when re-ingesting with epoch discovery.

    Matching is done by normalized path (case-insensitive, normalized backslashes).

    Args:
        client: Neo4j GraphClient
        facility: Facility ID (e.g., "epfl")
        tree_name: MDSplus tree name (e.g., "results")
        dry_run: If True, log but don't write

    Returns:
        Dict with merge statistics
    """
    # Find legacy nodes with valuable metadata that match epoch nodes
    # Normalize paths by uppercasing and removing extra backslashes
    # Use parameters to pass backslash strings to avoid escaping issues
    double_bs = "\\\\"  # Two backslash chars
    single_bs = "\\"  # One backslash char

    result = client.query(
        """
        MATCH (legacy:TreeNode)
        WHERE legacy.first_shot IS NULL
          AND legacy.tree_name = $tree_name
          AND (legacy.description IS NOT NULL
               OR legacy.physics_domain IS NOT NULL)
        WITH legacy,
             toUpper(replace(legacy.path, $double_bs, $single_bs)) AS norm_legacy
        MATCH (epoch:TreeNode)
        WHERE epoch.first_shot IS NOT NULL
          AND epoch.tree_name = $tree_name
          AND toUpper(replace(epoch.path, $double_bs, $single_bs)) = norm_legacy
        RETURN legacy.path AS legacy_path,
               epoch.path AS epoch_path,
               legacy.description AS legacy_desc,
               epoch.description AS epoch_desc,
               legacy.physics_domain AS legacy_domain,
               epoch.physics_domain AS epoch_domain,
               legacy.units AS legacy_units,
               epoch.units AS epoch_units
        """,
        tree_name=tree_name,
        double_bs=double_bs,
        single_bs=single_bs,
    )

    stats = {
        "matched": len(result),
        "descriptions_merged": 0,
        "physics_domains_merged": 0,
        "units_merged": 0,
    }

    if not result:
        logger.info(f"No legacy nodes with metadata to merge for {tree_name}")
        return stats

    logger.info(f"Found {len(result)} legacy nodes with metadata to merge")

    if dry_run:
        for r in result[:5]:
            logger.info(
                f"  [DRY RUN] Would merge: {r['legacy_path']} -> {r['epoch_path']}"
            )
        return stats

    # Merge metadata - prefer legacy if epoch is missing or generic
    for r in result:
        updates = {}

        # Merge description if legacy has better one
        legacy_desc = r["legacy_desc"] or ""
        epoch_desc = r["epoch_desc"] or ""
        if legacy_desc and (
            not epoch_desc
            or epoch_desc.lower() == "none"
            or len(legacy_desc) > len(epoch_desc)
        ):
            updates["description"] = legacy_desc
            stats["descriptions_merged"] += 1

        # Merge physics_domain if missing
        if r["legacy_domain"] and not r["epoch_domain"]:
            updates["physics_domain"] = r["legacy_domain"]
            stats["physics_domains_merged"] += 1

        # Merge units if epoch has generic 'dimensionless'
        legacy_units = r["legacy_units"] or ""
        epoch_units = r["epoch_units"] or ""
        if (
            legacy_units
            and legacy_units != "dimensionless"
            and (not epoch_units or epoch_units == "dimensionless")
        ):
            updates["units"] = legacy_units
            stats["units_merged"] += 1

        if updates:
            # Apply updates to epoch node - normalize paths for matching
            set_clauses = ", ".join(f"n.{k} = ${k}" for k in updates)
            norm_path = r["epoch_path"].upper().replace("\\\\", "\\")
            client.query(
                f"""
                MATCH (n:TreeNode)
                WHERE toUpper(replace(n.path, $double_bs, $single_bs)) = $norm_path
                  AND n.first_shot IS NOT NULL
                  AND n.tree_name = $tree_name
                SET {set_clauses}
                """,
                norm_path=norm_path,
                tree_name=tree_name,
                double_bs=double_bs,
                single_bs=single_bs,
                **updates,
            )

    logger.info(
        f"Merged metadata: {stats['descriptions_merged']} descriptions, "
        f"{stats['physics_domains_merged']} physics_domains, "
        f"{stats['units_merged']} units"
    )
    return stats


def cleanup_legacy_nodes(
    client: "GraphClient",
    facility: str,
    tree_name: str,
    dry_run: bool = False,
) -> dict:
    """Remove legacy nodes that have been superseded by epoch-aware nodes.

    Only removes legacy nodes (first_shot IS NULL) that have a matching
    epoch-aware node with the same normalized path. Preserves legacy nodes
    that don't have epoch equivalents.

    Should be called AFTER merge_legacy_metadata() to preserve valuable metadata.

    Args:
        client: Neo4j GraphClient
        facility: Facility ID
        tree_name: MDSplus tree name
        dry_run: If True, log but don't delete

    Returns:
        Dict with cleanup statistics
    """
    # Backslash strings for normalization
    double_bs = "\\\\"  # Two backslash chars
    single_bs = "\\"  # One backslash char

    # Count legacy nodes that have epoch equivalents
    # Normalize paths by removing extra backslashes
    result = client.query(
        """
        MATCH (legacy:TreeNode)
        WHERE legacy.first_shot IS NULL
          AND legacy.tree_name = $tree_name
        WITH legacy,
             toUpper(replace(legacy.path, $double_bs, $single_bs)) AS norm_legacy
        OPTIONAL MATCH (epoch:TreeNode)
        WHERE epoch.first_shot IS NOT NULL
          AND epoch.tree_name = $tree_name
          AND toUpper(replace(epoch.path, $double_bs, $single_bs)) = norm_legacy
        RETURN legacy.path AS path,
               epoch IS NOT NULL AS has_epoch
        """,
        tree_name=tree_name,
        double_bs=double_bs,
        single_bs=single_bs,
    )

    to_delete = [r["path"] for r in result if r["has_epoch"]]
    to_keep = [r["path"] for r in result if not r["has_epoch"]]

    stats = {
        "total_legacy": len(result),
        "to_delete": len(to_delete),
        "to_keep": len(to_keep),
        "deleted": 0,
    }

    logger.info(
        f"Legacy nodes for {tree_name}: {len(to_delete)} to delete, "
        f"{len(to_keep)} to keep (no epoch equivalent)"
    )

    if dry_run:
        for path in to_delete[:5]:
            logger.info(f"  [DRY RUN] Would delete: {path}")
        if len(to_delete) > 5:
            logger.info(f"  ... and {len(to_delete) - 5} more")
        return stats

    if to_delete:
        # Delete legacy nodes that have epoch equivalents
        client.query(
            """
            MATCH (legacy:TreeNode)
            WHERE legacy.first_shot IS NULL
              AND legacy.tree_name = $tree_name
            WITH legacy,
                 toUpper(replace(legacy.path, $double_bs, $single_bs)) AS norm_legacy
            MATCH (epoch:TreeNode)
            WHERE epoch.first_shot IS NOT NULL
              AND epoch.tree_name = $tree_name
              AND toUpper(replace(epoch.path, $double_bs, $single_bs)) = norm_legacy
            DETACH DELETE legacy
            """,
            tree_name=tree_name,
            double_bs=double_bs,
            single_bs=single_bs,
        )
        stats["deleted"] = len(to_delete)
        logger.info(f"Deleted {len(to_delete)} superseded legacy nodes")

    return stats
