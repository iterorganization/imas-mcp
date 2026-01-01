"""MDSplus tree structure ingestion to Neo4j.

Creates TreeModelVersion and TreeNode entities with proper relationships.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def ingest_epochs(
    client: "GraphClient",
    epochs: list[dict],
    dry_run: bool = False,
) -> int:
    """Ingest TreeModelVersion nodes to the graph.

    Args:
        client: Neo4j GraphClient
        epochs: List of epoch dicts from discover_epochs()
        dry_run: If True, log but don't write

    Returns:
        Number of epochs ingested
    """
    if dry_run:
        for epoch in epochs:
            logger.info(
                f"[DRY RUN] Would create TreeModelVersion: {epoch['id']} "
                f"(v{epoch['version']}, shots {epoch['first_shot']}-{epoch.get('last_shot', 'current')})"
            )
        return len(epochs)

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
