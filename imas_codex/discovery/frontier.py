"""
Frontier management for graph-led discovery.

The frontier is the set of FacilityPath nodes that are ready for scanning
(status='pending') or ready for expansion (expand_to > depth). This module
provides queries and utilities for managing the frontier.

Key concepts:
    - Frontier: Paths awaiting scan (status='pending' or expand_to > depth)
    - Coverage: Fraction of known paths that are scored
    - Seeding: Creating initial root paths for a facility
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryStats:
    """Statistics about discovery progress for a facility."""

    facility: str
    total: int = 0
    pending: int = 0
    scanned: int = 0
    scored: int = 0
    skipped: int = 0

    @property
    def frontier_size(self) -> int:
        """Number of paths awaiting scan."""
        return self.pending

    @property
    def coverage(self) -> float:
        """Fraction of known paths that are scored."""
        if self.total == 0:
            return 0.0
        return self.scored / self.total


def get_discovery_stats(facility: str) -> dict[str, Any]:
    """Get discovery statistics for a facility.

    Returns:
        Dict with counts: total, pending, scanned, scored, skipped
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            RETURN
                count(p) AS total,
                sum(CASE WHEN p.status = 'pending' THEN 1 ELSE 0 END) AS pending,
                sum(CASE WHEN p.status = 'scanned' THEN 1 ELSE 0 END) AS scanned,
                sum(CASE WHEN p.status = 'scored' THEN 1 ELSE 0 END) AS scored,
                sum(CASE WHEN p.status = 'skipped' THEN 1 ELSE 0 END) AS skipped
            """,
            facility=facility,
        )

        if result:
            return {
                "total": result[0]["total"],
                "pending": result[0]["pending"],
                "scanned": result[0]["scanned"],
                "scored": result[0]["scored"],
                "skipped": result[0]["skipped"],
            }

        return {"total": 0, "pending": 0, "scanned": 0, "scored": 0, "skipped": 0}


def get_frontier(
    facility: str,
    limit: int = 100,
    include_expansions: bool = True,
) -> list[dict[str, Any]]:
    """Get paths in the frontier (awaiting scan).

    Frontier includes:
    1. Paths with status='pending' (newly seeded or from parent expansion)
    2. Paths where expand_to > depth (marked for expansion by scorer)

    Args:
        facility: Facility ID
        limit: Maximum paths to return
        include_expansions: Include paths marked for expansion (expand_to > depth)

    Returns:
        List of dicts with path info: id, path, depth, status, parent_path_id
    """
    from imas_codex.graph import GraphClient

    if include_expansions:
        where_clause = "(p.status = 'pending' OR (p.expand_to IS NOT NULL AND p.expand_to > p.depth))"
    else:
        where_clause = "p.status = 'pending'"

    with GraphClient() as gc:
        result = gc.query(
            f"""
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {{id: $facility}})
            WHERE {where_clause}
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.status AS status, p.parent_path_id AS parent_path_id
            ORDER BY p.depth ASC, p.path ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
        )

        return list(result)


def get_scorable_paths(facility: str, limit: int = 100) -> list[dict[str, Any]]:
    """Get paths that are ready for scoring (scanned but not scored).

    Returns:
        List of dicts with path info, DirStats, and child_names for LLM context
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.status = 'scanned' AND p.score IS NULL
            RETURN p.id AS id, p.path AS path, p.depth AS depth,
                   p.total_files AS total_files, p.total_dirs AS total_dirs,
                   p.file_type_counts AS file_type_counts,
                   p.has_readme AS has_readme, p.has_makefile AS has_makefile,
                   p.has_git AS has_git, p.patterns_detected AS patterns_detected,
                   p.description AS description, p.child_names AS child_names
            ORDER BY p.depth ASC, p.path ASC
            LIMIT $limit
            """,
            facility=facility,
            limit=limit,
        )

        return list(result)


def seed_facility_roots(
    facility: str,
    root_paths: list[str] | None = None,
) -> int:
    """Create initial FacilityPath nodes for discovery.

    If no root paths provided, uses facility config's paths.actionable_paths
    or falls back to common root paths.

    Args:
        facility: Facility ID
        root_paths: Optional list of root paths to seed

    Returns:
        Number of paths created
    """
    from imas_codex.discovery import get_facility
    from imas_codex.graph import GraphClient

    # Get root paths from config if not provided
    if root_paths is None:
        config = get_facility(facility)
        paths_config = config.get("paths", {})

        # Look for actionable_paths or use defaults
        if isinstance(paths_config, dict):
            root_paths = paths_config.get("actionable_paths", [])
            if isinstance(root_paths, list) and root_paths:
                root_paths = [
                    p.get("path") if isinstance(p, dict) else p for p in root_paths
                ]
            else:
                # Fallback to common exploration roots
                root_paths = ["/home", "/work", "/opt"]
        else:
            root_paths = ["/home", "/work", "/opt"]

    now = datetime.now(UTC).isoformat()
    items = []

    for path in root_paths:
        path_id = f"{facility}:{path}"
        items.append(
            {
                "id": path_id,
                "facility_id": facility,
                "path": path,
                "path_type": "code_directory",  # Default, will be updated during scan
                "status": "pending",
                "depth": 0,
                "discovered_at": now,
            }
        )

    with GraphClient() as gc:
        # First ensure facility exists
        existing = gc.get_facility(facility)
        if not existing:
            # Create minimal facility node
            gc.create_facility(facility, name=facility)

        result = gc.create_nodes("FacilityPath", items)

    logger.info(f"Seeded {result['processed']} root paths for {facility}")
    return result["processed"]


def create_child_paths(
    facility: str,
    parent_path: str,
    child_paths: list[str],
) -> int:
    """Create child FacilityPath nodes from a parent.

    Args:
        facility: Facility ID
        parent_path: Parent path string
        child_paths: List of child path strings

    Returns:
        Number of paths created
    """
    from imas_codex.graph import GraphClient

    parent_id = f"{facility}:{parent_path}"
    parent_depth_result = None

    with GraphClient() as gc:
        # Get parent depth
        result = gc.query(
            "MATCH (p:FacilityPath {id: $id}) RETURN p.depth AS depth",
            id=parent_id,
        )
        parent_depth_result = result[0]["depth"] if result else 0

    parent_depth = parent_depth_result or 0
    now = datetime.now(UTC).isoformat()
    items = []

    for child_path in child_paths:
        child_id = f"{facility}:{child_path}"
        items.append(
            {
                "id": child_id,
                "facility_id": facility,
                "path": child_path,
                "path_type": "code_directory",
                "status": "pending",
                "depth": parent_depth + 1,
                "parent_path_id": parent_id,
                "discovered_at": now,
            }
        )

    if not items:
        return 0

    with GraphClient() as gc:
        result = gc.create_nodes("FacilityPath", items)

        # Create PARENT relationships
        gc.query(
            """
            UNWIND $children AS child
            MATCH (c:FacilityPath {id: child.id})
            MATCH (p:FacilityPath {id: child.parent_path_id})
            MERGE (c)-[:PARENT]->(p)
            """,
            children=items,
        )

    return result["processed"]


def mark_path_scanned(
    facility: str,
    path: str,
    stats: dict[str, Any],
) -> None:
    """Update path with scan results and mark as scanned.

    Args:
        facility: Facility ID
        path: Path string
        stats: DirStats dict with file_type_counts, total_files, etc.
    """
    from imas_codex.graph import GraphClient

    path_id = f"{facility}:{path}"
    now = datetime.now(UTC).isoformat()

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath {id: $id})
            SET p.status = 'scanned',
                p.scanned_at = $now,
                p.total_files = $total_files,
                p.total_dirs = $total_dirs,
                p.total_size_bytes = $total_size_bytes,
                p.size_skipped = $size_skipped,
                p.file_type_counts = $file_type_counts,
                p.has_readme = $has_readme,
                p.has_makefile = $has_makefile,
                p.has_git = $has_git,
                p.patterns_detected = $patterns_detected
            """,
            id=path_id,
            now=now,
            total_files=stats.get("total_files", 0),
            total_dirs=stats.get("total_dirs", 0),
            total_size_bytes=stats.get("total_size_bytes"),
            size_skipped=stats.get("size_skipped", False),
            file_type_counts=stats.get("file_type_counts", "{}"),
            has_readme=stats.get("has_readme", False),
            has_makefile=stats.get("has_makefile", False),
            has_git=stats.get("has_git", False),
            patterns_detected=stats.get("patterns_detected", []),
        )


def mark_paths_scored(
    facility: str,
    scores: list[dict[str, Any]],
) -> int:
    """Update multiple paths with LLM scores.

    Args:
        facility: Facility ID
        scores: List of dicts with path, score, score_code, score_data, score_imas,
                description, path_purpose, evidence, should_expand

    Returns:
        Number of paths updated
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    updated = 0

    with GraphClient() as gc:
        for score_data in scores:
            path = score_data["path"]
            path_id = f"{facility}:{path}"

            # Determine expand_to based on should_expand
            expand_to = None
            if score_data.get("should_expand"):
                # Get current depth and set expand_to = depth + 1
                result = gc.query(
                    "MATCH (p:FacilityPath {id: $id}) RETURN p.depth AS depth",
                    id=path_id,
                )
                if result:
                    expand_to = (result[0]["depth"] or 0) + 1

            gc.query(
                """
                MATCH (p:FacilityPath {id: $id})
                SET p.status = 'scored',
                    p.scored_at = $now,
                    p.score = $score,
                    p.score_code = $score_code,
                    p.score_data = $score_data,
                    p.score_imas = $score_imas,
                    p.description = $description,
                    p.path_purpose = $path_purpose,
                    p.evidence = $evidence,
                    p.expand_to = $expand_to
                """,
                id=path_id,
                now=now,
                score=score_data.get("score"),
                score_code=score_data.get("score_code"),
                score_data=score_data.get(
                    "score_data_score"
                ),  # Avoid conflict with params
                score_imas=score_data.get("score_imas"),
                description=score_data.get("description"),
                path_purpose=score_data.get("path_purpose"),
                evidence=score_data.get("evidence"),
                expand_to=expand_to,
            )
            updated += 1

    return updated


def mark_path_skipped(
    facility: str,
    path: str,
    reason: str,
) -> None:
    """Mark a path as skipped with a reason.

    Args:
        facility: Facility ID
        path: Path string
        reason: Skip reason
    """
    from imas_codex.graph import GraphClient

    path_id = f"{facility}:{path}"
    now = datetime.now(UTC).isoformat()

    with GraphClient() as gc:
        gc.query(
            """
            MATCH (p:FacilityPath {id: $id})
            SET p.status = 'skipped',
                p.skipped_at = $now,
                p.skip_reason = $reason
            """,
            id=path_id,
            now=now,
            reason=reason,
        )


def get_high_value_paths(
    facility: str,
    min_score: float = 0.7,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Get scored paths above a threshold.

    Args:
        facility: Facility ID
        min_score: Minimum score threshold
        limit: Maximum paths to return

    Returns:
        List of dicts with path info and scores
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE p.score >= $min_score
            RETURN p.id AS id, p.path AS path, p.score AS score,
                   p.description AS description, p.path_purpose AS path_purpose,
                   p.score_code AS score_code, p.score_data AS score_data,
                   p.score_imas AS score_imas
            ORDER BY p.score DESC
            LIMIT $limit
            """,
            facility=facility,
            min_score=min_score,
            limit=limit,
        )

        return list(result)


def clear_facility_paths(facility: str) -> int:
    """Delete all FacilityPath nodes for a facility.

    Use this before starting a fresh discovery.

    Args:
        facility: Facility ID

    Returns:
        Number of paths deleted
    """
    from imas_codex.graph import GraphClient

    with GraphClient() as gc:
        result = gc.query(
            """
            MATCH (p:FacilityPath)-[:FACILITY_ID]->(f:Facility {id: $facility})
            DETACH DELETE p
            RETURN count(p) AS deleted
            """,
            facility=facility,
        )

        return result[0]["deleted"] if result else 0


def persist_scan_results(
    facility: str,
    results: list[tuple[str, dict, list[str], str | None]],
    excluded: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Persist multiple scan results in a single transaction.

    Much faster than calling mark_path_scanned/create_child_paths per path.

    Args:
        facility: Facility ID
        results: List of (path, stats_dict, child_dirs, error) tuples
        excluded: Optional list of (path, parent_path, reason) for excluded dirs

    Returns:
        Dict with scanned, children_created, excluded, errors counts
    """
    from imas_codex.graph import GraphClient

    now = datetime.now(UTC).isoformat()
    scanned = 0
    children_created = 0
    excluded_count = 0
    errors = 0

    # Separate successes and errors
    successes = []
    all_children = []

    for path, stats, child_dirs, error in results:
        path_id = f"{facility}:{path}"
        if error:
            errors += 1
            # Mark as skipped
            successes.append(
                {
                    "id": path_id,
                    "status": "skipped",
                    "skip_reason": error,
                    "scanned_at": now,
                }
            )
        else:
            scanned += 1
            # Build update dict with child_names if available
            update_dict = {
                "id": path_id,
                "status": "scanned",
                "scanned_at": now,
                "total_files": stats.get("total_files", 0),
                "total_dirs": stats.get("total_dirs", 0),
                "has_readme": stats.get("has_readme", False),
                "has_makefile": stats.get("has_makefile", False),
                "has_git": stats.get("has_git", False),
            }
            # Store file_type_counts if available
            file_type_counts = stats.get("file_type_counts")
            if file_type_counts:
                import json

                if isinstance(file_type_counts, dict):
                    update_dict["file_type_counts"] = json.dumps(file_type_counts)
                else:
                    update_dict["file_type_counts"] = file_type_counts
            # Store child names for LLM scoring context (as JSON string)
            child_names = stats.get("child_names")
            if child_names:
                import json

                update_dict["child_names"] = json.dumps(child_names)
            # Store patterns detected
            patterns = stats.get("patterns_detected")
            if patterns:
                update_dict["patterns_detected"] = patterns
            successes.append(update_dict)
            # Prepare children
            for child_path in child_dirs:
                all_children.append(
                    {
                        "id": f"{facility}:{child_path}",
                        "facility_id": facility,
                        "path": child_path,
                        "parent_id": path_id,
                    }
                )

    with GraphClient() as gc:
        # Batch update scanned/skipped paths
        if successes:
            gc.query(
                """
                UNWIND $items AS item
                MATCH (p:FacilityPath {id: item.id})
                SET p.status = item.status,
                    p.scanned_at = item.scanned_at,
                    p.skip_reason = item.skip_reason,
                    p.total_files = item.total_files,
                    p.total_dirs = item.total_dirs,
                    p.has_readme = item.has_readme,
                    p.has_makefile = item.has_makefile,
                    p.has_git = item.has_git,
                    p.child_names = item.child_names
                """,
                items=successes,
            )

        # Batch create children
        if all_children:
            # First get parent depths
            parent_ids = list({c["parent_id"] for c in all_children})
            depth_result = gc.query(
                """
                UNWIND $ids AS id
                MATCH (p:FacilityPath {id: id})
                RETURN p.id AS id, p.depth AS depth
                """,
                ids=parent_ids,
            )
            depth_map = {r["id"]: r["depth"] or 0 for r in depth_result}

            # Add depth to children
            for child in all_children:
                child["depth"] = depth_map.get(child["parent_id"], 0) + 1
                child["status"] = "pending"
                child["path_type"] = "code_directory"
                child["discovered_at"] = now

            # Create child nodes
            gc.query(
                """
                UNWIND $children AS child
                MERGE (c:FacilityPath {id: child.id})
                ON CREATE SET c.facility_id = child.facility_id,
                              c.path = child.path,
                              c.path_type = child.path_type,
                              c.status = child.status,
                              c.depth = child.depth,
                              c.parent_path_id = child.parent_id,
                              c.discovered_at = child.discovered_at
                """,
                children=all_children,
            )

            # Create FACILITY_ID relationships
            gc.query(
                """
                UNWIND $children AS child
                MATCH (c:FacilityPath {id: child.id})
                MATCH (f:Facility {id: child.facility_id})
                MERGE (c)-[:FACILITY_ID]->(f)
                """,
                children=all_children,
            )

            # Create PARENT relationships
            gc.query(
                """
                UNWIND $children AS child
                MATCH (c:FacilityPath {id: child.id})
                MATCH (p:FacilityPath {id: child.parent_id})
                MERGE (c)-[:PARENT]->(p)
                """,
                children=all_children,
            )

            children_created = len(all_children)

        # Handle excluded directories (create with status='excluded')
        if excluded:
            excluded_nodes = []
            for path, parent_path, reason in excluded:
                parent_id = f"{facility}:{parent_path}"
                # Get parent depth
                depth_result = gc.query(
                    "MATCH (p:FacilityPath {id: $id}) RETURN p.depth AS depth",
                    id=parent_id,
                )
                parent_depth = depth_result[0]["depth"] if depth_result else 0

                excluded_nodes.append(
                    {
                        "id": f"{facility}:{path}",
                        "facility_id": facility,
                        "path": path,
                        "parent_id": parent_id,
                        "depth": (parent_depth or 0) + 1,
                        "status": "excluded",
                        "skip_reason": reason,
                        "discovered_at": now,
                    }
                )

            if excluded_nodes:
                gc.query(
                    """
                    UNWIND $nodes AS node
                    MERGE (p:FacilityPath {id: node.id})
                    ON CREATE SET p.facility_id = node.facility_id,
                                  p.path = node.path,
                                  p.status = node.status,
                                  p.skip_reason = node.skip_reason,
                                  p.depth = node.depth,
                                  p.parent_path_id = node.parent_id,
                                  p.discovered_at = node.discovered_at
                    """,
                    nodes=excluded_nodes,
                )

                # Create relationships
                gc.query(
                    """
                    UNWIND $nodes AS node
                    MATCH (c:FacilityPath {id: node.id})
                    MATCH (f:Facility {id: node.facility_id})
                    MERGE (c)-[:FACILITY_ID]->(f)
                    """,
                    nodes=excluded_nodes,
                )

                excluded_count = len(excluded_nodes)

    return {
        "scanned": scanned,
        "children_created": children_created,
        "excluded": excluded_count,
        "errors": errors,
    }
