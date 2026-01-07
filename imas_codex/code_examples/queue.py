"""Source file discovery for the ingestion pipeline.

Scouts use queue_source_files() to mark files as discovered for ingestion.
The CLI ingest command processes discovered files.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from imas_codex.graph import GraphClient

from .facility_reader import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)


@dataclass
class QueuedFile:
    """A file queued for ingestion."""

    path: str
    facility_id: str
    language: str
    interest_score: float = 0.5
    patterns_matched: list[str] | None = None
    parent_path_id: str | None = None
    discovered_by: str | None = None


def _generate_source_file_id(facility: str, path: str) -> str:
    """Generate unique ID for a SourceFile node."""
    return f"{facility}:{path}"


def _detect_language(path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(path).suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext, "python")


def queue_source_files(
    facility: str,
    file_paths: list[str],
    interest_score: float = 0.5,
    patterns_matched: list[str] | None = None,
    parent_path_id: str | None = None,
    discovered_by: str | None = None,
) -> dict[str, int]:
    """Discover source files for ingestion.

    Creates SourceFile nodes with status='discovered'. Files already in
    discovered/ingested status are skipped (idempotent).

    Args:
        facility: Facility ID (e.g., "epfl")
        file_paths: List of remote file paths to discover
        interest_score: Priority score (0.0-1.0, higher = sooner)
        patterns_matched: Patterns that matched (e.g., ["IMAS", "equilibrium"])
        parent_path_id: FacilityPath that contains these files
        discovered_by: Scout session identifier

    Returns:
        Dict with counts: {"discovered": N, "skipped": K, "errors": [...]}
    """
    if not file_paths:
        return {"discovered": 0, "skipped": 0, "errors": []}

    now = datetime.now(UTC).isoformat()
    items = []
    for path in file_paths:
        items.append(
            {
                "id": _generate_source_file_id(facility, path),
                "facility_id": facility,
                "path": path,
                "language": _detect_language(path),
                "status": "discovered",
                "interest_score": interest_score,
                "patterns_matched": patterns_matched or [],
                "parent_path_id": parent_path_id,
                "discovered_by": discovered_by,
                "discovered_at": now,
            }
        )

    try:
        with GraphClient() as client:
            # Check for existing SourceFile or CodeExample nodes
            # Skip if already queued/ready or already has a CodeExample
            existing = client.query(
                """
                UNWIND $items AS item
                OPTIONAL MATCH (sf:SourceFile {id: item.id})
                OPTIONAL MATCH (ce:CodeExample {source_file: item.path, facility_id: item.facility_id})
                RETURN item.id AS id,
                       sf.status AS sf_status,
                       ce.id AS ce_id
                """,
                items=items,
            )

            # Filter out existing files
            skip_ids = set()
            for row in existing:
                if row["sf_status"] in ("discovered", "ingested"):
                    skip_ids.add(row["id"])
                elif row["ce_id"] is not None:
                    skip_ids.add(row["id"])

            to_discover = [item for item in items if item["id"] not in skip_ids]

            if not to_discover:
                return {"discovered": 0, "skipped": len(skip_ids), "errors": []}

            # Create SourceFile nodes with UNWIND
            client.query(
                """
                UNWIND $items AS item
                MERGE (sf:SourceFile {id: item.id})
                SET sf += item
                WITH sf, item
                MATCH (f:Facility {id: item.facility_id})
                MERGE (sf)-[:FACILITY_ID]->(f)
                """,
                items=to_discover,
            )

            # Link to parent FacilityPath if provided
            if parent_path_id:
                client.query(
                    """
                    MATCH (sf:SourceFile)
                    WHERE sf.parent_path_id = $parent_id
                    MATCH (p:FacilityPath {id: $parent_id})
                    MERGE (p)-[:CONTAINS]->(sf)
                    """,
                    parent_id=parent_path_id,
                )

            logger.info(
                "Discovered %d files for ingestion (skipped %d)",
                len(to_discover),
                len(skip_ids),
            )
            return {
                "discovered": len(to_discover),
                "skipped": len(skip_ids),
                "errors": [],
            }

    except Exception as e:
        logger.exception("Failed to discover source files")
        return {"discovered": 0, "skipped": 0, "errors": [str(e)]}


def get_pending_files(
    facility: str,
    limit: int = 100,
    min_interest_score: float = 0.0,
) -> list[dict]:
    """Get pending SourceFile nodes for ingestion.

    Returns files that need processing:
    - discovered: Not yet processed
    - failed: With retry count < 3 (will be retried)

    Ordered by interest_score descending.

    Args:
        facility: Facility ID
        limit: Maximum number of files to return
        min_interest_score: Minimum interest score threshold

    Returns:
        List of SourceFile dicts with id, path, language, interest_score
    """
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
            WHERE sf.status = 'discovered'
               OR (sf.status = 'failed' AND coalesce(sf.retry_count, 0) < 3)
            AND coalesce(sf.interest_score, 0.5) >= $min_score
            RETURN sf.id AS id, sf.path AS path, sf.language AS language,
                   sf.interest_score AS interest_score, sf.status AS status,
                   sf.retry_count AS retry_count
            ORDER BY sf.interest_score DESC, sf.discovered_at ASC
            LIMIT $limit
            """,
            facility=facility,
            min_score=min_interest_score,
            limit=limit,
        )
        return list(result)


def update_source_file_status(
    source_file_id: str,
    status: str,
    code_example_id: str | None = None,
    error: str | None = None,
) -> None:
    """Update the status of a SourceFile node.

    Args:
        source_file_id: SourceFile ID
        status: New status (discovered, ingested, failed, stale)
        code_example_id: CodeExample ID if status is 'ingested'
        error: Error message if status is 'failed'
    """
    now = datetime.now(UTC).isoformat()

    with GraphClient() as client:
        if status == "ingested" and code_example_id:
            client.query(
                """
                MATCH (sf:SourceFile {id: $id})
                SET sf.status = $status,
                    sf.completed_at = $now,
                    sf.code_example_id = $code_example_id,
                    sf.error = null
                WITH sf
                MATCH (ce:CodeExample {id: $code_example_id})
                MERGE (sf)-[:PRODUCED]->(ce)
                """,
                id=source_file_id,
                status=status,
                now=now,
                code_example_id=code_example_id,
            )
        elif status == "failed":
            client.query(
                """
                MATCH (sf:SourceFile {id: $id})
                SET sf.status = $status,
                    sf.error = $error,
                    sf.retry_count = coalesce(sf.retry_count, 0) + 1
                """,
                id=source_file_id,
                status=status,
                error=error,
            )
        else:
            client.query(
                """
                MATCH (sf:SourceFile {id: $id})
                SET sf.status = $status
                """,
                id=source_file_id,
                status=status,
            )


def get_queue_stats(facility: str) -> dict[str, int]:
    """Get queue statistics for a facility.

    Args:
        facility: Facility ID

    Returns:
        Dict with counts by status: {"queued": N, "fetching": M, ...}
    """
    with GraphClient() as client:
        result = client.query(
            """
            MATCH (sf:SourceFile)-[:FACILITY_ID]->(f:Facility {id: $facility})
            RETURN sf.status AS status, count(*) AS count
            """,
            facility=facility,
        )
        return {row["status"]: row["count"] for row in result}


__all__ = [
    "QueuedFile",
    "get_pending_files",
    "get_queue_stats",
    "queue_source_files",
    "update_source_file_status",
]
