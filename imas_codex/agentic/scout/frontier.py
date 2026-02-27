"""Frontier tracking for scout exploration.

Implements the "moving frontier" concept:
- Track which paths have been explored vs remaining
- Calculate exploration progress and coverage
- Identify high-priority unexplored paths
"""

import logging
from dataclasses import dataclass

from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


@dataclass
class FrontierStats:
    """Statistics about the exploration frontier."""

    facility: str
    total_paths: int = 0
    discovered: int = 0
    scanned: int = 0
    scored: int = 0
    skipped: int = 0
    queued_files: int = 0
    ingested_files: int = 0

    @property
    def explored(self) -> int:
        """Paths that have been explored (any status beyond discovered)."""
        return self.scanned + self.scored + self.skipped

    @property
    def remaining(self) -> int:
        """Paths still waiting to be explored."""
        return self.discovered

    @property
    def coverage(self) -> float:
        """Fraction of paths that have been explored (0.0-1.0)."""
        if self.total_paths == 0:
            return 0.0
        return self.explored / self.total_paths

    def summary(self) -> str:
        """Human-readable summary of frontier state."""
        return f"""Frontier Status for {self.facility}:
  Total paths: {self.total_paths}
  Explored: {self.explored} ({self.coverage:.1%})
  Remaining: {self.remaining}
  Skipped (dead-ends): {self.skipped}
  Files queued: {self.queued_files}
  Files ingested: {self.ingested_files}

By status:
  discovered: {self.discovered}
  scanned: {self.scanned}
  scored: {self.scored}
"""


class FrontierManager:
    """Manages the exploration frontier for a facility.

    Tracks which paths have been explored and which remain.
    Provides methods to get high-priority unexplored paths.
    """

    def __init__(self, facility: str) -> None:
        self.facility = facility

    def get_stats(self) -> FrontierStats:
        """Get current frontier statistics from the graph."""
        stats = FrontierStats(facility=self.facility)

        try:
            with GraphClient() as client:
                # Count FacilityPath nodes by status
                result = client.query(
                    """
                    MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                    RETURN p.status AS status, count(*) AS count
                    """,
                    facility=self.facility,
                )
                for row in result:
                    status = row["status"] or "discovered"
                    count = row["count"]
                    if status == "discovered":
                        stats.discovered = count
                    elif status == "scanned":
                        stats.scanned = count
                    elif status == "scored":
                        stats.scored = count
                    elif status in ("skipped", "failed"):
                        stats.skipped = count

                stats.total_paths = (
                    stats.discovered + stats.scanned + stats.scored + stats.skipped
                )

                # Count CodeFile nodes
                file_result = client.query(
                    """
                    MATCH (sf:CodeFile)-[:AT_FACILITY]->(f:Facility {id: $facility})
                    RETURN sf.status AS status, count(*) AS count
                    """,
                    facility=self.facility,
                )
                for row in file_result:
                    status = row["status"] or "discovered"
                    count = row["count"]
                    if status in ("discovered", "queued"):
                        stats.queued_files += count
                    elif status == "ingested":
                        stats.ingested_files += count

        except Exception as e:
            logger.exception("Failed to get frontier stats: %s", e)

        return stats

    def get_unexplored_paths(
        self,
        limit: int = 20,
        min_interest_score: float = 0.0,
    ) -> list[dict]:
        """Get high-priority unexplored paths.

        Returns paths with status='discovered' ordered by interest_score.
        These are the next candidates for exploration.
        """
        try:
            with GraphClient() as client:
                result = client.query(
                    """
                    MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                    WHERE p.status = 'discovered'
                      AND coalesce(p.interest_score, 0.5) >= $min_score
                    RETURN p.id AS id,
                           p.path AS path,
                           p.interest_score AS interest_score,
                           p.discovered_at AS discovered_at
                    ORDER BY p.interest_score DESC
                    LIMIT $limit
                    """,
                    facility=self.facility,
                    min_score=min_interest_score,
                    limit=limit,
                )
                return list(result)
        except Exception as e:
            logger.exception("Failed to get unexplored paths: %s", e)
            return []

    def get_explored_paths(
        self,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict]:
        """Get recently explored paths.

        Args:
            limit: Maximum number of paths to return
            status: Filter by specific status (listed, scanned, analyzed)

        Returns:
            List of explored path dicts
        """
        try:
            with GraphClient() as client:
                if status:
                    result = client.query(
                        """
                        MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                        WHERE p.status = $status
                        RETURN p.id AS id,
                               p.path AS path,
                               p.status AS status,
                               p.interest_score AS interest_score
                        ORDER BY p.interest_score DESC
                        LIMIT $limit
                        """,
                        facility=self.facility,
                        status=status,
                        limit=limit,
                    )
                else:
                    result = client.query(
                        """
                        MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                        WHERE p.status IN ['scanned', 'scored']
                        RETURN p.id AS id,
                               p.path AS path,
                               p.status AS status,
                               p.interest_score AS interest_score
                        ORDER BY p.interest_score DESC
                        LIMIT $limit
                        """,
                        facility=self.facility,
                        limit=limit,
                    )
                return list(result)
        except Exception as e:
            logger.exception("Failed to get explored paths: %s", e)
            return []

    def mark_path_status(
        self,
        path: str,
        status: str,
        interest_score: float | None = None,
        **extra: str,
    ) -> None:
        """Update the status of a path in the frontier.

        Status lifecycle: discovered -> listed -> scanned -> analyzed
        Or: discovered -> skipped (for dead-ends)
        """
        path_id = f"{self.facility}:{path}"
        try:
            with GraphClient() as client:
                props = {"status": status}
                if interest_score is not None:
                    props["interest_score"] = interest_score
                props.update(extra)

                client.query(
                    """
                    MATCH (p:FacilityPath {id: $id})
                    SET p += $props
                    """,
                    id=path_id,
                    props=props,
                )
        except Exception as e:
            logger.exception("Failed to update path status: %s", e)

    def get_skipped_paths(self, limit: int = 50) -> list[dict]:
        """Get paths that were skipped as dead-ends."""
        try:
            with GraphClient() as client:
                result = client.query(
                    """
                    MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                    WHERE p.status = 'skipped'
                    RETURN p.id AS id,
                           p.path AS path,
                           p.skip_reason AS reason,
                           p.skipped_at AS skipped_at
                    ORDER BY p.skipped_at DESC
                    LIMIT $limit
                    """,
                    facility=self.facility,
                    limit=limit,
                )
                return list(result)
        except Exception as e:
            logger.exception("Failed to get skipped paths: %s", e)
            return []

    def reset_frontier(self, confirm: bool = False) -> int:
        """Reset all paths back to discovered status.

        WARNING: This is destructive! Only use for testing/development.

        Args:
            confirm: Must be True to actually reset

        Returns:
            Number of paths reset
        """
        if not confirm:
            logger.warning("reset_frontier called without confirm=True, skipping")
            return 0

        try:
            with GraphClient() as client:
                result = client.query(
                    """
                    MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
                    WHERE p.status <> 'discovered'
                    SET p.status = 'discovered'
                    RETURN count(*) AS reset_count
                    """,
                    facility=self.facility,
                )
                rows = list(result)
                count = rows[0]["reset_count"] if rows else 0
                logger.info("Reset %d paths to discovered status", count)
                return count
        except Exception as e:
            logger.exception("Failed to reset frontier: %s", e)
            return 0


def get_facility_frontier_summary(facility: str) -> str:
    """Get a formatted frontier summary for a facility."""
    manager = FrontierManager(facility)
    stats = manager.get_stats()
    return stats.summary()
