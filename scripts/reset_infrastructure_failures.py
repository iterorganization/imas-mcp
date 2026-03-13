#!/usr/bin/env python3
"""Reset signals that failed checks due to infrastructure bugs.

After fixing static scanner routing (G.1) and JPF path resolution (G.2),
run this script to reset infrastructure-failed signals so they can be
re-checked with the correct scanner routing.

Usage:
    uv run python scripts/reset_infrastructure_failures.py [--facility jet] [--dry-run]
"""

from __future__ import annotations

import argparse
import logging

from imas_codex.discovery.signals.parallel import INFRASTRUCTURE_CHECK_ERRORS
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


def reset_infrastructure_failures(
    facility: str = "jet",
    dry_run: bool = False,
) -> int:
    """Reset signals that failed due to infrastructure bugs.

    Removes CHECKED_WITH relationships where the error_type is an
    infrastructure error, and resets the signal status to 'enriched'
    so they can be re-claimed for checking.

    Returns the number of signals reset.
    """
    error_types = sorted(INFRASTRUCTURE_CHECK_ERRORS)

    with GraphClient() as gc:
        # First, count affected signals
        count_result = gc.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})-[c:CHECKED_WITH]->(da:DataAccess)
            WHERE c.success = false
              AND c.error_type IN $error_types
            RETURN count(s) AS count
            """,
            facility=facility,
            error_types=error_types,
        )
        count = count_result[0]["count"] if count_result else 0
        print(f"Found {count} signals with infrastructure check failures")
        print(f"  Error types: {error_types}")

        if dry_run:
            # Show breakdown by error type
            breakdown = gc.query(
                """
                MATCH (s:FacilitySignal {facility_id: $facility})-[c:CHECKED_WITH]->(da:DataAccess)
                WHERE c.success = false
                  AND c.error_type IN $error_types
                RETURN c.error_type AS error_type, count(s) AS count
                ORDER BY count DESC
                """,
                facility=facility,
                error_types=error_types,
            )
            for row in breakdown:
                print(f"  {row['error_type']}: {row['count']}")
            print("\nDry run — no changes made. Remove --dry-run to reset.")
            return 0

        if count == 0:
            print("No signals to reset.")
            return 0

        # Reset signals: delete CHECKED_WITH, clear claimed_at, set status
        gc.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})-[c:CHECKED_WITH]->(da:DataAccess)
            WHERE c.success = false
              AND c.error_type IN $error_types
            SET s.status = 'enriched',
                s.claimed_at = null
            DELETE c
            """,
            facility=facility,
            error_types=error_types,
        )

        print(f"Reset {count} signals to 'enriched' status.")
        print(f"Re-run checks with: uv run imas-codex discover signals {facility} --check-only")
        return count


def main():
    parser = argparse.ArgumentParser(
        description="Reset signals with infrastructure check failures"
    )
    parser.add_argument(
        "--facility", default="jet", help="Facility ID (default: jet)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be reset without making changes",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    reset_infrastructure_failures(facility=args.facility, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
