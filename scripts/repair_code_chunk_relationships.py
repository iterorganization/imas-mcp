"""Repair orphaned CodeChunk relationships.

One-time repair script for CodeChunks that have a `code_example_id` property
but are missing the corresponding HAS_CHUNK and CODE_EXAMPLE_ID relationships.

Usage:
    uv run python scripts/repair_code_chunk_relationships.py
    uv run python scripts/repair_code_chunk_relationships.py --dry-run
"""

import argparse
import logging

from imas_codex.graph.client import GraphClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def diagnose(gc: GraphClient) -> dict[str, int]:
    """Run diagnostic queries and return counts."""
    stats: dict[str, int] = {}

    # Total CodeChunks with code_example_id property
    logger.info("  Counting CodeChunks with code_example_id...")
    result = gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
        RETURN count(cc) AS total
    """)
    stats["total_with_property"] = result[0]["total"] if result else 0
    logger.info("    total_with_property: %d", stats["total_with_property"])

    # Missing HAS_CHUNK relationships
    logger.info("  Counting missing HAS_CHUNK...")
    result = gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
          AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
        RETURN count(cc) AS missing
    """)
    stats["missing_has_chunk"] = result[0]["missing"] if result else 0
    logger.info("    missing_has_chunk: %d", stats["missing_has_chunk"])

    # Missing CODE_EXAMPLE_ID relationships
    logger.info("  Counting missing CODE_EXAMPLE_ID...")
    result = gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
          AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
        RETURN count(cc) AS missing
    """)
    stats["missing_code_example_id"] = result[0]["missing"] if result else 0
    logger.info("    missing_code_example_id: %d", stats["missing_code_example_id"])

    # Breakdown by facility (only for missing HAS_CHUNK)
    if stats["missing_has_chunk"] > 0:
        logger.info("  Breakdown by facility...")
        result = gc.query("""
            MATCH (cc:CodeChunk)
            WHERE cc.code_example_id IS NOT NULL
              AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
            RETURN cc.facility_id AS facility, count(cc) AS count
            ORDER BY count DESC
        """)
        for r in result or []:
            key = f"missing_has_chunk_{r['facility']}"
            stats[key] = r["count"]
            logger.info("    %s: %d", key, r["count"])

    return stats


def repair(gc: GraphClient, *, dry_run: bool = False) -> dict[str, int]:
    """Execute repair queries."""
    stats: dict[str, int] = {}

    if dry_run:
        logger.info("DRY RUN — no changes will be made")
        return diagnose(gc)

    # Step 1: Create missing HAS_CHUNK relationships
    result = gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
          AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
        MATCH (ce:CodeExample {id: cc.code_example_id})
        MERGE (ce)-[:HAS_CHUNK]->(cc)
        RETURN count(cc) AS repaired
    """)
    stats["has_chunk_created"] = result[0]["repaired"] if result else 0
    logger.info("Created %d HAS_CHUNK relationships", stats["has_chunk_created"])

    # Step 2: Create missing CODE_EXAMPLE_ID relationships
    result = gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
          AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
        MATCH (ce:CodeExample {id: cc.code_example_id})
        MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)
        RETURN count(cc) AS repaired
    """)
    stats["code_example_id_created"] = result[0]["repaired"] if result else 0
    logger.info(
        "Created %d CODE_EXAMPLE_ID relationships",
        stats["code_example_id_created"],
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair orphaned CodeChunk relationships"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Diagnose without making changes"
    )
    args = parser.parse_args()

    gc = GraphClient()

    logger.info("=== CodeChunk Relationship Diagnosis ===")
    diag = diagnose(gc)

    if (
        diag.get("missing_has_chunk", 0) == 0
        and diag.get("missing_code_example_id", 0) == 0
    ):
        logger.info("\nNo repairs needed.")
        return

    if args.dry_run:
        logger.info("\nDRY RUN — no changes made. Run without --dry-run to repair.")
        return

    logger.info("\n=== Executing Repairs ===")
    stats = repair(gc, dry_run=False)
    for key, value in stats.items():
        logger.info("  %s: %d", key, value)
    logger.info("\nDone. Re-run with --dry-run to verify.")


if __name__ == "__main__":
    main()
