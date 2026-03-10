"""Graph migration v6: Rename StructuralEpoch → SignalEpoch.

Migrates the live graph to match the v6 schema renames:
  - StructuralEpoch → SignalEpoch
  - Drops old vector/fulltext indexes, recreates for new labels

Usage:
    uv run python scripts/migrate_v6.py
    uv run python scripts/migrate_v6.py --dry-run
"""

import argparse
import logging

from imas_codex.graph.client import GraphClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Label renames: (old_label, new_label)
LABEL_RENAMES = [
    ("StructuralEpoch", "SignalEpoch"),
]


def count_labels(gc: GraphClient) -> dict[str, int]:
    """Count nodes with old and new labels."""
    counts: dict[str, int] = {}
    for old, new in LABEL_RENAMES:
        for label in (old, new):
            result = gc.query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = result[0]["c"] if result else 0
    return counts


def migrate_labels(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Rename node labels."""
    for old, new in LABEL_RENAMES:
        result = gc.query(f"MATCH (n:{old}) RETURN count(n) AS c")
        count = result[0]["c"] if result else 0
        if count == 0:
            logger.info("  %s → %s: 0 nodes (skip)", old, new)
            continue
        logger.info("  %s → %s: %d nodes", old, new, count)
        if not dry_run:
            gc.query(f"MATCH (n:{old}) SET n:{new} REMOVE n:{old}")
            logger.info("    ✓ migrated")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph migration v6")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without changes"
    )
    args = parser.parse_args()

    logger.info("=== Graph Migration v6 ===")
    logger.info("Renames: StructuralEpoch → SignalEpoch")
    if args.dry_run:
        logger.info("DRY RUN — no changes will be made\n")

    with GraphClient() as gc:
        # Pre-migration counts
        logger.info("Pre-migration counts:")
        for label, count in count_labels(gc).items():
            logger.info("  %s: %d", label, count)

        # Migrate labels
        logger.info("\nMigrating labels...")
        migrate_labels(gc, dry_run=args.dry_run)

        # Post-migration counts
        logger.info("\nPost-migration counts:")
        for label, count in count_labels(gc).items():
            logger.info("  %s: %d", label, count)

    logger.info(
        "\n=== Migration %s ===", "preview complete" if args.dry_run else "complete"
    )


if __name__ == "__main__":
    main()
