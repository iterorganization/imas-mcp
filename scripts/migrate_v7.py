"""Graph migration v7: Unit normalization + SignalSource rename.

Migrates the live graph:
  1. Rename SignalGroup label → SignalSource
  2. Rename USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE relationships
  3. Re-normalize all Unit node IDs from UDUNITS dot-exponential (m.s^-1)
     to pint short notation (m/s, A/m**2, kg*m/s**2)
  4. Backfill data_source_node and data_source_path properties
  5. Set enrichment_source for signals missing it
  6. Clean up legacy pattern properties

Usage:
    uv run python scripts/migrate_v7.py
    uv run python scripts/migrate_v7.py --dry-run
"""

import argparse
import logging

from imas_codex.graph.client import GraphClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def migrate_signal_source_labels(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Rename SignalGroup → SignalSource labels."""
    result = gc.query("MATCH (n:SignalGroup) RETURN count(n) AS c")
    count = result[0]["c"] if result else 0
    logger.info("  SignalGroup → SignalSource: %d nodes", count)
    if count > 0 and not dry_run:
        gc.query("MATCH (n:SignalGroup) SET n:SignalSource REMOVE n:SignalGroup")
        logger.info("    ✓ migrated labels")


def migrate_signal_source_relationships(
    gc: GraphClient, *, dry_run: bool = False
) -> None:
    """Rename USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE relationships."""
    result = gc.query("MATCH ()-[r:USES_SIGNAL_GROUP]->() RETURN count(r) AS c")
    count = result[0]["c"] if result else 0
    logger.info("  USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE: %d relationships", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (m)-[r:USES_SIGNAL_GROUP]->(ss)
            CREATE (m)-[:USES_SIGNAL_SOURCE]->(ss)
            DELETE r
        """)
        logger.info("    ✓ migrated relationships")


def migrate_unit_nodes(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Re-normalize Unit node IDs from UDUNITS to pint short notation.

    For each Unit node, re-parses the symbol through pint and checks
    if the normalized form differs. If it does, merges into the new ID
    and redirects all HAS_UNIT relationships.
    """
    from imas_codex.units import normalize_unit_symbol

    # Get all current Unit nodes
    result = gc.query("MATCH (u:Unit) RETURN u.id AS id, u.symbol AS symbol")
    if not result:
        logger.info("  Unit nodes: 0 (skip)")
        return

    logger.info("  Unit nodes: %d total", len(result))

    renames = []
    for row in result:
        old_id = row["id"]
        new_id = normalize_unit_symbol(old_id)
        if new_id is None:
            new_id = old_id  # keep as-is if unparseable
        if new_id != old_id:
            renames.append({"old_id": old_id, "new_id": new_id})

    logger.info("  Unit nodes needing rename: %d", len(renames))
    for r in renames:
        logger.info("    %s → %s", r["old_id"], r["new_id"])

    if renames and not dry_run:
        # For each rename: merge into new ID, redirect relationships, delete old
        for r in renames:
            old_id = r["old_id"]
            new_id = r["new_id"]
            # Create or merge the target unit node
            gc.query(
                "MERGE (u:Unit {id: $new_id}) SET u.symbol = $new_id",
                new_id=new_id,
            )
            # Redirect HAS_UNIT relationships from old → new
            gc.query(
                """
                MATCH (n)-[r:HAS_UNIT]->(old:Unit {id: $old_id})
                MATCH (new:Unit {id: $new_id})
                CREATE (n)-[:HAS_UNIT]->(new)
                DELETE r
                """,
                old_id=old_id,
                new_id=new_id,
            )
            # Update unit property on nodes that reference the old symbol
            gc.query(
                """
                MATCH (n)
                WHERE n.unit = $old_id
                SET n.unit = $new_id
                """,
                old_id=old_id,
                new_id=new_id,
            )
            # Delete old unit node if no remaining relationships
            gc.query(
                """
                MATCH (old:Unit {id: $old_id})
                WHERE NOT exists { (old)<-[:HAS_UNIT]-() }
                DELETE old
                """,
                old_id=old_id,
            )
        logger.info("    ✓ migrated unit nodes")


def backfill_signal_properties(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Backfill data_source_node and data_source_path from edges."""
    # data_source_node
    result = gc.query("""
        MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
        WHERE s.data_source_node IS NULL
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info("  Backfill data_source_node: %d signals", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
            WHERE s.data_source_node IS NULL
            SET s.data_source_node = sn.id
        """)
        logger.info("    ✓ backfilled data_source_node")

    # data_source_path
    result = gc.query("""
        MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
        WHERE s.data_source_path IS NULL
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info("  Backfill data_source_path: %d signals", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)-[:HAS_DATA_SOURCE_NODE]->(sn:SignalNode)
            WHERE s.data_source_path IS NULL
            SET s.data_source_path = sn.path
        """)
        logger.info("    ✓ backfilled data_source_path")


def set_enrichment_source(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Set enrichment_source for directly-enriched signals missing it."""
    result = gc.query("""
        MATCH (s:FacilitySignal)
        WHERE s.status IN ['enriched', 'checked']
          AND s.enrichment_source IS NULL
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info("  Set enrichment_source='direct': %d signals", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)
            WHERE s.status IN ['enriched', 'checked']
              AND s.enrichment_source IS NULL
            SET s.enrichment_source = 'direct'
        """)
        logger.info("    ✓ set enrichment_source")

    # Rename old pattern_propagation → signal_source_propagation
    result = gc.query("""
        MATCH (s:FacilitySignal)
        WHERE s.enrichment_source = 'pattern_propagation'
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info("  Rename pattern_propagation → signal_source_propagation: %d", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)
            WHERE s.enrichment_source = 'pattern_propagation'
            SET s.enrichment_source = 'signal_source_propagation'
        """)
        logger.info("    ✓ renamed enrichment_source")

    # Rename signal_group_propagation → signal_source_propagation
    result = gc.query("""
        MATCH (s:FacilitySignal)
        WHERE s.enrichment_source = 'signal_group_propagation'
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info(
        "  Rename signal_group_propagation → signal_source_propagation: %d", count
    )
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)
            WHERE s.enrichment_source = 'signal_group_propagation'
            SET s.enrichment_source = 'signal_source_propagation'
        """)
        logger.info("    ✓ renamed enrichment_source")


def cleanup_legacy_properties(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Remove legacy pattern properties."""
    result = gc.query("""
        MATCH (s:FacilitySignal)
        WHERE s.pattern_representative_id IS NOT NULL
        RETURN count(s) AS c
    """)
    count = result[0]["c"] if result else 0
    logger.info("  Remove legacy pattern properties: %d signals", count)
    if count > 0 and not dry_run:
        gc.query("""
            MATCH (s:FacilitySignal)
            WHERE s.pattern_representative_id IS NOT NULL
            REMOVE s.pattern_representative_id, s.pattern_template
        """)
        logger.info("    ✓ cleaned up legacy properties")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph migration v7")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes only")
    args = parser.parse_args()

    logger.info("Graph migration v7: Unit normalization + SignalSource rename")
    logger.info("=" * 60)

    with GraphClient() as gc:
        logger.info("\n1. SignalGroup → SignalSource label rename")
        migrate_signal_source_labels(gc, dry_run=args.dry_run)

        logger.info("\n2. USES_SIGNAL_GROUP → USES_SIGNAL_SOURCE relationships")
        migrate_signal_source_relationships(gc, dry_run=args.dry_run)

        logger.info("\n3. Unit node normalization (UDUNITS → pint short)")
        migrate_unit_nodes(gc, dry_run=args.dry_run)

        logger.info("\n4. Backfill signal properties from edges")
        backfill_signal_properties(gc, dry_run=args.dry_run)

        logger.info("\n5. Set enrichment_source on enriched signals")
        set_enrichment_source(gc, dry_run=args.dry_run)

        logger.info("\n6. Clean up legacy pattern properties")
        cleanup_legacy_properties(gc, dry_run=args.dry_run)

    logger.info("\n" + "=" * 60)
    if args.dry_run:
        logger.info("DRY RUN complete — no changes made")
    else:
        logger.info("Migration complete")


if __name__ == "__main__":
    main()
