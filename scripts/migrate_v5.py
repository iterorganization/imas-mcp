"""Graph migration v5: Rename labels, relationships, and indexes.

Migrates the live graph to match the v5 schema renames:
  - DataNode → SignalNode
  - IMASPath → IMASNode
  - IMASPathChange → IMASNodeChange
  - FOLLOWS_PATTERN → MEMBER_OF
  - Drops old vector/fulltext indexes, recreates for new labels
  - Removes defunct nodes (old IMASMapping field-level, AgentSession)

Usage:
    uv run python scripts/migrate_v5.py
    uv run python scripts/migrate_v5.py --dry-run
"""

import argparse
import logging

from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_embedding_dimension

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Label renames: (old_label, new_label)
LABEL_RENAMES = [
    ("DataNode", "SignalNode"),
    ("IMASPath", "IMASNode"),
    ("IMASPathChange", "IMASNodeChange"),
]

# Relationship renames: (old_type, new_type)
RELATIONSHIP_RENAMES = [
    ("FOLLOWS_PATTERN", "MEMBER_OF"),
]

# Old vector indexes to drop
OLD_VECTOR_INDEXES = [
    "data_node_desc_embedding",
    "imas_path_embedding",
]

# Old fulltext indexes to drop
OLD_FULLTEXT_INDEXES = [
    "data_node_text",
    "imas_path_text",
]

# Old range indexes to drop (new label range indexes already created by schema)
# These are actually uniqueness constraints — drop the constraint, index drops too
OLD_CONSTRAINTS = [
    "imaspath_id",
    "imaspathchange_id",
]


def count_labels(gc: GraphClient) -> dict[str, int]:
    """Count nodes with old and new labels."""
    counts: dict[str, int] = {}
    for old, new in LABEL_RENAMES:
        for label in (old, new):
            result = gc.query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = result[0]["c"] if result else 0
    return counts


def count_relationships(gc: GraphClient) -> dict[str, int]:
    """Count relationships with old and new types."""
    counts: dict[str, int] = {}
    for old, new in RELATIONSHIP_RENAMES:
        for rel_type in (old, new):
            result = gc.query(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
            counts[rel_type] = result[0]["c"] if result else 0
    return counts


def list_indexes(gc: GraphClient) -> dict[str, dict]:
    """Get all indexes as name → info dict."""
    result = gc.query("SHOW INDEXES YIELD name, type, labelsOrTypes, properties")
    return {r["name"]: r for r in result}


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


def migrate_relationships(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Rename relationship types."""
    for old, new in RELATIONSHIP_RENAMES:
        result = gc.query(f"MATCH ()-[r:{old}]->() RETURN count(r) AS c")
        count = result[0]["c"] if result else 0
        if count == 0:
            logger.info("  %s → %s: 0 relationships (skip)", old, new)
            continue
        logger.info("  %s → %s: %d relationships", old, new, count)
        if not dry_run:
            gc.query(f"""
                CALL {{
                    MATCH (a)-[r:{old}]->(b)
                    CREATE (a)-[:{new}]->(b)
                    DELETE r
                }} IN TRANSACTIONS OF 1000 ROWS
            """)
            logger.info("    ✓ migrated")


def remove_defunct_nodes(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Remove defunct node types and stale properties from old schema."""
    # Old field-level IMASMapping nodes (status proposed/validated)
    result = gc.query("""
        MATCH (n:IMASMapping)
        WHERE n.source_path IS NOT NULL
        RETURN count(n) AS c
    """)
    old_mapping_count = result[0]["c"] if result else 0
    if old_mapping_count > 0:
        logger.info("  Old field-level IMASMapping nodes: %d", old_mapping_count)
        if not dry_run:
            gc.query("""
                MATCH (n:IMASMapping)
                WHERE n.source_path IS NOT NULL
                DETACH DELETE n
            """)
            logger.info("    ✓ removed")
    else:
        logger.info("  Old field-level IMASMapping nodes: 0 (skip)")

    # AgentSession nodes
    result = gc.query("MATCH (n:AgentSession) RETURN count(n) AS c")
    agent_count = result[0]["c"] if result else 0
    if agent_count > 0:
        logger.info("  AgentSession nodes: %d", agent_count)
        if not dry_run:
            gc.query("MATCH (n:AgentSession) DETACH DELETE n")
            logger.info("    ✓ removed")
    else:
        logger.info("  AgentSession nodes: 0 (skip)")

    # Remove stale properties from FacilitySignal (removed from schema in Phase 1d)
    stale_props = ["pattern_representative_id", "pattern_template"]
    for prop in stale_props:
        result = gc.query(f"""
            MATCH (n:FacilitySignal)
            WHERE n.{prop} IS NOT NULL
            RETURN count(n) AS c
        """)
        count = result[0]["c"] if result else 0
        if count > 0:
            logger.info("  FacilitySignal.%s: %d nodes", prop, count)
            if not dry_run:
                gc.query(f"""
                    MATCH (n:FacilitySignal)
                    WHERE n.{prop} IS NOT NULL
                    REMOVE n.{prop}
                """)
                logger.info("    ✓ removed")
        else:
            logger.info("  FacilitySignal.%s: 0 nodes (skip)", prop)


def migrate_indexes(gc: GraphClient, *, dry_run: bool = False) -> None:
    """Drop old indexes and create new ones for renamed labels."""
    dim = get_embedding_dimension()
    indexes = list_indexes(gc)

    # Drop old vector indexes
    for idx_name in OLD_VECTOR_INDEXES:
        if idx_name in indexes:
            logger.info("  Drop vector index: %s", idx_name)
            if not dry_run:
                gc.query(f"DROP INDEX {idx_name} IF EXISTS")
                logger.info("    ✓ dropped")
        else:
            logger.info("  Vector index %s: not found (skip)", idx_name)

    # Drop old fulltext indexes
    for idx_name in OLD_FULLTEXT_INDEXES:
        if idx_name in indexes:
            logger.info("  Drop fulltext index: %s", idx_name)
            if not dry_run:
                gc.query(f"DROP INDEX {idx_name} IF EXISTS")
                logger.info("    ✓ dropped")
        else:
            logger.info("  Fulltext index %s: not found (skip)", idx_name)

    # Drop old constraints (which also drops their backing range indexes)
    constraints = {r["name"] for r in gc.query("SHOW CONSTRAINTS YIELD name")}
    for constraint_name in OLD_CONSTRAINTS:
        if constraint_name in constraints:
            logger.info("  Drop constraint: %s", constraint_name)
            if not dry_run:
                gc.query(f"DROP CONSTRAINT {constraint_name} IF EXISTS")
                logger.info("    ✓ dropped")
        else:
            logger.info("  Constraint %s: not found (skip)", constraint_name)

    # Recreate vector indexes for new labels
    new_vector_indexes = [
        (
            "signal_node_desc_embedding",
            "SignalNode",
            "embedding",
        ),
        (
            "imas_node_embedding",
            "IMASNode",
            "embedding",
        ),
    ]
    for idx_name, label, prop in new_vector_indexes:
        if idx_name in indexes:
            logger.info("  Vector index %s: already exists (skip)", idx_name)
        else:
            logger.info(
                "  Create vector index: %s (%s.%s, dim=%d)", idx_name, label, prop, dim
            )
            if not dry_run:
                gc.query(f"""
                    CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
                    FOR (n:{label}) ON n.{prop}
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {dim},
                            `vector.similarity_function`: 'cosine'
                        }}
                    }}
                """)
                logger.info("    ✓ created")

    # Recreate fulltext indexes for new labels
    new_fulltext_indexes = [
        ("signal_node_text", "SignalNode", ["path", "description"]),
        ("imas_node_text", "IMASNode", ["id", "description"]),
    ]
    for idx_name, label, props in new_fulltext_indexes:
        if idx_name in indexes:
            logger.info("  Fulltext index %s: already exists (skip)", idx_name)
        else:
            prop_str = ", ".join(f"n.{p}" for p in props)
            logger.info(
                "  Create fulltext index: %s (%s on %s)", idx_name, label, props
            )
            if not dry_run:
                gc.query(f"""
                    CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS
                    FOR (n:{label}) ON EACH [{prop_str}]
                """)
                logger.info("    ✓ created")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate graph to v5 schema")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    args = parser.parse_args()

    mode = "DRY RUN" if args.dry_run else "LIVE"
    logger.info("=== Graph Migration v5 (%s) ===\n", mode)

    with GraphClient() as gc:
        # Pre-migration counts
        logger.info("Pre-migration counts:")
        label_counts = count_labels(gc)
        for label, count in sorted(label_counts.items()):
            logger.info("  %s: %d", label, count)
        rel_counts = count_relationships(gc)
        for rel_type, count in sorted(rel_counts.items()):
            logger.info("  %s: %d", rel_type, count)

        # Step 1: Rename labels
        logger.info("\n1. Label migrations:")
        migrate_labels(gc, dry_run=args.dry_run)

        # Step 2: Rename relationships
        logger.info("\n2. Relationship migrations:")
        migrate_relationships(gc, dry_run=args.dry_run)

        # Step 3: Remove defunct nodes
        logger.info("\n3. Remove defunct nodes:")
        remove_defunct_nodes(gc, dry_run=args.dry_run)

        # Step 4: Migrate indexes
        logger.info("\n4. Index migrations:")
        migrate_indexes(gc, dry_run=args.dry_run)

        # Post-migration counts
        logger.info("\nPost-migration counts:")
        label_counts = count_labels(gc)
        for label, count in sorted(label_counts.items()):
            logger.info("  %s: %d", label, count)
        rel_counts = count_relationships(gc)
        for rel_type, count in sorted(rel_counts.items()):
            logger.info("  %s: %d", rel_type, count)

    logger.info(
        "\n=== Migration %s ===", "preview complete" if args.dry_run else "complete"
    )


if __name__ == "__main__":
    main()
