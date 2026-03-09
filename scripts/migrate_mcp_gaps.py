#!/usr/bin/env python3
"""Migration script for MCP gap rectification (Stage 3).

Runs 8 migration steps in dependency order. Each step is idempotent.
Safe to rerun — uses MERGE and IF NOT EXISTS throughout.

Usage:
    uv run python scripts/migrate_mcp_gaps.py              # All steps
    uv run python scripts/migrate_mcp_gaps.py --step 2     # Single step
    uv run python scripts/migrate_mcp_gaps.py --dry-run    # Show queries only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def step1_fix_signal_contamination(gc, *, dry_run: bool = False) -> None:
    """Delete 289 contaminated JET FacilitySignals with wrong facility prefixes."""
    query = """
        MATCH (fs:FacilitySignal {facility_id: 'jet'})
        WHERE NOT fs.id STARTS WITH 'jet:'
        RETURN count(fs) AS count
    """
    result = gc.query(query)
    count = result[0]["count"] if result else 0
    logger.info("Step 1: Found %d contaminated JET signals", count)

    if count == 0:
        logger.info("Step 1: No contaminated signals to remove")
        return

    if dry_run:
        logger.info("Step 1: Would delete %d signals (dry-run)", count)
        return

    gc.query("""
        MATCH (fs:FacilitySignal {facility_id: 'jet'})
        WHERE NOT fs.id STARTS WITH 'jet:'
        DETACH DELETE fs
    """)
    logger.info("Step 1: Deleted %d contaminated signals", count)


def step2_create_data_access_nodes(gc, *, dry_run: bool = False) -> None:
    """Create general-purpose DataAccess nodes for JET."""
    nodes = [
        {
            "id": "jet:ppf:standard",
            "name": "PPF Standard Access",
            "facility_id": "jet",
            "method_type": "ppf",
            "library": "ppf",
            "access_type": "local",
            "connection_template": "ppfgo(pulse={shot})",
            "data_template": "ppfdata({shot}, '{dda}', '{dtype}')",
            "time_template": "ppftim({shot}, '{dda}', '{dtype}')",
            "data_source": "ppf",
            "description": "Standard PPF data access via ppfdata/ppfget functions",
        },
        {
            "id": "jet:ppf:python",
            "name": "PPF Python Access",
            "facility_id": "jet",
            "method_type": "ppf",
            "library": "ppf",
            "access_type": "local",
            "connection_template": "import ppf",
            "data_template": "ppf.ppfdata({shot}, '{dda}', '{dtype}')",
            "imports_template": "import ppf",
            "data_source": "ppf",
            "description": "Python PPF module access for JET processed data",
        },
        {
            "id": "jet:jpf:standard",
            "name": "JPF Standard Access",
            "facility_id": "jet",
            "method_type": "jpf",
            "library": "jpf",
            "access_type": "local",
            "connection_template": "conn = MDSplus.Connection('{server}')",
            "data_template": "conn.get('dpf(\"{signal_path}\", {shot})')",
            "data_source": "jpf",
            "description": "JET Primary Facility data access for raw diagnostic signals",
        },
        {
            "id": "jet:mdsplus:standard",
            "name": "MDSplus Standard Access",
            "facility_id": "jet",
            "method_type": "mdsplus",
            "library": "MDSplus",
            "access_type": "remote",
            "connection_template": "import MDSplus; conn = MDSplus.Connection('{server}')",
            "data_template": "conn.openTree('{tree}', {shot}); conn.get('{node_path}')",
            "data_source": "mdsplus",
            "description": "MDSplus tree access for JET experimental data",
        },
        {
            "id": "jet:uda:standard",
            "name": "UDA Standard Access",
            "facility_id": "jet",
            "method_type": "uda",
            "library": "pyuda",
            "access_type": "local",
            "connection_template": "import pyuda; client = pyuda.Client()",
            "data_template": "client.get('{signal}', {shot})",
            "data_source": "uda",
            "description": "Universal Data Access for JET data via IDAM/UDA client",
        },
    ]

    if dry_run:
        logger.info("Step 2: Would create %d DataAccess nodes (dry-run)", len(nodes))
        for n in nodes:
            logger.info("  %s: %s", n["id"], n["name"])
        return

    gc.create_nodes("DataAccess", nodes)
    logger.info("Step 2: Created/updated %d DataAccess nodes", len(nodes))


def step3_fix_code_relationships(gc, *, dry_run: bool = False) -> None:
    """Fix orphaned CodeChunk and CodeExample relationships."""
    queries = {
        "orphaned_chunks": """
            MATCH (cc:CodeChunk)
            WHERE cc.code_example_id IS NOT NULL
              AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
            RETURN count(cc) AS count
        """,
        "unlinked_examples": """
            MATCH (ce:CodeExample)
            WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
              AND NOT (ce)-[:FROM_FILE]->(:CodeFile)
            RETURN count(ce) AS count
        """,
    }

    for name, query in queries.items():
        result = gc.query(query)
        count = result[0]["count"] if result else 0
        logger.info("Step 3: %s = %d", name, count)

    if dry_run:
        logger.info("Step 3: Dry-run, skipping fixes")
        return

    # 3a: Fix orphaned CodeChunks
    gc.query("""
        MATCH (cc:CodeChunk)
        WHERE cc.code_example_id IS NOT NULL
          AND NOT (cc)-[:CODE_EXAMPLE_ID]->(:CodeExample)
        WITH cc
        MATCH (ce:CodeExample {id: cc.code_example_id})
        MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)
    """)

    # 3b: Create FROM_FILE relationships
    gc.query("""
        MATCH (ce:CodeExample)
        WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
          AND NOT (ce)-[:FROM_FILE]->(:CodeFile)
        WITH ce
        MATCH (cf:CodeFile)
        WHERE cf.path = ce.source_file AND cf.facility_id = ce.facility_id
        MERGE (ce)-[:FROM_FILE]->(cf)
    """)

    # 3c: Create HAS_EXAMPLE relationships
    gc.query("""
        MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
        WHERE NOT (cf)-[:HAS_EXAMPLE]->(ce)
        MERGE (cf)-[:HAS_EXAMPLE]->(ce)
    """)
    logger.info("Step 3: Code relationships fixed")


def step4_wiki_cross_refs(gc, *, dry_run: bool = False) -> None:
    """Create wiki cross-reference relationships for JET."""
    if dry_run:
        logger.info("Step 4: Would run link_chunks_to_entities('jet') (dry-run)")
        return

    from imas_codex.discovery.wiki.pipeline import link_chunks_to_entities

    stats = link_chunks_to_entities("jet")
    logger.info("Step 4: Wiki cross-refs created: %s", stats)


def step5_dedup_wiki_chunks(gc, *, dry_run: bool = False) -> None:
    """Deduplicate wiki chunks with identical text."""
    result = gc.query("""
        MATCH (c:WikiChunk {facility_id: 'jet'})
        WITH c.text AS text, count(*) AS cnt
        WHERE cnt > 1
        RETURN sum(cnt - 1) AS duplicates
    """)
    dups = result[0]["duplicates"] if result and result[0]["duplicates"] else 0
    logger.info("Step 5: Found %d duplicate wiki chunks", dups)

    if dups == 0:
        logger.info("Step 5: No duplicates to remove")
        return

    if dry_run:
        logger.info("Step 5: Would delete %d duplicate chunks (dry-run)", dups)
        return

    gc.query("""
        MATCH (c:WikiChunk {facility_id: 'jet'})
        WITH c.text AS text, collect(c) AS chunks
        WHERE size(chunks) > 1
        WITH text, head(chunks) AS keeper, tail(chunks) AS dupes
        UNWIND dupes AS dup
        DETACH DELETE dup
    """)
    logger.info("Step 5: Deleted %d duplicate chunks", dups)


def step6_create_fulltext_indexes(gc, *, dry_run: bool = False) -> None:
    """Create fulltext indexes for BM25 text search."""
    indexes = [
        (
            "wiki_chunk_text",
            "WikiChunk",
            ["n.text"],
        ),
        (
            "code_chunk_text",
            "CodeChunk",
            ["n.text", "n.function_name"],
        ),
        (
            "facility_signal_text",
            "FacilitySignal",
            ["n.name", "n.description", "n.node_path"],
        ),
        (
            "data_node_text",
            "DataNode",
            ["n.description", "n.canonical_path"],
        ),
    ]

    for idx_name, label, props in indexes:
        props_str = ", ".join(props)
        query = (
            f"CREATE FULLTEXT INDEX {idx_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON EACH [{props_str}] "
            "OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard-no-stop-words' } }"
        )
        if dry_run:
            logger.info("Step 6: Would create index %s on %s", idx_name, label)
            continue
        try:
            gc.query(query)
            logger.info("Step 6: Created fulltext index %s", idx_name)
        except Exception as e:
            logger.warning("Step 6: Index %s: %s", idx_name, e)


def step7_generate_datanode_embeddings(gc, *, dry_run: bool = False) -> None:
    """Generate embeddings for DataNodes with descriptions but no embeddings."""
    result = gc.query("""
        MATCH (n:DataNode)
        WHERE n.description IS NOT NULL AND n.embedding IS NULL
        RETURN count(n) AS count
    """)
    count = result[0]["count"] if result else 0
    logger.info("Step 7: Found %d DataNodes needing embeddings", count)

    if count == 0:
        logger.info("Step 7: All DataNodes already embedded")
        return

    if dry_run:
        logger.info("Step 7: Would embed %d DataNodes (dry-run)", count)
        return

    from datetime import UTC, datetime

    from imas_codex.embeddings import get_encoder

    encoder = get_encoder()
    batch_size = 100

    result = gc.query("""
        MATCH (n:DataNode)
        WHERE n.description IS NOT NULL AND n.embedding IS NULL
        RETURN n.id AS id, n.description AS description
    """)

    for i in range(0, len(result), batch_size):
        batch = result[i : i + batch_size]
        texts = [r["description"] for r in batch]
        vectors = encoder.embed_texts(texts)
        now = datetime.now(UTC).isoformat()

        items = []
        for r, vec in zip(batch, vectors, strict=True):
            items.append(
                {
                    "id": r["id"],
                    "embedding": vec.tolist(),
                    "embedded_at": now,
                }
            )

        gc.query(
            """
            UNWIND $items AS item
            MATCH (n:DataNode {id: item.id})
            SET n.embedding = item.embedding, n.embedded_at = item.embedded_at
            """,
            items=items,
        )
        logger.info(
            "Step 7: Embedded %d/%d DataNodes",
            min(i + batch_size, len(result)),
            len(result),
        )


def step8_cleanup_legacy_index(gc, *, dry_run: bool = False) -> None:
    """Remove legacy vector index aliases."""
    if dry_run:
        logger.info("Step 8: Would drop tree_node_desc_embedding index (dry-run)")
        return

    try:
        gc.query("DROP INDEX tree_node_desc_embedding IF EXISTS")
        logger.info("Step 8: Dropped legacy index tree_node_desc_embedding")
    except Exception:
        logger.info("Step 8: No legacy index to drop")


STEPS = {
    1: ("Fix signal contamination", step1_fix_signal_contamination),
    2: ("Create DataAccess nodes", step2_create_data_access_nodes),
    3: ("Fix code relationships", step3_fix_code_relationships),
    4: ("Wiki cross-references", step4_wiki_cross_refs),
    5: ("Dedup wiki chunks", step5_dedup_wiki_chunks),
    6: ("Create fulltext indexes", step6_create_fulltext_indexes),
    7: ("Generate DataNode embeddings", step7_generate_datanode_embeddings),
    8: ("Clean up legacy index", step8_cleanup_legacy_index),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP gap rectification migrations")
    parser.add_argument(
        "--step", type=int, choices=list(STEPS.keys()), help="Run a single step"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    args = parser.parse_args()

    from imas_codex.graph.client import GraphClient

    gc = GraphClient()

    steps = [args.step] if args.step else list(STEPS.keys())

    for step_num in steps:
        name, func = STEPS[step_num]
        logger.info("=== Step %d: %s ===", step_num, name)
        func(gc, dry_run=args.dry_run)
        logger.info("")

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
