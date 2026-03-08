"""Graph relationship creation for ingested content.

Post-ingestion processing to create relationships between Chunk nodes,
DataReference nodes, and data entities (DataNode, TDIFunction, IMASPath).

Architecture:
    Chunk -[:CONTAINS_REF]-> DataReference -[:RESOLVES_TO_*]-> Entity

DataReference nodes preserve the exact string found in content for provenance,
while typed resolution relationships link to actual data entities.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Pattern to strip channel/index suffixes for fuzzy matching
CHANNEL_SUFFIX_PATTERN = re.compile(r"[_:](?:CHANNEL_?)?\d+$", re.IGNORECASE)


def _normalize_for_matching(raw_path: str) -> str:
    """Normalize a path for fuzzy DataNode matching.

    Strips channel indices and numeric suffixes to match tree structure.
    E.g., \\ATLAS::DT196_MHD_001:CHANNEL_006 -> \\ATLAS::DT196_MHD:CHANNEL
    """
    normalized = raw_path.upper().rstrip(":.")
    while CHANNEL_SUFFIX_PATTERN.search(normalized):
        normalized = CHANNEL_SUFFIX_PATTERN.sub("", normalized)
    return normalized


def _generate_ref_id(facility: str, ref_type: str, raw_string: str) -> str:
    """Generate DataReference ID: facility:type:hash."""
    content = raw_string.encode("utf-8")
    hash_suffix = hashlib.md5(content).hexdigest()[:12]
    return f"{facility}:{ref_type}:{hash_suffix}"


@contextlib.contextmanager
def _get_client(graph_client: GraphClient | None = None) -> Iterator[GraphClient]:
    """Context manager that uses provided client or creates a new one."""
    if graph_client is not None:
        yield graph_client
    else:
        from imas_codex.graph import GraphClient

        client = GraphClient()
        with client:
            yield client


def link_chunks_to_imas_paths(graph_client: GraphClient | None = None) -> int:
    """Create REFERENCES_IMAS relationships for chunks with IDS references.

    Matches CodeChunk nodes that have related_ids metadata to
    existing IMASPath IDS root nodes (where ids field matches the IDS name).

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.

    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (c:CodeChunk)
        WHERE c.related_ids IS NOT NULL
        UNWIND c.related_ids AS ids_name
        MATCH (p:IMASPath)
        WHERE p.ids = ids_name AND p.id = ids_name
        MERGE (c)-[:REFERENCES_IMAS]->(p)
        RETURN count(*) AS created
    """

    with _get_client(graph_client) as client:
        result = client.query(cypher)
        count = result[0]["created"] if result else 0
        logger.info("Created %d REFERENCES_IMAS relationships", count)
        return count


def link_examples_to_facility(graph_client: GraphClient | None = None) -> int:
    """Create AT_FACILITY relationships from CodeExample nodes.

    Args:
        graph_client: Optional GraphClient instance.

    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (e:CodeExample)
        WHERE e.facility_id IS NOT NULL
        MATCH (f:Facility {id: e.facility_id})
        MERGE (e)-[:AT_FACILITY]->(f)
        RETURN count(*) AS created
    """

    with _get_client(graph_client) as client:
        result = client.query(cypher)
        count = result[0]["created"] if result else 0
        logger.info("Created %d AT_FACILITY relationships", count)
        return count


def link_chunks_to_tree_nodes(graph_client: GraphClient | None = None) -> int:
    """Create DataReference nodes and link to DataNodes for MDSplus paths.

    For chunks with mdsplus_paths metadata:
    1. Creates DataReference nodes (deduplicated by facility:type:hash)
    2. Creates CONTAINS_REF relationships from CodeChunk -> DataReference
    3. Computes normalized_path for fuzzy matching
    4. Creates RESOLVES_TO_NODE relationships from DataReference -> DataNode

    Args:
        graph_client: Optional GraphClient instance.

    Returns:
        Number of DataReference nodes created or linked
    """
    with _get_client(graph_client) as client:
        # Step 1: Create DataReference nodes from mdsplus_paths
        create_refs_simple = """
            MATCH (c:CodeChunk)
            WHERE c.mdsplus_paths IS NOT NULL
            MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
            WHERE e.facility_id IS NOT NULL
            UNWIND c.mdsplus_paths AS path
            WITH DISTINCT e.facility_id AS facility, path
            MERGE (d:DataReference {raw_string: path, facility_id: facility})
            ON CREATE SET
                d.id = facility + ':mdsplus_path:' + path,
                d.ref_type = 'mdsplus_path'
            WITH d, facility
            MATCH (f:Facility {id: facility})
            MERGE (d)-[:AT_FACILITY]->(f)
            RETURN count(d) AS refs_created
        """
        result = client.query(create_refs_simple)
        refs_created = result[0]["refs_created"] if result else 0
        logger.info("Created/matched %d DataReference nodes", refs_created)

        # Step 2: Create CONTAINS_REF relationships
        contains_ref = """
            MATCH (c:CodeChunk)
            WHERE c.mdsplus_paths IS NOT NULL
            MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
            WHERE e.facility_id IS NOT NULL
            UNWIND c.mdsplus_paths AS path
            WITH c, e.facility_id AS facility, path
            MATCH (d:DataReference {raw_string: path, facility_id: facility})
            MERGE (c)-[:CONTAINS_REF]->(d)
            RETURN count(*) AS contains_created
        """
        result = client.query(contains_ref)
        contains_count = result[0]["contains_created"] if result else 0
        logger.info("Created %d CONTAINS_REF relationships", contains_count)

        # Step 2.5: Compute normalized_path for fuzzy matching
        refs_to_normalize = client.query("""
            MATCH (d:DataReference {ref_type: 'mdsplus_path'})
            WHERE d.normalized_path IS NULL
            RETURN d.id AS id, d.raw_string AS raw
        """)
        if refs_to_normalize:
            updates = [
                {"id": r["id"], "normalized": _normalize_for_matching(r["raw"])}
                for r in refs_to_normalize
            ]
            client.query(
                """
                UNWIND $updates AS u
                MATCH (d:DataReference {id: u.id})
                SET d.normalized_path = u.normalized
                """,
                updates=updates,
            )
            logger.info("Computed normalized_path for %d refs", len(updates))

        # Step 3: Create RESOLVES_TO_NODE relationships
        resolve_to_tree = """
            MATCH (d:DataReference {ref_type: 'mdsplus_path'})
            WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
            MATCH (t:DataNode)
            WHERE t.path = d.raw_string
               OR t.path ENDS WITH substring(d.raw_string, 1)
               OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
               OR toUpper(t.path) = d.normalized_path
            MERGE (d)-[:RESOLVES_TO_NODE]->(t)
            RETURN count(*) AS resolved
        """
        result = client.query(resolve_to_tree)
        resolved_count = result[0]["resolved"] if result else 0
        logger.info("Created %d RESOLVES_TO_NODE relationships", resolved_count)

        # Step 4: Create RESOLVES_TO_IMAS_PATH via DataNode → IMASMapping → IMASPath
        result = client.query("""
            MATCH (dr:DataReference)-[:RESOLVES_TO_NODE]->(dn:DataNode)
            WHERE NOT (dr)-[:RESOLVES_TO_IMAS_PATH]->()
            MATCH (m:IMASMapping)-[:SOURCE_PATH]->(dn)
            MATCH (m)-[:TARGET_PATH]->(ip:IMASPath)
            MERGE (dr)-[:RESOLVES_TO_IMAS_PATH]->(ip)
            RETURN count(*) AS linked
        """)
        imas_linked = result[0]["linked"] if result else 0
        if imas_linked:
            logger.info("Created %d RESOLVES_TO_IMAS_PATH relationships", imas_linked)

        # Step 5: Create CALLS_TDI_FUNCTION for TDI call references
        result = client.query("""
            MATCH (dr:DataReference {ref_type: 'tdi_call'})
            WHERE NOT (dr)-[:CALLS_TDI_FUNCTION]->()
            MATCH (tdi:TDIFunction {facility_id: dr.facility_id})
            WHERE tdi.name = dr.raw_string
               OR dr.raw_string CONTAINS tdi.name
            MERGE (dr)-[:CALLS_TDI_FUNCTION]->(tdi)
            RETURN count(*) AS linked
        """)
        tdi_linked = result[0]["linked"] if result else 0
        if tdi_linked:
            logger.info("Created %d CALLS_TDI_FUNCTION relationships", tdi_linked)

        # Update ref_count on CodeChunk nodes
        client.query("""
            MATCH (c:CodeChunk)-[r:CONTAINS_REF]->(d:DataReference)
            WITH c, count(r) AS ref_count
            SET c.ref_count = ref_count
        """)

        return refs_created


def link_example_mdsplus_paths(
    graph_client: GraphClient,
    example_id: str,
) -> int:
    """Create DataReference nodes and link to DataNodes for a specific example.

    Args:
        graph_client: GraphClient instance
        example_id: ID of the CodeExample

    Returns:
        Number of DataReference nodes created/linked
    """
    # Create DataReference nodes and CONTAINS_REF relationships
    result = graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
        WHERE c.mdsplus_paths IS NOT NULL AND e.facility_id IS NOT NULL
        UNWIND c.mdsplus_paths AS path
        WITH c, e.facility_id AS facility, path
        MERGE (d:DataReference {raw_string: path, facility_id: facility})
        ON CREATE SET
            d.id = facility + ':mdsplus_path:' + path,
            d.ref_type = 'mdsplus_path'
        MERGE (c)-[:CONTAINS_REF]->(d)
        WITH d, facility
        MATCH (f:Facility {id: facility})
        MERGE (d)-[:AT_FACILITY]->(f)
        RETURN count(DISTINCT d) AS refs_created
        """,
        example_id=example_id,
    )
    refs_created = result[0]["refs_created"] if result else 0

    # Resolve to DataNodes
    graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
              -[:CONTAINS_REF]->(d:DataReference {ref_type: 'mdsplus_path'})
        WHERE NOT (d)-[:RESOLVES_TO_NODE]->()
        MATCH (t:DataNode)
        WHERE t.path = d.raw_string
           OR t.path ENDS WITH substring(d.raw_string, 1)
           OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
        MERGE (d)-[:RESOLVES_TO_NODE]->(t)
        """,
        example_id=example_id,
    )

    # Update ref_count
    graph_client.query(
        """
        MATCH (e:CodeExample {id: $example_id})-[:HAS_CHUNK]->(c:CodeChunk)
              -[r:CONTAINS_REF]->(d:DataReference)
        WITH c, count(r) AS ref_count
        SET c.ref_count = ref_count
        """,
        example_id=example_id,
    )

    return refs_created


def migrate_schema_relationships(
    dry_run: bool = False,
    graph_client: GraphClient | None = None,
) -> dict[str, int]:
    """One-time migration to create missing schema-defined relationships.

    Fixes drift between LinkML schema and graph state for already-ingested data.
    All operations are idempotent (MERGE, not CREATE).

    Migrations:
    1. CODE_EXAMPLE_ID: CodeChunk → CodeExample
    2. AT_FACILITY: CodeChunk → Facility (using denormalized facility_id)
    3. FROM_FILE: CodeExample → CodeFile
    4. PRODUCED: CodeFile → CodeExample
    5. TreeNode → DataNode label fix

    Args:
        dry_run: If True, only report counts without creating relationships.
        graph_client: Optional GraphClient instance.

    Returns:
        Dict with counts per migration step.
    """
    stats: dict[str, int] = {}

    with _get_client(graph_client) as client:
        # 1. Create CODE_EXAMPLE_ID relationships from existing property
        result = client.query("""
            MATCH (cc:CodeChunk)
            WHERE cc.code_example_id IS NOT NULL
            MATCH (ce:CodeExample {id: cc.code_example_id})
            WHERE NOT (cc)-[:CODE_EXAMPLE_ID]->(ce)
            RETURN count(cc) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["code_example_id_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                MATCH (ce:CodeExample {id: cc.code_example_id})
                MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)
                RETURN count(*) AS created
            """)
            stats["code_example_id_created"] = result[0]["created"] if result else 0
            logger.info(
                "Created %d CODE_EXAMPLE_ID relationships",
                stats["code_example_id_created"],
            )

        # 2. Create AT_FACILITY for CodeChunks missing the relationship
        result = client.query("""
            MATCH (cc:CodeChunk)
            WHERE cc.facility_id IS NOT NULL
            AND NOT (cc)-[:AT_FACILITY]->()
            RETURN count(cc) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["at_facility_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (cc:CodeChunk)
                WHERE cc.facility_id IS NOT NULL
                MATCH (f:Facility {id: cc.facility_id})
                MERGE (cc)-[:AT_FACILITY]->(f)
                RETURN count(*) AS created
            """)
            stats["at_facility_created"] = result[0]["created"] if result else 0
            logger.info(
                "Created %d AT_FACILITY relationships for CodeChunks",
                stats["at_facility_created"],
            )

        # 3. Create FROM_FILE relationships
        result = client.query("""
            MATCH (ce:CodeExample)
            WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
            AND NOT (ce)-[:FROM_FILE]->()
            MATCH (cf:CodeFile {path: ce.source_file})
            WHERE cf.facility_id = ce.facility_id
            RETURN count(ce) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["from_file_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (ce:CodeExample)
                WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
                MATCH (cf:CodeFile {path: ce.source_file})
                WHERE cf.facility_id = ce.facility_id
                MERGE (ce)-[:FROM_FILE]->(cf)
                RETURN count(*) AS created
            """)
            stats["from_file_created"] = result[0]["created"] if result else 0
            logger.info(
                "Created %d FROM_FILE relationships",
                stats["from_file_created"],
            )

        # 4. Create PRODUCED from CodeFile to CodeExample
        result = client.query("""
            MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
            WHERE NOT (cf)-[:PRODUCED]->(ce)
            RETURN count(*) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["produced_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
                MERGE (cf)-[:PRODUCED]->(ce)
                RETURN count(*) AS created
            """)
            stats["produced_created"] = result[0]["created"] if result else 0
            logger.info(
                "Created %d PRODUCED relationships",
                stats["produced_created"],
            )

        # 5. Fix TreeNode → DataNode label (add DataNode label to TreeNode nodes)
        result = client.query("""
            MATCH (n:TreeNode)
            WHERE NOT n:DataNode
            RETURN count(n) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["treenode_relabel_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (n:TreeNode)
                WHERE NOT n:DataNode
                SET n:DataNode
                RETURN count(n) AS relabeled
            """)
            stats["treenode_relabeled"] = result[0]["relabeled"] if result else 0
            logger.info(
                "Added DataNode label to %d TreeNode nodes",
                stats["treenode_relabeled"],
            )

    return stats


__all__ = [
    "link_chunks_to_imas_paths",
    "link_chunks_to_tree_nodes",
    "link_example_mdsplus_paths",
    "link_examples_to_facility",
    "migrate_schema_relationships",
]
