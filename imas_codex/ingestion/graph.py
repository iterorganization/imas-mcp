"""Graph relationship creation for ingested content.

Post-ingestion processing to create relationships between Chunk nodes,
DataReference nodes, and data entities (SignalNode, TDIFunction, IMASNode).

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
    """Normalize a path for fuzzy SignalNode matching.

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
    existing IMASNode IDS root nodes (where ids field matches the IDS name).

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.

    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (c:CodeChunk)
        WHERE c.related_ids IS NOT NULL
        UNWIND c.related_ids AS ids_name
        MATCH (p:IMASNode)
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


def link_chunks_to_data_nodes(graph_client: GraphClient | None = None) -> int:
    """Create DataReference nodes and link to DataNodes for MDSplus paths.

    For chunks with mdsplus_paths metadata:
    1. Creates DataReference nodes (deduplicated by facility:type:hash)
    2. Creates CONTAINS_REF relationships from CodeChunk -> DataReference
    3. Computes normalized_path for fuzzy matching
    4. Creates RESOLVES_TO_NODE relationships from DataReference -> SignalNode

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
            MATCH (t:SignalNode)
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

        # Step 4: Create RESOLVES_TO_IMAS_PATH via SignalNode → IMASMapping → IMASNode
        result = client.query("""
            MATCH (dr:DataReference)-[:RESOLVES_TO_NODE]->(dn:SignalNode)
            WHERE NOT (dr)-[:RESOLVES_TO_IMAS_PATH]->()
            MATCH (dn)-[:MEMBER_OF]->(sg:SignalGroup)-[:MAPS_TO_IMAS]->(ip:IMASNode)
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
        MATCH (t:SignalNode)
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


def _run_migration_step(
    client: GraphClient,
    *,
    key: str,
    stats: dict[str, int],
    dry_run: bool,
    count_query: str,
    apply_query: str,
    count_field: str = "pending",
    apply_field: str = "created",
    log_msg: str = "",
    batch_size: int = 0,
    **params: object,
) -> None:
    """Run a single idempotent migration step (count then apply).

    Args:
        batch_size: If >0, apply in batches using LIMIT. The apply_query
            must NOT already contain a LIMIT clause. Set this for steps
            that modify >10K nodes to avoid OOM.
    """
    result = client.query(count_query, **params)
    pending = result[0][count_field] if result else 0
    stats[f"{key}_pending"] = pending

    if not dry_run and pending > 0:
        if batch_size > 0:
            total = 0
            while True:
                result = client.query(apply_query, batch_size=batch_size, **params)
                batch_count = result[0][apply_field] if result else 0
                total += batch_count
                if batch_count == 0:
                    break
            stats[f"{key}_created"] = total
        else:
            result = client.query(apply_query, **params)
            stats[f"{key}_created"] = result[0][apply_field] if result else 0
        if log_msg:
            logger.info(log_msg, stats[f"{key}_created"])


def migrate_schema_relationships(
    dry_run: bool = False,
    graph_client: GraphClient | None = None,
) -> dict[str, int]:
    """Migrate graph to match the authoritative LinkML schema.

    Fixes drift between the LinkML schema and graph state for already-ingested
    data. All operations are idempotent (MERGE, not CREATE). Safe to run
    multiple times.

    Migrations:
     1. CODE_EXAMPLE_ID:   CodeChunk → CodeExample
     2. AT_FACILITY:       CodeChunk → Facility
     3. FROM_FILE:         CodeExample → CodeFile
     4. HAS_EXAMPLE:       CodeFile → CodeExample
     5. TreeNode → SignalNode:  Relabel + remove old label
     6. TreeNodePattern → SignalGroup:  Relabel + remove old label
     7. TreeModelVersion → StructuralEpoch:  Relabel + remove old label
     8. IN_DATA_SOURCE:    SignalNode → DataSource (from data_source_name)
     9. IN_TREE → IN_DATA_SOURCE:  Migrate legacy relationships
    10. RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE:  Migrate legacy relationships
    11. MDSplusTree cleanup:  Remove legacy MDSplusTree nodes
    12. Signal status fix: Map non-enum statuses to valid values
    13. Signal null status: Set discovered on null-status signals
    14. Ghost cleanup:     Remove empty FacilitySignal nodes
    15. SOURCE_NODE → HAS_DATA_SOURCE_NODE:  Migrate legacy relationships
    16. SAME_GEOMETRY cleanup:  Remove undeclared relationships
    17. ACCESSES_GEOMETRY cleanup:  Remove undeclared relationships
    18. Deprecated properties:  Remove _node_content, _node_type from CodeChunks
    19. PRODUCED → HAS_EXAMPLE:  Migrate renamed relationships
    20. SAME_SENSOR → MATCHES_SENSOR:  Migrate renamed relationships
    21. tree_name cleanup:  Remove deprecated property (replaced by data_source_name)
    22. AT_FACILITY:       CodeExample → Facility (missing edges)
    23. Garbage CodeChunks: Remove nodes with null id or null text
    24. Deduplicate CodeChunks: Remove duplicate nodes by id
    25. AT_FACILITY:       WikiPage → Facility (missing edges)
    26. AT_FACILITY:       DataAccess → Facility (missing edges)
    27. Spurious AT_FACILITY: Remove cross-facility edges where property ≠ edge
    28. HAS_CHUNK:         CodeExample → CodeChunk (orphan repair)
    29. Undeclared props:  Remove _related_ids, related_ids_count from CodeChunks
    30. DocSource twiki_raw: Normalize source_type to 'twiki'
    31. Constraint upgrade: Simple → composite (id, facility_id) where needed
    32. Orphan CodeChunks: Delete chunks whose parent CodeExample doesn't exist
    33. Status rollback: Roll back status on nodes missing lifecycle-required fields

    Args:
        dry_run: If True, only report counts without making changes.
        graph_client: Optional GraphClient instance.

    Returns:
        Dict with counts per migration step.
    """
    stats: dict[str, int] = {}

    with _get_client(graph_client) as client:
        # 1. Create CODE_EXAMPLE_ID relationships from existing property
        _run_migration_step(
            client,
            key="code_example_id",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                MATCH (ce:CodeExample {id: cc.code_example_id})
                WHERE NOT (cc)-[:CODE_EXAMPLE_ID]->(ce)
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                MATCH (ce:CodeExample {id: cc.code_example_id})
                WHERE NOT (cc)-[:CODE_EXAMPLE_ID]->(ce)
                WITH cc, ce LIMIT $batch_size
                MERGE (cc)-[:CODE_EXAMPLE_ID]->(ce)
                RETURN count(*) AS created
            """,
            log_msg="Created %d CODE_EXAMPLE_ID relationships",
            batch_size=5000,
        )

        # 2. Create AT_FACILITY for CodeChunks missing the relationship
        _run_migration_step(
            client,
            key="at_facility",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.facility_id IS NOT NULL
                AND NOT (cc)-[:AT_FACILITY]->()
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.facility_id IS NOT NULL
                AND NOT (cc)-[:AT_FACILITY]->()
                WITH cc LIMIT $batch_size
                MATCH (f:Facility {id: cc.facility_id})
                MERGE (cc)-[:AT_FACILITY]->(f)
                RETURN count(*) AS created
            """,
            log_msg="Created %d AT_FACILITY relationships for CodeChunks",
            batch_size=5000,
        )

        # 3. Create FROM_FILE relationships
        _run_migration_step(
            client,
            key="from_file",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (ce:CodeExample)
                WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
                AND NOT (ce)-[:FROM_FILE]->()
                MATCH (cf:CodeFile {path: ce.source_file})
                WHERE cf.facility_id = ce.facility_id
                RETURN count(ce) AS pending
            """,
            apply_query="""
                MATCH (ce:CodeExample)
                WHERE ce.source_file IS NOT NULL AND ce.facility_id IS NOT NULL
                AND NOT (ce)-[:FROM_FILE]->()
                WITH ce LIMIT $batch_size
                MATCH (cf:CodeFile {path: ce.source_file})
                WHERE cf.facility_id = ce.facility_id
                MERGE (ce)-[:FROM_FILE]->(cf)
                RETURN count(*) AS created
            """,
            log_msg="Created %d FROM_FILE relationships",
            batch_size=5000,
        )

        # 4. Create HAS_EXAMPLE from CodeFile to CodeExample
        _run_migration_step(
            client,
            key="has_example",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
                WHERE NOT (cf)-[:HAS_EXAMPLE]->(ce)
                RETURN count(*) AS pending
            """,
            apply_query="""
                MATCH (ce:CodeExample)-[:FROM_FILE]->(cf:CodeFile)
                MERGE (cf)-[:HAS_EXAMPLE]->(ce)
                RETURN count(*) AS created
            """,
            log_msg="Created %d HAS_EXAMPLE relationships",
        )

        # 5. Fix TreeNode → SignalNode label (add new, remove old)
        _run_migration_step(
            client,
            key="treenode_relabel",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (n) WHERE 'TreeNode' IN labels(n)
                AND NOT 'SignalNode' IN labels(n)
                RETURN count(n) AS pending
            """,
            apply_query="""
                MATCH (n) WHERE 'TreeNode' IN labels(n)
                WITH n LIMIT $batch_size
                SET n:SignalNode
                REMOVE n:TreeNode
                RETURN count(n) AS created
            """,
            log_msg="Relabeled %d TreeNode → SignalNode nodes",
            batch_size=10000,
        )

        # 6. Fix TreeNodePattern → SignalGroup label (add new, remove old)
        _run_migration_step(
            client,
            key="treenodepattern_relabel",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (n) WHERE 'TreeNodePattern' IN labels(n)
                RETURN count(n) AS pending
            """,
            apply_query="""
                MATCH (n) WHERE 'TreeNodePattern' IN labels(n)
                SET n:SignalGroup
                REMOVE n:TreeNodePattern
                RETURN count(n) AS created
            """,
            log_msg="Relabeled %d TreeNodePattern → SignalGroup nodes",
        )

        # 7. Fix TreeModelVersion → StructuralEpoch label (add new, remove old)
        _run_migration_step(
            client,
            key="treemodelversion_relabel",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (n) WHERE 'TreeModelVersion' IN labels(n)
                RETURN count(n) AS pending
            """,
            apply_query="""
                MATCH (n) WHERE 'TreeModelVersion' IN labels(n)
                SET n:StructuralEpoch
                REMOVE n:TreeModelVersion
                RETURN count(n) AS created
            """,
            log_msg="Relabeled %d TreeModelVersion → StructuralEpoch nodes",
        )

        # 8. Create IN_DATA_SOURCE from data_source_name property
        #    SignalNode nodes have data_source_name → DataSource.name
        #    Processes per-tree to avoid label scan on potentially corrupted
        #    DataSource index entries (OOM recovery artifact).
        pending_trees = client.query("""
            MATCH (n:SignalNode)
            WHERE n.data_source_name IS NOT NULL AND n.facility_id IS NOT NULL
            AND NOT (n)-[:IN_DATA_SOURCE]->()
            RETURN DISTINCT n.data_source_name AS name, n.facility_id AS fid,
                   count(n) AS cnt
        """)
        total_pending = sum(r["cnt"] for r in pending_trees) if pending_trees else 0
        stats["in_data_source_pending"] = total_pending

        if not dry_run and total_pending > 0:
            total_created = 0
            for tree in pending_trees:
                name, fid = tree["name"], tree["fid"]
                # Find or create DataSource node (param-bound avoids label scan)
                ds = client.query(
                    "MATCH (ds:DataSource {id: $fid + ':' + $name}) "
                    "RETURN elementId(ds) AS eid",
                    name=name,
                    fid=fid,
                )
                if not ds:
                    ds = client.query(
                        "CREATE (ds:DataSource {id: $fid + ':' + $name, "
                        "name: $name, facility_id: $fid}) "
                        "RETURN elementId(ds) AS eid",
                        name=name,
                        fid=fid,
                    )
                ds_eid = ds[0]["eid"]
                # Batch-create relationships using elementId lookup
                while True:
                    result = client.query(
                        "MATCH (n:SignalNode) "
                        "WHERE n.data_source_name = $name AND n.facility_id = $fid "
                        "AND NOT (n)-[:IN_DATA_SOURCE]->() "
                        "WITH n LIMIT 5000 "
                        "MATCH (ds) WHERE elementId(ds) = $ds_eid "
                        "CREATE (n)-[:IN_DATA_SOURCE]->(ds) "
                        "RETURN count(*) AS created",
                        name=name,
                        fid=fid,
                        ds_eid=ds_eid,
                    )
                    batch = result[0]["created"] if result else 0
                    total_created += batch
                    if batch == 0:
                        break
            stats["in_data_source_created"] = total_created
            logger.info("Created %d IN_DATA_SOURCE relationships", total_created)

        # 9. Migrate IN_TREE → IN_DATA_SOURCE relationships
        _run_migration_step(
            client,
            key="in_tree_migrate",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r]->() WHERE type(r) = 'IN_TREE'
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (a)-[r]->(b) WHERE type(r) = 'IN_TREE'
                WITH a, r, b LIMIT $batch_size
                MERGE (a)-[:IN_DATA_SOURCE]->(b)
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Migrated %d IN_TREE → IN_DATA_SOURCE relationships",
            batch_size=5000,
        )

        # 10. Migrate RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE relationships
        _run_migration_step(
            client,
            key="resolves_migrate",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r]->() WHERE type(r) = 'RESOLVES_TO_TREE_NODE'
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (a)-[r]->(b) WHERE type(r) = 'RESOLVES_TO_TREE_NODE'
                WITH a, r, b LIMIT $batch_size
                MERGE (a)-[:RESOLVES_TO_NODE]->(b)
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Migrated %d RESOLVES_TO_TREE_NODE → RESOLVES_TO_NODE relationships",
            batch_size=5000,
        )

        # 11. Remove legacy MDSplusTree nodes (data already in DataSource)
        _run_migration_step(
            client,
            key="mdsplustree_cleanup",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (t) WHERE 'MDSplusTree' IN labels(t)
                RETURN count(t) AS pending
            """,
            apply_query="""
                MATCH (t) WHERE 'MDSplusTree' IN labels(t)
                DETACH DELETE t
                RETURN count(t) AS created
            """,
            log_msg="Removed %d legacy MDSplusTree nodes",
        )

        # 12. Fix FacilitySignal non-enum status values
        #    Schema enum: discovered, enriched, checked, skipped, failed
        #    'scored'/'triaged' without descriptions → discovered
        #    'ingested' → enriched
        result = client.query("""
            MATCH (fs:FacilitySignal)
            WHERE fs.status IN ['scored', 'triaged', 'ingested']
            RETURN fs.status AS status, count(fs) AS cnt
        """)
        pending = sum(r["cnt"] for r in result) if result else 0
        stats["signal_status_pending"] = pending

        if not dry_run and pending > 0:
            # scored/triaged without description → discovered
            client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.status IN ['scored', 'triaged']
                AND fs.description IS NULL
                SET fs.status = 'discovered'
            """)
            # triaged with description → enriched
            client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.status = 'triaged'
                AND fs.description IS NOT NULL
                SET fs.status = 'enriched'
            """)
            # ingested → enriched
            client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.status = 'ingested'
                SET fs.status = 'enriched'
            """)
            result = client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.status IN ['scored', 'triaged', 'ingested']
                RETURN count(fs) AS remaining
            """)
            remaining = result[0]["remaining"] if result else 0
            stats["signal_status_created"] = pending - remaining
            logger.info(
                "Fixed %d FacilitySignal non-enum status values",
                stats["signal_status_created"],
            )

        # 13. Set null status on FacilitySignals with valid facility_id
        result = client.query("""
            MATCH (fs:FacilitySignal)
            WHERE fs.status IS NULL AND fs.id IS NOT NULL
            RETURN count(fs) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["signal_null_status_pending"] = pending

        if not dry_run and pending > 0:
            client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.status IS NULL AND fs.id IS NOT NULL
                SET fs.status = 'discovered'
            """)
            stats["signal_null_status_created"] = pending
            logger.info(
                "Set status='discovered' on %d FacilitySignals with null status",
                pending,
            )

        # 14. Remove ghost FacilitySignal nodes (null id, no properties)
        result = client.query("""
            MATCH (fs:FacilitySignal)
            WHERE fs.id IS NULL
            AND NOT (fs)-[]-()
            RETURN count(fs) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["ghost_signal_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (fs:FacilitySignal)
                WHERE fs.id IS NULL
                AND NOT (fs)-[]-()
                DELETE fs
                RETURN count(fs) AS created
            """)
            stats["ghost_signal_created"] = result[0]["created"] if result else 0
            logger.info(
                "Removed %d ghost FacilitySignal nodes",
                stats["ghost_signal_created"],
            )

        # 15. Migrate SOURCE_NODE → HAS_DATA_SOURCE_NODE
        _run_migration_step(
            client,
            key="source_node_migrate",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r]->() WHERE type(r) = 'SOURCE_NODE'
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (a)-[r]->(b) WHERE type(r) = 'SOURCE_NODE'
                WITH a, r, b LIMIT $batch_size
                MERGE (a)-[:HAS_DATA_SOURCE_NODE]->(b)
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Migrated %d SOURCE_NODE → HAS_DATA_SOURCE_NODE relationships",
            batch_size=5000,
        )

        # 16. Remove undeclared SAME_GEOMETRY relationships
        _run_migration_step(
            client,
            key="same_geometry_cleanup",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r:SAME_GEOMETRY]->()
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH ()-[r:SAME_GEOMETRY]->()
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Removed %d SAME_GEOMETRY relationships",
        )

        # 17. Remove undeclared ACCESSES_GEOMETRY relationships
        _run_migration_step(
            client,
            key="accesses_geometry_cleanup",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r:ACCESSES_GEOMETRY]->()
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH ()-[r:ACCESSES_GEOMETRY]->()
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Removed %d ACCESSES_GEOMETRY relationships",
        )

        # 18. Remove deprecated _node_content, _node_type from CodeChunks
        _run_migration_step(
            client,
            key="deprecated_props",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc._node_content IS NOT NULL OR cc._node_type IS NOT NULL
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc._node_content IS NOT NULL OR cc._node_type IS NOT NULL
                WITH cc LIMIT $batch_size
                REMOVE cc._node_content, cc._node_type
                RETURN count(cc) AS created
            """,
            log_msg="Cleaned %d CodeChunks with deprecated _node_* properties",
            batch_size=5000,
        )

        # 19. Migrate PRODUCED → HAS_EXAMPLE relationships
        _run_migration_step(
            client,
            key="produced_migrate",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r:PRODUCED]->()
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (a)-[r:PRODUCED]->(b)
                WITH a, r, b LIMIT $batch_size
                CREATE (a)-[:HAS_EXAMPLE]->(b)
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Migrated %d PRODUCED → HAS_EXAMPLE relationships",
            batch_size=100,
        )

        # 20. Migrate SAME_SENSOR → MATCHES_SENSOR relationships
        _run_migration_step(
            client,
            key="same_sensor_migrate",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH ()-[r:SAME_SENSOR]->()
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (a)-[r:SAME_SENSOR]->(b)
                WITH a, r, b LIMIT $batch_size
                CREATE (a)-[:MATCHES_SENSOR]->(b)
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Migrated %d SAME_SENSOR → MATCHES_SENSOR relationships",
            batch_size=100,
        )

        # 21. Remove deprecated tree_name property (replaced by data_source_name)
        _run_migration_step(
            client,
            key="tree_name_cleanup",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (n) WHERE n.tree_name IS NOT NULL
                RETURN count(n) AS pending
            """,
            apply_query="""
                MATCH (n) WHERE n.tree_name IS NOT NULL
                WITH n LIMIT $batch_size
                REMOVE n.tree_name
                RETURN count(n) AS created
            """,
            log_msg="Removed tree_name property from %d nodes",
            batch_size=500,
        )

        # 22. Create missing AT_FACILITY edges for CodeExample nodes
        _run_migration_step(
            client,
            key="code_example_at_facility",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (ce:CodeExample)
                WHERE ce.facility_id IS NOT NULL
                AND NOT (ce)-[:AT_FACILITY]->(:Facility)
                RETURN count(ce) AS pending
            """,
            apply_query="""
                MATCH (ce:CodeExample)
                WHERE ce.facility_id IS NOT NULL
                AND NOT (ce)-[:AT_FACILITY]->(:Facility)
                WITH ce LIMIT $batch_size
                MATCH (f:Facility {id: ce.facility_id})
                MERGE (ce)-[:AT_FACILITY]->(f)
                RETURN count(*) AS created
            """,
            log_msg="Created %d AT_FACILITY relationships for CodeExamples",
            batch_size=1000,
        )

        # 23. Delete garbage CodeChunk nodes (null id or null text)
        result = client.query("""
            MATCH (c:CodeChunk) WHERE c.id IS NULL OR c.text IS NULL
            RETURN count(c) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["garbage_codechunk_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (c:CodeChunk) WHERE c.id IS NULL OR c.text IS NULL
                DETACH DELETE c
                RETURN count(c) AS created
            """)
            stats["garbage_codechunk_created"] = result[0]["created"] if result else 0
            logger.info(
                "Removed %d garbage CodeChunk nodes (null id/text)",
                stats["garbage_codechunk_created"],
            )

        # 24. Deduplicate CodeChunk nodes (keep first, delete rest)
        result = client.query("""
            MATCH (c:CodeChunk)
            WITH c.id AS id, collect(c) AS nodes
            WHERE size(nodes) > 1
            UNWIND tail(nodes) AS dup
            RETURN count(dup) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["dedup_codechunk_pending"] = pending

        if not dry_run and pending > 0:
            result = client.query("""
                MATCH (c:CodeChunk)
                WITH c.id AS id, collect(c) AS nodes
                WHERE size(nodes) > 1
                UNWIND tail(nodes) AS dup
                DETACH DELETE dup
                RETURN count(dup) AS created
            """)
            stats["dedup_codechunk_created"] = result[0]["created"] if result else 0
            logger.info(
                "Deduplicated %d CodeChunk nodes",
                stats["dedup_codechunk_created"],
            )

        # 25. Create missing AT_FACILITY edges for WikiPage nodes
        _run_migration_step(
            client,
            key="wiki_page_at_facility",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (w:WikiPage)
                WHERE w.facility_id IS NOT NULL
                AND NOT (w)-[:AT_FACILITY]->(:Facility)
                RETURN count(w) AS pending
            """,
            apply_query="""
                MATCH (w:WikiPage)
                WHERE w.facility_id IS NOT NULL
                AND NOT (w)-[:AT_FACILITY]->(:Facility)
                WITH w LIMIT $batch_size
                MATCH (f:Facility {id: w.facility_id})
                MERGE (w)-[:AT_FACILITY]->(f)
                RETURN count(*) AS created
            """,
            log_msg="Created %d AT_FACILITY relationships for WikiPages",
            batch_size=1000,
        )

        # 26. Create missing AT_FACILITY edges for DataAccess nodes
        _run_migration_step(
            client,
            key="data_access_at_facility",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (da:DataAccess)
                WHERE da.facility_id IS NOT NULL
                AND NOT (da)-[:AT_FACILITY]->(:Facility)
                RETURN count(da) AS pending
            """,
            apply_query="""
                MATCH (da:DataAccess)
                WHERE da.facility_id IS NOT NULL
                AND NOT (da)-[:AT_FACILITY]->(:Facility)
                WITH da LIMIT $batch_size
                MATCH (f:Facility {id: da.facility_id})
                MERGE (da)-[:AT_FACILITY]->(f)
                RETURN count(*) AS created
            """,
            log_msg="Created %d AT_FACILITY relationships for DataAccess",
            batch_size=1000,
        )

        # 27. Remove spurious cross-facility AT_FACILITY edges
        #     Nodes have correct facility_id property but extra edges to
        #     wrong Facility nodes (created by buggy batch operations).
        _run_migration_step(
            client,
            key="spurious_at_facility",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (n)-[r:AT_FACILITY]->(f:Facility)
                WHERE n.facility_id IS NOT NULL AND n.facility_id <> f.id
                RETURN count(r) AS pending
            """,
            apply_query="""
                MATCH (n)-[r:AT_FACILITY]->(f:Facility)
                WHERE n.facility_id IS NOT NULL AND n.facility_id <> f.id
                WITH r LIMIT $batch_size
                DELETE r
                RETURN count(*) AS created
            """,
            log_msg="Removed %d spurious cross-facility AT_FACILITY edges",
            batch_size=5000,
        )

        # 28. Create missing HAS_CHUNK edges for orphan CodeChunks
        #     Chunks have code_example_id but no incoming HAS_CHUNK edge.
        _run_migration_step(
            client,
            key="orphan_has_chunk",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
                MATCH (ce:CodeExample {id: cc.code_example_id})
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                AND NOT (cc)<-[:HAS_CHUNK]-(:CodeExample)
                WITH cc LIMIT $batch_size
                MATCH (ce:CodeExample {id: cc.code_example_id})
                MERGE (ce)-[:HAS_CHUNK]->(cc)
                RETURN count(*) AS created
            """,
            log_msg="Created %d HAS_CHUNK relationships for orphan CodeChunks",
            batch_size=5000,
        )

        # 29. Remove undeclared _related_ids and related_ids_count properties
        _run_migration_step(
            client,
            key="undeclared_related_ids",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc._related_ids IS NOT NULL OR cc.related_ids_count IS NOT NULL
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc._related_ids IS NOT NULL OR cc.related_ids_count IS NOT NULL
                WITH cc LIMIT $batch_size
                REMOVE cc._related_ids, cc.related_ids_count
                RETURN count(cc) AS created
            """,
            log_msg="Cleaned %d CodeChunks with undeclared _related_ids properties",
            batch_size=5000,
        )

        # 30. Normalize DocSource source_type 'twiki_raw' → 'twiki'
        result = client.query("""
            MATCH (d:DocSource) WHERE d.source_type = 'twiki_raw'
            RETURN count(d) AS pending
        """)
        pending = result[0]["pending"] if result else 0
        stats["twiki_raw_pending"] = pending

        if not dry_run and pending > 0:
            client.query("""
                MATCH (d:DocSource) WHERE d.source_type = 'twiki_raw'
                SET d.source_type = 'twiki'
            """)
            stats["twiki_raw_created"] = pending
            logger.info("Normalized %d DocSource twiki_raw → twiki", pending)

        # 31. Fix constraints: upgrade simple → composite (id, facility_id)
        #     for all labels that the schema says need composite constraints
        from imas_codex.graph.schema import GraphSchema

        schema = GraphSchema()
        constraints = client.query("""
            SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties
            RETURN name, labelsOrTypes[0] AS label, properties
        """)
        constraint_map = {}
        for c in constraints or []:
            constraint_map[c["label"]] = (c["name"], c["properties"])

        constraint_fixes = 0
        for label in schema.node_labels:
            if not schema.needs_composite_constraint(label):
                continue
            existing = constraint_map.get(label)
            if existing and "facility_id" not in existing[1]:
                constraint_fixes += 1
                if not dry_run:
                    client.query(f"DROP CONSTRAINT {existing[0]} IF EXISTS")
                    id_field = schema.get_identifier(label)
                    constraint_name = f"{label.lower()}_{id_field}"
                    client.query(
                        f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE (n.{id_field}, n.facility_id) IS UNIQUE"
                    )
                    logger.info(
                        "Upgraded %s constraint to composite (id, facility_id)",
                        label,
                    )
        stats["constraint_upgrade_pending"] = constraint_fixes
        if not dry_run:
            stats["constraint_upgrade_created"] = constraint_fixes

        # 32. Delete orphan CodeChunks whose parent CodeExample doesn't exist
        _run_migration_step(
            client,
            key="orphan_codechunk_cleanup",
            stats=stats,
            dry_run=dry_run,
            count_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                AND NOT EXISTS { MATCH (ce:CodeExample {id: cc.code_example_id}) }
                RETURN count(cc) AS pending
            """,
            apply_query="""
                MATCH (cc:CodeChunk)
                WHERE cc.code_example_id IS NOT NULL
                AND NOT EXISTS { MATCH (ce:CodeExample {id: cc.code_example_id}) }
                WITH cc LIMIT $batch_size
                DETACH DELETE cc
                RETURN count(cc) AS created
            """,
            log_msg="Deleted %d orphan CodeChunks (missing parent CodeExample)",
            batch_size=5000,
        )

        # 33. Roll back status on nodes missing lifecycle-required fields
        #     Nodes that advanced to a status requiring certain fields but
        #     the fields are null — roll back so pipelines re-process them.
        status_rollbacks = [
            # CodeFile: scored needs score_composite → roll back to triaged
            (
                "CodeFile",
                "status",
                "scored",
                "score_composite",
                "triaged",
            ),
            # CodeFile: triaged/scored/ingested needs triage_composite → discovered
            (
                "CodeFile",
                "status",
                "triaged",
                "triage_composite",
                "discovered",
            ),
            # SignalNode: enriched needs description → discovered
            (
                "SignalNode",
                "enrichment_status",
                "enriched",
                "description",
                "discovered",
            ),
            # FacilityPath: triaged/scored/enriched needs description → discovered
            (
                "FacilityPath",
                "status",
                "triaged",
                "description",
                "discovered",
            ),
            # FacilityPath: scored/enriched needs score_composite → triaged
            (
                "FacilityPath",
                "status",
                "scored",
                "score_composite",
                "triaged",
            ),
        ]
        total_rollbacks = 0
        for (
            label,
            status_field,
            from_status,
            required_field,
            to_status,
        ) in status_rollbacks:
            result = client.query(
                f"MATCH (n:{label}) "
                f"WHERE n.{required_field} IS NULL AND n.{status_field} = $from_status "
                f"RETURN count(n) AS pending",
                from_status=from_status,
            )
            pending = result[0]["pending"] if result else 0
            if pending > 0 and not dry_run:
                client.query(
                    f"MATCH (n:{label}) "
                    f"WHERE n.{required_field} IS NULL AND n.{status_field} = $from_status "
                    f"SET n.{status_field} = $to_status",
                    from_status=from_status,
                    to_status=to_status,
                )
                logger.info(
                    "Rolled back %d %s from %s→%s (null %s)",
                    pending,
                    label,
                    from_status,
                    to_status,
                    required_field,
                )
            total_rollbacks += pending
        stats["status_rollback_pending"] = total_rollbacks
        if not dry_run:
            stats["status_rollback_created"] = total_rollbacks

    return stats


__all__ = [
    "link_chunks_to_imas_paths",
    "link_chunks_to_data_nodes",
    "link_example_mdsplus_paths",
    "link_examples_to_facility",
    "migrate_schema_relationships",
]
