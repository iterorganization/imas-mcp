"""Graph relationship creation for code examples.

Post-ingestion processing to create relationships between CodeChunk nodes,
DataReference nodes, and data entities (TreeNode, TDIFunction, IMASPath).

Architecture:
    CodeChunk -[:CONTAINS_REF]-> DataReference -[:RESOLVES_TO_*]-> Entity

DataReference nodes preserve the exact string found in code for provenance,
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
# Matches: _001, _00, :CHANNEL_006, etc.
CHANNEL_SUFFIX_PATTERN = re.compile(r"[_:](?:CHANNEL_?)?\d+$", re.IGNORECASE)


def _normalize_for_matching(raw_path: str) -> str:
    """Normalize a path for fuzzy TreeNode matching.

    Strips channel indices and numeric suffixes to match tree structure.
    E.g., \\ATLAS::DT196_MHD_001:CHANNEL_006 -> \\ATLAS::DT196_MHD:CHANNEL
    """
    normalized = raw_path.upper().rstrip(":.")
    # Iteratively strip channel/index suffixes
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
    """Context manager that uses provided client or creates a new one.

    If a client is provided, yields it without managing its lifecycle.
    If no client is provided, creates one and ensures it's closed.
    """
    if graph_client is not None:
        # Client provided - don't manage lifecycle
        yield graph_client
    else:
        # Create and manage our own client
        from imas_codex.graph import GraphClient

        client = GraphClient()
        with client:
            yield client


def link_chunks_to_imas_paths(graph_client: GraphClient | None = None) -> int:
    """Create REFERENCES_IMAS relationships for chunks with IDS references.

    Matches CodeChunk nodes that have related_ids metadata to
    existing IMASPath nodes (IDS-level paths where id = ids name).

    Note: IMASPath nodes must already exist from DD ingestion.
    This function does NOT create IMASPath nodes on-demand.

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
        WHERE p.id = ids_name AND p.ids = ids_name
        MERGE (c)-[:REFERENCES_IMAS]->(p)
        RETURN count(*) AS created
    """

    with _get_client(graph_client) as client:
        result = client.query(cypher)
        count = result[0]["created"] if result else 0
        logger.info("Created %d REFERENCES_IMAS relationships", count)
        return count


def link_examples_to_facility(graph_client: GraphClient | None = None) -> int:
    """Create FACILITY_ID relationships from CodeExample nodes.

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.

    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (e:CodeExample)
        WHERE e.facility_id IS NOT NULL
        MATCH (f:Facility {id: e.facility_id})
        MERGE (e)-[:FACILITY_ID]->(f)
        RETURN count(*) AS created
    """

    with _get_client(graph_client) as client:
        result = client.query(cypher)
        count = result[0]["created"] if result else 0
        logger.info("Created %d FACILITY_ID relationships", count)
        return count


def link_chunks_to_tree_nodes(graph_client: GraphClient | None = None) -> int:
    """Create DataReference nodes and link to TreeNodes for MDSplus paths.

    For chunks with mdsplus_paths metadata:
    1. Creates DataReference nodes (deduplicated by facility:type:hash)
    2. Creates CONTAINS_REF relationships from CodeChunk -> DataReference
    3. Computes normalized_path for fuzzy matching
    4. Creates RESOLVES_TO_TREE_NODE relationships from DataReference -> TreeNode

    Uses case-insensitive name matching on the final path component
    to handle path format variations.

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.

    Returns:
        Number of DataReference nodes created or linked
    """
    with _get_client(graph_client) as client:
        # Step 1: Create DataReference nodes from mdsplus_paths
        # This is idempotent - MERGE ensures no duplicates
        create_refs_simple = """
            MATCH (c:CodeChunk)
            WHERE c.mdsplus_paths IS NOT NULL
            MATCH (c)<-[:HAS_CHUNK]-(e:CodeExample)
            UNWIND c.mdsplus_paths AS path
            WITH DISTINCT coalesce(e.facility_id, 'epfl') AS facility, path
            MERGE (d:DataReference {raw_string: path, facility_id: facility})
            ON CREATE SET
                d.id = facility + ':mdsplus_path:' + path,
                d.ref_type = 'mdsplus_path'
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
            UNWIND c.mdsplus_paths AS path
            WITH c, coalesce(e.facility_id, 'epfl') AS facility, path
            MATCH (d:DataReference {raw_string: path, facility_id: facility})
            MERGE (c)-[:CONTAINS_REF]->(d)
            RETURN count(*) AS contains_created
        """
        result = client.query(contains_ref)
        contains_count = result[0]["contains_created"] if result else 0
        logger.info("Created %d CONTAINS_REF relationships", contains_count)

        # Step 2.5: Compute normalized_path for fuzzy matching (Python-side)
        # Fetch refs without normalized_path, compute it, update in batch
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

        # Step 3: Create RESOLVES_TO_TREE_NODE relationships
        # Uses multiple matching strategies for path format variations
        resolve_to_tree = """
            MATCH (d:DataReference {ref_type: 'mdsplus_path'})
            WHERE NOT (d)-[:RESOLVES_TO_TREE_NODE]->()
            MATCH (t:TreeNode)
            WHERE t.path = d.raw_string
               OR t.path ENDS WITH substring(d.raw_string, 1)
               OR toLower(split(t.path, ':')[-1]) = toLower(split(d.raw_string, '::')[-1])
               OR toUpper(t.path) = d.normalized_path
            MERGE (d)-[:RESOLVES_TO_TREE_NODE]->(t)
            RETURN count(*) AS resolved
        """
        result = client.query(resolve_to_tree)
        resolved_count = result[0]["resolved"] if result else 0
        logger.info("Created %d RESOLVES_TO_TREE_NODE relationships", resolved_count)

        # Update ref_count on CodeChunk nodes
        client.query("""
            MATCH (c:CodeChunk)-[r:CONTAINS_REF]->(d:DataReference)
            WITH c, count(r) AS ref_count
            SET c.ref_count = ref_count
        """)

        return refs_created


__all__ = [
    "link_chunks_to_imas_paths",
    "link_chunks_to_tree_nodes",
    "link_examples_to_facility",
]
