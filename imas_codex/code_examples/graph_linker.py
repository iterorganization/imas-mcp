"""Graph relationship creation for code examples.

Post-ingestion processing to create REFERENCES_IMAS relationships
between CodeChunk nodes and IMASPath nodes based on IDS references.
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


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
    """Create REFERENCES_NODE relationships for chunks with MDSplus paths.

    Matches CodeChunk nodes that have mdsplus_paths metadata to
    corresponding TreeNode entities. Uses suffix matching to handle
    paths extracted from f-strings.

    Args:
        graph_client: Optional GraphClient instance. If None, creates one.

    Returns:
        Number of relationships created
    """
    cypher = """
        MATCH (c:CodeChunk)
        WHERE c.mdsplus_paths IS NOT NULL
        UNWIND c.mdsplus_paths AS mds_path
        MATCH (t:TreeNode)
        WHERE t.path = mds_path OR t.path ENDS WITH substring(mds_path, 1)
        MERGE (c)-[:REFERENCES_NODE]->(t)
        RETURN count(*) AS created
    """

    with _get_client(graph_client) as client:
        result = client.query(cypher)
        count = result[0]["created"] if result else 0
        logger.info("Created %d REFERENCES_NODE relationships", count)
        return count


__all__ = [
    "link_chunks_to_imas_paths",
    "link_chunks_to_tree_nodes",
    "link_examples_to_facility",
]
