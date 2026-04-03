"""Search-test fixtures shared across the search test suite."""

from __future__ import annotations

import pytest

from tests.search.benchmark_data import ALL_QUERIES


def _collect_cluster_ids() -> set[str]:
    """Gather all cluster IDs referenced by benchmark queries."""
    ids: set[str] = set()
    for q in ALL_QUERIES:
        ids.update(q.expected_clusters)
    return ids


@pytest.fixture(scope="session")
def cluster_members() -> dict[str, list[str]]:
    """Pre-fetch cluster members for all referenced clusters.

    Returns a mapping of ``cluster_id`` → list of member path IDs.
    Returns an empty dict when Neo4j is unavailable or the cluster
    query fails — callers should handle the empty-dict case gracefully.
    """
    cluster_ids = _collect_cluster_ids()
    if not cluster_ids:
        return {}

    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception:
        return {}

    try:
        rows = client.query(
            """
            UNWIND $cids AS cid
            MATCH (n:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster {id: cid})
            RETURN c.id AS cluster_id, collect(n.id) AS members
            """,
            cids=list(cluster_ids),
        )
        return {row["cluster_id"]: row["members"] for row in rows}
    except Exception:
        return {}
    finally:
        client.close()
