"""Fixtures for graph quality tests.

All tests in this package require a live Neo4j connection. If Neo4j
is not reachable, tests are skipped automatically.
"""

import pytest

from imas_codex.graph.schema import GraphSchema, get_schema

# Apply graph marker to all tests in this directory
pytestmark = pytest.mark.graph


@pytest.fixture(scope="session")
def graph_client():
    """Session-scoped GraphClient that skips if Neo4j is unreachable."""
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        # Verify connectivity
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="session")
def schema() -> GraphSchema:
    """The global GraphSchema instance."""
    return get_schema()


@pytest.fixture(scope="session")
def label_counts(graph_client) -> dict[str, int]:
    """Node counts per label, fetched once per session."""
    return graph_client.get_label_counts()


@pytest.fixture(scope="session")
def graph_stats(graph_client) -> dict[str, int]:
    """Total node and relationship counts."""
    return graph_client.get_stats()


@pytest.fixture(scope="session")
def graph_labels(graph_client) -> set[str]:
    """Labels actually present in the graph."""
    results = graph_client.query("CALL db.labels() YIELD label RETURN label")
    return {r["label"] for r in results}


@pytest.fixture(scope="session")
def graph_relationship_types(graph_client) -> set[str]:
    """Relationship types actually present in the graph."""
    results = graph_client.query(
        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
    )
    return {r["relationshipType"] for r in results}


@pytest.fixture(scope="session")
def graph_constraints(graph_client) -> list[dict]:
    """All constraints in the graph."""
    return graph_client.query(
        "SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties "
        "RETURN name, type, labelsOrTypes, properties"
    )


@pytest.fixture(scope="session")
def graph_indexes(graph_client) -> list[dict]:
    """All indexes in the graph."""
    return graph_client.query(
        "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, state "
        "RETURN name, type, labelsOrTypes, properties, state"
    )


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    """Configured embedding dimension from pyproject.toml."""
    from imas_codex.settings import get_embedding_dimension

    return get_embedding_dimension()
