"""Fixtures for graph quality tests.

All tests in this package require a live Neo4j connection. If Neo4j
is not reachable, tests are skipped automatically at collection time
(visible as "skipped" rather than silently deselected).
"""

from pathlib import Path

import pytest

from imas_codex.graph.schema import GraphSchema, get_schema

# Apply graph marker to all tests in this directory
pytestmark = pytest.mark.graph

# ── Auto-skip when Neo4j is unreachable ──────────────────────────────────
# Checked once per session, cached in module-level variable.

_neo4j_available: bool | None = None


def _check_neo4j_available() -> bool:
    """Quick probe to see if Neo4j is reachable (cached)."""
    global _neo4j_available
    if _neo4j_available is not None:
        return _neo4j_available
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
        client.close()
        _neo4j_available = True
    except Exception:
        _neo4j_available = False
    return _neo4j_available


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip all graph-marked tests when Neo4j is not reachable.

    Runs the connectivity check once per session. Skipping is visible
    in test output ('s') so users know graph tests *exist* but were not
    executed, rather than being silently deselected.
    """
    if _check_neo4j_available():
        return
    skip_marker = pytest.mark.skip(reason="Neo4j not available")
    for item in items:
        if item.get_closest_marker("graph"):
            item.add_marker(skip_marker)


# =============================================================================
# Schema helpers - derive configuration from LinkML schemas
# =============================================================================


def _get_schemas_dir() -> Path:
    """Get the schemas directory."""
    return Path(__file__).parent.parent.parent / "imas_codex" / "schemas"


def _get_combined_embeddable_labels() -> list[str]:
    """Get all embeddable labels from all schemas (facility + imas_dd)."""
    schemas_dir = _get_schemas_dir()
    labels = set()
    for schema_name in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_name)
        labels |= set(gs.embeddable_labels)
    return sorted(labels)


def _get_combined_description_embeddable_labels() -> list[str]:
    """Get labels with description+embedding from all schemas."""
    schemas_dir = _get_schemas_dir()
    labels = set()
    for schema_name in ["facility.yaml", "imas_dd.yaml"]:
        gs = GraphSchema(schemas_dir / schema_name)
        labels |= set(gs.description_embeddable_labels)
    return sorted(labels)


# Module-level caches (computed once per test session)
_ALL_EMBEDDABLE_LABELS: list[str] | None = None
_DESCRIPTION_EMBEDDABLE_LABELS: list[str] | None = None


def get_all_embeddable_labels() -> list[str]:
    """Get all node labels with embedding slots (cached)."""
    global _ALL_EMBEDDABLE_LABELS
    if _ALL_EMBEDDABLE_LABELS is None:
        _ALL_EMBEDDABLE_LABELS = _get_combined_embeddable_labels()
    return _ALL_EMBEDDABLE_LABELS


def get_description_embeddable_labels() -> list[str]:
    """Get node labels with description+embedding (cached)."""
    global _DESCRIPTION_EMBEDDABLE_LABELS
    if _DESCRIPTION_EMBEDDABLE_LABELS is None:
        _DESCRIPTION_EMBEDDABLE_LABELS = _get_combined_description_embeddable_labels()
    return _DESCRIPTION_EMBEDDABLE_LABELS


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def graph_client():
    """Session-scoped GraphClient that skips if Neo4j is unreachable.

    Does NOT run ``initialize_schema()`` — that is heavy DDL work that
    overwhelms SSH tunnels.  Tests needing constraints should use the
    ``graph_with_schema`` fixture instead.
    """
    from imas_codex.graph.client import GraphClient

    try:
        client = GraphClient()
        client.get_stats()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield client
    client.close()


@pytest.fixture(scope="session")
def graph_with_schema(graph_client):
    """GraphClient with schema constraints initialized.

    Calls ``initialize_schema()`` once per session — idempotent but
    does many DDL round-trips.  Only used by constraint tests.
    Verifies connectivity after the DDL burst in case the tunnel dropped.
    """
    from imas_codex.graph.client import GraphClient

    try:
        graph_client.initialize_schema()
    except Exception:
        pass  # Non-fatal — constraint tests will report specifics

    # Verify the connection survived the DDL burst
    try:
        graph_client.query("RETURN 1 AS ping")
    except Exception:
        # Connection pool poisoned — recreate
        try:
            graph_client.close()
        except Exception:
            pass
        try:
            new_client = GraphClient()
            new_client.get_stats()
            yield new_client
            new_client.close()
            return
        except Exception as e:
            pytest.skip(f"Neo4j connection lost during schema init: {e}")

    yield graph_client


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
