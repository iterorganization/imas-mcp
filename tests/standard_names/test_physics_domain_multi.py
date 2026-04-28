"""Tests for multi-valued physics_domain on StandardName.

Verifies:
- physics_domain is stored as a list
- Append-only semantics (second write adds new domains)
- Deduplication (re-writing an existing domain is a no-op)
- HAS_PHYSICS_DOMAIN relationship count matches distinct domains
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.graph


@pytest.fixture()
def clean_sn(graph_client):
    """Delete test StandardName nodes before and after each test."""
    test_id = "test_multidomain_electron_temperature"
    graph_client.query(
        "MATCH (sn:StandardName {id: $id}) DETACH DELETE sn",
        id=test_id,
    )
    yield test_id
    graph_client.query(
        "MATCH (sn:StandardName {id: $id}) DETACH DELETE sn",
        id=test_id,
    )


class TestPhysicsDomainMultiValued:
    """Multi-valued physics_domain write semantics."""

    def test_write_single_domain_stores_list(self, graph_client, clean_sn):
        """Writing a single domain stores it as a list."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        count = write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["equilibrium"],
                    "pipeline_status": "named",
                }
            ]
        )
        assert count >= 1

        rows = graph_client.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.physics_domain AS pd",
            id=clean_sn,
        )
        pd = list(rows)[0]["pd"]
        assert isinstance(pd, list)
        assert pd == ["equilibrium"]

    def test_append_new_domain(self, graph_client, clean_sn):
        """Re-writing with a new domain appends it (order-insensitive)."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        # First write
        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["equilibrium"],
                    "pipeline_status": "named",
                }
            ]
        )
        # Second write — different domain
        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["transport"],
                    "pipeline_status": "named",
                }
            ]
        )

        rows = graph_client.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.physics_domain AS pd",
            id=clean_sn,
        )
        pd = set(list(rows)[0]["pd"])
        assert pd == {"equilibrium", "transport"}

    def test_no_duplicate_on_rewrite(self, graph_client, clean_sn):
        """Re-writing an existing domain does not duplicate it."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["equilibrium"],
                    "pipeline_status": "named",
                }
            ]
        )
        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["transport"],
                    "pipeline_status": "named",
                }
            ]
        )
        # Third write — duplicate domain
        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["transport"],
                    "pipeline_status": "named",
                }
            ]
        )

        rows = graph_client.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.physics_domain AS pd",
            id=clean_sn,
        )
        pd = list(rows)[0]["pd"]
        # Must be exactly 2, no duplicates
        assert sorted(pd) == ["equilibrium", "transport"]

    def test_has_physics_domain_edge_count(self, graph_client, clean_sn):
        """HAS_PHYSICS_DOMAIN relationship count matches distinct domains."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["equilibrium"],
                    "pipeline_status": "named",
                }
            ]
        )
        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": ["transport"],
                    "pipeline_status": "named",
                }
            ]
        )

        rows = graph_client.query(
            """
            MATCH (sn:StandardName {id: $id})-[:HAS_PHYSICS_DOMAIN]->(pd:PhysicsDomain)
            RETURN count(pd) AS edge_count
            """,
            id=clean_sn,
        )
        assert list(rows)[0]["edge_count"] == 2

    def test_scalar_input_wrapped_to_list(self, graph_client, clean_sn):
        """Scalar physics_domain input is silently wrapped to a list."""
        from imas_codex.standard_names.graph_ops import write_standard_names

        write_standard_names(
            [
                {
                    "id": clean_sn,
                    "source_types": ["dd"],
                    "physics_domain": "magnetics",  # scalar input
                    "pipeline_status": "named",
                }
            ]
        )

        rows = graph_client.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.physics_domain AS pd",
            id=clean_sn,
        )
        pd = list(rows)[0]["pd"]
        assert isinstance(pd, list)
        assert pd == ["magnetics"]
