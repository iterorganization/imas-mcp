"""Tests for physics_domain promote-on-higher-rank semantics on StandardName.

Verifies:
- physics_domain is stored as a scalar string (the highest-priority domain)
- source_domains accumulates all contributing domains (append-only)
- Re-writing an existing domain does not duplicate it in source_domains
- HAS_PHYSICS_DOMAIN relationship count matches distinct source domains
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
    """physics_domain promote-on-higher-rank write semantics."""

    def test_write_single_domain_stores_scalar(self, graph_client, clean_sn):
        """Writing a single domain stores it as a scalar string (not a list)."""
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
        assert isinstance(pd, str)
        assert pd == "equilibrium"

    def test_append_new_domain_tracks_in_source_domains(self, graph_client, clean_sn):
        """Re-writing with a new domain records it in source_domains (order-insensitive).

        physics_domain retains the highest-priority (lowest-rank) domain after promotion.
        source_domains accumulates all contributing domains.
        """
        from imas_codex.standard_names.graph_ops import write_standard_names

        # First write — equilibrium (rank 0, highest priority)
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
        # Second write — transport (rank 2, lower priority than equilibrium)
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
            """MATCH (sn:StandardName {id: $id})
               RETURN sn.physics_domain AS pd, sn.source_domains AS sd""",
            id=clean_sn,
        )
        row = list(rows)[0]
        # physics_domain is the promoted scalar (equilibrium outranks transport)
        assert row["pd"] == "equilibrium"
        # source_domains tracks all contributing domains
        assert set(row["sd"]) == {"equilibrium", "transport"}

    def test_no_duplicate_on_rewrite(self, graph_client, clean_sn):
        """Re-writing an existing domain does not duplicate it in source_domains."""
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
        # Third write — duplicate domain (transport again)
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
            """MATCH (sn:StandardName {id: $id})
               RETURN sn.source_domains AS sd""",
            id=clean_sn,
        )
        sd = list(rows)[0]["sd"]
        # Must be exactly 2 distinct domains, no duplicates
        assert sorted(sd) == ["equilibrium", "transport"]

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

    def test_scalar_input_stored_as_scalar(self, graph_client, clean_sn):
        """Scalar physics_domain input is stored as a scalar string."""
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
        assert isinstance(pd, str)
        assert pd == "magnetics"
