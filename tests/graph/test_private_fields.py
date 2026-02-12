"""Private field leak detection tests.

Verifies that fields annotated with `is_private: true` in the LinkML
schema have not leaked into the graph database. Private data (hostnames,
IPs, NFS mounts, exploration notes) must only live in *_private.yaml
files, never in the graph.
"""

import re

import pytest

pytestmark = pytest.mark.graph

# Patterns that suggest private infrastructure data leaked
_HOSTNAME_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # IPv4
    r"|"
    r"\b[a-zA-Z0-9-]+\.(?:iter\.org|local|internal)\b"  # Internal hostnames
)


class TestPrivateFieldsNotInGraph:
    """Private fields from LinkML schema must not appear on graph nodes."""

    def test_no_private_fields_on_facility(self, graph_client, schema):
        """Facility nodes must not have is_private annotated fields."""
        private_slots = schema.get_private_slots("Facility")
        if not private_slots:
            pytest.skip("No private slots defined for Facility")

        violations = []
        for slot_name in private_slots:
            result = graph_client.query(
                f"MATCH (f:Facility) WHERE f.{slot_name} IS NOT NULL "
                f"RETURN f.id AS id, f.{slot_name} AS val LIMIT 3"
            )
            if result:
                ids = [r["id"] for r in result]
                violations.append(f"Facility.{slot_name} present on: {ids}")

        assert not violations, "Private fields leaked to graph:\n  " + "\n  ".join(
            violations
        )

    def test_no_ssh_host_on_facility(self, graph_client):
        """ssh_host is private and must not be in the graph."""
        result = graph_client.query(
            "MATCH (f:Facility) WHERE f.ssh_host IS NOT NULL RETURN f.id AS id LIMIT 5"
        )
        assert not result, (
            f"Facility nodes with ssh_host in graph: {[r['id'] for r in result]}"
        )

    def test_no_exploration_notes_on_facility(self, graph_client):
        """exploration_notes is private and must not be in the graph."""
        result = graph_client.query(
            "MATCH (f:Facility) WHERE f.exploration_notes IS NOT NULL "
            "RETURN f.id AS id LIMIT 5"
        )
        assert not result, (
            f"Facility nodes with exploration_notes in graph: "
            f"{[r['id'] for r in result]}"
        )


class TestNoInfrastructureLeaks:
    """Detect accidental infrastructure data leaks across all nodes."""

    def test_no_hostnames_in_facility_properties(self, graph_client):
        """Facility properties should not contain raw hostnames or IPs."""
        # Check common string properties that might accidentally contain infra data
        result = graph_client.query(
            "MATCH (f:Facility) "
            "RETURN f.id AS id, f.name AS name, f.description AS desc"
        )
        violations = []
        for row in result:
            for field in ["name", "desc"]:
                val = row.get(field)
                if val and _HOSTNAME_PATTERN.search(str(val)):
                    violations.append(
                        f"Facility({row['id']}).{field} may contain "
                        f"infrastructure data: {val[:80]}"
                    )

        assert not violations, (
            "Possible infrastructure data in Facility properties:\n  "
            + "\n  ".join(violations)
        )
