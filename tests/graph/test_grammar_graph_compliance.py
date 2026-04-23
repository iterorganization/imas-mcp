"""Grammar-graph schema compliance tests (plan 29 E.8).

Validates that the grammar graph nodes and edges match the canonical
ISN SEGMENT_ORDER specification:
- Every segment in SEGMENT_ORDER exists as a GrammarSegment node.
- Every HAS_SEGMENT edge references a valid ISN segment name.
- Positional indices on HAS_SEGMENT edges match SEGMENT_ORDER ordering.

These are integration tests requiring a live Neo4j instance with
a synced grammar graph. Auto-skipped when Neo4j is unreachable.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.graph, pytest.mark.integration]


@pytest.fixture(scope="module")
def segment_order():
    """Import ISN SEGMENT_ORDER from the canonical location."""
    try:
        from imas_standard_names.grammar.constants import SEGMENT_ORDER

        return SEGMENT_ORDER
    except ImportError:
        pytest.skip("imas_standard_names not installed — cannot verify SEGMENT_ORDER")


@pytest.fixture(scope="module")
def grammar_segments(graph_client):
    """Fetch all GrammarSegment nodes from the graph."""
    results = graph_client.query(
        "MATCH (s:GrammarSegment) RETURN s.name AS name, s.position AS position"
    )
    return results


@pytest.fixture(scope="module")
def has_segment_edges(graph_client):
    """Fetch all HAS_SEGMENT edges with their segment properties."""
    results = graph_client.query(
        "MATCH (:StandardName)-[r:HAS_SEGMENT]->(:GrammarToken) "
        "RETURN DISTINCT r.segment AS segment, r.position AS position"
    )
    return results


class TestSegmentOrderCompliance:
    """Verify grammar graph aligns with ISN SEGMENT_ORDER."""

    def test_all_isn_segments_present_as_nodes(
        self, grammar_segments, segment_order
    ) -> None:
        """Every segment declared by ISN SEGMENT_ORDER must exist as a
        GrammarSegment node after a fresh sync."""
        graph_segment_names = {r["name"] for r in grammar_segments}
        isn_segments = set(segment_order)
        missing = isn_segments - graph_segment_names
        assert not missing, (
            f"ISN segments missing from graph: {sorted(missing)}. "
            f"Run `imas-codex sn sync-grammar` to populate."
        )

    def test_graph_segments_subset_of_isn(
        self, grammar_segments, segment_order
    ) -> None:
        """No GrammarSegment node should exist outside the ISN spec."""
        graph_segment_names = {r["name"] for r in grammar_segments}
        isn_segments = set(segment_order)
        extra = graph_segment_names - isn_segments
        assert not extra, (
            f"Graph has GrammarSegment nodes not in ISN SEGMENT_ORDER: "
            f"{sorted(extra)}. These may be stale from a prior version."
        )

    def test_segment_positions_match_isn_order(
        self, grammar_segments, segment_order
    ) -> None:
        """Each GrammarSegment.position must equal its index in SEGMENT_ORDER."""
        isn_positions = {name: idx for idx, name in enumerate(segment_order)}
        mismatches = []
        for seg in grammar_segments:
            name = seg["name"]
            if name in isn_positions:
                expected_pos = isn_positions[name]
                actual_pos = seg["position"]
                if actual_pos != expected_pos:
                    mismatches.append(f"{name}: graph={actual_pos}, ISN={expected_pos}")
        assert not mismatches, (
            f"GrammarSegment position mismatches: {'; '.join(mismatches)}"
        )

    def test_has_segment_edge_segments_in_allowlist(
        self, has_segment_edges, segment_order
    ) -> None:
        """Every segment_name on HAS_SEGMENT edges must be in ISN SEGMENT_ORDER."""
        isn_segments = set(segment_order)
        edge_segments = {r["segment"] for r in has_segment_edges if r["segment"]}
        invalid = edge_segments - isn_segments
        assert not invalid, (
            f"HAS_SEGMENT edges reference unknown segments: {sorted(invalid)}"
        )

    def test_has_segment_edge_positions_match_isn_index(
        self, has_segment_edges, segment_order
    ) -> None:
        """The position on each HAS_SEGMENT edge must match the ISN index."""
        isn_positions = {name: idx for idx, name in enumerate(segment_order)}
        mismatches = []
        for edge in has_segment_edges:
            seg = edge["segment"]
            pos = edge["position"]
            if seg and seg in isn_positions:
                expected = isn_positions[seg]
                if pos != expected:
                    mismatches.append(
                        f"segment={seg}: edge.position={pos}, ISN index={expected}"
                    )
        assert not mismatches, (
            f"HAS_SEGMENT edge position mismatches: {'; '.join(mismatches)}"
        )
