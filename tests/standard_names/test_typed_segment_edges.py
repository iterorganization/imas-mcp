"""Tests for typed grammar-segment edges written alongside HAS_SEGMENT.

W40 Phase 4: _write_segment_edges now writes 10 typed relationship types
(HAS_PHYSICAL_BASE, HAS_SUBJECT, HAS_TRANSFORMATION, …) in addition to the
generic HAS_SEGMENT edge.

All tests are mock-based — no live Neo4j required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_version_resolver():
    """Return a fixed token version so _write_segment_edges proceeds."""
    with patch(
        "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
        return_value="0.7.0rc31",
    ):
        yield


@pytest.fixture()
def mock_gc():
    """Mock GraphClient that records query calls."""
    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client


@pytest.fixture()
def _patch_isn():
    """Skip test if ISN is not installed."""
    pytest.importorskip("imas_standard_names")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_delete_calls(mock_gc: MagicMock) -> list:
    """Find DELETE queries (idempotency pre-clear)."""
    return [c for c in mock_gc.query.call_args_list if "DELETE" in str(c)]


def _find_merge_segment_calls(mock_gc: MagicMock) -> list:
    """Find the MERGE + FOREACH query that writes segment edges."""
    return [
        c
        for c in mock_gc.query.call_args_list
        if "HAS_SEGMENT" in str(c) and "MERGE" in str(c)
    ]


def _cypher_of(call) -> str:
    """Extract Cypher string from a mock call."""
    return call[0][0]


# ---------------------------------------------------------------------------
# Phase 4 Tests: typed edge Cypher is present in the MERGE query
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestTypedEdgeCypherPresent:
    """Verify that the 10 typed FOREACH clauses appear in the Cypher emitted
    by _write_segment_edges."""

    def test_has_physical_base_clause_in_cypher(self, mock_gc: MagicMock) -> None:
        """HAS_PHYSICAL_BASE FOREACH clause must be present in the MERGE query."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]  # DELETE, then MERGE/RETURN
        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert merge_calls, "Expected at least one MERGE/FOREACH query"
        cypher = _cypher_of(merge_calls[0])
        assert "HAS_PHYSICAL_BASE" in cypher

    def test_has_subject_clause_in_cypher(self, mock_gc: MagicMock) -> None:
        """HAS_SUBJECT FOREACH clause must be present in the MERGE query."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        cypher = _cypher_of(merge_calls[0])
        assert "HAS_SUBJECT" in cypher

    def test_all_ten_typed_edges_in_cypher(self, mock_gc: MagicMock) -> None:
        """All 10 typed edge labels must appear in the MERGE query Cypher."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert merge_calls
        cypher = _cypher_of(merge_calls[0])

        expected_edges = [
            "HAS_PHYSICAL_BASE",
            "HAS_SUBJECT",
            "HAS_TRANSFORMATION",
            "HAS_COMPONENT",
            "HAS_COORDINATE",
            "HAS_PROCESS",
            "HAS_POSITION",
            "HAS_REGION",
            "HAS_DEVICE",
            "HAS_GEOMETRIC_BASE",
        ]
        for edge_label in expected_edges:
            assert edge_label in cypher, (
                f"Expected {edge_label} in MERGE Cypher but not found"
            )


# ---------------------------------------------------------------------------
# Phase 4 Tests: idempotency — pre-delete covers all typed edge types
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestTypedEdgeIdempotency:
    """Verify the pre-delete covers all 10 typed edge labels."""

    def test_delete_covers_all_typed_labels(self, mock_gc: MagicMock) -> None:
        """The DELETE query must include all 10 typed edge labels plus HAS_SEGMENT."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        delete_calls = _find_delete_calls(mock_gc)
        assert delete_calls, "Expected a DELETE call"

        delete_cypher = _cypher_of(delete_calls[0])
        for label in [
            "HAS_SEGMENT",
            "HAS_PHYSICAL_BASE",
            "HAS_SUBJECT",
            "HAS_TRANSFORMATION",
            "HAS_COMPONENT",
            "HAS_COORDINATE",
            "HAS_PROCESS",
            "HAS_POSITION",
            "HAS_REGION",
            "HAS_DEVICE",
            "HAS_GEOMETRIC_BASE",
        ]:
            assert label in delete_cypher, (
                f"Expected {label} in DELETE Cypher but not found"
            )

    def test_delete_before_merge(self, mock_gc: MagicMock) -> None:
        """DELETE must execute before the MERGE/FOREACH query."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        calls = mock_gc.query.call_args_list
        delete_idx = next((i for i, c in enumerate(calls) if "DELETE" in str(c)), None)
        merge_idx = next(
            (
                i
                for i, c in enumerate(calls)
                if "HAS_SEGMENT" in str(c) and "MERGE" in str(c)
            ),
            None,
        )
        assert delete_idx is not None
        assert merge_idx is not None
        assert delete_idx < merge_idx, "DELETE must precede MERGE"


# ---------------------------------------------------------------------------
# Phase 4 Tests: segment-conditional FOREACH — correct segment guards
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestTypedEdgeSegmentGuards:
    """Verify each FOREACH clause is guarded by the correct segment value."""

    def test_physical_base_guard_in_cypher(self, mock_gc: MagicMock) -> None:
        """HAS_PHYSICAL_BASE FOREACH must guard on segment = 'physical_base'."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        cypher = _cypher_of(merge_calls[0])
        # The guard and the edge label must both be present
        assert "'physical_base'" in cypher
        assert "HAS_PHYSICAL_BASE" in cypher

    def test_transformation_guard_in_cypher(self, mock_gc: MagicMock) -> None:
        """HAS_TRANSFORMATION FOREACH must guard on segment = 'transformation'."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        cypher = _cypher_of(merge_calls[0])
        assert "'transformation'" in cypher
        assert "HAS_TRANSFORMATION" in cypher

    def test_multi_segment_name_edges_param_has_coordinate(
        self, mock_gc: MagicMock
    ) -> None:
        """poloidal_electron_temperature should include a coordinate token in edges_param
        (ISN grammar classifies 'poloidal' as coordinate for this name)."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        mock_gc.query.side_effect = [[], []]
        _write_segment_edges(mock_gc, ["poloidal_electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert merge_calls
        edges_param = merge_calls[0][1].get("edges", [])
        segments = {e["segment"] for e in edges_param}
        # ISN grammar: poloidal→coordinate, electron→subject, temperature→physical_base
        assert "coordinate" in segments, (
            f"poloidal_electron_temperature must have a coordinate segment; got: {segments}"
        )
        assert "subject" in segments
        assert "physical_base" in segments

    def test_token_miss_does_not_prevent_cypher_execution(
        self, mock_gc: MagicMock
    ) -> None:
        """When OPTIONAL MATCH returns no token (token-miss), the FOREACH guards
        on ``t IS NOT NULL`` prevent edge creation — but the query still returns
        matched=False without error."""
        from imas_codex.standard_names.graph_ops import _write_segment_edges

        # Simulate token-miss: MERGE returns matched=False for all tokens
        mock_gc.query.side_effect = [
            [],  # DELETE
            [
                {"token": "electron", "segment": "subject", "matched": False},
                {"token": "temperature", "segment": "physical_base", "matched": False},
            ],
        ]
        gaps = _write_segment_edges(mock_gc, ["electron_temperature"])
        # Should report gaps but not raise
        assert len(gaps) == 2
        assert all(g["sn_id"] == "electron_temperature" for g in gaps)
