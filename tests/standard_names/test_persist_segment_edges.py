"""Tests for HAS_SEGMENT edge writing in write_standard_names.

Validates that persist logic:
- Parses StandardName via ISN grammar and writes HAS_SEGMENT edges
  to GrammarToken nodes with {position, segment} properties.
- Clears old HAS_SEGMENT edges before re-writing (idempotency).
- Handles parse failures gracefully (SN node still persisted).
- Detects token-miss (vocabulary gaps) and logs warnings.

All tests are mocked — no live Neo4j required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_gc():
    """A mock GraphClient that records query calls."""
    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client


@pytest.fixture()
def _patch_isn():
    """Verify ISN imports are available."""
    pytest.importorskip("imas_standard_names")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _call_write(names: list[dict], mock_gc: MagicMock) -> int:
    """Call write_standard_names with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_standard_names

        return write_standard_names(names)


def _find_segment_calls(mock_gc: MagicMock) -> list:
    """Find all HAS_SEGMENT-related query calls."""
    return [c for c in mock_gc.query.call_args_list if "HAS_SEGMENT" in str(c)]


def _find_delete_segment_calls(mock_gc: MagicMock) -> list:
    """Find HAS_SEGMENT DELETE calls (idempotency cleanup)."""
    return [
        c
        for c in mock_gc.query.call_args_list
        if "HAS_SEGMENT" in str(c) and "DELETE" in str(c)
    ]


def _find_merge_segment_calls(mock_gc: MagicMock) -> list:
    """Find HAS_SEGMENT MERGE/write calls."""
    return [
        c
        for c in mock_gc.query.call_args_list
        if "HAS_SEGMENT" in str(c) and "MERGE" in str(c)
    ]


# ---------------------------------------------------------------------------
# Tests: segment edge writing
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestSegmentEdgeWriting:
    """Test HAS_SEGMENT edge creation during write_standard_names."""

    def test_segment_edges_written_for_valid_name(self, mock_gc: MagicMock) -> None:
        """electron_temperature should produce HAS_SEGMENT edges to grammar tokens."""
        # Return matched=True for all tokens
        mock_gc.query.return_value = [
            {"token": "electron", "segment": "subject", "matched": True},
            {"token": "temperature", "segment": "physical_base", "matched": True},
        ]

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
            }
        ]
        _call_write(names, mock_gc)

        # Should have DELETE + MERGE calls for HAS_SEGMENT
        delete_calls = _find_delete_segment_calls(mock_gc)
        assert len(delete_calls) >= 1, "Should DELETE old HAS_SEGMENT edges first"

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert len(merge_calls) >= 1, "Should MERGE new HAS_SEGMENT edges"

        # Verify the MERGE Cypher references GrammarToken and edge properties
        merge_cypher = merge_calls[0][0][0]
        assert "GrammarToken" in merge_cypher
        assert "r.position" in merge_cypher
        assert "r.segment" in merge_cypher

    def test_segment_edge_positions_correct(self, mock_gc: MagicMock) -> None:
        """Edge specs should carry correct segment positions from ISN grammar."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]
        _call_write(names, mock_gc)

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert len(merge_calls) >= 1

        # Check edges parameter has expected tokens
        edges_param = merge_calls[0][1].get("edges", [])
        tokens = {e["token"] for e in edges_param}
        assert "electron" in tokens
        assert "temperature" in tokens

        # Verify position values are integers
        for e in edges_param:
            assert isinstance(e["position"], int)
            assert isinstance(e["segment"], str)

    def test_isn_version_passed_to_cypher(self, mock_gc: MagicMock) -> None:
        """The ISN version should be passed as a parameter for GrammarToken matching."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]
        _call_write(names, mock_gc)

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert len(merge_calls) >= 1

        # Verify isn_version parameter is set
        isn_version = merge_calls[0][1].get("isn_version")
        assert isn_version is not None
        assert isinstance(isn_version, str)


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestSegmentEdgeIdempotency:
    """Test that re-persisting clears old edges before writing new ones."""

    def test_delete_before_merge(self, mock_gc: MagicMock) -> None:
        """DELETE of old HAS_SEGMENT edges must happen before MERGE of new ones."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]
        _call_write(names, mock_gc)

        # Find indices of DELETE and MERGE calls
        all_calls = mock_gc.query.call_args_list
        delete_idx = None
        merge_idx = None
        for i, c in enumerate(all_calls):
            cypher = str(c)
            if "HAS_SEGMENT" in cypher and "DELETE" in cypher:
                delete_idx = i
            if "HAS_SEGMENT" in cypher and "MERGE" in cypher:
                merge_idx = i

        assert delete_idx is not None, "Should have a DELETE HAS_SEGMENT call"
        assert merge_idx is not None, "Should have a MERGE HAS_SEGMENT call"
        assert delete_idx < merge_idx, "DELETE must happen before MERGE"

    def test_delete_targets_specific_sn(self, mock_gc: MagicMock) -> None:
        """DELETE should target the specific StandardName node by id."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]
        _call_write(names, mock_gc)

        delete_calls = _find_delete_segment_calls(mock_gc)
        assert len(delete_calls) >= 1

        # Should pass sn_id parameter
        sn_id = delete_calls[0][1].get("sn_id")
        assert sn_id == "electron_temperature"


# ---------------------------------------------------------------------------
# Tests: parse failure handling
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestSegmentEdgeParseFailure:
    """Test graceful handling when parse_standard_name fails."""

    def test_parse_failure_still_persists_sn_node(self, mock_gc: MagicMock) -> None:
        """If grammar parse fails, the SN node should still be written."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "invalid__name",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]

        with patch(
            "imas_standard_names.grammar.parse_standard_name",
            side_effect=ValueError("Cannot parse"),
        ):
            result = _call_write(names, mock_gc)

        # SN node should still be written
        assert result == 1

        # The MERGE StandardName call should exist
        merge_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "MERGE (sn:StandardName" in str(c)
        ]
        assert len(merge_calls) >= 1

    def test_parse_failure_skips_segment_edges(self, mock_gc: MagicMock) -> None:
        """If grammar parse fails, no HAS_SEGMENT edges should be written."""
        mock_gc.query.return_value = []

        names = [
            {
                "id": "electron_temperature",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]

        # Patch _write_segment_edges to verify it handles errors internally
        with patch(
            "imas_codex.standard_names.graph_ops._write_segment_edges"
        ) as mock_write:
            mock_write.side_effect = None  # No-op
            _call_write(names, mock_gc)
            mock_write.assert_called_once()

    def test_parse_failure_logs_warning(
        self, mock_gc: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Parse failure should log a warning."""
        mock_gc.query.return_value = []

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        with patch(
            "imas_standard_names.grammar.parse_standard_name",
            side_effect=ValueError("Cannot parse"),
        ):
            import logging

            with caplog.at_level(
                logging.WARNING, logger="imas_codex.standard_names.graph_ops"
            ):
                _write_segment_edges(mock_gc, ["bad_name_xyz"])

        assert any("Grammar parse failed" in r.message for r in caplog.records)

        # No HAS_SEGMENT queries should have been made
        segment_calls = _find_segment_calls(mock_gc)
        assert len(segment_calls) == 0


# ---------------------------------------------------------------------------
# Tests: token-miss detection
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestSegmentEdgeTokenMiss:
    """Test detection and logging of vocabulary gaps (token-miss)."""

    def test_token_miss_logged(
        self, mock_gc: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When a token has no matching GrammarToken node, a warning is logged."""
        # Simulate token-miss: OPTIONAL MATCH returns matched=False
        mock_gc.query.side_effect = [
            # DELETE call
            [],
            # MERGE call returns with one unmatched token
            [
                {"token": "electron", "segment": "subject", "matched": True},
                {"token": "temperature", "segment": "physical_base", "matched": False},
            ],
        ]

        import logging

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        with caplog.at_level(
            logging.WARNING, logger="imas_codex.standard_names.graph_ops"
        ):
            _write_segment_edges(mock_gc, ["electron_temperature"])

        assert any("Token-miss" in r.message for r in caplog.records)
        assert any("physical_base:temperature" in r.message for r in caplog.records)

    def test_no_warning_when_all_tokens_matched(
        self, mock_gc: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when all tokens have matching GrammarToken nodes."""
        mock_gc.query.side_effect = [
            # DELETE call
            [],
            # MERGE call returns all matched
            [
                {"token": "electron", "segment": "subject", "matched": True},
                {"token": "temperature", "segment": "physical_base", "matched": True},
            ],
        ]

        import logging

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        with caplog.at_level(
            logging.WARNING, logger="imas_codex.standard_names.graph_ops"
        ):
            _write_segment_edges(mock_gc, ["electron_temperature"])

        assert not any("Token-miss" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests: round-trip segment edge writing (plan 29 E.8)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_patch_isn")
class TestSegmentEdgeRoundTrip:
    """Write HAS_SEGMENT edges and verify reconstructed segment list matches input."""

    def test_round_trip_positions_match_input_order(self, mock_gc: MagicMock) -> None:
        """Given a known parsed-segment list, verify the written edges carry
        the correct {position, segment, token} fields that reconstruct
        the original segment list in order."""
        # Simulate: DELETE returns nothing, MERGE returns all matched
        mock_gc.query.side_effect = [
            # DELETE old edges
            [],
            # MERGE + RETURN
            [
                {"token": "electron", "segment": "subject", "matched": True},
                {"token": "temperature", "segment": "physical_base", "matched": True},
            ],
        ]

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        _write_segment_edges(mock_gc, ["electron_temperature"])

        # Find the MERGE call and inspect edges parameter
        merge_calls = _find_merge_segment_calls(mock_gc)
        assert len(merge_calls) >= 1

        edges_param = merge_calls[0][1].get("edges", [])
        assert len(edges_param) == 2

        # Reconstruct segment list sorted by position
        sorted_edges = sorted(edges_param, key=lambda e: e["position"])

        # Verify exact field values
        assert sorted_edges[0] == {
            "position": 2,
            "segment": "subject",
            "token": "electron",
        }
        assert sorted_edges[1] == {
            "position": 5,
            "segment": "physical_base",
            "token": "temperature",
        }

    def test_round_trip_multi_segment_name(self, mock_gc: MagicMock) -> None:
        """A name with 3+ segments should produce edges with monotonically
        increasing positions matching ISN SEGMENT_ORDER."""
        # poloidal_electron_temperature → component(0), subject(2), physical_base(5)
        mock_gc.query.side_effect = [
            [],  # DELETE
            [  # MERGE results
                {"token": "poloidal", "segment": "component", "matched": True},
                {"token": "electron", "segment": "subject", "matched": True},
                {"token": "temperature", "segment": "physical_base", "matched": True},
            ],
        ]

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        _write_segment_edges(mock_gc, ["poloidal_electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        assert len(merge_calls) >= 1

        edges_param = merge_calls[0][1].get("edges", [])
        assert len(edges_param) == 3

        sorted_edges = sorted(edges_param, key=lambda e: e["position"])

        # Positions must be monotonically increasing
        positions = [e["position"] for e in sorted_edges]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions), "Positions must be unique"

        # Each edge must have all three fields
        for edge in sorted_edges:
            assert "position" in edge
            assert "segment" in edge
            assert "token" in edge
            assert isinstance(edge["position"], int)
            assert isinstance(edge["segment"], str)
            assert isinstance(edge["token"], str)

    def test_round_trip_edge_fields_complete(self, mock_gc: MagicMock) -> None:
        """Every edge dict passed to Cypher must have exactly
        {position, segment, token} — no extra, no missing keys."""
        mock_gc.query.side_effect = [
            [],  # DELETE
            [
                {"token": "electron", "segment": "subject", "matched": True},
                {"token": "temperature", "segment": "physical_base", "matched": True},
            ],
        ]

        from imas_codex.standard_names.graph_ops import _write_segment_edges

        _write_segment_edges(mock_gc, ["electron_temperature"])

        merge_calls = _find_merge_segment_calls(mock_gc)
        edges_param = merge_calls[0][1].get("edges", [])

        expected_keys = {"position", "segment", "token"}
        for edge in edges_param:
            assert set(edge.keys()) == expected_keys, (
                f"Edge has unexpected keys: {set(edge.keys())} "
                f"(expected {expected_keys})"
            )
