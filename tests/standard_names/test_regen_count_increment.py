"""Regression tests: regen_count must be incremented when persisting regen output.

Bug 5 — ``regen_count`` was read in ``fetch_low_score_sources`` (line 332) to
cap regeneration iterations at 1, but was never incremented after a regen pass.
The fix adds ``regen_count = coalesce(regen_count, 0) + 1`` in
``write_standard_names`` when the candidate carries ``regen_increment=True``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRegenCountIncrement:
    """Verify regen_count is properly tracked through the persist path."""

    def _call_write(self, names: list[dict], mock_gc: MagicMock) -> int:
        """Call write_standard_names with a mocked GraphClient."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            return write_standard_names(names)

    def _find_merge_call(self, mock_gc: MagicMock):
        """Find the MERGE StandardName query call."""
        for c in mock_gc.query.call_args_list:
            cypher = c[0][0]
            if "MERGE (sn:StandardName" in cypher:
                return c
        raise AssertionError("No MERGE StandardName query found in calls")

    def test_regen_increment_in_cypher(self) -> None:
        """The MERGE Cypher must contain a regen_count CASE clause."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(
            [
                {
                    "id": "test_regen_name",
                    "source_types": ["dd"],
                    "source_id": "test/path",
                    "regen_increment": True,
                }
            ],
            mock_gc,
        )

        merge_call = self._find_merge_call(mock_gc)
        cypher = merge_call[0][0]
        assert "sn.regen_count" in cypher, "regen_count must appear in MERGE SET"
        assert "b.regen_increment" in cypher, (
            "regen_count clause must be gated on b.regen_increment"
        )

    def test_regen_increment_true_in_batch(self) -> None:
        """When regen_increment=True, batch dict must carry the flag."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(
            [
                {
                    "id": "test_regen_name",
                    "source_types": ["dd"],
                    "source_id": "test/path",
                    "regen_increment": True,
                }
            ],
            mock_gc,
        )

        merge_call = self._find_merge_call(mock_gc)
        batch = merge_call[1]["batch"]
        assert len(batch) == 1
        assert batch[0]["regen_increment"] is True

    def test_non_regen_has_null_increment(self) -> None:
        """When regen_increment is absent, batch dict must carry None."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(
            [
                {
                    "id": "test_fresh_name",
                    "source_types": ["dd"],
                    "source_id": "test/path",
                }
            ],
            mock_gc,
        )

        merge_call = self._find_merge_call(mock_gc)
        batch = merge_call[1]["batch"]
        assert len(batch) == 1
        assert batch[0]["regen_increment"] is None

    def test_regen_count_case_logic(self) -> None:
        """The CASE clause must increment only when regen_increment = true.

        Verify the exact Cypher pattern:
          CASE WHEN b.regen_increment = true
               THEN coalesce(sn.regen_count, 0) + 1
               ELSE sn.regen_count END
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(
            [
                {
                    "id": "test_regen_name",
                    "source_types": ["dd"],
                    "source_id": "test/path",
                    "regen_increment": True,
                }
            ],
            mock_gc,
        )

        merge_call = self._find_merge_call(mock_gc)
        cypher = merge_call[0][0]
        # Must have the increment logic
        assert "coalesce(sn.regen_count, 0) + 1" in cypher
        # Must be gated on the boolean flag
        assert "b.regen_increment = true" in cypher


class TestFetchLowScoreSourcesGate:
    """Verify the regen-eligibility gate reads regen_count correctly."""

    def test_gate_clause_present(self) -> None:
        """fetch_low_score_sources Cypher must gate on regen_count < 1."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            mock_gc = MagicMock()
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            mock_gc.query = MagicMock(return_value=[])

            from imas_codex.standard_names.graph_ops import fetch_low_score_sources

            fetch_low_score_sources(
                min_score=0.5,
                domain="test_domain",
                source_type="dd",
            )

            # Verify the query contains the regen_count gate
            call = mock_gc.query.call_args
            cypher = call[0][0]
            assert "coalesce(sn.regen_count, 0) < 1" in cypher
