"""Tests for the 'mixed' DD-unit source filter.

Verifies that paths with DD unit='mixed' are hard-rejected at source
extraction time via ``_apply_mixed_unit_filter``, and that the
``_is_unparseable_dd_unit`` helper no longer treats 'mixed' as a valid
dimensionless sentinel.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from imas_codex.standard_names.sources.dd import (
    _apply_mixed_unit_filter,
    _is_unparseable_dd_unit,
)


class TestMixedUnitFilter:
    """``_apply_mixed_unit_filter`` hard-rejects unit='mixed' rows."""

    def test_mixed_unit_rows_are_dropped(self) -> None:
        rows = [
            {
                "path": "equilibrium/time_slice/ggd/grid/metric/g11_covariant",
                "unit": "mixed",
                "description": "covariant metric tensor component",
            },
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "description": "electron temperature",
            },
        ]
        kept = _apply_mixed_unit_filter(rows, write_skipped=False)
        assert len(kept) == 1
        assert kept[0]["path"].endswith("/temperature")

    def test_non_mixed_units_pass_through(self) -> None:
        rows = [
            {"path": "a/b/c", "unit": "eV", "description": "d1"},
            {"path": "d/e/f", "unit": "1", "description": "d2"},
            {"path": "g/h/i", "unit": "-", "description": "d3"},
            {"path": "j/k/l", "unit": None, "description": "d4"},
        ]
        kept = _apply_mixed_unit_filter(rows, write_skipped=False)
        assert len(kept) == 4  # none of these are 'mixed'

    def test_all_mixed_returns_empty(self) -> None:
        rows = [
            {"path": "a/b", "unit": "mixed", "description": "d1"},
            {"path": "c/d", "unit": "mixed", "description": "d2"},
        ]
        kept = _apply_mixed_unit_filter(rows, write_skipped=False)
        assert kept == []

    def test_skip_records_written_to_graph(self) -> None:
        rows = [
            {
                "path": "equilibrium/time_slice/ggd/grid/metric/g22",
                "unit": "mixed",
                "description": "metric tensor",
            },
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources"
        ) as mock_write:
            mock_write.return_value = 1
            kept = _apply_mixed_unit_filter(rows, write_skipped=True)

        assert len(kept) == 0
        mock_write.assert_called_once()
        records = mock_write.call_args[0][0]
        assert len(records) == 1
        assert records[0]["skip_reason"] == "dd_unit_mixed_non_standard"
        assert "mixed" in records[0]["skip_reason_detail"]


class TestIsUnparseableDDUnitNoMixed:
    """``_is_unparseable_dd_unit`` must NOT treat 'mixed' as a valid sentinel."""

    def test_mixed_is_unparseable(self) -> None:
        # 'mixed' should be treated as unparseable (returns True)
        assert _is_unparseable_dd_unit("mixed") is True

    def test_dimensionless_sentinels_still_valid(self) -> None:
        for unit in ("1", "dimensionless", "-", "none"):
            assert _is_unparseable_dd_unit(unit) is False, (
                f"'{unit}' should be a valid dimensionless sentinel"
            )

    def test_real_units_still_valid(self) -> None:
        for unit in ("eV", "m", "Pa", "T", "A", "kg.m^-3"):
            assert _is_unparseable_dd_unit(unit) is False, (
                f"'{unit}' should be parseable"
            )
