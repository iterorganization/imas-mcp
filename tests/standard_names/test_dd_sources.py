"""Integration tests for DD source extraction with unit overrides.

Focuses on ``_apply_unit_overrides`` — the helper that runs after the
Cypher result set is assembled but before enrichment. Verifies that
override rules rewrite ``row['unit']`` and skip rules drop rows while
writing StandardNameSource records via graph_ops.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.sources.dd import _apply_unit_overrides


class TestApplyUnitOverrides:
    """Pure-Python behavior of ``_apply_unit_overrides``."""

    def test_override_rewrites_unit_and_records_provenance(self) -> None:
        rows = [
            {
                "path": "core_profiles/profiles_1d/ion/element/multiplicity",
                "unit": "Elementary Charge Unit",
                "description": "multiplicity of element",
            }
        ]
        kept = _apply_unit_overrides(rows, write_skipped=False)
        assert len(kept) == 1
        assert kept[0]["unit"] == "1"
        assert kept[0]["_unit_override"]["rule"] == "override"
        assert kept[0]["_unit_override"]["original_unit"] == "Elementary Charge Unit"

    def test_skip_drops_row_and_queues_skip_record(self) -> None:
        rows = [
            {
                "path": "equilibrium/time_slice/ggd/grid/space/"
                "objects_per_dimension/object/measure",
                "unit": "m^dimension",
                "description": "cell measure",
            },
            # A second row that should pass through unchanged:
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "description": "electron temp",
            },
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources"
        ) as mock_write:
            mock_write.return_value = 1
            kept = _apply_unit_overrides(rows, write_skipped=True)

        # Only the valid row survives
        assert len(kept) == 1
        assert kept[0]["path"].endswith("/temperature")

        # Graph write was called with the skipped row
        mock_write.assert_called_once()
        records = mock_write.call_args[0][0]
        assert len(records) == 1
        assert records[0]["source_type"] == "dd"
        assert records[0]["skip_reason"] == "dd_unit_unresolvable"

    def test_no_graph_write_when_no_skips(self) -> None:
        """Critical: when no rows are skipped, we must NOT call into the
        graph. Other extraction tests (e.g. gyrokinetics) mock GraphClient
        with fixed-length side_effect iterators — an unexpected extra
        query would break them."""
        rows = [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "description": "",
            }
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources"
        ) as mock_write:
            kept = _apply_unit_overrides(rows, write_skipped=True)
            mock_write.assert_not_called()
        assert len(kept) == 1

    def test_write_skipped_false_suppresses_graph_call(self) -> None:
        rows = [
            {
                "path": "pulse_schedule/ec/beam/power_launched/reference",
                "unit": "1",
                "description": "",
            }
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources"
        ) as mock_write:
            kept = _apply_unit_overrides(rows, write_skipped=False)
            mock_write.assert_not_called()
        assert kept == []

    def test_pass_through_preserves_row(self) -> None:
        rows = [
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "unit": "Wb",
                "description": "poloidal flux",
                "_extra": "preserved",
            }
        ]
        kept = _apply_unit_overrides(rows, write_skipped=False)
        assert kept[0]["unit"] == "Wb"
        assert "_unit_override" not in kept[0]
        assert kept[0]["_extra"] == "preserved"

    def test_mixed_batch(self) -> None:
        rows = [
            # override
            {
                "path": "spectrometer_mass/channel/atomic_mass",
                "unit": "Atomic Mass Unit",
                "description": "",
            },
            # skip
            {
                "path": "pulse_schedule/pf_active/coil/resistance_additional/reference",
                "unit": "1",
                "description": "",
            },
            # pass-through
            {
                "path": "equilibrium/time_slice/profiles_1d/psi",
                "unit": "Wb",
                "description": "",
            },
            # override to different unit
            {
                "path": "core_profiles/profiles_1d/ion/z_ion",
                "unit": "e",
                "description": "",
            },
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources"
        ) as mock_write:
            mock_write.return_value = 1
            kept = _apply_unit_overrides(rows, write_skipped=True)

        # 3 kept, 1 skipped
        assert len(kept) == 3
        by_path = {r["path"]: r for r in kept}
        assert by_path["spectrometer_mass/channel/atomic_mass"]["unit"] == "u"
        assert by_path["equilibrium/time_slice/profiles_1d/psi"]["unit"] == "Wb"
        assert by_path["core_profiles/profiles_1d/ion/z_ion"]["unit"] == "1"

        mock_write.assert_called_once()
        records = mock_write.call_args[0][0]
        assert len(records) == 1
        assert "pulse_schedule" in records[0]["source_id"]

    def test_empty_input(self) -> None:
        assert _apply_unit_overrides([], write_skipped=True) == []


class TestApplyUnitOverridesErrorHandling:
    """Graph failures must not block extraction."""

    def test_graph_exception_is_logged_not_raised(self, caplog) -> None:
        rows = [
            {
                "path": "equilibrium/time_slice/ggd/grid/space/"
                "objects_per_dimension/object/measure",
                "unit": "m^dimension",
                "description": "",
            }
        ]
        with patch(
            "imas_codex.standard_names.graph_ops.write_skipped_sources",
            side_effect=RuntimeError("neo4j unreachable"),
        ):
            import logging as _logging

            with caplog.at_level(_logging.WARNING):
                kept = _apply_unit_overrides(rows, write_skipped=True)

        # The skip still took effect on the returned list
        assert kept == []
        # A warning was logged
        assert any(
            "Failed to write skipped DD sources" in rec.message
            for rec in caplog.records
        )
