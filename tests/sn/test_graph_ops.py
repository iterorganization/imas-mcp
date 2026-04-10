"""Tests for standard name graph operations.

Tests write_standard_names coalesce behavior, relationship creation,
and get_validated_standard_names filtering — all mocked, no live Neo4j.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestWriteStandardNames:
    """Test write_standard_names Cypher generation and coalesce behavior."""

    def _call_write(self, names: list[dict], mock_gc: MagicMock) -> int:
        """Call write_standard_names with a mocked GraphClient."""
        with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.sn.graph_ops import write_standard_names

            return write_standard_names(names)

    def test_write_coalesce_preserves_existing(
        self, sample_standard_names: list[dict]
    ) -> None:
        """Re-running write with None fields should NOT overwrite existing data.

        The Cypher must use coalesce(b.field, sn.field) so that passing
        None for a field preserves whatever is already in the graph.
        """
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # First write: all fields populated
        self._call_write(sample_standard_names, mock_gc)

        # Verify MERGE query uses coalesce
        merge_call = mock_gc.query.call_args_list[0]
        cypher = merge_call[0][0]

        # Every field SET should use coalesce pattern
        assert "coalesce(b.source_type, sn.source_type)" in cypher
        assert "coalesce(b.description, sn.description)" in cypher
        assert "coalesce(b.documentation, sn.documentation)" in cypher
        assert "coalesce(b.kind, sn.kind)" in cypher
        assert "coalesce(b.tags, sn.tags)" in cypher
        assert "coalesce(b.links, sn.links)" in cypher
        assert "coalesce(b.imas_paths, sn.imas_paths)" in cypher
        assert "coalesce(b.validity_domain, sn.validity_domain)" in cypher
        assert "coalesce(b.constraints, sn.constraints)" in cypher
        assert "coalesce(b.confidence, sn.confidence)" in cypher
        assert "coalesce(b.process, sn.process)" in cypher

        # created_at should use coalesce(sn.created_at, datetime()) — preserve existing
        assert "coalesce(sn.created_at, datetime())" in cypher

    def test_write_empty_returns_zero(self) -> None:
        """Empty list should return 0 without touching the graph."""
        from imas_codex.sn.graph_ops import write_standard_names

        result = write_standard_names([])
        assert result == 0

    def test_dd_relationship_created(self, sample_standard_names: list[dict]) -> None:
        """DD-sourced names should create (IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        # Find the DD relationship query
        dd_calls = [
            call for call in mock_gc.query.call_args_list if "IMASNode" in str(call)
        ]
        assert len(dd_calls) >= 1, "Should create DD HAS_STANDARD_NAME relationship"

        dd_cypher = dd_calls[0][0][0]
        assert "HAS_STANDARD_NAME" in dd_cypher
        assert "MEASURES" not in dd_cypher  # Old relationship name must not appear
        assert "IMASNode" in dd_cypher

    def test_signal_relationship_created(
        self, sample_standard_names: list[dict]
    ) -> None:
        """Signal-sourced names should create (FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        # Find the signal relationship query
        signal_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "FacilitySignal" in str(call)
        ]
        assert len(signal_calls) >= 1, (
            "Should create signal HAS_STANDARD_NAME relationship"
        )

        signal_cypher = signal_calls[0][0][0]
        assert "HAS_STANDARD_NAME" in signal_cypher
        assert "MEASURES" not in signal_cypher
        assert "FacilitySignal" in signal_cypher

    def test_unit_relationship_created(self, sample_standard_names: list[dict]) -> None:
        """Names with units should create (StandardName)-[:CANONICAL_UNITS]->(Unit)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        # Find the CANONICAL_UNITS relationship query
        unit_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "CANONICAL_UNITS" in str(call)
        ]
        assert len(unit_calls) >= 1, "Should create CANONICAL_UNITS relationship"

        unit_cypher = unit_calls[0][0][0]
        assert "Unit" in unit_cypher
        assert "MERGE (u:Unit" in unit_cypher
        assert "MERGE (sn)-[:CANONICAL_UNITS]->(u)" in unit_cypher

    def test_no_unit_relationship_when_no_units(self) -> None:
        """Names without units should NOT create CANONICAL_UNITS relationships."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "test_name",
                "source_type": "dd",
                "source_id": "some/path",
            }
        ]
        self._call_write(names, mock_gc)

        unit_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "CANONICAL_UNITS" in str(call)
        ]
        assert len(unit_calls) == 0, "Should NOT create CANONICAL_UNITS when no units"

    def test_rich_fields_in_batch(self, sample_standard_names: list[dict]) -> None:
        """All rich fields should be included in the batch parameter."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        merge_call = mock_gc.query.call_args_list[0]
        batch = merge_call[1]["batch"]

        # First entry should have all rich fields
        first = batch[0]
        assert first["id"] == "electron_temperature"
        assert first["documentation"] is not None
        assert first["kind"] == "scalar"
        assert (
            "core_profiles" in (first.get("tags") or [])
            or first.get("tags") is not None
        )
        assert first["validity_domain"] == "core plasma"

    def test_empty_lists_become_none(self) -> None:
        """Empty list fields should be converted to None for coalesce to work."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "test_name",
                "source_type": "dd",
                "source_id": "some/path",
                "tags": [],
                "links": [],
                "imas_paths": [],
                "constraints": [],
            }
        ]
        self._call_write(names, mock_gc)

        merge_call = mock_gc.query.call_args_list[0]
        batch = merge_call[1]["batch"]
        first = batch[0]

        # Empty lists should become None so coalesce preserves existing
        assert first["tags"] is None
        assert first["links"] is None
        assert first["imas_paths"] is None
        assert first["constraints"] is None


class TestGetValidatedStandardNames:
    """Test get_validated_standard_names query filtering."""

    def test_confidence_filter(self) -> None:
        """Should filter by minimum confidence."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "electron_temperature",
                    "description": "Te",
                    "source": "dd",
                    "source_path": "core_profiles/profiles_1d/electrons/temperature",
                    "canonical_units": "eV",
                    "confidence": 0.95,
                    "ids_name": "core_profiles",
                }
            ]
        )

        with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.sn.graph_ops import get_validated_standard_names

            results = get_validated_standard_names(confidence_min=0.9)

        # Verify confidence_min was passed to query
        call_kwargs = mock_gc.query.call_args[1]
        assert call_kwargs["confidence_min"] == 0.9
        assert len(results) == 1

    def test_ids_filter(self) -> None:
        """Should filter by IDS name."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.sn.graph_ops import get_validated_standard_names

            get_validated_standard_names(ids_filter="equilibrium")

        # Verify ids_filter was passed to query
        call_kwargs = mock_gc.query.call_args[1]
        assert call_kwargs["ids_filter"] == "equilibrium"

        # Verify the Cypher includes the IDS filter clause
        cypher = mock_gc.query.call_args[0][0]
        assert "ids_filter" in cypher

    def test_no_filters(self) -> None:
        """With no filters, should return all standard names."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "name": "a",
                    "description": "",
                    "source": "dd",
                    "source_path": "x",
                    "canonical_units": None,
                    "confidence": 1.0,
                    "ids_name": None,
                },
                {
                    "name": "b",
                    "description": "",
                    "source": "dd",
                    "source_path": "y",
                    "canonical_units": None,
                    "confidence": 1.0,
                    "ids_name": None,
                },
            ]
        )

        with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.sn.graph_ops import get_validated_standard_names

            results = get_validated_standard_names()

        assert len(results) == 2


class TestGetExistingStandardNames:
    """Test deduplication query."""

    def test_returns_set_of_ids(self) -> None:
        """Should return a set of standard name IDs."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {"id": "electron_temperature"},
                {"id": "plasma_current"},
            ]
        )

        with patch("imas_codex.sn.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.sn.graph_ops import get_existing_standard_names

            result = get_existing_standard_names()

        assert isinstance(result, set)
        assert "electron_temperature" in result
        assert "plasma_current" in result
        assert len(result) == 2
