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
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            return write_standard_names(names)

    def _find_merge_call(self, mock_gc: MagicMock):
        """Find the MERGE StandardName query call from the call list."""
        for c in mock_gc.query.call_args_list:
            cypher = c[0][0]
            if "MERGE (sn:StandardName" in cypher:
                return c
        raise AssertionError("No MERGE StandardName query found in calls")

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
        merge_call = self._find_merge_call(mock_gc)
        cypher = merge_call[0][0]

        # Every field SET should use coalesce pattern
        assert "coalesce(b.source_types, sn.source_types)" in cypher
        assert "coalesce(b.description, sn.description)" in cypher
        assert "coalesce(b.documentation, sn.documentation)" in cypher
        assert "coalesce(b.kind, sn.kind)" in cypher
        assert "coalesce(b.tags, sn.tags)" in cypher
        assert "coalesce(b.links, sn.links)" in cypher
        assert "coalesce(b.source_paths, sn.source_paths)" in cypher
        assert "coalesce(b.validity_domain, sn.validity_domain)" in cypher
        assert "coalesce(b.constraints, sn.constraints)" in cypher
        assert "coalesce(b.confidence, sn.confidence)" in cypher
        assert "coalesce(b.process, sn.process)" in cypher

        # created_at should use coalesce(sn.created_at, datetime()) — preserve existing
        assert "coalesce(sn.created_at, datetime())" in cypher

    def test_write_empty_returns_zero(self) -> None:
        """Empty list should return 0 without touching the graph."""
        from imas_codex.standard_names.graph_ops import write_standard_names

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
        """Names with units should create (StandardName)-[:HAS_UNIT]->(Unit)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        # Find the HAS_UNIT relationship query
        unit_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "HAS_UNIT" in str(call) and "Unit" in str(call)
        ]
        assert len(unit_calls) >= 1, "Should create HAS_UNIT relationship"

        unit_cypher = unit_calls[0][0][0]
        assert "Unit" in unit_cypher
        assert "MERGE (u:Unit" in unit_cypher
        assert "MERGE (sn)-[:HAS_UNIT]->(u)" in unit_cypher

    def test_no_unit_relationship_when_no_units(self) -> None:
        """Names without units should NOT create HAS_UNIT relationships."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "test_name",
                "source_types": ["dd"],
                "source_id": "some/path",
            }
        ]
        self._call_write(names, mock_gc)

        unit_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "MERGE (u:Unit" in str(call)
        ]
        assert len(unit_calls) == 0, "Should NOT create HAS_UNIT when no units"

    def test_rich_fields_in_batch(self, sample_standard_names: list[dict]) -> None:
        """All rich fields should be included in the batch parameter."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        self._call_write(sample_standard_names, mock_gc)

        merge_call = self._find_merge_call(mock_gc)
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
                "source_types": ["dd"],
                "source_id": "some/path",
                "tags": [],
                "links": [],
                "source_paths": [],
                "constraints": [],
            }
        ]
        self._call_write(names, mock_gc)

        merge_call = self._find_merge_call(mock_gc)
        batch = merge_call[1]["batch"]
        first = batch[0]

        # Empty lists should become None so coalesce preserves existing
        assert first["tags"] is None
        assert first["links"] is None
        assert first["source_paths"] is None
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
                    "unit": "eV",
                    "confidence": 0.95,
                    "ids_name": "core_profiles",
                }
            ]
        )

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import get_validated_standard_names

            results = get_validated_standard_names(confidence_min=0.9)

        # Verify confidence_min was passed to query
        call_kwargs = mock_gc.query.call_args[1]
        assert call_kwargs["confidence_min"] == 0.9
        assert len(results) == 1

    def test_ids_filter(self) -> None:
        """Should filter by IDS name."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import get_validated_standard_names

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
                    "unit": None,
                    "confidence": 1.0,
                    "ids_name": None,
                },
                {
                    "name": "b",
                    "description": "",
                    "source": "dd",
                    "source_path": "y",
                    "unit": None,
                    "confidence": 1.0,
                    "ids_name": None,
                },
            ]
        )

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import get_validated_standard_names

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

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import get_existing_standard_names

            result = get_existing_standard_names()

        assert isinstance(result, set)
        assert "electron_temperature" in result
        assert "plasma_current" in result
        assert len(result) == 2


# =============================================================================
# TestResetStandardNames
# =============================================================================


class TestResetStandardNames:
    """Test reset_standard_names query logic."""

    def _call_reset(self, mock_gc: MagicMock, **kwargs) -> int:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import reset_standard_names

            return reset_standard_names(**kwargs)

    def test_dry_run_returns_count_without_modifying(self) -> None:
        """dry_run=True should return count from the count query only."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 3}])

        count = self._call_reset(mock_gc, from_status="drafted", dry_run=True)

        assert count == 3
        # Only one query should be called (the count query)
        assert mock_gc.query.call_count == 1

    def test_returns_zero_for_empty_graph(self) -> None:
        """When no nodes match, reset returns 0 and makes no modification queries."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        count = self._call_reset(mock_gc, from_status="drafted")

        assert count == 0
        # Only the count query; no DELETE or SET queries
        assert mock_gc.query.call_count == 1

    def test_clears_transient_fields(self) -> None:
        """Reset should null out embedding, embedded_at, model, generated_at, confidence."""
        mock_gc = MagicMock()
        # First call = count query; subsequent calls = relationship + set queries
        mock_gc.query = MagicMock(return_value=[{"n": 2}])

        self._call_reset(mock_gc, from_status="drafted")

        # Collect all Cypher strings passed
        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)

        assert "sn.embedding = null" in all_cypher
        assert "sn.embedded_at = null" in all_cypher
        assert "sn.model = null" in all_cypher
        assert "sn.generated_at = null" in all_cypher
        assert "sn.confidence = null" in all_cypher

    def test_removes_has_standard_name_relationships(self) -> None:
        """Reset should delete HAS_STANDARD_NAME relationships."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 1}])

        self._call_reset(mock_gc, from_status="drafted")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "HAS_STANDARD_NAME" in all_cypher

    def test_removes_unit_relationships(self) -> None:
        """Reset should delete HAS_UNIT relationships."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 1}])

        self._call_reset(mock_gc, from_status="drafted")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "HAS_UNIT" in all_cypher

    def test_to_status_sets_review_status(self) -> None:
        """When to_status is given, SET clause should include review_status."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 1}])

        self._call_reset(mock_gc, from_status="drafted", to_status="extracted")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "review_status" in all_cypher

        # Verify to_status kwarg was passed
        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        to_statuses = [kw.get("to_status") for kw in all_kwargs if "to_status" in kw]
        assert "extracted" in to_statuses

    def test_source_filter_included_in_cypher(self) -> None:
        """source_filter should appear in the WHERE clause."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_reset(mock_gc, from_status="drafted", source_filter="dd")

        # Check source_filter param was passed
        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        sources = [
            kw.get("source_filter") for kw in all_kwargs if "source_filter" in kw
        ]
        assert "dd" in sources

    def test_ids_filter_uses_starts_with(self) -> None:
        """ids_filter should restrict via STARTS WITH prefix on src.id."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_reset(mock_gc, from_status="drafted", ids_filter="equilibrium")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "STARTS WITH" in all_cypher

        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        prefixes = [kw.get("ids_prefix") for kw in all_kwargs if "ids_prefix" in kw]
        assert "equilibrium/" in prefixes


# =============================================================================
# TestClearStandardNames
# =============================================================================


class TestClearStandardNames:
    """Test clear_standard_names deletion logic."""

    def _call_clear(self, mock_gc: MagicMock, **kwargs) -> int:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import clear_standard_names

            return clear_standard_names(**kwargs)

    def test_dry_run_returns_count_without_deleting(self) -> None:
        """dry_run=True should return count without issuing DELETE."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 5}])

        count = self._call_clear(mock_gc, dry_run=True)

        assert count == 5
        # Only the count query
        assert mock_gc.query.call_count == 1
        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "DELETE" not in all_cypher

    def test_default_status_filter_means_all_statuses(self) -> None:
        """status_filter=None should mean 'no filter' (all statuses, excl accepted)."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_clear(mock_gc)

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        # Should not constrain to specific status IN list
        assert "IN $statuses" not in all_cypher
        # Should still exclude accepted by default
        assert "<> 'accepted'" in all_cypher

    def test_accepted_not_deleted_without_flag(self) -> None:
        """Without include_accepted, 'accepted' should not be in statuses list."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_clear(mock_gc, status_filter=["drafted"])

        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        statuses_lists = [kw.get("statuses") for kw in all_kwargs if "statuses" in kw]
        for sl in statuses_lists:
            assert "accepted" not in sl, (
                "accepted should not appear without include_accepted"
            )

    def test_include_accepted_adds_to_statuses(self) -> None:
        """include_accepted=True should add 'accepted' to effective_statuses."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_clear(mock_gc, status_filter=["drafted"], include_accepted=True)

        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        statuses_lists = [kw.get("statuses") for kw in all_kwargs if "statuses" in kw]
        assert any("accepted" in sl for sl in statuses_lists)

    def test_returns_zero_for_empty_graph(self) -> None:
        """When no nodes match, returns 0 and makes no DELETE queries."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        count = self._call_clear(mock_gc)

        assert count == 0
        assert mock_gc.query.call_count == 1

    def test_detach_delete_without_ids_filter(self) -> None:
        """Without ids_filter, should DETACH DELETE matching nodes."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 2}])

        self._call_clear(mock_gc)

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "DETACH DELETE" in all_cypher

    def test_relationship_first_with_ids_filter(self) -> None:
        """With ids_filter, should remove relationships before deleting orphan nodes."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 3}])

        self._call_clear(mock_gc, ids_filter="core_profiles")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        # Relationship delete should appear (DELETE r pattern)
        assert "DELETE r" in all_cypher
        # Node delete should also appear
        assert "DETACH DELETE sn" in all_cypher

    def test_ids_filter_uses_starts_with(self) -> None:
        """ids_filter should use STARTS WITH prefix matching."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_clear(mock_gc, ids_filter="magnetics")

        all_cypher = " ".join(call[0][0] for call in mock_gc.query.call_args_list)
        assert "STARTS WITH" in all_cypher

        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        prefixes = [kw.get("ids_prefix") for kw in all_kwargs if "ids_prefix" in kw]
        assert "magnetics/" in prefixes

    def test_source_filter_passed_as_param(self) -> None:
        """source_filter should be passed as a query parameter."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"n": 0}])

        self._call_clear(mock_gc, source_filter="signals")

        all_kwargs = [call[1] for call in mock_gc.query.call_args_list]
        sources = [
            kw.get("source_filter") for kw in all_kwargs if "source_filter" in kw
        ]
        assert "signals" in sources
