"""Tests for standard name graph operations.

Tests write_standard_names coalesce behavior, relationship creation,
get_validated_standard_names filtering, and reset/clear filter plumbing
— all mocked, no live Neo4j.
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


# =============================================================================
# TestCocosScalarDefaulting — Fix #3 from D.3 senior review §4.3
# =============================================================================


class TestCocosScalarDefaulting:
    """Test that persist_composed_batch defaults cocos_transformation_type
    to ``one_like`` for safe scalar quantities."""

    def test_safe_scalar_gets_one_like(self) -> None:
        """A scalar with a safe unit and no prior COCOS type gets ``one_like``."""
        from imas_codex.standard_names.graph_ops import (
            SAFE_SCALAR_COCOS_UNITS,
            persist_composed_batch,
        )

        # Verify the constant is accessible and non-empty
        assert len(SAFE_SCALAR_COCOS_UNITS) > 0
        assert "eV" in SAFE_SCALAR_COCOS_UNITS

        candidates = [
            {
                "id": "electron_temperature",
                "kind": "scalar",
                "unit": "eV",
                # no cocos_transformation_type set
            }
        ]

        with patch(
            "imas_codex.standard_names.graph_ops.write_standard_names"
        ) as mock_w:
            mock_w.return_value = 1
            with patch("imas_codex.embeddings.description.embed_descriptions_batch"):
                persist_composed_batch(candidates, compose_model="test/model")

        assert candidates[0]["cocos_transformation_type"] == "one_like"

    def test_existing_cocos_type_not_overridden(self) -> None:
        """A scalar with an existing (non-one_like) COCOS type keeps it."""
        from imas_codex.standard_names.graph_ops import persist_composed_batch

        candidates = [
            {
                "id": "poloidal_magnetic_flux",
                "kind": "scalar",
                "unit": "eV",
                "cocos_transformation_type": "psi_like",
            }
        ]

        with patch(
            "imas_codex.standard_names.graph_ops.write_standard_names"
        ) as mock_w:
            mock_w.return_value = 1
            with patch("imas_codex.embeddings.description.embed_descriptions_batch"):
                persist_composed_batch(candidates, compose_model="test/model")

        assert candidates[0]["cocos_transformation_type"] == "psi_like"

    def test_vector_not_defaulted(self) -> None:
        """A vector quantity does NOT get ``one_like`` defaulted."""
        from imas_codex.standard_names.graph_ops import persist_composed_batch

        candidates = [
            {
                "id": "position_of_magnetic_axis",
                "kind": "vector",
                "unit": "m",
                # no cocos_transformation_type
            }
        ]

        with patch(
            "imas_codex.standard_names.graph_ops.write_standard_names"
        ) as mock_w:
            mock_w.return_value = 1
            with patch("imas_codex.embeddings.description.embed_descriptions_batch"):
                persist_composed_batch(candidates, compose_model="test/model")

        assert candidates[0].get("cocos_transformation_type") is None

    def test_unsafe_unit_not_defaulted(self) -> None:
        """A scalar with an unsafe unit (Wb, T, A) does NOT get defaulted."""
        from imas_codex.standard_names.graph_ops import persist_composed_batch

        for unsafe_unit in ("Wb", "T", "A"):
            candidates = [
                {
                    "id": "test_quantity",
                    "kind": "scalar",
                    "unit": unsafe_unit,
                    # no cocos_transformation_type
                }
            ]

            with patch(
                "imas_codex.standard_names.graph_ops.write_standard_names"
            ) as mock_w:
                mock_w.return_value = 1
                with patch(
                    "imas_codex.embeddings.description.embed_descriptions_batch"
                ):
                    persist_composed_batch(candidates, compose_model="test/model")

            assert candidates[0].get("cocos_transformation_type") is None, (
                f"Unit {unsafe_unit} should NOT default to one_like"
            )


class TestResetStandardNamesFilters:
    """Test that reset_standard_names builds Cypher WHERE clauses for new filters."""

    def _call_reset(self, mock_gc: MagicMock, **kwargs) -> int:
        """Call reset_standard_names with a mocked GraphClient."""
        # Return count > 0 from count query so reset runs, then 0 for remainder
        mock_gc.query = MagicMock(side_effect=lambda cypher, **kw: [{"n": 5}])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import reset_standard_names

            return reset_standard_names(**kwargs)

    def _get_count_cypher(self, mock_gc: MagicMock) -> str:
        """Extract the first Cypher query (the count query)."""
        return mock_gc.query.call_args_list[0][0][0]

    def _get_count_params(self, mock_gc: MagicMock) -> dict:
        """Extract params from the first Cypher query."""
        return mock_gc.query.call_args_list[0][1]

    def test_since_filter(self) -> None:
        """--since should add a generated_at >= datetime() clause."""
        mock_gc = MagicMock()
        self._call_reset(mock_gc, from_status="drafted", since="2026-04-19T10:00")
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "datetime($since)" in cypher
        assert params["since"] == "2026-04-19T10:00"

    def test_before_filter(self) -> None:
        """--before should add a generated_at < datetime() clause."""
        mock_gc = MagicMock()
        self._call_reset(mock_gc, from_status="drafted", before="2026-05-01")
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "datetime($before)" in cypher
        assert params["before"] == "2026-05-01"

    def test_below_score_filter(self) -> None:
        """--below-score should add a reviewer_score < clause."""
        mock_gc = MagicMock()
        self._call_reset(mock_gc, from_status="drafted", below_score=0.6)
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "sn.reviewer_score < $below_score" in cypher
        assert params["below_score"] == 0.6

    def test_tiers_filter(self) -> None:
        """--tier should add a review_tier IN clause."""
        mock_gc = MagicMock()
        self._call_reset(mock_gc, from_status="drafted", tiers=["poor", "adequate"])
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "sn.review_tier IN $tiers" in cypher
        assert params["tiers"] == ["poor", "adequate"]

    def test_validation_status_filter(self) -> None:
        """--validation_status should add a validation_status = clause."""
        mock_gc = MagicMock()
        self._call_reset(
            mock_gc, from_status="drafted", validation_status="quarantined"
        )
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "sn.validation_status = $validation_status" in cypher
        assert params["validation_status"] == "quarantined"

    def test_combined_filters(self) -> None:
        """Multiple filters should all appear in the Cypher WHERE clause."""
        mock_gc = MagicMock()
        self._call_reset(
            mock_gc,
            from_status="drafted",
            since="2026-04-01",
            below_score=0.5,
            tiers=["poor"],
        )
        cypher = self._get_count_cypher(mock_gc)
        assert "datetime($since)" in cypher
        assert "sn.reviewer_score < $below_score" in cypher
        assert "sn.review_tier IN $tiers" in cypher

    def test_no_filters_backward_compat(self) -> None:
        """Without new filters, Cypher should only contain from_status clause."""
        mock_gc = MagicMock()
        self._call_reset(mock_gc, from_status="drafted")
        cypher = self._get_count_cypher(mock_gc)
        assert "sn.review_status = $from_status" in cypher
        assert "datetime" not in cypher
        assert "reviewer_score" not in cypher
        assert "review_tier" not in cypher
        assert "validation_status" not in cypher


class TestClearStandardNamesFilters:
    """Test that clear_standard_names builds Cypher WHERE clauses for new filters."""

    def _call_clear(self, mock_gc: MagicMock, **kwargs) -> int:
        """Call clear_standard_names with a mocked GraphClient."""
        mock_gc.query = MagicMock(side_effect=lambda cypher, **kw: [{"n": 3}])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import clear_standard_names

            return clear_standard_names(**kwargs)

    def _get_count_cypher(self, mock_gc: MagicMock) -> str:
        """Extract the first Cypher query (the count query)."""
        return mock_gc.query.call_args_list[0][0][0]

    def _get_count_params(self, mock_gc: MagicMock) -> dict:
        """Extract params from the first Cypher query."""
        return mock_gc.query.call_args_list[0][1]

    def test_since_filter(self) -> None:
        """--since filter should appear in clear Cypher."""
        mock_gc = MagicMock()
        self._call_clear(mock_gc, status_filter=["drafted"], since="2026-04-19T10:00")
        cypher = self._get_count_cypher(mock_gc)
        params = self._get_count_params(mock_gc)
        assert "datetime($since)" in cypher
        assert params["since"] == "2026-04-19T10:00"

    def test_below_score_filter(self) -> None:
        """--below-score filter should appear in clear Cypher."""
        mock_gc = MagicMock()
        self._call_clear(mock_gc, status_filter=["drafted"], below_score=0.6)
        cypher = self._get_count_cypher(mock_gc)
        assert "sn.reviewer_score < $below_score" in cypher

    def test_tiers_filter(self) -> None:
        """--tier filter should appear in clear Cypher."""
        mock_gc = MagicMock()
        self._call_clear(mock_gc, status_filter=["drafted"], tiers=["poor"])
        cypher = self._get_count_cypher(mock_gc)
        assert "sn.review_tier IN $tiers" in cypher


class TestWriteSkippedSources:
    """Test write_skipped_sources Cypher structure and behavior."""

    def _call_write(self, records: list[dict], mock_gc: MagicMock) -> int:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_skipped_sources

            return write_skipped_sources(records)

    def test_empty_returns_zero_no_graph_call(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock()
        count = self._call_write([], mock_gc)
        assert count == 0
        mock_gc.query.assert_not_called()

    def test_write_dd_skipped_source(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"affected": 1}])

        count = self._call_write(
            [
                {
                    "source_type": "dd",
                    "source_id": "equilibrium/time_slice/ggd/space/object/measure",
                    "skip_reason": "dd_unit_unresolvable",
                    "skip_reason_detail": "Jinja template: m^dimension",
                    "description": "cell measure",
                }
            ],
            mock_gc,
        )
        assert count == 1
        mock_gc.query.assert_called_once()
        cypher, _ = mock_gc.query.call_args[0], mock_gc.query.call_args[1]
        cypher_text = cypher[0]
        assert "MERGE (sns:StandardNameSource" in cypher_text
        assert "sns.status = 'skipped'" in cypher_text
        assert "sns.skip_reason = src.skip_reason" in cypher_text
        # claim fields cleared on match
        assert "sns.claim_token = null" in cypher_text
        # relationship creation for DD sources
        assert "MERGE (imas:IMASNode" in cypher_text
        assert "FROM_DD_PATH" in cypher_text

        sources = mock_gc.query.call_args[1]["sources"]
        assert len(sources) == 1
        assert sources[0]["id"] == "dd:equilibrium/time_slice/ggd/space/object/measure"
        assert sources[0]["source_type"] == "dd"
        assert sources[0]["dd_path"] == sources[0]["source_id"]
        assert sources[0]["signal"] is None
        assert sources[0]["skip_reason"] == "dd_unit_unresolvable"

    def test_write_signals_skipped_source(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"affected": 1}])

        self._call_write(
            [
                {
                    "source_type": "signals",
                    "source_id": "tcv:ip_raw",
                    "skip_reason": "unavailable",
                    "skip_reason_detail": "no data in MDSplus",
                }
            ],
            mock_gc,
        )
        sources = mock_gc.query.call_args[1]["sources"]
        assert sources[0]["source_type"] == "signals"
        assert sources[0]["signal"] == "tcv:ip_raw"
        assert sources[0]["dd_path"] is None

    def test_batch_writes(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"affected": 3}])
        count = self._call_write(
            [
                {
                    "source_type": "dd",
                    "source_id": f"ids/path_{i}",
                    "skip_reason": "dd_unit_context_dependent",
                    "skip_reason_detail": "sentinel unit",
                }
                for i in range(3)
            ],
            mock_gc,
        )
        assert count == 3
        assert len(mock_gc.query.call_args[1]["sources"]) == 3


class TestListSkippedSources:
    """Test list_skipped_sources filtering."""

    def _call(self, mock_gc: MagicMock, **kwargs) -> list[dict]:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import list_skipped_sources

            return list_skipped_sources(**kwargs)

    def test_default_filter_is_skipped_only(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        self._call(mock_gc)
        cypher = mock_gc.query.call_args[0][0]
        assert "sns.status = 'skipped'" in cypher
        assert "AND sns.skip_reason = $reason" not in cypher

    def test_reason_filter_passes_parameter(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        self._call(mock_gc, reason="dd_unit_unresolvable", limit=50)
        cypher = mock_gc.query.call_args[0][0]
        assert "AND sns.skip_reason = $reason" in cypher
        kwargs = mock_gc.query.call_args[1]
        assert kwargs["reason"] == "dd_unit_unresolvable"
        assert kwargs["limit"] == 50


class TestGetSkippedSourceCounts:
    """Test get_skipped_source_counts aggregation."""

    def _call(self, mock_gc: MagicMock, **kwargs) -> dict[str, int]:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import (
                get_skipped_source_counts,
            )

            return get_skipped_source_counts(**kwargs)

    def test_returns_dict_keyed_by_reason(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {"skip_reason": "dd_unit_unresolvable", "cnt": 140},
                {"skip_reason": "dd_unit_context_dependent", "cnt": 1263},
            ]
        )
        result = self._call(mock_gc)
        assert result == {
            "dd_unit_unresolvable": 140,
            "dd_unit_context_dependent": 1263,
        }

    def test_source_type_filter(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        self._call(mock_gc, source_type="dd")
        cypher = mock_gc.query.call_args[0][0]
        assert "AND sns.source_type = $source_type" in cypher
        assert mock_gc.query.call_args[1]["source_type"] == "dd"
