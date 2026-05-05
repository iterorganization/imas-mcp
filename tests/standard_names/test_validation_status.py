"""Tests for the validation_status lifecycle on StandardName.

Covers:
- Schema enum values
- _is_quarantined() decision logic
- mark_names_validated() sets validation_status in Cypher
- persist_composed_batch() defaults to "pending"
- Downstream query filters (get_validated_names, get_validated_standard_names,
  review pipeline _load_all_names)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

# =============================================================================
# Schema enum
# =============================================================================


class TestValidationStatusEnum:
    """Verify the StandardNameValidationStatus enum exists and has correct values."""

    def test_enum_values(self) -> None:
        from imas_codex.graph.models import StandardNameValidationStatus

        assert StandardNameValidationStatus.pending == "pending"
        assert StandardNameValidationStatus.valid == "valid"
        assert StandardNameValidationStatus.quarantined == "quarantined"

    def test_needs_revision_removed(self) -> None:
        """`needs_revision` has been dropped — regen is driven by --min-score."""
        from imas_codex.graph.models import StandardNameValidationStatus

        assert not hasattr(StandardNameValidationStatus, "needs_revision")
        values = {m.value for m in StandardNameValidationStatus}
        assert "needs_revision" not in values

    def test_enum_member_count(self) -> None:
        from imas_codex.graph.models import StandardNameValidationStatus

        assert len(StandardNameValidationStatus) == 3

    def test_standard_name_model_has_field(self) -> None:
        """StandardName Pydantic model includes validation_status."""
        from imas_codex.graph.models import StandardName

        fields = StandardName.model_fields
        assert "validation_status" in fields


# =============================================================================
# _is_quarantined() decision logic
# =============================================================================


class TestIsQuarantined:
    """Test the _is_quarantined helper that gates valid vs quarantined."""

    def test_clean_entry_not_quarantined(self) -> None:
        from imas_codex.standard_names.workers import _is_quarantined

        issues: list[str] = []
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is False

    def test_parse_error_quarantined(self) -> None:
        from imas_codex.standard_names.workers import _is_quarantined

        issues = ["parse_error: grammar round-trip failed for bad_name"]
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is True

    def test_grammar_ambiguity_quarantined(self) -> None:
        from imas_codex.standard_names.workers import _is_quarantined

        issues = ["grammar:ambiguity:component_coordinate_overlap: some_name"]
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is True

    def test_pydantic_failure_quarantined(self) -> None:
        from imas_codex.standard_names.workers import _is_quarantined

        issues = ["[pydantic:name] bad name format"]
        summary = {"pydantic": {"passed": False, "error_count": 1}}
        assert _is_quarantined(issues, summary) is True

    def test_semantic_issues_not_quarantined(self) -> None:
        """Semantic warnings are advisory — NOT critical."""
        from imas_codex.standard_names.workers import _is_quarantined

        issues = ["[semantic] description mentions path fragment"]
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is False

    def test_description_issues_not_quarantined(self) -> None:
        """Description quality hints are advisory — NOT critical."""
        from imas_codex.standard_names.workers import _is_quarantined

        issues = ["[description] too short"]
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is False

    def test_empty_summary_not_quarantined(self) -> None:
        """Missing pydantic key defaults to passed."""
        from imas_codex.standard_names.workers import _is_quarantined

        assert _is_quarantined([], {}) is False

    def test_multiple_issues_parse_error_wins(self) -> None:
        """If any critical issue is present, quarantine."""
        from imas_codex.standard_names.workers import _is_quarantined

        issues = [
            "[semantic] minor issue",
            "parse_error: grammar round-trip failed for x",
            "[description] quality",
        ]
        summary = {"pydantic": {"passed": True, "error_count": 0}}
        assert _is_quarantined(issues, summary) is True


# =============================================================================
# mark_names_validated() Cypher includes validation_status
# =============================================================================


class TestMarkNamesValidated:
    """Verify mark_names_validated passes validation_status to graph."""

    def test_validation_status_in_cypher(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"marked": 2}])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import mark_names_validated

            results = [
                {
                    "id": "electron_temperature",
                    "validation_issues": [],
                    "validation_layer_summary": json.dumps({}),
                    "validation_status": "valid",
                },
                {
                    "id": "bad_name",
                    "validation_issues": ["parse_error: failed"],
                    "validation_layer_summary": json.dumps({}),
                    "validation_status": "quarantined",
                },
            ]
            mark_names_validated("test-token", results)

        # Verify Cypher contains validation_status
        cypher_call = mock_gc.query.call_args
        cypher = cypher_call[0][0]
        assert "sn.validation_status = b.validation_status" in cypher

        # Verify batch contains validation_status values
        batch = cypher_call[1]["batch"]
        statuses = {b["validation_status"] for b in batch}
        assert statuses == {"valid", "quarantined"}

    def test_default_validation_status(self) -> None:
        """Missing validation_status defaults to 'valid'."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"marked": 1}])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import mark_names_validated

            results = [
                {
                    "id": "electron_temperature",
                    "validation_issues": [],
                    "validation_layer_summary": json.dumps({}),
                    # No validation_status key
                },
            ]
            mark_names_validated("test-token", results)

        batch = mock_gc.query.call_args[1]["batch"]
        assert batch[0]["validation_status"] == "valid"


# =============================================================================
# =============================================================================
# write_standard_names() includes validation_status in Cypher
# =============================================================================


class TestWriteStandardNamesValidationStatus:
    """Verify write_standard_names Cypher and batch include validation_status."""

    def test_cypher_coalesce_validation_status(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(
                [
                    {
                        "id": "electron_temperature",
                        "source_types": ["dd"],
                        "source_id": "test",
                        "validation_status": "pending",
                    }
                ]
            )

        # Find the MERGE query
        for c in mock_gc.query.call_args_list:
            cypher = c[0][0]
            if "MERGE (sn:StandardName" in cypher:
                assert "coalesce(b.validation_status, sn.validation_status)" in cypher
                batch = c[1]["batch"]
                assert batch[0]["validation_status"] == "pending"
                break
        else:
            pytest.fail("No MERGE StandardName query found")


# =============================================================================
# Downstream query filters
# =============================================================================


class TestDownstreamValidationFilters:
    """Verify downstream queries include validation_status='valid' filter."""

    def test_get_validated_names_filters_valid(self) -> None:
        """consolidation query requires validation_status='valid'."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import get_validated_names

            get_validated_names()

        cypher = mock_gc.query.call_args[0][0]
        assert "sn.validation_status = 'valid'" in cypher

    def test_get_validated_standard_names_filters_valid(self) -> None:
        """publish query requires validation_status='valid'."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import (
                get_validated_standard_names,
            )

            get_validated_standard_names()

        cypher = mock_gc.query.call_args[0][0]
        assert "sn.validation_status = 'valid'" in cypher

    def test_review_pipeline_filters_valid(self) -> None:
        """Review pipeline _load_all_names query includes validation_status='valid'."""
        # Read the pipeline source file directly to verify the WHERE clause
        from pathlib import Path

        import imas_codex.standard_names.review.pipeline as pipeline_mod

        source_file = Path(pipeline_mod.__file__)
        source_text = source_file.read_text()
        assert "validation_status = 'valid'" in source_text
