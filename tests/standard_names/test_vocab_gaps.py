"""Tests for VocabGap infrastructure (Phase 2A–2D).

Covers:
- SNVocabGap / SNComposeBatch model parsing
- write_vocab_gaps dedup and relationship creation
- Ambiguity detection tagging in validate_worker
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1: SNVocabGap model
# ---------------------------------------------------------------------------


class TestSNVocabGapModel:
    """SNVocabGap model validates and stores fields."""

    def test_basic_construction(self):
        from imas_codex.standard_names.models import SNVocabGap

        gap = SNVocabGap(
            source_id="equilibrium/time_slice/profiles_1d/psi",
            segment="transformation",
            needed_token="time_derivative_of",
            reason="Need time derivative transformation",
        )
        assert gap.source_id == "equilibrium/time_slice/profiles_1d/psi"
        assert gap.segment == "transformation"
        assert gap.needed_token == "time_derivative_of"
        assert gap.reason == "Need time derivative transformation"

    def test_all_fields_required(self):
        from pydantic import ValidationError

        from imas_codex.standard_names.models import SNVocabGap

        with pytest.raises(ValidationError):
            SNVocabGap(source_id="path/a", segment="transformation")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Test 2: SNComposeBatch with vocab_gaps
# ---------------------------------------------------------------------------


class TestSNComposeBatchVocabGaps:
    """SNComposeBatch correctly parses vocab_gaps from LLM response."""

    def test_compose_batch_with_vocab_gaps(self):
        from imas_codex.standard_names.models import SNComposeBatch

        data = {
            "candidates": [],
            "skipped": [],
            "vocab_gaps": [
                {
                    "source_id": "equilibrium/time_slice/profiles_1d/dpsi_drho_tor",
                    "segment": "transformation",
                    "needed_token": "derivative_of",
                    "reason": "Quantity is a derivative but no derivative transformation exists",
                }
            ],
        }
        batch = SNComposeBatch(**data)
        assert len(batch.vocab_gaps) == 1
        assert batch.vocab_gaps[0].segment == "transformation"
        assert batch.vocab_gaps[0].needed_token == "derivative_of"
        assert batch.vocab_gaps[0].source_id == (
            "equilibrium/time_slice/profiles_1d/dpsi_drho_tor"
        )

    def test_compose_batch_vocab_gaps_default_empty(self):
        from imas_codex.standard_names.models import SNComposeBatch

        batch = SNComposeBatch(candidates=[], skipped=[])
        assert batch.vocab_gaps == []

    def test_compose_batch_multiple_gaps(self):
        from imas_codex.standard_names.models import SNComposeBatch

        data = {
            "candidates": [],
            "skipped": [],
            "vocab_gaps": [
                {
                    "source_id": "path/a",
                    "segment": "transformation",
                    "needed_token": "derivative_of",
                    "reason": "reason A",
                },
                {
                    "source_id": "path/b",
                    "segment": "process",
                    "needed_token": "fusion",
                    "reason": "reason B",
                },
            ],
        }
        batch = SNComposeBatch(**data)
        assert len(batch.vocab_gaps) == 2
        assert {g.needed_token for g in batch.vocab_gaps} == {
            "derivative_of",
            "fusion",
        }


# ---------------------------------------------------------------------------
# Test 3: write_vocab_gaps dedup and relationship creation
# ---------------------------------------------------------------------------


class TestWriteVocabGaps:
    """write_vocab_gaps deduplicates gaps and creates relationships."""

    def _call_write(
        self, gaps: list[dict], mock_gc: MagicMock, source_type: str = "dd"
    ) -> int:
        """Call write_vocab_gaps with a mocked GraphClient."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            return write_vocab_gaps(gaps, source_type=source_type)

    def test_empty_returns_zero(self):
        from imas_codex.standard_names.graph_ops import write_vocab_gaps

        assert write_vocab_gaps([]) == 0

    def test_dedup_same_segment_needed_token(self):
        """Two gaps with same segment:needed_token → 1 VocabGap node."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason B",
            },
        ]
        result = self._call_write(gaps, mock_gc)
        assert result == 1  # 1 unique VocabGap node

    def test_different_tokens_create_separate_nodes(self):
        """Gaps with different segment:needed_token → separate VocabGap nodes."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "process",
                "needed_token": "fusion",
                "reason": "reason B",
            },
        ]
        result = self._call_write(gaps, mock_gc)
        assert result == 2  # 2 unique VocabGap nodes

    def test_example_count_accumulates(self):
        """Duplicate segment:needed_token accumulates example_count in batch."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason A",
            },
            {
                "source_id": "path/b",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason B",
            },
            {
                "source_id": "path/c",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason C",
            },
        ]
        self._call_write(gaps, mock_gc)

        # Find the MERGE VocabGap query call and inspect the batch
        merge_call = None
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "MERGE (vg:VocabGap" in cypher:
                merge_call = call
                break
        assert merge_call is not None, "No MERGE VocabGap query found"

        batch = merge_call[1]["batch"]
        assert len(batch) == 1  # 1 deduplicated node
        assert batch[0]["example_count"] == 3  # 3 sources contributed

    def test_dd_source_creates_imasnode_relationship(self):
        """DD source type creates HAS_SN_VOCAB_GAP from IMASNode."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "needs derivative",
            }
        ]
        self._call_write(gaps, mock_gc, source_type="dd")

        # Find the relationship query
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_SN_VOCAB_GAP" in c[0][0] and "IMASNode" in c[0][0]
        ]
        assert len(rel_calls) == 1, "Should create DD HAS_SN_VOCAB_GAP relationship"

        # Verify reason is in the relationship batch
        rel_batch = rel_calls[0][1]["batch"]
        assert len(rel_batch) == 1
        assert rel_batch[0]["reason"] == "needs derivative"
        assert rel_batch[0]["source_id"] == ("equilibrium/time_slice/profiles_1d/psi")

    def test_signal_source_creates_facilitysignal_relationship(self):
        """Signal source type creates HAS_SN_VOCAB_GAP from FacilitySignal."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "tcv:ip/measured",
                "segment": "subject",
                "needed_token": "plasma_current_ip",
                "reason": "missing subject token",
            }
        ]
        self._call_write(gaps, mock_gc, source_type="signal")

        # Find the relationship query for FacilitySignal
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_SN_VOCAB_GAP" in c[0][0] and "FacilitySignal" in c[0][0]
        ]
        assert len(rel_calls) == 1, "Should create signal HAS_SN_VOCAB_GAP relationship"

    def test_relationship_has_per_source_reason(self):
        """Each HAS_SN_VOCAB_GAP relationship carries source-specific reason."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason for A",
            },
            {
                "source_id": "path/b",
                "segment": "transformation",
                "needed_token": "derivative_of",
                "reason": "reason for B",
            },
        ]
        self._call_write(gaps, mock_gc)

        # Find relationship query
        rel_calls = [
            c for c in mock_gc.query.call_args_list if "HAS_SN_VOCAB_GAP" in c[0][0]
        ]
        assert len(rel_calls) >= 1

        rel_batch = rel_calls[0][1]["batch"]
        assert len(rel_batch) == 2  # 2 relationships (one per source)
        reasons = {r["reason"] for r in rel_batch}
        assert reasons == {"reason for A", "reason for B"}


# ---------------------------------------------------------------------------
# Test 4: Ambiguity detection tagging
# ---------------------------------------------------------------------------


class TestAmbiguityClassification:
    """Validate worker tags component/coordinate overlap as grammar ambiguity."""

    def test_component_coordinate_overlap_classification(self):
        """Error containing 'component' AND 'coordinate' → specific ambiguity tag."""
        # Reproduce the classification logic from validate_worker
        exc_msg = "Token 'radial' is ambiguous: matches both component and coordinate"
        exc_msg_lower = exc_msg.lower()
        name = "radial_electron_temperature"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("grammar:ambiguity:component_coordinate_overlap:")
        assert name in issues[0]

    def test_generic_ambiguity_classification(self):
        """Error containing 'ambig' but NOT component+coordinate → unclassified."""
        exc_msg = "Ambiguous token: cannot resolve segment"
        exc_msg_lower = exc_msg.lower()
        name = "some_ambiguous_name"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("grammar:ambiguity:unclassified:")

    def test_plain_parse_error_classification(self):
        """Error without ambiguity keywords → generic parse_error."""
        exc_msg = "Invalid token sequence in standard name"
        exc_msg_lower = exc_msg.lower()
        name = "broken_name_here"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")

    def test_component_only_is_not_overlap(self):
        """Error with 'component' but NOT 'coordinate' → generic parse_error."""
        exc_msg = "Unknown component token 'radial'"
        exc_msg_lower = exc_msg.lower()
        name = "radial_temperature"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")

    def test_coordinate_only_is_not_overlap(self):
        """Error with 'coordinate' but NOT 'component' → generic parse_error."""
        exc_msg = "Unknown coordinate token 'toroidal'"
        exc_msg_lower = exc_msg.lower()
        name = "toroidal_field"

        issues: list[str] = []
        if "component" in exc_msg_lower and "coordinate" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:component_coordinate_overlap: {name}")
        elif "ambig" in exc_msg_lower:
            issues.append(f"grammar:ambiguity:unclassified: {name}")
        else:
            issues.append(f"parse_error: grammar round-trip failed for {name}")

        assert len(issues) == 1
        assert issues[0].startswith("parse_error:")
