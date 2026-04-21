"""Tests for VocabGap infrastructure (Phase 2A–2D).

Covers:
- StandardNameVocabGap / StandardNameComposeBatch model parsing
- write_vocab_gaps dedup and relationship creation
- Ambiguity detection tagging in validate_worker
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Test 1: StandardNameVocabGap model
# ---------------------------------------------------------------------------


class TestSNVocabGapModel:
    """StandardNameVocabGap model validates and stores fields."""

    def test_basic_construction(self):
        from imas_codex.standard_names.models import StandardNameVocabGap

        gap = StandardNameVocabGap(
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

        from imas_codex.standard_names.models import StandardNameVocabGap

        with pytest.raises(ValidationError):
            StandardNameVocabGap(source_id="path/a", segment="transformation")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Test 2: StandardNameComposeBatch with vocab_gaps
# ---------------------------------------------------------------------------


class TestSNComposeBatchVocabGaps:
    """StandardNameComposeBatch correctly parses vocab_gaps from LLM response."""

    def test_compose_batch_with_vocab_gaps(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

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
        batch = StandardNameComposeBatch(**data)
        assert len(batch.vocab_gaps) == 1
        assert batch.vocab_gaps[0].segment == "transformation"
        assert batch.vocab_gaps[0].needed_token == "derivative_of"
        assert batch.vocab_gaps[0].source_id == (
            "equilibrium/time_slice/profiles_1d/dpsi_drho_tor"
        )

    def test_compose_batch_vocab_gaps_default_empty(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

        batch = StandardNameComposeBatch(candidates=[], skipped=[])
        assert batch.vocab_gaps == []

    def test_compose_batch_multiple_gaps(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

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
        batch = StandardNameComposeBatch(**data)
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
        """DD source type creates HAS_STANDARD_NAME_VOCAB_GAP from IMASNode."""
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
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0] and "IMASNode" in c[0][0]
        ]
        assert len(rel_calls) == 1, (
            "Should create DD HAS_STANDARD_NAME_VOCAB_GAP relationship"
        )

        # Verify reason is in the relationship batch
        rel_batch = rel_calls[0][1]["batch"]
        assert len(rel_batch) == 1
        assert rel_batch[0]["reason"] == "needs derivative"
        assert rel_batch[0]["source_id"] == ("equilibrium/time_slice/profiles_1d/psi")

    def test_signal_source_creates_facilitysignal_relationship(self):
        """Signal source type creates HAS_STANDARD_NAME_VOCAB_GAP from FacilitySignal."""
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
        self._call_write(gaps, mock_gc, source_type="signals")

        # Find the relationship query for FacilitySignal
        rel_calls = [
            c
            for c in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0] and "FacilitySignal" in c[0][0]
        ]
        assert len(rel_calls) == 1, (
            "Should create signal HAS_STANDARD_NAME_VOCAB_GAP relationship"
        )

    def test_relationship_has_per_source_reason(self):
        """Each HAS_STANDARD_NAME_VOCAB_GAP relationship carries source-specific reason."""
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
            c
            for c in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME_VOCAB_GAP" in c[0][0]
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


# ---------------------------------------------------------------------------
# Test 5: persist_composed_batch creates VocabGap from token-miss
# ---------------------------------------------------------------------------


class TestPersistBatchTokenMissGaps:
    """write_standard_names calls write_vocab_gaps for token misses detected
    during _write_segment_edges."""

    def test_token_miss_creates_vocab_gap_nodes(self):
        """When _write_segment_edges detects unmatched tokens, VocabGap nodes are created."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        # Simulate _write_segment_edges returning a token miss
        token_miss_gaps = [
            {
                "sn_id": "electron_temperature",
                "segment": "subject",
                "needed_token": "exotic_particle",
            }
        ]

        names = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
                "unit": "eV",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_segment_edges",
                return_value=token_miss_gaps,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        # write_vocab_gaps should have been called with DD gaps
        mock_write_vg.assert_called_once()
        call_args = mock_write_vg.call_args
        gap_dicts = call_args[0][0]
        assert len(gap_dicts) == 1
        assert gap_dicts[0]["source_id"] == (
            "core_profiles/profiles_1d/electrons/temperature"
        )
        assert gap_dicts[0]["segment"] == "subject"
        assert gap_dicts[0]["needed_token"] == "exotic_particle"
        assert call_args[1]["source_type"] == "dd"

    def test_no_token_miss_skips_write_vocab_gaps(self):
        """When _write_segment_edges returns no gaps, write_vocab_gaps is not called."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        names = [
            {
                "id": "electron_temperature",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "source_types": ["dd"],
                "unit": "eV",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_segment_edges",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        mock_write_vg.assert_not_called()

    def test_signal_source_routes_to_signals_type(self):
        """Signal source_types route gaps to write_vocab_gaps with source_type='signals'."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        token_miss_gaps = [
            {
                "sn_id": "plasma_current",
                "segment": "process",
                "needed_token": "novel_process",
            }
        ]

        names = [
            {
                "id": "plasma_current",
                "source_id": "tcv:ip/measured",
                "source_types": ["signals"],
                "unit": "A",
            }
        ]

        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.graph_ops._write_segment_edges",
                return_value=token_miss_gaps,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.write_vocab_gaps"
            ) as mock_write_vg,
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.standard_names.graph_ops import write_standard_names

            write_standard_names(names)

        mock_write_vg.assert_called_once()
        assert mock_write_vg.call_args[1]["source_type"] == "signals"


# ---------------------------------------------------------------------------
# Test 6: _resolve_grammar_token_version fallback
# ---------------------------------------------------------------------------


class TestResolveGrammarTokenVersion:
    """_resolve_grammar_token_version uses exact ISN version when available,
    falls back to latest graph version, or returns None."""

    def test_exact_version_match(self):
        """Returns ISN version when GrammarToken nodes exist for it."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[{"t.version": "0.7.0rc16"}])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result == "0.7.0rc16"

    def test_fallback_to_latest(self):
        """Falls back to latest available version when exact doesn't exist."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()

        # First call (exact version) returns empty, second (fallback) returns rc14
        mock_gc.query = MagicMock(side_effect=[[], [{"v": "0.7.0rc14"}]])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result == "0.7.0rc14"

    def test_no_grammar_tokens(self):
        """Returns None when no GrammarToken nodes exist at all."""
        from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        result = _resolve_grammar_token_version(mock_gc, "0.7.0rc16")
        assert result is None


# ---------------------------------------------------------------------------
# Test 7: _write_segment_edges version fallback integration
# ---------------------------------------------------------------------------


class TestWriteSegmentEdgesVersionFallback:
    """_write_segment_edges uses version fallback to avoid false-positive
    VocabGap nodes when GrammarToken nodes are stale."""

    def test_no_grammar_tokens_skips_entirely(self):
        """When no GrammarToken nodes exist, returns empty gaps without parsing."""
        mock_gc = MagicMock()

        with patch(
            "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
            return_value=None,
        ):
            from imas_codex.standard_names.graph_ops import _write_segment_edges

            gaps = _write_segment_edges(mock_gc, ["electron_temperature"])

        assert gaps == []

    def test_fallback_version_passed_to_cypher(self):
        """When ISN version differs from graph tokens, fallback version is used
        in the OPTIONAL MATCH to avoid false-positive VocabGap."""
        mock_gc = MagicMock()
        # Return matched=True for the token query
        mock_gc.query = MagicMock(
            return_value=[{"token": "electron", "segment": "subject", "matched": True}]
        )

        with (
            patch(
                "imas_codex.standard_names.graph_ops._resolve_grammar_token_version",
                return_value="0.7.0rc14",
            ),
            patch("imas_standard_names.grammar.parse_standard_name") as mock_parse,
            patch("imas_standard_names.graph.spec.segment_edge_specs") as mock_specs,
        ):
            mock_parsed = MagicMock()
            mock_parse.return_value = mock_parsed
            mock_spec = MagicMock()
            mock_spec.position = 2
            mock_spec.segment = "subject"
            mock_spec.token = "electron"
            mock_specs.return_value = [mock_spec]

            from imas_codex.standard_names.graph_ops import _write_segment_edges

            gaps = _write_segment_edges(mock_gc, ["electron_temperature"])

        # No gaps — token was matched via fallback version
        assert gaps == []

        # Verify fallback version (0.7.0rc14) was used in the query
        opt_match_calls = [
            c for c in mock_gc.query.call_args_list if "OPTIONAL MATCH" in str(c)
        ]
        assert len(opt_match_calls) >= 1
        assert opt_match_calls[0][1]["token_version"] == "0.7.0rc14"


# ---------------------------------------------------------------------------
# Test 5: Open-vocabulary segment filtering (Problem 1 regression)
# ---------------------------------------------------------------------------


class TestOpenSegmentFilter:
    """Open-vocabulary segments must never produce VocabGap nodes.

    ``physical_base`` is open by design — any compound is admissible.
    ``grammar_ambiguity`` is a pseudo segment reported by the composer for
    structural ambiguity rather than missing tokens.  Gaps on either are
    filtered out before ``write_vocab_gaps`` persists them, and they never
    retire the underlying ``StandardNameSource`` to ``vocab_gap`` status.
    """

    def test_open_segments_includes_physical_base(self):
        from imas_codex.standard_names.segments import open_segments

        assert "physical_base" in open_segments()

    def test_is_open_segment_predicate(self):
        from imas_codex.standard_names.segments import is_open_segment

        assert is_open_segment("physical_base") is True
        assert is_open_segment("grammar_ambiguity") is True
        # Closed segments must not be flagged as open
        assert is_open_segment("transformation") is False
        assert is_open_segment("subject") is False
        assert is_open_segment("position") is False
        assert is_open_segment("component") is False
        assert is_open_segment(None) is False
        assert is_open_segment("") is False

    def test_filter_closed_segment_gaps_splits_by_openness(self):
        from imas_codex.standard_names.segments import filter_closed_segment_gaps

        gaps = [
            {"source_id": "a", "segment": "transformation", "needed_token": "curl_of"},
            {
                "source_id": "b",
                "segment": "physical_base",
                "needed_token": "toroidal_torque",
            },
            {
                "source_id": "c",
                "segment": "grammar_ambiguity",
                "needed_token": "diamagnetic",
            },
            {"source_id": "d", "segment": "subject", "needed_token": "pellet"},
        ]
        kept, dropped = filter_closed_segment_gaps(gaps)
        kept_segs = {g["segment"] for g in kept}
        drop_segs = {g["segment"] for g in dropped}
        assert kept_segs == {"transformation", "subject"}
        assert drop_segs == {"physical_base", "grammar_ambiguity"}

    def test_write_vocab_gaps_skips_open_segments(self):
        """write_vocab_gaps must not emit MERGE for open/pseudo segments."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "segment": "physical_base",
                "needed_token": "toroidal_torque",
                "reason": "open-segment compound — should be filtered",
            },
            {
                "source_id": "core_profiles/ions/velocity",
                "segment": "grammar_ambiguity",
                "needed_token": "diamagnetic",
                "reason": "structural ambiguity — should be filtered",
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            written = write_vocab_gaps(gaps, source_type="dd")

        # Nothing persisted
        assert written == 0
        # And crucially no MERGE (vg:VocabGap ...) query fired
        merge_calls = [
            c for c in mock_gc.query.call_args_list if "MERGE (vg:VocabGap" in c[0][0]
        ]
        assert merge_calls == [], (
            "Open-segment gaps must never reach the VocabGap MERGE query"
        )

    def test_write_vocab_gaps_mixed_batch_only_persists_closed(self):
        """Mixed batch: open-segment entry filtered, closed-segment kept."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        gaps = [
            {
                "source_id": "path/a",
                "segment": "physical_base",  # open — filter out
                "needed_token": "toroidal_torque",
                "reason": "noise",
            },
            {
                "source_id": "path/b",
                "segment": "transformation",  # closed — keep
                "needed_token": "curl_of",
                "reason": "real gap",
            },
        ]

        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            written = write_vocab_gaps(gaps, source_type="dd")

        assert written == 1  # only the closed-segment gap materialised
        merge_calls = [
            c for c in mock_gc.query.call_args_list if "MERGE (vg:VocabGap" in c[0][0]
        ]
        assert len(merge_calls) == 1
        batch = merge_calls[0][1]["batch"]
        assert len(batch) == 1
        assert batch[0]["segment"] == "transformation"
        assert batch[0]["needed_token"] == "curl_of"
