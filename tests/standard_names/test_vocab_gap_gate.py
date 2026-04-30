"""Tests for the VocabGap 3-way classifier gate in write_vocab_gaps.

Verifies that VocabGap nodes are classified as absent, wrong_slot_placement,
or ambiguous_known_token based on the ISN segment-token index.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_segment_token_map(mapping: dict[str, tuple[str, ...]]) -> dict:
    """Build a mock SEGMENT_TOKEN_MAP from a simplified mapping."""
    return mapping


class TestVocabGapClassification:
    """Mock the gap-write call site; verify correct category assignment."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        """Clear lru_cache between tests so mock data takes effect."""
        from imas_codex.standard_names.segments import (
            _segment_token_index,
            open_segments,
        )

        _segment_token_index.cache_clear()
        open_segments.cache_clear()
        yield
        _segment_token_index.cache_clear()
        open_segments.cache_clear()

    def _run_write_vocab_gaps(self, gaps, segment_token_map):
        """Run write_vocab_gaps with a mocked ISN and GraphClient."""
        mock_gc_instance = MagicMock()
        mock_gc_instance.query = MagicMock()
        mock_gc_instance.__enter__ = MagicMock(return_value=mock_gc_instance)
        mock_gc_instance.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "imas_codex.standard_names.segments.SEGMENT_TOKEN_MAP",
                segment_token_map,
                create=True,
            ),
            patch(
                "imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP",
                segment_token_map,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "imas_standard_names": MagicMock(),
                    "imas_standard_names.grammar": MagicMock(),
                    "imas_standard_names.grammar.constants": MagicMock(
                        SEGMENT_TOKEN_MAP=segment_token_map
                    ),
                },
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=mock_gc_instance,
            ),
        ):
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            write_vocab_gaps(gaps, "dd", skip_segment_filter=True)

        # Extract the batch passed to the first MERGE query (gap nodes)
        calls = mock_gc_instance.query.call_args_list
        assert calls, "Expected at least one query call"
        # First call is the MERGE for VocabGap nodes
        first_call = calls[0]
        batch = first_call.kwargs.get("batch", first_call[1].get("batch", []))
        return {node["id"]: node for node in batch}

    def test_absent_token(self):
        """Token not in any segment → category='absent'."""
        stm = {
            "subject": ("electron", "ion"),
            "process": ("heating", "cooling"),
            "physical_base": (),  # open
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "needed_token": "frobnicating",
                "reason": "test",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        gap_id = "vocab_gap:process:frobnicating"
        assert gap_id in nodes
        assert nodes[gap_id]["category"] == "absent"
        assert nodes[gap_id]["actual_segments"] == []

    def test_wrong_slot_placement(self):
        """Token exists in one segment but reported on another → 'wrong_slot_placement'."""
        stm = {
            "subject": ("electron", "ion", "particle"),
            "process": ("heating", "cooling"),
            "physical_base": (),
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "needed_token": "ion",
                "reason": "LLM placed ion in process",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        gap_id = "vocab_gap:process:ion"
        assert gap_id in nodes
        assert nodes[gap_id]["category"] == "wrong_slot_placement"
        assert nodes[gap_id]["actual_segments"] == ["subject"]

    def test_ambiguous_known_token(self):
        """Token in multiple segments, reported on none of them → 'ambiguous_known_token'."""
        stm = {
            "subject": ("parallel", "radial"),
            "orientation": ("parallel", "toroidal"),
            "process": ("heating",),
            "physical_base": (),
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "needed_token": "parallel",
                "reason": "LLM confused",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        gap_id = "vocab_gap:process:parallel"
        assert gap_id in nodes
        assert nodes[gap_id]["category"] == "ambiguous_known_token"
        assert set(nodes[gap_id]["actual_segments"]) == {"subject", "orientation"}

    def test_mixed_batch(self):
        """A batch with all three categories produces correct per-gap classifications."""
        stm = {
            "subject": ("electron", "ion"),
            "orientation": ("parallel", "radial"),
            "qualifier": ("parallel",),  # overlap with orientation
            "process": ("heating",),
            "physical_base": (),
        }
        gaps = [
            # absent
            {
                "source_id": "a",
                "segment": "process",
                "needed_token": "turbulating",
                "reason": "test",
            },
            # wrong_slot (ion in subject, reported on process)
            {
                "source_id": "b",
                "segment": "process",
                "needed_token": "ion",
                "reason": "test",
            },
            # ambiguous (parallel in orientation+qualifier, reported on process)
            {
                "source_id": "c",
                "segment": "process",
                "needed_token": "parallel",
                "reason": "test",
            },
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)

        assert nodes["vocab_gap:process:turbulating"]["category"] == "absent"
        assert nodes["vocab_gap:process:ion"]["category"] == "wrong_slot_placement"
        assert (
            nodes["vocab_gap:process:parallel"]["category"] == "ambiguous_known_token"
        )
