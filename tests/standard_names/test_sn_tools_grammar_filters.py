"""Unit tests for grammar-slot filters and ``list_grammar_vocabulary`` MCP tool.

Verifies:

- ``_search_standard_names`` post-filters rows by ``grammar_*`` kwargs.
- ``_list_grammar_vocabulary`` validates the segment allowlist, dispatches
  a Cypher query using the correct property name, and formats the output
  as a markdown table.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.llm.sn_tools import (
    GRAMMAR_SEGMENTS,
    _list_grammar_vocabulary,
    _search_standard_names,
)


@pytest.fixture
def mock_gc() -> MagicMock:
    """Minimal GraphClient stub with ``query()`` method."""

    gc = MagicMock()
    gc.query = MagicMock(return_value=[])
    return gc


# ---------------------------------------------------------------------------
# _search_standard_names grammar_* post-filtering
# ---------------------------------------------------------------------------


class TestSearchGrammarFilters:
    """``_search_standard_names`` filters rows by grammar_* kwargs."""

    def _make_rows(self) -> list[dict]:
        return [
            {
                "name": "electron_temperature",
                "description": "T_e",
                "kind": "scalar",
                "grammar_subject": "electron",
                "grammar_physical_base": "temperature",
                "grammar_component": None,
                "score": 1.0,
            },
            {
                "name": "ion_temperature",
                "description": "T_i",
                "kind": "scalar",
                "grammar_subject": "ion",
                "grammar_physical_base": "temperature",
                "grammar_component": None,
                "score": 0.9,
            },
            {
                "name": "toroidal_electron_velocity",
                "description": "v_e,tor",
                "kind": "scalar",
                "grammar_subject": "electron",
                "grammar_physical_base": "velocity",
                "grammar_component": "toroidal",
                "score": 0.8,
            },
        ]

    def test_subject_filter_keeps_only_matching(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_sn", lambda *a, **kw: rows
        )
        # Force keyword path by breaking embedding
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names("temp", grammar_subject="electron", gc=mock_gc)
        assert "electron_temperature" in out
        assert "toroidal_electron_velocity" in out
        assert "ion_temperature" not in out

    def test_multiple_filters_are_conjunctive(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_sn", lambda *a, **kw: rows
        )
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names(
            "v",
            grammar_subject="electron",
            grammar_component="toroidal",
            gc=mock_gc,
        )
        assert "toroidal_electron_velocity" in out
        assert "electron_temperature" not in out
        assert "ion_temperature" not in out

    def test_filter_case_insensitive(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_sn", lambda *a, **kw: rows
        )
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names("t", grammar_subject="ELECTRON", gc=mock_gc)
        assert "electron_temperature" in out
        assert "ion_temperature" not in out

    def test_no_filter_returns_all(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_sn", lambda *a, **kw: rows
        )
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names("t", gc=mock_gc)
        assert "electron_temperature" in out
        assert "ion_temperature" in out
        assert "toroidal_electron_velocity" in out


# ---------------------------------------------------------------------------
# _list_grammar_vocabulary
# ---------------------------------------------------------------------------


class TestListGrammarVocabulary:
    """Vocabulary listing validates segment and formats results."""

    def test_unknown_segment_returns_error_with_valid_list(self, mock_gc):
        out = _list_grammar_vocabulary("bogus", gc=mock_gc)
        assert "Unknown grammar segment" in out
        # Some known segment names should appear in the valid list
        assert "physical_base" in out
        assert "component" in out
        # Should not issue a query for invalid segment
        mock_gc.query.assert_not_called()

    def test_known_segment_dispatches_correct_property(self, mock_gc):
        mock_gc.query.return_value = [
            {"token": "radial", "n": 37},
            {"token": "toroidal", "n": 32},
        ]
        out = _list_grammar_vocabulary("component", gc=mock_gc)
        # Cypher should reference the grammar_component property
        cypher_arg = mock_gc.query.call_args[0][0]
        assert "sn.grammar_component" in cypher_arg
        # Output is a markdown table with tokens and counts
        assert "## Grammar Vocabulary: `component`" in out
        assert "| radial | 37 |" in out
        assert "| toroidal | 32 |" in out
        assert "2 distinct tokens across 69 StandardName nodes." in out

    def test_empty_result_returns_descriptive_message(self, mock_gc):
        mock_gc.query.return_value = []
        out = _list_grammar_vocabulary("region", gc=mock_gc)
        assert "No StandardName nodes have the `grammar_region` property set." in out

    def test_all_declared_segments_are_accepted(self, mock_gc):
        mock_gc.query.return_value = []
        for seg in GRAMMAR_SEGMENTS:
            out = _list_grammar_vocabulary(seg, gc=mock_gc)
            assert "Unknown grammar segment" not in out, (
                f"segment {seg!r} rejected by allowlist"
            )

    def test_segment_case_insensitive(self, mock_gc):
        mock_gc.query.return_value = [{"token": "toroidal", "n": 1}]
        out = _list_grammar_vocabulary("Component", gc=mock_gc)
        assert "Unknown grammar segment" not in out
        assert "## Grammar Vocabulary: `component`" in out
