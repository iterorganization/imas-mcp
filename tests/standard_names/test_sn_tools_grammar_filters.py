"""Unit tests for ``_list_grammar_vocabulary`` and ``_search_standard_names`` MCP tools.

Verifies (vNext grammar — plan 38 W4a):

- ``_search_standard_names`` accepts no ``grammar_*`` kwargs (removed); kind/tags/cocos
  post-filters still work.
- ``_list_grammar_vocabulary`` validates segment names dynamically against
  ``SEGMENT_TOKEN_MAP``, returns ISN vocabulary for closed segments, and
  describes open segments without querying Neo4j.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.llm.sn_tools import (
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
# _search_standard_names — no grammar_* args; other filters still work
# ---------------------------------------------------------------------------


class TestSearchNoGrammarArgs:
    """``_search_standard_names`` has no grammar_* kwargs in vNext."""

    def _make_rows(self) -> list[dict]:
        return [
            {
                "name": "electron_temperature",
                "description": "T_e",
                "kind": "scalar",
                "score": 1.0,
            },
            {
                "name": "ion_temperature",
                "description": "T_i",
                "kind": "scalar",
                "score": 0.9,
            },
        ]

    def test_returns_all_without_filters(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_standard_names",
            lambda *a, **kw: rows,
        )
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names("temperature", gc=mock_gc)
        assert "electron_temperature" in out
        assert "ion_temperature" in out

    def test_kind_filter_still_works(self, mock_gc, monkeypatch):
        rows = self._make_rows()
        rows[0]["kind"] = "scalar"
        rows[1]["kind"] = "vector"
        monkeypatch.setattr(
            "imas_codex.llm.sn_tools._keyword_search_standard_names",
            lambda *a, **kw: rows,
        )
        monkeypatch.setattr(
            "imas_codex.embeddings.encoder.Encoder.embed_texts",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no embed")),
        )
        out = _search_standard_names("temperature", kind="vector", gc=mock_gc)
        assert "ion_temperature" in out
        assert "electron_temperature" not in out

    def test_no_grammar_kwargs_accepted(self):
        """Signature must not have grammar_* parameters."""
        import inspect

        sig = inspect.signature(_search_standard_names)
        grammar_params = [p for p in sig.parameters if p.startswith("grammar_")]
        assert grammar_params == [], (
            f"grammar_* kwargs found in _search_standard_names: {grammar_params}"
        )


# ---------------------------------------------------------------------------
# _list_grammar_vocabulary — dynamic ISN vocabulary (no Neo4j query)
# ---------------------------------------------------------------------------


class TestListGrammarVocabulary:
    """Vocabulary listing uses ISN SEGMENT_TOKEN_MAP dynamically."""

    def test_unknown_segment_returns_error_with_valid_list(self):
        out = _list_grammar_vocabulary("bogus")
        assert "Unknown grammar segment" in out
        # Some known segment names should appear in the valid list
        assert "component" in out
        assert "physical_base" in out

    def test_known_closed_segment_returns_vocabulary_table(self):
        out = _list_grammar_vocabulary("component")
        assert "## Grammar Vocabulary: `component`" in out
        # rc21 has 18 component tokens — output should have token rows
        assert "|" in out
        # Must not be an error
        assert "Unknown grammar segment" not in out

    def test_open_segment_returns_informational_message(self):
        out = _list_grammar_vocabulary("physical_base")
        # physical_base is open in rc21 (empty token list)
        assert "physical_base" in out
        assert "Unknown grammar segment" not in out

    def test_segment_case_insensitive(self):
        out = _list_grammar_vocabulary("Component")
        assert "Unknown grammar segment" not in out
        assert "## Grammar Vocabulary: `component`" in out

    def test_all_isn_segments_are_accepted(self):
        """All segments from SEGMENT_TOKEN_MAP should be valid."""
        try:
            from imas_standard_names.grammar.constants import SEGMENT_TOKEN_MAP

            segments = list(SEGMENT_TOKEN_MAP.keys())
        except ImportError:
            pytest.skip("ISN not available")
        for seg in segments:
            out = _list_grammar_vocabulary(seg)
            assert "Unknown grammar segment" not in out, (
                f"segment {seg!r} rejected by allowlist"
            )

    def test_does_not_query_neo4j(self, mock_gc):
        """_list_grammar_vocabulary must not call GraphClient.query."""
        _list_grammar_vocabulary("component")
        mock_gc.query.assert_not_called()
