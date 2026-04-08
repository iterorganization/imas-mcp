"""Unit tests for phrase-aware BM25 query rewriting helpers."""

import pytest

from imas_codex.tools.graph_search import _build_phrase_aware_query, _escape_lucene


class TestEscapeLucene:
    def test_no_special_chars(self):
        assert _escape_lucene("electron") == "electron"

    def test_parens(self):
        result = _escape_lucene("q(95)")
        assert "\\(" in result
        assert "\\)" in result

    def test_plus_minus(self):
        assert _escape_lucene("a+b-c") == "a\\+b\\-c"

    def test_colon(self):
        assert _escape_lucene("field:value") == "field\\:value"

    def test_empty(self):
        assert _escape_lucene("") == ""


class TestBuildPhraseAwareQuery:
    def test_single_word_unchanged(self):
        result = _build_phrase_aware_query("ip")
        assert result == "ip"

    def test_two_words_phrase(self):
        result = _build_phrase_aware_query("electron temperature")
        assert '"electron temperature"' in result
        assert "electron" in result
        assert "temperature" in result

    def test_three_words_bigrams(self):
        result = _build_phrase_aware_query("magnetic flux poloidal")
        assert '"magnetic flux"' in result
        assert '"flux poloidal"' in result
        assert "magnetic" in result
        assert "flux" in result
        assert "poloidal" in result

    def test_three_words_no_cross_bigram(self):
        """'magnetic poloidal' should NOT appear as a phrase (not adjacent)."""
        result = _build_phrase_aware_query("magnetic flux poloidal")
        assert '"magnetic poloidal"' not in result

    def test_special_chars_escaped(self):
        """q(95) must not produce invalid Lucene syntax."""
        result = _build_phrase_aware_query("q(95)")
        # Single word — no phrases, just escaped term
        assert "\\(" in result
        assert "\\)" in result
        # No unescaped parens
        assert "q(95)" not in result

    def test_multi_word_with_special_chars(self):
        result = _build_phrase_aware_query("q(95) profile")
        # Should contain a bigram phrase and individual terms
        assert "OR" in result
        # Both words present (escaped)
        assert "profile" in result

    def test_or_separator(self):
        result = _build_phrase_aware_query("electron temperature")
        assert " OR " in result

    def test_whitespace_stripped(self):
        result = _build_phrase_aware_query("  ip  ")
        assert result == "ip"
