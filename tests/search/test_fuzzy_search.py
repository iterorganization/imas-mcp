"""Tests for Lucene fuzzy search query building."""

from __future__ import annotations

import pytest


class TestBuildLuceneQuery:
    """Verify _build_lucene_query builds correct Lucene syntax."""

    def test_single_term(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("temperature")
        # Should have field boosting
        assert "description:temperature^3" in result
        assert "name:temperature^2" in result
        assert "keywords:temperature^2" in result
        # 11 chars → fuzzy ~2
        assert "temperature~2" in result

    def test_multi_term(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("electron temperature")
        assert "description:electron^3" in result
        assert "description:temperature^3" in result
        # Both terms linked with AND
        assert " AND " in result
        # Fuzzy OR clause
        assert " OR " in result
        # electron (8 chars) → ~2, temperature (11 chars) → ~2
        assert "electron~2" in result
        assert "temperature~2" in result

    def test_short_term_no_fuzzy(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("ip")
        # Should have field boosting
        assert "description:ip^3" in result
        # 2 chars → no fuzzy variant
        # Just the base query, no OR fuzzy
        assert result.count("OR") == 3  # only the 3 field-level ORs

    def test_medium_term_fuzzy_1(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("psi")
        # 3 chars → no fuzzy
        assert "psi~" not in result
        # "temp" would be 4 chars → fuzzy ~1
        result2 = _build_lucene_query("temp")
        assert "temp~1" in result2

    def test_long_term_fuzzy_2(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("elongation")
        # 10 chars → ~2
        assert "elongation~2" in result

    def test_misspelling_coverage(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        # Common misspelling: "temperture" (10 chars) → ~2
        result = _build_lucene_query("temperture")
        assert "temperture~2" in result

    def test_special_chars_escaped(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("rho_tor_norm")
        # Underscores should be escaped
        assert "rho\\_tor\\_norm" in result or "rho_tor_norm" in result

    def test_empty_query(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("")
        assert result == ""  # or handle gracefully

    def test_whitespace_only_query(self):
        from imas_codex.tools.graph_search import _build_lucene_query

        result = _build_lucene_query("   ")
        assert result == ""


class TestEscapeLucene:
    """Verify Lucene special character escaping."""

    def test_basic_escape(self):
        from imas_codex.tools.graph_search import _escape_lucene

        assert _escape_lucene("test") == "test"
        assert _escape_lucene("a+b") == "a\\+b"
        assert _escape_lucene("a:b") == "a\\:b"

    def test_brackets(self):
        from imas_codex.tools.graph_search import _escape_lucene

        assert _escape_lucene("[0]") == "\\[0\\]"
