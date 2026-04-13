"""Tests for standard name search helpers."""

import pytest


def test_search_similar_names_returns_list():
    """search_similar_names returns a list (possibly empty if no graph)."""
    from imas_codex.standard_names.search import search_similar_names

    result = search_similar_names("electron temperature")
    assert isinstance(result, list)


def test_search_similar_names_graceful_failure():
    """search_similar_names doesn't raise on missing graph."""
    from imas_codex.standard_names.search import search_similar_names

    # Even with garbage input, should return empty list, not raise
    result = search_similar_names("")
    assert isinstance(result, list)
    assert result == []


def test_search_similar_names_empty_query():
    """Empty or whitespace-only queries return empty list immediately."""
    from imas_codex.standard_names.search import search_similar_names

    assert search_similar_names("") == []
    assert search_similar_names("   ") == []
