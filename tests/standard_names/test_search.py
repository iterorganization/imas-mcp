"""Tests for standard name search helpers."""

import pytest


def test_search_standard_names_vector_returns_list():
    """search_standard_names_vector returns a list (possibly empty if no graph)."""
    from imas_codex.standard_names.search import search_standard_names_vector

    result = search_standard_names_vector("electron temperature")
    assert isinstance(result, list)


def test_search_standard_names_vector_graceful_failure():
    """search_standard_names_vector doesn't raise on missing graph."""
    from imas_codex.standard_names.search import search_standard_names_vector

    # Even with garbage input, should return empty list, not raise
    result = search_standard_names_vector("")
    assert isinstance(result, list)
    assert result == []


def test_search_standard_names_vector_empty_query():
    """Empty or whitespace-only queries return empty list immediately."""
    from imas_codex.standard_names.search import search_standard_names_vector

    assert search_standard_names_vector("") == []
    assert search_standard_names_vector("   ") == []
