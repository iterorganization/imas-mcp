"""Tests for domain query functions.

These tests validate the domain query functions (find_signals, find_wiki,
find_imas, find_code, find_tree_nodes, map_signals_to_imas, facility_overview)
without requiring Neo4j — they test structure and error handling using mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.domain_queries import (
    facility_overview,
    find_code,
    find_imas,
    find_signals,
    find_tree_nodes,
    find_wiki,
    map_signals_to_imas,
)


@pytest.fixture
def mock_gc():
    """Mock GraphClient for testing without Neo4j."""
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def mock_embed():
    """Mock embed function."""
    return MagicMock(return_value=[0.1] * 256)


class TestFindSignals:
    """Test find_signals domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_signals(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_calls_query_with_facility(self, mock_gc, mock_embed):
        find_signals(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert mock_gc.query.called
        # Should have facility parameter
        call_kwargs = mock_gc.query.call_args
        assert "tcv" in str(call_kwargs)

    def test_semantic_search(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "id": "tcv:ip",
                "name": "ip",
                "description": "Plasma current",
                "score": 0.9,
            }
        ]
        result = find_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        assert mock_embed.called
        assert len(result) == 1
        assert result[0]["id"] == "tcv:ip"

    def test_requires_facility(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="facility"):
            find_signals(gc=mock_gc, embed_fn=mock_embed)

    def test_include_access_true(self, mock_gc, mock_embed):
        find_signals(
            facility="tcv", include_access=True, gc=mock_gc, embed_fn=mock_embed
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "DataAccess" in cypher

    def test_include_access_false(self, mock_gc, mock_embed):
        find_signals(
            facility="tcv",
            include_access=False,
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "DataAccess" not in cypher


class TestFindWiki:
    """Test find_wiki domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_wiki(query="equilibrium", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_requires_query(self, mock_gc, mock_embed):
        with pytest.raises(TypeError):
            find_wiki(gc=mock_gc, embed_fn=mock_embed)

    def test_calls_embedding(self, mock_gc, mock_embed):
        find_wiki(query="plasma", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_called_once_with("plasma")

    def test_includes_page_context(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "text": "content",
                "page_title": "Test Page",
                "page_url": "http://...",
                "score": 0.8,
            }
        ]
        find_wiki(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "WikiPage" in cypher


class TestFindImas:
    """Test find_imas domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_imas(
            query="electron temperature", gc=mock_gc, embed_fn=mock_embed
        )
        assert isinstance(result, list)

    def test_semantic_search(self, mock_gc, mock_embed):
        find_imas(query="electron temperature", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_called_once()
        assert mock_gc.query.called

    def test_filters_deprecated(self, mock_gc, mock_embed):
        find_imas(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "DEPRECATED_IN" in cypher


class TestFindCode:
    """Test find_code domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_code(
            query="equilibrium reconstruction", gc=mock_gc, embed_fn=mock_embed
        )
        assert isinstance(result, list)

    def test_with_facility_filter(self, mock_gc, mock_embed):
        find_code(query="test", facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "facility_id" in cypher


class TestFindTreeNodes:
    """Test find_tree_nodes domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_tree_nodes(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_with_tree_filter(self, mock_gc, mock_embed):
        find_tree_nodes(
            facility="tcv", tree_name="results", gc=mock_gc, embed_fn=mock_embed
        )
        call_kwargs = mock_gc.query.call_args
        assert "results" in str(call_kwargs)

    def test_with_semantic_search(self, mock_gc, mock_embed):
        find_tree_nodes(
            query="electron density",
            facility="tcv",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        # Should use vector search when query is provided
        cypher = mock_gc.query.call_args[0][0]
        # Should have vector search or WHERE clause
        assert "embedding" in cypher.lower() or "description" in cypher.lower()


class TestMapSignalsToImas:
    """Test map_signals_to_imas domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = map_signals_to_imas(facility="tcv", gc=mock_gc)
        assert isinstance(result, list)

    def test_includes_imas_paths(self, mock_gc, mock_embed):
        map_signals_to_imas(facility="tcv", gc=mock_gc)
        cypher = mock_gc.query.call_args[0][0]
        assert "MAPS_TO_IMAS" in cypher
        assert "IMASPath" in cypher


class TestFacilityOverview:
    """Test facility_overview domain query."""

    def test_returns_dict(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "diagnostics": 5,
                "trees": 3,
                "signals": 100,
                "wiki_pages": 20,
                "code_files": 50,
            }
        ]
        result = facility_overview(facility="tcv", gc=mock_gc)
        assert isinstance(result, dict)

    def test_has_facility_key(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "diagnostics": 0,
                "trees": 0,
                "signals": 0,
                "wiki_pages": 0,
                "code_files": 0,
            }
        ]
        result = facility_overview(facility="tcv", gc=mock_gc)
        assert result["facility"] == "tcv"


class TestFunctionSignatures:
    """Verify function signatures and help text."""

    def test_find_signals_has_docstring(self):
        assert find_signals.__doc__ is not None
        assert "facility" in find_signals.__doc__

    def test_find_wiki_has_docstring(self):
        assert find_wiki.__doc__ is not None

    def test_find_imas_has_docstring(self):
        assert find_imas.__doc__ is not None

    def test_all_functions_importable(self):
        from imas_codex.graph.domain_queries import (
            facility_overview,
            find_code,
            find_imas,
            find_signals,
            find_tree_nodes,
            find_wiki,
            map_signals_to_imas,
        )
