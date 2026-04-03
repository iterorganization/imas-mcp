"""Tests for domain query functions.

These tests validate the domain query functions (find_signals, find_wiki,
find_imas, find_code, find_data_nodes, map_signals_to_imas, facility_overview)
without requiring Neo4j — they test structure and error handling using mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.domain_queries import (
    facility_overview,
    find_code,
    find_data_nodes,
    find_imas,
    find_signals,
    find_wiki,
    map_signals_to_imas,
    wiki_page_chunks,
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

    def test_requires_query_or_keyword(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="query.*keyword"):
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

    def test_text_contains_keyword_only(self, mock_gc, mock_embed):
        """Keyword-only search without semantic query."""
        find_wiki(text_contains="fishbone", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_not_called()
        cypher = mock_gc.query.call_args[0][0]
        assert "CONTAINS" in cypher

    def test_page_title_contains(self, mock_gc, mock_embed):
        """Filter by page title substring."""
        find_wiki(page_title_contains="fishbone", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "title" in cypher.lower()
        assert "CONTAINS" in cypher

    def test_semantic_with_text_filter(self, mock_gc, mock_embed):
        """Combined semantic + keyword filtering."""
        find_wiki(
            query="instabilities",
            text_contains="fishbone",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        mock_embed.assert_called_once()
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "CONTAINS" in cypher

    def test_semantic_with_title_filter(self, mock_gc, mock_embed):
        """Semantic search filtered by page title."""
        find_wiki(
            query="kink mode",
            page_title_contains="fishbone",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "title" in cypher.lower()


class TestWikiPageChunks:
    """Test wiki_page_chunks helper."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_filters_by_title(self, mock_gc, mock_embed):
        wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "title" in cypher.lower()
        assert "CONTAINS" in cypher

    def test_with_facility(self, mock_gc, mock_embed):
        wiki_page_chunks("fishbone", facility="jet", gc=mock_gc, embed_fn=mock_embed)
        call_kwargs = mock_gc.query.call_args
        assert "jet" in str(call_kwargs)

    def test_with_text_contains(self, mock_gc, mock_embed):
        wiki_page_chunks(
            "fishbone",
            text_contains="team",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        # Should have two CONTAINS conditions
        assert cypher.count("CONTAINS") >= 2

    def test_returns_page_context(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "page_title": "Controlling fishbones",
                "page_url": "http://...",
                "facility": "jet",
                "section": "Team",
                "text": "content",
            }
        ]
        result = wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        assert result[0]["page_title"] == "Controlling fishbones"


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


class TestFindDataNodes:
    """Test find_data_nodes domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_data_nodes(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_with_tree_filter(self, mock_gc, mock_embed):
        find_data_nodes(
            facility="tcv", data_source_name="results", gc=mock_gc, embed_fn=mock_embed
        )
        call_kwargs = mock_gc.query.call_args
        assert "results" in str(call_kwargs)

    def test_with_semantic_search(self, mock_gc, mock_embed):
        find_data_nodes(
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
        assert "MEMBER_OF" in cypher
        assert "IMASNode" in cypher


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
            find_data_nodes,
            find_imas,
            find_signals,
            find_wiki,
            map_signals_to_imas,
        )
