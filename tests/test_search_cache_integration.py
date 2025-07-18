"""
Integration tests for search caching in the MCP server.
"""

import pytest
from unittest.mock import AsyncMock, patch

from imas_mcp.server import Server


@pytest.mark.asyncio
class TestServerSearchCaching:
    """Test search caching integration with the MCP server."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        return Server(ids_set={"core_profiles", "equilibrium"})

    async def test_cache_integration_basic(self, server):
        """Test basic caching integration with search_imas."""
        # Clear cache to start fresh
        server.search_cache.clear()

        # Mock the search composer to return predictable results
        class MockResult:
            def to_dict(self):
                return {
                    "path": "core_profiles/profiles_1d/electrons/temperature",
                    "score": 0.95,
                    "documentation": "Electron temperature profile",
                    "units": "eV",
                    "ids_name": "core_profiles",
                }

            @property
            def document(self):
                class MockDoc:
                    raw_data = {}

                    class MockMeta:
                        path_name = "test/path"

                    metadata = MockMeta()

                return MockDoc()

        mock_results = [MockResult()]

        with patch.object(server.search_composer, "search", return_value=mock_results):
            # First call should miss cache and execute search
            result1 = await server.search_imas("plasma temperature")

            assert (
                result1.get("cache_hit") is not True
            )  # Should not have cache_hit flag on first call
            assert server.search_cache.stats["misses"] == 1
            assert server.search_cache.stats["sets"] == 1

            # Second identical call should hit cache
            result2 = await server.search_imas("plasma temperature")

            assert result2.get("cache_hit") is True
            assert server.search_cache.stats["hits"] == 1
            assert server.search_cache.stats["sets"] == 1  # Should not increment

            # Results should be identical (except cache_hit flag)
            result1_no_flag = {k: v for k, v in result1.items() if k != "cache_hit"}
            result2_no_flag = {k: v for k, v in result2.items() if k != "cache_hit"}
            assert result1_no_flag == result2_no_flag

    async def test_cache_with_different_parameters(self, server):
        """Test that different search parameters create different cache entries."""
        server.search_cache.clear()

        class MockResult:
            def to_dict(self):
                return {"path": "test/path", "score": 0.9}

            @property
            def document(self):
                class MockDoc:
                    raw_data = {}

                    class MockMeta:
                        path_name = "test/path"

                    metadata = MockMeta()

                return MockDoc()

        mock_results = [MockResult()]

        with patch.object(server.search_composer, "search", return_value=mock_results):
            # Different queries should miss cache
            await server.search_imas("plasma temperature")
            await server.search_imas("magnetic field")  # Different query

            assert server.search_cache.stats["misses"] == 2
            assert server.search_cache.stats["sets"] == 2

            # Different parameters should miss cache
            await server.search_imas(
                "plasma temperature", ids_name="equilibrium"
            )  # Different IDS
            await server.search_imas(
                "plasma temperature", max_results=20
            )  # Different max_results
            await server.search_imas(
                "plasma temperature", search_mode="lexical"
            )  # Different mode

            assert server.search_cache.stats["misses"] == 5
            assert server.search_cache.stats["sets"] == 5

    async def test_cache_does_not_store_errors(self, server):
        """Test that error results are not cached."""
        server.search_cache.clear()

        # Mock search composer to raise an exception
        with patch.object(
            server.search_composer, "search", side_effect=Exception("Search failed")
        ):
            result = await server.search_imas("failing query")

            assert "error" in result
            assert server.search_cache.stats["sets"] == 0  # Should not cache errors

    async def test_cache_does_not_store_empty_results(self, server):
        """Test that empty results are not cached."""
        server.search_cache.clear()

        # Mock empty results
        with patch.object(server.search_composer, "search", return_value=[]):
            result = await server.search_imas("empty query")

            assert result["total_results"] == 0
            assert (
                server.search_cache.stats["sets"] == 0
            )  # Should not cache empty results

    async def test_cache_with_ai_context(self, server):
        """Test caching behavior with AI context."""
        server.search_cache.clear()

        class MockResult:
            def to_dict(self):
                return {"path": "test/path", "score": 0.9}

            @property
            def document(self):
                class MockDoc:
                    raw_data = {}

                    class MockMeta:
                        path_name = "test/path"

                    metadata = MockMeta()

                return MockDoc()

        mock_results = [MockResult()]

        # Mock AI context
        mock_context = AsyncMock()

        with patch.object(server.search_composer, "search", return_value=mock_results):
            # Call with AI context
            result1 = await server.search_imas("plasma temperature", ctx=mock_context)

            assert "ai_insights" in result1
            assert server.search_cache.stats["sets"] == 1

            # Second call should hit cache
            result2 = await server.search_imas("plasma temperature", ctx=mock_context)

            assert result2["cache_hit"] is True
            assert server.search_cache.stats["hits"] == 1

    async def test_cache_stats_integration(self, server):
        """Test cache statistics through server integration."""
        server.search_cache.clear()

        class MockResult:
            def to_dict(self):
                return {"path": "test/path", "score": 0.9}

            @property
            def document(self):
                class MockDoc:
                    raw_data = {}

                    class MockMeta:
                        path_name = "test/path"

                    metadata = MockMeta()

                return MockDoc()

        mock_results = [MockResult()]

        with patch.object(server.search_composer, "search", return_value=mock_results):
            # Multiple searches to build stats
            await server.search_imas("query1")  # miss + set
            await server.search_imas("query1")  # hit
            await server.search_imas("query2")  # miss + set
            await server.search_imas("query1")  # hit

            stats = server.search_cache.get_stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 2
            assert stats["sets"] == 2
            assert stats["total_requests"] == 4
            assert stats["hit_rate"] == 0.5
            assert stats["cache_size"] == 2

    async def test_cache_clear_functionality(self, server):
        """Test cache clearing through server."""

        class MockResult:
            def to_dict(self):
                return {"path": "test/path", "score": 0.9}

            @property
            def document(self):
                class MockDoc:
                    raw_data = {}

                    class MockMeta:
                        path_name = "test/path"

                    metadata = MockMeta()

                return MockDoc()

        mock_results = [MockResult()]

        with patch.object(server.search_composer, "search", return_value=mock_results):
            # Add some cached results
            await server.search_imas("query1")
            await server.search_imas("query2")

            assert server.search_cache.get_stats()["cache_size"] == 2

            # Clear cache
            server.search_cache.clear()

            assert server.search_cache.get_stats()["cache_size"] == 0

            # Next search should miss cache
            await server.search_imas("query1")
            stats = server.search_cache.get_stats()
            assert stats["misses"] == 3  # Previous misses + 1
