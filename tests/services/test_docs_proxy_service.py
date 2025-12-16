"""
Tests for DocsProxyService.

Tests HTTP request handling, result parsing, and error handling.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from imas_mcp.exceptions import DocsServerError
from imas_mcp.services.docs_proxy_service import (
    DocsProxyService,
    LibraryNotFoundError,
    Settings,
)
from imas_mcp.services.docs_server_manager import (
    DocsServerManager,
    DocsServerUnavailableError,
    PortAllocationError,
)


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()

        assert settings.docs_timeout == 30
        assert settings.default_docs_limit == 5
        assert settings.max_docs_limit == 20

    def test_settings_from_env(self, monkeypatch):
        """Test settings from environment variables."""
        monkeypatch.setenv("DOCS_TIMEOUT", "60")
        monkeypatch.setenv("DEFAULT_DOCS_LIMIT", "10")
        monkeypatch.setenv("MAX_DOCS_LIMIT", "50")

        settings = Settings()

        assert settings.docs_timeout == 60
        assert settings.default_docs_limit == 10
        assert settings.max_docs_limit == 50


class TestLibraryNotFoundError:
    """Tests for LibraryNotFoundError."""

    def test_error_message(self):
        """Test error message formatting."""
        error = LibraryNotFoundError("unknown-lib", ["lib1", "lib2"])

        assert error.library == "unknown-lib"
        assert error.available_libraries == ["lib1", "lib2"]
        assert "unknown-lib" in str(error)
        assert "lib1, lib2" in str(error)


class TestDocsProxyServiceInit:
    """Tests for DocsProxyService initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        service = DocsProxyService()

        assert service.settings is not None
        assert service.docs_manager is not None
        assert isinstance(service.timeout, int)

    def test_init_with_custom_settings(self):
        """Test initialization with custom settings."""
        settings = Settings()
        settings.docs_timeout = 60

        service = DocsProxyService(settings=settings)

        assert service.timeout == 60

    def test_init_with_custom_manager(self):
        """Test initialization with custom manager."""
        manager = MagicMock(spec=DocsServerManager)

        service = DocsProxyService(docs_manager=manager)

        assert service.docs_manager is manager


class TestParseSearchResults:
    """Tests for _parse_search_results method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService instance."""
        manager = MagicMock(spec=DocsServerManager)
        return DocsProxyService(docs_manager=manager)

    def test_parse_empty_text(self, service):
        """Test parsing empty text."""
        result = service._parse_search_results("")

        assert result["results"] == []
        assert result["count"] == 0

    def test_parse_single_result(self, service):
        """Test parsing a single result."""
        text = """------------------------------------------------------------
Result 1: https://example.com/docs

This is the content of the first result.
It has multiple lines."""

        result = service._parse_search_results(text)

        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com/docs"
        assert "content of the first result" in result["results"][0]["content"]

    def test_parse_multiple_results(self, service):
        """Test parsing multiple results."""
        text = """------------------------------------------------------------
Result 1: https://example.com/page1

Content for page 1.

------------------------------------------------------------
Result 2: https://example.com/page2

Content for page 2."""

        result = service._parse_search_results(text)

        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["url"] == "https://example.com/page1"
        assert result["results"][1]["url"] == "https://example.com/page2"

    def test_parse_result_without_separator(self, service):
        """Test parsing results without dashes separator."""
        text = """Result 1: https://example.com

Content here"""

        result = service._parse_search_results(text)

        assert result["count"] == 1
        assert result["results"][0]["url"] == "https://example.com"

    def test_parse_result_default_fields(self, service):
        """Test that parsed results have default fields."""
        text = """Result 1: https://example.com

Content"""

        result = service._parse_search_results(text)

        assert result["results"][0]["score"] is None
        assert result["results"][0]["mimeType"] == "text/plain"


class TestSearchDocs:
    """Tests for search_docs method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        manager.is_running = True
        manager.allocated_port = 12345
        manager.base_url = "http://127.0.0.1:12345"
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_search_without_library(self, service):
        """Test search without library returns error."""
        result = await service.search_docs("test query")

        assert "error" in result
        assert "Library parameter is required" in result["error"]
        assert result["library_required"] is True

    @pytest.mark.asyncio
    async def test_search_with_library(self, service):
        """Test search with library makes HTTP request."""
        mock_response = {"results": [], "query": "test"}

        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.search_docs("test query", library="test-lib")

            mock_request.assert_called_once()
            assert result["query"] == "test query"
            assert result["library"] == "test-lib"

    @pytest.mark.asyncio
    async def test_search_limit_clamping(self, service):
        """Test that limit is clamped to valid range."""
        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {}

            # Test limit too high
            await service.search_docs("query", library="lib", limit=100)

            call_args = mock_request.call_args
            # limit should be clamped to max_docs_limit (20)
            assert call_args[1]["json"]["limit"] == 20

    @pytest.mark.asyncio
    async def test_search_with_version(self, service):
        """Test search with version parameter."""
        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = {}

            result = await service.search_docs("query", library="lib", version="1.0.0")

            call_args = mock_request.call_args
            assert call_args[1]["json"]["version"] == "1.0.0"
            assert result["version"] == "1.0.0"


class TestListAvailableLibraries:
    """Tests for list_available_libraries method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        manager.is_running = True
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_list_libraries_dict_format(self, service):
        """Test parsing library list from dict format."""
        mock_response = {
            "result": {
                "data": [
                    {"library": "lib1"},
                    {"library": "lib2"},
                ]
            }
        }

        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.list_available_libraries()

            assert result == ["lib1", "lib2"]

    @pytest.mark.asyncio
    async def test_list_libraries_name_format(self, service):
        """Test parsing library list with 'name' key."""
        mock_response = {
            "result": {
                "data": [
                    {"name": "lib1"},
                    {"name": "lib2"},
                ]
            }
        }

        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.list_available_libraries()

            assert result == ["lib1", "lib2"]

    @pytest.mark.asyncio
    async def test_list_libraries_string_format(self, service):
        """Test parsing library list from string format."""
        mock_response = {"result": {"data": ["lib1", "lib2"]}}

        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.list_available_libraries()

            assert result == ["lib1", "lib2"]

    @pytest.mark.asyncio
    async def test_list_libraries_list_format(self, service):
        """Test parsing library list when response is a list."""
        mock_response = ["lib1", "lib2"]

        with patch.object(
            service, "_make_http_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.list_available_libraries()

            assert result == ["lib1", "lib2"]


class TestProxyListLibraries:
    """Tests for proxy_list_libraries method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        manager.is_running = True
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_proxy_list_empty(self, service):
        """Test parsing 'No libraries indexed yet' message."""
        mock_response = {"text": "No libraries indexed yet."}

        with patch.object(
            service, "_make_mcp_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.proxy_list_libraries()

            assert result == []

    @pytest.mark.asyncio
    async def test_proxy_list_with_libraries(self, service):
        """Test parsing library list from bullet points."""
        mock_response = {
            "text": """Indexed libraries:

- lib1
- lib2
- lib3"""
        }

        with patch.object(
            service, "_make_mcp_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.proxy_list_libraries()

            assert result == ["lib1", "lib2", "lib3"]


class TestProxySearchDocs:
    """Tests for proxy_search_docs method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        manager.is_running = True
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_proxy_search_success(self, service):
        """Test successful proxy search."""
        mock_response = {
            "text": """------------------------------------------------------------
Result 1: https://example.com

Content here"""
        }

        with patch.object(
            service, "_make_mcp_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.return_value = mock_response

            result = await service.proxy_search_docs(
                query="test", library="lib", version="1.0", limit=5
            )

            assert result["success"] is True
            assert result["query"] == "test"
            assert result["library"] == "lib"
            assert result["count"] == 1


class TestValidateLibraryExists:
    """Tests for validate_library_exists method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_validate_exists(self, service):
        """Test validation when library exists."""
        with patch.object(
            service, "list_available_libraries", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = ["lib1", "lib2"]

            result = await service.validate_library_exists("lib1")

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_not_exists(self, service):
        """Test validation when library doesn't exist."""
        with patch.object(
            service, "list_available_libraries", new_callable=AsyncMock
        ) as mock_list:
            mock_list.return_value = ["lib1", "lib2"]

            result = await service.validate_library_exists("unknown")

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_on_error(self, service):
        """Test validation returns False on error."""
        with patch.object(
            service, "list_available_libraries", new_callable=AsyncMock
        ) as mock_list:
            mock_list.side_effect = DocsServerError("Server error")

            result = await service.validate_library_exists("lib")

            assert result is False


class TestFindBestVersion:
    """Tests for find_best_version method."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = MagicMock(spec=DocsServerManager)
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_find_best_version_latest(self, service):
        """Test finding best version when 'latest' requested."""
        with patch.object(
            service, "get_library_versions", new_callable=AsyncMock
        ) as mock_versions:
            mock_versions.return_value = ["2.0.0", "1.0.0"]

            result = await service.find_best_version("lib", "latest")

            assert result == "2.0.0"

    @pytest.mark.asyncio
    async def test_find_best_version_specific(self, service):
        """Test finding best version with specific version."""
        result = await service.find_best_version("lib", "1.5.0")

        assert result == "1.5.0"

    @pytest.mark.asyncio
    async def test_find_best_version_latest_no_versions(self, service):
        """Test finding best version when no versions available."""
        with patch.object(
            service, "get_library_versions", new_callable=AsyncMock
        ) as mock_versions:
            mock_versions.return_value = []

            result = await service.find_best_version("lib", "latest")

            assert result == "latest"


class TestContextManager:
    """Tests for context manager functionality."""

    @pytest.fixture
    def service(self):
        """Create a DocsProxyService with mock manager."""
        manager = AsyncMock(spec=DocsServerManager)
        return DocsProxyService(docs_manager=manager)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, service):
        """Test async context manager."""
        async with service:
            pass

        service.docs_manager.start_server.assert_called_once()
        service.docs_manager.stop_server.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, service):
        """Test close method."""
        await service.close()

        service.docs_manager.stop_server.assert_called_once()
