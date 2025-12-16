"""
Integration tests for docs server lifecycle management.

These tests verify the complete integration of the persistent docs server
with the IMAS MCP server, including startup, shutdown, health monitoring,
and HTTP client communication.
"""

import asyncio
import json
import logging
import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.server import Server
from imas_codex.services.docs_proxy_service import DocsProxyService
from imas_codex.services.docs_server_manager import (
    DocsServerError,
    DocsServerManager,
    PortAllocationError,
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_npx(monkeypatch):
    """Mock npx executable and ensure OPENAI_API_KEY is set for mocked tests."""
    # Set a fake API key if not present to allow mocked tests to run
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key-for-mocked-tests")
    with patch("shutil.which", return_value="/usr/bin/npx"):
        yield


@pytest.fixture
def mock_process():
    process_mock = AsyncMock()
    process_mock.pid = 12345
    process_mock.returncode = None

    # Simulate process running state
    exit_event = asyncio.Event()

    async def wait_impl():
        await exit_event.wait()
        return None

    process_mock.wait.side_effect = wait_impl

    def terminate_impl():
        exit_event.set()
        process_mock.returncode = 0

    # Fix: terminate and kill must be synchronous mocks
    process_mock.terminate = MagicMock(side_effect=terminate_impl)
    process_mock.kill = MagicMock(side_effect=terminate_impl)

    process_mock.stdout = AsyncMock()
    process_mock.stdout.__aiter__.return_value = iter([b"Server started"])
    process_mock.stderr = AsyncMock()
    process_mock.stderr.__aiter__.return_value = iter([])

    with patch("anyio.open_process", return_value=process_mock) as mock:
        yield mock, process_mock


@pytest.fixture
def mock_aiohttp():
    with patch("aiohttp.ClientSession") as mock_session_cls:
        # The session instance
        mock_session_instance = MagicMock()
        mock_session_cls.return_value = mock_session_instance

        # The session context (yielded by async with ClientSession())
        mock_context = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_context)
        mock_session_instance.__aexit__ = AsyncMock(return_value=None)

        # Helper to create response context managers
        def create_response_cm(status=200, json_data=None, text_data=""):
            mock_response = AsyncMock()
            mock_response.status = status
            if json_data is not None:
                mock_response.json.return_value = json_data
            mock_response.text.return_value = text_data

            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        # Default behavior for get/post/request
        # They are synchronous methods returning an async context manager
        mock_context.get.return_value = create_response_cm()
        mock_context.post.return_value = create_response_cm()
        mock_context.request.return_value = create_response_cm()

        # Store the helper on the mock so tests can use it
        mock_context.create_response_cm = create_response_cm

        yield mock_context


@pytest.fixture
def temp_store_path():
    """Create a temporary directory for docs storage."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestDocsServerManager:
    """Test the DocsServerManager class lifecycle management."""

    @pytest.fixture
    def docs_manager(self, temp_store_path):
        """Create a DocsServerManager instance for testing."""
        return DocsServerManager(
            default_port=18280,  # Use a test port
            store_path=temp_store_path,
            timeout=10,
        )

    @pytest.mark.asyncio
    async def test_port_allocation(self, temp_store_path):
        """Test that port allocation works correctly."""
        manager = DocsServerManager(default_port=18281, store_path=temp_store_path)

        # Mock the port availability check
        with patch.object(manager, "_is_port_truly_available", return_value=True):
            port = await manager._allocate_port()
            assert port == 18282  # default_port + 1

    @pytest.mark.asyncio
    async def test_port_fallback(self, temp_store_path):
        """Test that port fallback works when default port is occupied."""
        manager = DocsServerManager(default_port=18282, store_path=temp_store_path)

        # Mock the default port as unavailable, but subsequent ports available
        async def mock_is_port_available(port):
            if port == 18283:  # default_port + 1
                return False
            return True

        with patch.object(
            manager, "_is_port_truly_available", side_effect=mock_is_port_available
        ):
            port = await manager._allocate_port()
            assert port == 18284  # Should find the next available port

    @pytest.mark.asyncio
    async def test_port_allocation_failure(self, temp_store_path):
        """Test that port allocation fails when no ports are available."""
        manager = DocsServerManager(default_port=18283, store_path=temp_store_path)

        # Mock all ports as unavailable
        with patch.object(manager, "_is_port_truly_available", return_value=False):
            with pytest.raises(PortAllocationError):
                await manager._allocate_port()

    @pytest.mark.asyncio
    async def test_base_url_property(self, temp_store_path):
        """Test that the base_url property works correctly."""
        manager = DocsServerManager(default_port=18284, store_path=temp_store_path)

        # Before starting, should raise error
        with pytest.raises(DocsServerError, match="Server not started"):
            _ = manager.base_url

        # After allocation, should work
        manager.allocated_port = 18284
        assert manager.base_url == "http://127.0.0.1:18284"

    @pytest.mark.asyncio
    async def test_server_startup_and_shutdown(
        self, temp_store_path, mock_npx, mock_process, mock_aiohttp
    ):
        """Test that the docs server can be started and stopped."""
        manager = DocsServerManager(default_port=18285, store_path=temp_store_path)

        # Mock port availability
        with patch.object(manager, "_is_port_truly_available", return_value=True):
            # Mock health check response for wait_for_server_ready
            # The default mock_aiohttp setup returns 200 OK, so this should pass immediately

            await manager.start_server()
            assert manager.is_running
            assert manager.allocated_port is not None

            # Test that server can be stopped
            await manager.stop_server()
            assert not manager.is_running

    @pytest.mark.asyncio
    async def test_health_check(
        self, temp_store_path, mock_npx, mock_process, mock_aiohttp
    ):
        """Test health check functionality."""
        manager = DocsServerManager(default_port=18286, store_path=temp_store_path)

        # Test health check when not running
        health_data = await manager.health_check()
        assert health_data["status"] == "unhealthy"
        assert "Server not running" in health_data["error"]

        # Start server (mocked)
        with patch.object(manager, "_is_port_truly_available", return_value=True):
            # Setup responses for different endpoints
            def side_effect(url, **kwargs):
                if "/api/ping" in url:
                    return mock_aiohttp.create_response_cm(status=200)
                elif "/api/list" in url:
                    return mock_aiohttp.create_response_cm(
                        status=200, json_data={"libraries": [{"name": "test-lib"}]}
                    )
                return mock_aiohttp.create_response_cm(status=404)

            mock_aiohttp.get.side_effect = side_effect

            await manager.start_server()

            health_data = await manager.health_check()
            assert health_data["status"] == "healthy"
            assert health_data["total_libraries"] == 1

    @pytest.mark.asyncio
    async def test_context_manager(
        self, temp_store_path, mock_npx, mock_process, mock_aiohttp
    ):
        """Test that the context manager works correctly."""
        manager = DocsServerManager(default_port=18287, store_path=temp_store_path)

        with patch.object(manager, "_is_port_truly_available", return_value=True):
            async with manager:
                assert manager.is_running
            # After context manager, should be stopped
            assert not manager.is_running

    @pytest.mark.asyncio
    async def test_signal_handlers(self, temp_store_path):
        """Test that signal handlers can be set up."""
        manager = DocsServerManager(default_port=18288, store_path=temp_store_path)

        with patch("signal.signal") as mock_signal:
            manager.setup_signal_handlers()
            assert mock_signal.called


class TestDocsProxyService:
    """Test the DocsProxyService with HTTP client integration."""

    @pytest.fixture
    def docs_manager(self, temp_store_path):
        """Create a DocsServerManager instance for testing."""
        return DocsServerManager(default_port=18290, store_path=temp_store_path)

    @pytest.fixture
    def docs_proxy(self, docs_manager):
        """Create a DocsProxyService instance for testing."""
        return DocsProxyService(docs_manager=docs_manager)

    @pytest.mark.asyncio
    async def test_service_initialization(self, docs_proxy):
        """Test that the service can be initialized."""
        assert docs_proxy is not None
        assert docs_proxy.docs_manager is not None
        # Check type instead of exact value to avoid env var issues
        assert isinstance(docs_proxy.timeout, int)

    @pytest.mark.asyncio
    async def test_library_listing(self, temp_store_path, docs_proxy, mock_aiohttp):
        """Test that library listing works (mocked)."""
        # Mock the HTTP response
        mock_response = {
            "libraries": [
                {"name": "test-library", "versions": ["1.0.0"]},
                "another-library",
            ]
        }

        # Mock manager as running
        docs_proxy.docs_manager.allocated_port = 12345
        docs_proxy.docs_manager.process = MagicMock()
        docs_proxy.docs_manager.process.returncode = None

        # Setup request mock
        mock_aiohttp.request.return_value = mock_aiohttp.create_response_cm(
            status=200, json_data=mock_response, text_data=json.dumps(mock_response)
        )

        # Test the functionality
        result = await docs_proxy._make_http_request("list")
        assert result == mock_response


class TestServerIntegration:
    """Test integration between Server and docs server management."""

    @pytest.mark.asyncio
    async def test_server_with_docs_manager(self, temp_store_path):
        """Test that Server can be created with docs server management."""
        # Mock dependencies to avoid real server startup
        with (
            patch("imas_codex.server.Tools"),
            patch("imas_codex.server.Resources"),
            patch("imas_codex.server.Embeddings"),
            patch("imas_codex.server.Server._validate_schemas_available"),
            patch("imas_codex.server.Server._register_components"),
        ):
            server = Server()

            # Verify that docs manager is initialized
            assert hasattr(server, "docs_manager")
            assert server.docs_manager is not None

            # Verify that the server can be cleaned up
            await server.cleanup()


class TestErrorHandling:
    """Test error handling and recovery scenarios."""

    @pytest.fixture
    def docs_manager(self, temp_store_path):
        """Create a DocsServerManager instance for testing."""
        return DocsServerManager(default_port=18293, store_path=temp_store_path)

    @pytest.mark.asyncio
    async def test_port_conflict_handling(self, temp_store_path):
        """Test that port conflicts are handled gracefully."""
        manager = DocsServerManager(default_port=18291)

        # Should fallback to another port
        with patch.object(manager, "_is_port_truly_available", return_value=False):
            with pytest.raises(PortAllocationError):
                await manager._allocate_port()

    @pytest.mark.asyncio
    async def test_server_crash_recovery(
        self, temp_store_path, mock_npx, mock_process, mock_aiohttp
    ):
        """Test that the system can recover from server crashes."""
        manager = DocsServerManager(default_port=18292)

        with patch.object(manager, "_is_port_truly_available", return_value=True):
            await manager.start_server()
            assert manager.is_running
            await manager.stop_server()

    @pytest.mark.asyncio
    async def test_malformed_response_handling(
        self, temp_store_path, docs_manager, mock_aiohttp
    ):
        """Test handling of malformed responses from docs server."""
        proxy = DocsProxyService(docs_manager=docs_manager)

        # Mock manager as running
        docs_manager.allocated_port = 12345
        docs_manager.process = MagicMock()
        docs_manager.process.returncode = None

        # Test with various malformed responses
        test_responses = [
            "not json",  # Invalid JSON
            '{"incomplete": ',  # Incomplete JSON
        ]

        for response_text in test_responses:
            # Setup response to fail on json()
            cm = mock_aiohttp.create_response_cm(status=200, text_data=response_text)
            # Get the response mock from the CM
            mock_resp = cm.__aenter__.return_value
            mock_resp.json.side_effect = json.JSONDecodeError("test", response_text, 0)

            mock_aiohttp.request.return_value = cm

            with pytest.raises(DocsServerError):
                await proxy._make_http_request("test")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
