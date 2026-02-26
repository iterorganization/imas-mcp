"""Extended tests for server.py module."""

import pytest

from imas_codex.server import Server
from tests.conftest import _create_mock_graph_client


class TestServerExtended:
    """Extended tests for the Server class."""

    def test_unsupported_transport_raises_error(self, server):
        """Server raises error for unsupported transport."""
        with pytest.raises(ValueError, match="Unsupported transport"):
            server.run(transport="invalid")

    def test_uptime_seconds_returns_non_negative(self, server):
        """uptime_seconds returns non-negative value."""
        uptime = server.uptime_seconds()

        assert uptime >= 0.0

    def test_get_version_returns_string(self, server):
        """_get_version returns a string."""
        version = server._get_version()

        assert isinstance(version, str)

    @pytest.mark.asyncio
    async def test_async_context_manager(self, server):
        """Server works as async context manager."""
        async with server:
            assert server is not None


class TestServerComponents:
    """Tests for server component initialization."""

    def test_server_has_tools(self, server):
        """Server has tools component."""
        assert server.tools is not None

    def test_server_has_resources(self, server):
        """Server has resources component."""
        assert server.resources is not None

    def test_server_has_graph_client(self, server):
        """Server has graph_client."""
        assert server.graph_client is not None

    def test_server_has_mcp(self, server):
        """Server has MCP instance."""
        assert server.mcp is not None

    def test_server_started_at_is_set(self, server):
        """Server has started_at timestamp."""
        from datetime import datetime

        assert server.started_at is not None
        assert isinstance(server.started_at, datetime)

    def test_server_with_ids_set(self):
        """Server initializes with ids_set filter."""
        mock_gc = _create_mock_graph_client()
        server = Server(ids_set={"equilibrium", "core_profiles"}, graph_client=mock_gc)

        assert server.ids_set == {"equilibrium", "core_profiles"}
        assert server.tools is not None

    def test_server_initializes(self):
        """Server initializes with graph client."""
        mock_gc = _create_mock_graph_client()
        server = Server(graph_client=mock_gc)

        assert server.tools is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
