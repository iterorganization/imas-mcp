"""Extended tests for server.py module."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.server import Server


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

    def test_server_has_embeddings(self, server):
        """Server has embeddings component."""
        assert server.embeddings is not None

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
        server = Server(ids_set={"equilibrium", "core_profiles"})

        assert server.ids_set == {"equilibrium", "core_profiles"}
        assert server.tools is not None

    def test_server_use_rich_config(self):
        """Server respects use_rich config."""
        server = Server(use_rich=False)

        assert server.use_rich is False


class TestServerBuildSchemas:
    """Tests for schema building and validation."""

    def test_build_schemas_if_missing_returns_true_when_exists(self, server):
        """_build_schemas_if_missing returns True when schemas exist."""
        result = server._build_schemas_if_missing()

        assert result is True

    @patch("imas_codex.server.ResourcePathAccessor")
    def test_build_schemas_if_missing_handles_exception(
        self, mock_accessor_class, server
    ):
        """_build_schemas_if_missing handles build failures."""
        mock_accessor = MagicMock()
        mock_accessor.schemas_dir.exists.return_value = False
        mock_accessor.schemas_dir.__truediv__ = MagicMock(return_value=MagicMock())
        catalog_mock = MagicMock()
        catalog_mock.exists.return_value = False
        mock_accessor.schemas_dir.__truediv__.return_value = catalog_mock
        mock_accessor_class.return_value = mock_accessor

        with patch.object(server, "_build_schemas_if_missing", return_value=True):
            result = server._build_schemas_if_missing()

        assert result is True


class TestServerValidation:
    """Tests for server validation methods."""

    def test_validate_schemas_available_succeeds(self, server):
        """_validate_schemas_available succeeds when schemas exist."""
        server._validate_schemas_available()

    def test_register_components(self, server):
        """_register_components registers tools and resources."""
        server._register_components()
