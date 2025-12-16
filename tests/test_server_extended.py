"""Extended tests for server.py module."""

import asyncio

import pytest

from imas_mcp.server import Server


class TestServerExtended:
    """Extended tests for the Server class."""

    def test_unsupported_transport_raises_error(self):
        """Server raises error for unsupported transport."""
        server = Server(ids_set={"equilibrium"})

        with pytest.raises(ValueError, match="Unsupported transport"):
            server.run(transport="invalid")

    def test_uptime_seconds_returns_non_negative(self):
        """uptime_seconds returns non-negative value."""
        server = Server(ids_set={"equilibrium"})

        uptime = server.uptime_seconds()

        assert uptime >= 0.0

    def test_get_version_returns_string(self):
        """_get_version returns a string."""
        server = Server(ids_set={"equilibrium"})

        version = server._get_version()

        assert isinstance(version, str)

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Server works as async context manager."""
        async with Server(ids_set={"equilibrium"}) as server:
            assert server is not None

    def test_setup_signal_handlers_does_not_raise(self):
        """setup_signal_handlers completes without error."""
        server = Server(ids_set={"equilibrium"})

        # Should not raise
        server.setup_signal_handlers()
