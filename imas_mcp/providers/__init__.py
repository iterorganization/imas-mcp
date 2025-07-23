"""
Base class for MCP providers.

This module defines the interface for providers that can be registered
with the IMAS MCP server, whether they provide tools, resources, or both.
"""

from abc import ABC, abstractmethod

from fastmcp import FastMCP


class MCPProvider(ABC):
    """Base class for MCP providers.

    Providers can register tools, resources, or both with the MCP server.
    The separation between tools and resources is handled at the registration
    level using mcp.tool() and mcp.resource() decorators, not through
    separate provider classes.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        pass

    @abstractmethod
    def register(self, mcp: FastMCP):
        """Register this provider's capabilities with the MCP server.

        Args:
            mcp: FastMCP server instance to register with
        """
        pass
