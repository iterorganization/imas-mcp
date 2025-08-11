"""
Base provider interface for MCP tools and resources.

This module defines the common interface for MCP providers that can register
tools and resources with the FastMCP server. This enables a composable
architecture where the server acts as an integrator.
"""

from abc import ABC, abstractmethod

from fastmcp import FastMCP


class MCPProvider(ABC):
    """Base class for MCP providers (tools and resources)."""

    @abstractmethod
    def register(self, mcp: FastMCP) -> None:
        """Register tools and/or resources with the MCP server.

        Args:
            mcp: The FastMCP server instance to register with
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging and identification."""
        pass
