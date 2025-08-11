"""
MCP tool decorator for marking methods as MCP tools.

This decorator provides a standard way to mark methods as MCP tools
and attach descriptions that can be used by the MCP framework.
"""

from collections.abc import Callable
from typing import Any


def mcp_tool(description: str):
    """
    Decorator to mark methods as MCP tools with descriptions.

    This decorator adds metadata to methods to identify them as MCP tools
    and provides a description that can be used by the MCP framework
    for tool discovery and documentation.

    Args:
        description: Human-readable description of what the tool does

    Returns:
        Decorated function with MCP tool metadata
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._mcp_tool = True
        func._mcp_description = description
        return func

    return decorator
