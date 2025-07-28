"""
Base tool functionality for IMAS MCP tools.

This module contains common functionality shared across all tool implementations.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Base class for all IMAS MCP tools."""

    def __init__(self):
        self.logger = logger

    @abstractmethod
    def get_tool_name(self) -> str:
        """Return the name of this tool."""
        pass

    def _create_error_response(
        self, error_message: str, query: str = ""
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return {
            "error": error_message,
            "query": query,
            "tool": self.get_tool_name(),
            "status": "error",
        }
