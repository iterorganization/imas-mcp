"""
Overview tool implementation.

This module contains the get_overview tool logic extracted from the monolithic Tools class.
"""

from .base import BaseTool


class Overview(BaseTool):
    """Tool for getting IMAS overview."""

    def get_tool_name(self) -> str:
        return "get_overview"

    # TODO: Extract get_overview method from tools.py in Phase 5
