"""
Search tool implementation.

This module contains the search_imas tool logic extracted from the monolithic Tools class.
"""

from .base import BaseTool


class Search(BaseTool):
    """Tool for searching IMAS data paths."""

    def get_tool_name(self) -> str:
        return "search_imas"

    # TODO: Extract search_imas method from tools.py in Phase 4
