"""
Explain tool implementation.

This module contains the explain_concept tool logic extracted from the monolithic Tools class.
"""

from .base import BaseTool


class Explain(BaseTool):
    """Tool for explaining IMAS concepts."""

    def get_tool_name(self) -> str:
        return "explain_concept"

    # TODO: Extract explain_concept method from tools.py in Phase 5
