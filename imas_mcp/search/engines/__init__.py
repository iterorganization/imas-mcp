"""
Search engines for IMAS MCP.

This package contains different search engine implementations for the IMAS MCP system.
Each engine provides a specific search strategy (semantic, lexical, hybrid).
"""

from .base_engine import SearchEngine, SearchEngineError, MockSearchEngine

__all__ = ["SearchEngine", "SearchEngineError", "MockSearchEngine"]
