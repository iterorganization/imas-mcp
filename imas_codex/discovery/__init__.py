"""
Discovery Engine for remote facility exploration.

This module provides configuration management for remote fusion facilities.
Exploration is done via direct SSH; see imas_codex/config/README.md for guidance.
"""

from imas_codex.discovery.config import get_config, list_facilities

__all__ = [
    "get_config",
    "list_facilities",
]
