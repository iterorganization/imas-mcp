"""
Discovery Engine for remote facility exploration.

This module provides tools for surveying remote fusion facilities via SSH,
discovering available tools, and cataloging filesystem structures.
"""

from imas_codex.discovery.config import get_config, list_facilities
from imas_codex.discovery.connection import FacilityConnection, ScriptResult
from imas_codex.discovery.prompts import load_prompt
from imas_codex.discovery.sandbox import CommandSandbox

__all__ = [
    # Configuration
    "get_config",
    "list_facilities",
    # Connection
    "FacilityConnection",
    "ScriptResult",
    # Prompts
    "load_prompt",
    # Sandbox
    "CommandSandbox",
]
