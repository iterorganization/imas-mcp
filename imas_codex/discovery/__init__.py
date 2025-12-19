"""
Discovery Engine for remote facility exploration.

This module provides tools for surveying remote fusion facilities via SSH/Fabric,
discovering available tools, and cataloging filesystem structures.

The main entry point is the Investigator class, which runs an agentic LLM loop
to explore remote facilities.
"""

from imas_codex.discovery.config import get_config, list_facilities
from imas_codex.discovery.connection import FacilityConnection, ScriptResult
from imas_codex.discovery.investigator import InvestigationResult, Investigator
from imas_codex.discovery.prompts import PromptLoader, get_prompt_loader, load_prompt
from imas_codex.discovery.sandbox import CommandSandbox

__all__ = [
    # Main interface
    "Investigator",
    "InvestigationResult",
    # Configuration
    "get_config",
    "list_facilities",
    # Connection
    "FacilityConnection",
    "ScriptResult",
    # Prompts
    "PromptLoader",
    "get_prompt_loader",
    "load_prompt",
    # Sandbox
    "CommandSandbox",
]
