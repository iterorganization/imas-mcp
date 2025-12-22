"""
Agents module for remote facility exploration.

This module provides the exploration agent architecture where autonomous
subagents explore remote fusion facilities via SSH, orchestrated by a
frontier LLM (Claude Opus 4.5) using a ReAct loop.
"""

from imas_codex.agents.exploration import (
    ExplorationAgent,
    ExplorationResult,
    ExplorationState,
)

# Keep FileExplorerAgent as alias for backwards compatibility
from imas_codex.agents.file_explorer import FileExplorerAgent
from imas_codex.agents.knowledge import (
    FacilityKnowledge,
    load_knowledge,
    persist_learnings,
)
from imas_codex.agents.server import AgentsServer

__all__ = [
    # Server
    "AgentsServer",
    # Exploration
    "ExplorationAgent",
    "ExplorationResult",
    "ExplorationState",
    # Knowledge
    "FacilityKnowledge",
    "load_knowledge",
    "persist_learnings",
    # Legacy alias
    "FileExplorerAgent",
]
