"""
Agents module for remote facility exploration.

This module provides prompt-driven exploration of remote fusion facilities.
The Cursor chat LLM orchestrates exploration via terminal commands,
with session logging and learning persistence.

No subagents - the LLM in chat is the orchestrator.
"""

from imas_codex.agents.server import AgentsServer

__all__ = [
    "AgentsServer",
]
