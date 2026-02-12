"""
Agents module for autonomous facility exploration and enrichment.

This module provides:
1. CodeAgent-based agents for exploration and enrichment
2. MCP server for tool-based exploration (AgentsServer)
3. Reusable tools for graph queries, SSH, and search
4. Cost monitoring with budget enforcement

The CodeAgents write Python code to invoke tools, enabling:
- Loops and conditionals for complex workflows
- Self-debugging through code inspection
- Adaptive problem-solving
- Multi-agent orchestration
"""

# Agent creation and configuration
from imas_codex.agentic.agents import (
    PRESETS,
    AgentConfig,
    create_agent,
    create_litellm_model,
    create_orchestrator,
    get_agent_monitor,
    get_enrichment_agent,
    get_model_for_task,
    get_model_id,
    get_scout_agent,
)

# Enrichment
from imas_codex.agentic.enrich import (
    BatchProgress,
    EnrichmentResult,
    batch_enrich_paths,
    batch_enrich_paths_sync,
    compose_batches,
    discover_nodes_to_enrich,
    estimate_enrichment_cost,
    get_parent_path,
    quick_task,
    quick_task_sync,
)

# Exploration
from imas_codex.agentic.explore import (
    ExplorationAgent,
    ExplorationProgress,
    ExplorationResult,
    explore_facility,
    explore_facility_sync,
)

# LLM configuration (for legacy wiki code that needs LlamaIndex)
from imas_codex.agentic.llm import get_llm

# Cost monitoring and progress display
from imas_codex.agentic.monitor import (
    AgentMonitor,
    AgentProgressDisplay,
    BudgetExhaustedError,
    create_progress_callback,
    create_step_callback,
    estimate_cost,
    estimate_task_cost,
)

# MCP Server
from imas_codex.agentic.server import AgentsServer

# Session management
from imas_codex.agentic.session import CostTracker, LLMSession, create_session

# Tools
from imas_codex.agentic.tools import (
    get_all_tools,
    get_enrichment_tools,
    get_exploration_tools,
)

__all__ = [
    # Agent configuration
    "AgentConfig",
    "create_agent",
    "create_litellm_model",
    "create_orchestrator",
    "get_agent_monitor",
    "get_enrichment_agent",
    "get_scout_agent",
    "get_model_for_task",
    "get_model_id",
    "PRESETS",
    # Enrichment
    "EnrichmentResult",
    "BatchProgress",
    "batch_enrich_paths",
    "batch_enrich_paths_sync",
    "compose_batches",
    "discover_nodes_to_enrich",
    "estimate_enrichment_cost",
    "get_parent_path",
    "quick_task",
    "quick_task_sync",
    # Exploration
    "ExplorationAgent",
    "ExplorationProgress",
    "ExplorationResult",
    "explore_facility",
    "explore_facility_sync",
    # Cost monitoring and progress
    "AgentMonitor",
    "AgentProgressDisplay",
    "BudgetExhaustedError",
    "create_progress_callback",
    "create_step_callback",
    "estimate_cost",
    "estimate_task_cost",
    # MCP Server
    "AgentsServer",
    # LLM (for legacy wiki code)
    "get_llm",
    # Session management
    "LLMSession",
    "CostTracker",
    "create_session",
    # Tools
    "get_all_tools",
    "get_enrichment_tools",
    "get_exploration_tools",
]
