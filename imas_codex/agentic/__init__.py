"""
Agents module for autonomous facility exploration and enrichment.

This module provides:
1. smolagents CodeAgents for autonomous code-generating agents
2. MCP server for tool-based exploration (AgentsServer)
3. Legacy LlamaIndex ReActAgents (deprecated, use smolagents)
4. Reusable tools for graph queries, SSH, and search
5. Cost monitoring with budget enforcement

The CodeAgents write Python code to invoke tools, enabling:
- Loops and conditionals for complex workflows
- Self-debugging through code inspection
- Adaptive problem-solving
- Multi-agent orchestration
"""

# smolagents (preferred)
from imas_codex.agentic.agents import (
    AgentConfig,
    create_agent,
    create_litellm_model,
    create_orchestrator,
    get_agent_monitor,
    get_enrichment_agent,
    get_exploration_agent,
    get_model_for_task,
)

# Legacy (LlamaIndex-based, deprecated)
from imas_codex.agentic.explore import (
    ExplorationResult,
    create_exploration_agent as create_exploration_agent_legacy,
    run_exploration,
    run_exploration_sync,
)
from imas_codex.agentic.llm import (
    DEFAULT_MODEL,
    MODELS,
    get_llm,
    get_model_for_task as get_model_for_task_legacy,
    get_model_id,
)
from imas_codex.agentic.monitor import (
    AgentMonitor,
    BudgetExhaustedError,
    create_step_callback,
    estimate_cost,
    estimate_task_cost,
)
from imas_codex.agentic.react import (
    AgentConfig as AgentConfigLegacy,
    BatchProgress,
    EnrichmentResult,
    batch_enrich_paths,
    compose_batches,
    create_agent as create_agent_legacy,
    discover_nodes_to_enrich,
    estimate_enrichment_cost,
    get_enrichment_agent as get_enrichment_agent_legacy,
    get_exploration_agent as get_exploration_agent_legacy,
    get_mapping_agent as get_mapping_agent_legacy,
    get_parent_path,
    quick_agent_task,
    react_batch_enrich_paths,
    run_agent,
    run_agent_sync,
)
from imas_codex.agentic.server import AgentsServer
from imas_codex.agentic.session import CostTracker, LLMSession, create_session
from imas_codex.agentic.smol_explore import (
    ExplorationProgress,
    SmolExplorationAgent,
    SmolExplorationResult,
    explore_facility,
    explore_facility_sync,
)
from imas_codex.agentic.smolagents_tools import (
    get_all_tools,
    get_enrichment_tools,
    get_exploration_tools,
)
from imas_codex.agentic.tools import (
    get_all_tools as get_all_tools_legacy,
    get_exploration_tools as get_exploration_tools_legacy,
    get_graph_tool,
    get_imas_tools,
    get_search_tools,
    get_ssh_tools,
)

__all__ = [
    # smolagents (preferred)
    "AgentConfig",
    "AgentMonitor",
    "BudgetExhaustedError",
    "create_agent",
    "create_litellm_model",
    "create_orchestrator",
    "create_step_callback",
    "estimate_cost",
    "estimate_task_cost",
    "get_agent_monitor",
    "get_enrichment_agent",
    "get_exploration_agent",
    "get_model_for_task",
    "get_enrichment_tools",
    "get_exploration_tools",
    "get_all_tools",
    # Exploration (smolagents)
    "SmolExplorationAgent",
    "SmolExplorationResult",
    "ExplorationProgress",
    "explore_facility",
    "explore_facility_sync",
    # MCP Server
    "AgentsServer",
    # LLM configuration (legacy)
    "get_llm",
    "get_model_id",
    "DEFAULT_MODEL",
    "MODELS",
    # Session management (legacy)
    "LLMSession",
    "CostTracker",
    "create_session",
    # Legacy agent execution (deprecated - use smolagents)
    "run_agent",
    "run_agent_sync",
    "run_exploration",
    "run_exploration_sync",
    "quick_agent_task",
    "batch_enrich_paths",
    "react_batch_enrich_paths",
    "discover_nodes_to_enrich",
    "estimate_enrichment_cost",
    "compose_batches",
    "get_parent_path",
    "EnrichmentResult",
    "ExplorationResult",
    "BatchProgress",
    # Legacy tools (deprecated)
    "get_imas_tools",
    "get_graph_tool",
    "get_ssh_tools",
    "get_search_tools",
]
