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

Imports are lazy to avoid pulling in heavy dependencies (smolagents,
huggingface_hub) when only lightweight submodules like prompt_loader
are needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imas_codex.agentic.agents import (
        PRESETS as PRESETS,
        AgentConfig as AgentConfig,
        create_agent as create_agent,
        create_litellm_model as create_litellm_model,
        get_agent_monitor as get_agent_monitor,
        get_enrichment_agent as get_enrichment_agent,
        get_model_id as get_model_id,
    )
    from imas_codex.agentic.enrich import (
        BatchProgress as BatchProgress,
        EnrichmentResult as EnrichmentResult,
        batch_enrich_paths as batch_enrich_paths,
        batch_enrich_paths_sync as batch_enrich_paths_sync,
        compose_batches as compose_batches,
        discover_nodes_to_enrich as discover_nodes_to_enrich,
        estimate_enrichment_cost as estimate_enrichment_cost,
        get_parent_path as get_parent_path,
        quick_task as quick_task,
        quick_task_sync as quick_task_sync,
    )
    from imas_codex.agentic.explore import (
        ExplorationAgent as ExplorationAgent,
        ExplorationProgress as ExplorationProgress,
        ExplorationResult as ExplorationResult,
        explore_facility as explore_facility,
        explore_facility_sync as explore_facility_sync,
    )
    from imas_codex.agentic.monitor import (
        AgentMonitor as AgentMonitor,
        AgentProgressDisplay as AgentProgressDisplay,
        BudgetExhaustedError as BudgetExhaustedError,
        create_progress_callback as create_progress_callback,
        create_step_callback as create_step_callback,
        estimate_cost as estimate_cost,
        estimate_task_cost as estimate_task_cost,
    )
    from imas_codex.agentic.server import AgentsServer as AgentsServer
    from imas_codex.agentic.tools import (
        get_all_tools as get_all_tools,
        get_enrichment_tools as get_enrichment_tools,
        get_exploration_tools as get_exploration_tools,
    )

# Map attribute names to (module, name) for lazy loading
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # agents
    "PRESETS": (".agents", "PRESETS"),
    "AgentConfig": (".agents", "AgentConfig"),
    "create_agent": (".agents", "create_agent"),
    "create_litellm_model": (".agents", "create_litellm_model"),
    "get_agent_monitor": (".agents", "get_agent_monitor"),
    "get_enrichment_agent": (".agents", "get_enrichment_agent"),
    "get_model_id": (".agents", "get_model_id"),
    # enrich
    "BatchProgress": (".enrich", "BatchProgress"),
    "EnrichmentResult": (".enrich", "EnrichmentResult"),
    "batch_enrich_paths": (".enrich", "batch_enrich_paths"),
    "batch_enrich_paths_sync": (".enrich", "batch_enrich_paths_sync"),
    "compose_batches": (".enrich", "compose_batches"),
    "discover_nodes_to_enrich": (".enrich", "discover_nodes_to_enrich"),
    "estimate_enrichment_cost": (".enrich", "estimate_enrichment_cost"),
    "get_parent_path": (".enrich", "get_parent_path"),
    "quick_task": (".enrich", "quick_task"),
    "quick_task_sync": (".enrich", "quick_task_sync"),
    # explore
    "ExplorationAgent": (".explore", "ExplorationAgent"),
    "ExplorationProgress": (".explore", "ExplorationProgress"),
    "ExplorationResult": (".explore", "ExplorationResult"),
    "explore_facility": (".explore", "explore_facility"),
    "explore_facility_sync": (".explore", "explore_facility_sync"),
    # monitor
    "AgentMonitor": (".monitor", "AgentMonitor"),
    "AgentProgressDisplay": (".monitor", "AgentProgressDisplay"),
    "BudgetExhaustedError": (".monitor", "BudgetExhaustedError"),
    "create_progress_callback": (".monitor", "create_progress_callback"),
    "create_step_callback": (".monitor", "create_step_callback"),
    "estimate_cost": (".monitor", "estimate_cost"),
    "estimate_task_cost": (".monitor", "estimate_task_cost"),
    # server
    "AgentsServer": (".server", "AgentsServer"),
    # tools
    "get_all_tools": (".tools", "get_all_tools"),
    "get_enrichment_tools": (".tools", "get_enrichment_tools"),
    "get_exploration_tools": (".tools", "get_exploration_tools"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        # Cache on the module to avoid repeated lookups
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
