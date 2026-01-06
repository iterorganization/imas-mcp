"""
Agents module for remote facility exploration.

This module provides:
1. MCP server for tool-based exploration (AgentsServer)
2. LlamaIndex ReActAgents for autonomous exploration
3. Reusable tools for graph queries, SSH, and search

The ReActAgents can autonomously:
- Query the Neo4j knowledge graph
- Execute SSH commands on remote facilities
- Search code examples and IMAS paths
- Synthesize information across multiple sources
"""

from imas_codex.agents.llm import DEFAULT_MODEL, MODELS, get_llm, get_model_id
from imas_codex.agents.react import (
    AgentConfig,
    BatchProgress,
    EnrichmentResult,
    batch_enrich_paths,
    compose_batches,
    create_agent,
    discover_nodes_to_enrich,
    estimate_enrichment_cost,
    get_enrichment_agent,
    get_exploration_agent,
    get_mapping_agent,
    get_parent_path,
    quick_agent_task,
    react_batch_enrich_paths,
    run_agent,
    run_agent_sync,
)
from imas_codex.agents.server import AgentsServer
from imas_codex.agents.tools import (
    get_all_tools,
    get_exploration_tools,
    get_graph_tool,
    get_imas_tools,
    get_search_tools,
    get_ssh_tools,
)

__all__ = [
    # MCP Server
    "AgentsServer",
    # LLM configuration
    "get_llm",
    "get_model_id",
    "DEFAULT_MODEL",
    "MODELS",
    # Agent factories
    "get_enrichment_agent",
    "get_mapping_agent",
    "get_exploration_agent",
    "create_agent",
    "AgentConfig",
    # Agent execution
    "run_agent",
    "run_agent_sync",
    "quick_agent_task",
    "batch_enrich_paths",
    "react_batch_enrich_paths",
    "discover_nodes_to_enrich",
    "estimate_enrichment_cost",
    "compose_batches",
    "get_parent_path",
    "EnrichmentResult",
    "BatchProgress",
    # Tools
    "get_exploration_tools",
    "get_imas_tools",
    "get_graph_tool",
    "get_ssh_tools",
    "get_search_tools",
    "get_all_tools",
]
