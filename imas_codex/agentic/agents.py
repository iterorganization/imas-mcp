"""
Smolagents CodeAgent factory with LiteLLM/OpenRouter integration.

Provides autonomous agents that generate Python code to invoke tools,
enabling adaptive problem-solving through code inspection and execution.

Architecture:
    - CodeAgent: Generates Python code to call tools (vs JSON tool calls)
    - LiteLLM: Unified interface to OpenRouter models
    - Budget limits: Cost tracking with configurable USD limits
    - Multi-agent: Managed agents for specialized tasks

Usage:
    from imas_codex.agentic.agents import create_agent, AgentConfig

    config = AgentConfig(
        name="enrichment",
        instructions="Enrich TreeNode metadata with physics descriptions",
        tools=get_enrichment_tools(),
        cost_limit_usd=5.0,
    )
    agent = create_agent(config)
    result = agent.run("Enrich paths from LIUQE equilibrium")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from smolagents import CodeAgent, LiteLLMModel, Tool

from imas_codex.agentic.monitor import AgentMonitor, create_step_callback
from imas_codex.agentic.prompt_loader import load_prompts

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Load prompts at module level
_PROMPTS = load_prompts()


def _get_prompt(name: str) -> str:
    """Get prompt by name with fallback."""
    prompt = _PROMPTS.get(name)
    if prompt is None:
        logger.warning(f"Prompt '{name}' not found")
        return ""
    return prompt.content


# Default model fallback if config loading fails
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


def _load_model_config() -> dict[str, str]:
    """Load model configuration from pyproject.toml."""
    try:
        import tomllib
        from pathlib import Path

        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                config = tomllib.load(f)
            models = config.get("tool", {}).get("imas-codex", {}).get("models", {})
            return {
                "default": models.get("default", DEFAULT_MODEL),
                "discovery": models.get("discovery", DEFAULT_MODEL),
                "evaluation": models.get("evaluation", DEFAULT_MODEL),
                "enrichment": models.get("enrichment", DEFAULT_MODEL),
                "exploration": models.get("exploration", DEFAULT_MODEL),
                "score": models.get("score", DEFAULT_MODEL),
                "captioning": models.get("captioning", DEFAULT_MODEL),
                "presets": models.get("presets", {}),
            }
    except Exception:
        pass
    return {
        "default": DEFAULT_MODEL,
        "discovery": DEFAULT_MODEL,
        "evaluation": DEFAULT_MODEL,
        "enrichment": DEFAULT_MODEL,
        "exploration": DEFAULT_MODEL,
        "score": DEFAULT_MODEL,
        "captioning": DEFAULT_MODEL,
        "presets": {},
    }


# Load config at module import
_MODEL_CONFIG = _load_model_config()

# Model presets for convenience (used by get_model_id)
MODELS = _MODEL_CONFIG.get("presets", {})
if not MODELS:
    MODELS = {
        "gemini-flash": "google/gemini-3-flash-preview",
        "gemini-pro": "google/gemini-3-pro-preview",
        "claude-haiku": "anthropic/claude-haiku-4.5",
        "claude-sonnet": "anthropic/claude-sonnet-4.5",
        "claude-opus": "anthropic/claude-opus-4.5",
    }


def get_model_id(preset: str) -> str:
    """Get full model ID from a preset name or return as-is."""
    return MODELS.get(preset, preset)


def get_model_for_task(task: str) -> str:
    """Get the configured model for a task type."""
    return _MODEL_CONFIG.get(task, _MODEL_CONFIG["default"])


def create_litellm_model(
    model: str | None = None,
    task: str = "default",
    temperature: float = 0.3,
    max_tokens: int = 16384,
) -> LiteLLMModel:
    """Create a LiteLLM model configured for OpenRouter.

    Args:
        model: Model identifier (e.g., 'anthropic/claude-sonnet-4.5')
        task: Task type for model selection if model not specified
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response

    Returns:
        Configured LiteLLMModel instance
    """
    # Resolve model from task config if not specified
    if model is None:
        model = get_model_for_task(task)

    # OpenRouter requires 'openrouter/' prefix for LiteLLM
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    # Get API key from environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it in .env or export it."
        )

    return LiteLLMModel(
        model_id=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


@dataclass
class AgentConfig:
    """Configuration for a smolagents CodeAgent.

    Attributes:
        name: Agent identifier for logging and tracking
        instructions: Custom instructions inserted into system prompt
        tools: List of Tool instances the agent can use
        model: LLM model identifier (None = use task default)
        task: Task type for model selection ('enrichment', 'exploration', etc.)
        temperature: LLM sampling temperature
        max_tokens: Maximum tokens in LLM response
        max_steps: Maximum agent iterations
        cost_limit_usd: Budget limit in USD (None = no limit)
        planning_interval: Steps between planning phases (None = no planning)
        verbose: Enable verbose logging
        managed_agents: Sub-agents this agent can delegate to
    """

    name: str
    instructions: str = ""
    tools: list[Tool] = field(default_factory=list)
    model: str | None = None
    task: str = "default"
    temperature: float = 0.3
    max_tokens: int = 16384
    max_steps: int = 20
    cost_limit_usd: float | None = None
    planning_interval: int | None = None
    verbose: bool = False
    managed_agents: list[CodeAgent] = field(default_factory=list)


def create_agent(
    config: AgentConfig,
    step_callbacks: list[Callable] | None = None,
) -> CodeAgent:
    """Create a CodeAgent from configuration.

    The CodeAgent generates Python code to invoke tools, enabling:
    - Loops and conditionals for complex workflows
    - Variable reuse across tool calls
    - Self-debugging through code inspection
    - Adaptive problem-solving

    Args:
        config: AgentConfig with name, instructions, tools, etc.
        step_callbacks: Optional callbacks invoked after each step

    Returns:
        Configured CodeAgent instance

    Example:
        config = AgentConfig(
            name="explorer",
            instructions="Explore facility and discover code files",
            tools=get_exploration_tools(),
            task="exploration",
            cost_limit_usd=2.0,
        )
        agent = create_agent(config)
        result = agent.run("Find IMAS integration code at /work/codes")
    """
    # Create LLM model
    llm = create_litellm_model(
        model=config.model,
        task=config.task,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    # Build callbacks list
    callbacks = list(step_callbacks or [])

    # Add cost monitoring callback if budget limit set
    monitor = None
    if config.cost_limit_usd is not None:
        monitor = AgentMonitor(
            agent_name=config.name,
            cost_limit_usd=config.cost_limit_usd,
        )
        callbacks.append(create_step_callback(monitor))

    # Load system prompt from prompts directory if available
    system_prompt = _get_prompt(f"{config.name}-system")
    if not system_prompt and config.instructions:
        system_prompt = config.instructions

    # Create agent
    agent = CodeAgent(
        tools=config.tools,
        model=llm,
        instructions=system_prompt,
        max_steps=config.max_steps,
        planning_interval=config.planning_interval,
        step_callbacks=callbacks if callbacks else None,
        managed_agents=config.managed_agents if config.managed_agents else None,
        name=config.name,
    )

    # Attach monitor for external access
    if monitor:
        agent._monitor = monitor  # type: ignore[attr-defined]

    logger.info(
        f"Created CodeAgent '{config.name}' with {len(config.tools)} tools, "
        f"model={config.model or get_model_for_task(config.task)}"
    )

    return agent


def get_agent_monitor(agent: CodeAgent) -> AgentMonitor | None:
    """Get the cost monitor attached to an agent, if any."""
    return getattr(agent, "_monitor", None)


# =============================================================================
# Pre-configured Agent Factories
# =============================================================================


def get_enrichment_agent(
    verbose: bool = False,
    model: str | None = None,
    cost_limit_usd: float = 5.0,
) -> CodeAgent:
    """Create an agent configured for TreeNode metadata enrichment.

    This agent excels at:
    - Analyzing MDSplus path naming conventions
    - Cross-referencing with code examples
    - Generating physics-accurate descriptions
    - Batch processing with JSON output

    Args:
        verbose: Enable verbose output
        model: Override model (default: enrichment model from config)
        cost_limit_usd: Budget limit in USD

    Returns:
        CodeAgent configured for enrichment tasks
    """
    from imas_codex.agentic.tools import get_enrichment_tools

    config = AgentConfig(
        name="enrich",
        instructions=_get_prompt("discovery/enricher"),
        tools=get_enrichment_tools(),
        model=model,
        task="enrichment",
        max_steps=10,
        cost_limit_usd=cost_limit_usd,
        verbose=verbose,
    )
    return create_agent(config)


def get_scout_agent(
    facility: str,
    verbose: bool = False,
    model: str | None = None,
    cost_limit_usd: float = 5.0,
) -> CodeAgent:
    """Create an agent configured for facility scouting (exploration).

    This agent excels at:
    - Discovering source files via SSH
    - Navigating codebases efficiently
    - Identifying high-value code (IMAS, equilibrium, transport)
    - Documenting findings in the knowledge graph

    Args:
        facility: Facility ID to explore (e.g., "iter", "tcv")
        verbose: Enable verbose output
        model: Override model (default: exploration model from config)
        cost_limit_usd: Budget limit in USD

    Returns:
        CodeAgent configured for scouting tasks
    """
    from imas_codex.agentic.tools import get_exploration_tools

    tools = get_exploration_tools(facility)

    # Add facility context to instructions
    instructions = _get_prompt("exploration/facility")
    if not instructions:
        instructions = "You are an expert at exploring fusion facility codebases."
    instructions += f"\n\n## Current Session\nFacility: {facility}\n"

    config = AgentConfig(
        name="scout",
        instructions=instructions,
        tools=tools,
        model=model,
        task="exploration",
        max_steps=30,
        cost_limit_usd=cost_limit_usd,
        planning_interval=5,  # Re-plan every 5 steps
        verbose=verbose,
    )
    return create_agent(config)


# Backward compatibility alias (deprecated)
get_exploration_agent = get_scout_agent


# =============================================================================
# Multi-Agent Orchestration
# =============================================================================


def create_orchestrator(
    task: str,
    managed_agents: list[CodeAgent],
    instructions: str | None = None,
    cost_limit_usd: float = 10.0,
) -> CodeAgent:
    """Create an orchestrator agent that delegates to specialized agents.

    The orchestrator can invoke managed agents as tools, enabling:
    - Task decomposition across specialized agents
    - Parallel exploration of different paths
    - Aggregation of results from multiple agents

    Args:
        task: Task type for model selection
        managed_agents: List of specialized agents to delegate to
        instructions: Custom orchestration instructions
        cost_limit_usd: Budget limit for the orchestrator

    Returns:
        Orchestrator CodeAgent

    Example:
        enricher = get_enrichment_agent()
        scout = get_scout_agent("iter")

        orchestrator = create_orchestrator(
            task="enrichment",
            managed_agents=[enricher, scout],
            instructions="Explore facility then enrich discovered paths",
        )
        result = orchestrator.run("Process ITER facility")
    """
    if not instructions:
        instructions = (
            "You are an orchestrator agent. Delegate tasks to specialized agents "
            "based on their capabilities. Aggregate and synthesize their results."
        )

    config = AgentConfig(
        name="orchestrator",
        instructions=instructions,
        tools=[],  # Orchestrator uses managed agents, not direct tools
        task=task,
        max_steps=15,
        cost_limit_usd=cost_limit_usd,
        managed_agents=managed_agents,
    )
    return create_agent(config)
