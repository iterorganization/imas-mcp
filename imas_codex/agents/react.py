"""
LlamaIndex ReActAgent configurations for different exploration tasks.

Provides pre-configured agents for:
- Metadata enrichment
- IMAS mapping discovery
- Facility exploration
- Code analysis
"""

import asyncio
import logging
from dataclasses import dataclass, field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agents.llm import get_llm
from imas_codex.agents.tools import get_exploration_tools

logger = logging.getLogger(__name__)


# System prompts for different agent personalities
ENRICHMENT_SYSTEM_PROMPT = """You are a tokamak physics expert enriching MDSplus TreeNode metadata for the TCV tokamak at EPFL.

Your task is to analyze TreeNode paths and provide accurate physics descriptions.

Guidelines:
- Be DEFINITIVE in descriptions - avoid hedging language like "likely" or "probably"
- Use proper physics terminology and units
- Reference TCV-specific knowledge (LIUQE, ASTRA, CXRS, etc.)
- If uncertain, gather more information using tools before describing
- Set description to null rather than guessing

Available context:
- Neo4j graph with TreeNodes, code examples, and facility data
- SSH access to MDSplus database for real node metadata
- Code examples showing how paths are used
- IMAS Data Dictionary for standard physics definitions

When enriching a path:
1. Check existing graph data for context
2. Search code examples for usage patterns
3. Query MDSplus via SSH if needed
4. Synthesize into a concise, accurate description
"""

MAPPING_SYSTEM_PROMPT = """You are an expert in fusion data standards, specializing in mapping facility-specific data to IMAS (Integrated Modelling and Analysis Suite).

Your task is to discover semantic mappings between MDSplus TreeNodes and IMAS Data Dictionary paths.

Guidelines:
- Focus on physics equivalence, not just name similarity
- Consider units, coordinates, and array dimensions
- Note transformation requirements (unit conversions, coordinate systems)
- Distinguish between exact matches and approximate mappings
- Document confidence level and any assumptions

Mapping process:
1. Understand the TreeNode's physics meaning from graph/MDSplus
2. Search IMAS DD for semantically equivalent paths
3. Verify units and structure compatibility
4. Document the mapping with confidence and notes
"""

EXPLORATION_SYSTEM_PROMPT = """You are an expert at exploring fusion facility data systems and codebases.

Your task is to systematically discover and document data structures, analysis codes, and their relationships.

Guidelines:
- Be thorough but efficient - use batch operations when possible
- Document findings immediately in the knowledge graph
- Prioritize high-value physics domains (equilibrium, profiles, transport)
- Note connections between codes, diagnostics, and data paths
- Flag interesting patterns for deeper investigation

Exploration approach:
1. Start with known entry points (tree names, code directories)
2. Use SSH tools for discovery (rg, fd, ls)
3. Cross-reference with existing graph data
4. Identify patterns and relationships
5. Queue files for ingestion when appropriate
"""


@dataclass
class AgentConfig:
    """Configuration for a LlamaIndex ReActAgent."""

    name: str
    system_prompt: str
    tools: list[FunctionTool] = field(default_factory=list)
    model: str = "google/gemini-3-flash-preview"
    temperature: float = 0.3
    verbose: bool = False
    max_iterations: int = 10


def create_agent(config: AgentConfig) -> ReActAgent:
    """
    Create a ReActAgent from configuration.

    Args:
        config: AgentConfig with name, system prompt, tools, etc.

    Returns:
        Configured ReActAgent instance
    """
    llm = get_llm(model=config.model, temperature=config.temperature)

    tools = config.tools or get_exploration_tools()

    agent = ReActAgent(
        name=config.name,
        description=config.system_prompt[:200],  # Short description
        system_prompt=config.system_prompt,
        tools=tools,
        llm=llm,
        verbose=config.verbose,
    )

    logger.info(f"Created agent '{config.name}' with {len(tools)} tools")
    return agent


def get_enrichment_agent(verbose: bool = False) -> ReActAgent:
    """
    Get an agent configured for TreeNode metadata enrichment.

    This agent excels at:
    - Analyzing MDSplus path naming conventions
    - Cross-referencing with code examples
    - Querying MDSplus for metadata
    - Generating physics-accurate descriptions

    Args:
        verbose: Enable verbose reasoning output

    Returns:
        ReActAgent configured for enrichment tasks
    """
    config = AgentConfig(
        name="enrichment-agent",
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        verbose=verbose,
    )
    return create_agent(config)


def get_mapping_agent(verbose: bool = False) -> ReActAgent:
    """
    Get an agent configured for IMAS mapping discovery.

    This agent excels at:
    - Understanding TreeNode physics semantics
    - Searching IMAS DD for equivalents
    - Analyzing unit and coordinate compatibility
    - Documenting mappings with confidence

    Args:
        verbose: Enable verbose reasoning output

    Returns:
        ReActAgent configured for mapping tasks
    """
    config = AgentConfig(
        name="mapping-agent",
        system_prompt=MAPPING_SYSTEM_PROMPT,
        verbose=verbose,
    )
    return create_agent(config)


def get_exploration_agent(verbose: bool = False) -> ReActAgent:
    """
    Get an agent configured for facility exploration.

    This agent excels at:
    - Discovering data structures via SSH
    - Navigating codebases efficiently
    - Identifying high-value targets
    - Documenting findings in the graph

    Args:
        verbose: Enable verbose reasoning output

    Returns:
        ReActAgent configured for exploration tasks
    """
    config = AgentConfig(
        name="exploration-agent",
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        verbose=verbose,
    )
    return create_agent(config)


async def run_agent(
    agent: ReActAgent,
    message: str,
    stream_events: bool = False,
) -> str:
    """
    Run an agent with a message and return the response.

    Args:
        agent: The ReActAgent to run
        message: User message/task
        stream_events: If True, print events as they occur

    Returns:
        Agent's final response as string
    """
    handler = agent.run(user_msg=message)

    if stream_events:
        async for event in handler.stream_events():
            if hasattr(event, "tool_name"):
                tool_name = event.tool_name
                tool_kwargs = getattr(event, "tool_kwargs", {})
                logger.info(f"[TOOL] {tool_name}: {tool_kwargs}")
            elif hasattr(event, "tool_output"):
                output = str(event.tool_output)[:200]
                logger.info(f"[RESULT] {output}...")

    result = await handler
    return str(result)


def run_agent_sync(
    agent: ReActAgent,
    message: str,
    stream_events: bool = False,
) -> str:
    """
    Synchronous wrapper for run_agent.

    Args:
        agent: The ReActAgent to run
        message: User message/task
        stream_events: If True, print events as they occur

    Returns:
        Agent's final response as string
    """
    return asyncio.run(run_agent(agent, message, stream_events))


# Convenience function for quick agent tasks
async def quick_agent_task(
    task: str,
    agent_type: str = "enrichment",
    verbose: bool = False,
) -> str:
    """
    Run a quick agent task without manual agent setup.

    Args:
        task: The task description
        agent_type: One of 'enrichment', 'mapping', 'exploration'
        verbose: Enable verbose output

    Returns:
        Agent's response

    Example:
        result = await quick_agent_task(
            "Describe what \\RESULTS::ASTRA is used for",
            agent_type="enrichment"
        )
    """
    agent_factories = {
        "enrichment": get_enrichment_agent,
        "mapping": get_mapping_agent,
        "exploration": get_exploration_agent,
    }

    if agent_type not in agent_factories:
        msg = f"Unknown agent type: {agent_type}. Choose from {list(agent_factories.keys())}"
        raise ValueError(msg)

    agent = agent_factories[agent_type](verbose=verbose)
    return await run_agent(agent, task, stream_events=verbose)


# Batch processing utilities
@dataclass
class EnrichmentResult:
    """Result of enriching a single TreeNode."""

    path: str
    description: str | None
    physics_domain: str | None
    confidence: str
    error: str | None = None


async def batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    verbose: bool = False,
) -> list[EnrichmentResult]:
    """
    Batch enrich multiple TreeNode paths using the enrichment agent.

    Args:
        paths: List of MDSplus paths to enrich
        tree_name: Tree name for context
        verbose: Enable verbose output

    Returns:
        List of EnrichmentResult for each path
    """
    agent = get_enrichment_agent(verbose=verbose)
    results = []

    for path in paths:
        try:
            response = await run_agent(
                agent,
                f"Analyze the TreeNode path {path} in the {tree_name} tree. "
                f"Provide: 1) A physics description, 2) Physics domain, 3) Confidence level. "
                f"Use available tools to gather context before answering.",
            )

            # Parse response (agent should format consistently)
            results.append(
                EnrichmentResult(
                    path=path,
                    description=response,  # Full response as description
                    physics_domain=None,  # Would need structured output
                    confidence="medium",
                )
            )
        except Exception as e:
            logger.exception(f"Failed to enrich {path}")
            results.append(
                EnrichmentResult(
                    path=path,
                    description=None,
                    physics_domain=None,
                    confidence="low",
                    error=str(e),
                )
            )

    return results
