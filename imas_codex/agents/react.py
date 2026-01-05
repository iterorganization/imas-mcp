"""
LlamaIndex ReActAgent configurations for different exploration tasks.

Provides pre-configured agents for:
- Metadata enrichment
- IMAS mapping discovery
- Facility exploration
- Code analysis
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

from imas_codex.agents.llm import get_llm
from imas_codex.agents.tools import get_exploration_tools
from imas_codex.graph import GraphClient

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
    units: str | None
    confidence: str
    error: str | None = None
    elapsed_seconds: float = 0.0


# Physics domains for TreeNode enrichment
PHYSICS_DOMAINS = [
    "equilibrium",
    "magnetics",
    "heating",
    "diagnostics",
    "transport",
    "mhd",
    "control",
    "machine",
    "neutral_beam",
    "spectroscopy",
]


def _build_batch_prompt(
    tree_name: str,
    nodes: list[dict],
) -> str:
    """Build LLM prompt for batch enrichment."""
    prompt = f"""You are a tokamak physics expert enriching MDSplus TreeNode metadata for the TCV tokamak at EPFL.

For each path, analyze the naming convention and any provided code context to generate:
- description: 1-2 sentence physics description. Be DIRECT and DEFINITIVE - do NOT use hedging language like "likely", "probably", "may represent". State what the node IS, not what it "might be".
- physics_domain: One of: {", ".join(PHYSICS_DOMAINS)}
- units: SI units for this quantity (e.g., "A", "Wb", "m", "s", "eV", "m^-3"). Use null if dimensionless or unknown.
- confidence:
  - high = standard physics quantity with well-known abbreviation (I_P, PSI, Q, ne, Te, etc.)
  - medium = clear from context or naming pattern but not a standard abbreviation
  - low = uncertain, set description to null instead of guessing

If you cannot determine the meaning with confidence, set description to null rather than guessing.

TCV-specific knowledge:
- LIUQE: TCV's main equilibrium reconstruction code (both FORTRAN and MATLAB versions)
- ASTRA: 1.5D transport code for plasma simulations
- CXRS: Charge Exchange Recombination Spectroscopy diagnostic
- THOMSON: Thomson scattering diagnostic for Te/ne profiles
- FIR: Far-Infrared interferometer for line-integrated density
- BOLO: Bolometer arrays for radiated power
- GPI: Gas Puff Imaging diagnostic
- PROFFIT: Profile fitting analysis code
- IBS: Integrated Beam Simulation or Ion Beam System
- RHO: Normalized toroidal flux coordinate (sqrt of normalized toroidal flux)
- _95 suffix: quantity at 95% normalized flux surface
- _AXIS suffix: quantity on magnetic axis

Tree: {tree_name}

Paths to describe (respond with JSON array):
"""

    for node in nodes:
        path = node["path"]
        prompt += f"\n- {path}"
        if node.get("units") and node["units"] != "dimensionless":
            prompt += f" (current units: {node['units']})"
        if node.get("description") and node["description"] != "None":
            prompt += f" (current: {node['description'][:100]})"

    prompt += """

Respond with a JSON array only, no markdown. Be definitive in descriptions:
[{"path": "...", "description": "...", "physics_domain": "...", "units": "...", "confidence": "high|medium|low"}]
"""
    return prompt


def _parse_llm_response(content: str) -> list[dict]:
    """Parse LLM response, extracting JSON."""
    content = content.strip()
    # Remove markdown code blocks if present
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    return json.loads(content)


async def _call_llm_batch(
    model: str,
    tree_name: str,
    nodes: list[dict],
    temperature: float = 0.3,
) -> list[EnrichmentResult]:
    """Call LLM for batch enrichment."""
    from imas_codex.agents.llm import get_llm

    prompt = _build_batch_prompt(tree_name, nodes)

    llm = get_llm(model=model, temperature=temperature)
    response = await llm.acomplete(prompt)

    try:
        results = _parse_llm_response(response.text)
        return [
            EnrichmentResult(
                path=item["path"],
                description=item.get("description"),
                physics_domain=item.get("physics_domain"),
                units=item.get("units"),
                confidence=item.get("confidence", "low"),
            )
            for item in results
        ]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response: {e}")
        logger.debug(f"Response was: {response.text[:500]}")
        return [
            EnrichmentResult(
                path=n["path"],
                description=None,
                physics_domain=None,
                units=None,
                confidence="low",
                error=f"JSON parse error: {e}",
            )
            for n in nodes
        ]


def _save_enrichments_to_graph(results: list[EnrichmentResult]) -> int:
    """
    Save enrichment results to Neo4j graph.

    Returns number of nodes updated.
    """
    updates = [
        {
            "path": r.path,
            "description": r.description,
            "physics_domain": r.physics_domain,
            "units": r.units,
            "enrichment_confidence": r.confidence,
            "enrichment_source": "llm_agent",
        }
        for r in results
        if r.description  # Only update if we got a description
    ]

    if not updates:
        return 0

    with GraphClient() as gc:
        gc.query(
            """
            UNWIND $updates AS u
            MATCH (t:TreeNode {path: u.path})
            SET t.description = u.description,
                t.physics_domain = u.physics_domain,
                t.units = COALESCE(u.units, t.units),
                t.enrichment_confidence = u.enrichment_confidence,
                t.enrichment_source = u.enrichment_source
            """,
            updates=updates,
        )

    return len(updates)


def _get_nodes_for_enrichment(
    paths: list[str],
    tree_name: str,
) -> list[dict]:
    """Fetch current node state from graph for enrichment."""
    with GraphClient() as gc:
        result = gc.query(
            """
            UNWIND $paths AS path
            MATCH (t:TreeNode {path: path})
            RETURN t.path AS path, t.tree_name AS tree,
                   t.units AS units, t.description AS description
            """,
            paths=paths,
        )
        return [dict(r) for r in result]


async def batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    verbose: bool = False,
    batch_size: int = 20,
    model: str = "google/gemini-2.0-flash-001",
    dry_run: bool = False,
) -> list[EnrichmentResult]:
    """
    Batch enrich multiple TreeNode paths with LLM-generated metadata.

    This function:
    1. Fetches current node state from the graph
    2. Calls LLM in batches for efficient processing
    3. Persists enrichments back to the graph

    Args:
        paths: List of MDSplus paths to enrich
        tree_name: Tree name for context
        verbose: Enable verbose output
        batch_size: Paths per LLM request (default: 20)
        model: LLM model to use
        dry_run: If True, don't persist to graph

    Returns:
        List of EnrichmentResult for each path
    """
    start_time = time.perf_counter()

    # Get current node state
    nodes = _get_nodes_for_enrichment(paths, tree_name)
    if not nodes:
        logger.warning(f"No TreeNodes found for paths: {paths[:3]}...")
        # Create placeholder nodes for paths not in graph
        nodes = [
            {"path": p, "tree": tree_name, "units": None, "description": None}
            for p in paths
        ]

    if verbose:
        logger.info(f"Processing {len(nodes)} nodes in batches of {batch_size}")

    all_results: list[EnrichmentResult] = []

    # Process in batches
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        batch_start = time.perf_counter()

        if verbose:
            batch_num = i // batch_size + 1
            total_batches = (len(nodes) + batch_size - 1) // batch_size
            logger.info(f"Batch {batch_num}/{total_batches}...")

        results = await _call_llm_batch(model, tree_name, batch)

        batch_elapsed = time.perf_counter() - batch_start
        for r in results:
            r.elapsed_seconds = batch_elapsed / len(results)

        all_results.extend(results)

        # Persist after each batch (not in dry run)
        if not dry_run:
            updated = _save_enrichments_to_graph(results)
            if verbose:
                logger.info(f"  Persisted {updated} enrichments to graph")

    total_elapsed = time.perf_counter() - start_time

    # Summary stats
    enriched = sum(1 for r in all_results if r.description)
    high_conf = sum(1 for r in all_results if r.confidence == "high")
    errors = sum(1 for r in all_results if r.error)

    logger.info(
        f"Enrichment complete: {enriched}/{len(all_results)} enriched, "
        f"{high_conf} high confidence, {errors} errors, "
        f"{total_elapsed:.1f}s total"
    )

    return all_results
