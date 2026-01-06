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

Tool priority (use in this order):
1. query_neo4j - ALWAYS check graph first for existing metadata and sibling nodes
2. search_code_examples - Find real usage patterns with units/descriptions in code
3. get_tree_structure - Understand sibling nodes and hierarchy
4. ssh_mdsplus_query - LAST RESORT only if above insufficient (slow, ~5s per query)

IMPORTANT: The graph already contains rich context:
- Previously enriched TreeNodes with descriptions
- Code examples showing how paths are used
- Sibling nodes that reveal naming patterns
SSH queries are slow and should be a last resort.

When enriching a path:
1. Query the graph for this node and its siblings (same parent path)
2. Search code examples for usage patterns
3. Only use SSH if the above don't provide enough information
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


# =============================================================================
# TreeNode Enrichment
# =============================================================================


@dataclass
class EnrichmentResult:
    """Result of enriching a single TreeNode."""

    path: str
    description: str | None
    physics_domain: str | None
    units: str | None
    confidence: str
    # IMAS mapping preparation fields
    sign_convention: str | None = None
    coordinate_system: str | None = None
    array_structure: str | None = None
    normalization: str | None = None
    error_node: str | None = None
    data_quality: str | None = None
    # Metadata
    has_context: bool = False
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

# Patterns for metadata/internal nodes to skip during discovery
# These are matched as complete path segments (ending the path or followed by colon)
SKIP_PATTERNS = {
    "IGNORE",
    "FOO",
    "BAR",
    "VERSION_NUM",
    "COMMENT",
    "ERROR_BAR",
    "UNITS",
    "CONFIDENCE",
    "TRIAL",
    "USER_NAME",
    "TIME_INDEX",
    "QUALITY",
}


def _should_skip_path(path: str) -> bool:
    """Check if path should be skipped (metadata/internal node).

    Matches patterns as complete path segments, not substrings.
    E.g., "BAR" matches ":BAR" but not ":BARATRON".
    """
    upper = path.upper()
    for pattern in SKIP_PATTERNS:
        # Check if pattern appears as a complete segment
        # Pattern at end: path ends with :PATTERN
        if upper.endswith(":" + pattern):
            return True
        # Pattern in middle: path contains :PATTERN:
        if ":" + pattern + ":" in upper:
            return True
    return False


def _build_batch_prompt(
    tree_name: str,
    nodes: list[dict],
) -> str:
    """Build LLM prompt for batch enrichment."""
    prompt = f"""You are a tokamak physics expert enriching MDSplus TreeNode metadata for the TCV tokamak at EPFL.

For each path, analyze the naming convention and any provided code context to generate:

REQUIRED fields:
- description: 1-2 sentence physics description. Be DIRECT and DEFINITIVE.
- physics_domain: One of: {", ".join(PHYSICS_DOMAINS)}
- units: SI units (e.g., "A", "Wb", "m^-3"). Use null if dimensionless/unknown.
- confidence: high|medium|low

IMAS MAPPING fields (include when determinable):
- sign_convention: Sign/direction convention. CRITICAL for currents, fluxes, fields.
  Examples: "positive clockwise viewed from above (COCOS 11)", "positive outward"
- array_structure: Dimension semantics for arrays. E.g., "(time, rho)", "(R, Z, time)"
- error_node: Path to associated error bar node if known

Confidence levels:
- high = standard physics abbreviation (I_P, PSI, Q, ne, Te)
- medium = clear from context but not standard
- low = uncertain, set description to null

TCV-specific knowledge:
- LIUQE: Equilibrium reconstruction (COCOS 17: Bphi>0, Ip>0 counter-clockwise from above)
- ASTRA: 1.5D transport code
- CXRS: Charge Exchange Recombination Spectroscopy
- THOMSON: Thomson scattering (Te/ne profiles)
- FIR: Far-Infrared interferometer (line-integrated density)
- BOLO: Bolometer arrays (radiated power)
- RHO: Normalized toroidal flux coordinate
- _95 suffix: value at 95% flux surface
- _AXIS suffix: value on magnetic axis

Tree: {tree_name}

Paths to describe:
"""

    for node in nodes:
        path = node["path"]
        prompt += f"\n- {path}"
        if node.get("units") and node["units"] != "dimensionless":
            prompt += f" (current units: {node['units']})"
        if node.get("description") and node["description"] != "None":
            prompt += f" (current: {node['description'][:100]})"
        # Add code context if available
        if node.get("snippets"):
            snippets = node["snippets"][:2]  # Max 2 snippets
            prompt += f"\n  Code context: {snippets[0][:200]}..."

    prompt += """

Respond with JSON array only (no markdown):
[{"path": "...", "description": "...", "physics_domain": "...", "units": "...", "confidence": "...", "sign_convention": "...", "array_structure": "...", "error_node": "..."}]

Omit optional fields if not determinable. Be definitive in descriptions.
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
                sign_convention=item.get("sign_convention"),
                array_structure=item.get("array_structure"),
                error_node=item.get("error_node"),
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
    updates = []
    for r in results:
        if not r.description:
            continue
        update = {
            "path": r.path,
            "description": r.description,
            "physics_domain": r.physics_domain,
            "units": r.units,
            "enrichment_confidence": r.confidence,
            "enrichment_source": "llm_agent",
        }
        # Add optional IMAS mapping fields if present
        if r.sign_convention:
            update["sign_convention"] = r.sign_convention
        if r.array_structure:
            update["array_structure"] = r.array_structure
        if r.error_node:
            update["error_node"] = r.error_node
        updates.append(update)

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
                t.enrichment_source = u.enrichment_source,
                t.sign_convention = COALESCE(u.sign_convention, t.sign_convention),
                t.array_structure = COALESCE(u.array_structure, t.array_structure),
                t.error_node = COALESCE(u.error_node, t.error_node)
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


def discover_nodes_to_enrich(
    tree_name: str | None = None,
    with_context_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """
    Discover TreeNodes needing enrichment from Neo4j.

    Finds nodes that:
    - Have first_shot set (are valid epoch-aware nodes)
    - Have no description or description is "None"
    - Are not metadata/internal nodes (TRIAL, IGNORE, etc.)

    Args:
        tree_name: Filter to specific tree (e.g., "results")
        with_context_only: Only return nodes with code context
        limit: Maximum nodes to return

    Returns:
        List of node dicts with path, tree, units, has_context, snippets
    """
    with GraphClient() as gc:
        # Build WHERE clauses
        where_clauses = [
            "t.first_shot IS NOT NULL",
            '(t.description IS NULL OR t.description = "None")',
        ]
        if tree_name:
            where_clauses.append(f't.tree_name = "{tree_name}"')

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            OPTIONAL MATCH (d:DataReference)-[:RESOLVES_TO_TREE_NODE]->(t)
            OPTIONAL MATCH (c:CodeChunk)-[:CONTAINS_REF]->(d)
            WITH t, collect(DISTINCT substring(c.content, 0, 300)) AS snippets
            RETURN t.path AS path, t.tree_name AS tree, t.units AS units,
                   size(snippets) > 0 AS has_context, snippets
            ORDER BY has_context DESC, t.path
            {limit_clause}
        """

        result = gc.query(query)

        nodes = []
        for r in result:
            if _should_skip_path(r["path"]):
                continue
            if with_context_only and not r["has_context"]:
                continue
            nodes.append(
                {
                    "path": r["path"],
                    "tree": r["tree"],
                    "units": r["units"],
                    "has_context": r["has_context"],
                    "snippets": r["snippets"] or [],
                }
            )

        return nodes


def estimate_enrichment_cost(
    node_count: int,
    batch_size: int = 20,
) -> dict:
    """
    Estimate LLM cost for enrichment.

    Args:
        node_count: Number of nodes to enrich
        batch_size: Paths per LLM request

    Returns:
        Dict with num_batches, input_tokens, output_tokens, estimated_cost
    """
    num_batches = (node_count + batch_size - 1) // batch_size
    # Estimate ~200 tokens prompt overhead + 50 tokens per path
    input_tokens = num_batches * (200 + batch_size * 50)
    # Estimate ~30 output tokens per path
    output_tokens = num_batches * batch_size * 30
    # Gemini Flash pricing: $0.10/1M input, $0.40/1M output
    cost = (input_tokens / 1_000_000) * 0.10 + (output_tokens / 1_000_000) * 0.40

    return {
        "num_batches": num_batches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
    }


async def batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    verbose: bool = False,
    batch_size: int = 20,
    model: str = "google/gemini-3-flash-preview",
    dry_run: bool = False,
    context_map: dict[str, list[str]] | None = None,
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
        context_map: Optional dict mapping paths to code snippets

    Returns:
        List of EnrichmentResult for each path
    """
    start_time = time.perf_counter()
    context_map = context_map or {}

    # Get current node state
    nodes = _get_nodes_for_enrichment(paths, tree_name)
    if not nodes:
        logger.warning(f"No TreeNodes found for paths: {paths[:3]}...")
        # Create placeholder nodes for paths not in graph
        nodes = [
            {"path": p, "tree": tree_name, "units": None, "description": None}
            for p in paths
        ]

    # Merge context snippets into nodes
    for node in nodes:
        path = node["path"]
        if path in context_map:
            node["snippets"] = context_map[path]

    if verbose:
        with_ctx = sum(1 for n in nodes if n.get("snippets"))
        logger.info(
            f"Processing {len(nodes)} nodes ({with_ctx} with context) "
            f"in batches of {batch_size}"
        )

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
