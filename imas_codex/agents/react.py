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
from imas_codex.agents.prompt_loader import load_prompts
from imas_codex.agents.tools import get_exploration_tools
from imas_codex.core.physics_domain import PhysicsDomain
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Load prompts from markdown files
_PROMPTS = load_prompts()


def _get_prompt(name: str) -> str:
    """Get a prompt by name, with fallback to empty string."""
    prompt = _PROMPTS.get(name)
    if prompt is None:
        logger.warning(f"Prompt '{name}' not found in prompts directory")
        return ""
    return prompt.content


def get_physics_domain_values() -> list[str]:
    """Get all valid physics domain values from the enum."""
    return [d.value for d in PhysicsDomain]


# System prompts loaded from prompts directory
ENRICHMENT_SYSTEM_PROMPT = _get_prompt("enrichment-system")
MAPPING_SYSTEM_PROMPT = _get_prompt("mapping-system")
EXPLORATION_SYSTEM_PROMPT = _get_prompt("exploration-system")


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
        tools=tools,
        llm=llm,
        verbose=config.verbose,
        system_prompt=config.system_prompt,
        max_iterations=config.max_iterations,
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
    physics_domain: PhysicsDomain | None
    units: str | None
    confidence: str
    # IMAS mapping preparation fields
    sign_convention: str | None = None
    dimensions: list[str] | None = None
    error_node: str | None = None
    # Metadata
    has_context: bool = False
    error: str | None = None
    elapsed_seconds: float = 0.0


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
    """Build LLM prompt for batch enrichment using the prompt template."""
    # Build paths section
    paths_lines = []
    for node in nodes:
        path = node["path"]
        line = f"- {path}"
        if node.get("units") and node["units"] != "dimensionless":
            line += f" (current units: {node['units']})"
        if node.get("description") and node["description"] != "None":
            line += f" (current: {node['description'][:100]})"
        # Add code context if available
        if node.get("snippets"):
            snippets = node["snippets"][:2]  # Max 2 snippets
            line += f"\n  Code context: {snippets[0][:200]}..."
        paths_lines.append(line)

    paths_section = "\n".join(paths_lines)

    # Get prompt template and fill in placeholders
    template = _get_prompt("enrichment-batch")
    if not template:
        raise ValueError("enrichment-batch prompt not found")

    return template.format(
        physics_domains=", ".join(get_physics_domain_values()),
        tree_name=tree_name,
        paths_section=paths_section,
    )


def _parse_llm_response(content: str) -> list[dict]:
    """Parse LLM response, extracting JSON."""
    content = content.strip()
    
    # Try to find JSON array block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if match:
        content = match.group(1)
    elif content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: try to find any list structure
        match = re.search(r"(\[.*\])", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


def _parse_physics_domain(value: str | None) -> PhysicsDomain | None:
    """Parse physics domain string to enum, with fallback to GENERAL."""
    if not value:
        return None
    try:
        return PhysicsDomain(value)
    except ValueError:
        logger.warning(f"Unknown physics domain '{value}', falling back to GENERAL")
        return PhysicsDomain.GENERAL


async def _call_llm_batch(
    model: str,
    tree_name: str,
    nodes: list[dict],
    temperature: float = 0.3,
) -> list[EnrichmentResult]:
    """Call LLM for batch enrichment.

    Uses higher max_tokens (16384) to accommodate large batch responses.
    Each path generates ~100-150 output tokens in JSON format.
    """
    from imas_codex.agents.llm import get_llm

    prompt = _build_batch_prompt(tree_name, nodes)

    # Calculate max_tokens based on batch size (~150 tokens per path)
    estimated_output_tokens = len(nodes) * 150
    max_tokens = max(4096, min(estimated_output_tokens + 500, 65536))

    llm = get_llm(model=model, temperature=temperature, max_tokens=max_tokens)
    response = await llm.acomplete(prompt)

    try:
        results = _parse_llm_response(response.text)
        return [
            EnrichmentResult(
                path=item["path"],
                description=item.get("description"),
                physics_domain=_parse_physics_domain(item.get("physics_domain")),
                units=item.get("units"),
                confidence=item.get("confidence", "low"),
                sign_convention=item.get("sign_convention"),
                dimensions=item.get("dimensions"),
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

    Creates:
    - TreeNode property updates (description, physics_domain, etc.)
    - HAS_UNIT relationships to Unit nodes (creating Unit if needed)
    - HAS_ERROR relationships to error TreeNodes (if they exist)

    Returns number of nodes updated.
    """
    updates = []
    unit_links = []
    error_links = []

    for r in results:
        if not r.description:
            continue
        update = {
            "path": r.path,
            "description": r.description,
            "physics_domain": r.physics_domain.value if r.physics_domain else None,
            "enrichment_confidence": r.confidence,
            "enrichment_source": "llm_agent",
        }
        # Add optional IMAS mapping fields if present
        if r.sign_convention:
            update["sign_convention"] = r.sign_convention
        if r.dimensions:
            update["dimensions"] = r.dimensions
        updates.append(update)

        # Track unit relationships
        if r.units:
            unit_links.append({"path": r.path, "unit_symbol": r.units})

        # Track error node relationships
        if r.error_node:
            error_links.append({"path": r.path, "error_path": r.error_node})

    if not updates:
        return 0

    with GraphClient() as gc:
        # Update TreeNode properties
        gc.query(
            """
            UNWIND $updates AS u
            MATCH (t:TreeNode {path: u.path})
            SET t.description = u.description,
                t.physics_domain = u.physics_domain,
                t.enrichment_confidence = u.enrichment_confidence,
                t.enrichment_source = u.enrichment_source,
                t.sign_convention = COALESCE(u.sign_convention, t.sign_convention),
                t.dimensions = COALESCE(u.dimensions, t.dimensions)
            """,
            updates=updates,
        )

        # Create/link Unit nodes
        if unit_links:
            gc.query(
                """
                UNWIND $links AS link
                MATCH (t:TreeNode {path: link.path})
                MERGE (u:Unit {symbol: link.unit_symbol})
                MERGE (t)-[:HAS_UNIT]->(u)
                """,
                links=unit_links,
            )

        # Create HAS_ERROR relationships to existing error nodes
        if error_links:
            gc.query(
                """
                UNWIND $links AS link
                MATCH (t:TreeNode {path: link.path})
                MATCH (e:TreeNode {path: link.error_path})
                MERGE (t)-[:HAS_ERROR]->(e)
                """,
                links=error_links,
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
    batch_size: int = 200,
) -> dict:
    """
    Estimate LLM cost and time for enrichment.

    Args:
        node_count: Number of nodes to enrich
        batch_size: Paths per LLM request (default: 200, optimal for Gemini 3 Flash)

    Returns:
        Dict with num_batches, input_tokens, output_tokens, estimated_cost, estimated_hours
    """
    num_batches = (node_count + batch_size - 1) // batch_size
    # Estimate ~12 tokens per path input (from benchmark)
    input_tokens = num_batches * (500 + batch_size * 12)
    # Estimate ~150 output tokens per path (from benchmark)
    output_tokens = num_batches * batch_size * 150
    # Gemini Flash pricing: $0.10/1M input, $0.40/1M output
    cost = (input_tokens / 1_000_000) * 0.10 + (output_tokens / 1_000_000) * 0.40
    # Time estimate: ~53s per batch of 200 (from benchmark)
    seconds_per_batch = 53 * (batch_size / 200)
    estimated_hours = (num_batches * seconds_per_batch) / 3600

    return {
        "num_batches": num_batches,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
        "estimated_hours": estimated_hours,
    }


async def batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    verbose: bool = False,
    batch_size: int = 200,
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

    Batch size of 200 is optimal for Gemini 3 Flash:
    - 100% success rate (no JSON truncation)
    - ~3.8 paths/sec throughput
    - ~12 hours for 167k nodes

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


# =============================================================================
# Smart Batch Composition
# =============================================================================


def _get_parent_path(path: str) -> str:
    """Extract parent path from MDSplus path.

    Examples:
        \\RESULTS::LIUQE:PSI -> \\RESULTS::LIUQE
        \\RESULTS::THOMSON:NE:ERROR_BAR -> \\RESULTS::THOMSON:NE
        \\RESULTS::TOP -> \\RESULTS
    """
    # Handle :: separator (tree::subtree)
    if "::" in path:
        tree_part, node_part = path.split("::", 1)
        if ":" in node_part:
            # Has sub-nodes, get parent
            parent_node = ":".join(node_part.split(":")[:-1])
            return f"{tree_part}::{parent_node}" if parent_node else tree_part
        return tree_part
    return path


def compose_batches(
    paths: list[str],
    batch_size: int = 50,
    group_by_parent: bool = True,
) -> list[list[str]]:
    """
    Compose smart batches of related paths for efficient enrichment.

    Groups paths by parent to maximize context sharing. Sibling paths
    (same parent) are processed together so one context query serves all.

    Args:
        paths: List of MDSplus paths to batch
        batch_size: Maximum paths per batch (default: 50 for ReAct)
        group_by_parent: If True, group siblings together

    Returns:
        List of batches, each batch is a list of related paths
    """
    if not group_by_parent:
        # Simple chunking without grouping
        return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    # Group paths by parent
    parent_groups: dict[str, list[str]] = {}
    for path in paths:
        parent = _get_parent_path(path)
        if parent not in parent_groups:
            parent_groups[parent] = []
        parent_groups[parent].append(path)

    # Build batches respecting parent groups
    batches: list[list[str]] = []
    current_batch: list[str] = []

    # Sort parents by group size (process larger groups first for efficiency)
    sorted_parents = sorted(parent_groups.keys(), key=lambda p: -len(parent_groups[p]))

    for parent in sorted_parents:
        group = parent_groups[parent]

        if len(group) > batch_size:
            # Large group: split into multiple batches
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            for i in range(0, len(group), batch_size):
                batches.append(group[i : i + batch_size])
        elif len(current_batch) + len(group) <= batch_size:
            # Fits in current batch
            current_batch.extend(group)
        else:
            # Start new batch
            if current_batch:
                batches.append(current_batch)
            current_batch = group.copy()

    if current_batch:
        batches.append(current_batch)

    return batches


# =============================================================================
# ReAct Batch Enrichment
# =============================================================================


async def _run_react_batch_enrichment(
    agent: ReActAgent,
    paths: list[str],
    tree_name: str,
) -> list[EnrichmentResult]:
    """
    Run ReAct agent to enrich a batch of related paths.

    The agent will:
    1. Use tools to gather context for the batch
    2. Generate enrichments for all paths

    Args:
        agent: Configured ReActAgent with tools
        paths: Batch of related paths to enrich
        tree_name: Tree name for context

    Returns:
        List of EnrichmentResult for each path
    """
    # Build the task prompt
    paths_list = "\n".join(f"- {p}" for p in paths)
    parent = _get_parent_path(paths[0]) if paths else "unknown"

    task = f"""Enrich the following {len(paths)} MDSplus TreeNode paths from the {tree_name} tree.

These paths share the parent: {parent}

Paths to enrich:
{paths_list}

## Instructions

1. First, gather context using tools:
   - Query the graph for existing metadata on these paths and their siblings
   - Search code examples for usage patterns (try searching for "{parent}" or key terms)
   - Only use SSH if graph/code search is insufficient

2. Then, output a JSON array with enrichments for ALL {len(paths)} paths.

Remember: Be DEFINITIVE in descriptions. Use proper physics terminology.
Output ONLY the JSON array, no other text.
"""

    try:
        response = await agent.run(task)
        response_text = str(response)

        # Parse JSON from response
        results = _parse_llm_response(response_text)

        return [
            EnrichmentResult(
                path=item.get("path", paths[i] if i < len(paths) else "unknown"),
                description=item.get("description"),
                physics_domain=_parse_physics_domain(item.get("physics_domain")),
                units=item.get("units"),
                confidence=item.get("confidence", "low"),
                sign_convention=item.get("sign_convention"),
                dimensions=item.get("dimensions"),
                error_node=item.get("error_node"),
            )
            for i, item in enumerate(results)
        ]
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse agent response: {e}")
        return [
            EnrichmentResult(
                path=p,
                description=None,
                physics_domain=None,
                units=None,
                confidence="low",
                error=f"JSON parse error: {e}",
            )
            for p in paths
        ]
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return [
            EnrichmentResult(
                path=p,
                description=None,
                physics_domain=None,
                units=None,
                confidence="low",
                error=str(e),
            )
            for p in paths
        ]


def get_batch_enrichment_agent(
    verbose: bool = False,
    model: str = "google/gemini-3-flash-preview",
) -> ReActAgent:
    """
    Create a ReActAgent configured for batch enrichment.

    Uses tools for context gathering but expects batch JSON output.

    Args:
        verbose: Enable verbose agent output
        model: LLM model to use

    Returns:
        Configured ReActAgent
    """
    from imas_codex.agents.tools import get_enrichment_tools

    # Get system prompt and fill in physics domains
    system_prompt = _get_prompt("enrichment-react-batch")
    if system_prompt:
        system_prompt = system_prompt.format(
            physics_domains=", ".join(get_physics_domain_values())
        )
    else:
        # Fallback if prompt file not found
        system_prompt = ENRICHMENT_SYSTEM_PROMPT

    llm = get_llm(model=model, temperature=0.3, max_tokens=16384)
    tools = get_enrichment_tools()

    return ReActAgent(
        tools=tools,
        llm=llm,
        verbose=verbose,
        system_prompt=system_prompt,
        max_iterations=10,
    )


async def react_batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    batch_size: int = 200,  # Updated default based on benchmark (optimal throughput)
    verbose: bool = False,
    dry_run: bool = False,
    model: str = "google/gemini-3-flash-preview",
) -> list[EnrichmentResult]:
    """
    Enrich TreeNode paths using ReAct agent with smart batching.

    This function:
    1. Composes smart batches (groups related paths by parent)
    2. For each batch, runs ReAct agent to gather context and enrich
    3. Persists enrichments to graph

    Benchmark Results (Gemini 3 Flash):
    - Batch 10: 0.7 paths/sec
    - Batch 50: 1.1 paths/sec
    - Batch 200: 1.8 paths/sec (Optimal)
    
    Compared to direct LLM batch (3.8 paths/sec), ReAct is slower but
    provides much higher quality by gathering context from:
    - Knowledge Graph (sibling nodes)
    - Code Usage (how the path is actually used)
    - MDSplus (live metadata)

    Args:
        paths: List of MDSplus paths to enrich
        tree_name: Tree name for context
        batch_size: Paths per ReAct batch (default: 200)
        verbose: Enable verbose output
        dry_run: If True, don't persist to graph
        model: LLM model to use

    Returns:
        List of EnrichmentResult for all paths
    """
    start_time = time.perf_counter()

    # Compose smart batches
    batches = compose_batches(paths, batch_size=batch_size, group_by_parent=True)

    logger.info(
        f"Processing {len(paths)} paths in {len(batches)} smart batches "
        f"(avg {len(paths)/len(batches):.1f} paths/batch)"
    )

    # Create agent once, reuse for all batches
    agent = get_batch_enrichment_agent(verbose=verbose, model=model)

    all_results: list[EnrichmentResult] = []

    for i, batch in enumerate(batches, 1):
        batch_start = time.perf_counter()
        parent = _get_parent_path(batch[0]) if batch else "unknown"

        logger.info(f"Batch {i}/{len(batches)}: {len(batch)} paths from {parent}")

        # Retry logic for network issues
        max_retries = 3
        results = []
        for attempt in range(max_retries):
            try:
                results = await _run_react_batch_enrichment(agent, batch, tree_name)
                # Check for empty/error results that aren't parse errors
                valid_results = sum(1 for r in results if not r.error)
                if valid_results > 0 or not results:
                     break
                logger.warning(f"Batch {i} attempt {attempt+1} returned no valid results, retrying...")
            except Exception as e:
                logger.warning(f"Batch {i} attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Batch {i} failed after {max_retries} attempts")
                    # Return error results for this batch
                    results = [
                        EnrichmentResult(
                            path=p,
                            description=None,
                            physics_domain=None,
                            units=None,
                            confidence="low",
                            error=f"Batch failed: {str(e)}",
                        )
                        for p in batch
                    ]

        batch_elapsed = time.perf_counter() - batch_start
        for r in results:
            r.elapsed_seconds = batch_elapsed / len(results) if results else 0

        all_results.extend(results)

        # Persist after each batch
        if not dry_run and results:
            updated = _save_enrichments_to_graph(results)
            logger.info(f"  Persisted {updated} enrichments ({batch_elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - start_time

    # Summary
    enriched = sum(1 for r in all_results if r.description)
    high_conf = sum(1 for r in all_results if r.confidence == "high")
    errors = sum(1 for r in all_results if r.error)

    logger.info(
        f"ReAct enrichment complete: {enriched}/{len(all_results)} enriched, "
        f"{high_conf} high confidence, {errors} errors, "
        f"{total_elapsed:.1f}s total ({len(all_results)/total_elapsed:.1f} paths/sec)"
    )

    return all_results
