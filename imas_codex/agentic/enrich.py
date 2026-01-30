"""
TreeNode metadata enrichment agent.

Provides:
- EnrichmentResult: Result of enriching a single TreeNode
- batch_enrich_paths: Main enrichment function for CLI
- quick_task: Simple task runner for ad-hoc agent tasks
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from imas_codex.agentic.agents import (
    AgentConfig,
    create_agent,
    get_model_for_task,
)
from imas_codex.agentic.monitor import AgentMonitor, create_step_callback
from imas_codex.agentic.prompt_loader import load_prompts
from imas_codex.agentic.tools import (
    get_enrichment_tools,
    get_exploration_tools,
)
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


@dataclass
class BatchProgress:
    """Progress state for batch enrichment."""

    batch_num: int
    total_batches: int
    parent_path: str
    paths_processed: int
    paths_total: int
    enriched: int
    errors: int
    high_confidence: int
    elapsed_seconds: float


# Callback type for progress updates
ProgressCallback = Callable[[BatchProgress], None] | None


def get_physics_domain_values() -> list[str]:
    """Get all valid physics domain values from the enum."""
    return [d.value for d in PhysicsDomain]


def _parse_physics_domain(value: str | None) -> PhysicsDomain | None:
    """Parse physics domain string to enum."""
    if not value:
        return None
    try:
        return PhysicsDomain(value.lower())
    except ValueError:
        return None


def _parse_json_response(content: str) -> list[dict]:
    """Parse JSON array from agent response."""
    # Try to find JSON array in response
    import re

    # Try direct parse first
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find JSON array in text
    match = re.search(r"\[[\s\S]*\]", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Find individual JSON objects
    objects = []
    for match in re.finditer(r"\{[^{}]+\}", content):
        try:
            objects.append(json.loads(match.group()))
        except json.JSONDecodeError:
            continue

    return objects


def get_parent_path(path: str) -> str:
    """Get parent path for grouping related nodes."""
    parts = path.split(":")
    if len(parts) > 2:
        return ":".join(parts[:-1])
    return parts[0] if parts else path


def compose_batches(
    paths: list[str],
    batch_size: int = 100,
    group_by_parent: bool = True,
) -> list[list[str]]:
    """
    Compose paths into batches, optionally grouping by parent.

    Args:
        paths: List of paths to batch
        batch_size: Maximum paths per batch
        group_by_parent: If True, group related paths together

    Returns:
        List of batches
    """
    if not group_by_parent:
        return [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]

    # Group by parent
    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        parent = get_parent_path(path)
        groups[parent].append(path)

    # Compose batches from groups
    batches: list[list[str]] = []
    current_batch: list[str] = []

    for parent in sorted(groups.keys()):
        group_paths = groups[parent]

        # If group is larger than batch_size, split it
        if len(group_paths) > batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
            for i in range(0, len(group_paths), batch_size):
                batches.append(group_paths[i : i + batch_size])
        # If adding this group would exceed batch_size, start new batch
        elif len(current_batch) + len(group_paths) > batch_size:
            if current_batch:
                batches.append(current_batch)
            current_batch = group_paths.copy()
        else:
            current_batch.extend(group_paths)

    if current_batch:
        batches.append(current_batch)

    return batches


def _save_enrichments_to_graph(results: list[EnrichmentResult], dry_run: bool) -> int:
    """Save enrichment results to Neo4j graph."""
    if dry_run:
        return 0

    saved = 0
    with GraphClient() as gc:
        for result in results:
            if result.error or not result.description:
                continue

            props = {
                "description": result.description,
                "enrichment_status": "enriched",
                "enrichment_confidence": result.confidence,
            }

            if result.physics_domain:
                props["physics_domain"] = result.physics_domain.value

            if result.units:
                props["units"] = result.units

            if result.sign_convention:
                props["sign_convention"] = result.sign_convention

            if result.dimensions:
                props["dimensions"] = result.dimensions

            if result.error_node:
                props["error_node"] = result.error_node

            query = """
                MATCH (t:TreeNode {path: $path})
                SET t += $props
                RETURN t.path AS path
            """
            result_data = gc.query(query, {"path": result.path, "props": props})
            if result_data:
                saved += 1

    return saved


def discover_nodes_to_enrich(
    tree_name: str | None = None,
    status: str = "pending",
    with_context_only: bool = False,
    limit: int | None = None,
) -> list[dict]:
    """
    Discover TreeNodes that need enrichment.

    Args:
        tree_name: Filter to specific tree
        status: Target status (pending, enriched, stale, all)
        with_context_only: Only nodes with code context
        limit: Maximum nodes to return

    Returns:
        List of node dicts with path, tree, has_context
    """
    where_clauses = []

    if status != "all":
        where_clauses.append(
            f"(t.enrichment_status = '{status}' OR t.enrichment_status IS NULL)"
        )

    if tree_name:
        where_clauses.append(f"t.tree_name = '{tree_name}'")

    if with_context_only:
        where_clauses.append("exists((t)-[:APPEARS_IN]->(:CodeChunk))")

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        MATCH (t:TreeNode)
        {where}
        OPTIONAL MATCH (t)-[:APPEARS_IN]->(c:CodeChunk)
        WITH t, count(c) > 0 AS has_context
        RETURN t.path AS path, t.tree_name AS tree, has_context
        ORDER BY has_context DESC, t.path
        {limit_clause}
    """

    with GraphClient() as gc:
        return gc.query(query)


def estimate_enrichment_cost(
    num_paths: int,
    batch_size: int = 100,
    model: str = "google/gemini-3-pro-preview",
) -> dict:
    """
    Estimate cost and time for enrichment.

    Returns dict with:
    - estimated_cost: USD
    - estimated_hours: Time
    - paths_per_second: Expected throughput
    """
    num_batches = (num_paths + batch_size - 1) // batch_size

    # Benchmarked rates
    if "pro" in model.lower():
        paths_per_sec = 0.4
        cost_per_1k_paths = 0.50
    else:
        paths_per_sec = 2.5
        cost_per_1k_paths = 0.05

    estimated_time_sec = num_paths / paths_per_sec
    estimated_cost = (num_paths / 1000) * cost_per_1k_paths

    return {
        "estimated_cost": estimated_cost,
        "estimated_hours": estimated_time_sec / 3600,
        "paths_per_second": paths_per_sec,
        "num_batches": num_batches,
    }


async def quick_task(
    task: str,
    agent_type: str = "enrichment",
    verbose: bool = False,
    cost_limit_usd: float | None = None,
) -> str:
    """
    Run a quick agent task using smolagents CodeAgent.

    Args:
        task: The task description
        agent_type: One of 'enrichment', 'mapping', 'exploration'
        verbose: Enable verbose output
        cost_limit_usd: Optional cost budget

    Returns:
        Agent's response
    """
    # Get tools based on agent type
    if agent_type == "exploration":
        tools = get_exploration_tools()
        system_prompt = _get_prompt("exploration/facility")
    else:
        tools = get_enrichment_tools()
        system_prompt = _get_prompt("discovery/enricher")

    model = get_model_for_task(agent_type)

    config = AgentConfig(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        max_steps=15,
        verbose=verbose,
    )

    agent = create_agent(config)
    monitor = AgentMonitor(cost_limit_usd=cost_limit_usd)

    # Run the agent
    result = agent.run(task, step_callbacks=[create_step_callback(monitor)])

    logger.info(
        f"Task completed: {monitor.steps} steps, "
        f"${monitor.total_cost:.4f} cost, "
        f"{monitor.tool_calls} tool calls"
    )

    return str(result)


def quick_task_sync(
    task: str,
    agent_type: str = "enrichment",
    verbose: bool = False,
    cost_limit_usd: float | None = None,
) -> str:
    """Synchronous wrapper for quick_task."""
    return asyncio.run(quick_task(task, agent_type, verbose, cost_limit_usd))


async def _run_batch_enrichment(
    paths: list[str],
    tree_name: str,
    model: str,
    verbose: bool = False,
) -> list[EnrichmentResult]:
    """
    Run smolagents CodeAgent to enrich a batch of paths.

    The agent generates Python code to:
    1. Query the graph for existing metadata
    2. Search code examples for usage patterns
    3. Generate enrichments for all paths
    """
    tools = get_enrichment_tools()
    physics_domains = ", ".join(get_physics_domain_values())
    parent = get_parent_path(paths[0]) if paths else "unknown"
    paths_list = "\n".join(f"- {p}" for p in paths)

    system_prompt = f"""You are an enrichment agent for MDSplus TreeNodes.

Your task is to generate physics-accurate descriptions for TreeNode paths.
You have tools to query the knowledge graph and search code examples.

## Available Tools
- query_neo4j: Query the knowledge graph for node metadata
- search_code_examples: Find how paths are used in code
- search_imas_paths: Find related IMAS data dictionary paths

## Enrichment Guidelines
- Be DEFINITIVE in descriptions, not speculative
- Include physical meaning and typical use
- Identify correct SI units
- Assign appropriate physics domain

## Output Format
Output a JSON array with enrichments. Each object must have:
- "path": The exact MDSplus path
- "description": 1-2 sentence physics description
- "physics_domain": One of: {physics_domains}
- "units": SI units (e.g., "A", "Wb", "m^-3") or null
- "confidence": "high", "medium", or "low"

Optional fields: sign_convention, dimensions, error_node
"""

    task = f"""Enrich these {len(paths)} MDSplus paths from the {tree_name} tree.
Parent path: {parent}

Paths:
{paths_list}

Steps:
1. Query the graph: `query_neo4j("MATCH (t:TreeNode) WHERE t.path CONTAINS '{parent}' RETURN t.path, t.description LIMIT 20")`
2. Search code: `search_code_examples("{parent}")`
3. Generate enrichments as JSON array for ALL {len(paths)} paths

Output ONLY the JSON array."""

    config = AgentConfig(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        max_steps=10,
        verbose=verbose,
    )

    agent = create_agent(config)
    monitor = AgentMonitor()

    try:
        result = agent.run(task, step_callbacks=[create_step_callback(monitor)])
        response_text = str(result)

        # Parse JSON from response
        parsed = _parse_json_response(response_text)

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
            for i, item in enumerate(parsed)
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


async def batch_enrich_paths(
    paths: list[str],
    tree_name: str = "results",
    batch_size: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    progress_callback: ProgressCallback = None,
) -> list[EnrichmentResult]:
    """
    Enrich TreeNode paths using smolagents CodeAgent.

    Args:
        paths: List of MDSplus paths to enrich
        tree_name: Tree name for context
        batch_size: Paths per batch (auto-selected if None)
        verbose: Enable verbose output
        dry_run: If True, don't persist to graph
        model: LLM model to use
        progress_callback: Optional callback for progress updates

    Returns:
        List of EnrichmentResult for all paths
    """
    effective_model = model or get_model_for_task("enrichment")

    # Auto-select batch size based on model
    if batch_size is None:
        batch_size = 200 if "pro" in effective_model.lower() else 100

    start_time = time.perf_counter()
    batches = compose_batches(paths, batch_size=batch_size, group_by_parent=True)

    logger.info(
        f"Processing {len(paths)} paths in {len(batches)} batches "
        f"(avg {len(paths) / len(batches):.1f} paths/batch)"
    )

    all_results: list[EnrichmentResult] = []
    paths_processed = 0
    enriched_count = 0
    error_count = 0
    high_conf_count = 0

    for i, batch in enumerate(batches, 1):
        parent = get_parent_path(batch[0]) if batch else "unknown"
        logger.info(f"Batch {i}/{len(batches)}: {len(batch)} paths from {parent}")

        # Run enrichment with retry
        max_retries = 3
        results = []
        for attempt in range(max_retries):
            try:
                results = await _run_batch_enrichment(
                    batch, tree_name, effective_model, verbose
                )
                valid_results = sum(1 for r in results if not r.error)
                if valid_results > 0:
                    break
                logger.warning(f"Batch {i} attempt {attempt + 1}: no valid results")
            except Exception as e:
                logger.warning(f"Batch {i} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    results = [
                        EnrichmentResult(
                            path=p,
                            description=None,
                            physics_domain=None,
                            units=None,
                            confidence="low",
                            error=f"Failed after {max_retries} attempts",
                        )
                        for p in batch
                    ]

        # Save to graph
        if not dry_run:
            _save_enrichments_to_graph(results, dry_run=False)

        # Update counts
        paths_processed += len(batch)
        for r in results:
            if r.error:
                error_count += 1
            elif r.description:
                enriched_count += 1
                if r.confidence == "high":
                    high_conf_count += 1

        all_results.extend(results)

        # Progress callback
        if progress_callback:
            elapsed = time.perf_counter() - start_time
            progress_callback(
                BatchProgress(
                    batch_num=i,
                    total_batches=len(batches),
                    parent_path=parent,
                    paths_processed=paths_processed,
                    paths_total=len(paths),
                    enriched=enriched_count,
                    errors=error_count,
                    high_confidence=high_conf_count,
                    elapsed_seconds=elapsed,
                )
            )

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"Enrichment complete: {enriched_count} enriched, "
        f"{error_count} errors, {elapsed:.1f}s"
    )

    return all_results


def batch_enrich_paths_sync(
    paths: list[str],
    tree_name: str = "results",
    batch_size: int | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    model: str | None = None,
    progress_callback: ProgressCallback = None,
) -> list[EnrichmentResult]:
    """Synchronous wrapper for batch_enrich_paths."""
    return asyncio.run(
        batch_enrich_paths(
            paths, tree_name, batch_size, verbose, dry_run, model, progress_callback
        )
    )
