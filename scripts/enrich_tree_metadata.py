#!/usr/bin/env python3
"""Enrich MDSplus TreeNode metadata using LLM inference.

Uses Gemini Flash via OpenRouter to generate descriptions, physics domains,
and discover IMAS mappings for TreeNodes.

Usage:
    uv run enrich-tree-metadata --facility epfl
    uv run enrich-tree-metadata --facility epfl --tree results --dry-run
    uv run enrich-tree-metadata --facility epfl --with-context-only
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click
from dotenv import load_dotenv

# Load .env from home directory or project
load_dotenv(Path.home() / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# Patterns for metadata/internal nodes to skip
SKIP_PATTERNS = {
    ":IGNORE",
    ":FOO",
    ":BAR",
    ":VERSION_NUM",
    ":COMMENT",
    ":ERROR_BAR",
    ":UNITS",
    ":CONFIDENCE",
    ":TRIAL",
    ":USER_NAME",
    ":TIME_INDEX",
    ":QUALITY",
}

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


@dataclass
class EnrichmentResult:
    """Result from LLM enrichment."""

    path: str
    description: str | None
    physics_domain: str | None
    confidence: str
    suggested_units: str | None = None
    source: str = "llm"


def should_skip_path(path: str) -> bool:
    """Check if path should be skipped (metadata/internal node)."""
    upper = path.upper()
    return any(pattern in upper for pattern in SKIP_PATTERNS)


def get_openrouter_client():
    """Create OpenAI client configured for OpenRouter."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    return OpenAI(api_key=api_key, base_url=base_url)


def ssh_query_node(facility: str, tree_name: str, path: str) -> dict | None:
    """Query MDSplus node metadata via SSH."""
    # Escape path for shell
    escaped_path = path.replace("\\", "\\\\").replace('"', '\\"')

    cmd = f"""python3 -c "
import MDSplus
try:
    tree = MDSplus.Tree('{tree_name}', 80000)
    node = tree.getNode('{escaped_path}')
    print('usage:', node.usage)
    print('description:', str(node.description) if node.description else '')
    print('units:', str(node.units) if node.units else '')
except Exception as e:
    print('error:', str(e))
"
"""
    try:
        result = subprocess.run(
            ["ssh", facility, cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            output = {}
            for line in result.stdout.strip().split("\n"):
                if ":" in line:
                    key, _, value = line.partition(":")
                    output[key.strip()] = value.strip()
            return output
    except Exception as e:
        logger.warning(f"SSH query failed for {path}: {e}")
    return None


def build_enrichment_prompt(
    tree_name: str,
    nodes: list[dict],
    code_context: dict[str, list[str]] | None = None,
) -> str:
    """Build the LLM prompt for batch enrichment."""
    prompt = f"""You are a tokamak physics expert enriching MDSplus TreeNode metadata for the TCV tokamak at EPFL.

For each path, analyze the naming convention and any provided code context to generate:
- description: 1-2 sentence physics description. Be DIRECT and DEFINITIVE - do NOT use hedging language like "likely", "probably", "may represent". State what the node IS, not what it "might be".
- physics_domain: One of: {", ".join(PHYSICS_DOMAINS)}
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
- RHO: Normalized toroidal flux coordinate (sqrt of normalized toroidal flux)
- _95 suffix: quantity at 95% normalized flux surface
- _AXIS suffix: quantity on magnetic axis

Tree: {tree_name}

Paths to describe (respond with JSON array):
"""

    for node in nodes:
        path = node["path"]
        prompt += f"\n- {path}"
        if node.get("units"):
            prompt += f" (units: {node['units']})"
        if code_context and path in code_context:
            snippets = code_context[path][:2]  # Max 2 snippets
            prompt += f"\n  Code context: {snippets[0][:200]}..."

    prompt += """

Respond with a JSON array only, no markdown. Be definitive in descriptions:
[{"path": "...", "description": "...", "physics_domain": "...", "confidence": "high|medium|low"}]
"""
    return prompt


async def enrich_batch(
    client,
    model: str,
    tree_name: str,
    nodes: list[dict],
    code_context: dict[str, list[str]] | None = None,
    facility: str | None = None,
) -> list[EnrichmentResult]:
    """Enrich a batch of nodes using LLM."""
    prompt = build_enrichment_prompt(tree_name, nodes, code_context)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )

        content = response.choices[0].message.content
        # Extract JSON from response
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        results = json.loads(content)

        enriched = []
        for item in results:
            enriched.append(
                EnrichmentResult(
                    path=item["path"],
                    description=item.get("description"),
                    physics_domain=item.get("physics_domain"),
                    confidence=item.get("confidence", "low"),
                    source="llm",
                )
            )
        return enriched

    except Exception as e:
        logger.error(f"Batch enrichment failed: {e}")
        return []


def get_nodes_to_enrich(
    client, tree_name: str | None = None, with_context_only: bool = False
) -> list[dict]:
    """Get TreeNodes needing enrichment from Neo4j."""
    from imas_codex.graph import GraphClient

    with GraphClient() as graph:
        # Build query
        where_clauses = [
            "t.first_shot IS NOT NULL",
            "(t.description IS NULL OR t.description = 'None')",
        ]
        if tree_name:
            where_clauses.append(f"t.tree_name = '{tree_name}'")

        query = f"""
            MATCH (t:TreeNode)
            WHERE {" AND ".join(where_clauses)}
            OPTIONAL MATCH (d:DataReference)-[:RESOLVES_TO_TREE_NODE]->(t)
            OPTIONAL MATCH (c:CodeChunk)-[:CONTAINS_REF]->(d)
            WITH t, collect(DISTINCT substring(c.content, 0, 300)) AS snippets
            RETURN t.path AS path, t.tree_name AS tree, t.units AS units,
                   size(snippets) > 0 AS has_context, snippets
            ORDER BY has_context DESC, t.path
        """

        result = graph.query(query)

        nodes = []
        for r in result:
            if should_skip_path(r["path"]):
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


def save_results(graph_client, results: list[EnrichmentResult], dry_run: bool = False):
    """Save enrichment results to Neo4j."""
    if dry_run:
        for r in results[:10]:
            logger.info(f"  {r.path}: {r.description} [{r.confidence}]")
        if len(results) > 10:
            logger.info(f"  ... and {len(results) - 10} more")
        return

    # Batch update
    updates = [
        {
            "path": r.path,
            "description": r.description,
            "physics_domain": r.physics_domain,
            "enrichment_confidence": r.confidence,
            "enrichment_source": r.source,
        }
        for r in results
        if r.description  # Only update if we got a description
    ]

    if updates:
        graph_client.query(
            """
            UNWIND $updates AS u
            MATCH (t:TreeNode {path: u.path})
            SET t.description = u.description,
                t.physics_domain = u.physics_domain,
                t.enrichment_confidence = u.enrichment_confidence,
                t.enrichment_source = u.enrichment_source
            """,
            updates=updates,
        )


@click.command()
@click.option("--facility", "-f", default="epfl", help="Facility SSH alias")
@click.option("--tree", "-t", help="Specific tree to enrich (default: all)")
@click.option(
    "--model",
    "-m",
    default="google/gemini-2.0-flash-001",
    help="LLM model to use",
)
@click.option("--batch-size", "-b", default=50, help="Nodes per LLM request")
@click.option("--dry-run", is_flag=True, help="Preview without saving")
@click.option(
    "--with-context-only",
    is_flag=True,
    help="Only enrich nodes with code context",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.option("--limit", "-n", type=int, help="Limit number of nodes to process")
def main(
    facility: str,
    tree: str | None,
    model: str,
    batch_size: int,
    dry_run: bool,
    with_context_only: bool,
    verbose: bool,
    limit: int | None,
):
    """Enrich MDSplus TreeNode metadata using LLM inference."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    click.echo("TreeNode Metadata Enrichment")
    click.echo(f"Model: {model}")
    click.echo(f"Facility: {facility}")
    if tree:
        click.echo(f"Tree: {tree}")
    click.echo()

    # Get nodes to enrich
    click.echo("Fetching nodes to enrich...")
    nodes = get_nodes_to_enrich(
        None, tree_name=tree, with_context_only=with_context_only
    )

    if limit:
        nodes = nodes[:limit]

    click.echo(f"Found {len(nodes)} nodes to enrich")
    with_context = sum(1 for n in nodes if n["has_context"])
    click.echo(f"  With code context: {with_context}")
    click.echo(f"  Path only: {len(nodes) - with_context}")

    if not nodes:
        click.echo("No nodes to enrich!")
        return 0

    # Estimate cost
    num_batches = (len(nodes) + batch_size - 1) // batch_size
    input_tokens = num_batches * (200 + batch_size * 50)
    output_tokens = num_batches * batch_size * 30
    cost = (input_tokens / 1_000_000) * 0.10 + (output_tokens / 1_000_000) * 0.40
    click.echo(f"\nEstimated cost: ${cost:.2f} ({num_batches} batches)")

    if dry_run:
        click.echo("\n[DRY RUN] Would process these nodes:")
        for n in nodes[:5]:
            ctx = " (has code context)" if n["has_context"] else ""
            click.echo(f"  {n['path']}{ctx}")
        if len(nodes) > 5:
            click.echo(f"  ... and {len(nodes) - 5} more")
        return 0

    # Initialize clients
    client = get_openrouter_client()

    from imas_codex.graph import GraphClient

    with GraphClient() as graph:
        # Process in batches
        all_results = []
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            batch_num = i // batch_size + 1
            click.echo(f"\nBatch {batch_num}/{num_batches}...")

            # Build code context for nodes that have it
            code_context = {}
            for n in batch:
                if n["snippets"]:
                    code_context[n["path"]] = n["snippets"]

            # Group by tree
            tree_name = batch[0]["tree"]

            # Call LLM
            results = asyncio.run(
                enrich_batch(
                    client,
                    model,
                    tree_name,
                    batch,
                    code_context,
                    facility,
                )
            )

            all_results.extend(results)

            # Show progress
            enriched = sum(1 for r in results if r.description)
            click.echo(f"  Enriched {enriched}/{len(batch)} nodes")

        # Save results
        click.echo(f"\nSaving {len(all_results)} results...")
        save_results(graph, all_results, dry_run=False)

        # Summary
        total_enriched = sum(1 for r in all_results if r.description)
        high_conf = sum(1 for r in all_results if r.confidence == "high")
        click.echo("\n=== Summary ===")
        click.echo(f"Total enriched: {total_enriched}/{len(nodes)}")
        click.echo(f"High confidence: {high_conf}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
