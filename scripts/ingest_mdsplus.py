#!/usr/bin/env python3
"""
Ingest MDSplus tree structure from a remote facility into the knowledge graph.

This script connects to a facility via SSH, introspects MDSplus trees using
a reference shot, and batch-ingests TreeNode data into Neo4j.

For large trees (>1000 nodes), this script is more efficient than LLM-driven
ingestion because it:
1. Runs the introspection in a single SSH session
2. Batches Neo4j writes with UNWIND
3. Automatically computes parent_path for hierarchy traversal

Usage:
    # Ingest a single tree
    uv run ingest-mdsplus epfl results --shot 84469

    # Ingest multiple trees
    uv run ingest-mdsplus epfl results tcv_shot magnetics --shot 84469

    # Dry run to see what would be ingested
    uv run ingest-mdsplus epfl results --shot 84469 --dry-run

    # Limit number of nodes (for testing)
    uv run ingest-mdsplus epfl results --shot 84469 --limit 100
"""

import json
import logging
import subprocess
import sys

import click

from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def compute_parent_path(path: str) -> str | None:
    """Compute parent path from MDSplus node path.

    MDSplus hierarchy separators:
    - :: separates tree name from path
    - . separates nodes within the tree
    - : separates variants at same level

    Examples:
        \\RESULTS::LIUQE:PSI -> \\RESULTS::LIUQE
        \\RESULTS::LIUQE -> \\RESULTS
        \\MAGNETICS::TOP.HARDWARE.CALIB -> \\MAGNETICS::TOP.HARDWARE
        \\MAGNETICS::TOP.HARDWARE -> \\MAGNETICS::TOP
        \\MAGNETICS::TOP -> \\MAGNETICS
        \\MAGNETICS -> None (root)
    """
    # First try : separator (for variants like LIUQE:PSI)
    if ":" in path.split("::")[-1] if "::" in path else ":" in path:
        # Has : after :: - use : as separator
        parts = path.rsplit(":", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]

    # Try . separator (for hierarchy like TOP.HARDWARE)
    if "." in path:
        parts = path.rsplit(".", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]

    # Try :: separator (for tree root like \\MAGNETICS::TOP)
    if "::" in path:
        parts = path.split("::", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]

    return None


def introspect_tree_ssh(
    facility: str,
    tree_name: str,
    shot: int,
    limit: int | None = None,
) -> list[dict]:
    """Introspect MDSplus tree via SSH and return node data.

    Runs a Python script on the remote facility that:
    1. Opens the tree at the given shot
    2. Enumerates all nodes with getNodeWild('***')
    3. Extracts path, node_type, units, description for each
    4. Returns JSON array of node dicts

    Args:
        facility: SSH host alias (e.g., "epfl")
        tree_name: MDSplus tree name (e.g., "results")
        shot: Shot number to use for introspection
        limit: Optional limit on number of nodes to fetch

    Returns:
        List of node dicts with keys: path, node_type, units, description
    """
    limit_clause = f"[:{limit}]" if limit else ""

    # Python code to run on remote facility
    remote_script = f'''
import json
import MDSplus

tree = MDSplus.Tree("{tree_name}", {shot})
nodes = list(tree.getNodeWild("***")){limit_clause}

result = []
for node in nodes:
    try:
        usage = str(node.usage)
        # Map MDSplus usage to our TreeNodeType enum
        usage_map = {{
            "STRUCTURE": "STRUCTURE",
            "SIGNAL": "SIGNAL",
            "NUMERIC": "NUMERIC",
            "TEXT": "TEXT",
            "AXIS": "AXIS",
            "SUBTREE": "SUBTREE",
            "ACTION": "ACTION",
            "DISPATCH": "DISPATCH",
            "TASK": "TASK",
        }}
        node_type = usage_map.get(usage, "STRUCTURE")

        # Try to get units (may fail for non-data nodes)
        try:
            units = node.units if hasattr(node, "units") else ""
        except Exception:
            units = ""

        # Try to get description
        try:
            desc = str(node.node_name) if hasattr(node, "node_name") else ""
        except Exception:
            desc = ""

        result.append({{
            "path": str(node.path),
            "node_type": node_type,
            "units": units or "dimensionless",
            "description": desc,
        }})
    except Exception as e:
        # Skip nodes that fail introspection
        pass

print(json.dumps(result))
'''

    # Run via SSH
    cmd = ["ssh", facility, f"python3 -c '{remote_script}'"]
    logger.info(f"Running introspection on {facility} for {tree_name} shot {shot}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for large trees
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout introspecting {tree_name} on {facility}")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"SSH command failed: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON output: {e}")
        raise


def ingest_tree_nodes(
    facility_id: str,
    tree_name: str,
    nodes: list[dict],
    shot: int,
    batch_size: int = 100,
) -> dict:
    """Ingest tree nodes into Neo4j.

    Args:
        facility_id: Facility identifier (e.g., "epfl")
        tree_name: MDSplus tree name
        nodes: List of node dicts from introspection
        shot: Reference shot number
        batch_size: Nodes per UNWIND batch

    Returns:
        Dict with ingestion stats
    """
    # Prepare nodes with computed fields
    prepared = []
    for node in nodes:
        path = node["path"]
        prepared.append(
            {
                "path": path,
                "tree_name": tree_name,
                "facility_id": facility_id,
                "node_type": node.get("node_type", "STRUCTURE"),
                "units": node.get("units", "dimensionless"),
                "description": node.get("description", ""),
                "example_shot": shot,
                "parent_path": compute_parent_path(path),
            }
        )

    # Batch insert with UNWIND
    with GraphClient() as client:
        processed = 0
        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            client.query(
                """
                UNWIND $batch AS item
                MERGE (n:TreeNode {path: item.path})
                SET n += item
                WITH n, item
                MATCH (f:Facility {id: item.facility_id})
                MERGE (n)-[:FACILITY_ID]->(f)
                """,
                batch=batch,
            )
            processed += len(batch)
            logger.info(f"Ingested {processed}/{len(prepared)} nodes")

        # Create TREE_NAME relationships
        client.query(
            """
            MATCH (n:TreeNode {tree_name: $tree_name})
            MATCH (t:MDSplusTree {name: $tree_name})
            MERGE (n)-[:TREE_NAME]->(t)
            """,
            tree_name=tree_name,
        )

        # Update MDSplusTree stats
        client.query(
            """
            MATCH (t:MDSplusTree {name: $tree_name})
            SET t.node_count_ingested = $count,
                t.ingestion_status = 'ingested',
                t.reference_shot = $shot,
                t.last_ingested = datetime()
            """,
            tree_name=tree_name,
            count=len(prepared),
            shot=shot,
        )

    return {
        "tree": tree_name,
        "nodes_ingested": len(prepared),
        "reference_shot": shot,
    }


@click.command()
@click.argument("facility")
@click.argument("trees", nargs=-1, required=True)
@click.option("--shot", "-s", type=int, required=True, help="Reference shot number")
@click.option("--limit", "-l", type=int, help="Limit number of nodes (for testing)")
@click.option(
    "--batch-size", "-b", type=int, default=100, help="Batch size for Neo4j writes"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be ingested without writing"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
def ingest_mdsplus(
    facility: str,
    trees: tuple[str, ...],
    shot: int,
    limit: int | None,
    batch_size: int,
    dry_run: bool,
    verbose: bool,
    quiet: bool,
) -> int:
    """Ingest MDSplus tree structure from a remote facility.

    FACILITY is the SSH host alias (e.g., "epfl").
    TREES are the MDSplus tree names to ingest (e.g., "results", "tcv_shot").

    Examples:
        ingest-mdsplus epfl results --shot 84469
        ingest-mdsplus epfl results tcv_shot --shot 84469 -v
        ingest-mdsplus epfl results --shot 84469 --limit 100 --dry-run
    """
    # Set up logging
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = []
    for tree_name in trees:
        logger.info(f"Processing {tree_name} tree...")

        try:
            # Introspect tree via SSH
            nodes = introspect_tree_ssh(facility, tree_name, shot, limit)
            logger.info(f"Found {len(nodes)} nodes in {tree_name}")

            if dry_run:
                click.echo(f"{tree_name}: {len(nodes)} nodes (dry run)")
                # Show sample
                for node in nodes[:5]:
                    click.echo(f"  {node['path']} ({node['node_type']})")
                if len(nodes) > 5:
                    click.echo(f"  ... and {len(nodes) - 5} more")
                continue

            # Ingest to Neo4j
            result = ingest_tree_nodes(
                facility_id=facility,
                tree_name=tree_name,
                nodes=nodes,
                shot=shot,
                batch_size=batch_size,
            )
            results.append(result)
            click.echo(f"✓ {tree_name}: {result['nodes_ingested']} nodes ingested")

        except Exception as e:
            logger.exception(f"Failed to process {tree_name}")
            click.echo(f"✗ {tree_name}: {e}", err=True)
            return 1

    if not dry_run and results:
        total = sum(r["nodes_ingested"] for r in results)
        click.echo(f"\nTotal: {total} nodes ingested across {len(results)} trees")

    return 0


if __name__ == "__main__":
    sys.exit(ingest_mdsplus())
