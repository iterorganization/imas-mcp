"""MDSplus node metadata extraction.

Extracts real units, descriptions (from COMMENT nodes), and other
metadata from MDSplus trees.
"""

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

# Remote script to extract full metadata including COMMENT data
METADATA_SCRIPT = """
import json
import MDSplus

tree_name = "{tree_name}"
shot = {shot}

t = MDSplus.Tree(tree_name, shot)
all_nodes = list(t.getNodeWild("***"))

# Build lookup for COMMENT nodes
comment_data = {{}}
for node in all_nodes:
    path = str(node.path)
    if path.endswith(":COMMENT"):
        try:
            data = node.data()
            if data:
                parent_path = path[:-8]  # Remove ":COMMENT"
                comment_data[parent_path] = str(data)[:500]
        except:
            pass

# Extract metadata for all non-COMMENT nodes
result = []
for node in all_nodes:
    path = str(node.path)
    if path.endswith(":COMMENT"):
        continue  # Skip COMMENT nodes themselves

    try:
        # Get usage/node type
        usage = str(node.usage)
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

        # Get units (strip whitespace)
        try:
            units = str(node.units).strip()
            if not units or units == " ":
                units = ""
        except:
            units = ""

        # Get tags
        try:
            tags = list(node.tags) if hasattr(node, "tags") else []
        except:
            tags = []

        # Get description from COMMENT child if available
        description = comment_data.get(path, "")

        record = {{
            "path": path,
            "node_type": node_type,
        }}

        # Only include non-empty values
        if units:
            record["units"] = units
        if description:
            record["description"] = description
        if tags:
            record["tags"] = tags

        result.append(record)
    except Exception:
        pass

print(json.dumps(result))
"""


def extract_metadata(
    facility: str,
    tree_name: str,
    shot: int,
    ssh_timeout: int = 120,
) -> list[dict]:
    """Extract full metadata from MDSplus tree at a specific shot.

    Args:
        facility: SSH host alias (e.g., "epfl")
        tree_name: MDSplus tree name
        shot: Shot number to query
        ssh_timeout: SSH timeout in seconds

    Returns:
        List of dicts with path, node_type, units, description, tags
    """
    script = METADATA_SCRIPT.format(tree_name=tree_name, shot=shot)

    # Escape single quotes
    escaped_script = script.replace("'", "'\"'\"'")
    cmd = ["ssh", facility, f"python3 -c '{escaped_script}'"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ssh_timeout,
            check=True,
        )
        metadata = json.loads(result.stdout)
        logger.info(f"Extracted metadata for {len(metadata)} nodes at shot {shot}")
        return metadata

    except subprocess.TimeoutExpired:
        logger.error(f"SSH timeout after {ssh_timeout}s extracting metadata")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"SSH failed: {e.stderr}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata response: {e}")
        return []


def enrich_graph_metadata(
    client: "GraphClient",  # noqa: F821
    facility: str,
    tree_name: str,
    shot: int,
    dry_run: bool = False,
) -> dict[str, int]:
    """Enrich TreeNode records in graph with real metadata.

    Args:
        client: Neo4j GraphClient
        facility: Facility ID
        tree_name: MDSplus tree name
        shot: Shot to extract metadata from
        dry_run: If True, log but don't write

    Returns:
        Dict with counts: units_updated, descriptions_updated, types_updated
    """
    # Extract metadata from MDSplus
    metadata = extract_metadata(facility, tree_name, shot)

    if not metadata:
        return {"units_updated": 0, "descriptions_updated": 0, "types_updated": 0}

    # Count what we found
    with_units = sum(1 for m in metadata if m.get("units"))
    with_desc = sum(1 for m in metadata if m.get("description"))

    logger.info(f"Found {with_units} nodes with units, {with_desc} with descriptions")

    if dry_run:
        # Show samples
        sample_units = [m for m in metadata if m.get("units")][:5]
        sample_desc = [m for m in metadata if m.get("description")][:5]

        if sample_units:
            logger.info("Sample units:")
            for m in sample_units:
                logger.info(f"  {m['path']}: {m['units']}")

        if sample_desc:
            logger.info("Sample descriptions:")
            for m in sample_desc:
                logger.info(f"  {m['path']}: {m['description'][:60]}...")

        return {
            "units_updated": with_units,
            "descriptions_updated": with_desc,
            "types_updated": len(metadata),
        }

    # Update nodes with units
    units_records = [m for m in metadata if m.get("units")]
    if units_records:
        client.query(
            """
            UNWIND $records AS r
            MATCH (n:TreeNode {path: r.path, facility_id: $facility})
            WHERE n.tree_name = $tree
            SET n.units = r.units
            """,
            records=units_records,
            facility=facility,
            tree=tree_name,
        )
        logger.info(f"Updated units for {len(units_records)} nodes")

    # Update nodes with descriptions
    desc_records = [m for m in metadata if m.get("description")]
    if desc_records:
        client.query(
            """
            UNWIND $records AS r
            MATCH (n:TreeNode {path: r.path, facility_id: $facility})
            WHERE n.tree_name = $tree
            SET n.description = r.description
            """,
            records=desc_records,
            facility=facility,
            tree=tree_name,
        )
        logger.info(f"Updated descriptions for {len(desc_records)} nodes")

    # Update node types
    type_records = [{"path": m["path"], "node_type": m["node_type"]} for m in metadata]
    if type_records:
        client.query(
            """
            UNWIND $records AS r
            MATCH (n:TreeNode {path: r.path, facility_id: $facility})
            WHERE n.tree_name = $tree
            SET n.node_type = r.node_type
            """,
            records=type_records,
            facility=facility,
            tree=tree_name,
        )
        logger.info(f"Updated node types for {len(type_records)} nodes")

    return {
        "units_updated": len(units_records),
        "descriptions_updated": len(desc_records),
        "types_updated": len(type_records),
    }
