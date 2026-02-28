"""Static/machine-description tree extraction and ingestion.

Extracts time-invariant constructional data from MDSplus static trees:
vessel geometry, coil positions, tile contours, magnetic probe positions,
flux loop positions, Green's functions, and mesh grids.

Static trees differ from shot-dependent signal trees:
- Versioned by machine configuration, not by shot-to-shot changes
- Opened by version number (shot=1..N), not experimental shot number
- Contain geometry and parameters that change only during shutdowns

Generic: works with any facility that has static_trees configured
in its facility YAML.

Usage:
    from imas_codex.mdsplus.static import discover_static_tree, ingest_static_tree
    from imas_codex.graph import GraphClient

    # Extract from facility
    data = discover_static_tree("tcv", "static", versions=[1,2,3,4,5,6,7,8])

    # Ingest to Neo4j
    with GraphClient() as client:
        ingest_static_tree(client, "tcv", data)
"""

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from imas_codex.mdsplus.ingestion import compute_canonical_path, normalize_mdsplus_path
from imas_codex.remote.executor import async_run_python_script, run_python_script

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


def get_static_tree_config(facility: str) -> list[dict[str, Any]]:
    """Load static_trees config from facility YAML.

    Args:
        facility: Facility identifier (e.g., "tcv")

    Returns:
        List of static tree config dicts, or empty list if none configured.
    """
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    data_sources = config.get("data_sources", {})
    mdsplus = data_sources.get("mdsplus", {})
    return mdsplus.get("static_trees", [])


def _load_mdsplus_config(facility: str) -> dict[str, Any]:
    """Load MDSplus config section from facility YAML."""
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    return config.get("data_sources", {}).get("mdsplus", {})


def _resolve_versions(
    facility: str,
    tree_name: str,
    versions: list[int] | None,
) -> tuple[list[int], bool]:
    """Resolve version list and extract_values from config if not provided.

    Returns:
        (versions, extract_values) tuple.
    """
    extract_values = False
    if versions is None:
        configs = get_static_tree_config(facility)
        for cfg in configs:
            if cfg.get("tree_name") == tree_name:
                ver_list = cfg.get("versions", [])
                versions = [v["version"] for v in ver_list]
                if cfg.get("extract_values") is not None:
                    extract_values = cfg["extract_values"]
                break
        if versions is None:
            versions = [1]
    return versions, extract_values


def discover_static_tree_version(
    facility: str,
    tree_name: str,
    version: int,
    extract_values: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Extract a single version of a static tree from a remote facility.

    Runs the remote script for one version only, keeping SSH sessions
    short enough to avoid timeouts on large trees.

    Args:
        facility: SSH host alias (e.g., "tcv")
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number to extract
        extract_values: Whether to extract numerical data
        timeout: SSH timeout in seconds

    Returns:
        Dict with version data: {"tree_name": str, "versions": {"N": {...}}, "diff": {}}
    """
    mdsplus = _load_mdsplus_config(facility)
    exclude_names = mdsplus.get("exclude_node_names", [])
    setup_commands = mdsplus.get("setup_commands")

    input_data = {
        "tree_name": tree_name,
        "versions": [version],
        "extract_values": extract_values,
        "exclude_names": exclude_names,
    }

    logger.info(
        "Extracting static tree %s v%d from %s (values=%s)",
        tree_name,
        version,
        facility,
        extract_values,
    )

    output = run_python_script(
        "extract_static_tree.py",
        input_data=input_data,
        ssh_host=facility,
        timeout=timeout,
        setup_commands=setup_commands,
    )

    data = json.loads(output)

    ver_str = str(version)
    ver_data = data.get("versions", {}).get(ver_str, {})
    if "error" in ver_data:
        logger.warning("Version %d: %s", version, ver_data["error"])
    else:
        logger.info(
            "Version %d: %d nodes, %d tags",
            version,
            ver_data.get("node_count", 0),
            len(ver_data.get("tags", {})),
        )

    return data


async def async_discover_static_tree_version(
    facility: str,
    tree_name: str,
    version: int,
    extract_values: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Async version of discover_static_tree_version.

    Uses async_run_python_script for non-blocking SSH calls,
    allowing concurrent version extraction.
    """
    mdsplus = _load_mdsplus_config(facility)
    exclude_names = mdsplus.get("exclude_node_names", [])
    setup_commands = mdsplus.get("setup_commands")

    input_data = {
        "tree_name": tree_name,
        "versions": [version],
        "extract_values": extract_values,
        "exclude_names": exclude_names,
    }

    logger.info(
        "Extracting static tree %s v%d from %s (values=%s)",
        tree_name,
        version,
        facility,
        extract_values,
    )

    output = await async_run_python_script(
        "extract_static_tree.py",
        input_data=input_data,
        ssh_host=facility,
        timeout=timeout,
        setup_commands=setup_commands,
    )

    data = json.loads(output)

    ver_str = str(version)
    ver_data = data.get("versions", {}).get(ver_str, {})
    if "error" in ver_data:
        logger.warning("Version %d: %s", version, ver_data["error"])
    else:
        logger.info(
            "Version %d: %d nodes, %d tags",
            version,
            ver_data.get("node_count", 0),
            len(ver_data.get("tags", {})),
        )

    return data


async def async_extract_units_for_version(
    facility: str,
    tree_name: str,
    version: int,
    timeout: int = 180,
    batch_size: int = 5000,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, str]:
    """Async version of extract_units_for_version.

    Uses async_run_python_script for non-blocking SSH calls.
    """
    mdsplus = _load_mdsplus_config(facility)
    setup_commands = mdsplus.get("setup_commands")

    all_units: dict[str, str] = {}
    offset = 0
    total_nodes = 0
    total_checked = 0
    batch_num = 0

    logger.info(
        "Extracting units from %s:%s v%d (batch_size=%d)...",
        facility,
        tree_name,
        version,
        batch_size,
    )

    while True:
        batch_num += 1
        input_data = {
            "tree_name": tree_name,
            "version": version,
            "node_types": ["NUMERIC", "SIGNAL"],
            "offset": offset,
            "limit": batch_size,
        }

        try:
            output = ""
            output = await async_run_python_script(
                "extract_units.py",
                input_data=input_data,
                ssh_host=facility,
                timeout=timeout,
                setup_commands=setup_commands,
            )
            json_line = output.strip().split("\n")[0]
            data = json.loads(json_line)
        except Exception:
            logger.exception(
                "Units batch %d failed for %s:%s v%d (offset=%d)",
                batch_num,
                facility,
                tree_name,
                version,
                offset,
            )
            if output:
                logger.debug("Raw output (first 500 chars): %s", output[:500])
            break

        if "error" in data:
            logger.warning(
                "Units batch %d error v%d: %s", batch_num, version, data["error"]
            )
            break

        batch_units = data.get("units", {})
        all_units.update(batch_units)
        total_nodes = data.get("total_nodes", 0)
        batch_checked = data.get("batch_checked", 0)
        total_checked += batch_checked

        logger.info(
            "  Batch %d: checked %d nodes (offset %d-%d), found %d units",
            batch_num,
            batch_checked,
            offset,
            offset + batch_checked,
            len(batch_units),
        )

        if on_progress:
            on_progress(total_checked, total_nodes, len(all_units))

        offset += batch_size
        if offset >= total_nodes:
            break

    logger.info(
        "Units extraction complete: %d paths with units out of %d checked",
        len(all_units),
        total_checked,
    )
    return all_units


def merge_version_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-version extraction results into a single data dict.

    Combines version data from multiple single-version extractions and
    computes structural diffs across all versions.

    Args:
        results: List of dicts from discover_static_tree_version().

    Returns:
        Combined dict with all versions and cross-version diff.
    """
    if not results:
        return {"tree_name": "", "versions": {}, "diff": {}}

    tree_name = results[0].get("tree_name", "")
    merged_versions: dict[str, dict] = {}

    for r in results:
        for ver_str, ver_data in r.get("versions", {}).items():
            merged_versions[ver_str] = ver_data

    # Recompute diff across all merged versions
    diff = _compute_diff(merged_versions)

    return {"tree_name": tree_name, "versions": merged_versions, "diff": diff}


def _compute_diff(
    version_data: dict[str, dict],
) -> dict[str, dict[str, list[str]]]:
    """Compute structural differences between consecutive versions."""
    sorted_versions = sorted(version_data.keys(), key=int)
    added: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}

    prev_paths: set[str] | None = None
    for ver in sorted_versions:
        data = version_data[ver]
        if "error" in data:
            continue
        current_paths = {n["path"] for n in data.get("nodes", [])}
        if prev_paths is not None:
            new_paths = sorted(current_paths - prev_paths)
            gone_paths = sorted(prev_paths - current_paths)
            if new_paths:
                added[ver] = new_paths
            if gone_paths:
                removed[ver] = gone_paths
        prev_paths = current_paths

    return {"added": added, "removed": removed}


def discover_static_tree(
    facility: str,
    tree_name: str,
    versions: list[int] | None = None,
    extract_values: bool = False,
    timeout: int = 300,
) -> dict[str, Any]:
    """Extract static tree structure and values from a remote facility.

    Extracts each version individually to avoid SSH timeout on large trees,
    then merges results and computes cross-version diffs.

    Args:
        facility: SSH host alias (e.g., "tcv")
        tree_name: MDSplus tree name (e.g., "static")
        versions: Version numbers to extract (default: all from config)
        extract_values: Whether to extract numerical data (R/Z, matrices)
        timeout: SSH timeout per version in seconds

    Returns:
        Dict with version data, structural diffs, and tag mappings.
        Structure: {"tree_name": str, "versions": {ver: {...}}, "diff": {...}}
    """
    versions, extract_values_cfg = _resolve_versions(facility, tree_name, versions)
    if not extract_values:
        extract_values = extract_values_cfg

    results = []
    for ver in versions:
        data = discover_static_tree_version(
            facility=facility,
            tree_name=tree_name,
            version=ver,
            extract_values=extract_values,
            timeout=timeout,
        )
        results.append(data)

    return merge_version_results(results)


def extract_units_for_version(
    facility: str,
    tree_name: str,
    version: int,
    timeout: int = 180,
    batch_size: int = 5000,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, str]:
    """Extract units from data-bearing nodes in batches.

    Runs multiple SSH jobs, each processing ``batch_size`` nodes.
    This avoids a single long-running session that can timeout, provides
    streamed progress updates, and preserves partial results on failure.

    Each batch takes ~90s for 5000 nodes at ~18ms/node.

    Args:
        facility: SSH host alias (e.g., "tcv")
        tree_name: MDSplus tree name (e.g., "static")
        version: Version number to extract units from (typically the latest)
        timeout: SSH timeout per batch in seconds (default: 120)
        batch_size: Nodes per SSH call (default: 5000)
        on_progress: Optional callback(checked, total, found) called after each batch

    Returns:
        Dict mapping path → unit string.
        Partial results on batch failure (logged, not raised).
    """
    mdsplus = _load_mdsplus_config(facility)
    setup_commands = mdsplus.get("setup_commands")

    # First batch discovers total node count
    all_units: dict[str, str] = {}
    offset = 0
    total_nodes = 0
    total_checked = 0
    batch_num = 0

    logger.info(
        "Extracting units from %s:%s v%d (batch_size=%d)...",
        facility,
        tree_name,
        version,
        batch_size,
    )

    while True:
        batch_num += 1
        input_data = {
            "tree_name": tree_name,
            "version": version,
            "node_types": ["NUMERIC", "SIGNAL"],
            "offset": offset,
            "limit": batch_size,
        }

        try:
            output = ""
            output = run_python_script(
                "extract_units.py",
                input_data=input_data,
                ssh_host=facility,
                timeout=timeout,
                setup_commands=setup_commands,
            )
            # Parse only the first line of output — MDSplus C library or
            # Python cleanup may print extra lines after our JSON.
            json_line = output.strip().split("\n")[0]
            data = json.loads(json_line)
        except Exception:
            logger.exception(
                "Units batch %d failed for %s:%s v%d (offset=%d)",
                batch_num,
                facility,
                tree_name,
                version,
                offset,
            )
            if output:
                logger.debug("Raw output (first 500 chars): %s", output[:500])
            break

        if "error" in data:
            logger.warning(
                "Units batch %d error v%d: %s", batch_num, version, data["error"]
            )
            break

        batch_units = data.get("units", {})
        all_units.update(batch_units)
        total_nodes = data.get("total_nodes", 0)
        batch_checked = data.get("batch_checked", 0)
        total_checked += batch_checked

        logger.info(
            "  Batch %d: checked %d nodes (offset %d-%d), found %d units",
            batch_num,
            batch_checked,
            offset,
            offset + batch_checked,
            len(batch_units),
        )

        if on_progress:
            on_progress(total_checked, total_nodes, len(all_units))

        offset += batch_size
        if offset >= total_nodes:
            break

    logger.info(
        "Units extraction complete: %d paths with units out of %d checked",
        len(all_units),
        total_checked,
    )
    return all_units


def merge_units_into_data(
    data: dict[str, Any],
    units: dict[str, str],
) -> int:
    """Merge extracted units into tree data, updating nodes in-place.

    Matches by exact path. Units from the latest version are applied
    to all versions containing that path (units are constructional
    and don't change between versions).

    Args:
        data: Tree data dict from merge_version_results()
        units: Path → unit string mapping from extract_units_for_version()

    Returns:
        Number of nodes updated with units.
    """
    updated = 0
    for ver_data in data.get("versions", {}).values():
        if "error" in ver_data:
            continue
        for node in ver_data.get("nodes", []):
            path = node.get("path", "")
            if path in units:
                node["units"] = units[path]
                updated += 1
    return updated


def ingest_static_tree(
    client: "GraphClient | None",
    facility: str,
    data: dict[str, Any],
    version_config: list[dict] | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Ingest static tree data into the Neo4j graph.

    Creates:
    - TreeModelVersion nodes for each static tree version
    - TreeNode nodes with applicability ranges (first_shot/last_shot)
    - INTRODUCED_IN / REMOVED_IN / AT_FACILITY relationships
    - Tag metadata on TreeNodes

    Args:
        client: Neo4j GraphClient
        facility: Facility identifier
        data: Output from discover_static_tree()
        version_config: Version configs with first_shot info (from facility YAML).
            If None, loaded from facility config.
        dry_run: If True, log but don't write

    Returns:
        Dict with counts: versions_created, nodes_created, values_stored
    """
    tree_name = data["tree_name"]
    versions = data.get("versions", {})
    diff = data.get("diff", {})

    stats = {"versions_created": 0, "nodes_created": 0, "values_stored": 0}

    if not versions:
        logger.warning("No version data to ingest")
        return stats

    # Load version config for first_shot mapping
    if version_config is None:
        configs = get_static_tree_config(facility)
        for cfg in configs:
            if cfg.get("tree_name") == tree_name:
                version_config = cfg.get("versions", [])
                break
    if version_config is None:
        version_config = []

    # Build first_shot lookup
    first_shot_map: dict[int, int] = {}
    description_map: dict[int, str] = {}
    for vc in version_config:
        ver = vc["version"]
        if "first_shot" in vc:
            first_shot_map[ver] = vc["first_shot"]
        if "description" in vc:
            description_map[ver] = vc["description"]

    # Sort versions for last_shot computation
    sorted_versions = sorted(first_shot_map.keys())

    # Compute last_shot for each version (one before the next version starts)
    last_shot_map: dict[int, int | None] = {}
    for i, ver in enumerate(sorted_versions):
        if i + 1 < len(sorted_versions):
            next_first = first_shot_map[sorted_versions[i + 1]]
            last_shot_map[ver] = next_first - 1
        else:
            last_shot_map[ver] = None  # Current version — no upper bound

    # Phase 1: Create TreeModelVersion nodes
    epoch_records = []
    for ver_str, ver_data in versions.items():
        if "error" in ver_data:
            continue
        ver_num = int(ver_str)
        epoch_id = f"{facility}:{tree_name}:v{ver_num}"
        first_shot = first_shot_map.get(ver_num, ver_num)  # Fallback to version number
        last_shot = last_shot_map.get(ver_num)
        desc = description_map.get(ver_num, "")

        record = {
            "id": epoch_id,
            "facility_id": facility,
            "tree_name": tree_name,
            "version": ver_num,
            "first_shot": first_shot,
            "node_count": ver_data.get("node_count", 0),
            "fingerprint": f"static_v{ver_num}",
            "is_static": True,
            "description": desc,
        }
        if last_shot is not None:
            record["last_shot"] = last_shot

        epoch_records.append(record)

    if dry_run:
        for rec in epoch_records:
            logger.info(
                "[DRY RUN] TreeModelVersion: %s (v%d, shots %d-%s) — %s",
                rec["id"],
                rec["version"],
                rec["first_shot"],
                rec.get("last_shot", "current"),
                rec.get("description", ""),
            )
    else:
        if epoch_records:
            client.query(
                """
                UNWIND $epochs AS epoch
                MERGE (v:TreeModelVersion {id: epoch.id})
                SET v += epoch
                WITH v, epoch
                MATCH (f:Facility {id: epoch.facility_id})
                MERGE (v)-[:AT_FACILITY]->(f)
                """,
                epochs=epoch_records,
            )
            # Create SUCCEEDS chain
            for i in range(1, len(epoch_records)):
                client.query(
                    """
                    MATCH (curr:TreeModelVersion {id: $curr_id})
                    MATCH (prev:TreeModelVersion {id: $prev_id})
                    MERGE (curr)-[:SUCCEEDS]->(prev)
                    """,
                    curr_id=epoch_records[i]["id"],
                    prev_id=epoch_records[i - 1]["id"],
                )

    stats["versions_created"] = len(epoch_records)

    # Phase 2: Build super tree — all nodes with applicability ranges
    # Use the most complete version (last one) as the base
    all_paths: dict[str, dict[str, Any]] = {}

    for ver_str in sorted(versions.keys(), key=int):
        ver_data = versions[ver_str]
        if "error" in ver_data:
            continue
        ver_num = int(ver_str)
        first_shot = first_shot_map.get(ver_num, ver_num)
        epoch_id = f"{facility}:{tree_name}:v{ver_num}"

        for node in ver_data.get("nodes", []):
            path = node["path"]
            if path not in all_paths:
                all_paths[path] = {
                    "first_seen_version": ver_num,
                    "first_shot": first_shot,
                    "introduced_version": epoch_id,
                    "last_shot": None,
                    "removed_version": None,
                    "node": node,
                }

    # Mark removals from diff data
    for ver_str, removed_paths in diff.get("removed", {}).items():
        ver_num = int(ver_str)
        epoch_id = f"{facility}:{tree_name}:v{ver_num}"
        prev_first_shot = first_shot_map.get(ver_num, ver_num)
        for path in removed_paths:
            if path in all_paths and all_paths[path]["removed_version"] is None:
                all_paths[path]["last_shot"] = prev_first_shot - 1
                all_paths[path]["removed_version"] = epoch_id

    # Build TreeNode records
    node_records = []
    value_records = []
    for path, info in all_paths.items():
        node = info["node"]
        normalized = normalize_mdsplus_path(path)
        canonical = compute_canonical_path(path)
        node_id = f"{facility}:{tree_name}:{normalized}"

        record: dict[str, Any] = {
            "id": node_id,
            "path": normalized,
            "canonical_path": canonical,
            "tree_name": tree_name,
            "facility_id": facility,
            "node_type": node.get("node_type", "STRUCTURE"),
            "first_shot": info["first_shot"],
            "introduced_version": info["introduced_version"],
            "source": "static_tree_extraction",
            "is_static": True,
        }

        if info["last_shot"] is not None:
            record["last_shot"] = info["last_shot"]
        if info["removed_version"] is not None:
            record["removed_version"] = info["removed_version"]
        if node.get("units"):
            record["units"] = node["units"]
        if node.get("description"):
            record["description"] = node["description"]
        if node.get("tags"):
            record["tags"] = node["tags"]

        # Parent path
        parent = _compute_parent_path(normalized)
        if parent:
            record["parent_path"] = parent

        node_records.append(record)

        # Track values separately (don't store large arrays in Neo4j properties)
        if node.get("value") is not None or node.get("value_summary") is not None:
            val_record = {
                "node_id": node_id,
                "path": normalized,
            }
            if node.get("shape") is not None:
                val_record["shape"] = node["shape"]
            if node.get("dtype"):
                val_record["dtype"] = node["dtype"]
            if node.get("value") is not None:
                val = node["value"]
                # Store scalars and small arrays directly on the node
                if isinstance(val, int | float):
                    val_record["scalar_value"] = val
                elif isinstance(val, list) and len(val) <= 100:
                    val_record["array_value"] = val
                # Large arrays: store shape/summary only
            if node.get("value_summary"):
                val_record["value_summary"] = node["value_summary"]
            value_records.append(val_record)

    if dry_run:
        logger.info("[DRY RUN] Would create %d TreeNode records", len(node_records))
        # Show samples by node type
        by_type: dict[str, int] = {}
        for r in node_records:
            t = r.get("node_type", "STRUCTURE")
            by_type[t] = by_type.get(t, 0) + 1
        for t, count in sorted(by_type.items()):
            logger.info("  %s: %d nodes", t, count)
        if value_records:
            logger.info("[DRY RUN] Would store values for %d nodes", len(value_records))
    else:
        # Batch insert TreeNodes
        batch_size = 500
        for i in range(0, len(node_records), batch_size):
            batch = node_records[i : i + batch_size]
            client.query(
                """
                UNWIND $nodes AS node
                MERGE (n:TreeNode {path: node.path, facility_id: node.facility_id})
                SET n.id = node.id,
                    n.tree_name = node.tree_name,
                    n.canonical_path = node.canonical_path,
                    n.parent_path = node.parent_path,
                    n.first_shot = node.first_shot,
                    n.last_shot = node.last_shot,
                    n.introduced_version = node.introduced_version,
                    n.removed_version = node.removed_version,
                    n.node_type = node.node_type,
                    n.source = node.source,
                    n.units = node.units,
                    n.description = node.description,
                    n.tags = node.tags,
                    n.is_static = node.is_static
                """,
                nodes=batch,
            )

        # Store scalar values directly on nodes
        scalar_batch = [v for v in value_records if "scalar_value" in v]
        if scalar_batch:
            client.query(
                """
                UNWIND $values AS val
                MATCH (n:TreeNode {id: val.node_id})
                SET n.scalar_value = val.scalar_value,
                    n.shape = val.shape,
                    n.dtype = val.dtype
                """,
                values=scalar_batch,
            )

        # Store small array values
        array_batch = [v for v in value_records if "array_value" in v]
        if array_batch:
            client.query(
                """
                UNWIND $values AS val
                MATCH (n:TreeNode {id: val.node_id})
                SET n.array_value = val.array_value,
                    n.shape = val.shape,
                    n.dtype = val.dtype
                """,
                values=array_batch,
            )

        # Create relationships
        client.query(
            """
            MATCH (n:TreeNode)
            WHERE n.tree_name = $tree_name AND n.facility_id = $facility
              AND n.is_static = true
            WITH n
            MATCH (f:Facility {id: $facility})
            MERGE (n)-[:AT_FACILITY]->(f)
            """,
            tree_name=tree_name,
            facility=facility,
        )

        client.query(
            """
            MATCH (n:TreeNode)
            WHERE n.tree_name = $tree_name AND n.facility_id = $facility
              AND n.is_static = true AND n.introduced_version IS NOT NULL
            WITH n
            MATCH (v:TreeModelVersion {id: n.introduced_version})
            MERGE (n)-[:INTRODUCED_IN]->(v)
            """,
            tree_name=tree_name,
            facility=facility,
        )

        client.query(
            """
            MATCH (n:TreeNode)
            WHERE n.tree_name = $tree_name AND n.facility_id = $facility
              AND n.is_static = true AND n.removed_version IS NOT NULL
            WITH n
            MATCH (v:TreeModelVersion {id: n.removed_version})
            MERGE (n)-[:REMOVED_IN]->(v)
            """,
            tree_name=tree_name,
            facility=facility,
        )

        # Parent-child relationships
        client.query(
            """
            MATCH (child:TreeNode)
            WHERE child.tree_name = $tree_name AND child.facility_id = $facility
              AND child.is_static = true AND child.parent_path IS NOT NULL
            WITH child
            MATCH (parent:TreeNode {path: child.parent_path, facility_id: $facility})
            WHERE parent.tree_name = $tree_name
            MERGE (parent)-[:HAS_NODE]->(child)
            """,
            tree_name=tree_name,
            facility=facility,
        )

    stats["nodes_created"] = len(node_records)
    stats["values_stored"] = len(value_records)

    logger.info(
        "Static tree %s:%s — %d versions, %d nodes, %d values",
        facility,
        tree_name,
        stats["versions_created"],
        stats["nodes_created"],
        stats["values_stored"],
    )

    return stats


def _compute_parent_path(path: str) -> str | None:
    """Compute parent path for a tree node.

    Handles both `.` and `:` hierarchy separators used by MDSplus.

    Examples:
        \\STATIC::TOP.C.R -> \\STATIC::TOP.C
        \\STATIC::VESSEL:VAL:R -> \\STATIC::VESSEL:VAL
        \\STATIC::TOP -> None
    """
    if "::" in path:
        tree_part, node_part = path.split("::", 1)
        last_dot = node_part.rfind(".")
        last_colon = node_part.rfind(":")
        last_sep = max(last_dot, last_colon)
        if last_sep < 0:
            return None
        parent_node = node_part[:last_sep]
        return f"{tree_part}::{parent_node}"
    else:
        if "." not in path:
            return None
        return ".".join(path.rsplit(".", 1)[:-1])
