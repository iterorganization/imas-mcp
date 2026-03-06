"""MDSplus tree extraction.

Extracts tree structure from any MDSplus tree — versioned (static),
shot-scoped (dynamic), or epoched. The extraction uses the same remote
script (``extract_tree.py``) for all tree types: MDSplus
``Tree(name, shot)`` works identically whether the shot number
represents a machine-configuration version or an experimental shot.

Usage:
    from imas_codex.mdsplus.extraction import discover_tree, extract_tree_version
    from imas_codex.graph import GraphClient

    # Extract a single version/shot
    data = extract_tree_version("tcv", "static", shot=8)

    # Extract all configured versions
    data = discover_tree("tcv", "static", shots=[1,2,3,4,5,6,7,8])
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
    """Load versioned tree configs from facility YAML.

    Returns trees from the unified ``trees`` list that have versions
    configured (i.e., versioned/static trees).

    Args:
        facility: Facility identifier (e.g., "tcv")

    Returns:
        List of tree config dicts that have versions, or empty list.
    """
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    data_systems = config.get("data_systems", {})
    mdsplus = data_systems.get("mdsplus", {})
    all_trees = mdsplus.get("trees", [])
    return [t for t in all_trees if t.get("versions")]


def get_static_tree_graph_state(
    client: "GraphClient",
    facility: str,
    data_source_name: str,
    ver_list: list[int],
) -> dict[str, Any]:
    """Query graph for existing static tree state.

    Returns information about which versions are already extracted,
    which nodes exist, and which have been enriched.

    Args:
        client: Neo4j GraphClient
        facility: Facility identifier
        data_source_name: Static tree name
        ver_list: Version numbers to check

    Returns:
        Dict with:
          - ingested_versions: set of version ints already in graph
          - version_node_counts: {version_int: node_count} from graph
          - total_nodes: total DataNode count for this tree
          - enriched_nodes: count of DataNodes with descriptions
          - unenriched_paths: list of paths needing enrichment
    """
    # Check which StructuralEpochs exist
    epoch_ids = [f"{facility}:{data_source_name}:v{v}" for v in ver_list]
    result = client.query(
        """
        UNWIND $ids AS eid
        OPTIONAL MATCH (v:StructuralEpoch {id: eid})
        RETURN eid, v.version AS version, v.node_count AS node_count
        """,
        ids=epoch_ids,
    )

    ingested_versions: set[int] = set()
    version_node_counts: dict[int, int] = {}
    for row in result:
        if row["version"] is not None:
            ver = int(row["version"])
            ingested_versions.add(ver)
            version_node_counts[ver] = int(row["node_count"] or 0)

    # Count DataNodes and enrichment state
    node_stats = client.query(
        """
        MATCH (n:DataNode)
        WHERE n.data_source_name = $data_source_name AND n.facility_id = $facility
        RETURN
            count(n) AS total,
            sum(CASE WHEN n.description IS NOT NULL AND n.description <> ''
                THEN 1 ELSE 0 END) AS enriched
        """,
        data_source_name=data_source_name,
        facility=facility,
    )

    total_nodes = 0
    enriched_nodes = 0
    if node_stats:
        total_nodes = int(node_stats[0].get("total", 0))
        enriched_nodes = int(node_stats[0].get("enriched", 0))

    # Get paths that need enrichment (data-bearing nodes without descriptions)
    unenriched = client.query(
        """
        MATCH (n:DataNode)
        WHERE n.data_source_name = $data_source_name AND n.facility_id = $facility
          AND n.node_type IN ['NUMERIC', 'SIGNAL', 'AXIS', 'TEXT']
          AND (n.description IS NULL OR n.description = '')
        RETURN n.path AS path, n.node_type AS node_type,
               n.tags AS tags, n.unit AS unit
        """,
        data_source_name=data_source_name,
        facility=facility,
    )

    unenriched_paths = [dict(r) for r in unenriched]

    return {
        "ingested_versions": ingested_versions,
        "version_node_counts": version_node_counts,
        "total_nodes": total_nodes,
        "enriched_nodes": enriched_nodes,
        "unenriched_paths": unenriched_paths,
    }


def _load_mdsplus_config(facility: str) -> dict[str, Any]:
    """Load MDSplus config section from facility YAML."""
    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    return config.get("data_systems", {}).get("mdsplus", {})


def _resolve_shots(
    facility: str,
    data_source_name: str,
    shots: list[int] | None,
) -> list[int]:
    """Resolve shot/version list from config if not provided.

    For versioned trees, returns version numbers.
    For shot-scoped trees, falls back to reference_shot.

    Returns:
        List of shot/version numbers.
    """
    if shots is not None:
        return shots

    from imas_codex.discovery.base.facility import get_facility

    config = get_facility(facility)
    mdsplus = config.get("data_systems", {}).get("mdsplus", {})
    all_trees = mdsplus.get("trees", [])

    for cfg in all_trees:
        if cfg.get("source_name") == data_source_name:
            ver_list = cfg.get("versions", [])
            if ver_list:
                return [v["version"] for v in ver_list]
            break
        # Check subtrees
        for sub in cfg.get("subtrees", []):
            if sub.get("source_name") == data_source_name:
                ref = cfg.get("reference_shot") or mdsplus.get("reference_shot")
                if ref:
                    return [ref]
                break

    ref = mdsplus.get("reference_shot")
    return [ref] if ref else [1]


def extract_tree_version(
    facility: str,
    data_source_name: str,
    shot: int,
    timeout: int = 300,
    node_usages: list[str] | None = None,
) -> dict[str, Any]:
    """Extract a single version/shot of a tree from a remote facility.

    Runs the remote script for one shot/version, keeping SSH sessions
    short enough to avoid timeouts on large trees.

    Args:
        facility: SSH host alias (e.g., "tcv")
        data_source_name: MDSplus tree name (e.g., "static", "results")
        shot: Shot or version number to extract
        timeout: SSH timeout in seconds
        node_usages: If set, only extract nodes with these usage types.
            None extracts all nodes (full structure).

    Returns:
        Dict with version data: {"data_source_name": str, "versions": {"N": {...}}, "diff": {}}
    """
    mdsplus = _load_mdsplus_config(facility)
    exclude_names = mdsplus.get("exclude_node_names", [])
    setup_commands = mdsplus.get("setup_commands")

    input_data: dict[str, Any] = {
        "data_source_name": data_source_name,
        "shots": [shot],
        "exclude_names": exclude_names,
    }
    if node_usages:
        input_data["node_usages"] = node_usages

    logger.info(
        "Extracting tree %s shot=%d from %s",
        data_source_name,
        shot,
        facility,
    )

    output = run_python_script(
        "extract_tree.py",
        input_data=input_data,
        ssh_host=facility,
        timeout=timeout,
        setup_commands=setup_commands,
    )

    data = json.loads(output)

    ver_str = str(shot)
    ver_data = data.get("versions", {}).get(ver_str, {})
    if "error" in ver_data:
        logger.warning("Shot %d: %s", shot, ver_data["error"])
    else:
        logger.info(
            "Shot %d: %d nodes, %d tags",
            shot,
            ver_data.get("node_count", 0),
            len(ver_data.get("tags", {})),
        )

    return data


# Backward-compatible alias
discover_static_tree_version = extract_tree_version


async def async_extract_tree_version(
    facility: str,
    data_source_name: str,
    shot: int,
    timeout: int = 300,
    node_usages: list[str] | None = None,
) -> dict[str, Any]:
    """Async version of extract_tree_version.

    Uses async_run_python_script for non-blocking SSH calls,
    allowing concurrent version/shot extraction.
    """
    mdsplus = _load_mdsplus_config(facility)
    exclude_names = mdsplus.get("exclude_node_names", [])
    setup_commands = mdsplus.get("setup_commands")

    input_data: dict[str, Any] = {
        "data_source_name": data_source_name,
        "shots": [shot],
        "exclude_names": exclude_names,
    }
    if node_usages:
        input_data["node_usages"] = node_usages

    logger.info(
        "Extracting tree %s shot=%d from %s",
        data_source_name,
        shot,
        facility,
    )

    output = await async_run_python_script(
        "extract_tree.py",
        input_data=input_data,
        ssh_host=facility,
        timeout=timeout,
        setup_commands=setup_commands,
    )

    data = json.loads(output)

    ver_str = str(shot)
    ver_data = data.get("versions", {}).get(ver_str, {})
    if "error" in ver_data:
        logger.warning("Shot %d: %s", shot, ver_data["error"])
    else:
        logger.info(
            "Shot %d: %d nodes, %d tags",
            shot,
            ver_data.get("node_count", 0),
            len(ver_data.get("tags", {})),
        )

    return data


# Backward-compatible alias
async_discover_static_tree_version = async_extract_tree_version


async def async_extract_units_for_version(
    facility: str,
    data_source_name: str,
    version: int,
    timeout: int = 180,
    batch_size: int = 500,
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
        data_source_name,
        version,
        batch_size,
    )

    while True:
        batch_num += 1
        input_data = {
            "data_source_name": data_source_name,
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
                data_source_name,
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
        return {"data_source_name": "", "versions": {}, "diff": {}}

    data_source_name = results[0].get("data_source_name", "")
    merged_versions: dict[str, dict] = {}

    for r in results:
        for ver_str, ver_data in r.get("versions", {}).items():
            merged_versions[ver_str] = ver_data

    # Recompute diff across all merged versions
    diff = _compute_diff(merged_versions)

    return {
        "data_source_name": data_source_name,
        "versions": merged_versions,
        "diff": diff,
    }


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


def discover_tree(
    facility: str,
    data_source_name: str,
    shots: list[int] | None = None,
    timeout: int = 300,
    node_usages: list[str] | None = None,
) -> dict[str, Any]:
    """Extract tree structure from a remote facility.

    Extracts each shot/version individually to avoid SSH timeout on
    large trees, then merges results and computes cross-version diffs.

    Works for any tree type: versioned (static), shot-scoped (dynamic),
    or epoched. The ``shots`` parameter accepts version numbers for
    versioned trees or shot numbers for dynamic trees.

    Args:
        facility: SSH host alias (e.g., "tcv")
        data_source_name: MDSplus tree name (e.g., "static", "results")
        shots: Shot/version numbers to extract (default: from config)
        timeout: SSH timeout per shot/version in seconds
        node_usages: If set, only extract nodes with these usage types.

    Returns:
        Dict with version data, structural diffs, and tag mappings.
        Structure: {"data_source_name": str, "versions": {shot: {...}}, "diff": {...}}
    """
    shots = _resolve_shots(facility, data_source_name, shots)

    results = []
    for shot in shots:
        data = extract_tree_version(
            facility=facility,
            data_source_name=data_source_name,
            shot=shot,
            timeout=timeout,
            node_usages=node_usages,
        )
        results.append(data)

    return merge_version_results(results)


# Backward-compatible alias
discover_static_tree = discover_tree


def extract_units_for_version(
    facility: str,
    data_source_name: str,
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
        data_source_name: MDSplus tree name (e.g., "static")
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
        data_source_name,
        version,
        batch_size,
    )

    while True:
        batch_num += 1
        input_data = {
            "data_source_name": data_source_name,
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
                data_source_name,
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
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, int]:
    """Ingest static tree data into the Neo4j graph.

    Creates:
    - StructuralEpoch nodes for each static tree version
    - DataNode nodes with applicability ranges (first_shot/last_shot)
    - INTRODUCED_IN / REMOVED_IN / AT_FACILITY relationships
    - Tag metadata on DataNodes

    Args:
        client: Neo4j GraphClient
        facility: Facility identifier
        data: Output from discover_static_tree()
        version_config: Version configs with first_shot info (from facility YAML).
            If None, loaded from facility config.
        dry_run: If True, log but don't write
        on_progress: Callback(written, total, detail) called after each
            batch write to report ingestion progress.

    Returns:
        Dict with counts: versions_created, nodes_created
    """
    data_source_name = data["data_source_name"]
    versions = data.get("versions", {})
    diff = data.get("diff", {})

    stats = {"versions_created": 0, "nodes_created": 0}

    if not versions:
        logger.warning("No version data to ingest")
        return stats

    # Load version config for first_shot mapping
    if version_config is None:
        configs = get_static_tree_config(facility)
        for cfg in configs:
            if cfg.get("source_name") == data_source_name:
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

    # Phase 1: Create StructuralEpoch nodes
    epoch_records = []
    for ver_str, ver_data in versions.items():
        if "error" in ver_data:
            continue
        ver_num = int(ver_str)
        epoch_id = f"{facility}:{data_source_name}:v{ver_num}"
        first_shot = first_shot_map.get(ver_num, ver_num)  # Fallback to version number
        last_shot = last_shot_map.get(ver_num)
        desc = description_map.get(ver_num, "")

        record = {
            "id": epoch_id,
            "facility_id": facility,
            "data_source_name": data_source_name,
            "version": ver_num,
            "first_shot": first_shot,
            "node_count": ver_data.get("node_count", 0),
            "fingerprint": f"static_v{ver_num}",
            "description": desc,
        }
        if last_shot is not None:
            record["last_shot"] = last_shot

        epoch_records.append(record)

    if dry_run:
        for rec in epoch_records:
            logger.info(
                "[DRY RUN] StructuralEpoch: %s (v%d, shots %d-%s) — %s",
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
                MERGE (v:StructuralEpoch {id: epoch.id})
                SET v += epoch
                WITH v, epoch
                MATCH (f:Facility {id: epoch.facility_id})
                MERGE (v)-[:AT_FACILITY]->(f)
                WITH v, epoch
                MERGE (t:DataSource {name: epoch.data_source_name})
                ON CREATE SET t.facility_id = epoch.facility_id
                MERGE (v)-[:IN_DATA_SOURCE]->(t)
                """,
                epochs=epoch_records,
            )
            # Create SUCCEEDS chain
            for i in range(1, len(epoch_records)):
                client.query(
                    """
                    MATCH (curr:StructuralEpoch {id: $curr_id})
                    MATCH (prev:StructuralEpoch {id: $prev_id})
                    MERGE (curr)-[:HAS_PREDECESSOR]->(prev)
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
        epoch_id = f"{facility}:{data_source_name}:v{ver_num}"

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
        epoch_id = f"{facility}:{data_source_name}:v{ver_num}"
        prev_first_shot = first_shot_map.get(ver_num, ver_num)
        for path in removed_paths:
            if path in all_paths and all_paths[path]["removed_version"] is None:
                all_paths[path]["last_shot"] = prev_first_shot - 1
                all_paths[path]["removed_version"] = epoch_id

    # Build DataNode records
    #
    # Build a fullpath → normalized-tag-path lookup first. MDSplus returns
    # tag-aliased paths (e.g. \STATIC::DBRDR_A_A) for nodes whose structural
    # fullpath is \STATIC::TOP.GREENS.GREEN_A_A:DBRDR. The tag path loses
    # parent hierarchy, so we need to map fullpath parents back to the matching
    # tag-path node that's actually stored in the graph.
    fullpath_to_tagpath: dict[str, str] = {}
    for path, info in all_paths.items():
        fullpath = info["node"].get("fullpath")
        normalized = normalize_mdsplus_path(path)
        if fullpath:
            fp_normalized = normalize_mdsplus_path(fullpath)
            fullpath_to_tagpath[fp_normalized] = normalized

    node_records = []
    for path, info in all_paths.items():
        node = info["node"]
        normalized = normalize_mdsplus_path(path)
        canonical = compute_canonical_path(path)
        node_id = f"{facility}:{data_source_name}:{normalized}"

        record: dict[str, Any] = {
            "id": node_id,
            "path": normalized,
            "canonical_path": canonical,
            "data_source_name": data_source_name,
            "facility_id": facility,
            "node_type": node.get("node_type", "STRUCTURE"),
            "first_shot": info["first_shot"],
            "introduced_version": info["introduced_version"],
            "source": "static_tree_extraction",
        }

        if info["last_shot"] is not None:
            record["last_shot"] = info["last_shot"]
        if info["removed_version"] is not None:
            record["removed_version"] = info["removed_version"]
        if node.get("units"):
            record["unit"] = node["units"]
        if node.get("description"):
            record["description"] = node["description"]
        if node.get("tags"):
            record["tags"] = node["tags"]

        # Parent path — try tag-path first, then resolve via fullpath.
        # Tag-aliased nodes like \STATIC::DBRDR_A_A have no hierarchy after ::
        # so _compute_parent_path returns None. In that case, use fullpath
        # (\STATIC::TOP.GREENS.GREEN_A_A:DBRDR) to find the structural parent
        # (\STATIC::TOP.GREENS.GREEN_A_A) and map it back to its tag path.
        parent = _compute_parent_path(normalized)
        if not parent:
            fullpath = node.get("fullpath")
            if fullpath:
                fp_normalized = normalize_mdsplus_path(fullpath)
                fp_parent = _compute_parent_path(fp_normalized)
                if fp_parent:
                    # Resolve the fullpath parent to its stored tag-path alias
                    parent = fullpath_to_tagpath.get(fp_parent, fp_parent)
        if parent:
            record["parent_path"] = parent

        node_records.append(record)

    if dry_run:
        logger.info("[DRY RUN] Would create %d DataNode records", len(node_records))
        # Show samples by node type
        by_type: dict[str, int] = {}
        for r in node_records:
            t = r.get("node_type", "STRUCTURE")
            by_type[t] = by_type.get(t, 0) + 1
        for t, count in sorted(by_type.items()):
            logger.info("  %s: %d nodes", t, count)
    else:
        # Batch insert DataNodes
        batch_size = 500
        total_records = len(node_records)
        written = 0
        for i in range(0, total_records, batch_size):
            batch = node_records[i : i + batch_size]
            client.query(
                """
                UNWIND $nodes AS node
                MERGE (n:DataNode {path: node.path, facility_id: node.facility_id})
                SET n.id = node.id,
                    n.data_source_name = node.data_source_name,
                    n.canonical_path = node.canonical_path,
                    n.parent_path = node.parent_path,
                    n.first_shot = node.first_shot,
                    n.last_shot = node.last_shot,
                    n.introduced_version = node.introduced_version,
                    n.removed_version = node.removed_version,
                    n.node_type = node.node_type,
                    n.source = node.source,
                    n.unit = node.unit,
                    n.description = node.description,
                    n.tags = node.tags
                """,
                nodes=batch,
            )
            written += len(batch)
            if on_progress:
                batch_num = i // batch_size + 1
                total_batches = (total_records + batch_size - 1) // batch_size
                on_progress(
                    written,
                    total_records,
                    f"MERGE batch {batch_num}/{total_batches}",
                )

        # Create relationships
        if on_progress:
            on_progress(written, total_records, "creating relationships...")
        client.query(
            """
            MATCH (n:DataNode)
            WHERE n.data_source_name = $data_source_name AND n.facility_id = $facility
            WITH n
            MATCH (f:Facility {id: $facility})
            MERGE (n)-[:AT_FACILITY]->(f)
            WITH n
            MERGE (t:DataSource {name: $data_source_name})
            ON CREATE SET t.facility_id = $facility
            MERGE (n)-[:IN_DATA_SOURCE]->(t)
            """,
            data_source_name=data_source_name,
            facility=facility,
        )

        client.query(
            """
            MATCH (n:DataNode)
            WHERE n.data_source_name = $data_source_name AND n.facility_id = $facility
              AND n.introduced_version IS NOT NULL
            WITH n
            MATCH (v:StructuralEpoch {id: n.introduced_version})
            MERGE (n)-[:INTRODUCED_IN]->(v)
            """,
            data_source_name=data_source_name,
            facility=facility,
        )

        client.query(
            """
            MATCH (n:DataNode)
            WHERE n.data_source_name = $data_source_name AND n.facility_id = $facility
              AND n.removed_version IS NOT NULL
            WITH n
            MATCH (v:StructuralEpoch {id: n.removed_version})
            MERGE (n)-[:REMOVED_IN]->(v)
            """,
            data_source_name=data_source_name,
            facility=facility,
        )

        # Parent-child relationships
        client.query(
            """
            MATCH (child:DataNode)
            WHERE child.data_source_name = $data_source_name AND child.facility_id = $facility
              AND child.parent_path IS NOT NULL
            WITH child
            MATCH (parent:DataNode {path: child.parent_path, facility_id: $facility})
            WHERE parent.data_source_name = $data_source_name
            MERGE (parent)-[:HAS_NODE]->(child)
            """,
            data_source_name=data_source_name,
            facility=facility,
        )

    stats["nodes_created"] = len(node_records)

    logger.info(
        "Static tree %s:%s — %d versions, %d nodes",
        facility,
        data_source_name,
        stats["versions_created"],
        stats["nodes_created"],
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
