#!/usr/bin/env python3
"""
Build the IMAS Data Dictionary Knowledge Graph.

This script populates Neo4j with IMAS DD structure including:
- DDVersion nodes for all available DD versions
- IDS nodes for top-level structures
- IMASPath nodes with hierarchical relationships
- Unit and CoordinateSpec nodes
- Version tracking (INTRODUCED_IN, DEPRECATED_IN, RENAMED_TO)
- PathChange nodes for metadata changes with semantic classification
- SemanticCluster nodes from existing clusters (optional)

The graph is augmented incrementally, not rebuilt - this preserves
links to facility data (TreeNodes, IMASMappings).
"""

import json
import logging
import sys

import click
import imas

from imas_codex import dd_version as current_dd_version
from imas_codex.core.physics_categorization import physics_categorizer
from imas_codex.core.progress_monitor import create_progress_monitor
from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# Keywords for semantic classification of documentation changes
SIGN_CONVENTION_KEYWORDS = [
    "upwards",
    "downwards",
    "increasing",
    "decreasing",
    "positive",
    "negative",
    "clockwise",
    "anti-clockwise",
    "normal vector",
    "surface normal",
    "sign convention",
]

COORDINATE_CONVENTION_KEYWORDS = [
    "right-handed",
    "left-handed",
    "coordinate system",
    "reference frame",
]


def classify_doc_change(old_doc: str, new_doc: str) -> tuple[str, list[str]]:
    """
    Classify a documentation change by physics significance.

    Returns:
        Tuple of (semantic_type, keywords_detected)
    """
    old_lower = old_doc.lower() if old_doc else ""
    new_lower = new_doc.lower() if new_doc else ""

    keywords_found = []

    # Check for new sign convention text
    for kw in SIGN_CONVENTION_KEYWORDS:
        if kw in new_lower and kw not in old_lower:
            keywords_found.append(kw)

    if keywords_found:
        return "sign_convention", keywords_found

    # Check for coordinate convention changes
    for kw in COORDINATE_CONVENTION_KEYWORDS:
        if kw in new_lower and kw not in old_lower:
            keywords_found.append(kw)

    if keywords_found:
        return "coordinate_convention", keywords_found

    # Check if documentation was significantly expanded (clarification)
    if new_doc and old_doc and len(new_doc) > len(old_doc) * 1.5:
        return "definition_clarification", []

    return "none", []


def get_all_dd_versions() -> list[str]:
    """Get all available DD versions, sorted."""
    versions = imas.dd_zip.dd_xml_versions()
    return sorted(versions)


def extract_paths_for_version(version: str, ids_filter: set[str] | None = None) -> dict:
    """
    Extract all paths from a DD version.

    Returns:
        Dict with structure:
        {
            "ids_info": {ids_name: {description, path_count, ...}},
            "paths": {full_path: {name, data_type, units, ...}},
            "units": set of unit strings
        }
    """
    factory = imas.IDSFactory(version)
    ids_names = set(factory.ids_names())

    if ids_filter:
        ids_names = ids_names & ids_filter

    ids_info = {}
    paths = {}
    units = set()

    for ids_name in sorted(ids_names):
        try:
            ids_def = factory.new(ids_name)
            metadata = ids_def.metadata

            # IDS-level info
            ids_info[ids_name] = {
                "name": ids_name,
                "description": metadata.documentation or "",
                "physics_domain": physics_categorizer.get_domain_for_ids(
                    ids_name
                ).value,
            }

            # Extract all paths recursively
            _extract_paths_recursive(metadata, ids_name, "", paths, units)

        except Exception as e:
            logger.warning(f"Error processing {ids_name} in {version}: {e}")

    # Compute IDS stats
    for ids_name in ids_info:
        ids_paths = [p for p in paths if p.startswith(f"{ids_name}/")]
        ids_info[ids_name]["path_count"] = len(ids_paths)
        ids_info[ids_name]["leaf_count"] = sum(
            1
            for p in ids_paths
            if paths[p].get("data_type") not in ("STRUCTURE", "STRUCT_ARRAY")
        )

    return {
        "ids_info": ids_info,
        "paths": paths,
        "units": units,
    }


def _extract_paths_recursive(
    metadata,
    ids_name: str,
    prefix: str,
    paths: dict,
    units: set,
) -> None:
    """Recursively extract paths from IDS metadata."""
    for child in metadata:
        child_path = f"{prefix}{child.name}" if prefix else child.name
        full_path = f"{ids_name}/{child_path}"

        # Extract path info
        path_info = {
            "name": child.name,
            "documentation": child.documentation or "",
            "units": child.units or "",
            "data_type": f"{child.data_type.name}_{child.ndim}D"
            if child.data_type
            else None,
            "ndim": child.ndim,
            "node_type": child.type.name.lower() if child.type.name else None,
            "maxoccur": child.maxoccur,
            "parent_path": f"{ids_name}/{prefix.rstrip('/')}" if prefix else ids_name,
        }

        # Handle structure types
        if child.data_type:
            if child.data_type.name == "STRUCTURE":
                path_info["data_type"] = "STRUCTURE"
            elif child.data_type.name == "STRUCT_ARRAY":
                path_info["data_type"] = "STRUCT_ARRAY"

        # Collect coordinates
        if child.ndim > 0:
            coordinates = []
            for coord in child.coordinates:
                coord_str = str(coord) if coord else ""
                coordinates.append(coord_str)
            path_info["coordinates"] = coordinates

        # Track units
        if child.units and child.units not in ("", "as_parent"):
            units.add(child.units)

        paths[full_path] = path_info

        # Recurse into children (structures and struct arrays)
        if child.data_type and child.data_type.name in ("STRUCTURE", "STRUCT_ARRAY"):
            _extract_paths_recursive(child, ids_name, f"{child_path}/", paths, units)


def compute_version_changes(
    old_paths: dict[str, dict],
    new_paths: dict[str, dict],
) -> dict:
    """
    Compute changes between two versions.

    Returns:
        Dict with:
        - added: paths in new but not old
        - removed: paths in old but not new
        - changed: paths with metadata changes
    """
    old_set = set(old_paths.keys())
    new_set = set(new_paths.keys())

    added = new_set - old_set
    removed = old_set - new_set
    common = old_set & new_set

    # Check for metadata changes in common paths
    changed = {}
    for path in common:
        old_info = old_paths[path]
        new_info = new_paths[path]

        changes = []
        for field in ("units", "documentation", "data_type", "node_type"):
            old_val = old_info.get(field, "")
            new_val = new_info.get(field, "")
            if old_val != new_val:
                changes.append(
                    {
                        "field": field,
                        "old_value": str(old_val) if old_val else "",
                        "new_value": str(new_val) if new_val else "",
                    }
                )

        if changes:
            changed[path] = changes

    return {
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def load_path_mappings(version: str) -> dict:
    """Load path mappings from build_path_map output if available."""
    from imas_codex.resource_path_accessor import ResourcePathAccessor

    try:
        path_accessor = ResourcePathAccessor(dd_version=version)
        mapping_file = path_accessor.mappings_dir / "path_mappings.json"

        if mapping_file.exists():
            with open(mapping_file) as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Could not load path mappings: {e}")

    return {"old_to_new": {}, "new_to_old": {}}


def build_dd_graph(
    client: GraphClient,
    versions: list[str] | None = None,
    include_clusters: bool = False,
    dry_run: bool = False,
    ids_filter: set[str] | None = None,
    use_rich: bool | None = None,
) -> dict:
    """
    Build the IMAS DD graph.

    Args:
        client: Neo4j GraphClient
        versions: List of versions to process (None = all available)
        ids_filter: Optional set of IDS names to include
        include_clusters: Whether to import semantic clusters
        dry_run: If True, don't write to graph
        use_rich: Force rich progress (True), logging (False), or auto (None)

    Returns:
        Statistics about the build
    """
    all_versions = get_all_dd_versions()

    if versions is None:
        versions = all_versions
    else:
        # Validate versions
        for v in versions:
            if v not in all_versions:
                raise ValueError(f"Unknown DD version: {v}")

    logger.info(f"Building DD graph for {len(versions)} versions")
    if ids_filter:
        logger.info(f"Filtering to IDS: {sorted(ids_filter)}")

    # Create progress monitor
    progress = create_progress_monitor(
        use_rich=use_rich,
        logger=logger,
        item_names=versions,
        description_template="Processing: {item}",
    )

    stats = {
        "versions_processed": 0,
        "ids_created": 0,
        "paths_created": 0,
        "units_created": 0,
        "path_changes_created": 0,
        "clusters_created": 0,
    }

    # Ensure indexes exist for performance
    if not dry_run:
        _ensure_indexes(client)

    # First pass: create DDVersion nodes with predecessor chain
    logger.info("Creating DDVersion nodes...")
    if not dry_run:
        _create_version_nodes(client, versions)
    stats["versions_processed"] = len(versions)

    # Create Unit nodes (collect all unique units first)
    logger.info("Collecting units across all versions...")
    all_units: set[str] = set()
    version_data: dict[str, dict] = {}

    progress.start_processing(versions, "Extracting paths")
    for version in versions:
        progress.set_current_item(version)
        try:
            data = extract_paths_for_version(version, ids_filter=ids_filter)
            version_data[version] = data
            all_units.update(data["units"])
        except Exception as e:
            logger.error(f"Error extracting {version}: {e}")
            progress.update_progress(version, error=str(e))
            continue
        progress.update_progress(version)
    progress.finish_processing()

    # Create Unit nodes
    logger.info(f"Creating {len(all_units)} Unit nodes...")
    if not dry_run:
        _create_unit_nodes(client, all_units)
    stats["units_created"] = len(all_units)

    # Create CoordinateSpec nodes for index-based coordinates
    logger.info("Creating CoordinateSpec nodes...")
    coord_specs = {"1...N", "1...1", "1...2", "1...3", "1...4", "1...6"}
    if not dry_run:
        _create_coordinate_spec_nodes(client, coord_specs)

    # Create IDS and IMASPath nodes, tracking version changes
    logger.info("Creating IDS and IMASPath nodes with version tracking...")
    prev_paths: dict[str, dict] = {}

    progress.start_processing(versions, "Building graph")
    for i, version in enumerate(versions):
        progress.set_current_item(version)

        if version not in version_data:
            progress.update_progress(version, error="No data")
            continue

        data = version_data[version]

        # Compute changes from previous version
        changes = compute_version_changes(prev_paths, data["paths"])

        if not dry_run:
            # Batch create/update IDS nodes
            _batch_upsert_ids_nodes(client, data["ids_info"], version, i == 0)
            stats["ids_created"] = max(stats["ids_created"], len(data["ids_info"]))

            # Batch create IMASPath nodes for new paths
            new_paths_data = {p: data["paths"][p] for p in changes["added"]}
            _batch_create_path_nodes(client, new_paths_data, version)
            stats["paths_created"] += len(changes["added"])

            # Batch mark deprecated paths
            _batch_mark_paths_deprecated(client, changes["removed"], version)

            # Batch create PathChange nodes for metadata changes
            change_count = _batch_create_path_changes(
                client, changes["changed"], version
            )
            stats["path_changes_created"] += change_count

        prev_paths = data["paths"]
        progress.update_progress(version)

    progress.finish_processing()

    # Batch create RENAMED_TO relationships from path mappings
    logger.info("Creating RENAMED_TO relationships...")
    if not dry_run:
        mappings = load_path_mappings(current_dd_version)
        _batch_create_renamed_to(client, mappings.get("old_to_new", {}))

    # Import semantic clusters if requested
    if include_clusters:
        logger.info("Importing semantic clusters...")
        cluster_count = _import_clusters(client, dry_run)
        stats["clusters_created"] = cluster_count

    # Update graph metadata
    if not dry_run:
        client.query(
            """
            MERGE (m:_GraphMeta)
            SET m.dd_graph_built = datetime(),
                m.dd_versions = $versions,
                m.dd_current_version = $current
        """,
            versions=versions,
            current=current_dd_version,
        )

    return stats


def _create_version_nodes(client: GraphClient, versions: list[str]) -> None:
    """Create DDVersion nodes with predecessor chain using batch operations."""
    sorted_versions = sorted(versions)

    # Build version data with predecessors
    version_data = []
    for i, version in enumerate(sorted_versions):
        version_data.append(
            {
                "id": version,
                "predecessor": sorted_versions[i - 1] if i > 0 else None,
                "is_current": version == current_dd_version,
            }
        )

    # Batch create all version nodes
    client.query(
        """
        UNWIND $versions AS v
        MERGE (ver:DDVersion {id: v.id})
        SET ver.is_current = v.is_current,
            ver.created_at = datetime()
    """,
        versions=version_data,
    )

    # Batch create predecessor relationships
    predecessors = [v for v in version_data if v["predecessor"] is not None]
    if predecessors:
        client.query(
            """
            UNWIND $versions AS v
            MATCH (ver:DDVersion {id: v.id})
            MATCH (prev:DDVersion {id: v.predecessor})
            MERGE (ver)-[:PREDECESSOR]->(prev)
        """,
            versions=predecessors,
        )


def _ensure_indexes(client: GraphClient) -> None:
    """Ensure required indexes exist for optimal query performance."""
    logger.debug("Ensuring DD indexes exist...")

    # IMASPath.id - critical for MERGE and relationship creation
    client.query("CREATE INDEX imaspath_id IF NOT EXISTS FOR (p:IMASPath) ON (p.id)")

    # IDS.name - for IDS relationship lookups
    client.query("CREATE INDEX ids_name IF NOT EXISTS FOR (i:IDS) ON (i.name)")

    # DDVersion.id - for version relationship lookups
    client.query("CREATE INDEX ddversion_id IF NOT EXISTS FOR (v:DDVersion) ON (v.id)")

    # Unit.symbol - for unit relationship lookups
    client.query("CREATE INDEX unit_symbol IF NOT EXISTS FOR (u:Unit) ON (u.symbol)")


def _create_unit_nodes(client: GraphClient, units: set[str]) -> None:
    """Create Unit nodes."""
    unit_list = [{"symbol": u} for u in units if u]
    if unit_list:
        client.query(
            """
            UNWIND $units AS unit
            MERGE (u:Unit {symbol: unit.symbol})
        """,
            units=unit_list,
        )


def _create_coordinate_spec_nodes(client: GraphClient, specs: set[str]) -> None:
    """Create CoordinateSpec nodes for index-based coordinates."""
    spec_list = []
    for spec in specs:
        is_bounded = spec != "1...N"
        max_size = None
        if is_bounded:
            try:
                max_size = int(spec.split("...")[1])
            except (IndexError, ValueError):
                pass
        spec_list.append(
            {
                "id": spec,
                "is_bounded": is_bounded,
                "max_size": max_size,
            }
        )

    if spec_list:
        client.query(
            """
            UNWIND $specs AS spec
            MERGE (c:CoordinateSpec {id: spec.id})
            SET c.is_bounded = spec.is_bounded,
                c.max_size = spec.max_size
        """,
            specs=spec_list,
        )


def _batch_upsert_ids_nodes(
    client: GraphClient,
    ids_info_map: dict[str, dict],
    version: str,
    is_first_version: bool,
) -> None:
    """Batch create or update IDS nodes."""
    ids_list = [
        {
            "name": ids_name,
            "description": info.get("description", ""),
            "physics_domain": info.get("physics_domain", "general"),
            "path_count": info.get("path_count", 0),
            "leaf_count": info.get("leaf_count", 0),
        }
        for ids_name, info in ids_info_map.items()
    ]

    if not ids_list:
        return

    # Batch create/update IDS nodes
    client.query(
        """
        UNWIND $ids_list AS ids_data
        MERGE (ids:IDS {name: ids_data.name})
        SET ids.description = ids_data.description,
            ids.physics_domain = ids_data.physics_domain,
            ids.path_count = ids_data.path_count,
            ids.leaf_count = ids_data.leaf_count
    """,
        ids_list=ids_list,
    )

    # Batch create INTRODUCED_IN relationships for first version
    if is_first_version:
        client.query(
            """
            UNWIND $ids_list AS ids_data
            MATCH (ids:IDS {name: ids_data.name})
            MATCH (v:DDVersion {id: $version})
            MERGE (ids)-[:INTRODUCED_IN]->(v)
        """,
            ids_list=ids_list,
            version=version,
        )


def _batch_create_path_nodes(
    client: GraphClient,
    paths_data: dict[str, dict],
    version: str,
    batch_size: int = 1000,
) -> None:
    """Batch create IMASPath nodes with relationships.

    Uses multiple batched queries to avoid memory issues with large datasets.
    """
    # Prepare path data for batch insertion
    path_list = []
    for path, path_info in paths_data.items():
        ids_name = path.split("/")[0]
        physics_domain = physics_categorizer.get_domain_for_ids(ids_name).value

        path_list.append(
            {
                "id": path,
                "name": path_info.get("name", ""),
                "documentation": path_info.get("documentation", ""),
                "data_type": path_info.get("data_type"),
                "ndim": path_info.get("ndim", 0),
                "node_type": path_info.get("node_type"),
                "physics_domain": physics_domain,
                "maxoccur": path_info.get("maxoccur"),
                "ids_name": ids_name,
                "parent_path": path_info.get("parent_path"),
                "units": path_info.get("units", ""),
            }
        )

    if not path_list:
        return

    # Process in batches to avoid memory issues
    for i in range(0, len(path_list), batch_size):
        batch = path_list[i : i + batch_size]

        # Step 1: Create IMASPath nodes
        client.query(
            """
            UNWIND $paths AS p
            MERGE (path:IMASPath {id: p.id})
            SET path.name = p.name,
                path.documentation = p.documentation,
                path.data_type = p.data_type,
                path.ndim = p.ndim,
                path.node_type = p.node_type,
                path.physics_domain = p.physics_domain,
                path.maxoccur = p.maxoccur
        """,
            paths=batch,
        )

        # Step 2: Create IDS relationships
        client.query(
            """
            UNWIND $paths AS p
            MATCH (path:IMASPath {id: p.id})
            MATCH (ids:IDS {name: p.ids_name})
            MERGE (path)-[:IDS]->(ids)
        """,
            paths=batch,
        )

        # Step 3: Create PARENT relationships (filter out root-level paths)
        parent_paths = [
            p for p in batch if p["parent_path"] and p["parent_path"] != p["ids_name"]
        ]
        if parent_paths:
            client.query(
                """
                UNWIND $paths AS p
                MATCH (path:IMASPath {id: p.id})
                MERGE (parent:IMASPath {id: p.parent_path})
                MERGE (path)-[:PARENT]->(parent)
            """,
                paths=parent_paths,
            )

        # Step 4: Create HAS_UNIT relationships (filter out empty units)
        unit_paths = [p for p in batch if p["units"] and p["units"] != ""]
        if unit_paths:
            client.query(
                """
                UNWIND $paths AS p
                MATCH (path:IMASPath {id: p.id})
                MATCH (u:Unit {symbol: p.units})
                MERGE (path)-[:HAS_UNIT]->(u)
            """,
                paths=unit_paths,
            )

        # Step 5: Create INTRODUCED_IN relationships
        client.query(
            """
            UNWIND $paths AS p
            MATCH (path:IMASPath {id: p.id})
            MATCH (v:DDVersion {id: $version})
            MERGE (path)-[:INTRODUCED_IN]->(v)
        """,
            paths=batch,
            version=version,
        )


def _batch_mark_paths_deprecated(
    client: GraphClient, paths: set[str], version: str
) -> None:
    """Batch mark paths as deprecated in a specific version."""
    if not paths:
        return

    path_list = [{"path": p} for p in paths]
    client.query(
        """
        UNWIND $paths AS p
        MATCH (path:IMASPath {id: p.path})
        MATCH (v:DDVersion {id: $version})
        MERGE (path)-[:DEPRECATED_IN]->(v)
    """,
        paths=path_list,
        version=version,
    )


def _batch_create_path_changes(
    client: GraphClient,
    changes: dict[str, list[dict]],
    version: str,
) -> int:
    """Batch create PathChange nodes for metadata changes with semantic classification."""
    if not changes:
        return 0

    change_list = []
    for path, path_changes in changes.items():
        for change in path_changes:
            change_data = {
                "id": f"{path}:{change['field']}:{version}",
                "path": path,
                "change_type": change["field"],
                "old_value": change.get("old_value", ""),
                "new_value": change.get("new_value", ""),
                "semantic_type": None,
                "keywords_detected": None,
            }

            # Classify documentation changes
            if change["field"] == "documentation":
                semantic_type, keywords = classify_doc_change(
                    change.get("old_value", ""),
                    change.get("new_value", ""),
                )
                change_data["semantic_type"] = semantic_type
                if keywords:
                    change_data["keywords_detected"] = json.dumps(keywords)

            change_list.append(change_data)

    if not change_list:
        return 0

    # Create PathChange nodes with semantic classification
    client.query(
        """
        UNWIND $changes AS c
        MERGE (change:PathChange {id: c.id})
        SET change.change_type = c.change_type,
            change.old_value = c.old_value,
            change.new_value = c.new_value,
            change.semantic_type = c.semantic_type,
            change.keywords_detected = c.keywords_detected
    """,
        changes=change_list,
    )

    # Create relationships
    client.query(
        """
        UNWIND $changes AS c
        MATCH (change:PathChange {id: c.id})
        MATCH (p:IMASPath {id: c.path})
        MATCH (v:DDVersion {id: $version})
        MERGE (change)-[:PATH]->(p)
        MERGE (change)-[:VERSION]->(v)
    """,
        changes=change_list,
        version=version,
    )

    return len(change_list)


def _batch_create_renamed_to(client: GraphClient, mappings: dict[str, dict]) -> None:
    """Batch create RENAMED_TO relationships between paths."""
    if not mappings:
        return

    rename_list = []
    for old_path, mapping in mappings.items():
        new_path = mapping.get("new_path")
        if new_path:
            rename_list.append({"old_path": old_path, "new_path": new_path})

    if rename_list:
        client.query(
            """
            UNWIND $renames AS r
            MATCH (old:IMASPath {id: r.old_path})
            MATCH (new:IMASPath {id: r.new_path})
            MERGE (old)-[:RENAMED_TO]->(new)
        """,
            renames=rename_list,
        )


def _import_clusters(client: GraphClient, dry_run: bool) -> int:
    """Import semantic clusters from existing cluster data."""
    try:
        from imas_codex.core.clusters import Clusters

        clusters_manager = Clusters()
        if not clusters_manager.is_available():
            logger.warning("Cluster data not available")
            return 0

        cluster_data = clusters_manager.get_clusters()
        cluster_count = 0

        for cluster_id, cluster in cluster_data.items():
            if dry_run:
                cluster_count += 1
                continue

            # Create SemanticCluster node
            label = cluster.get("label", f"cluster_{cluster_id}")
            physics_domain = cluster.get("physics_domain", "general")
            paths = cluster.get("paths", [])
            cross_ids = cluster.get("cross_ids", False)

            client.query(
                """
                MERGE (c:SemanticCluster {id: $cluster_id})
                SET c.label = $label,
                    c.physics_domain = $physics_domain,
                    c.path_count = $path_count,
                    c.cross_ids = $cross_ids
            """,
                cluster_id=str(cluster_id),
                label=label,
                physics_domain=physics_domain,
                path_count=len(paths),
                cross_ids=cross_ids,
            )

            # Create IN_CLUSTER relationships
            for path_info in paths:
                if isinstance(path_info, dict):
                    path = path_info.get("path", "")
                    distance = path_info.get("distance", 0.0)
                else:
                    path = str(path_info)
                    distance = 0.0

                client.query(
                    """
                    MATCH (p:IMASPath {id: $path})
                    MATCH (c:SemanticCluster {id: $cluster_id})
                    MERGE (p)-[r:IN_CLUSTER]->(c)
                    SET r.distance = $distance
                """,
                    path=path,
                    cluster_id=str(cluster_id),
                    distance=distance,
                )

            cluster_count += 1

        return cluster_count

    except Exception as e:
        logger.error(f"Error importing clusters: {e}")
        return 0


@click.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all logging except errors")
@click.option(
    "--all-versions",
    is_flag=True,
    help="Process all available DD versions (default: current only)",
)
@click.option(
    "--from-version",
    type=str,
    help="Start from a specific version (for incremental updates)",
)
@click.option(
    "--ids-filter",
    type=str,
    help="Specific IDS names to include as a space-separated string (e.g., 'equilibrium core_profiles')",
)
@click.option(
    "--include-clusters",
    is_flag=True,
    help="Import semantic clusters into graph",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview changes without writing to graph",
)
@click.option(
    "--no-rich",
    is_flag=True,
    help="Disable rich progress bar, use plain logging",
)
def build_dd_graph_cli(
    verbose: bool,
    quiet: bool,
    all_versions: bool,
    from_version: str | None,
    ids_filter: str | None,
    include_clusters: bool,
    dry_run: bool,
    no_rich: bool,
) -> int:
    """Build the IMAS Data Dictionary Knowledge Graph.

    This command populates Neo4j with IMAS DD structure including version
    tracking, path hierarchy, units, and optionally semantic clusters.

    Examples:
        build-dd-graph                        # Build current version only
        build-dd-graph --all-versions         # Build all 34 versions
        build-dd-graph --from-version 4.0.0   # Incremental from 4.0.0
        build-dd-graph --include-clusters     # Include semantic clusters
        build-dd-graph --dry-run -v           # Preview without writing
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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress imas library's verbose logging
    logging.getLogger("imas").setLevel(logging.WARNING)

    try:
        # Determine versions to process
        available_versions = get_all_dd_versions()

        if all_versions:
            versions = available_versions
        elif from_version:
            # Get all versions from the specified one onwards
            try:
                start_idx = available_versions.index(from_version)
                versions = available_versions[start_idx:]
            except ValueError:
                click.echo(f"Error: Unknown version {from_version}", err=True)
                click.echo(
                    f"Available: {', '.join(available_versions[:5])}...", err=True
                )
                return 1
        else:
            # Just current version
            versions = [current_dd_version]

        logger.info(f"Processing {len(versions)} DD versions")
        logger.info(f"Versions: {versions[0]} → {versions[-1]}")

        # Parse IDS filter if provided
        ids_set: set[str] | None = None
        if ids_filter:
            ids_set = set(ids_filter.split())
            logger.info(f"Filtering to IDS: {sorted(ids_set)}")

        if dry_run:
            click.echo("DRY RUN - no changes will be written to graph")

        # Determine rich usage
        use_rich: bool | None = None
        if no_rich:
            use_rich = False

        # Build graph
        with GraphClient() as client:
            stats = build_dd_graph(
                client=client,
                versions=versions,
                ids_filter=ids_set,
                include_clusters=include_clusters,
                dry_run=dry_run,
                use_rich=use_rich,
            )

        # Report results
        click.echo("\n=== Build Complete ===")
        click.echo(f"Versions processed: {stats['versions_processed']}")
        click.echo(f"IDS nodes: {stats['ids_created']}")
        click.echo(f"IMASPath nodes created: {stats['paths_created']}")
        click.echo(f"Unit nodes: {stats['units_created']}")
        click.echo(f"PathChange nodes: {stats['path_changes_created']}")
        if include_clusters:
            click.echo(f"Cluster nodes: {stats['clusters_created']}")

        return 0

    except Exception as e:
        logger.error(f"Error building DD graph: {e}")
        if verbose:
            logger.exception("Full traceback:")
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(build_dd_graph_cli())
