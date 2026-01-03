#!/usr/bin/env python3
"""
Build the IMAS Data Dictionary Knowledge Graph.

This script populates Neo4j with IMAS DD structure including:
- DDVersion nodes for all available DD versions
- IDS nodes for top-level structures
- DDPath nodes with hierarchical relationships
- Unit and CoordinateSpec nodes
- Version tracking (INTRODUCED_IN, DEPRECATED_IN, RENAMED_TO)
- PathChange nodes for metadata changes
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

    # Create IDS and DDPath nodes, tracking version changes
    logger.info("Creating IDS and DDPath nodes with version tracking...")
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
            # Create/update IDS nodes
            for ids_name, ids_info in data["ids_info"].items():
                _upsert_ids_node(client, ids_name, ids_info, version, i == 0)
            stats["ids_created"] = max(stats["ids_created"], len(data["ids_info"]))

            # Create DDPath nodes for new paths
            for path in changes["added"]:
                path_info = data["paths"][path]
                _create_path_node(client, path, path_info, version)
            stats["paths_created"] += len(changes["added"])

            # Mark deprecated paths
            for path in changes["removed"]:
                _mark_path_deprecated(client, path, version)

            # Create PathChange nodes for metadata changes
            for path, path_changes in changes["changed"].items():
                for change in path_changes:
                    _create_path_change(client, path, version, change)
                    stats["path_changes_created"] += 1

        prev_paths = data["paths"]
        progress.update_progress(version)

    progress.finish_processing()

    # Create RENAMED_TO relationships from path mappings
    logger.info("Creating RENAMED_TO relationships...")
    if not dry_run:
        mappings = load_path_mappings(current_dd_version)
        for old_path, mapping in mappings.get("old_to_new", {}).items():
            new_path = mapping.get("new_path")
            if new_path:
                _create_renamed_to(client, old_path, new_path)

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
    """Create DDVersion nodes with predecessor chain."""
    # Sort versions to ensure proper ordering
    sorted_versions = sorted(versions)

    for i, version in enumerate(sorted_versions):
        predecessor = sorted_versions[i - 1] if i > 0 else None
        is_current = version == current_dd_version

        client.query(
            """
            MERGE (v:DDVersion {id: $version})
            SET v.is_current = $is_current,
                v.created_at = datetime()
            WITH v
            CALL {
                WITH v
                MATCH (prev:DDVersion {id: $predecessor})
                MERGE (v)-[:PREDECESSOR]->(prev)
            }
            RETURN v
        """,
            version=version,
            predecessor=predecessor,
            is_current=is_current,
        )


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


def _upsert_ids_node(
    client: GraphClient,
    ids_name: str,
    ids_info: dict,
    version: str,
    is_first_version: bool,
) -> None:
    """Create or update IDS node."""
    client.query(
        """
        MERGE (ids:IDS {name: $name})
        SET ids.description = $description,
            ids.physics_domain = $physics_domain,
            ids.path_count = $path_count,
            ids.leaf_count = $leaf_count
        WITH ids
        MATCH (v:DDVersion {id: $version})
        // Only set introduced_version if this is first version or IDS is new
        FOREACH (_ IN CASE WHEN $is_first THEN [1] ELSE [] END |
            MERGE (ids)-[:INTRODUCED_IN]->(v)
        )
    """,
        name=ids_name,
        description=ids_info.get("description", ""),
        physics_domain=ids_info.get("physics_domain", "general"),
        path_count=ids_info.get("path_count", 0),
        leaf_count=ids_info.get("leaf_count", 0),
        version=version,
        is_first=is_first_version,
    )


def _create_path_node(
    client: GraphClient,
    path: str,
    path_info: dict,
    version: str,
) -> None:
    """Create DDPath node with relationships."""
    # Extract IDS name from path
    ids_name = path.split("/")[0]

    # Determine physics domain
    physics_domain = physics_categorizer.get_domain_for_ids(ids_name).value

    client.query(
        """
        MERGE (p:DDPath {id: $path})
        SET p.name = $name,
            p.documentation = $documentation,
            p.data_type = $data_type,
            p.ndim = $ndim,
            p.node_type = $node_type,
            p.physics_domain = $physics_domain,
            p.maxoccur = $maxoccur
        WITH p

        // Link to IDS
        MATCH (ids:IDS {name: $ids_name})
        MERGE (p)-[:IDS]->(ids)
        WITH p

        // Link to parent path
        FOREACH (_ IN CASE WHEN $parent_path IS NOT NULL AND $parent_path <> $ids_name THEN [1] ELSE [] END |
            MERGE (parent:DDPath {id: $parent_path})
            MERGE (p)-[:PARENT]->(parent)
        )
        WITH p

        // Link to unit
        FOREACH (_ IN CASE WHEN $units IS NOT NULL AND $units <> '' THEN [1] ELSE [] END |
            MERGE (u:Unit {symbol: $units})
            MERGE (p)-[:HAS_UNIT]->(u)
        )
        WITH p

        // Link to introduced version
        MATCH (v:DDVersion {id: $version})
        MERGE (p)-[:INTRODUCED_IN]->(v)
    """,
        path=path,
        name=path_info.get("name", ""),
        documentation=path_info.get("documentation", ""),
        data_type=path_info.get("data_type"),
        ndim=path_info.get("ndim", 0),
        node_type=path_info.get("node_type"),
        physics_domain=physics_domain,
        maxoccur=path_info.get("maxoccur"),
        ids_name=ids_name,
        parent_path=path_info.get("parent_path"),
        units=path_info.get("units", ""),
        version=version,
    )


def _mark_path_deprecated(client: GraphClient, path: str, version: str) -> None:
    """Mark a path as deprecated in a specific version."""
    client.query(
        """
        MATCH (p:DDPath {id: $path})
        MATCH (v:DDVersion {id: $version})
        MERGE (p)-[:DEPRECATED_IN]->(v)
    """,
        path=path,
        version=version,
    )


def _create_path_change(
    client: GraphClient,
    path: str,
    version: str,
    change: dict,
) -> None:
    """Create a PathChange node for metadata changes."""
    change_id = f"{path}:{change['field']}:{version}"

    client.query(
        """
        MERGE (c:PathChange {id: $change_id})
        SET c.change_type = $change_type,
            c.old_value = $old_value,
            c.new_value = $new_value
        WITH c
        MATCH (p:DDPath {id: $path})
        MATCH (v:DDVersion {id: $version})
        MERGE (c)-[:PATH]->(p)
        MERGE (c)-[:VERSION]->(v)
    """,
        change_id=change_id,
        change_type=change["field"],
        old_value=change.get("old_value", ""),
        new_value=change.get("new_value", ""),
        path=path,
        version=version,
    )


def _create_renamed_to(client: GraphClient, old_path: str, new_path: str) -> None:
    """Create RENAMED_TO relationship between paths."""
    client.query(
        """
        MATCH (old:DDPath {id: $old_path})
        MATCH (new:DDPath {id: $new_path})
        MERGE (old)-[:RENAMED_TO]->(new)
    """,
        old_path=old_path,
        new_path=new_path,
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
                    MATCH (p:DDPath {id: $path})
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
        click.echo(f"DDPath nodes created: {stats['paths_created']}")
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
