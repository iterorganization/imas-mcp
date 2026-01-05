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
- Embeddings for IMASPath nodes with content-based caching

The graph is augmented incrementally, not rebuilt - this preserves
links to facility data (TreeNodes, IMASMappings).

Embedding Pipeline:
- Generates enriched_text by concatenating IDS context, documentation, units
- Computes SHA256 hash of enriched_text for cache busting
- Only regenerates embeddings for paths where hash changed
- Stores embedding vectors in Neo4j for vector search
"""

import hashlib
import json
import logging
import sys

import click
import imas
import numpy as np

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


# =============================================================================
# Embedding Generation Functions
# =============================================================================


def generate_embedding_text(
    path: str,
    path_info: dict,
    ids_info: dict | None = None,
) -> str:
    """
    Generate embedding text by concatenating contextual information.

    Produces coherent prose describing the data path, its physics context,
    and measurement characteristics for better semantic clustering. Uses
    natural language formatting optimized for sentence transformer embedding.

    Args:
        path: Full path (e.g., "equilibrium/time_slice/profiles_1d/psi")
        path_info: Path metadata dict
        ids_info: Optional IDS-level metadata for context

    Returns:
        Natural language prose optimized for sentence transformer embedding
    """
    from imas_codex.core.unit_loader import get_unit_dimensionality, get_unit_name

    sentences = []

    # Extract components
    ids_name = path.split("/")[0]
    path_name = path_info.get("name", path.split("/")[-1])

    # Core identity with readable path formatting (convert / to " in ")
    path_readable = path_name.replace("_", " ")
    ids_readable = ids_name.replace("_", " ")

    # Include parent context in path name for better semantic context
    path_parts = path.split("/")
    if len(path_parts) > 2:
        parent_context = " in ".join(p.replace("_", " ") for p in path_parts[1:-1] if p)
        if parent_context:
            sentences.append(
                f"The {path_readable} in {parent_context} field in the {ids_readable} IDS."
            )
        else:
            sentences.append(f"The {path_readable} field in the {ids_readable} IDS.")
    else:
        sentences.append(f"The {path_readable} field in the {ids_readable} IDS.")

    # IDS-level context if available
    if ids_info and ids_info.get("description"):
        ids_desc = ids_info["description"]
        if len(ids_desc) < 200:
            sentences.append(f"The {ids_readable} IDS contains {ids_desc.lower()}")

    # Primary documentation
    doc = path_info.get("documentation", "")
    if doc:
        doc_clean = doc.strip()
        if doc_clean and not doc_clean.endswith("."):
            doc_clean += "."
        sentences.append(doc_clean)

    # Units with pint-based expansion for better semantic matching
    units = path_info.get("units", "")
    if units and units not in ("", "none", "1", "as_parent"):
        unit_parts = []

        # Get expanded unit name (e.g., "eV" -> "electron volt")
        unit_name = get_unit_name(units)
        if unit_name and unit_name != units:
            unit_parts.append(f"measured in {unit_name} ({units})")
        else:
            unit_parts.append(f"measured in {units}")

        # Add dimensionality for physics context (e.g., "[energy]", "[length]")
        dimensionality = get_unit_dimensionality(units)
        if dimensionality and dimensionality != "dimensionless":
            unit_parts.append(f"representing {dimensionality}")

        if unit_parts:
            unit_sentence = " ".join(unit_parts)
            # Capitalize first letter without lowercasing rest (preserves eV, Pa)
            sentences.append(unit_sentence[0].upper() + unit_sentence[1:] + ".")

    # Physics domain context
    physics_domain = path_info.get("physics_domain", "")
    if not physics_domain:
        # Derive from IDS name if not provided
        physics_domain = physics_categorizer.get_domain_for_ids(ids_name).value

    if physics_domain and physics_domain != "general":
        domain_readable = physics_domain.replace("_", " ")
        sentences.append(f"Related to {domain_readable} physics.")

    # Data type in natural language
    data_type = path_info.get("data_type", "")
    if data_type and data_type not in ("STRUCTURE", "STRUCT_ARRAY"):
        ndim = path_info.get("ndim", 0)
        if ndim == 0:
            sentences.append("This is a scalar value.")
        elif ndim == 1:
            sentences.append("This is a one-dimensional array.")
        elif ndim == 2:
            sentences.append("This is a two-dimensional array.")
        elif ndim == 3:
            sentences.append("This is a three-dimensional array.")
        elif ndim > 3:
            sentences.append(f"This is a {ndim}-dimensional array.")

    # Coordinate system in natural language
    coordinates = path_info.get("coordinates", [])
    if coordinates and isinstance(coordinates, list):
        valid_coords = [str(c) for c in coordinates if c]
        if valid_coords:
            if len(valid_coords) == 1:
                sentences.append(f"Indexed along the {valid_coords[0]} coordinate.")
            else:
                coords_formatted = ", ".join(valid_coords[:-1])
                sentences.append(
                    f"Indexed along the {coords_formatted} and {valid_coords[-1]} coordinates."
                )

    return " ".join(sentences)


def compute_content_hash(text: str) -> str:
    """
    Compute SHA256 hash of content for cache busting.

    Args:
        text: Text to hash (typically enriched_text)

    Returns:
        First 16 characters of SHA256 hex digest
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def generate_embeddings_batch(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    use_rich: bool | None = None,
) -> np.ndarray:
    """
    Generate embeddings for a batch of texts using sentence transformer.

    Args:
        texts: List of text strings to embed
        model_name: Sentence transformer model name
        batch_size: Batch size for encoding
        use_rich: Force rich progress (True), logging (False), or auto (None)

    Returns:
        Numpy array of embeddings (N x 384 for MiniLM)
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    total_batches = (len(texts) + batch_size - 1) // batch_size
    if total_batches <= 1:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    batch_names = [
        f"{min((i + 1) * batch_size, len(texts))}/{len(texts)}"
        for i in range(total_batches)
    ]

    progress = create_progress_monitor(
        use_rich=use_rich,
        logger=logger,
        item_names=batch_names,
        description_template="Embedding: {item}",
    )

    embeddings_list = []
    progress.start_processing(batch_names, "Generating embeddings")
    try:
        for i in range(0, len(texts), batch_size):
            texts_processed = min((i // batch_size + 1) * batch_size, len(texts))
            batch_name = f"{texts_processed}/{len(texts)}"
            progress.set_current_item(batch_name)

            batch_texts = texts[i : i + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings_list.append(batch_embeddings)
            progress.update_progress(batch_name)
    finally:
        progress.finish_processing()

    return np.vstack(embeddings_list)


def get_existing_embedding_hashes(
    client: GraphClient,
    paths: list[str],
    batch_size: int = 1000,
) -> dict[str, str]:
    """
    Query existing embedding hashes from the graph for cache validation.

    Args:
        client: GraphClient instance
        paths: List of path IDs to check
        batch_size: Query batch size

    Returns:
        Dict mapping path_id to embedding_hash (only for paths with existing hashes)
    """
    result = {}

    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        records = client.query(
            """
            UNWIND $paths AS path_id
            MATCH (p:IMASPath {id: path_id})
            WHERE p.embedding_hash IS NOT NULL
            RETURN p.id AS path_id, p.embedding_hash AS hash
            """,
            paths=batch,
        )
        for record in records:
            result[record["path_id"]] = record["hash"]

    return result


def update_path_embeddings(
    client: GraphClient,
    paths_data: dict[str, dict],
    ids_info: dict[str, dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 500,
    force_rebuild: bool = False,
    use_rich: bool | None = None,
) -> dict[str, int]:
    """
    Generate and store embeddings for IMASPath nodes with content-based caching.

    Only regenerates embeddings for paths where the content hash has changed,
    minimizing recompute on graph rebuilds. Model name is stored on DDVersion,
    not on individual paths.

    Args:
        client: GraphClient instance
        paths_data: Dict mapping path_id to path metadata
        ids_info: Dict mapping ids_name to IDS metadata
        model_name: Sentence transformer model name
        batch_size: Batch size for embedding generation
        force_rebuild: If True, regenerate all embeddings regardless of cache
        use_rich: Force rich progress (True), logging (False), or auto (None)

    Returns:
        Dict with stats: {"updated": N, "cached": M, "total": N+M}
    """
    if not paths_data:
        return {"updated": 0, "cached": 0, "total": 0}

    # Step 1: Generate embedding text and compute hashes for all paths
    path_ids = list(paths_data.keys())
    embedding_texts = {}
    content_hashes = {}

    for path_id, path_info in paths_data.items():
        ids_name = path_id.split("/")[0]
        ids_meta = ids_info.get(ids_name, {})
        text = generate_embedding_text(path_id, path_info, ids_meta)
        embedding_texts[path_id] = text
        content_hashes[path_id] = compute_content_hash(text)

    # Step 2: Get existing hashes from graph to determine what needs update
    if not force_rebuild:
        existing_hashes = get_existing_embedding_hashes(client, path_ids)
        paths_to_update = [
            pid for pid in path_ids if content_hashes[pid] != existing_hashes.get(pid)
        ]
        cached_count = len(path_ids) - len(paths_to_update)
    else:
        paths_to_update = path_ids
        cached_count = 0

    if not paths_to_update:
        logger.info(f"All {len(path_ids)} embeddings up to date (cached)")
        return {"updated": 0, "cached": len(path_ids), "total": len(path_ids)}

    logger.info(
        f"Generating embeddings for {len(paths_to_update)} paths "
        f"({cached_count} cached)"
    )

    # Step 3: Generate embeddings for paths that need update
    texts_to_embed = [embedding_texts[pid] for pid in paths_to_update]
    embeddings = generate_embeddings_batch(
        texts_to_embed, model_name, batch_size, use_rich=use_rich
    )

    # Step 4: Store embeddings in graph in batches with progress tracking
    store_batch_size = 100
    total_store_batches = (
        len(paths_to_update) + store_batch_size - 1
    ) // store_batch_size

    if total_store_batches > 1:
        store_batch_names = [
            f"{min((i + 1) * store_batch_size, len(paths_to_update))}/{len(paths_to_update)}"
            for i in range(total_store_batches)
        ]
        store_progress = create_progress_monitor(
            use_rich=use_rich,
            logger=logger,
            item_names=store_batch_names,
            description_template="Storing: {item}",
        )
        store_progress.start_processing(store_batch_names, "Storing embeddings")
    else:
        store_progress = None

    try:
        for i in range(0, len(paths_to_update), store_batch_size):
            batch_paths = paths_to_update[i : i + store_batch_size]
            batch_data = []

            if store_progress:
                paths_stored = min(
                    (i // store_batch_size + 1) * store_batch_size, len(paths_to_update)
                )
                batch_name = f"{paths_stored}/{len(paths_to_update)}"
                store_progress.set_current_item(batch_name)

            for j, path_id in enumerate(batch_paths):
                embedding_idx = i + j
                batch_data.append(
                    {
                        "path_id": path_id,
                        "embedding_text": embedding_texts[path_id],
                        "embedding": embeddings[embedding_idx].tolist(),
                        "embedding_hash": content_hashes[path_id],
                    }
                )

            client.query(
                """
                UNWIND $batch AS b
                MATCH (p:IMASPath {id: b.path_id})
                SET p.embedding_text = b.embedding_text,
                    p.embedding = b.embedding,
                    p.embedding_hash = b.embedding_hash
                """,
                batch=batch_data,
            )

            if store_progress:
                paths_stored = min(
                    (i // store_batch_size + 1) * store_batch_size, len(paths_to_update)
                )
                batch_name = f"{paths_stored}/{len(paths_to_update)}"
                store_progress.update_progress(batch_name)
    finally:
        if store_progress:
            store_progress.finish_processing()

    return {
        "updated": len(paths_to_update),
        "cached": cached_count,
        "total": len(path_ids),
    }


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
    include_embeddings: bool = True,
    dry_run: bool = False,
    ids_filter: set[str] | None = None,
    use_rich: bool | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    force_embeddings: bool = False,
) -> dict:
    """
    Build the IMAS DD graph.

    Args:
        client: Neo4j GraphClient
        versions: List of versions to process (None = all available)
        ids_filter: Optional set of IDS names to include
        include_clusters: Whether to import semantic clusters
        include_embeddings: Whether to generate path embeddings (default True)
        dry_run: If True, don't write to graph
        use_rich: Force rich progress (True), logging (False), or auto (None)
        embedding_model: Sentence transformer model for embeddings
        force_embeddings: Force regenerate all embeddings (ignore cache)

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
        "embeddings_updated": 0,
        "embeddings_cached": 0,
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

    # Generate and store embeddings for current version paths
    if include_embeddings and not dry_run:
        current_version_data = version_data.get(current_dd_version)
        if current_version_data:
            logger.info(f"Generating embeddings for {current_dd_version}...")
            embedding_stats = update_path_embeddings(
                client=client,
                paths_data=current_version_data["paths"],
                ids_info=current_version_data["ids_info"],
                model_name=embedding_model,
                force_rebuild=force_embeddings,
                use_rich=use_rich,
            )
            stats["embeddings_updated"] = embedding_stats["updated"]
            stats["embeddings_cached"] = embedding_stats["cached"]

            # Update DDVersion with embedding metadata
            client.query(
                """
                MATCH (v:DDVersion {id: $version})
                SET v.embeddings_built_at = datetime(),
                    v.embeddings_model = $model,
                    v.embeddings_count = $count
                """,
                version=current_dd_version,
                model=embedding_model,
                count=embedding_stats["total"],
            )
        else:
            logger.warning(f"No data for current version {current_dd_version}")

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

    # SemanticCluster.id - for cluster lookups
    client.query(
        "CREATE INDEX semanticcluster_id IF NOT EXISTS FOR (c:SemanticCluster) ON (c.id)"
    )

    # Vector indexes for semantic search
    _ensure_vector_indexes(client)


def _ensure_vector_indexes(client: GraphClient) -> None:
    """Create vector indexes for semantic search on embeddings.

    Creates two vector indexes:
    - imas_path_embedding: For searching IMASPath nodes by embedding
    - cluster_centroid: For hierarchical cluster-first search
    """
    # Check if vector indexes already exist
    try:
        existing = client.query("SHOW INDEXES YIELD name RETURN collect(name) AS names")
        existing_names = set(existing[0]["names"]) if existing else set()
    except Exception:
        existing_names = set()

    # Create IMASPath embedding index (384 dims for all-MiniLM-L6-v2)
    if "imas_path_embedding" not in existing_names:
        try:
            client.query("""
                CREATE VECTOR INDEX imas_path_embedding IF NOT EXISTS
                FOR (p:IMASPath) ON p.embedding
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            logger.info("Created vector index: imas_path_embedding")
        except Exception as e:
            logger.warning(f"Failed to create imas_path_embedding index: {e}")

    # Create SemanticCluster centroid index
    if "cluster_centroid" not in existing_names:
        try:
            client.query("""
                CREATE VECTOR INDEX cluster_centroid IF NOT EXISTS
                FOR (c:SemanticCluster) ON c.centroid
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)
            logger.info("Created vector index: cluster_centroid")
        except Exception as e:
            logger.warning(f"Failed to create cluster_centroid index: {e}")


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


def _import_clusters(
    client: GraphClient,
    dry_run: bool,
) -> int:
    """Import semantic clusters from existing cluster data with centroids.

    Stores cluster centroids for hierarchical semantic search - first match
    clusters by centroid similarity, then search within matching clusters.
    Model name is tracked on DDVersion, not on clusters.

    Args:
        client: GraphClient instance
        dry_run: If True, don't write to graph

    Returns:
        Number of clusters imported
    """
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

            # Extract cluster properties
            label = cluster.get("label", f"cluster_{cluster_id}")
            physics_domain = cluster.get("physics_domain", "general")
            paths = cluster.get("paths", [])
            cross_ids = cluster.get("cross_ids", False)
            centroid = cluster.get("centroid")
            similarity_score = cluster.get("similarity_score", 0.0)
            ids_names = cluster.get("ids_names", [])
            scope = cluster.get("scope", "global")

            # Build cluster properties dict
            cluster_props = {
                "cluster_id": str(cluster_id),
                "label": label,
                "physics_domain": physics_domain,
                "path_count": len(paths),
                "cross_ids": cross_ids,
                "similarity_score": similarity_score,
                "scope": scope,
            }

            # Add centroid if available (list of floats for Neo4j vector index)
            if centroid and isinstance(centroid, list):
                cluster_props["centroid"] = centroid

            # Add IDS names as array
            if ids_names:
                cluster_props["ids_names"] = ids_names

            # Create SemanticCluster node with all properties
            client.query(
                """
                MERGE (c:SemanticCluster {id: $cluster_id})
                SET c.label = $label,
                    c.physics_domain = $physics_domain,
                    c.path_count = $path_count,
                    c.cross_ids = $cross_ids,
                    c.similarity_score = $similarity_score,
                    c.scope = $scope,
                    c.centroid = $centroid,
                    c.ids_names = $ids_names
                """,
                **cluster_props,
            )

            # Batch create IN_CLUSTER relationships for efficiency
            path_memberships = []
            for path_info in paths:
                if isinstance(path_info, dict):
                    path = path_info.get("path", "")
                    distance = path_info.get("distance", 0.0)
                else:
                    path = str(path_info)
                    distance = 0.0
                if path:
                    path_memberships.append({"path": path, "distance": distance})

            if path_memberships:
                client.query(
                    """
                    UNWIND $memberships AS m
                    MATCH (p:IMASPath {id: m.path})
                    MATCH (c:SemanticCluster {id: $cluster_id})
                    MERGE (p)-[r:IN_CLUSTER]->(c)
                    SET r.distance = m.distance
                    """,
                    memberships=path_memberships,
                    cluster_id=str(cluster_id),
                )

            cluster_count += 1

        # Update DDVersion with cluster metadata
        client.query(
            """
            MATCH (v:DDVersion {is_current: true})
            SET v.clusters_built_at = datetime(),
                v.clusters_count = $count
            """,
            count=cluster_count,
        )

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
    "--include-embeddings/--no-embeddings",
    default=True,
    help="Generate embeddings for IMASPath nodes (default: enabled)",
)
@click.option(
    "--force-embeddings",
    is_flag=True,
    help="Force regenerate all embeddings (ignore cache)",
)
@click.option(
    "--embedding-model",
    type=str,
    default="all-MiniLM-L6-v2",
    help="Sentence transformer model for embeddings",
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
    include_embeddings: bool,
    force_embeddings: bool,
    embedding_model: str,
    dry_run: bool,
    no_rich: bool,
) -> int:
    """Build the IMAS Data Dictionary Knowledge Graph.

    This command populates Neo4j with IMAS DD structure including version
    tracking, path hierarchy, units, embeddings, and optionally semantic clusters.

    Embeddings are cached using content-based hashing - only paths with changed
    content will have embeddings regenerated on subsequent runs.

    Examples:
        build-dd-graph                        # Build current version with embeddings
        build-dd-graph --all-versions         # Build all 34 versions
        build-dd-graph --from-version 4.0.0   # Incremental from 4.0.0
        build-dd-graph --include-clusters     # Include semantic clusters
        build-dd-graph --force-embeddings     # Regenerate all embeddings
        build-dd-graph --no-embeddings        # Skip embedding generation
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
                include_embeddings=include_embeddings,
                dry_run=dry_run,
                use_rich=use_rich,
                embedding_model=embedding_model,
                force_embeddings=force_embeddings,
            )

        # Report results
        click.echo("\n=== Build Complete ===")
        click.echo(f"Versions processed: {stats['versions_processed']}")
        click.echo(f"IDS nodes: {stats['ids_created']}")
        click.echo(f"IMASPath nodes created: {stats['paths_created']}")
        click.echo(f"Unit nodes: {stats['units_created']}")
        click.echo(f"PathChange nodes: {stats['path_changes_created']}")
        if include_embeddings:
            click.echo(f"Embeddings updated: {stats['embeddings_updated']}")
            click.echo(f"Embeddings cached: {stats['embeddings_cached']}")
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
