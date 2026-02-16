"""
Build the IMAS Data Dictionary Knowledge Graph.

This module provides functions to populate Neo4j with IMAS DD structure:
- DDVersion nodes for all available DD versions
- IDS nodes for top-level structures
- IMASPath nodes with hierarchical relationships
- Unit and IMASCoordinateSpec nodes
- Version tracking (INTRODUCED_IN, DEPRECATED_IN, RENAMED_TO)
- IMASPathChange nodes for metadata changes with semantic classification
- IMASSemanticCluster nodes from existing clusters (optional)
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
import re
from contextlib import contextmanager
from pathlib import Path

import imas
import numpy as np

from imas_codex import dd_version as current_dd_version
from imas_codex.core.exclusions import ExclusionChecker
from imas_codex.core.physics_categorization import physics_categorizer
from imas_codex.core.progress_monitor import (
    create_build_monitor,
    create_progress_monitor,
)
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)


@contextmanager
def suppress_third_party_logging(level: int = logging.WARNING):
    """Temporarily suppress noisy third-party logging during operations.

    Raises log level for sentence_transformers and other verbose libraries
    to prevent interference with progress bars.
    """
    noisy_loggers = [
        "sentence_transformers",
        "sentence_transformers.SentenceTransformer",
        "transformers",
        "huggingface_hub",
        "linkml_runtime",
    ]
    original_levels = {}
    for name in noisy_loggers:
        log = logging.getLogger(name)
        original_levels[name] = log.level
        log.setLevel(level)

    try:
        yield
    finally:
        for name, orig_level in original_levels.items():
            logging.getLogger(name).setLevel(orig_level)


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
    Compute SHA256 hash of content for change tracking.

    Args:
        text: Text to hash (typically enriched_text)

    Returns:
        First 16 characters of SHA256 hex digest
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute_embedding_hash(text: str, model_name: str) -> str:
    """
    Compute hash for embedding cache validation including model name.

    Includes model name so changing the embedding model invalidates
    cached embeddings even if the content text hasn't changed.

    Args:
        text: Embedding text content
        model_name: Full model name (e.g., "Qwen/Qwen3-Embedding-0.6B")

    Returns:
        First 16 characters of SHA256 hex digest
    """
    combined = f"{model_name}:{text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# Embedding change types for tracking what changed between versions
class EmbeddingChangeType:
    """Types of embedding content changes detected."""

    UNITS_CHANGED = "UNITS_CHANGED"
    DOCUMENTATION_CHANGED = "DOCUMENTATION_CHANGED"
    COORDINATES_CHANGED = "COORDINATES_CHANGED"
    DATA_TYPE_CHANGED = "DATA_TYPE_CHANGED"
    PHYSICS_DOMAIN_CHANGED = "PHYSICS_DOMAIN_CHANGED"
    IDS_CONTEXT_CHANGED = "IDS_CONTEXT_CHANGED"


def detect_embedding_changes(
    old_text: str | None, new_text: str, path_info: dict
) -> list[str]:
    """
    Detect what changed between old and new embedding text.

    Compares semantic sections of embedding text to identify specific
    changes in units, documentation, coordinates, etc.

    Args:
        old_text: Previous embedding text (None if new path)
        new_text: Current embedding text
        path_info: Path metadata for additional context

    Returns:
        List of EmbeddingChangeType values for detected changes
    """
    if old_text is None:
        return []  # New path, not a change

    if old_text == new_text:
        return []  # No change

    changes = []

    # Detect units change
    old_units = _extract_units_section(old_text)
    new_units = _extract_units_section(new_text)
    if old_units != new_units:
        changes.append(EmbeddingChangeType.UNITS_CHANGED)

    # Detect documentation change (longest sentence typically)
    old_doc = _extract_documentation_section(old_text)
    new_doc = _extract_documentation_section(new_text)
    if old_doc != new_doc:
        changes.append(EmbeddingChangeType.DOCUMENTATION_CHANGED)

    # Detect coordinates change
    if "coordinate" in old_text.lower() or "coordinate" in new_text.lower():
        old_coords = _extract_coordinate_section(old_text)
        new_coords = _extract_coordinate_section(new_text)
        if old_coords != new_coords:
            changes.append(EmbeddingChangeType.COORDINATES_CHANGED)

    # Detect data type change
    old_dtype = _extract_data_type_section(old_text)
    new_dtype = _extract_data_type_section(new_text)
    if old_dtype != new_dtype:
        changes.append(EmbeddingChangeType.DATA_TYPE_CHANGED)

    # Detect physics domain change
    if "physics" in old_text.lower() or "physics" in new_text.lower():
        old_domain = _extract_physics_domain_section(old_text)
        new_domain = _extract_physics_domain_section(new_text)
        if old_domain != new_domain:
            changes.append(EmbeddingChangeType.PHYSICS_DOMAIN_CHANGED)

    # Detect IDS context change
    old_ids = _extract_ids_context_section(old_text)
    new_ids = _extract_ids_context_section(new_text)
    if old_ids != new_ids:
        changes.append(EmbeddingChangeType.IDS_CONTEXT_CHANGED)

    return changes


def _extract_units_section(text: str) -> str | None:
    """Extract units portion of embedding text."""
    match = re.search(r"[Mm]easured in ([^.]+)\.", text)
    return match.group(1) if match else None


def _extract_documentation_section(text: str) -> str | None:
    """Extract documentation portion (usually 3rd sentence)."""
    sentences = text.split(". ")
    # Documentation is usually after IDS context, before units
    for i, sent in enumerate(sentences):
        if i >= 2 and not sent.startswith(
            ("Measured", "Related to", "This is a", "Indexed")
        ):
            return sent
    return None


def _extract_coordinate_section(text: str) -> str | None:
    """Extract coordinate information."""
    match = re.search(r"[Ii]ndexed along ([^.]+)\.", text)
    return match.group(1) if match else None


def _extract_data_type_section(text: str) -> str | None:
    """Extract data type description."""
    match = re.search(r"This is a ([^.]+)\.", text)
    return match.group(1) if match else None


def _extract_physics_domain_section(text: str) -> str | None:
    """Extract physics domain."""
    match = re.search(r"Related to ([^.]+) physics\.", text)
    return match.group(1) if match else None


def _extract_ids_context_section(text: str) -> str | None:
    """Extract IDS context description."""
    match = re.search(r"The [^.]+ IDS contains ([^.]+)", text)
    return match.group(1) if match else None


# Error field pattern for identifying error paths and extracting base path and type
ERROR_FIELD_PATTERN = re.compile(r"^(.+?)_error_(upper|lower|index)$")


def filter_embeddable_paths(
    paths_data: dict[str, dict],
    exclusion_checker: ExclusionChecker | None = None,
) -> tuple[dict[str, dict], dict[str, tuple[str, str]]]:
    """
    Filter paths to only those that should have embeddings generated.

    Error fields and metadata paths are excluded from embedding generation
    but still exist as nodes in the graph. GGD paths are included by default
    (configurable via `include-ggd` setting in pyproject.toml).

    Args:
        paths_data: Dict mapping path_id to path metadata
        exclusion_checker: Optional ExclusionChecker (uses default if None)

    Returns:
        Tuple of (embeddable_paths, error_relationships) where:
        - embeddable_paths: Dict of paths that should have embeddings
        - error_relationships: Dict mapping error_path -> (data_path, error_type)
          where error_type is "upper", "lower", or "index"
    """
    checker = exclusion_checker or ExclusionChecker()
    embeddable = {}
    error_relationships: dict[str, tuple[str, str]] = {}

    for path_id, path_info in paths_data.items():
        exclusion_reason = checker.get_exclusion_reason(path_id)

        if exclusion_reason is None:
            # Not excluded - will be embedded
            embeddable[path_id] = path_info
        elif exclusion_reason == "error_field":
            # Error field - extract base path and error type for linking
            name = path_info.get("name", "")
            match = ERROR_FIELD_PATTERN.match(name)
            if match:
                base_name = match.group(1)
                error_type = match.group(2)  # "upper", "lower", or "index"
                # Construct the data path by replacing error suffix in full path
                parent_path = path_info.get("parent_path", "")
                if parent_path:
                    data_path = f"{parent_path}/{base_name}"
                    error_relationships[path_id] = (data_path, error_type)

    return embeddable, error_relationships


def generate_embeddings_batch(
    texts: list[str],
    model_name: str | None = None,
    batch_size: int = 256,
    use_rich: bool | None = None,
) -> np.ndarray:
    """
    Generate embeddings for a batch of texts using Encoder.

    Args:
        texts: List of text strings to embed
        model_name: Model name (defaults to configured model from settings)
        batch_size: Batch size for encoding
        use_rich: Force rich progress (True), logging (False), or auto (None)

    Returns:
        Numpy array of embeddings (N x embedding_dimension)
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_embedding_model

    if model_name is None:
        model_name = get_embedding_model()

    config = EncoderConfig(
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        use_rich=use_rich if use_rich is not None else True,
    )
    encoder = Encoder(config=config)
    return encoder.embed_texts(texts)


def get_existing_embedding_data(
    client: GraphClient,
    paths: list[str],
    batch_size: int = 1000,
) -> dict[str, dict]:
    """
    Query existing embedding data from the graph for change detection.

    Args:
        client: GraphClient instance
        paths: List of path IDs to check
        batch_size: Query batch size

    Returns:
        Dict mapping path_id to {hash, text} (only for paths with existing data)
    """
    result = {}

    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        records = client.query(
            """
            UNWIND $paths AS path_id
            MATCH (p:IMASPath {id: path_id})
            WHERE p.embedding_hash IS NOT NULL
            RETURN p.id AS path_id, p.embedding_hash AS hash, p.embedding_text AS text
            """,
            paths=batch,
        )
        for record in records:
            result[record["path_id"]] = {
                "hash": record["hash"],
                "text": record["text"],
            }

    return result


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


def record_embedding_changes(
    client: GraphClient,
    changes: list[dict],
    dd_version: str,
) -> int:
    """
    Record embedding content changes as relationships in the graph.

    Creates HAS_EMBEDDING_CHANGE relationships linking IMASPath nodes to
    EmbeddingChange nodes that track what changed and when.

    Args:
        client: GraphClient instance
        changes: List of dicts with path_id, change_types, old_text, new_text
        dd_version: DD version when change was detected

    Returns:
        Number of change records created
    """
    if not changes:
        return 0

    # Batch create EmbeddingChange nodes and relationships
    change_data = []
    for change in changes:
        for change_type in change["change_types"]:
            change_data.append(
                {
                    "path_id": change["path_id"],
                    "change_type": change_type,
                    "old_text_hash": compute_content_hash(change["old_text"])
                    if change.get("old_text")
                    else None,
                    "new_text_hash": compute_content_hash(change["new_text"]),
                    "dd_version": dd_version,
                }
            )

    if not change_data:
        return 0

    # Use MERGE to avoid duplicates for same path/change_type/version
    client.query(
        """
        UNWIND $changes AS c
        MATCH (p:IMASPath {id: c.path_id})
        MERGE (p)-[r:HAS_EMBEDDING_CHANGE {
            change_type: c.change_type,
            dd_version: c.dd_version
        }]->(ec:EmbeddingChange {
            path_id: c.path_id,
            change_type: c.change_type,
            dd_version: c.dd_version
        })
        ON CREATE SET
            ec.detected_at = datetime(),
            ec.old_hash = c.old_text_hash,
            ec.new_hash = c.new_text_hash
        """,
        changes=change_data,
    )

    return len(change_data)


def update_path_embeddings(
    client: GraphClient,
    paths_data: dict[str, dict],
    ids_info: dict[str, dict],
    model_name: str | None = None,
    batch_size: int = 500,
    force_rebuild: bool = False,
    use_rich: bool | None = None,
    dd_version: str | None = None,
    track_changes: bool = True,
) -> dict[str, int]:
    """
    Generate and store embeddings for IMASPath nodes with content-based caching.

    Only regenerates embeddings for paths where the content hash has changed,
    minimizing recompute on graph rebuilds. Model name is stored on DDVersion,
    not on individual paths.

    When track_changes is True, detects what changed (units, documentation, etc.)
    and records EmbeddingChange nodes linked to affected paths.

    Args:
        client: GraphClient instance
        paths_data: Dict mapping path_id to path metadata
        ids_info: Dict mapping ids_name to IDS metadata
        model_name: Embedding model name (defaults to configured model from settings)
        batch_size: Batch size for embedding generation
        force_rebuild: If True, regenerate all embeddings regardless of cache
        use_rich: Force rich progress (True), logging (False), or auto (None)
        dd_version: DD version for change tracking (uses current if None)
        track_changes: If True, detect and record what changed in embeddings

    Returns:
        Dict with stats: {"updated": N, "cached": M, "total": N+M, "changes": C}
    """
    if not paths_data:
        return {"updated": 0, "cached": 0, "total": 0, "changes": 0}

    # Resolve model name early for hash computation
    from imas_codex.settings import get_embedding_model

    resolved_model = model_name or get_embedding_model()

    # Step 1: Generate embedding text and compute hashes for all paths
    # Hash includes model name so changing model invalidates cached embeddings
    path_ids = list(paths_data.keys())
    embedding_texts = {}
    content_hashes = {}

    for path_id, path_info in paths_data.items():
        ids_name = path_id.split("/")[0]
        ids_meta = ids_info.get(ids_name, {})
        text = generate_embedding_text(path_id, path_info, ids_meta)
        embedding_texts[path_id] = text
        content_hashes[path_id] = compute_embedding_hash(text, resolved_model)

    # Step 2: Get existing data from graph for cache validation AND change detection
    existing_data = {}
    if not force_rebuild or track_changes:
        existing_data = get_existing_embedding_data(client, path_ids)

    if not force_rebuild:
        paths_to_update = [
            pid
            for pid in path_ids
            if content_hashes[pid] != existing_data.get(pid, {}).get("hash")
        ]
        cached_count = len(path_ids) - len(paths_to_update)
    else:
        paths_to_update = path_ids
        cached_count = 0

    if not paths_to_update:
        logger.debug(f"All {len(path_ids)} embeddings up to date (cached)")
        return {
            "updated": 0,
            "cached": len(path_ids),
            "total": len(path_ids),
            "changes": 0,
        }

    logger.debug(
        f"Generating embeddings for {len(paths_to_update)} paths "
        f"({cached_count} cached)"
    )

    # Step 3: Detect changes for paths being updated (if tracking enabled)
    changes_detected = []
    if track_changes:
        for path_id in paths_to_update:
            old_data = existing_data.get(path_id)
            if old_data and old_data.get("text"):
                change_types = detect_embedding_changes(
                    old_data["text"],
                    embedding_texts[path_id],
                    paths_data[path_id],
                )
                if change_types:
                    changes_detected.append(
                        {
                            "path_id": path_id,
                            "change_types": change_types,
                            "old_text": old_data["text"],
                            "new_text": embedding_texts[path_id],
                        }
                    )

    # Step 4: Generate embeddings for paths that need update
    texts_to_embed = [embedding_texts[pid] for pid in paths_to_update]
    embeddings = generate_embeddings_batch(
        texts_to_embed, resolved_model, batch_size, use_rich=use_rich
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

    # Step 6: Record embedding changes if tracking enabled
    changes_count = 0
    if track_changes and changes_detected:
        version = dd_version or current_dd_version
        changes_count = record_embedding_changes(client, changes_detected, version)
        if changes_count > 0:
            logger.debug(f"Recorded {changes_count} embedding content changes")

    return {
        "updated": len(paths_to_update),
        "cached": cached_count,
        "total": len(path_ids),
        "changes": changes_count,
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


def _compute_build_hash(
    versions: list[str],
    ids_filter: set[str] | None,
    embedding_model: str | None,
    include_clusters: bool,
    include_embeddings: bool,
) -> str:
    """Compute a deterministic hash of the build parameters.

    Used to detect whether a previous build with the same parameters
    already populated the graph, allowing fast no-op re-runs.
    """
    parts = [
        ",".join(sorted(versions)),
        ",".join(sorted(ids_filter)) if ids_filter else "",
        embedding_model or "",
        str(include_clusters),
        str(include_embeddings),
    ]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _check_graph_up_to_date(
    client: GraphClient,
    build_hash: str,
    versions: list[str],
    include_embeddings: bool,
    include_clusters: bool,
) -> bool:
    """Check if the graph already contains a build matching the given hash.

    Validates:
    - DDVersion.build_hash on the current version matches
    - All requested versions exist
    - Embedding coverage is complete (if embeddings were requested)
    - Clusters exist (if clusters were requested)

    Returns True only when the graph is fully up-to-date.
    """
    try:
        result = client.query(
            """
            MATCH (d:DDVersion {is_current: true})
            RETURN d.build_hash AS hash
            """
        )
        if not result:
            return False

        meta = result[0]
        if meta.get("hash") != build_hash:
            return False

        # Verify all requested versions exist
        ver_result = client.query(
            "MATCH (d:DDVersion) RETURN collect(d.id) AS versions"
        )
        stored_versions = ver_result[0]["versions"] if ver_result else []
        if set(versions) != set(stored_versions):
            return False

        # Verify embeddings are present
        if include_embeddings:
            emb_result = client.query(
                """
                MATCH (p:IMASPath)
                WITH count(p) AS total,
                     count(CASE WHEN p.embedding IS NOT NULL THEN 1 END) AS with_emb
                RETURN total, with_emb
                """
            )
            if emb_result:
                row = emb_result[0]
                if row["total"] == 0 or row["with_emb"] < row["total"] * 0.95:
                    return False

        # Verify clusters are present
        if include_clusters:
            cl_result = client.query(
                "MATCH (c:IMASSemanticCluster) RETURN count(c) AS cnt"
            )
            if not cl_result or cl_result[0]["cnt"] == 0:
                return False

        return True
    except Exception as e:
        logger.debug(f"Graph up-to-date check failed: {e}")
        return False


def build_dd_graph(
    client: GraphClient,
    versions: list[str] | None = None,
    include_clusters: bool = False,
    include_embeddings: bool = True,
    dry_run: bool = False,
    ids_filter: set[str] | None = None,
    use_rich: bool | None = None,
    embedding_model: str | None = None,
    force_embeddings: bool = False,
) -> dict:
    """
    Build the IMAS DD graph.

    On a second run with identical parameters the function verifies the
    graph is already up-to-date via a build hash stored on the current ``DDVersion``
    and returns immediately, avoiding any expensive re-extraction or
    re-embedding.

    Args:
        client: Neo4j GraphClient
        versions: List of versions to process (None = all available)
        ids_filter: Optional set of IDS names to include
        include_clusters: Whether to import semantic clusters
        include_embeddings: Whether to generate path embeddings (default True)
        dry_run: If True, don't write to graph
        use_rich: Force rich progress (True), logging (False), or auto (None)
        embedding_model: Embedding model name (defaults to configured model from settings)
        force_embeddings: Force regenerate all embeddings (ignore cache)

    Returns:
        Statistics about the build
    """
    all_versions = get_all_dd_versions()

    if versions is None:
        versions = all_versions
    else:
        for v in versions:
            if v not in all_versions:
                raise ValueError(f"Unknown DD version: {v}")

    stats = {
        "versions_processed": 0,
        "ids_created": 0,
        "paths_created": 0,
        "units_created": 0,
        "path_changes_created": 0,
        "clusters_created": 0,
        "embeddings_updated": 0,
        "embeddings_cached": 0,
        "definitions_changed": 0,
        "error_relationships": 0,
        "paths_filtered": 0,
        "skipped": False,
    }

    # --- Hash-based idempotency check ---
    build_hash = _compute_build_hash(
        versions, ids_filter, embedding_model, include_clusters, include_embeddings
    )

    if not dry_run and not force_embeddings:
        if _check_graph_up_to_date(
            client, build_hash, versions, include_embeddings, include_clusters
        ):
            logger.info(
                "Graph already up-to-date (build hash %s) — skipping rebuild",
                build_hash,
            )
            stats["versions_processed"] = len(versions)
            stats["skipped"] = True
            return stats

    # --- Build with progress tracking ---
    monitor = create_build_monitor(use_rich=use_rich, logger=logger)

    with monitor.managed_build():
        # Ensure indexes exist for performance
        if not dry_run:
            monitor.status("Creating indexes...")
            _ensure_indexes(client)

        # Create DDVersion nodes
        if not dry_run:
            monitor.status(f"Creating {len(versions)} version nodes...")
            _create_version_nodes(client, versions)
        stats["versions_processed"] = len(versions)

        # Phase 1: Extract paths from all versions
        all_units: set[str] = set()
        version_data: dict[str, dict] = {}

        with monitor.phase(
            "Extract paths",
            items=versions,
            description_template="Extracting {item}",
        ) as phase:
            for version in versions:
                try:
                    data = extract_paths_for_version(version, ids_filter=ids_filter)
                    version_data[version] = data
                    all_units.update(data["units"])
                    phase.update(version)
                except Exception as e:
                    logger.error(f"Error extracting {version}: {e}")
                    phase.update(version, error=str(e))
            phase.set_detail(
                f"{sum(len(d['paths']) for d in version_data.values())} paths"
            )

        # Create Unit / CoordinateSpec nodes
        if not dry_run:
            monitor.status(f"Creating {len(all_units)} unit nodes...")
            _create_unit_nodes(client, all_units)
        stats["units_created"] = len(all_units)

        all_coord_specs: set[str] = set()
        for ver_data in version_data.values():
            for path_info in ver_data["paths"].values():
                for coord_str in path_info.get("coordinates", []):
                    if coord_str and coord_str.startswith("1..."):
                        all_coord_specs.add(coord_str)
        if not dry_run:
            monitor.status(f"Creating {len(all_coord_specs)} coordinate spec nodes...")
            _create_coordinate_spec_nodes(client, all_coord_specs)

        # Phase 2: Build graph nodes (IDS + IMASPath + relationships)
        prev_paths: dict[str, dict] = {}

        with monitor.phase(
            "Build graph",
            items=versions,
            description_template="Building {item}",
        ) as phase:
            for i, version in enumerate(versions):
                if version not in version_data:
                    phase.update(version, error="No data")
                    continue

                data = version_data[version]
                changes = compute_version_changes(prev_paths, data["paths"])

                if not dry_run:
                    _batch_upsert_ids_nodes(client, data["ids_info"], version, i == 0)
                    stats["ids_created"] = max(
                        stats["ids_created"], len(data["ids_info"])
                    )

                    new_paths_data = {p: data["paths"][p] for p in changes["added"]}
                    _batch_create_path_nodes(client, new_paths_data, version)
                    stats["paths_created"] += len(changes["added"])

                    _batch_mark_paths_deprecated(client, changes["removed"], version)

                    change_count = _batch_create_path_changes(
                        client, changes["changed"], version
                    )
                    stats["path_changes_created"] += change_count

                prev_paths = data["paths"]
                phase.update(version)

        # RENAMED_TO relationships
        if not dry_run:
            monitor.status("Creating RENAMED_TO relationships...")
            mappings = load_path_mappings(current_dd_version)
            _batch_create_renamed_to(client, mappings.get("old_to_new", {}))

        # Phase 3: Embeddings
        if include_embeddings and not dry_run:
            merged_paths: dict[str, dict] = {}
            merged_ids_info: dict[str, dict] = {}
            for version in versions:
                vdata = version_data.get(version)
                if vdata:
                    merged_paths.update(vdata["paths"])
                    merged_ids_info.update(vdata["ids_info"])

            if merged_paths:
                embeddable_paths, error_relationships = filter_embeddable_paths(
                    merged_paths
                )
                stats["paths_filtered"] = len(merged_paths) - len(embeddable_paths)
                monitor.status(
                    f"Embedding {len(embeddable_paths)} paths "
                    f"(filtered {stats['paths_filtered']})..."
                )

                if error_relationships:
                    stats["error_relationships"] = _batch_create_error_relationships(
                        client, error_relationships
                    )

                embedding_stats = update_path_embeddings(
                    client=client,
                    paths_data=embeddable_paths,
                    ids_info=merged_ids_info,
                    model_name=embedding_model,
                    force_rebuild=force_embeddings,
                    use_rich=use_rich,
                    dd_version=current_dd_version,
                    track_changes=True,
                )
                stats["embeddings_updated"] = embedding_stats["updated"]
                stats["embeddings_cached"] = embedding_stats["cached"]

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

        # Phase 4: Clusters
        if include_clusters:
            monitor.status("Importing semantic clusters...")
            cluster_count = _import_clusters(client, dry_run, use_rich=use_rich)
            stats["clusters_created"] = cluster_count

        # Persist build hash on current DDVersion for idempotency on next run
        if not dry_run:
            client.query(
                """
                MATCH (d:DDVersion {id: $current})
                SET d.build_hash = $hash
                """,
                current=current_dd_version,
                hash=build_hash,
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

    # IMASSemanticCluster.id - for cluster lookups
    client.query(
        "CREATE CONSTRAINT imassemanticcluster_id IF NOT EXISTS "
        "FOR (c:IMASSemanticCluster) REQUIRE c.id IS UNIQUE"
    )

    # Vector indexes for semantic search
    client.ensure_vector_indexes()


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
    """Create IMASCoordinateSpec nodes for index-based coordinates."""
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
            MERGE (c:IMASCoordinateSpec {id: spec.id})
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
            ids.leaf_count = ids_data.leaf_count,
            ids.dd_version = $version
    """,
        ids_list=ids_list,
        version=version,
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
                "coordinates": path_info.get("coordinates", []),
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
                path.maxoccur = p.maxoccur,
                path.ids = p.ids_name
        """,
            paths=batch,
        )

        # Step 2: Create IDS relationships
        client.query(
            """
            UNWIND $paths AS p
            MATCH (path:IMASPath {id: p.id})
            MATCH (ids:IDS {name: p.ids_name})
            MERGE (path)-[:IN_IDS]->(ids)
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
                MERGE (path)-[:HAS_PARENT]->(parent)
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

        # Step 5: Create HAS_COORDINATE relationships
        # Each coordinate links to either an IMASCoordinateSpec (e.g., "1...N") or IMASPath
        coord_rels = []
        for p in batch:
            if not p["coordinates"]:
                continue
            ids_name = p["ids_name"]
            for dim, coord_str in enumerate(p["coordinates"], start=1):
                if not coord_str:
                    continue
                # Index-based coordinates link to IMASCoordinateSpec nodes
                if coord_str.startswith("1..."):
                    coord_rels.append(
                        {
                            "source_id": p["id"],
                            "target_id": coord_str,
                            "dimension": dim,
                            "is_spec": True,
                        }
                    )
                else:
                    # Path references are relative to IDS root
                    target_path = f"{ids_name}/{coord_str}"
                    coord_rels.append(
                        {
                            "source_id": p["id"],
                            "target_id": target_path,
                            "dimension": dim,
                            "is_spec": False,
                        }
                    )

        # Create relationships to IMASCoordinateSpec nodes
        spec_rels = [r for r in coord_rels if r["is_spec"]]
        if spec_rels:
            client.query(
                """
                UNWIND $rels AS r
                MATCH (path:IMASPath {id: r.source_id})
                MATCH (spec:IMASCoordinateSpec {id: r.target_id})
                MERGE (path)-[rel:HAS_COORDINATE]->(spec)
                SET rel.dimension = r.dimension
            """,
                rels=spec_rels,
            )

        # Create relationships to IMASPath coordinate nodes
        path_rels = [r for r in coord_rels if not r["is_spec"]]
        if path_rels:
            client.query(
                """
                UNWIND $rels AS r
                MATCH (path:IMASPath {id: r.source_id})
                MATCH (coord:IMASPath {id: r.target_id})
                MERGE (path)-[rel:HAS_COORDINATE]->(coord)
                SET rel.dimension = r.dimension
            """,
                rels=path_rels,
            )

        # Step 6: Create INTRODUCED_IN relationships (only if path doesn't already have one)
        # A path should only have one INTRODUCED_IN relationship - to the first version
        # where it appeared. If it was deprecated and re-added, we don't re-introduce it.
        client.query(
            """
            UNWIND $paths AS p
            MATCH (path:IMASPath {id: p.id})
            WHERE NOT EXISTS { (path)-[:INTRODUCED_IN]->(:DDVersion) }
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
    """Batch create IMASPathChange nodes for metadata changes with semantic classification."""
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

    # Create IMASPathChange nodes with semantic classification
    client.query(
        """
        UNWIND $changes AS c
        MERGE (change:IMASPathChange {id: c.id})
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
        MATCH (change:IMASPathChange {id: c.id})
        MATCH (p:IMASPath {id: c.path})
        MATCH (v:DDVersion {id: $version})
        MERGE (change)-[:FOR_IMAS_PATH]->(p)
        MERGE (change)-[:IN_VERSION]->(v)
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


def _batch_create_error_relationships(
    client: GraphClient,
    error_relationships: dict[str, tuple[str, str]],
    batch_size: int = 1000,
) -> int:
    """Batch create HAS_ERROR relationships between error paths and data paths.

    Error paths (e.g., psi_error_upper) are linked to their data paths (e.g., psi)
    via HAS_ERROR relationships with an error_type property. This allows semantic
    search to find the primary data path without embedding redundant error field
    descriptions.

    Args:
        client: GraphClient instance
        error_relationships: Dict mapping error_path_id to (data_path_id, error_type)
            where error_type is "upper", "lower", or "index"

    Returns:
        Number of relationships created
    """
    if not error_relationships:
        return 0

    rel_list = [
        {"error_path": error_path, "data_path": data_path, "error_type": error_type}
        for error_path, (data_path, error_type) in error_relationships.items()
    ]

    created = 0
    for i in range(0, len(rel_list), batch_size):
        batch = rel_list[i : i + batch_size]
        result = client.query(
            """
            UNWIND $rels AS r
            MATCH (err:IMASPath {id: r.error_path})
            MATCH (data:IMASPath {id: r.data_path})
            MERGE (data)-[rel:HAS_ERROR]->(err)
            SET rel.error_type = r.error_type
            RETURN count(*) as created
            """,
            rels=batch,
        )
        if result:
            created += result[0].get("created", 0)

    logger.debug(f"Created {created} HAS_ERROR relationships")
    return created


def _cleanup_stale_embeddings(client: GraphClient) -> int:
    """Remove embeddings from deprecated paths.

    .. deprecated::
        Deprecated paths now retain embeddings to support DD evolution queries.
        Kept for manual cleanup if needed, but no longer called during builds.

    Returns:
        Number of paths cleaned up
    """
    result = client.query("""
        MATCH (p:IMASPath)-[:DEPRECATED_IN]->(:DDVersion)
        WHERE p.embedding IS NOT NULL
        SET p.embedding = null,
            p.embedding_text = null,
            p.embedding_hash = null
        RETURN count(p) as cleaned
    """)
    cleaned = result[0]["cleaned"] if result else 0
    if cleaned > 0:
        logger.debug(f"Cleaned up {cleaned} stale embeddings from deprecated paths")
    return cleaned


def _import_clusters(
    client: GraphClient,
    dry_run: bool,
    use_rich: bool | None = None,
) -> int:
    """Build semantic clusters and store them in the graph (graph-first).

    Pipeline:
    1. Run HDBSCAN clustering via the Clusters manager (reads embeddings
       from IMASPath nodes in the graph).
    2. Import cluster metadata + IN_CLUSTER relationships in batches.
    3. Compute cluster embeddings as the mean of member path embeddings
       directly in Neo4j — the graph is the primary store.
    4. Export cluster embeddings to .npz as a backup for CI environments
       without a live graph.

    Model name is tracked on DDVersion, not on clusters.

    Args:
        client: GraphClient instance
        dry_run: If True, don't write to graph
        use_rich: Force rich progress setting

    Returns:
        Number of clusters imported
    """
    with suppress_third_party_logging():
        try:
            from imas_codex.core.clusters import Clusters
            from imas_codex.embeddings.config import EncoderConfig

            encoder_config = EncoderConfig()
            clusters_manager = Clusters(
                encoder_config=encoder_config,
                graph_client=client,
            )

            # Single call — get_clusters() triggers build internally if
            # the data is stale. Do NOT also call is_available() as that
            # would run the dependency check a second time and trigger a
            # redundant HDBSCAN rebuild (the build takes minutes, which
            # exceeds the 1-second dependency_check_interval).
            cluster_data = clusters_manager.get_clusters()
            if not cluster_data:
                logger.warning("No cluster data produced")
                return 0

            if not dry_run:
                # --- Clear stale clusters before import ---
                deleted = 0
                while True:
                    result = client.query(
                        """
                        MATCH (c:IMASSemanticCluster)
                        WITH c LIMIT 5000
                        DETACH DELETE c
                        RETURN count(c) AS deleted
                        """
                    )
                    batch = result[0]["deleted"] if result else 0
                    deleted += batch
                    if batch < 5000:
                        break
                if deleted > 0:
                    logger.info("Cleared %d stale cluster nodes before import", deleted)

            # --- Batch import cluster nodes ---
            cluster_batch: list[dict] = []
            membership_batch: list[dict] = []
            cluster_count = 0

            for cluster in cluster_data:
                cluster_id = cluster.get("id", str(cluster_count))
                if dry_run:
                    cluster_count += 1
                    continue

                label = cluster.get("label", f"cluster_{cluster_id}")
                physics_domain = cluster.get("physics_domain", "general")
                paths = cluster.get("paths", [])
                cross_ids = cluster.get("cross_ids", False)
                similarity_score = cluster.get(
                    "similarity_score", cluster.get("similarity", 0.0)
                )
                ids_names = cluster.get("ids_names", cluster.get("ids", []))
                scope = cluster.get("scope", "global")

                cluster_batch.append({
                    "cluster_id": str(cluster_id),
                    "label": label,
                    "physics_domain": physics_domain,
                    "path_count": len(paths),
                    "cross_ids": cross_ids,
                    "similarity_score": similarity_score,
                    "scope": scope,
                    "ids_names": ids_names if ids_names else [],
                })

                for path_info in paths:
                    if isinstance(path_info, dict):
                        path = path_info.get("path", "")
                        distance = path_info.get("distance", 0.0)
                    else:
                        path = str(path_info)
                        distance = 0.0
                    if path:
                        membership_batch.append({
                            "cluster_id": str(cluster_id),
                            "path": path,
                            "distance": distance,
                        })

                cluster_count += 1

            if not dry_run and cluster_batch:
                # Batch create all cluster nodes
                client.query(
                    """
                    UNWIND $clusters AS c
                    MERGE (n:IMASSemanticCluster {id: c.cluster_id})
                    SET n.label = c.label,
                        n.physics_domain = c.physics_domain,
                        n.path_count = c.path_count,
                        n.cross_ids = c.cross_ids,
                        n.similarity_score = c.similarity_score,
                        n.scope = c.scope,
                        n.ids_names = c.ids_names
                    """,
                    clusters=cluster_batch,
                )
                logger.info("Created %d cluster nodes", len(cluster_batch))

                # Batch create IN_CLUSTER relationships
                batch_size = 2000
                for i in range(0, len(membership_batch), batch_size):
                    batch = membership_batch[i : i + batch_size]
                    client.query(
                        """
                        UNWIND $memberships AS m
                        MATCH (p:IMASPath {id: m.path})
                        MATCH (c:IMASSemanticCluster {id: m.cluster_id})
                        MERGE (p)-[r:IN_CLUSTER]->(c)
                        SET r.distance = m.distance
                        """,
                        memberships=batch,
                    )
                logger.info(
                    "Created %d IN_CLUSTER relationships", len(membership_batch)
                )

                # --- Graph-first: compute cluster embeddings from member embeddings ---
                centroid_result = client.query(
                    """
                    MATCH (p:IMASPath)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                    WHERE p.embedding IS NOT NULL
                    WITH c, collect(p.embedding) AS embeddings, count(p) AS member_count
                    WHERE member_count > 0
                    WITH c, embeddings, member_count,
                         size(embeddings[0]) AS dim
                    WITH c, member_count, dim,
                         [i IN range(0, dim - 1) |
                            reduce(s = 0.0, emb IN embeddings | s + emb[i]) / member_count
                         ] AS cluster_emb
                    SET c.embedding = cluster_emb
                    RETURN count(c) AS embeddings_set
                    """
                )
                embeddings_set = (
                    centroid_result[0]["embeddings_set"] if centroid_result else 0
                )
                missing = cluster_count - embeddings_set
                logger.info(
                    "Computed %d/%d cluster embeddings from graph",
                    embeddings_set,
                    cluster_count,
                )
                if missing > 0:
                    logger.warning(
                        "%d clusters have no embedding (members may lack embeddings)",
                        missing,
                    )

                # --- Export cluster embeddings to .npz as CI fallback ---
                _export_cluster_embeddings_npz(client, clusters_manager.file_path.parent)

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


def _export_cluster_embeddings_npz(
    client: GraphClient,
    output_dir: Path,
    filename: str = "cluster_embeddings.npz",
) -> None:
    """Export cluster embeddings from the graph to .npz as a CI fallback.

    Args:
        client: GraphClient instance
        output_dir: Directory to write the .npz file
        filename: Output filename
    """
    try:
        results = client.query(
            """
            MATCH (c:IMASSemanticCluster)
            WHERE c.embedding IS NOT NULL
            RETURN c.id AS cluster_id, c.embedding AS embedding
            ORDER BY c.id
            """
        )
        if not results:
            logger.debug("No cluster embeddings to export")
            return

        cluster_ids = [r["cluster_id"] for r in results]
        centroids = np.array([r["embedding"] for r in results], dtype=np.float32)

        output_file = output_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_file,
            centroids=centroids,
            centroid_cluster_ids=np.array(cluster_ids, dtype=object),
            # Empty arrays for label embeddings (not exported from graph)
            label_embeddings=np.array([], dtype=np.float32),
            label_cluster_ids=np.array([], dtype=object),
        )
        logger.info(
            "Exported %d cluster embeddings to %s",
            len(cluster_ids),
            output_file.name,
        )
    except Exception as e:
        logger.warning("Failed to export cluster embeddings to .npz: %s", e)


def import_semantic_clusters(
    client: GraphClient,
    dry_run: bool = False,
    use_rich: bool | None = None,
) -> int:
    """Public API for importing semantic clusters into the graph.

    Thin wrapper around ``_import_clusters`` so external callers
    (e.g. ``imas-codex imas clusters sync``) have a stable name.
    """
    return _import_clusters(client, dry_run=dry_run, use_rich=use_rich)


# Public API exports
__all__ = [
    "build_dd_graph",
    "clear_dd_graph",
    "get_all_dd_versions",
    "extract_paths_for_version",
    "compute_version_changes",
    "import_semantic_clusters",
    "load_path_mappings",
]


def clear_dd_graph(
    client: GraphClient,
    batch_size: int = 5000,
) -> dict[str, int]:
    """Clear all IMAS Data Dictionary nodes from the graph.

    Deletes DD-specific nodes in referential-integrity order, cascading to
    orphaned nodes. Cross-references from facility nodes (MAPS_TO_IMAS,
    MENTIONS_IMAS, etc.) are detached before deletion.

    Unit nodes are only deleted if no non-DD nodes reference them.

    Deletion order:
    1. EmbeddingChange (leaf, references IMASPath + DDVersion)
    2. IMASPathChange (references IMASPath + DDVersion)
    3. IMASSemanticCluster (references IMASPath via IN_CLUSTER)
    4. IdentifierSchema (referenced by IMASPath)
    5. IMASPath (bulk — the largest set)
    6. IDS (referenced by IMASPath)
    7. DDVersion (referenced by IMASPath, IMASPathChange)
    8. IMASCoordinateSpec (referenced by coordinate relationships)
    9. Orphaned Unit nodes (not referenced by facility nodes)

    Args:
        client: GraphClient instance
        batch_size: Nodes to delete per batch (default 5000)

    Returns:
        Dict with counts per node type deleted
    """
    results: dict[str, int] = {}

    # Define deletion order: (label, result_key)
    # Order matters — delete children before parents
    node_types = [
        ("EmbeddingChange", "embedding_changes"),
        ("IMASPathChange", "path_changes"),
        ("IMASSemanticCluster", "clusters"),
        ("IdentifierSchema", "identifier_schemas"),
        ("IMASPath", "paths"),
        ("IDS", "ids_nodes"),
        ("DDVersion", "versions"),
        ("IMASCoordinateSpec", "coordinate_specs"),
    ]

    for label, key in node_types:
        total = 0
        while True:
            result = client.query(
                f"""
                MATCH (n:{label})
                WITH n LIMIT $batch_size
                DETACH DELETE n
                RETURN count(n) AS deleted
                """,
                batch_size=batch_size,
            )
            deleted = result[0]["deleted"] if result else 0
            total += deleted
            if deleted < batch_size:
                break
        results[key] = total
        if total > 0:
            logger.debug(f"Deleted {total} {label} nodes")

    # Delete orphaned Unit nodes (not referenced by any non-DD node)
    orphan_result = client.query("""
        MATCH (u:Unit)
        WHERE NOT (u)<-[:HAS_UNIT]-()
        DETACH DELETE u
        RETURN count(u) AS deleted
    """)
    results["orphaned_units"] = orphan_result[0]["deleted"] if orphan_result else 0
    if results["orphaned_units"] > 0:
        logger.debug(f"Deleted {results['orphaned_units']} orphaned Unit nodes")

    # Drop DD-specific vector indexes (they're empty now)
    # Derive DD index names from schema rather than hardcoding
    from imas_codex.graph.schema import GraphSchema

    dd_schema = GraphSchema(
        schema_path=str(Path(__file__).parent.parent / "schemas" / "imas_dd.yaml")
    )
    dd_index_names = [idx[0] for idx in dd_schema.vector_indexes]
    for index_name in dd_index_names:
        try:
            client.query(f"DROP INDEX {index_name} IF EXISTS")
            logger.debug(f"Dropped vector index: {index_name}")
        except Exception:
            pass  # Index may not exist

    total_deleted = sum(results.values())
    logger.info(f"Cleared {total_deleted} DD nodes from graph")
    return results
