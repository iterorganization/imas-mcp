"""
Build the IMAS Data Dictionary Knowledge Graph.

This module provides functions to populate Neo4j with IMAS DD structure:
- DDVersion nodes for all available DD versions
- IDS nodes for top-level structures
- IMASNode nodes with hierarchical relationships
- Unit and IMASCoordinateSpec nodes
- Version tracking (INTRODUCED_IN, DEPRECATED_IN, RENAMED_TO)
- IMASNodeChange nodes for metadata changes with semantic classification
- IMASSemanticCluster nodes from existing clusters (optional)
- Embeddings for IMASNode nodes with content-based caching

The graph is augmented incrementally, not rebuilt - this preserves
links to facility data (DataNodes, IMASMappings).

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
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import imas
import numpy as np

from imas_codex import dd_version as current_dd_version
from imas_codex.core.paths import strip_path_annotations
from imas_codex.core.physics_categorization import physics_categorizer
from imas_codex.core.progress_monitor import (
    create_progress_monitor,
)
from imas_codex.graph.client import GraphClient

logger = logging.getLogger(__name__)

# Bump when metadata properties change (new DD XML attributes, schema fields,
# etc.) to invalidate the build hash and force re-extraction on next --force.
_BUILD_SCHEMA_VERSION = 2

# Alias for backward-compat within this module
_strip_dd_indices = strip_path_annotations


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

FIELD_TO_CHANGE_TYPE = {
    "ndim": "structure_changed",
    "identifier_enum_name": "structure_changed",
    "maxoccur": "maxoccur_changed",
}

DATA_FORMAT_KEYWORDS = [
    "dimension",
    "second dimension",
    "first dimension",
    "column",
    "matrix elements",
    "encoding",
    "polarity",
    "format",
    "sides",
    "index convention",
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

    # Check for data format/encoding convention changes
    format_keywords = []
    for kw in DATA_FORMAT_KEYWORDS:
        if kw in new_lower and kw not in old_lower:
            format_keywords.append(kw)
    if format_keywords:
        return "data_format_change", format_keywords

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
    """Generate embedding text for an IMAS DD node.

    Concatenates the full IMAS path with the enriched description.
    The path provides terminal segment matching (ip, b0, psi) and
    hierarchy context; the description provides physics semantics
    with abbreviations from the enrichment prompt.

    Documentation excerpts and keywords are deliberately excluded —
    at dim 256 (Matryoshka), longer text dilutes the primary signal
    and degrades retrieval quality. The enriched descriptions already
    contain abbreviations and physics context from the LLM enrichment.

    Args:
        path: Full path (e.g., "equilibrium/time_slice/profiles_1d/psi")
        path_info: Path metadata dict
        ids_info: Optional IDS-level metadata for context

    Returns:
        Concise text optimized for dim-256 sentence transformer embedding
    """
    desc = (path_info.get("description") or "").strip()
    doc = (path_info.get("documentation") or "").strip()

    # Primary text: prefer enriched description, fall back to documentation
    primary = desc if desc else doc
    if not primary:
        return ""

    # Full IMAS path + primary description only
    # At dim 256, adding doc excerpts/keywords dilutes cosine similarity
    return f"{path}. {primary}"


def compute_content_hash(text: str) -> str:
    """
    Compute SHA256 hash of content for change tracking.

    Args:
        text: Text to hash (typically enriched_text)

    Returns:
        First 16 characters of SHA256 hex digest
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# Mapping of accessor terminal names to keywords for parent inheritance.
# When a concept node (STRUCTURE/STRUCT_ARRAY) has children with these names,
# the keywords are added to the parent to improve BM25/CONTAINS discoverability.
KEYWORD_WORTHY_CHILDREN: dict[str, list[str]] = {
    "r": ["radius", "major_radius", "R"],
    "z": ["height", "vertical", "Z"],
    "phi": ["toroidal_angle", "azimuthal"],
    "time": ["timebase", "temporal"],
    "measured": ["measured", "measurement"],
    "reconstructed": ["reconstructed", "reconstruction"],
    "parallel": ["parallel_component"],
    "poloidal": ["poloidal_component"],
    "radial": ["radial_component"],
    "toroidal": ["toroidal_component"],
    "value": ["value", "scalar"],
    "data": ["data", "profile_data"],
    "psi": ["poloidal_flux", "psi"],
    "rho_tor_norm": ["normalized_toroidal_flux", "rho"],
}


def inherit_child_keywords(gc: "GraphClient") -> int:
    """Add child accessor names as keywords on parent concept nodes.

    Concept nodes (STRUCTURE/STRUCT_ARRAY) inherit relevant child accessor
    names so queries like "radius of X-point" match the parent through
    BM25/CONTAINS keyword search even when the child node itself is excluded
    from search results (template-enriched accessor terminal).

    Returns:
        Number of parent nodes updated
    """
    mappings = [
        {"name": k, "extra_keywords": v} for k, v in KEYWORD_WORTHY_CHILDREN.items()
    ]
    result = gc.query(
        """
        UNWIND $mappings AS m
        MATCH (child:IMASNode)
        WHERE child.name = m.name AND child.enrichment_source = 'template'
        MATCH (child)-[:HAS_PARENT]->(parent:IMASNode)
        WHERE parent.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
        WITH parent, collect(DISTINCT m.extra_keywords) AS kw_lists
        WITH parent, reduce(acc = [], kws IN kw_lists | acc + kws) AS all_kws
        SET parent.keywords = [x IN (coalesce(parent.keywords, []) + all_kws) WHERE x IS NOT NULL | x]
        RETURN count(parent) AS updated
        """,
        mappings=mappings,
    )
    updated = result[0]["updated"] if result else 0
    logger.info("inherit_child_keywords: updated %d parent nodes", updated)
    return updated


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


def _detect_error_relationships(
    paths_data: dict[str, dict],
) -> dict[str, tuple[str, str]]:
    """Detect error field relationships from path data.

    Identifies error paths (e.g., psi_error_upper) and maps them to their
    base data paths (e.g., psi) for HAS_ERROR relationship creation.

    Args:
        paths_data: Dict mapping path_id to path metadata

    Returns:
        Dict mapping error_path -> (data_path, error_type)
        where error_type is "upper", "lower", or "index"
    """
    error_relationships: dict[str, tuple[str, str]] = {}

    for path_id, path_info in paths_data.items():
        name = path_info.get("name", "")
        match = ERROR_FIELD_PATTERN.match(name)
        if match:
            base_name = match.group(1)
            error_type = match.group(2)  # "upper", "lower", or "index"
            parent_path = path_info.get("parent_path", "")
            if parent_path:
                data_path = f"{parent_path}/{base_name}"
                error_relationships[path_id] = (data_path, error_type)

    return error_relationships


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
            MATCH (p:IMASNode {id: path_id})
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
            MATCH (p:IMASNode {id: path_id})
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

    Creates HAS_EMBEDDING_CHANGE relationships linking IMASNode nodes to
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
        MATCH (p:IMASNode {id: c.path_id})
        MERGE (p)-[r:HAS_EMBEDDING_CHANGE {
            change_type: c.change_type,
            dd_version: c.dd_version
        }]->(ec:EmbeddingChange {
            path_id: c.path_id,
            change_type: c.change_type,
            dd_version: c.dd_version
        })
        ON CREATE SET
            ec.id = c.path_id + ':' + c.dd_version + ':' + c.change_type,
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
    on_store_batch: "Callable[[list[str], float], None] | None" = None,
) -> dict[str, int]:
    """
    Generate and store embeddings for IMASNode nodes with content-based caching.

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

            import time as _time

            store_start = _time.time()
            client.query(
                """
                UNWIND $batch AS b
                MATCH (p:IMASNode {id: b.path_id})
                SET p.embedding_text = b.embedding_text,
                    p.embedding = b.embedding,
                    p.embedding_hash = b.embedding_hash,
                    p.status = 'embedded',
                    p.claimed_at = null,
                    p.claim_token = null
                """,
                batch=batch_data,
            )
            store_time = _time.time() - store_start

            if on_store_batch:
                on_store_batch(
                    [bd["path_id"] for bd in batch_data],
                    store_time,
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


def extract_cocos_for_version(version: str) -> tuple[int | None, int]:
    """Extract COCOS value and labeled field count from DD XML.

    Args:
        version: DD version string (e.g., "3.42.0", "4.0.0")

    Returns:
        Tuple of (cocos_value, cocos_labeled_field_count).
        cocos_value is None for versions that don't declare COCOS in XML.
    """
    tree = imas.dd_zip.dd_etree(version)
    root = tree.getroot() if hasattr(tree, "getroot") else tree

    cocos_el = root.find("cocos")
    cocos_val = int(cocos_el.text) if cocos_el is not None else None

    labeled_count = sum(
        1 for f in root.iter("field") if f.get("cocos_label_transformation")
    )

    return cocos_val, labeled_count


def extract_cocos_labels_for_version(version: str) -> dict[str, str]:
    """Extract per-field cocos_label_transformation from DD XML.

    Args:
        version: DD version string

    Returns:
        Dict mapping field path to cocos_label_transformation value.
        Only includes fields that have the attribute.
    """
    tree = imas.dd_zip.dd_etree(version)
    root = tree.getroot() if hasattr(tree, "getroot") else tree

    labels = {}
    for ids_el in root.findall("IDS"):
        ids_name = ids_el.get("name", "")
        for field in ids_el.iter("field"):
            cocos_label = field.get("cocos_label_transformation")
            if cocos_label:
                field_path = field.get("path", "")
                if field_path and ids_name:
                    full_path = f"{ids_name}/{field_path}"
                    labels[full_path] = cocos_label

    return labels


def _backfill_cocos_labels(client: "GraphClient", version_data: dict[str, dict]) -> int:
    """Backfill COCOS labels for DD versions where XML lacks them.

    DD versions 4.0.0–4.1.1 removed ``cocos_label_transformation`` from
    the XML schema while retaining ``cocos_transformation_expression``.
    This leaves ~200-250 COCOS-dependent paths unlabelled in 4.x.

    Three inference sources, applied in priority order:

    1. **Forward-port** from the last labelled 3.x version (typically
       3.42.2 with ~680 labelled fields).
    2. **Expression parsing** — extract label names from
       ``cocos_transformation_expression`` (e.g. ``"- {psi_like}"``).
    3. **imas-python sign flip paths** — paths identified by
       :func:`get_sign_flip_paths` are ``psi_like`` by definition.

    Every labelled path also receives a ``cocos_label_source`` tag for
    provenance (``xml``, ``inferred_forward``, ``inferred_expression``,
    or ``inferred_sign_flip``).

    Returns:
        Count of labels backfilled.
    """
    sorted_versions = sorted(version_data.keys())
    latest_version = sorted_versions[-1]
    latest_paths = version_data[latest_version]["paths"]

    # Build reference labels from the latest 3.x version
    last_3x: str | None = None
    for v in reversed(sorted_versions):
        if v.startswith("3."):
            last_3x = v
            break

    ref_labels: dict[str, str] = {}
    if last_3x:
        for path, info in version_data[last_3x]["paths"].items():
            label = info.get("cocos_label_transformation")
            if label:
                ref_labels[path] = label

    updates: list[dict] = []
    handled: set[str] = set()

    for path, info in latest_paths.items():
        if info.get("cocos_label_transformation"):
            continue  # Already labelled from XML

        # Source 1: Forward-port from 3.x
        if path in ref_labels:
            updates.append(
                {"id": path, "label": ref_labels[path], "source": "inferred_forward"}
            )
            handled.add(path)
            continue

        # Source 2: Parse cocos_transformation_expression
        expr = info.get("cocos_transformation_expression", "")
        if expr:
            match = re.search(r"\{(\w+_like)\}", expr)
            if match:
                updates.append(
                    {
                        "id": path,
                        "label": match.group(1),
                        "source": "inferred_expression",
                    }
                )
                handled.add(path)
                continue

    # Source 3: imas-python sign flip paths for remaining unlabelled
    try:
        from imas_codex.cocos.transforms import (
            get_sign_flip_paths,
            list_ids_with_sign_flips,
        )

        for ids_name in list_ids_with_sign_flips():
            for rel_path in get_sign_flip_paths(ids_name):
                full_path = f"{ids_name}/{rel_path}"
                if full_path in handled:
                    continue
                if full_path not in latest_paths:
                    continue
                if latest_paths[full_path].get("cocos_label_transformation"):
                    continue
                updates.append(
                    {
                        "id": full_path,
                        "label": "psi_like",
                        "source": "inferred_sign_flip",
                    }
                )
                handled.add(full_path)
    except ImportError:
        logger.debug(
            "imas_codex.cocos.transforms not available, skipping sign flip backfill"
        )

    # Apply backfilled labels in batches
    if updates:
        for i in range(0, len(updates), 1000):
            batch = updates[i : i + 1000]
            client.query(
                """
                UNWIND $updates AS u
                MATCH (p:IMASNode {id: u.id})
                SET p.cocos_label_transformation = u.label,
                    p.cocos_label_source = u.source
                """,
                updates=batch,
            )
        logger.info(
            "Backfilled %d COCOS labels (forward=%d, expression=%d, sign_flip=%d)",
            len(updates),
            sum(1 for u in updates if u["source"] == "inferred_forward"),
            sum(1 for u in updates if u["source"] == "inferred_expression"),
            sum(1 for u in updates if u["source"] == "inferred_sign_flip"),
        )

    # Mark XML-sourced labels for provenance tracking
    client.query(
        """
        MATCH (p:IMASNode)
        WHERE p.cocos_label_transformation IS NOT NULL
        AND p.cocos_label_source IS NULL
        SET p.cocos_label_source = 'xml'
        """
    )

    return len(updates)


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

    # Extract IDS-level lifecycle metadata from DD XML
    tree = imas.dd_zip.dd_etree(version)
    root = tree.getroot() if hasattr(tree, "getroot") else tree
    ids_xml_meta = {}
    for ids_el in root.findall("IDS"):
        name = ids_el.get("name", "")
        if name:
            ids_xml_meta[name] = {
                "lifecycle_status": ids_el.get("lifecycle_status"),
                "lifecycle_version": ids_el.get("lifecycle_version"),
                "lifecycle_last_change": ids_el.get("lifecycle_last_change"),
                "ids_type": ids_el.get("type"),
                "maxoccur": int(ids_el.get("maxoccur"))
                if ids_el.get("maxoccur")
                else None,
            }

    # Extract field-level metadata from DD XML (attributes not in metadata API)
    field_xml_meta: dict[str, dict] = {}
    for ids_el in root.findall("IDS"):
        ids_name_xml = ids_el.get("name", "")
        for field_el in ids_el.iter("field"):
            field_path = field_el.get("path", "")
            if field_path and ids_name_xml:
                full_path = f"{ids_name_xml}/{field_path}"
                meta: dict = {}
                # Lifecycle
                ls = field_el.get("lifecycle_status")
                if ls:
                    meta["lifecycle_status"] = ls
                    meta["lifecycle_version"] = field_el.get("lifecycle_version")
                # Timebasepath
                tbp = field_el.get("timebasepath")
                if tbp:
                    meta["timebasepath"] = tbp
                # Human-readable path
                path_doc = field_el.get("path_doc")
                if path_doc and path_doc != field_path:
                    meta["path_doc"] = path_doc
                # Introduction version
                iav = (
                    field_el.get("introduced_after_version")
                    or field_el.get("introduced_after")
                    or field_el.get("introduced-after-version")
                )
                if iav:
                    meta["introduced_after_version"] = iav
                # Non-backward-compatible changes
                nbc_ver = field_el.get("change_nbc_version")
                if nbc_ver:
                    meta["change_nbc_version"] = nbc_ver
                    meta["change_nbc_description"] = field_el.get(
                        "change_nbc_description", ""
                    )
                    meta["change_nbc_previous_name"] = field_el.get(
                        "change_nbc_previous_name", ""
                    )
                    nbc_prev_type = field_el.get("change_nbc_previous_type")
                    if nbc_prev_type:
                        meta["change_nbc_previous_type"] = nbc_prev_type
                # COCOS transformation expression
                cocos_expr = field_el.get("cocos_transformation_expression")
                if cocos_expr:
                    meta["cocos_transformation_expression"] = cocos_expr
                # Alternative coordinate
                alt_coord = field_el.get("alternative_coordinate1")
                if alt_coord:
                    meta["alternative_coordinate1"] = alt_coord
                # URL
                url = field_el.get("url")
                if url:
                    meta["url"] = url
                # Identifier schema (doc_identifier)
                doc_id = field_el.get("doc_identifier")
                if doc_id:
                    meta["doc_identifier"] = doc_id
                # Coordinate same-as references (per dimension)
                for dim in range(1, 7):
                    csa = field_el.get(f"coordinate{dim}_same_as")
                    if csa:
                        meta[f"coordinate{dim}_same_as"] = csa
                if meta:
                    field_xml_meta[full_path] = meta

    ids_info = {}
    paths = {}
    units = set()

    for ids_name in sorted(ids_names):
        try:
            ids_def = factory.new(ids_name)
            metadata = ids_def.metadata

            # IDS-level info (merge API + XML metadata)
            xml_meta = ids_xml_meta.get(ids_name, {})
            ids_info[ids_name] = {
                "name": ids_name,
                "documentation": metadata.documentation or "",
                "physics_domain": physics_categorizer.get_domain_for_ids(
                    ids_name
                ).value,
                "lifecycle_status": xml_meta.get("lifecycle_status"),
                "lifecycle_version": xml_meta.get("lifecycle_version"),
                "lifecycle_last_change": xml_meta.get("lifecycle_last_change"),
                "ids_type": xml_meta.get("ids_type"),
            }

            # Extract all paths recursively
            _extract_paths_recursive(
                metadata, ids_name, "", paths, units, field_xml_meta
            )

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
    field_xml_meta: dict[str, dict] | None = None,
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

        # COCOS label transformation (e.g., psi_like, ip_like, q_like)
        cocos_label = getattr(child, "cocos_label_transformation", None)
        if cocos_label:
            path_info["cocos_label_transformation"] = str(cocos_label)

        # Merge field-level XML metadata (lifecycle, timebasepath, etc.)
        if field_xml_meta:
            xml_meta = field_xml_meta.get(full_path)
            if xml_meta:
                for key in (
                    "lifecycle_status",
                    "lifecycle_version",
                    "timebasepath",
                    "path_doc",
                    "introduced_after_version",
                    "change_nbc_version",
                    "change_nbc_description",
                    "change_nbc_previous_name",
                    "change_nbc_previous_type",
                    "cocos_transformation_expression",
                    "alternative_coordinate1",
                    "url",
                    "doc_identifier",
                    "coordinate1_same_as",
                    "coordinate2_same_as",
                    "coordinate3_same_as",
                    "coordinate4_same_as",
                    "coordinate5_same_as",
                    "coordinate6_same_as",
                ):
                    val = xml_meta.get(key)
                    if val:
                        path_info[key] = val

        # Extract identifier_enum name from metadata API
        # imas-python raises KeyError for missing identifier schemas
        # (e.g. transport_solver_numerics_computation_mode), so we need
        # to catch that rather than relying on getattr's AttributeError.
        try:
            id_enum = child.identifier_enum
        except (AttributeError, KeyError):
            id_enum = None
        if id_enum and hasattr(id_enum, "__name__"):
            path_info["identifier_enum_name"] = id_enum.__name__
            # Store enum values for IdentifierSchema creation
            if hasattr(id_enum, "__members__"):
                path_info["_identifier_enum_values"] = {
                    m.name: m.value for m in id_enum
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
            _extract_paths_recursive(
                child, ids_name, f"{child_path}/", paths, units, field_xml_meta
            )


def _units_changed(old_raw: str, new_raw: str) -> tuple[bool, str]:
    """Compare units via pint normalization.

    Returns:
        (changed, change_subtype) where change_subtype is one of:
        - 'cosmetic'           — same after normalization (not stored)
        - 'sentinel_resolved'  — as_parent/placeholder → concrete unit
        - 'dim_equivalent'     — different symbol, same dimension (e.g. J.m^-3 → Pa)
        - 'dim_incompatible'   — genuinely different physics (e.g. W → Wb)
    """
    from imas_codex.units import normalize_unit_symbol, unit_registry

    old_norm = normalize_unit_symbol(old_raw)
    new_norm = normalize_unit_symbol(new_raw)

    # Tier 1: Identical after normalization → not a change
    if old_norm == new_norm:
        return False, "cosmetic"

    # Tier 2: One or both are sentinels/unparseable
    if old_norm is None or new_norm is None:
        return True, "sentinel_resolved"

    # Tier 3: Both parse — check dimensional compatibility
    try:
        old_p = unit_registry.parse_expression(old_norm)
        new_p = unit_registry.parse_expression(new_norm)
        if old_p.is_compatible_with(new_p):
            return True, "dim_equivalent"
    except Exception:
        pass

    return True, "dim_incompatible"


def _classify_breaking_level(change_type: str, change: dict) -> str:
    """Classify a change as breaking/advisory/informational."""
    RULES = {
        "path_removed": "breaking",
        "path_renamed": "breaking",  # Non-NBC renames break interfaces
        "data_type": "breaking",
        "cocos_label_transformation": "breaking",
        "coordinates_changed": "advisory",
        "lifecycle_status": "advisory",
        "node_type": "advisory",
        "timebasepath": "informational",
        "maxoccur_changed": "advisory",
        "structure_changed": "advisory",
        "path_added": "informational",
        "convention_change": "breaking",
    }

    if change_type == "units":
        subtype = change.get("unit_change_subtype", "")
        if subtype == "dim_incompatible":
            return "breaking"
        if subtype == "dim_equivalent":
            return "advisory"
        return "informational"  # sentinel_resolved

    if change_type == "documentation":
        semantic = change.get("semantic_type", "none")
        if semantic == "sign_convention":
            return "breaking"  # sign convention changes break external codes
        if semantic == "coordinate_convention":
            return "advisory"
        if semantic == "data_format_change":
            return "breaking"
        return "informational"

    return RULES.get(change_type, "informational")


def _detect_renames(added: set[str], removed: set[str], new_paths: dict) -> list[dict]:
    """Detect renames using NBC metadata.

    Priority:
    1. NBC previous_name metadata (ground truth from DD maintainers)
    """
    renames = []

    # Priority 1: NBC metadata
    for path_id in added:
        path_info = new_paths.get(path_id, {})
        prev_name = path_info.get("change_nbc_previous_name")
        if prev_name and prev_name in removed:
            renames.append(
                {
                    "old_path": prev_name,
                    "new_path": path_id,
                    "source": "nbc_metadata",
                }
            )

    return renames


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
        - changed: paths with metadata changes (including renames,
          path_added, path_removed)
    """
    old_set = set(old_paths.keys())
    new_set = set(new_paths.keys())

    added = new_set - old_set
    removed = old_set - new_set
    common = old_set & new_set

    # Detect renames from NBC metadata
    renames = _detect_renames(added, removed, new_paths)
    matched_added = {r["new_path"] for r in renames}
    matched_removed = {r["old_path"] for r in renames}

    # Check for metadata changes in common paths
    changed: dict[str, list[dict]] = {}
    for path in common:
        old_info = old_paths[path]
        new_info = new_paths[path]

        changes: list[dict] = []
        for field in (
            "units",
            "documentation",
            "data_type",
            "node_type",
            "cocos_label_transformation",
            "lifecycle_status",
            "coordinates",
            "timebasepath",
            "maxoccur",
            "ndim",
            "identifier_enum_name",
        ):
            old_val = old_info.get(field, "")
            new_val = new_info.get(field, "")

            if field == "units":
                unit_changed, subtype = _units_changed(
                    str(old_val) if old_val else "",
                    str(new_val) if new_val else "",
                )
                if not unit_changed:
                    continue
                change_entry = {
                    "field": "units",
                    "old_value": str(old_val) if old_val else "",
                    "new_value": str(new_val) if new_val else "",
                    "unit_change_subtype": subtype,
                }
                change_entry["breaking_level"] = _classify_breaking_level(
                    "units", change_entry
                )
                changes.append(change_entry)
            elif field == "maxoccur":
                old_mo = old_info.get("maxoccur")
                new_mo = new_info.get("maxoccur")
                if old_mo == new_mo:
                    continue
                change_entry = {
                    "field": "maxoccur",
                    "old_value": str(old_mo) if old_mo is not None else "unbounded",
                    "new_value": str(new_mo) if new_mo is not None else "unbounded",
                }
                change_entry["breaking_level"] = _classify_breaking_level(
                    "maxoccur_changed", change_entry
                )
                changes.append(change_entry)
            else:
                if str(old_val) != str(new_val):
                    change_entry = {
                        "field": field,
                        "old_value": str(old_val) if old_val else "",
                        "new_value": str(new_val) if new_val else "",
                    }
                    # Classify doc changes semantically BEFORE computing breaking level
                    if field == "documentation":
                        semantic_type, keywords = classify_doc_change(
                            change_entry["old_value"], change_entry["new_value"]
                        )
                        change_entry["semantic_type"] = semantic_type
                        change_entry["keywords_detected"] = keywords
                    change_entry["breaking_level"] = _classify_breaking_level(
                        FIELD_TO_CHANGE_TYPE.get(field, field), change_entry
                    )
                    changes.append(change_entry)

        if changes:
            changed[path] = changes

    # Add rename change events
    for r in renames:
        changes_for_path = changed.setdefault(r["new_path"], [])
        # NBC-detected renames are handled by IMAS access layer (non-breaking)
        level = "informational" if r.get("source") == "nbc_metadata" else "breaking"
        changes_for_path.append(
            {
                "field": "path_renamed",
                "old_value": r["old_path"],
                "new_value": r["new_path"],
                "breaking_level": level,
            }
        )

    # Add path_added events (excluding renames)
    for path_id in added:
        if path_id not in matched_added:
            changes_for_path = changed.setdefault(path_id, [])
            changes_for_path.append(
                {
                    "field": "path_added",
                    "old_value": "",
                    "new_value": path_id,
                    "breaking_level": "informational",
                }
            )

    # Add path_removed events (excluding renames)
    for path_id in removed:
        if path_id not in matched_removed:
            changes_for_path = changed.setdefault(path_id, [])
            changes_for_path.append(
                {
                    "field": "path_removed",
                    "old_value": path_id,
                    "new_value": "",
                    "breaking_level": "breaking",
                }
            )

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


def _compute_cluster_content_hash(sorted_paths: list[str]) -> str:
    """Compute a content hash for a cluster based on its sorted member paths.

    This hash serves as the cluster's stable identity across HDBSCAN runs.
    If the same set of paths clusters together, the hash matches and the
    existing label/description/embeddings are preserved.
    """
    return hashlib.sha256("\n".join(sorted_paths).encode("utf-8")).hexdigest()[:16]


def _compute_build_hash(
    versions: list[str],
    ids_filter: set[str] | None,
) -> str:
    """Compute a deterministic hash of the build parameters.

    Used to detect whether a previous build with the same parameters
    already populated the graph, allowing fast no-op re-runs.
    Includes _BUILD_SCHEMA_VERSION so metadata additions invalidate
    the hash automatically.
    """
    parts = [
        str(_BUILD_SCHEMA_VERSION),
        ",".join(sorted(versions)),
        ",".join(sorted(ids_filter)) if ids_filter else "",
    ]
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


def _check_graph_up_to_date(
    client: GraphClient,
    build_hash: str,
    versions: list[str],
) -> bool:
    """Check if the graph already contains a build matching the given hash.

    Validates:
    - DDVersion.build_hash on the current version matches
    - All requested versions exist
    - Embedding coverage is complete
    - Clusters exist

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
        emb_result = client.query(
            """
            MATCH (p:IMASNode)
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
        cl_result = client.query("MATCH (c:IMASSemanticCluster) RETURN count(c) AS cnt")
        if not cl_result or cl_result[0]["cnt"] == 0:
            return False

        return True
    except Exception as e:
        logger.debug(f"Graph up-to-date check failed: {e}")
        return False


def _check_build_nodes_exist(
    client: GraphClient,
    build_hash: str,
    versions: list[str],
) -> bool:
    """Check if extract+build nodes already exist (regardless of downstream state).

    Lighter check than ``_check_graph_up_to_date`` — only verifies that
    DDVersion nodes exist with a matching build_hash and IMASNode nodes
    are present.  Does NOT check embedding or cluster completeness, so
    extract+build can be skipped while enrich/embed/cluster continue.
    """
    try:
        result = client.query(
            """
            MATCH (d:DDVersion {is_current: true})
            RETURN d.build_hash AS hash
            """
        )
        if not result or result[0].get("hash") != build_hash:
            return False

        ver_result = client.query(
            "MATCH (d:DDVersion) RETURN collect(d.id) AS versions"
        )
        stored_versions = ver_result[0]["versions"] if ver_result else []
        if set(versions) != set(stored_versions):
            return False

        # Check that IMASNodes exist (any status)
        node_result = client.query("MATCH (p:IMASNode) RETURN count(p) AS cnt")
        if not node_result or node_result[0]["cnt"] == 0:
            return False

        return True
    except Exception as e:
        logger.debug(f"Build-nodes-exist check failed: {e}")
        return False


# =============================================================================
# Phase Functions (called by workers)
# =============================================================================


def phase_extract(
    versions: list[str],
    ids_filter: set[str] | None = None,
    on_progress: "Callable[[int, int], None] | None" = None,
) -> tuple[dict[str, dict], set[str]]:
    """Extract paths from DD XML for all versions.

    Pure parsing — no graph client needed.

    Returns:
        (version_data, all_units) tuple where version_data maps
        version string to extracted data dict.
    """
    all_units: set[str] = set()
    version_data: dict[str, dict] = {}
    total = len(versions)

    if on_progress:
        on_progress(0, total)

    for i, version in enumerate(versions):
        try:
            data = extract_paths_for_version(version, ids_filter=ids_filter)
            version_data[version] = data
            all_units.update(data["units"])
        except Exception as e:
            logger.error("Error extracting %s: %s", version, e)
        if on_progress:
            on_progress(i + 1, total)

    return version_data, all_units


def phase_build(
    client: GraphClient,
    versions: list[str],
    version_data: dict[str, dict],
    all_units: set[str],
    *,
    dry_run: bool = False,
    on_progress: "Callable[[int, int], None] | None" = None,
    on_version: "Callable[[str], None] | None" = None,
) -> dict[str, int]:
    """Create graph nodes from extracted data.

    Creates Unit, CoordinateSpec, IdentifierSchema nodes; IDS and
    IMASNode nodes per version with hierarchical relationships;
    RENAMED_TO, COCOS label, and field metadata updates.

    Args:
        on_progress: Called with ``(paths_processed, total_paths)``.
        on_version: Called with the version string at the start of each version.

    Returns:
        Stats dict with creation/update counts.
    """
    stats: dict[str, int] = {
        "ids_created": 0,
        "paths_created": 0,
        "units_created": 0,
        "path_changes_created": 0,
        "definitions_changed": 0,
        "cocos_labels_updated": 0,
        "cocos_labels_backfilled": 0,
        "identifier_schemas_created": 0,
        "error_relationships": 0,
        "orphaned_units_deleted": 0,
    }

    # Create Unit / CoordinateSpec nodes
    if not dry_run:
        _create_unit_nodes(client, all_units)
    stats["units_created"] = len(all_units)

    all_coord_specs: set[str] = set()
    for ver_data in version_data.values():
        for path_info in ver_data["paths"].values():
            for coord_str in path_info.get("coordinates", []):
                if coord_str and coord_str.startswith("1..."):
                    all_coord_specs.add(coord_str)
    if not dry_run:
        _create_coordinate_spec_nodes(client, all_coord_specs)

    # Create IdentifierSchema nodes from the latest version's enum data
    if not dry_run and version_data:
        latest_ver = sorted(version_data.keys())[-1]
        latest_paths = version_data[latest_ver]["paths"]
        id_schemas = _collect_identifier_schemas(latest_paths)
        if id_schemas:
            _create_identifier_schema_nodes(client, id_schemas)
        stats["identifier_schemas_created"] = len(id_schemas)

    # Per-version: IDS + IMASNode + relationships + path changes
    prev_paths: dict[str, dict] = {}
    total_paths = sum(
        len(version_data[v]["paths"]) for v in versions if v in version_data
    )
    paths_processed = 0

    if on_progress:
        on_progress(0, total_paths)

    for i, version in enumerate(versions):
        if version not in version_data:
            continue

        if on_version:
            on_version(version)

        data = version_data[version]
        changes = compute_version_changes(prev_paths, data["paths"])

        if not dry_run:
            _batch_upsert_ids_nodes(client, data["ids_info"], version, i == 0)
            stats["ids_created"] = max(stats["ids_created"], len(data["ids_info"]))

            new_paths_data = {p: data["paths"][p] for p in changes["added"]}
            _batch_create_path_nodes(client, new_paths_data, version)
            stats["paths_created"] += len(changes["added"])

            _batch_mark_paths_deprecated(client, changes["removed"], version)

            change_count = _batch_create_path_changes(
                client, changes["changed"], version
            )
            stats["path_changes_created"] += change_count

            # Count paths with documentation (definition) changes
            for path_changes in changes["changed"].values():
                if any(c["field"] == "documentation" for c in path_changes):
                    stats["definitions_changed"] += 1

        prev_paths = data["paths"]
        paths_processed += len(data["paths"])

        if on_progress:
            on_progress(paths_processed, total_paths)

    # RENAMED_TO relationships
    if not dry_run:
        mappings = load_path_mappings(current_dd_version)
        _batch_create_renamed_to(client, mappings.get("old_to_new", {}))

    # Update cocos_label_transformation on existing IMASNode nodes
    if not dry_run and version_data:
        latest_version = sorted(version_data.keys())[-1]
        latest_data = version_data[latest_version]
        latest_labeled = set()
        cocos_updates = []
        for path, info in latest_data["paths"].items():
            label = info.get("cocos_label_transformation")
            if label:
                latest_labeled.add(path)
                cocos_updates.append({"id": path, "cocos_label_transformation": label})
        if cocos_updates:
            for i in range(0, len(cocos_updates), 1000):
                batch = cocos_updates[i : i + 1000]
                client.query(
                    """
                    UNWIND $paths AS p
                    MATCH (path:IMASNode {id: p.id})
                    SET path.cocos_label_transformation = p.cocos_label_transformation
                    """,
                    paths=batch,
                )
            stats["cocos_labels_updated"] = len(cocos_updates)

        # Clear stale labels on paths that lost COCOS sensitivity
        client.query(
            """
            MATCH (p:IMASNode)
            WHERE p.cocos_label_transformation IS NOT NULL
            AND NOT p.id IN $labeled_paths
            SET p.cocos_label_transformation = null
            """,
            labeled_paths=list(latest_labeled),
        )

        # Backfill COCOS labels for versions where XML lacks them (4.0.0–4.1.x)
        stats["cocos_labels_backfilled"] = _backfill_cocos_labels(client, version_data)

    # Final-pass: field-level metadata from latest version
    if not dry_run and version_data:
        latest_version = sorted(version_data.keys())[-1]
        latest_paths = version_data[latest_version]["paths"]
        meta_updates = []
        for path, info in latest_paths.items():
            update: dict = {"id": path}
            has_update = False
            for key in (
                "lifecycle_status",
                "lifecycle_version",
                "timebasepath",
                "path_doc",
                "introduced_after_version",
                "change_nbc_version",
                "change_nbc_description",
                "change_nbc_previous_name",
                "change_nbc_previous_type",
                "cocos_transformation_expression",
                "alternative_coordinate1",
                "url",
                "coordinate1_same_as",
                "coordinate2_same_as",
                "coordinate3_same_as",
                "coordinate4_same_as",
                "coordinate5_same_as",
                "coordinate6_same_as",
            ):
                val = info.get(key)
                update[key] = val
                if val:
                    has_update = True
            if has_update:
                meta_updates.append(update)
        if meta_updates:
            for i in range(0, len(meta_updates), 1000):
                batch = meta_updates[i : i + 1000]
                client.query(
                    """
                    UNWIND $paths AS p
                    MATCH (path:IMASNode {id: p.id})
                    SET path.lifecycle_status = p.lifecycle_status,
                        path.lifecycle_version = p.lifecycle_version,
                        path.timebasepath = p.timebasepath,
                        path.path_doc = p.path_doc,
                        path.introduced_after_version = p.introduced_after_version,
                        path.change_nbc_version = p.change_nbc_version,
                        path.change_nbc_description = p.change_nbc_description,
                        path.change_nbc_previous_name = p.change_nbc_previous_name,
                        path.change_nbc_previous_type = p.change_nbc_previous_type,
                        path.cocos_transformation_expression = p.cocos_transformation_expression,
                        path.alternative_coordinate1 = p.alternative_coordinate1,
                        path.url = p.url,
                        path.coordinate1_same_as = p.coordinate1_same_as,
                        path.coordinate2_same_as = p.coordinate2_same_as,
                        path.coordinate3_same_as = p.coordinate3_same_as,
                        path.coordinate4_same_as = p.coordinate4_same_as,
                        path.coordinate5_same_as = p.coordinate5_same_as,
                        path.coordinate6_same_as = p.coordinate6_same_as
                    """,
                    paths=batch,
                )

        # Create HAS_IDENTIFIER_SCHEMA relationships
        id_rels = []
        for path, info in latest_paths.items():
            enum_name = info.get("identifier_enum_name")
            if enum_name:
                id_rels.append({"id": path, "schema_name": enum_name})
        if id_rels:
            client.query(
                """
                UNWIND $rels AS r
                MATCH (path:IMASNode {id: r.id})
                MATCH (schema:IdentifierSchema {id: r.schema_name})
                MERGE (path)-[:HAS_IDENTIFIER_SCHEMA]->(schema)
            """,
                rels=id_rels,
            )

        # Create COORDINATE_SAME_AS relationships
        csa_rels = []
        for path, info in latest_paths.items():
            ids_name = path.split("/", 1)[0]
            for dim in range(1, 7):
                csa_raw = info.get(f"coordinate{dim}_same_as")
                if not csa_raw or " OR " in csa_raw:
                    continue
                target_path = f"{ids_name}/{_strip_dd_indices(csa_raw)}"
                csa_rels.append(
                    {
                        "source_id": path,
                        "target_id": target_path,
                        "dimension": dim,
                    }
                )
        if csa_rels:
            for i in range(0, len(csa_rels), 1000):
                batch = csa_rels[i : i + 1000]
                client.query(
                    """
                    UNWIND $rels AS r
                    MATCH (path:IMASNode {id: r.source_id})
                    MATCH (target:IMASNode {id: r.target_id})
                    MERGE (path)-[rel:COORDINATE_SAME_AS]->(target)
                    SET rel.dimension = r.dimension
                """,
                    rels=batch,
                )

    # Create HAS_ERROR relationships from error field naming patterns
    if not dry_run and version_data:
        all_paths: dict[str, dict] = {}
        for vdata in version_data.values():
            all_paths.update(vdata["paths"])
        error_relationships = _detect_error_relationships(all_paths)
        if error_relationships:
            stats["error_relationships"] = _batch_create_error_relationships(
                client, error_relationships
            )

    # Clean up orphaned Unit nodes that have no incoming HAS_UNIT relationships.
    # Units from older DD versions may be renamed/removed in later versions.
    if not dry_run:
        count_result = client.query(
            "MATCH (u:Unit) WHERE NOT (u)<-[:HAS_UNIT]-() RETURN count(u) AS orphans"
        )
        orphaned = count_result[0]["orphans"] if count_result else 0
        if orphaned > 0:
            client.query("MATCH (u:Unit) WHERE NOT (u)<-[:HAS_UNIT]-() DETACH DELETE u")
            logger.info(
                "Deleted %d orphaned Unit nodes (no HAS_UNIT relationships)",
                orphaned,
            )
        stats["orphaned_units_deleted"] = orphaned

    return stats


def phase_enrich(
    client: GraphClient,
    *,
    model: str | None = None,
    batch_size: int = 50,
    ids_filter: set[str] | None = None,
    force: bool = False,
    on_progress: "Callable[[int, int], None] | None" = None,
    on_cost: "Callable[[float], None] | None" = None,
    on_items: "Callable[[list[dict], float], None] | None" = None,
) -> dict[str, int | float]:
    """LLM enrichment of path descriptions.

    Enriches ALL IMASNode paths across all DD versions, not just the
    current version. Wraps :func:`enrich_imas_paths` and returns stats
    suitable for merging into the build stats dict.
    """
    from imas_codex.graph.dd_enrichment import enrich_imas_paths

    enrichment_stats = enrich_imas_paths(
        client=client,
        model=model,
        batch_size=batch_size,
        ids_filter=ids_filter,
        use_rich=False,
        force=force,
        on_progress=on_progress,
        on_cost=on_cost,
        on_items=on_items,
    )

    # Enrich identifier schemas
    from imas_codex.graph.dd_identifier_enrichment import enrich_identifier_schemas

    ident_stats = enrich_identifier_schemas(
        client,
        model=model,
        force=force,
    )

    # Enrich IDS nodes (depends on identifier enrichment for context)
    from imas_codex.graph.dd_ids_enrichment import enrich_ids_nodes

    ids_stats = enrich_ids_nodes(
        client,
        model=model,
        force=force,
    )

    return {
        "enriched_llm": enrichment_stats.get("enriched_llm", 0),
        "enriched_template": enrichment_stats.get("enriched_template", 0),
        "enrichment_cached": enrichment_stats.get("enrichment_cached", 0),
        "enrichment_cost": enrichment_stats.get("enrichment_cost", 0.0)
        + ident_stats.get("cost", 0.0)
        + ids_stats.get("cost", 0.0),
        "identifier_schemas_enriched": ident_stats.get("enriched", 0),
        "ids_enriched": ids_stats.get("enriched", 0),
    }


def phase_embed(
    client: GraphClient,
    versions: list[str],
    version_data: dict[str, dict],
    *,
    enriched_llm_count: int = 0,
    force: bool = False,
    force_reembed: bool = False,
    on_progress: "Callable[[int, int], None] | None" = None,
    on_items: "Callable[[list[dict], float], None] | None" = None,
) -> dict[str, int]:
    """Generate vector embeddings for DD paths.

    Merges version data, filters embeddable paths, merges enriched
    descriptions from the graph, then generates embeddings.

    Returns:
        Stats dict with embedding counts.
    """
    stats: dict[str, int] = {
        "paths_filtered": 0,
        "error_relationships": 0,
        "embeddings_updated": 0,
        "embeddings_cached": 0,
    }

    merged_paths: dict[str, dict] = {}
    merged_ids_info: dict[str, dict] = {}
    for version in versions:
        vdata = version_data.get(version)
        if vdata:
            merged_paths.update(vdata["paths"])
            merged_ids_info.update(vdata["ids_info"])

    # Detect error relationships before filtering (needs all paths)
    all_paths_for_errors = dict(merged_paths)

    # Filter merged version data to data nodes only — error/metadata skip embedding
    merged_paths = {
        path: info
        for path, info in merged_paths.items()
        if _classify_node(path, info.get("name", path.split("/")[-1])) == "data"
    }

    # Also include graph paths not in version_data (from previous builds)
    # so that incremental builds still embed all paths.
    # Only embed data nodes — error/metadata nodes are excluded.
    existing_paths_query = """
    MATCH (p:IMASNode)
    WHERE p.embedding IS NULL
      AND p.node_category = 'data'
    RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
           p.data_type AS data_type, p.ids AS ids, p.units AS units,
           p.description AS description, p.keywords AS keywords,
           p.cocos_label_transformation AS cocos_label_transformation,
           p.physics_domain AS physics_domain,
           p.node_type AS node_type, p.ndim AS ndim
    """
    for r in client.query(existing_paths_query):
        pid = r["id"]
        if pid not in merged_paths:
            merged_paths[pid] = {k: v for k, v in r.items() if v is not None}

    # Merge enriched descriptions from graph into path data
    # Always merge - enrichment may have run in a previous build
    enriched_paths_query = """
    MATCH (p:IMASNode)
    WHERE p.description IS NOT NULL
    RETURN p.id AS id,
           p.description AS description,
           p.keywords AS keywords,
           p.enrichment_source AS enrichment_source
    """
    enriched_count = 0
    for r in client.query(enriched_paths_query):
        pid = r["id"]
        if pid in merged_paths:
            merged_paths[pid] = dict(merged_paths[pid])
            merged_paths[pid]["description"] = r["description"]
            merged_paths[pid]["keywords"] = r["keywords"] or []
            enriched_count += 1

    if not merged_paths:
        return stats

    error_relationships = _detect_error_relationships(all_paths_for_errors)

    if on_progress:
        on_progress(0, len(merged_paths))

    if error_relationships:
        stats["error_relationships"] = _batch_create_error_relationships(
            client, error_relationships
        )

    # Force rebuild if enrichment produced new descriptions
    force_for_embed = force_reembed or enriched_llm_count > 0 or enriched_count > 0

    # Track cumulative progress across store batches
    _embed_stored = [0]

    def _on_store_batch(path_ids: list[str], batch_time: float) -> None:
        _embed_stored[0] += len(path_ids)
        if on_progress:
            on_progress(
                _embed_stored[0] + embedding_stats.get("cached", 0),
                len(merged_paths),
            )
        if on_items:
            stream_items = [{"primary_text": pid} for pid in path_ids]
            on_items(stream_items, batch_time)

    embedding_stats = update_path_embeddings(
        client=client,
        paths_data=merged_paths,
        ids_info=merged_ids_info,
        force_rebuild=force or force_for_embed,
        use_rich=False,
        dd_version=current_dd_version,
        track_changes=True,
        on_store_batch=_on_store_batch,
    )
    stats["embeddings_updated"] = embedding_stats["updated"]
    stats["embeddings_cached"] = embedding_stats["cached"]

    # Final progress update for cached paths (no store batches emitted)
    if on_progress:
        on_progress(
            embedding_stats["updated"] + embedding_stats["cached"],
            len(merged_paths),
        )

    from imas_codex.settings import get_embedding_model

    client.query(
        """
        MATCH (v:DDVersion {id: $version})
        SET v.embeddings_built_at = datetime(),
            v.embeddings_model = $model,
            v.embeddings_count = $count
        """,
        version=current_dd_version,
        model=get_embedding_model(),
        count=embedding_stats["total"],
    )

    # Embed enriched identifier schemas
    from imas_codex.graph.dd_identifier_enrichment import embed_identifier_schemas

    ident_embed_stats = embed_identifier_schemas(client, force_reembed=force_reembed)
    stats["identifier_embeddings_updated"] = ident_embed_stats["updated"]
    stats["identifier_embeddings_cached"] = ident_embed_stats["cached"]

    # Embed enriched IDS nodes
    from imas_codex.graph.dd_ids_enrichment import embed_ids_nodes

    ids_embed_stats = embed_ids_nodes(client, force_reembed=force_reembed)
    stats["ids_embeddings_updated"] = ids_embed_stats["updated"]
    stats["ids_embeddings_cached"] = ids_embed_stats["cached"]

    return stats


def phase_embed_stale(
    client: GraphClient,
    batch_size: int = 500,
) -> int:
    """Re-embed paths whose descriptions changed since initial embedding.

    Called after enrich completes when embed ran concurrently.  Finds
    paths where the content hash no longer matches (because enrichment
    updated the description after the initial embed pass) and
    regenerates their embeddings.

    Returns:
        Number of paths re-embedded.
    """
    from imas_codex.settings import get_embedding_model

    resolved_model = get_embedding_model()

    # Find paths with embeddings whose description was enriched after embedding
    stale_query = """
    MATCH (p:IMASNode)
    WHERE p.embedding IS NOT NULL
      AND p.description IS NOT NULL
      AND p.embedding_hash IS NOT NULL
      AND p.enrichment_source IS NOT NULL
    RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
           p.data_type AS data_type, p.ids AS ids, p.units AS units,
           p.description AS description, p.keywords AS keywords,
           p.cocos_label_transformation AS cocos_label_transformation,
           p.physics_domain AS physics_domain,
           p.node_type AS node_type, p.ndim AS ndim,
           p.embedding_hash AS existing_hash
    """
    rows = client.query(stale_query)
    if not rows:
        return 0

    # Build path data and check which hashes are stale
    stale_paths: dict[str, dict] = {}
    for r in rows:
        pid = r["id"]
        path_info = {
            k: v for k, v in r.items() if v is not None and k != "existing_hash"
        }
        text = generate_embedding_text(pid, path_info, {})
        new_hash = compute_embedding_hash(text, resolved_model)
        if new_hash != r["existing_hash"]:
            stale_paths[pid] = path_info

    if not stale_paths:
        return 0

    logger.info("Re-embedding %d paths with stale embeddings", len(stale_paths))

    embedding_stats = update_path_embeddings(
        client=client,
        paths_data=stale_paths,
        ids_info={},
        force_rebuild=True,
        use_rich=False,
        track_changes=False,
    )
    return embedding_stats["updated"]


# ---------------------------------------------------------------------------
# Deterministic cluster definitions
# ---------------------------------------------------------------------------

CANONICAL_CROSS_IDS = {
    "electron_temperature": {
        "pattern": r".*/electrons/temperature$",
        "description": "Electron temperature profiles across all IDS",
    },
    "ion_temperature": {
        "pattern": r".*/ions.*/temperature$",
        "description": "Ion temperature profiles across all IDS",
    },
    "electron_density": {
        "pattern": r".*/electrons/density$",
        "description": "Electron density profiles across all IDS",
    },
    "poloidal_flux": {
        "pattern": r".*/psi$",
        "description": "Poloidal magnetic flux profiles",
    },
    "plasma_current": {
        "pattern": r".*/ip$",
        "description": "Total plasma current",
    },
    "toroidal_field": {
        "pattern": r".*/b0$|.*/b_field_tor",
        "description": "Toroidal magnetic field on axis",
    },
    "safety_factor": {
        "pattern": r".*/q$|.*/q_profile",
        "description": "Safety factor profile",
    },
    "pressure": {
        "pattern": r".*/pressure$|.*/pressure_thermal$",
        "description": "Plasma pressure profiles",
    },
    "boundary_shape_r": {
        "pattern": r".*/boundary.*/r$|.*/outline/r$|.*/lcfs/r$",
        "description": "Plasma boundary R coordinates",
    },
    "boundary_shape_z": {
        "pattern": r".*/boundary.*/z$|.*/outline/z$|.*/lcfs/z$",
        "description": "Plasma boundary Z coordinates",
    },
}


def _create_cocos_clusters(client: GraphClient) -> int:
    """Create deterministic clusters grouping paths by COCOS label type.

    Unlike HDBSCAN clusters (statistical), these are authoritative groupings
    derived from DD metadata. Tagged with source='cocos_metadata'.

    Returns count of clusters created.
    """
    labels = client.query("""
        MATCH (p:IMASNode)
        WHERE p.cocos_label_transformation IS NOT NULL
          AND p.node_category = 'data'
        RETURN p.cocos_label_transformation AS label,
               collect(p.id) AS paths,
               collect(DISTINCT p.ids) AS ids_names,
               count(p) AS cnt
        ORDER BY label
    """)

    if not labels:
        return 0

    clusters = []
    for row in labels:
        label = row["label"]
        cluster_id = f"cocos_{label}"
        ids_names = row["ids_names"]
        clusters.append(
            {
                "id": cluster_id,
                "label": f"{label} COCOS-dependent fields",
                "description": (
                    f"Fields with {label} COCOS transformation sensitivity. "
                    f"These fields change sign or scale under COCOS convention "
                    f"changes. Spans {len(ids_names)} IDS: "
                    f"{', '.join(sorted(ids_names)[:10])}."
                ),
                "path_count": row["cnt"],
                "cross_ids": len(ids_names) > 1,
                "scope": "global",
                "source": "cocos_metadata",
                "ids_names": sorted(ids_names),
            }
        )

    # Create cluster nodes
    client.query(
        """
        UNWIND $clusters AS c
        MERGE (cl:IMASSemanticCluster {id: c.id})
        SET cl.label = c.label,
            cl.description = c.description,
            cl.path_count = c.path_count,
            cl.cross_ids = c.cross_ids,
            cl.scope = c.scope,
            cl.source = c.source,
            cl.ids_names = c.ids_names
        """,
        clusters=clusters,
    )

    # Create IN_CLUSTER relationships
    for row in labels:
        label = row["label"]
        cluster_id = f"cocos_{label}"
        client.query(
            """
            MATCH (p:IMASNode)
            WHERE p.cocos_label_transformation = $label
              AND p.node_category = 'data'
            MATCH (cl:IMASSemanticCluster {id: $cluster_id})
            MERGE (p)-[:IN_CLUSTER]->(cl)
            """,
            label=label,
            cluster_id=cluster_id,
        )

    logger.info("Created %d COCOS clusters", len(clusters))
    return len(clusters)


def _create_physics_clusters(client: GraphClient) -> int:
    """Create deterministic cross-IDS clusters for canonical physics quantities.

    Uses regex patterns from ``CANONICAL_CROSS_IDS`` to match IMASNode paths.
    Tagged with source='physics_canonical'.

    Returns count of clusters created.
    """
    all_paths = client.query("""
        MATCH (p:IMASNode)
        WHERE p.node_category = 'data'
        RETURN p.id AS id, p.ids AS ids
    """)

    if not all_paths:
        return 0

    created = 0

    for key, config in CANONICAL_CROSS_IDS.items():
        pattern = re.compile(config["pattern"])
        matching = [p for p in all_paths if pattern.search(p["id"])]

        if len(matching) < 2:
            continue

        ids_names = sorted({p["ids"] for p in matching})
        cluster_id = f"physics_{key}"

        client.query(
            """
            MERGE (cl:IMASSemanticCluster {id: $id})
            SET cl.label = $label,
                cl.description = $desc,
                cl.path_count = $cnt,
                cl.cross_ids = $cross_ids,
                cl.scope = 'global',
                cl.source = 'physics_canonical',
                cl.ids_names = $ids_names
            """,
            id=cluster_id,
            label=key.replace("_", " "),
            desc=config["description"],
            cnt=len(matching),
            cross_ids=len(ids_names) > 1,
            ids_names=ids_names,
        )

        path_ids = [p["id"] for p in matching]
        client.query(
            """
            UNWIND $paths AS pid
            MATCH (p:IMASNode {id: pid})
            MATCH (cl:IMASSemanticCluster {id: $cluster_id})
            MERGE (p)-[:IN_CLUSTER]->(cl)
            """,
            paths=path_ids,
            cluster_id=cluster_id,
        )

        created += 1

    logger.info("Created %d physics clusters", created)
    return created


def phase_cluster(
    client: GraphClient,
    *,
    dry_run: bool = False,
    force_reembed: bool = False,
    on_progress: "Callable[[int, int], None] | None" = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> int:
    """Import semantic clusters.

    Runs HDBSCAN statistical clustering, then adds deterministic clusters
    for COCOS-sensitive fields and canonical physics quantities.

    Args:
        force_reembed: If True, skip hash checks and re-embed all
            cluster text (labels/descriptions).
        on_progress: Optional callback ``(processed, total)`` for
            live progress updates during cluster import.
        stop_check: Optional callable returning True when the
            operation should be interrupted (e.g. Ctrl+C).

    Returns:
        Number of cluster nodes created.
    """
    hdbscan_count = _import_clusters(
        client,
        dry_run,
        use_rich=False,
        force_reembed=force_reembed,
        on_progress=on_progress,
        stop_check=stop_check,
    )

    if dry_run:
        return hdbscan_count

    # Deterministic clusters from DD metadata
    cocos_count = _create_cocos_clusters(client)
    physics_count = _create_physics_clusters(client)
    logger.info(
        "Created %d COCOS clusters and %d physics clusters",
        cocos_count,
        physics_count,
    )

    return hdbscan_count + cocos_count + physics_count


def _create_version_nodes(client: GraphClient, versions: list[str]) -> None:
    """Create DDVersion nodes with predecessor chain and COCOS metadata."""
    sorted_versions = sorted(versions)

    # Build version data with predecessors, successors, and COCOS info
    version_data = []
    for i, version in enumerate(sorted_versions):
        cocos_val, cocos_labeled = extract_cocos_for_version(version)
        major = int(version.split(".")[0])
        prev_major = int(sorted_versions[i - 1].split(".")[0]) if i > 0 else major
        version_data.append(
            {
                "id": version,
                "predecessor": sorted_versions[i - 1] if i > 0 else None,
                "successor": sorted_versions[i + 1]
                if i < len(sorted_versions) - 1
                else None,
                "is_current": version == current_dd_version,
                "is_major_boundary": (i > 0 and major != prev_major),
                "cocos": cocos_val,
                "cocos_labeled_fields": cocos_labeled,
            }
        )

    # Batch create all version nodes
    client.query(
        """
        UNWIND $versions AS v
        MERGE (ver:DDVersion {id: v.id})
        SET ver.is_current = v.is_current,
            ver.is_major_boundary = v.is_major_boundary,
            ver.cocos = v.cocos,
            ver.cocos_labeled_fields = v.cocos_labeled_fields,
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
            MERGE (ver)-[:HAS_PREDECESSOR]->(prev)
        """,
            versions=predecessors,
        )

    # Batch create successor relationships (symmetric with HAS_PREDECESSOR)
    successors = [v for v in version_data if v["successor"] is not None]
    if successors:
        client.query(
            """
            UNWIND $versions AS v
            MATCH (ver:DDVersion {id: v.id})
            MATCH (next:DDVersion {id: v.successor})
            MERGE (ver)-[:HAS_SUCCESSOR]->(next)
        """,
            versions=successors,
        )

    # Create COCOS reference nodes and link to DDVersions
    _create_cocos_nodes(client)

    # Link DDVersion nodes to their COCOS nodes
    versions_with_cocos = [v for v in version_data if v["cocos"] is not None]
    if versions_with_cocos:
        client.query(
            """
            UNWIND $versions AS v
            MATCH (ver:DDVersion {id: v.id})
            MATCH (cocos:COCOS {id: v.cocos})
            MERGE (ver)-[:HAS_COCOS]->(cocos)
        """,
            versions=versions_with_cocos,
        )


def _create_cocos_nodes(client: GraphClient) -> None:
    """Create all 16 COCOS reference nodes from the Sauter table.

    COCOS nodes are shared reference data (like Unit nodes). Each encodes
    the four parameters from Table I of Sauter & Medvedev, CPC 184 (2013).
    """
    from imas_codex.cocos.calculator import VALID_COCOS, cocos_to_parameters

    cocos_data = []
    for cocos_val in sorted(VALID_COCOS):
        params = cocos_to_parameters(cocos_val)
        psi_norm = "full ψ" if params.e_bp == 1 else "ψ/(2π)"
        handedness_rpz = "right-handed" if params.sigma_r_phi_z == 1 else "left-handed"
        handedness_rtp = (
            "right-handed" if params.sigma_rho_theta_phi == 1 else "left-handed"
        )
        psi_dir = "increasing" if params.sigma_bp == 1 else "decreasing"

        # Derived properties from Sauter Table I
        phi_increasing_ccw = params.sigma_r_phi_z == 1
        # θ direction from front depends on product σρθφ·σRφZ
        theta_increasing_cw = (params.sigma_rho_theta_phi * params.sigma_r_phi_z) == 1
        psi_increasing_outward = params.sigma_bp == 1
        # sign(q) = σIp·σB0·σρθφ; for co-current (σIp·σB0>0): sign(q) = σρθφ
        sign_q = params.sigma_rho_theta_phi
        # sign(dp/dψ) = -σIp·σBp; for positive Ip: sign(dp/dψ) = -σBp
        sign_dp_dpsi = -params.sigma_bp

        cocos_data.append(
            {
                "id": cocos_val,
                "sigma_bp": params.sigma_bp,
                "e_bp": params.e_bp,
                "sigma_r_phi_z": params.sigma_r_phi_z,
                "sigma_rho_theta_phi": params.sigma_rho_theta_phi,
                "phi_increasing_ccw": phi_increasing_ccw,
                "theta_increasing_cw": theta_increasing_cw,
                "psi_increasing_outward": psi_increasing_outward,
                "sign_q": sign_q,
                "sign_dp_dpsi": sign_dp_dpsi,
                "description": (
                    f"COCOS {cocos_val}: ψ {psi_dir} outward, {psi_norm}, "
                    f"(R,φ,Z) {handedness_rpz}, (ρ,θ,φ) {handedness_rtp}"
                ),
            }
        )

    client.query(
        """
        UNWIND $cocos_list AS c
        MERGE (cocos:COCOS {id: c.id})
        SET cocos.sigma_bp = c.sigma_bp,
            cocos.e_bp = c.e_bp,
            cocos.sigma_r_phi_z = c.sigma_r_phi_z,
            cocos.sigma_rho_theta_phi = c.sigma_rho_theta_phi,
            cocos.phi_increasing_ccw = c.phi_increasing_ccw,
            cocos.theta_increasing_cw = c.theta_increasing_cw,
            cocos.psi_increasing_outward = c.psi_increasing_outward,
            cocos.sign_q = c.sign_q,
            cocos.sign_dp_dpsi = c.sign_dp_dpsi,
            cocos.description = c.description
    """,
        cocos_list=cocos_data,
    )

    logger.debug("Created %d COCOS reference nodes", len(cocos_data))


def _ensure_indexes(client: GraphClient) -> None:
    """Ensure required indexes exist for optimal query performance."""
    logger.debug("Ensuring DD indexes exist...")

    # IMASNode.id - critical for MERGE and relationship creation
    client.query("CREATE INDEX imaspath_id IF NOT EXISTS FOR (p:IMASNode) ON (p.id)")

    # IMASNode.node_category - indexed filtering for search/enrichment/embedding
    client.query(
        "CREATE INDEX imas_node_category IF NOT EXISTS "
        "FOR (p:IMASNode) ON (p.node_category)"
    )

    # IDS.name - for IDS relationship lookups
    client.query("CREATE INDEX ids_name IF NOT EXISTS FOR (i:IDS) ON (i.name)")

    # DDVersion.id - for version relationship lookups
    client.query("CREATE INDEX ddversion_id IF NOT EXISTS FOR (v:DDVersion) ON (v.id)")

    # Unit.symbol - for unit relationship lookups
    client.query("CREATE INDEX unit_symbol IF NOT EXISTS FOR (u:Unit) ON (u.symbol)")

    # COCOS.id - for COCOS reference lookups
    client.query("CREATE INDEX cocos_id IF NOT EXISTS FOR (c:COCOS) ON (c.id)")

    # IMASSemanticCluster.id - for cluster lookups
    client.query(
        "CREATE CONSTRAINT imassemanticcluster_id IF NOT EXISTS "
        "FOR (c:IMASSemanticCluster) REQUIRE c.id IS UNIQUE"
    )

    # Composite indexes for precomputed query properties
    client.query(
        "CREATE INDEX imas_node_category_ids IF NOT EXISTS "
        "FOR (p:IMASNode) ON (p.node_category, p.ids)"
    )
    client.query(
        "CREATE INDEX imas_node_is_leaf IF NOT EXISTS FOR (p:IMASNode) ON (p.is_leaf)"
    )
    client.query(
        "CREATE INDEX imas_node_path_lower IF NOT EXISTS "
        "FOR (p:IMASNode) ON (p.path_lower)"
    )

    # Vector indexes for semantic search
    client.ensure_vector_indexes()

    # Fulltext indexes for BM25 text search
    client.ensure_fulltext_indexes()


def _create_unit_nodes(client: GraphClient, units: set[str]) -> None:
    """Create Unit nodes with pint-normalized symbols.

    Sentinel/unparseable units are filtered out — only pint-parseable
    units get Unit nodes.
    """
    from imas_codex.units import normalize_unit_symbol

    unit_list = []
    for u in units:
        if not u:
            continue
        normalized = normalize_unit_symbol(u)
        if normalized is None:
            continue  # skip sentinels
        unit_list.append({"id": normalized, "symbol": normalized})

    if unit_list:
        client.query(
            """
            UNWIND $units AS unit
            MERGE (u:Unit {id: unit.id})
            SET u.symbol = unit.symbol
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


def _collect_identifier_schemas(paths: dict[str, dict]) -> dict[str, dict]:
    """Collect IdentifierSchema data with full option metadata from DD XML.

    Uses ``imas_data_dictionaries.get_identifier_xml()`` to extract per-option
    ``description`` and ``units`` alongside the ``index`` and ``name`` already
    captured.  Also extracts the XML ``<header>`` text as the schema-level
    description.

    Falls back to path-metadata extraction when the XML API is unavailable.
    """
    import xml.etree.ElementTree as ET

    from imas_data_dictionaries import dd_identifiers, get_identifier_xml

    # Count field references per enum name from extracted paths
    referenced: dict[str, int] = {}
    for _path, info in paths.items():
        enum_name = info.get("identifier_enum_name")
        if enum_name:
            referenced[enum_name] = referenced.get(enum_name, 0) + 1

    available = set(dd_identifiers())
    schemas: dict[str, dict] = {}

    for enum_name, field_count in referenced.items():
        if enum_name not in available:
            continue
        xml_bytes = get_identifier_xml(enum_name)
        root = ET.fromstring(xml_bytes)

        # Extract header as schema description
        header = root.find("header")
        description = header.text.strip() if header is not None and header.text else ""

        # Extract per-option metadata from <int> elements
        options = []
        for child in root:
            if child.tag == "int" and child.text:
                options.append(
                    {
                        "index": int(child.text.strip()),
                        "name": child.attrib.get("name", ""),
                        "description": child.attrib.get("description", ""),
                        "units": child.attrib.get("units", ""),
                    }
                )
        options.sort(key=lambda o: o["index"])

        schemas[enum_name] = {
            "id": enum_name,
            "name": enum_name,
            "documentation": description,
            "options": json.dumps(options),
            "option_count": len(options),
            "field_count": field_count,
            "source": f"utilities/{enum_name}.xml",
        }

    return schemas


def _create_identifier_schema_nodes(
    client: GraphClient, schemas: dict[str, dict]
) -> None:
    """Create IdentifierSchema nodes in the graph."""
    schema_list = list(schemas.values())
    client.query(
        """
        UNWIND $schemas AS s
        MERGE (schema:IdentifierSchema {id: s.id})
        SET schema.name = s.name,
            schema.documentation = s.documentation,
            schema.option_count = s.option_count,
            schema.options = s.options,
            schema.field_count = s.field_count,
            schema.source = s.source
    """,
        schemas=schema_list,
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
            "id": ids_name,
            "name": ids_name,
            "documentation": info.get("documentation", ""),
            "physics_domain": info.get("physics_domain", "general"),
            "path_count": info.get("path_count", 0),
            "leaf_count": info.get("leaf_count", 0),
            "lifecycle_status": info.get("lifecycle_status"),
            "lifecycle_version": info.get("lifecycle_version"),
            "lifecycle_last_change": info.get("lifecycle_last_change"),
            "ids_type": info.get("ids_type"),
        }
        for ids_name, info in ids_info_map.items()
    ]

    if not ids_list:
        return

    # Batch create/update IDS nodes
    client.query(
        """
        UNWIND $ids_list AS ids_data
        MERGE (ids:IDS {id: ids_data.id})
        SET ids.name = ids_data.name,
            ids.documentation = ids_data.documentation,
            ids.physics_domain = ids_data.physics_domain,
            ids.path_count = ids_data.path_count,
            ids.leaf_count = ids_data.leaf_count,
            ids.dd_version = $version,
            ids.lifecycle_status = ids_data.lifecycle_status,
            ids.lifecycle_version = ids_data.lifecycle_version,
            ids.lifecycle_last_change = ids_data.lifecycle_last_change,
            ids.ids_type = ids_data.ids_type
    """,
        ids_list=ids_list,
        version=version,
    )

    # Batch create INTRODUCED_IN relationships for first version
    if is_first_version:
        client.query(
            """
            UNWIND $ids_list AS ids_data
            MATCH (ids:IDS {id: ids_data.id})
            MATCH (v:DDVersion {id: $version})
            MERGE (ids)-[:INTRODUCED_IN]->(v)
        """,
            ids_list=ids_list,
            version=version,
        )


def _classify_node(path_id: str, name: str) -> str:
    """Classify an IMASNode path into a node_category.

    Returns 'data', 'error', or 'metadata' based on path structure.
    Delegates to ExclusionChecker for consistent classification logic.
    Used at node creation time to set the indexed node_category property.
    """
    from imas_codex.core.exclusions import ExclusionChecker

    # Use a checker with both error fields and ggd included so that
    # _is_error_field and _is_metadata_path are the only classification
    # criteria — we want to classify, not exclude based on settings.
    checker = ExclusionChecker(include_error_fields=False, include_ggd=True)

    if checker._is_error_field(name):
        return "error"
    if checker._is_metadata_path(path_id):
        return "metadata"

    return "data"


def _batch_create_path_nodes(
    client: GraphClient,
    paths_data: dict[str, dict],
    version: str,
    batch_size: int = 1000,
) -> None:
    """Batch create IMASNode nodes with relationships.

    Uses multiple batched queries to avoid memory issues with large datasets.
    """
    # Prepare path data for batch insertion
    path_list = []
    for path, path_info in paths_data.items():
        ids_name = path.split("/")[0]
        physics_domain = physics_categorizer.get_domain_for_ids(ids_name).value
        name = path_info.get("name", "")

        doc = path_info.get("documentation", "")
        node_type = path_info.get("node_type")
        path_list.append(
            {
                "id": path,
                "name": name,
                "node_category": _classify_node(path, name),
                "documentation": doc,
                "data_type": path_info.get("data_type"),
                "ndim": path_info.get("ndim", 0),
                "node_type": node_type,
                "depth": path.count("/"),
                "is_leaf": node_type not in ("structure", "struct_array"),
                "path_lower": path.lower(),
                "doc_length": len(doc),
                "physics_domain": physics_domain,
                "maxoccur": path_info.get("maxoccur"),
                "ids_name": ids_name,
                "parent_path": path_info.get("parent_path"),
                "unit": path_info.get("units", ""),
                "coordinates": path_info.get("coordinates", []),
                "cocos_label_transformation": path_info.get(
                    "cocos_label_transformation"
                ),
                "lifecycle_status": path_info.get("lifecycle_status"),
                "lifecycle_version": path_info.get("lifecycle_version"),
                "timebasepath": path_info.get("timebasepath"),
                "path_doc": path_info.get("path_doc"),
                "introduced_after_version": path_info.get("introduced_after_version"),
                "change_nbc_version": path_info.get("change_nbc_version"),
                "change_nbc_description": path_info.get("change_nbc_description"),
                "change_nbc_previous_name": path_info.get("change_nbc_previous_name"),
                "change_nbc_previous_type": path_info.get("change_nbc_previous_type"),
                "cocos_transformation_expression": path_info.get(
                    "cocos_transformation_expression"
                ),
                "alternative_coordinate1": path_info.get("alternative_coordinate1"),
                "url": path_info.get("url"),
                "identifier_enum_name": path_info.get("identifier_enum_name"),
                "coordinate1_same_as": path_info.get("coordinate1_same_as"),
                "coordinate2_same_as": path_info.get("coordinate2_same_as"),
                "coordinate3_same_as": path_info.get("coordinate3_same_as"),
                "coordinate4_same_as": path_info.get("coordinate4_same_as"),
                "coordinate5_same_as": path_info.get("coordinate5_same_as"),
                "coordinate6_same_as": path_info.get("coordinate6_same_as"),
            }
        )

    if not path_list:
        return

    # Process in batches to avoid memory issues
    for i in range(0, len(path_list), batch_size):
        batch = path_list[i : i + batch_size]

        # Step 1: Create IMASNode nodes
        client.query(
            """
            UNWIND $paths AS p
            MERGE (path:IMASNode {id: p.id})
            SET path.name = p.name,
                path.node_category = p.node_category,
                path.documentation = p.documentation,
                path.data_type = p.data_type,
                path.ndim = p.ndim,
                path.node_type = p.node_type,
                path.physics_domain = p.physics_domain,
                path.maxoccur = p.maxoccur,
                path.ids = p.ids_name,
                path.depth = p.depth,
                path.is_leaf = p.is_leaf,
                path.path_lower = p.path_lower,
                path.doc_length = p.doc_length,
                path.cocos_label_transformation = p.cocos_label_transformation,
                path.lifecycle_status = p.lifecycle_status,
                path.lifecycle_version = p.lifecycle_version,
                path.timebasepath = p.timebasepath,
                path.path_doc = p.path_doc,
                path.introduced_after_version = p.introduced_after_version,
                path.change_nbc_version = p.change_nbc_version,
                path.change_nbc_description = p.change_nbc_description,
                path.change_nbc_previous_name = p.change_nbc_previous_name,
                path.change_nbc_previous_type = p.change_nbc_previous_type,
                path.cocos_transformation_expression = p.cocos_transformation_expression,
                path.alternative_coordinate1 = p.alternative_coordinate1,
                path.url = p.url,
                path.unit = p.unit,
                path.coordinate1_same_as = p.coordinate1_same_as,
                path.coordinate2_same_as = p.coordinate2_same_as,
                path.coordinate3_same_as = p.coordinate3_same_as,
                path.coordinate4_same_as = p.coordinate4_same_as,
                path.coordinate5_same_as = p.coordinate5_same_as,
                path.coordinate6_same_as = p.coordinate6_same_as,
                path.status = CASE
                    WHEN path.status IS NULL THEN 'built'
                    ELSE path.status
                END
        """,
            paths=batch,
        )

        # Step 2: Create IDS relationships
        client.query(
            """
            UNWIND $paths AS p
            MATCH (path:IMASNode {id: p.id})
            MATCH (ids:IDS {id: p.ids_name})
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
                MATCH (path:IMASNode {id: p.id})
                MERGE (parent:IMASNode {id: p.parent_path})
                MERGE (path)-[:HAS_PARENT]->(parent)
            """,
                paths=parent_paths,
            )

        # Step 4: Create HAS_UNIT relationships (filter out empty/sentinel units)
        # Normalize unit symbols to match pint-normalized Unit nodes
        from imas_codex.units import normalize_unit_symbol

        unit_paths = []
        for p in batch:
            if p["unit"] and p["unit"] != "":
                normalized = normalize_unit_symbol(p["unit"])
                if normalized:
                    unit_paths.append({**p, "unit": normalized})
        if unit_paths:
            client.query(
                """
                UNWIND $paths AS p
                MATCH (path:IMASNode {id: p.id})
                MATCH (u:Unit {id: p.unit})
                MERGE (path)-[:HAS_UNIT]->(u)
            """,
                paths=unit_paths,
            )

        # Step 5: Create HAS_COORDINATE relationships
        # Each coordinate links to either an IMASCoordinateSpec (e.g., "1...N") or IMASNode
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
                    # Path references are relative to IDS root.
                    # Strip DD index notation (itime), (i1) etc. to match
                    # our stored IMASNode IDs which omit indices.
                    # Handle cross-IDS refs like "IDS:magnetics/flux_loop"
                    if coord_str.startswith("IDS:"):
                        target_path = coord_str[4:]  # strip "IDS:" prefix
                    else:
                        target_path = f"{ids_name}/{_strip_dd_indices(coord_str)}"
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
                MATCH (path:IMASNode {id: r.source_id})
                MATCH (spec:IMASCoordinateSpec {id: r.target_id})
                MERGE (path)-[rel:HAS_COORDINATE]->(spec)
                SET rel.dimension = r.dimension
            """,
                rels=spec_rels,
            )

        # Create relationships to IMASNode coordinate nodes
        path_rels = [r for r in coord_rels if not r["is_spec"]]
        if path_rels:
            client.query(
                """
                UNWIND $rels AS r
                MATCH (path:IMASNode {id: r.source_id})
                MATCH (coord:IMASNode {id: r.target_id})
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
            MATCH (path:IMASNode {id: p.id})
            WHERE NOT EXISTS { (path)-[:INTRODUCED_IN]->(:DDVersion) }
            MATCH (v:DDVersion {id: $version})
            MERGE (path)-[:INTRODUCED_IN]->(v)
        """,
            paths=batch,
            version=version,
        )

        # Step 7: Create HAS_IDENTIFIER_SCHEMA relationships
        id_schema_paths = [p for p in batch if p.get("identifier_enum_name")]
        if id_schema_paths:
            client.query(
                """
                UNWIND $paths AS p
                MATCH (path:IMASNode {id: p.id})
                MATCH (schema:IdentifierSchema {id: p.identifier_enum_name})
                MERGE (path)-[:HAS_IDENTIFIER_SCHEMA]->(schema)
            """,
                paths=id_schema_paths,
            )

        # Step 8: Create COORDINATE_SAME_AS relationships
        # Links fields that share the same coordinate grid (per dimension)
        csa_rels = []
        for p in batch:
            ids_name = p["ids_name"]
            for dim in range(1, 7):
                csa_raw = p.get(f"coordinate{dim}_same_as")
                if not csa_raw or " OR " in csa_raw:
                    # Skip edge cases like "frame(itime)/counts_n OR 1...1"
                    continue
                target_path = f"{ids_name}/{_strip_dd_indices(csa_raw)}"
                csa_rels.append(
                    {
                        "source_id": p["id"],
                        "target_id": target_path,
                        "dimension": dim,
                    }
                )
        if csa_rels:
            client.query(
                """
                UNWIND $rels AS r
                MATCH (path:IMASNode {id: r.source_id})
                MATCH (target:IMASNode {id: r.target_id})
                MERGE (path)-[rel:COORDINATE_SAME_AS]->(target)
                SET rel.dimension = r.dimension
            """,
                rels=csa_rels,
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
        MATCH (path:IMASNode {id: p.path})
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
    """Batch create IMASNodeChange nodes for metadata changes with semantic classification."""
    if not changes:
        return 0

    change_list = []
    for path, path_changes in changes.items():
        for change in path_changes:
            change_data = {
                "id": f"{path}:{change['field']}:{version}",
                "path": path,
                "change_type": FIELD_TO_CHANGE_TYPE.get(
                    change["field"], change["field"]
                ),
                "old_value": change.get("old_value", ""),
                "new_value": change.get("new_value", ""),
                "semantic_type": None,
                "keywords_detected": None,
                "unit_change_subtype": change.get("unit_change_subtype"),
            }

            # Use pre-computed semantic classification from compute_version_changes()
            if change["field"] == "documentation":
                change_data["semantic_type"] = change.get("semantic_type")
                kw = change.get("keywords_detected")
                if kw and not isinstance(kw, str):
                    change_data["keywords_detected"] = json.dumps(kw)
                elif kw:
                    change_data["keywords_detected"] = kw
                # Fallback for legacy callers that don't pre-compute
                if change_data["semantic_type"] is None:
                    semantic_type, keywords = classify_doc_change(
                        change.get("old_value", ""),
                        change.get("new_value", ""),
                    )
                    change_data["semantic_type"] = semantic_type
                    if keywords:
                        change_data["keywords_detected"] = json.dumps(keywords)

            # Persist breaking_level — use pre-computed value from
            # compute_version_changes(), fall back to classifying here.
            change_data["breaking_level"] = change.get(
                "breaking_level"
            ) or _classify_breaking_level(change["field"], change_data)

            change_list.append(change_data)

    if not change_list:
        return 0

    # Create IMASNodeChange nodes with semantic classification
    client.query(
        """
        UNWIND $changes AS c
        MERGE (change:IMASNodeChange {id: c.id})
        SET change.change_type = c.change_type,
            change.old_value = c.old_value,
            change.new_value = c.new_value,
            change.semantic_type = c.semantic_type,
            change.keywords_detected = c.keywords_detected,
            change.breaking_level = c.breaking_level,
            change.unit_change_subtype = c.unit_change_subtype
    """,
        changes=change_list,
    )

    # Create relationships
    client.query(
        """
        UNWIND $changes AS c
        MATCH (change:IMASNodeChange {id: c.id})
        MATCH (p:IMASNode {id: c.path})
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
            MATCH (old:IMASNode {id: r.old_path})
            MATCH (new:IMASNode {id: r.new_path})
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
            MATCH (err:IMASNode {id: r.error_path})
            MATCH (data:IMASNode {id: r.data_path})
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
        MATCH (p:IMASNode)-[:DEPRECATED_IN]->(:DDVersion)
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


def _compute_cluster_centroid_embeddings(client: GraphClient) -> int:
    """Compute normalized centroid embeddings for semantic clusters."""
    centroid_result = client.query(
        """
        MATCH (p:IMASNode)-[:IN_CLUSTER]->(c:IMASSemanticCluster)
        WHERE p.embedding IS NOT NULL
        WITH c, collect(p.embedding) AS embeddings, count(p) AS member_count
        WHERE member_count > 0
        WITH c, embeddings, member_count,
             size(embeddings[0]) AS dim
        WITH c, member_count, dim,
             [i IN range(0, dim - 1) |
                reduce(s = 0.0, emb IN embeddings | s + emb[i]) / member_count
             ] AS centroid_raw
        WITH c, centroid_raw,
             sqrt(reduce(s = 0.0, x IN centroid_raw | s + x * x)) AS norm
        WITH c, CASE WHEN norm > 0
             THEN [x IN centroid_raw | x / norm]
             ELSE centroid_raw
             END AS cluster_emb
        SET c.embedding = cluster_emb
        RETURN count(c) AS embeddings_set
        """
    )
    return centroid_result[0]["embeddings_set"] if centroid_result else 0


def _import_clusters(
    client: GraphClient,
    dry_run: bool,
    use_rich: bool | None = None,
    force_reembed: bool = False,
    on_progress: "Callable[[int, int], None] | None" = None,
    stop_check: "Callable[[], bool] | None" = None,
) -> int:
    """Build semantic clusters from graph embeddings and merge into the graph.

    Graph-native pipeline — no file dependencies:
    1. Run HDBSCAN on IMASNode embeddings read directly from the graph.
    2. Compute a content hash for each cluster from its sorted member paths.
    3. MERGE cluster nodes using content hash as ID, preserving existing
       labels and descriptions on unchanged clusters.
    4. Compute cluster centroid embeddings in Neo4j.
    5. Embed cluster text (labels/descriptions) with per-cluster hash caching.
    6. Delete stale clusters that no longer appear in the new computation.

    Args:
        client: GraphClient instance
        dry_run: If True, don't write to graph
        use_rich: Force rich progress setting
        on_progress: Optional callback ``(processed, total)`` for
            live progress updates.
        stop_check: Optional callable returning True when the
            operation should be interrupted (e.g. Ctrl+C).

    Returns:
        Number of clusters created/updated
    """

    def _stopped() -> bool:
        return stop_check is not None and stop_check()

    with suppress_third_party_logging():
        try:
            from imas_codex.clusters.hierarchical import HierarchicalClusterer

            # Step 1: Load embedded IMASNode IDs from graph
            logger.info("Loading embeddings from graph for clustering...")
            emb_result = client.query(
                """
                MATCH (p:IMASNode)
                WHERE p.embedding IS NOT NULL
                RETURN p.id AS id, p.embedding AS embedding
                ORDER BY p.id
                """
            )
            if not emb_result or len(emb_result) < 10:
                logger.warning(
                    "Insufficient embeddings in graph for clustering (%d)",
                    len(emb_result) if emb_result else 0,
                )
                return 0

            path_ids = [r["id"] for r in emb_result]
            embeddings = np.array(
                [r["embedding"] for r in emb_result], dtype=np.float32
            )
            logger.info(
                "Loaded %d embeddings (dim=%d) from graph",
                embeddings.shape[0],
                embeddings.shape[1],
            )

            # Step 1.5: Check cluster input hash — skip if IMASNode set unchanged
            cluster_input_hash = hashlib.sha256(
                "\n".join(path_ids).encode("utf-8")
            ).hexdigest()[:16]

            if not force_reembed:
                stored = client.query(
                    """
                    MATCH (v:DDVersion {is_current: true})
                    RETURN v.cluster_input_hash AS hash
                    """
                )
                if stored and stored[0].get("hash") == cluster_input_hash:
                    # Verify clusters actually exist in the graph
                    cl_count = client.query(
                        "MATCH (c:IMASSemanticCluster) RETURN count(c) AS cnt"
                    )
                    if cl_count and cl_count[0]["cnt"] > 0:
                        count = cl_count[0]["cnt"]
                        logger.info(
                            "Clustering skipped — %d embedded paths unchanged "
                            "(hash %s), %d clusters in graph",
                            len(path_ids),
                            cluster_input_hash[:8],
                            count,
                        )
                        if on_progress:
                            on_progress(count, count)
                        return count

            if _stopped():
                logger.info("Clustering interrupted after loading embeddings")
                return 0

            # Step 2: Run HDBSCAN clustering
            logger.info("Running hierarchical clustering...")
            clusterer = HierarchicalClusterer(
                min_cluster_size=2,
                min_samples=2,
                cluster_selection_method="eom",
            )
            clusters = clusterer.cluster_all_levels(
                embeddings=embeddings,
                paths=path_ids,
                include_global=True,
                include_domain=True,
                include_ids=True,
            )
            if not clusters:
                logger.warning("HDBSCAN produced no clusters")
                return 0

            logger.info("HDBSCAN produced %d clusters", len(clusters))

            if _stopped():
                logger.info("Clustering interrupted after HDBSCAN")
                return 0

            if on_progress:
                on_progress(0, len(clusters))

            if dry_run:
                return len(clusters)

            # Step 3: Build cluster data with content-hash IDs
            new_cluster_ids: set[str] = set()
            cluster_batch: list[dict] = []
            membership_batch: list[dict] = []

            for cluster in clusters:
                sorted_paths = sorted(cluster.paths)
                content_hash = _compute_cluster_content_hash(sorted_paths)
                new_cluster_ids.add(content_hash)

                ids_names = sorted({p.split("/")[0] for p in cluster.paths})
                is_cross_ids = len(ids_names) > 1
                scope = getattr(cluster, "scope", "global")

                cluster_batch.append(
                    {
                        "cluster_id": content_hash,
                        "path_count": len(cluster.paths),
                        "cross_ids": is_cross_ids,
                        "similarity_score": round(cluster.similarity_score, 4),
                        "scope": scope,
                        "ids_names": ids_names,
                    }
                )

                for path in cluster.paths:
                    membership_batch.append(
                        {
                            "cluster_id": content_hash,
                            "path": path,
                        }
                    )

            # Step 4: Get existing cluster IDs for stale detection
            if _stopped():
                logger.info("Clustering interrupted before graph merge")
                return 0

            existing_result = client.query(
                "MATCH (c:IMASSemanticCluster) "
                "WHERE c.source IS NULL OR c.source = 'hdbscan' "
                "RETURN c.id AS id"
            )
            existing_ids = (
                {r["id"] for r in existing_result} if existing_result else set()
            )

            # Step 5: MERGE cluster nodes (preserves existing label/description)
            if _stopped():
                logger.info("Clustering interrupted before node merge")
                return 0

            logger.info("Merging %d cluster nodes...", len(cluster_batch))
            for i in range(0, len(cluster_batch), 1000):
                batch = cluster_batch[i : i + 1000]
                client.query(
                    """
                    UNWIND $clusters AS c
                    MERGE (n:IMASSemanticCluster {id: c.cluster_id})
                    SET n.path_count = c.path_count,
                        n.cross_ids = c.cross_ids,
                        n.similarity_score = c.similarity_score,
                        n.scope = c.scope,
                        n.ids_names = c.ids_names,
                        n.source = 'hdbscan'
                    """,
                    clusters=batch,
                )
                if on_progress:
                    on_progress(
                        min(i + 1000, len(cluster_batch)),
                        len(cluster_batch),
                    )

            # Step 6: Clear old IN_CLUSTER relationships for clusters being updated
            #         then recreate them. This handles membership changes.
            if _stopped():
                logger.info("Clustering interrupted after node merge")
                return len(clusters)

            updated_ids = list(new_cluster_ids & existing_ids)
            if updated_ids:
                for i in range(0, len(updated_ids), 1000):
                    batch = updated_ids[i : i + 1000]
                    client.query(
                        """
                        UNWIND $ids AS cid
                        MATCH (c:IMASSemanticCluster {id: cid})<-[r:IN_CLUSTER]-()
                        DELETE r
                        """,
                        ids=batch,
                    )

            # Step 7: Create IN_CLUSTER relationships
            if _stopped():
                logger.info("Clustering interrupted before membership creation")
                return len(clusters)

            logger.info(
                "Creating %d IN_CLUSTER relationships...", len(membership_batch)
            )
            for i in range(0, len(membership_batch), 2000):
                batch = membership_batch[i : i + 2000]
                client.query(
                    """
                    UNWIND $memberships AS m
                    MATCH (p:IMASNode {id: m.path})
                    MATCH (c:IMASSemanticCluster {id: m.cluster_id})
                    MERGE (p)-[:IN_CLUSTER]->(c)
                    """,
                    memberships=batch,
                )

            # Step 8: Compute cluster centroid embeddings in Neo4j
            # Mean of unit vectors then L2-normalize to restore unit length
            if _stopped():
                logger.info("Clustering interrupted before centroid computation")
                return len(clusters)

            embeddings_set = _compute_cluster_centroid_embeddings(client)
            logger.info(
                "Computed %d/%d cluster centroid embeddings",
                embeddings_set,
                len(clusters),
            )

            # Step 8.1: Create REPRESENTATIVE_PATH relationships
            # Link each cluster to its representative (most central) path node
            client.query("""
                MATCH (c:IMASSemanticCluster)
                WHERE c.representative_path IS NOT NULL
                MATCH (p:IMASNode {id: c.representative_path})
                MERGE (c)-[:REPRESENTATIVE_PATH]->(p)
            """)

            # Step 8.5: Generate labels for unlabeled clusters via LLM
            if _stopped():
                logger.info("Clustering interrupted before LLM labeling")
                return len(clusters)

            _label_unlabeled_clusters(client, batch_size=50)

            # Step 9: Embed cluster labels and descriptions (with hash caching)
            if _stopped():
                logger.info("Clustering interrupted before text embedding")
                return len(clusters)

            _embed_cluster_text(
                client,
                embedding_batch_size=256,
                use_rich=use_rich,
                force_reembed=force_reembed,
            )

            # Step 10: Delete stale clusters no longer in the computation
            if _stopped():
                logger.info("Clustering interrupted before stale cleanup")
                return len(clusters)

            stale_ids = existing_ids - new_cluster_ids
            if stale_ids:
                logger.info("Removing %d stale clusters...", len(stale_ids))
                stale_list = list(stale_ids)
                for i in range(0, len(stale_list), 1000):
                    batch = stale_list[i : i + 1000]
                    client.query(
                        """
                        UNWIND $ids AS cid
                        MATCH (c:IMASSemanticCluster {id: cid})
                        DETACH DELETE c
                        """,
                        ids=batch,
                    )
                logger.info("Deleted %d stale clusters", len(stale_ids))

            # Backfill source on pre-existing HDBSCAN clusters that lack it
            client.query("""
                MATCH (c:IMASSemanticCluster)
                WHERE c.source IS NULL
                SET c.source = 'hdbscan'
            """)

            # Update DDVersion with cluster metadata + input hash
            client.query(
                """
                MATCH (v:DDVersion {is_current: true})
                SET v.clusters_built_at = datetime(),
                    v.clusters_count = $count,
                    v.cluster_input_hash = $hash
                """,
                count=len(clusters),
                hash=cluster_input_hash,
            )

            return len(clusters)

        except Exception as e:
            logger.error(f"Error building clusters: {e}")
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


def _label_unlabeled_clusters(
    client: GraphClient,
    batch_size: int = 50,
) -> int:
    """Generate LLM labels for clusters that lack them.

    Reads unlabeled clusters from the graph with their member paths,
    calls ClusterLabeler in batches, and updates cluster nodes with
    label, description, physics_concepts, and tags.

    Args:
        client: GraphClient instance
        batch_size: Number of clusters per LLM call

    Returns:
        Number of clusters labeled
    """
    # Fetch unlabeled clusters with their member paths and documentation
    unlabeled = client.query("""
        MATCH (c:IMASSemanticCluster)
        WHERE c.label IS NULL
        OPTIONAL MATCH (m:IMASNode)-[:IN_CLUSTER]->(c)
        WITH c,
             collect(m.id) AS paths,
             collect(DISTINCT m.ids) AS ids_names,
             [pair IN collect([m.id, m.documentation]) WHERE pair[1] IS NOT NULL] AS doc_pairs
        RETURN c.id AS id, c.scope AS scope, c.cross_ids AS cross_ids,
               paths, ids_names,
               doc_pairs AS path_doc_pairs
        ORDER BY c.id
    """)

    if not unlabeled:
        logger.info("No unlabeled clusters found — skipping LLM labeling")
        return 0

    logger.info("Labeling %d unlabeled clusters via LLM...", len(unlabeled))

    try:
        from imas_codex.clusters.labeler import ClusterLabeler

        labeler = ClusterLabeler()
        labeled_count = 0

        for i in range(0, len(unlabeled), batch_size):
            batch = unlabeled[i : i + batch_size]

            # Format for ClusterLabeler
            cluster_dicts = []
            for row in batch:
                # Build path_docs dict from doc_pairs
                path_docs = {}
                for pair in row.get("path_doc_pairs", []):
                    if pair and len(pair) == 2:
                        path_docs[pair[0]] = pair[1]
                cluster_dicts.append(
                    {
                        "id": row["id"],
                        "paths": row["paths"] or [],
                        "path_docs": path_docs,
                        "ids_names": row["ids_names"] or [],
                        "is_cross_ids": bool(row.get("cross_ids")),
                        "scope": row.get("scope", "global"),
                    }
                )

            labels = labeler.label_clusters(cluster_dicts, batch_size=batch_size)

            # Update graph with labels
            updates = []
            for label in labels:
                updates.append(
                    {
                        "id": label.cluster_id,
                        "label": label.label,
                        "description": label.description,
                        "physics_concepts": label.physics_concepts,
                        "tags": label.tags,
                    }
                )

            if updates:
                client.query(
                    """
                    UNWIND $updates AS u
                    MATCH (c:IMASSemanticCluster {id: u.id})
                    SET c.label = u.label,
                        c.description = u.description,
                        c.physics_concepts = u.physics_concepts,
                        c.tags = u.tags
                    """,
                    updates=updates,
                )
                labeled_count += len(updates)

            logger.info(
                "Labeled %d/%d clusters",
                min(i + batch_size, len(unlabeled)),
                len(unlabeled),
            )

        logger.info("Cluster labeling complete: %d clusters labeled", labeled_count)
        return labeled_count

    except Exception as e:
        logger.error("Error during cluster labeling: %s", e)
        return 0


def _embed_cluster_text(
    client: GraphClient,
    embedding_batch_size: int = 256,
    use_rich: bool | None = None,
    force_reembed: bool = False,
) -> None:
    """Embed cluster labels and descriptions with per-cluster hash caching.

    Generates two embedding vectors per cluster:
    - label_embedding: from the short label text (3-6 words)
    - description_embedding: from the longer description (1-2 sentences)

    Only re-embeds clusters whose text_embedding_hash has changed (i.e.,
    label or description text changed, or the embedding model changed).
    When force_reembed is True, skips hash comparison and re-embeds everything.
    Clusters without labels are skipped.

    Args:
        client: GraphClient instance
        embedding_batch_size: Number of texts to embed per batch
        use_rich: Force rich progress setting
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_embedding_dimension, get_embedding_model

    # Fetch all clusters with labels/descriptions
    results = client.query("""
        MATCH (c:IMASSemanticCluster)
        WHERE c.label IS NOT NULL
        RETURN c.id AS id, c.label AS label, c.description AS description,
               c.text_embedding_hash AS existing_hash
        ORDER BY c.id
    """)

    if not results:
        logger.warning("No clusters with labels found for embedding")
        return

    dim = get_embedding_dimension()
    model_name = get_embedding_model()

    # Ensure vector indexes exist
    for index_name, prop_name in [
        ("cluster_label_embedding", "label_embedding"),
        ("cluster_description_embedding", "description_embedding"),
    ]:
        client.query(f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:IMASSemanticCluster) ON n.{prop_name}
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine',
                    `vector.quantization.enabled`: true
                }}
            }}
        """)

    # Compute text hashes and filter to clusters needing re-embedding
    clusters_to_embed = []
    cached_count = 0
    for r in results:
        label = r["label"]
        description = r["description"] or label
        text_hash = compute_embedding_hash(f"{label}|{description}", model_name)
        if not force_reembed and text_hash == r.get("existing_hash"):
            cached_count += 1
            continue
        clusters_to_embed.append(
            {
                "id": r["id"],
                "label": label,
                "description": description,
                "text_hash": text_hash,
            }
        )

    if not clusters_to_embed:
        logger.info("All %d cluster text embeddings up to date (cached)", cached_count)
        return

    logger.info(
        "Embedding %d cluster labels/descriptions (%d cached, %d-dim)",
        len(clusters_to_embed),
        cached_count,
        dim,
    )

    config = EncoderConfig(normalize_embeddings=True, use_rich=use_rich or False)
    encoder = Encoder(config=config)

    labels = [c["label"] for c in clusters_to_embed]
    descriptions = [c["description"] for c in clusters_to_embed]

    label_embeddings = encoder.embed_texts(labels)
    description_embeddings = encoder.embed_texts(descriptions)

    # Write to graph in batches
    batch_size = 500
    for i in range(0, len(clusters_to_embed), batch_size):
        batch_clusters = clusters_to_embed[i : i + batch_size]
        batch_label_embs = label_embeddings[i : i + batch_size]
        batch_desc_embs = description_embeddings[i : i + batch_size]

        batch_data = [
            {
                "id": c["id"],
                "label_embedding": lemb.tolist(),
                "description_embedding": demb.tolist(),
                "text_embedding_hash": c["text_hash"],
            }
            for c, lemb, demb in zip(
                batch_clusters, batch_label_embs, batch_desc_embs, strict=True
            )
        ]

        client.query(
            """
            UNWIND $batch AS b
            MATCH (c:IMASSemanticCluster {id: b.id})
            SET c.label_embedding = b.label_embedding,
                c.description_embedding = b.description_embedding,
                c.text_embedding_hash = b.text_embedding_hash
            """,
            batch=batch_data,
        )

    logger.info(
        "Embedded %d cluster labels/descriptions (%d cached)",
        len(clusters_to_embed),
        cached_count,
    )


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
    "phase_extract",
    "phase_build",
    "phase_enrich",
    "phase_embed",
    "phase_cluster",
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
    orphaned nodes. Cross-references from facility nodes (IMASMapping,
    MENTIONS_IMAS, etc.) are detached before deletion.

    Unit nodes are only deleted if no non-DD nodes reference them.

    Deletion order:
    1. EmbeddingChange (leaf, references IMASNode + DDVersion)
    2. IMASNodeChange (references IMASNode + DDVersion)
    3. IMASSemanticCluster (references IMASNode via IN_CLUSTER)
    4. IdentifierSchema (referenced by IMASNode)
    5. IMASNode (bulk — the largest set)
    6. IDS (referenced by IMASNode)
    7. DDVersion (referenced by IMASNode, IMASNodeChange)
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
        ("IMASNodeChange", "path_changes"),
        ("IMASSemanticCluster", "clusters"),
        ("IdentifierSchema", "identifier_schemas"),
        ("IMASNode", "paths"),
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
