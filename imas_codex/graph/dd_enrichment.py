"""LLM enrichment for IMAS Data Dictionary paths.

Generates physics-aware descriptions, keywords, and physics_domain updates
for IMASNode paths to improve semantic search quality. Uses batch LLM calls
with tree hierarchy context to produce rich descriptions.

Follows the idempotent pattern: paths are only enriched once unless the
context or model changes (tracked via enrichment_hash).

Usage:
    from imas_codex.graph.dd_enrichment import enrich_imas_paths

    stats = enrich_imas_paths(
        client=graph_client,
        version="4.0.0",
        model="google/gemini-3-flash-preview",
        batch_size=50,
    )
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Response Models
# =============================================================================


class IMASPathEnrichmentResult(BaseModel):
    """Enrichment result for a single IMAS path."""

    path_index: int = Field(description="1-based index matching the input batch order")
    description: str = Field(
        description=(
            "Clear physics description (2-3 sentences, 150-300 characters) "
            "that names the physical quantity with its standard abbreviation. "
            "Do NOT repeat units, data type, coordinates, or COCOS info."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Searchable keywords (up to 8) — physics abbreviations/symbols, "
            "concepts, measurement types, diagnostic names, and related "
            "terms not already in the description or path name"
        ),
    )
    physics_domain: str | None = Field(
        default=None,
        description=(
            "Primary physics domain for this path. Use ONLY if the path clearly "
            "belongs to a domain different from the IDS-level physics_domain. "
            "Leave null to inherit from IDS."
        ),
    )


class IMASPathEnrichmentBatch(BaseModel):
    """Batch enrichment response from the LLM."""

    results: list[IMASPathEnrichmentResult] = Field(
        description="Enrichment results, one per input path in batch order"
    )


# =============================================================================
# Boilerplate Detection
# =============================================================================

# Patterns for error/validity fields that use template descriptions
BOILERPLATE_PATTERNS = [
    re.compile(r"_error_index$"),
    re.compile(r"_error_lower$"),
    re.compile(r"_error_upper$"),
    re.compile(r"_validity$"),
    re.compile(r"_validity_timed$"),
    re.compile(r"^validity$"),
    re.compile(r"^validity_timed$"),
]

# Accessor terminal classification — nodes that derive meaning from their parent
# and should be template-enriched rather than LLM-enriched.
ACCESSOR_TERMINAL_NAMES = frozenset(
    {
        # Generic data containers
        "value",
        "data",
        "values",
        # Time bases
        "time",
        # Grid references
        "grid_index",
        "grid_subset_index",
        "index",
        # Interpolation
        "coefficients",
        # Geometric components
        "r",
        "z",
        "phi",
        "x",
        "y",
        # Directional components
        "parallel",
        "poloidal",
        "radial",
        "toroidal",
        "diamagnetic",
        # Validity (also covered by BOILERPLATE but explicit here)
        "validity",
        "validity_timed",
        # Fit results
        "measured",
        "reconstructed",
        "chi_squared",
        # Labels/identifiers
        "label",
        # GGD structure
        "neighbours",
        "nodes",
        "measure",
        "space",
        "geometry",
        "geometry_2d",
        "dim1",
        "dim2",
        "closed",
        # Other common accessors
        "surface",
        "weight",
        "multiplicity",
        "a",
        "d",
        "v",
    }
)

ACCESSOR_TERMINAL_SUFFIXES = ("_coefficients", "_n")

# Template descriptions for accessor terminals, keyed by name
ACCESSOR_TEMPLATES: dict[str, str] = {
    "value": "{parent_doc_short}",
    "data": "Data array for {parent_name_readable}.",
    "values": "Array of values for {parent_name_readable}.",
    "time": "Time base for {parent_name_readable}.",
    "coefficients": "Interpolation coefficients for {parent_name_readable}.",
    "grid_index": "Grid index reference for {parent_name_readable}.",
    "grid_subset_index": "Grid subset index for {parent_name_readable}.",
    "index": "Integer index for {parent_name_readable}.",
    "label": "String label for {parent_name_readable}.",
    "r": "Major radius (R) coordinate of {parent_name_readable}.",
    "z": "Vertical (Z) coordinate of {parent_name_readable}.",
    "phi": "Toroidal angle (φ) coordinate of {parent_name_readable}.",
    "x": "X coordinate of {parent_name_readable}.",
    "y": "Y coordinate of {parent_name_readable}.",
    "parallel": "Parallel component of {parent_name_readable}.",
    "poloidal": "Poloidal component of {parent_name_readable}.",
    "radial": "Radial component of {parent_name_readable}.",
    "toroidal": "Toroidal component of {parent_name_readable}.",
    "diamagnetic": "Diamagnetic component of {parent_name_readable}.",
    "validity": "Integer validity flag for {parent_name_readable}.",
    "validity_timed": "Time-dependent validity array for {parent_name_readable}.",
    "measured": "Measured value of {parent_name_readable}.",
    "reconstructed": "Reconstructed value of {parent_name_readable}.",
    "chi_squared": "Chi-squared goodness of fit for {parent_name_readable}.",
    "neighbours": "Neighbor indices for {parent_name_readable}.",
    "nodes": "Node indices for {parent_name_readable}.",
    "measure": "Measure (area/volume) of {parent_name_readable}.",
    "space": "Space reference for {parent_name_readable}.",
    "geometry": "Geometry specification for {parent_name_readable}.",
    "geometry_2d": "2D geometry specification for {parent_name_readable}.",
    "dim1": "First dimension of {parent_name_readable}.",
    "dim2": "Second dimension of {parent_name_readable}.",
    "closed": "Closed surface flag for {parent_name_readable}.",
    "surface": "Surface specification of {parent_name_readable}.",
    "weight": "Weight factor for {parent_name_readable}.",
    "multiplicity": "Multiplicity count for {parent_name_readable}.",
}

# Structural metadata subtrees identical across all IDS
METADATA_PREFIXES = (
    "ids_properties/",
    "/ids_properties/",
    "/code/",
)

# Known metadata leaf fields and their descriptions
METADATA_FIELD_TEMPLATES: dict[str, tuple[str, list[str]]] = {
    "homogeneous_time": (
        "Integer flag indicating the time mode of this IDS: "
        "0 = heterogeneous (fields may have independent time bases), "
        "1 = homogeneous (all time-dependent fields share a single time vector), "
        "2 = independent (no time correlation between fields).",
        ["time", "homogeneous", "metadata"],
    ),
    "comment": (
        "Free-text annotation describing this IDS occurrence, "
        "such as code settings, run conditions, or data provenance notes.",
        ["comment", "annotation", "metadata"],
    ),
    "provider": (
        "Name of the person or system that created this data entry.",
        ["provider", "author", "metadata"],
    ),
    "creation_date": (
        "ISO 8601 timestamp recording when this IDS occurrence was written.",
        ["timestamp", "creation", "metadata"],
    ),
    "version_put": (
        "Container recording the exact IMAS software versions used when writing this data.",
        ["version", "provenance", "software"],
    ),
    "data_dictionary": (
        "Version string of the IMAS Data Dictionary used when this IDS was written.",
        ["version", "data-dictionary", "schema"],
    ),
    "access_layer": (
        "Version of the IMAS Access Layer software used for I/O.",
        ["version", "access-layer", "software"],
    ),
    "access_layer_language": (
        "Programming language binding of the Access Layer (e.g. Python, Fortran).",
        ["language", "binding", "access-layer"],
    ),
    "occurrence_type": (
        "Classifier for this IDS occurrence indicating whether it contains "
        "core plasma data, edge data, or another category.",
        ["occurrence", "classification", "metadata"],
    ),
}


def is_boilerplate_path(path_id: str) -> bool:
    """Check if a path is a boilerplate error/validity/metadata field."""
    name = path_id.split("/")[-1]
    if any(p.search(name) for p in BOILERPLATE_PATTERNS):
        return True
    # Structural metadata subtrees (ids_properties/*, code/*)
    parts = path_id.split("/")
    if len(parts) >= 2 and parts[1] in ("ids_properties", "code"):
        return True
    return False


def _is_boilerplate_sibling(name: str) -> bool:
    """Check if a sibling name is a boilerplate error/validity field."""
    return any(p.search(name) for p in BOILERPLATE_PATTERNS)


def is_accessor_terminal(path_id: str, name: str | None = None) -> bool:
    """Check if a node is an accessor terminal (template-enriched, not embedded).

    Accessor terminals are leaf nodes whose meaning derives from their parent.
    They include error/validity fields, metadata subtrees, and generic
    data containers like 'value', 'time', 'r', 'z', etc.
    """
    if name is None:
        name = path_id.split("/")[-1]
    return classify_node(path_id, name) == "accessor"


# Force-include physics concepts — protects real physics quantities from
# frequency-based misclassification (Layer 2 of classify_node)
FORCE_INCLUDE_CONCEPTS = frozenset(
    {
        "psi",
        "density",
        "temperature",
        "pressure",
        "flux",
        "current",
        "voltage",
        "power",
        "energy",
        "frequency",
        "velocity",
        "momentum",
        "conductivity",
        "resistivity",
        "elongation",
        "triangularity",
        "b0",
        "b_field_r",
        "b_field_z",
        "q",
        "rho_tor_norm",
        "rho_pol_norm",
    }
)

# Regex patterns for unknown future accessor terminals (Layer 4)
ACCESSOR_REGEX_PATTERNS = [
    # Error/uncertainty bounds (current + future)
    re.compile(
        r"_(error|uncertainty|confidence|reliability)_(upper|lower|index|bound)$"
    ),
    # Standalone validity variants
    re.compile(r"^(error|uncertainty|validity)(_timed)?$"),
    # Coefficients (interpolation)
    re.compile(r"_coefficients$"),
    # Normalized variants
    re.compile(r"_n$"),
    # Flag/status patterns (future-proof)
    re.compile(r"_(flag|status|validate|check)$"),
    # Scale/offset patterns
    re.compile(r"_(scale|offset|weight|bias)$"),
]


def classify_node(path_id: str, name: str, node_stats: dict | None = None) -> str:
    """Classify a data node as 'concept' or 'accessor'.

    Layers evaluated in order. First match wins (short-circuit).

    Layer 1: Error/metadata (absolute, no false positives)
    Layer 2: Force-include physics concepts (semantic veto)
    Layer 3: Explicit accessor names (conservative list)
    Layer 4: Regex suffix/prefix patterns (future-proof)
    Layer 5: Frequency + structural heuristic (data-driven)
    Default: concept
    """
    # Layer 1: Error/metadata (absolute, no false positives)
    if _is_error_or_metadata(name, path_id):
        return "accessor"

    # Layer 2: Force-include physics concepts (semantic veto)
    if name in FORCE_INCLUDE_CONCEPTS:
        return "concept"

    # Layer 3: Explicit accessor names (conservative list)
    if name in ACCESSOR_TERMINAL_NAMES:
        return "accessor"

    # Layer 4: Regex suffix/prefix patterns (future-proof)
    if _matches_accessor_pattern(name):
        return "accessor"

    # Layer 5: Frequency + structural heuristic (data-driven)
    if node_stats:
        occ = node_stats.get("occurrence_count", 0)
        struct_ratio = node_stats.get("structure_parent_ratio", 0.0)
        if occ >= 20 and struct_ratio >= 0.95:
            logger.info(
                "Accessor by heuristic: %s (occ=%d, ratio=%.2f)",
                name,
                occ,
                struct_ratio,
            )
            return "accessor"

    # Default: concept
    return "concept"


def _is_error_or_metadata(name: str, path_id: str) -> bool:
    """Layer 1: Error fields and structural metadata subtrees."""
    if any(p.search(name) for p in BOILERPLATE_PATTERNS):
        return True
    parts = path_id.split("/")
    if len(parts) >= 2 and parts[1] in ("ids_properties", "code"):
        return True
    return False


def _matches_accessor_pattern(name: str) -> bool:
    """Layer 4: Regex patterns for unknown future accessors."""
    return any(p.search(name) for p in ACCESSOR_REGEX_PATTERNS)


def generate_template_description(
    path_id: str, path_info: dict, parent_info: dict | None = None
) -> dict[str, Any]:
    """Generate a template description for boilerplate and accessor terminal paths.

    Covers four categories:
    - Accessor terminals (value, time, r, z, etc.) — uses parent context
    - Error fields (_error_upper, _error_lower, _error_index)
    - Validity fields (_validity, _validity_timed)
    - Structural metadata (ids_properties/*, code/*)

    Returns dict with description, keywords suitable for
    direct graph update. No LLM call needed.
    """
    name = path_info.get("name", path_id.split("/")[-1])
    ids_name = path_id.split("/")[0]
    parts = path_id.split("/")

    # --- Accessor terminal templates (with parent context) ---
    if name in ACCESSOR_TEMPLATES and parent_info:
        parent_name = parent_info.get("name", "parent")
        parent_name_readable = parent_name.replace("_", " ")
        parent_doc = parent_info.get("description", "") or parent_info.get(
            "documentation", ""
        )
        parent_doc_short = (
            (parent_doc[:150].rstrip(".") + ".")
            if parent_doc
            else f"{parent_name_readable}."
        )

        template = ACCESSOR_TEMPLATES[name]
        desc = template.format(
            parent_name_readable=parent_name_readable,
            parent_doc_short=parent_doc_short,
        )

        return {
            "description": desc,
            "keywords": [name, parent_name],
            "enrichment_source": "template",
        }

    # --- Suffix patterns (_coefficients, _n) with parent context ---
    if any(name.endswith(s) for s in ACCESSOR_TERMINAL_SUFFIXES) and parent_info:
        parent_name = parent_info.get("name", "parent")
        parent_name_readable = parent_name.replace("_", " ")
        if name.endswith("_coefficients"):
            base = name[: -len("_coefficients")].replace("_", " ")
            desc = f"Interpolation coefficients for {base} of {parent_name_readable}."
        elif name.endswith("_n"):
            base = name[:-2].replace("_", " ")
            desc = f"Normalized {base} of {parent_name_readable}."
        else:
            desc = f"{name.replace('_', ' ').capitalize()} of {parent_name_readable}."

        return {
            "description": desc,
            "keywords": [name, parent_name],
            "enrichment_source": "template",
        }

    # --- Structural metadata (ids_properties/*, code/*) ---
    if len(parts) >= 2 and parts[1] in ("ids_properties", "code"):
        return _generate_metadata_template(path_id, name, ids_name, parts)

    # --- Error / validity fields ---
    base_name = name
    error_type = None
    for suffix in (
        "_error_index",
        "_error_lower",
        "_error_upper",
        "_validity",
        "_validity_timed",
    ):
        if name.endswith(suffix):
            base_name = name[: -len(suffix)]
            error_type = suffix[1:]  # Remove leading underscore
            break

    base_readable = base_name.replace("_", " ")
    data_type = path_info.get("data_type", "")
    ndim = _ndim_from_dtype(data_type)

    if error_type == "error_upper":
        desc = (
            f"Upper bound of the uncertainty on {base_readable}{ndim}. "
            f"In the IMAS error convention, if only error_upper is populated "
            f"the error is symmetric and represents one standard deviation. "
            f"When both error_upper and error_lower are set, they define the "
            f"asymmetric uncertainty envelope around the {base_readable} value."
        )
        keywords = ["error", "uncertainty", "standard-deviation", base_name]
    elif error_type == "error_lower":
        desc = (
            f"Lower bound of the uncertainty on {base_readable}{ndim}. "
            f"Populated only for asymmetric errors; when absent, the error "
            f"is symmetric and fully described by error_upper alone."
        )
        keywords = ["error", "uncertainty", "asymmetric", base_name]
    elif error_type == "error_index":
        desc = (
            f"Integer index into the error description vector for {base_readable}. "
            f"Links this measurement to a specific error model or systematic "
            f"error source documented in the parent structure."
        )
        keywords = ["error", "index", "systematic", base_name]
    elif error_type == "validity":
        desc = (
            f"Validity flag for {base_readable} (INT_0D). "
            f"Integer code: 0 = valid, negative = invalid or not available, "
            f"positive = valid with caveats. Used to filter unreliable data."
        )
        keywords = ["validity", "quality", "status", base_name]
    elif error_type == "validity_timed":
        desc = (
            f"Time-dependent validity flag for {base_readable}. "
            f"Array of integer codes aligned with the time base, marking "
            f"each time slice as valid (0), invalid (negative), or "
            f"conditionally valid (positive)."
        )
        keywords = ["validity", "time-varying", "quality", base_name]
    else:
        desc = f"Data field in {ids_name}."
        keywords = [base_name]

    return {
        "description": desc,
        "keywords": keywords[:5],
        "enrichment_source": "template",
    }


def _ndim_from_dtype(data_type: str) -> str:
    """Extract dimensionality hint from IMAS data type string."""
    if not data_type:
        return ""
    # FLT_1D → " (1D array)", FLT_0D → " (scalar)", CPX_3D → " (3D array)"
    for dim in ("0D", "1D", "2D", "3D", "4D", "5D", "6D"):
        if dim in data_type:
            return " (scalar)" if dim == "0D" else f" ({dim} array)"
    return ""


def _generate_metadata_template(
    path_id: str, name: str, ids_name: str, parts: list[str]
) -> dict[str, Any]:
    """Generate template for ids_properties/* and code/* subtrees."""
    subtree = parts[1]  # 'ids_properties' or 'code'
    depth = len(parts) - 2  # depth within the subtree

    # Check for known field templates
    if name in METADATA_FIELD_TEMPLATES:
        desc, keywords = METADATA_FIELD_TEMPLATES[name]
        return {
            "description": desc,
            "keywords": keywords[:5],
            "enrichment_source": "template",
        }

    # Subtree roots
    if depth == 0:
        if subtree == "ids_properties":
            desc = (
                f"Metadata container for the {ids_name} IDS. Holds provenance, "
                f"versioning, time homogeneity flag, and plugin information "
                f"that describe how this data was produced and stored."
            )
            keywords = ["metadata", "provenance", "ids_properties", ids_name]
        else:  # code
            desc = (
                f"Container for the code that produced this {ids_name} data. "
                f"Records the code name, version, repository, commit hash, "
                f"parameters, and linked libraries for full reproducibility."
            )
            keywords = ["code", "provenance", "software", ids_name]
        return {
            "description": desc,
            "keywords": keywords[:5],
            "enrichment_source": "template",
        }

    # Common provenance fields that appear in multiple subtrees
    readable = name.replace("_", " ")
    provenance_fields: dict[str, str] = {
        "name": f"Identifier name for this {subtree} component.",
        "description": f"Description of this {subtree} component.",
        "commit": "Version control commit hash for reproducibility.",
        "version": "Software version string.",
        "repository": "URL of the source code repository.",
        "parameters": "Input parameters or configuration used for this run.",
        "path": f"IDS path reference within the {subtree} subtree.",
        "index": "Integer index identifying this item within its array.",
    }

    if name in provenance_fields:
        desc = provenance_fields[name]
        keywords = [name, "provenance", subtree]
    elif "plugin" in path_id or "infrastructure" in path_id:
        desc = (
            f"Access Layer plugin metadata ({readable}) recording which "
            f"I/O backend was used and its version."
        )
        keywords = ["plugin", "access-layer", name]
    elif "library" in path_id:
        desc = f"External library dependency ({readable}) linked by the producing code."
        keywords = ["library", "dependency", name]
    elif "provenance" in path_id:
        desc = f"Data provenance record ({readable}) tracking origins and processing."
        keywords = ["provenance", "lineage", name]
    else:
        desc = f"Metadata field ({readable}) in the {subtree} structure of {ids_name}."
        keywords = [subtree, name, "metadata"]

    return {
        "description": desc,
        "keywords": keywords[:5],
        "enrichment_source": "template",
    }


# =============================================================================
# Enrichment Hash
# =============================================================================


def compute_enrichment_hash(context_text: str, model_name: str) -> str:
    """Compute hash for enrichment idempotency.

    Includes model name so changing the model invalidates cache.
    """
    combined = f"{model_name}:{context_text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Context Gathering
# =============================================================================


def gather_path_context(
    client: GraphClient,
    paths: list[dict],
    ids_info: dict[str, dict],
) -> list[dict]:
    """Gather rich context for a batch of paths.

    For each path, collects:
    - Full parent chain (ancestors from IDS root)
    - Sibling paths (immediate siblings under same parent)
    - Child summary (for STRUCTURE/STRUCT_ARRAY nodes)
    - IDS-level context (description, COCOS)
    - Unit and coordinate information

    Args:
        client: Graph client for queries
        paths: List of path dicts with id, name, documentation, etc.
        ids_info: IDS metadata keyed by IDS name

    Returns:
        Enriched path contexts for prompt construction
    """
    enriched = []

    # Batch query for parent chains and siblings
    path_ids = [p["id"] for p in paths]

    # Query sibling paths (same parent)
    sibling_query = """
    UNWIND $path_ids AS pid
    MATCH (p:IMASNode {id: pid})
    OPTIONAL MATCH (p)-[:HAS_PARENT]->(parent:IMASNode)
    OPTIONAL MATCH (sibling:IMASNode)-[:HAS_PARENT]->(parent)
    WHERE sibling.id <> pid
    RETURN pid AS path_id,
           parent.id AS parent_id,
           collect(DISTINCT sibling.name) AS siblings,
           parent.documentation AS parent_doc
    """
    sibling_results = {
        r["path_id"]: r for r in client.query(sibling_query, path_ids=path_ids)
    }

    # Query full ancestor documentation chain (parent → grandparent → ... → IDS root)
    ancestor_query = """
    UNWIND $path_ids AS pid
    MATCH (p:IMASNode {id: pid})
    OPTIONAL MATCH chain = (p)-[:HAS_PARENT*]->(ancestor:IMASNode)
    WITH pid,
         [node IN nodes(chain)[1..] | {
           id: node.id,
           name: node.name,
           documentation: coalesce(node.description, node.documentation)
         }] AS ancestors
    RETURN pid AS path_id, ancestors
    """
    ancestor_results = {
        r["path_id"]: r["ancestors"] or []
        for r in client.query(ancestor_query, path_ids=path_ids)
    }

    # Query child summary for structure nodes
    children_query = """
    UNWIND $path_ids AS pid
    MATCH (p:IMASNode {id: pid})
    WHERE p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
    OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(p)
    RETURN pid AS path_id,
           collect(DISTINCT child.name) AS children
    """
    children_results = {
        r["path_id"]: r["children"]
        for r in client.query(children_query, path_ids=path_ids)
    }

    # Query unit and coordinate info
    meta_query = """
    UNWIND $path_ids AS pid
    MATCH (p:IMASNode {id: pid})
    OPTIONAL MATCH (p)-[:HAS_UNIT]->(u:Unit)
    OPTIONAL MATCH (p)-[:HAS_COORDINATE]->(c:IMASCoordinateSpec)
    OPTIONAL MATCH (p)-[:IN_CLUSTER]->(cl:IMASSemanticCluster)
    RETURN pid AS path_id,
           u.id AS unit,
           collect(DISTINCT c.id) AS coordinates,
           cl.label AS cluster_label
    """
    meta_results = {
        r["path_id"]: r for r in client.query(meta_query, path_ids=path_ids)
    }

    for path in paths:
        path_id = path["id"]
        ids_name = path_id.split("/")[0]

        ctx = {
            **path,
            "ids_description": ids_info.get(ids_name, {}).get("description", ""),
            "ids_cocos": ids_info.get(ids_name, {}).get("cocos", None),
            "parent_chain": _build_parent_chain(path_id),
            "ancestors": ancestor_results.get(path_id, []),
            "siblings": sibling_results.get(path_id, {}).get("siblings", []),
            "parent_doc": sibling_results.get(path_id, {}).get("parent_doc"),
            "children": children_results.get(path_id, []),
            "unit": meta_results.get(path_id, {}).get("unit"),
            "coordinates": meta_results.get(path_id, {}).get("coordinates", []),
            "cluster_label": meta_results.get(path_id, {}).get("cluster_label"),
        }
        enriched.append(ctx)

    return enriched


def _build_parent_chain(path_id: str) -> list[str]:
    """Extract parent chain from path ID.

    Example: "equilibrium/time_slice/profiles_1d/psi" ->
             ["equilibrium", "time_slice", "profiles_1d"]
    """
    parts = path_id.split("/")
    return parts[:-1] if len(parts) > 1 else []


# =============================================================================
# Prompt Construction
# =============================================================================


def build_enrichment_messages(
    batch_contexts: list[dict],
    ids_info: dict[str, dict],
) -> list[dict[str, Any]]:
    """Build LLM messages for enrichment batch.

    Args:
        batch_contexts: Enriched path contexts from gather_path_context
        ids_info: IDS metadata

    Returns:
        Messages list for call_llm_structured
    """
    from imas_codex.llm.prompt_loader import render_prompt

    # Group by IDS for coherent context
    ids_groups: dict[str, list[dict]] = {}
    for ctx in batch_contexts:
        ids_name = ctx["id"].split("/")[0]
        if ids_name not in ids_groups:
            ids_groups[ids_name] = []
        ids_groups[ids_name].append(ctx)

    # Build the batch context for the prompt
    batch_data = []
    for idx, ctx in enumerate(batch_contexts, 1):
        entry = {
            "index": idx,
            "path": ctx["id"],
            "name": ctx.get("name", ctx["id"].split("/")[-1]),
            "documentation": ctx.get("documentation", ""),
            "data_type": ctx.get("data_type", ""),
            "parent_chain": " / ".join(ctx.get("parent_chain", [])),
            "ancestors": ctx.get("ancestors", []),
            "siblings": ctx.get("siblings", []),
            "children": ctx.get("children", []),
            "parent_doc": ctx.get("parent_doc"),
            "unit": ctx.get("unit"),
            "coordinates": ctx.get("coordinates", []),
            "cluster_label": ctx.get("cluster_label"),
            "cocos_label": ctx.get("cocos_label_transformation"),
            "ids_description": ctx.get("ids_description", ""),
        }
        batch_data.append(entry)

    # Render the system prompt with schema context
    system_prompt = render_prompt(
        "imas/enrichment",
        context={"batch": batch_data},
    )

    # Build user message with batch data
    # Emit IDS descriptions once per IDS group to avoid repetition
    user_lines = ["Enrich the following IMAS paths:\n"]
    emitted_ids_descriptions: set[str] = set()

    for entry in batch_data:
        ids_name = entry["path"].split("/")[0]
        user_lines.append(f"\n### Path {entry['index']}: `{entry['path']}`")
        user_lines.append(f"- Name: {entry['name']}")
        if entry["documentation"]:
            user_lines.append(f"- Documentation: {entry['documentation']}")
        if entry["data_type"]:
            user_lines.append(f"- Data type: {entry['data_type']}")
        if entry["parent_chain"]:
            user_lines.append(f"- Parent chain: {entry['parent_chain']}")
        # Include ancestor documentation chain for hierarchy context
        ancestors = entry.get("ancestors", [])
        if ancestors:
            ancestor_lines = []
            for anc in ancestors:
                anc_name = anc.get("name", "")
                anc_doc = anc.get("documentation", "")
                if anc_doc:
                    ancestor_lines.append(f"  - {anc.get('id', anc_name)}: {anc_doc}")
                else:
                    ancestor_lines.append(f"  - {anc.get('id', anc_name)}")
            user_lines.append("- Ancestor context (parent → root):")
            user_lines.extend(ancestor_lines)
        elif entry.get("parent_doc"):
            user_lines.append(f"- Parent documentation: {entry['parent_doc']}")
        if entry["siblings"]:
            # Filter out boilerplate error/validity siblings and cap at 30
            meaningful = [
                s for s in entry["siblings"] if not _is_boilerplate_sibling(s)
            ]
            if len(meaningful) > 30:
                shown = meaningful[:30]
                user_lines.append(
                    f"- Siblings ({len(meaningful)} total, showing 30): "
                    f"{', '.join(shown)}"
                )
            else:
                user_lines.append(f"- Siblings: {', '.join(meaningful)}")
        if entry["children"]:
            user_lines.append(f"- Children: {', '.join(entry['children'])}")
        if entry["unit"]:
            user_lines.append(f"- Unit: {entry['unit']}")
        if entry["coordinates"]:
            user_lines.append(f"- Coordinates: {', '.join(entry['coordinates'])}")
        if entry["cluster_label"]:
            user_lines.append(f"- Cluster: {entry['cluster_label']}")
        # Only emit IDS description once per IDS to avoid repetition
        if entry["ids_description"] and ids_name not in emitted_ids_descriptions:
            user_lines.append(f"- IDS description: {entry['ids_description']}")
            emitted_ids_descriptions.add(ids_name)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# =============================================================================
# Main Enrichment Worker
# =============================================================================


def enrich_imas_paths(
    client: GraphClient,
    version: str | None = None,
    *,
    model: str | None = None,
    batch_size: int = 50,
    ids_filter: set[str] | None = None,
    use_rich: bool | None = None,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    on_cost: Callable[[float], None] | None = None,
    on_items: Callable[[list[dict], float], None] | None = None,
) -> dict[str, Any]:
    """Enrich IMAS paths with LLM-generated descriptions.

    Enriches ALL IMASNode paths across all DD versions, not just paths
    introduced in a specific version. Each path node is unique in the
    graph so there is no risk of duplicate enrichment.

    For each path lacking a description (or with mismatched enrichment_hash):
    1. Gather rich context (hierarchy, siblings, metadata)
    2. Call LLM to generate description, keywords
    3. Update graph with enrichment fields
    4. Optionally propagate physics_domain back to graph

    Boilerplate paths (error/validity fields) get template descriptions
    without LLM calls.

    Args:
        client: Neo4j GraphClient
        version: Unused, kept for API compatibility. All versions are enriched.
        model: LLM model for enrichment (defaults to language model from settings)
        batch_size: Paths per LLM call (default 50)
        ids_filter: Optional set of IDS names to filter
        use_rich: Force rich progress (True), logging (False), or auto (None)
        force: Re-enrich all paths regardless of hash

    Returns:
        Statistics dict with enriched_llm, enriched_template, cached, cost, etc.
    """
    from imas_codex.core.progress_monitor import create_build_monitor
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.settings import get_model

    if model is None:
        model = get_model("language")

    stats = {
        "enriched_llm": 0,
        "enriched_template": 0,
        "enrichment_cached": 0,
        "enrichment_cost": 0.0,
        "enrichment_tokens": 0,
        "physics_domains_updated": 0,
    }

    monitor = create_build_monitor(use_rich=use_rich, logger=logger)

    # Query unenriched paths
    filter_clause = ""
    if ids_filter:
        filter_clause = "AND p.ids IN $ids_filter"

    if force:
        # Force re-enrich all paths across all versions
        paths_query = f"""
        MATCH (p:IMASNode)
        WHERE 1=1 {filter_clause}
        RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
               p.data_type AS data_type, p.ids AS ids,
               p.cocos_label_transformation AS cocos_label_transformation,
               p.enrichment_hash AS enrichment_hash
        ORDER BY p.id
        """
    else:
        # Only paths lacking descriptions or with stale hashes
        paths_query = f"""
        MATCH (p:IMASNode)
        WHERE (p.description IS NULL OR p.enrichment_hash IS NULL)
        {filter_clause}
        RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
               p.data_type AS data_type, p.ids AS ids,
               p.cocos_label_transformation AS cocos_label_transformation,
               p.enrichment_hash AS enrichment_hash
        ORDER BY p.id
        """

    params: dict[str, Any] = {}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    all_paths = list(client.query(paths_query, **params))
    logger.info(f"Found {len(all_paths)} paths to enrich")

    if not all_paths:
        return stats

    if on_progress:
        on_progress(0, len(all_paths))

    # Query IDS info for context
    ids_query = """
    MATCH (i:IDS)
    RETURN i.id AS id, i.description AS description, i.physics_domain AS physics_domain
    """
    ids_info = {r["id"]: r for r in client.query(ids_query)}

    # Separate boilerplate/accessor vs LLM paths
    boilerplate_paths = []
    llm_paths = []
    for path in all_paths:
        name = path["id"].split("/")[-1]
        if is_accessor_terminal(path["id"], name):
            boilerplate_paths.append(path)
        else:
            llm_paths.append(path)

    # Process boilerplate/accessor paths (no LLM)
    if boilerplate_paths:
        monitor.status(
            f"Generating template descriptions for {len(boilerplate_paths)} "
            f"boilerplate/accessor paths..."
        )

        # Query parent info for accessor terminals that need parent context
        accessor_ids = [
            p["id"] for p in boilerplate_paths if not is_boilerplate_path(p["id"])
        ]
        parent_map: dict[str, dict] = {}
        if accessor_ids:
            parent_results = client.query(
                """
                UNWIND $path_ids AS pid
                MATCH (n:IMASNode {id: pid})-[:HAS_PARENT]->(parent:IMASNode)
                RETURN pid AS path_id, parent.id AS parent_id,
                       parent.name AS parent_name,
                       coalesce(parent.description, parent.documentation) AS parent_doc
                """,
                path_ids=accessor_ids,
            )
            parent_map = {
                r["path_id"]: {
                    "name": r["parent_name"],
                    "documentation": r["parent_doc"],
                }
                for r in parent_results
            }

        template_updates = []
        for path in boilerplate_paths:
            parent_info = parent_map.get(path["id"])
            template = generate_template_description(
                path["id"], path, parent_info=parent_info
            )
            template_hash = compute_enrichment_hash(
                f"{path['documentation']}", "template"
            )
            template_updates.append(
                {
                    "id": path["id"],
                    "description": template["description"],
                    "keywords": template["keywords"],
                    "enrichment_hash": template_hash,
                    "enrichment_model": "template",
                    "enrichment_source": "template",
                }
            )

        # Batch update boilerplate paths
        _batch_update_enrichments(client, template_updates)
        stats["enriched_template"] = len(template_updates)

        if on_progress:
            on_progress(len(boilerplate_paths), len(all_paths))

    # Process LLM paths in batches
    if llm_paths:
        total_batches = (len(llm_paths) + batch_size - 1) // batch_size
        batch_items = [f"Batch {i + 1}/{total_batches}" for i in range(total_batches)]

        with monitor.phase(
            "Enrich paths",
            items=batch_items,
            description_template="{item}",
            item_label="batches",
        ) as phase:
            for batch_idx in range(0, len(llm_paths), batch_size):
                batch = llm_paths[batch_idx : batch_idx + batch_size]
                batch_num = batch_idx // batch_size

                # Gather context for this batch
                batch_contexts = gather_path_context(client, batch, ids_info)

                # Check hashes for cached entries
                to_enrich = []
                cached = []
                for ctx in batch_contexts:
                    # Build context string for hash
                    ctx_str = f"{ctx['id']}:{ctx.get('documentation', '')}:{ctx.get('siblings', [])}"
                    expected_hash = compute_enrichment_hash(ctx_str, model)
                    if not force and ctx.get("enrichment_hash") == expected_hash:
                        cached.append(ctx)
                    else:
                        ctx["_expected_hash"] = expected_hash
                        to_enrich.append(ctx)

                stats["enrichment_cached"] += len(cached)

                if not to_enrich:
                    phase.update(batch_items[batch_num])
                    continue

                # Build messages and call LLM
                messages = build_enrichment_messages(to_enrich, ids_info)

                try:
                    batch_start = time.time()
                    result, cost, tokens = call_llm_structured(
                        model=model,
                        messages=messages,
                        response_model=IMASPathEnrichmentBatch,
                        service="data-dictionary",
                    )
                    stats["enrichment_cost"] += cost
                    stats["enrichment_tokens"] += tokens
                    batch_time = time.time() - batch_start

                    if on_cost:
                        on_cost(stats["enrichment_cost"])

                    # Build updates from LLM results
                    updates = []
                    for enrichment in result.results:
                        if enrichment.path_index < 1 or enrichment.path_index > len(
                            to_enrich
                        ):
                            logger.warning(
                                f"Invalid path_index {enrichment.path_index} in enrichment result"
                            )
                            continue

                        ctx = to_enrich[enrichment.path_index - 1]
                        update = {
                            "id": ctx["id"],
                            "description": enrichment.description,
                            "keywords": enrichment.keywords[:5],
                            "enrichment_hash": ctx["_expected_hash"],
                            "enrichment_model": model,
                            "enrichment_source": "llm",
                        }

                        # Handle physics_domain update if provided
                        if enrichment.physics_domain:
                            update["physics_domain"] = enrichment.physics_domain
                            stats["physics_domains_updated"] += 1

                        updates.append(update)

                    # Batch update graph
                    _batch_update_enrichments(client, updates)
                    stats["enriched_llm"] += len(updates)

                    # Stream enriched items for display
                    if on_items and updates:
                        stream_items = [
                            {
                                "primary_text": u["id"],
                                "description": u.get("description", ""),
                                "physics_domain": u.get("physics_domain", ""),
                            }
                            for u in updates
                        ]
                        on_items(stream_items, batch_time)

                except Exception as e:
                    logger.error(f"Error enriching batch {batch_num + 1}: {e}")
                    # Continue with next batch

                phase.update(batch_items[batch_num])

                if on_progress:
                    processed = len(boilerplate_paths) + min(
                        batch_idx + batch_size, len(llm_paths)
                    )
                    on_progress(processed, len(all_paths))

    return stats


def _batch_update_enrichments(
    client: GraphClient,
    updates: list[dict],
    batch_size: int = 500,
) -> None:
    """Batch update enrichment fields on IMASNode nodes.

    Updates: description, keywords, enrichment_hash,
    enrichment_model, enrichment_source, and optionally physics_domain.
    """
    for i in range(0, len(updates), batch_size):
        batch = updates[i : i + batch_size]
        client.query(
            """
            UNWIND $updates AS u
            MATCH (p:IMASNode {id: u.id})
            SET p.description = u.description,
                p.keywords = u.keywords,
                p.enrichment_hash = u.enrichment_hash,
                p.enrichment_model = u.enrichment_model,
                p.enrichment_source = u.enrichment_source,
                p.status = 'enriched',
                p.claimed_at = null,
                p.claim_token = null
            WITH p, u
            WHERE u.physics_domain IS NOT NULL
            SET p.physics_domain = u.physics_domain
            """,
            updates=batch,
        )


# =============================================================================
# Pass 2 — Refinement
# =============================================================================


def compute_refinement_hash(
    pass1_description: str,
    sibling_descriptions: list[str],
    cluster_peers: list[str],
    model_name: str,
) -> str:
    """Hash of Pass 2 inputs for idempotent re-refinement."""
    combined = (
        f"{model_name}:{pass1_description}:"
        + ":".join(sorted(sibling_descriptions))
        + ":".join(sorted(cluster_peers))
    )
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def gather_refinement_context(
    client: GraphClient,
    paths: list[dict],
) -> list[dict]:
    """Gather sibling and cluster peer context for refinement.

    For each path, fetches:
    - Own Pass 1 description
    - Sibling descriptions (same parent's children)
    - Cluster peer descriptions (IN_CLUSTER relationship, other IDSs)

    Uses two batch UNWIND queries for efficiency.
    """
    path_ids = [p["id"] for p in paths]

    # Query 1: Sibling descriptions (same parent, with descriptions)
    sibling_results: dict[str, list[dict]] = {}
    if path_ids:
        rows = client.query(
            """
            UNWIND $path_ids AS pid
            MATCH (p:IMASNode {id: pid})-[:HAS_PARENT]->(parent:IMASNode)
                  <-[:HAS_PARENT]-(sib:IMASNode)
            WHERE sib.id <> pid AND sib.description IS NOT NULL
            RETURN pid AS path_id,
                   collect(DISTINCT {name: sib.name,
                                     description: sib.description})[..20]
                   AS siblings
            """,
            path_ids=path_ids,
        )
        sibling_results = {r["path_id"]: r["siblings"] for r in rows}

    # Query 2: Cluster peer descriptions (different IDS, with descriptions)
    cluster_results: dict[str, list[dict]] = {}
    if path_ids:
        rows = client.query(
            """
            UNWIND $path_ids AS pid
            MATCH (p:IMASNode {id: pid})-[:IN_CLUSTER]->
                  (:IMASSemanticCluster)<-[:IN_CLUSTER]-(peer:IMASNode)
            WHERE peer.id <> pid
              AND peer.ids <> p.ids
              AND peer.description IS NOT NULL
            RETURN pid AS path_id,
                   collect(DISTINCT {path: peer.id,
                                     description: peer.description})[..10]
                   AS cluster_peers
            """,
            path_ids=path_ids,
        )
        cluster_results = {r["path_id"]: r["cluster_peers"] for r in rows}

    # Merge context for each path
    enriched: list[dict] = []
    for path in paths:
        pid = path["id"]
        ctx = {
            **path,
            "siblings": sibling_results.get(pid, []),
            "cluster_peers": cluster_results.get(pid, []),
        }
        enriched.append(ctx)

    return enriched


def build_refinement_messages(
    contexts: list[dict],
    ids_info: dict[str, dict],
) -> list[dict[str, Any]]:
    """Build LLM messages for the refinement prompt.

    Args:
        contexts: Enriched path contexts from gather_refinement_context
        ids_info: IDS metadata keyed by IDS name

    Returns:
        Messages list for call_llm_structured
    """
    from imas_codex.llm.prompt_loader import render_prompt

    # Build the batch context for the prompt template
    batch_data = []
    for idx, ctx in enumerate(contexts, 1):
        entry = {
            "index": idx,
            "path": ctx["id"],
            "name": ctx.get("name", ctx["id"].split("/")[-1]),
            "data_type": ctx.get("data_type", ""),
            "pass1_description": ctx.get("description", ""),
            "siblings": ctx.get("siblings", []),
            "cluster_peers": ctx.get("cluster_peers", []),
        }
        batch_data.append(entry)

    # Render the system prompt with schema context
    system_prompt = render_prompt(
        "imas/refinement",
        context={"batch": batch_data},
    )

    # Build user message with batch data
    user_lines = ["Refine the following IMAS path descriptions:\n"]
    emitted_ids: set[str] = set()

    for entry in batch_data:
        ids_name = entry["path"].split("/")[0]
        user_lines.append(f"\n### Path {entry['index']}: `{entry['path']}`")
        user_lines.append(f"- Name: {entry['name']}")
        if entry["data_type"]:
            user_lines.append(f"- Data type: {entry['data_type']}")
        if entry["pass1_description"]:
            user_lines.append(f"- Pass 1 description: {entry['pass1_description']}")

        # Sibling context
        siblings = entry.get("siblings", [])
        if siblings:
            sib_lines = []
            for s in siblings[:15]:
                desc_preview = s["description"][:100]
                sib_lines.append(f"  - **{s['name']}**: {desc_preview}")
            user_lines.append(f"- Siblings ({len(siblings)} with descriptions):")
            user_lines.extend(sib_lines)

        # Cluster peer context
        peers = entry.get("cluster_peers", [])
        if peers:
            peer_lines = []
            for p in peers[:8]:
                desc_preview = p["description"][:120]
                peer_lines.append(f"  - `{p['path']}`: {desc_preview}")
            user_lines.append(f"- Cluster peers in other IDSs ({len(peers)}):")
            user_lines.extend(peer_lines)

        # IDS description (once per IDS)
        ids_desc = ids_info.get(ids_name, {}).get("description", "")
        if ids_desc and ids_name not in emitted_ids:
            user_lines.append(f"- IDS description: {ids_desc}")
            emitted_ids.add(ids_name)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
