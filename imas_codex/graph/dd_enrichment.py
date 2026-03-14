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
            "Physics-aware description of this IMAS path (2-4 sentences). "
            "Explains what the quantity measures, its physical significance, "
            "and its role in the IDS structure. Do NOT repeat units, data type, "
            "or coordinate information already in metadata fields."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Searchable keywords (max 5) — physics concepts, measurement types, "
            "and related terms not already in the documentation"
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
]


def is_boilerplate_path(path_id: str) -> bool:
    """Check if a path is a boilerplate error/validity field."""
    name = path_id.split("/")[-1]
    return any(p.search(name) for p in BOILERPLATE_PATTERNS)


def generate_template_description(path_id: str, path_info: dict) -> dict[str, Any]:
    """Generate a template description for boilerplate paths.

    Returns dict with description, keywords suitable for
    direct graph update. No LLM call needed.
    """
    name = path_info.get("name", path_id.split("/")[-1])
    ids_name = path_id.split("/")[0]

    # Extract the base field name (remove the error/validity suffix)
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

    # Build template description
    base_readable = base_name.replace("_", " ")

    if error_type and "error" in error_type:
        desc = (
            f"Error metadata for the {base_readable} field. "
            f"Provides standardized error reporting for {base_readable} measurements."
        )
        keywords = ["error", "uncertainty", base_name]
    elif error_type == "validity":
        desc = (
            f"Validity status indicator for the {base_readable} field. "
            f"Integer code indicating data quality or processing status."
        )
        keywords = ["validity", "status", "quality", base_name]
    elif error_type == "validity_timed":
        desc = (
            f"Time-varying validity status for the {base_readable} field. "
            f"Array of validity codes aligned with the time base."
        )
        keywords = ["validity", "time-varying", "status", base_name]
    else:
        desc = f"Data field in {ids_name}"
        keywords = [base_name]

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
           collect(DISTINCT sibling.name)[0..10] AS siblings,
           parent.documentation AS parent_doc
    """
    sibling_results = {
        r["path_id"]: r for r in client.query(sibling_query, path_ids=path_ids)
    }

    # Query child summary for structure nodes
    children_query = """
    UNWIND $path_ids AS pid
    MATCH (p:IMASNode {id: pid})
    WHERE p.data_type IN ['STRUCTURE', 'STRUCT_ARRAY']
    OPTIONAL MATCH (child:IMASNode)-[:HAS_PARENT]->(p)
    RETURN pid AS path_id,
           collect(DISTINCT child.name)[0..15] AS children
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
            "siblings": ctx.get("siblings", []),
            "children": ctx.get("children", []),
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
    user_lines = ["Enrich the following IMAS paths:\n"]
    for entry in batch_data:
        user_lines.append(f"\n### Path {entry['index']}: `{entry['path']}`")
        user_lines.append(f"- Name: {entry['name']}")
        if entry["documentation"]:
            user_lines.append(f"- Documentation: {entry['documentation']}")
        if entry["data_type"]:
            user_lines.append(f"- Data type: {entry['data_type']}")
        if entry["parent_chain"]:
            user_lines.append(f"- Parent chain: {entry['parent_chain']}")
        if entry["siblings"]:
            user_lines.append(f"- Siblings: {', '.join(entry['siblings'][:8])}")
        if entry["children"]:
            user_lines.append(f"- Children: {', '.join(entry['children'][:10])}")
        if entry["unit"]:
            user_lines.append(f"- Unit: {entry['unit']}")
        if entry["coordinates"]:
            user_lines.append(f"- Coordinates: {', '.join(entry['coordinates'])}")
        if entry["cluster_label"]:
            user_lines.append(f"- Cluster: {entry['cluster_label']}")
        if entry["cocos_label"]:
            user_lines.append(f"- COCOS label: {entry['cocos_label']}")
        if entry["ids_description"]:
            user_lines.append(f"- IDS description: {entry['ids_description'][:200]}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


# =============================================================================
# Main Enrichment Worker
# =============================================================================


def enrich_imas_paths(
    client: GraphClient,
    version: str,
    *,
    model: str | None = None,
    batch_size: int = 50,
    ids_filter: set[str] | None = None,
    use_rich: bool | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Enrich IMAS paths with LLM-generated descriptions.

    For each path lacking a description (or with mismatched enrichment_hash):
    1. Gather rich context (hierarchy, siblings, metadata)
    2. Call LLM to generate description, keywords
    3. Update graph with enrichment fields
    4. Optionally propagate physics_domain back to graph

    Boilerplate paths (error/validity fields) get template descriptions
    without LLM calls.

    Args:
        client: Neo4j GraphClient
        version: DD version to enrich (e.g., "4.0.0")
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
        # Force re-enrich all paths
        paths_query = f"""
        MATCH (p:IMASNode)-[:INTRODUCED_IN]->(v:DDVersion {{id: $version}})
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
        MATCH (p:IMASNode)-[:INTRODUCED_IN]->(v:DDVersion {{id: $version}})
        WHERE (p.description IS NULL OR p.enrichment_hash IS NULL)
        {filter_clause}
        RETURN p.id AS id, p.name AS name, p.documentation AS documentation,
               p.data_type AS data_type, p.ids AS ids,
               p.cocos_label_transformation AS cocos_label_transformation,
               p.enrichment_hash AS enrichment_hash
        ORDER BY p.id
        """

    params = {"version": version}
    if ids_filter:
        params["ids_filter"] = list(ids_filter)

    all_paths = list(client.query(paths_query, **params))
    logger.info(f"Found {len(all_paths)} paths to enrich for version {version}")

    if not all_paths:
        return stats

    # Query IDS info for context
    ids_query = """
    MATCH (i:IDS)
    RETURN i.id AS id, i.description AS description, i.physics_domain AS physics_domain
    """
    ids_info = {r["id"]: r for r in client.query(ids_query)}

    # Separate boilerplate vs LLM paths
    boilerplate_paths = []
    llm_paths = []
    for path in all_paths:
        if is_boilerplate_path(path["id"]):
            boilerplate_paths.append(path)
        else:
            llm_paths.append(path)

    # Process boilerplate paths (no LLM)
    if boilerplate_paths:
        monitor.status(
            f"Generating template descriptions for {len(boilerplate_paths)} boilerplate paths..."
        )
        template_updates = []
        for path in boilerplate_paths:
            template = generate_template_description(path["id"], path)
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
                    result, cost, tokens = call_llm_structured(
                        model=model,
                        messages=messages,
                        response_model=IMASPathEnrichmentBatch,
                    )
                    stats["enrichment_cost"] += cost
                    stats["enrichment_tokens"] += tokens

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

                except Exception as e:
                    logger.error(f"Error enriching batch {batch_num + 1}: {e}")
                    # Continue with next batch

                phase.update(batch_items[batch_num])

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
                p.enrichment_source = u.enrichment_source
            WITH p, u
            WHERE u.physics_domain IS NOT NULL
            SET p.physics_domain = u.physics_domain
            """,
            updates=batch,
        )
