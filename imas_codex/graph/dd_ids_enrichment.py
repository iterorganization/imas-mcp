"""LLM enrichment and embedding for IMAS IDS nodes.

Generates physics-aware descriptions, keywords, and embeddings for IDS
(Interface Data Structure) nodes to enable semantic search at the IDS level.

Follows the same idempotent pattern as dd_identifier_enrichment.py:
- Hash-based caching prevents redundant LLM calls
- Batch processing for efficiency
- Embedding generation with vector index creation

Usage:
    from imas_codex.graph.dd_ids_enrichment import enrich_ids_nodes

    stats = enrich_ids_nodes(client=graph_client, model="...")
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Callable

    from imas_codex.graph import GraphClient

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Response Models
# =============================================================================


class IDSEnrichmentResult(BaseModel):
    """Enrichment result for a single IDS."""

    ids_index: int = Field(description="1-based index matching the input batch order")
    description: str = Field(
        description=(
            "Physics-aware description of this IDS (3-5 sentences). "
            "Explains what physics the IDS captures, typical data sources, "
            "measurement systems or simulations that produce this data, "
            "and how it relates to other IDSs in the physics workflow."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Searchable keywords (up to 8) — physics domains, measurement "
            "types, analysis methods, diagnostic categories"
        ),
    )


class IDSEnrichmentBatch(BaseModel):
    """Batch enrichment response from the LLM."""

    results: list[IDSEnrichmentResult] = Field(
        description="Enrichment results, one per input IDS in batch order"
    )


# =============================================================================
# Hash computation
# =============================================================================


def _compute_enrichment_hash(context_text: str, model_name: str) -> str:
    """Compute hash for enrichment idempotency."""
    combined = f"{model_name}:{context_text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Context Gathering
# =============================================================================


def _gather_ids_context(
    client: GraphClient,
    ids_list: list[dict],
) -> list[dict]:
    """Gather rich context for each IDS to provide to the LLM.

    For each IDS, collects:
    - Top-level structural sections (first 2 levels of the IDS tree)
    - Identifier schemas used by paths in this IDS
    - Sibling IDS in the same physics domain
    - Path/leaf counts and lifecycle info (already on node)
    """
    ids_names = [ids["id"] for ids in ids_list]

    # Query top-level sections (depth 1 and 2 children)
    sections_query = """
    UNWIND $ids_names AS ids_name
    MATCH (p:IMASNode {ids: ids_name})
    WHERE size(split(p.id, '/')) <= 3
      AND p.node_category = 'data'
    RETURN p.ids AS ids_name,
           p.id AS path,
           p.name AS name,
           p.documentation AS documentation,
           p.data_type AS data_type
    ORDER BY p.id
    """
    sections_by_ids: dict[str, list[dict]] = {n: [] for n in ids_names}
    for r in client.query(sections_query, ids_names=ids_names):
        sections_by_ids[r["ids_name"]].append(
            {
                "path": r["path"],
                "name": r["name"],
                "documentation": r["documentation"] or "",
                "data_type": r["data_type"] or "",
            }
        )

    # Query identifier schemas used by each IDS
    ident_query = """
    UNWIND $ids_names AS ids_name
    MATCH (p:IMASNode {ids: ids_name})-[:HAS_IDENTIFIER_SCHEMA]->(s:IdentifierSchema)
    RETURN p.ids AS ids_name,
           s.name AS schema_name,
           s.documentation AS schema_documentation,
           s.option_count AS option_count
    """
    ident_by_ids: dict[str, list[dict]] = {n: [] for n in ids_names}
    seen_schemas: dict[str, set[str]] = {n: set() for n in ids_names}
    for r in client.query(ident_query, ids_names=ids_names):
        ids_name = r["ids_name"]
        schema_name = r["schema_name"]
        if schema_name not in seen_schemas[ids_name]:
            seen_schemas[ids_name].add(schema_name)
            ident_by_ids[ids_name].append(
                {
                    "name": schema_name,
                    "documentation": r["schema_documentation"] or "",
                    "option_count": r["option_count"] or 0,
                }
            )

    # Query physics domain grouping (sibling IDS)
    domain_query = """
    MATCH (i:IDS)
    RETURN i.id AS name, i.physics_domain AS domain
    """
    domain_groups: dict[str, list[str]] = {}
    for r in client.query(domain_query):
        domain = r["domain"] or "general"
        domain_groups.setdefault(domain, []).append(r["name"])

    # Build enriched context
    enriched = []
    for ids in ids_list:
        ids_name = ids["id"]
        domain = ids.get("physics_domain") or "general"
        siblings = [n for n in domain_groups.get(domain, []) if n != ids_name]

        enriched.append(
            {
                **ids,
                "sections": sections_by_ids.get(ids_name, []),
                "identifier_schemas": ident_by_ids.get(ids_name, []),
                "domain_siblings": siblings,
            }
        )

    return enriched


# =============================================================================
# Main Enrichment Function
# =============================================================================


def enrich_ids_nodes(
    client: GraphClient,
    *,
    model: str | None = None,
    batch_size: int = 30,
    force: bool = False,
    on_items: Callable[[list[dict], float], None] | None = None,
) -> dict[str, Any]:
    """Enrich IDS nodes with LLM-generated descriptions and keywords.

    For each IDS lacking enrichment (or with mismatched hash):
    1. Gather context (sections, identifier schemas, domain siblings)
    2. Call LLM to generate description and keywords
    3. Update graph with description, keywords

    Args:
        client: Neo4j GraphClient
        model: LLM model for enrichment (defaults to language model)
        batch_size: IDS per LLM call
        force: Re-enrich all IDS regardless of hash
        on_items: Optional callback for streaming enriched item labels

    Returns:
        Statistics dict with counts.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.settings import get_model

    if model is None:
        model = get_model("language")

    stats: dict[str, Any] = {
        "enriched": 0,
        "cached": 0,
        "cost": 0.0,
        "tokens": 0,
    }

    # Query IDS nodes
    if force:
        query = """
        MATCH (i:IDS)
        RETURN i.id AS id, i.name AS name, i.documentation AS documentation,
               i.physics_domain AS physics_domain,
               i.path_count AS path_count, i.leaf_count AS leaf_count,
               i.max_depth AS max_depth,
               i.lifecycle_status AS lifecycle_status,
               i.ids_type AS ids_type,
               i.enrichment_hash AS enrichment_hash
        ORDER BY i.id
        """
    else:
        query = """
        MATCH (i:IDS)
        WHERE i.description IS NULL OR i.enrichment_hash IS NULL
        RETURN i.id AS id, i.name AS name, i.documentation AS documentation,
               i.physics_domain AS physics_domain,
               i.path_count AS path_count, i.leaf_count AS leaf_count,
               i.max_depth AS max_depth,
               i.lifecycle_status AS lifecycle_status,
               i.ids_type AS ids_type,
               i.enrichment_hash AS enrichment_hash
        ORDER BY i.id
        """

    all_ids = list(client.query(query))
    logger.info(f"Found {len(all_ids)} IDS nodes to enrich")

    if not all_ids:
        return stats

    # Gather context for all IDS
    all_ids_ctx = _gather_ids_context(client, all_ids)

    # Process in batches
    for batch_idx in range(0, len(all_ids_ctx), batch_size):
        batch = all_ids_ctx[batch_idx : batch_idx + batch_size]

        # Check hashes for cached entries
        to_enrich = []
        for ids_ctx in batch:
            ctx_str = (
                f"{ids_ctx['id']}:{ids_ctx.get('documentation', '')}:"
                f"{ids_ctx.get('path_count', 0)}:"
                f"{json.dumps(ids_ctx.get('sections', []), sort_keys=True)}"
            )
            expected_hash = _compute_enrichment_hash(ctx_str, model)
            if not force and ids_ctx.get("enrichment_hash") == expected_hash:
                stats["cached"] += 1
            else:
                ids_ctx["_expected_hash"] = expected_hash
                to_enrich.append(ids_ctx)

        if not to_enrich:
            continue

        # Build prompt context
        batch_data = []
        for idx, ids_ctx in enumerate(to_enrich, 1):
            batch_data.append(
                {
                    "index": idx,
                    "name": ids_ctx["name"],
                    "documentation": ids_ctx.get("documentation") or "",
                    "physics_domain": ids_ctx.get("physics_domain") or "general",
                    "path_count": ids_ctx.get("path_count") or 0,
                    "leaf_count": ids_ctx.get("leaf_count") or 0,
                    "lifecycle_status": ids_ctx.get("lifecycle_status") or "",
                    "ids_type": ids_ctx.get("ids_type") or "",
                    "sections": ids_ctx.get("sections", []),
                    "identifier_schemas": ids_ctx.get("identifier_schemas", []),
                    "domain_siblings": ids_ctx.get("domain_siblings", []),
                }
            )

        # Render system prompt
        system_prompt = render_prompt(
            "imas/ids_enrichment",
            context={"batch": batch_data},
        )

        # Build user message with full context
        user_lines = ["Enrich the following IMAS IDS definitions:\n"]
        for entry in batch_data:
            user_lines.append(f"\n### IDS {entry['index']}: `{entry['name']}`")
            if entry["documentation"]:
                user_lines.append(f"- Raw DD description: {entry['documentation']}")
            user_lines.append(f"- Physics domain: {entry['physics_domain']}")
            user_lines.append(f"- Type: {entry['ids_type'] or 'dynamic'}")
            if entry["lifecycle_status"]:
                user_lines.append(f"- Lifecycle: {entry['lifecycle_status']}")
            user_lines.append(
                f"- Paths: {entry['path_count']} total, {entry['leaf_count']} leaves"
            )

            # Structural sections
            if entry["sections"]:
                user_lines.append("- Key structural sections:")
                for sec in entry["sections"]:
                    line = f"    - `{sec['path']}` ({sec['data_type']})"
                    if sec["documentation"]:
                        line += f": {sec['documentation']}"
                    user_lines.append(line)

            # Identifier schemas
            if entry["identifier_schemas"]:
                user_lines.append("- Identifier schemas used:")
                for ident in entry["identifier_schemas"]:
                    line = f"    - {ident['name']}"
                    if ident["documentation"]:
                        line += f" — {ident['documentation']}"
                    user_lines.append(line)

            # Domain siblings
            if entry["domain_siblings"]:
                user_lines.append(
                    f"- Related IDS (same domain): "
                    f"{', '.join(entry['domain_siblings'])}"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ]

        try:
            result, cost, tokens = call_llm_structured(
                model=model,
                messages=messages,
                response_model=IDSEnrichmentBatch,
            )
            stats["cost"] += cost
            stats["tokens"] += tokens

            # Build updates
            updates = []
            for enrichment in result.results:
                if enrichment.ids_index < 1 or enrichment.ids_index > len(to_enrich):
                    logger.warning(
                        f"Invalid ids_index {enrichment.ids_index} in result"
                    )
                    continue

                ids_ctx = to_enrich[enrichment.ids_index - 1]
                updates.append(
                    {
                        "id": ids_ctx["id"],
                        "description": enrichment.description,
                        "keywords": enrichment.keywords[:8],
                        "enrichment_hash": ids_ctx["_expected_hash"],
                        "enrichment_source": "llm",
                    }
                )

            # Batch update graph
            if updates:
                client.query(
                    """
                    UNWIND $updates AS u
                    MATCH (i:IDS {id: u.id})
                    SET i.description = u.description,
                        i.keywords = u.keywords,
                        i.enrichment_hash = u.enrichment_hash,
                        i.enrichment_source = u.enrichment_source
                    """,
                    updates=updates,
                )
                stats["enriched"] += len(updates)
                if on_items:
                    on_items(
                        [
                            {
                                "primary_text": update["id"],
                                "description": update["description"],
                            }
                            for update in updates
                        ],
                        0.0,
                    )

        except Exception as e:
            logger.error(f"Error enriching IDS batch: {e}")

    return stats


# =============================================================================
# Embedding Generation
# =============================================================================


def embed_ids_nodes(
    client: GraphClient,
    *,
    force_reembed: bool = False,
    on_items: Callable[[list[dict], float], None] | None = None,
) -> dict[str, int]:
    """Generate embeddings for enriched IDS nodes.

    Only re-embeds IDS whose content hash has changed.

    Returns:
        Stats dict with updated/cached counts.
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_embedding_dimension, get_embedding_model

    stats = {"updated": 0, "cached": 0}

    # Fetch IDS with enriched descriptions
    results = client.query("""
        MATCH (i:IDS)
        WHERE i.description IS NOT NULL AND i.enrichment_source IS NOT NULL
        RETURN i.id AS id, i.name AS name,
               i.description AS description,
               i.keywords AS keywords,
               i.embedding_hash AS existing_hash
        ORDER BY i.id
    """)

    if not results:
        logger.info("No enriched IDS nodes to embed")
        return stats

    dim = get_embedding_dimension()
    model_name = get_embedding_model()

    # Ensure vector index exists
    client.query(f"""
        CREATE VECTOR INDEX ids_embedding IF NOT EXISTS
        FOR (n:IDS) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine',
                `vector.quantization.enabled`: true
            }}
        }}
    """)

    # Compute text and hashes, filter to IDS needing re-embedding
    to_embed = []
    for r in results:
        keywords_str = ", ".join(r.get("keywords") or [])
        text = f"{r['name']}: {r['description']} Keywords: {keywords_str}"
        text_hash = hashlib.sha256(f"{model_name}:{text}".encode()).hexdigest()[:16]

        if not force_reembed and r.get("existing_hash") == text_hash:
            stats["cached"] += 1
            continue

        to_embed.append(
            {
                "id": r["id"],
                "text": text,
                "hash": text_hash,
            }
        )

    if not to_embed:
        logger.info(f"All {len(results)} IDS embeddings up to date")
        return stats

    # Generate embeddings
    encoder = Encoder(
        config=EncoderConfig(
            model_name=model_name,
            normalize_embeddings=True,
        )
    )
    texts = [item["text"] for item in to_embed]
    embeddings = encoder.embed_texts(texts)

    # Store embeddings
    batch_data = []
    for i, item in enumerate(to_embed):
        batch_data.append(
            {
                "id": item["id"],
                "embedding": embeddings[i].tolist(),
                "embedding_hash": item["hash"],
            }
        )

    client.query(
        """
        UNWIND $batch AS b
        MATCH (i:IDS {id: b.id})
        SET i.embedding = b.embedding,
            i.embedding_hash = b.embedding_hash
        """,
        batch=batch_data,
    )
    stats["updated"] = len(to_embed)
    if on_items and batch_data:
        on_items(
            [{"primary_text": item["id"]} for item in batch_data],
            0.0,
        )

    logger.info(f"IDS embeddings: {stats['updated']} updated, {stats['cached']} cached")
    return stats
