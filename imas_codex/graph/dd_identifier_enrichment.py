"""LLM enrichment for IMAS identifier schemas.

Generates physics-aware descriptions and keywords for IdentifierSchema
nodes to improve semantic search quality. Uses batch LLM calls with
option context to produce rich descriptions.

Follows the idempotent pattern: schemas are only enriched once unless
the context or model changes (tracked via enrichment_hash).

Usage:
    from imas_codex.graph.dd_identifier_enrichment import enrich_identifier_schemas

    stats = enrich_identifier_schemas(client=graph_client, model="...")
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


class IdentifierEnrichmentResult(BaseModel):
    """Enrichment result for a single identifier schema."""

    schema_index: int = Field(
        description="1-based index matching the input batch order"
    )
    description: str = Field(
        description=(
            "Physics-aware description of this identifier schema (2-4 sentences). "
            "Explains what the enumeration controls and why the options matter."
        )
    )
    keywords: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Searchable keywords — physics concepts, measurement types",
    )


class IdentifierEnrichmentBatch(BaseModel):
    """Batch enrichment response from the LLM."""

    results: list[IdentifierEnrichmentResult] = Field(
        description="Enrichment results, one per input schema in batch order"
    )


# =============================================================================
# Hash computation
# =============================================================================


def _compute_enrichment_hash(context_text: str, model_name: str) -> str:
    """Compute hash for enrichment idempotency."""
    combined = f"{model_name}:{context_text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]


# =============================================================================
# Main Enrichment Function
# =============================================================================


def enrich_identifier_schemas(
    client: GraphClient,
    *,
    model: str | None = None,
    batch_size: int = 30,
    force: bool = False,
    on_items: Callable[[list[dict], float], None] | None = None,
) -> dict[str, Any]:
    """Enrich IdentifierSchema nodes with LLM-generated descriptions.

    For each schema lacking a description (or with mismatched enrichment_hash):
    1. Gather schema context (name, options, field count)
    2. Call LLM to generate description and keywords
    3. Update graph with enrichment fields

    Args:
        client: Neo4j GraphClient
        model: LLM model for enrichment (defaults to language model)
        batch_size: Schemas per LLM call
        force: Re-enrich all schemas regardless of hash
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

    # Query schemas needing enrichment
    if force:
        query = """
        MATCH (s:IdentifierSchema)
        RETURN s.id AS id, s.name AS name, s.documentation AS documentation,
               s.options AS options, s.option_count AS option_count,
               s.field_count AS field_count, s.source AS source,
               s.enrichment_hash AS enrichment_hash
        ORDER BY s.id
        """
    else:
        query = """
        MATCH (s:IdentifierSchema)
        WHERE s.description IS NULL OR s.enrichment_hash IS NULL
        RETURN s.id AS id, s.name AS name, s.documentation AS documentation,
               s.options AS options, s.option_count AS option_count,
               s.field_count AS field_count, s.source AS source,
               s.enrichment_hash AS enrichment_hash
        ORDER BY s.id
        """

    all_schemas = list(client.query(query))
    logger.info(f"Found {len(all_schemas)} identifier schemas to enrich")

    if not all_schemas:
        return stats

    # Process in batches
    for batch_idx in range(0, len(all_schemas), batch_size):
        batch = all_schemas[batch_idx : batch_idx + batch_size]

        # Check hashes for cached entries
        to_enrich = []
        for schema in batch:
            ctx_str = f"{schema['id']}:{schema.get('options', '')}:{schema.get('documentation', '')}"
            expected_hash = _compute_enrichment_hash(ctx_str, model)
            if not force and schema.get("enrichment_hash") == expected_hash:
                stats["cached"] += 1
            else:
                schema["_expected_hash"] = expected_hash
                to_enrich.append(schema)

        if not to_enrich:
            continue

        # Build prompt context
        batch_data = []
        for idx, schema in enumerate(to_enrich, 1):
            options = []
            if schema.get("options"):
                try:
                    options = json.loads(schema["options"])
                except (json.JSONDecodeError, TypeError):
                    pass

            batch_data.append(
                {
                    "index": idx,
                    "name": schema["name"],
                    "documentation": schema.get("documentation") or "",
                    "option_count": schema.get("option_count") or len(options),
                    "options": options,
                    "field_count": schema.get("field_count") or 0,
                    "source": schema.get("source") or "",
                }
            )

        # Render system prompt
        system_prompt = render_prompt(
            "imas/identifier_enrichment",
            context={"batch": batch_data},
        )

        # Build user message with full option context
        user_lines = ["Enrich the following IMAS identifier schemas:\n"]
        for entry in batch_data:
            user_lines.append(f"\n### Schema {entry['index']}: `{entry['name']}`")
            if entry["documentation"]:
                user_lines.append(f"- Header: {entry['documentation']}")
            user_lines.append(f"- Option count: {entry['option_count']}")
            if entry["field_count"]:
                user_lines.append(f"- Used by {entry['field_count']} fields in the DD")
            if entry["source"]:
                user_lines.append(f"- Source: {entry['source']}")
            if entry["options"]:
                user_lines.append("- Options:")
                for opt in entry["options"]:
                    idx = opt.get("index", "")
                    name = opt.get("name", "")
                    desc = opt.get("description", "")
                    units = opt.get("units", "")
                    line = f"    - {idx}: {name}"
                    if desc:
                        line += f" — {desc}"
                    if units:
                        line += f" [{units}]"
                    user_lines.append(line)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_lines)},
        ]

        try:
            result, cost, tokens = call_llm_structured(
                model=model,
                messages=messages,
                response_model=IdentifierEnrichmentBatch,
            )
            stats["cost"] += cost
            stats["tokens"] += tokens

            # Build updates
            updates = []
            for enrichment in result.results:
                if enrichment.schema_index < 1 or enrichment.schema_index > len(
                    to_enrich
                ):
                    logger.warning(
                        f"Invalid schema_index {enrichment.schema_index} in result"
                    )
                    continue

                schema = to_enrich[enrichment.schema_index - 1]
                updates.append(
                    {
                        "id": schema["id"],
                        "description": enrichment.description,
                        "keywords": enrichment.keywords[:5],
                        "enrichment_hash": schema["_expected_hash"],
                        "enrichment_source": "llm",
                    }
                )

            # Batch update graph
            if updates:
                client.query(
                    """
                    UNWIND $updates AS u
                    MATCH (s:IdentifierSchema {id: u.id})
                    SET s.description = u.description,
                        s.keywords = u.keywords,
                        s.enrichment_hash = u.enrichment_hash,
                        s.enrichment_source = u.enrichment_source
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
            logger.error(f"Error enriching identifier batch: {e}")

    return stats


# =============================================================================
# Embedding Generation
# =============================================================================


def embed_identifier_schemas(
    client: GraphClient,
    *,
    force_reembed: bool = False,
    on_items: Callable[[list[dict], float], None] | None = None,
) -> dict[str, int]:
    """Generate embeddings for enriched IdentifierSchema nodes.

    Only re-embeds schemas whose content hash has changed.

    Returns:
        Stats dict with updated/cached counts.
    """
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.embeddings.encoder import Encoder
    from imas_codex.settings import get_embedding_dimension, get_embedding_model

    stats = {"updated": 0, "cached": 0}

    # Fetch schemas with enriched descriptions
    results = client.query("""
        MATCH (s:IdentifierSchema)
        WHERE s.description IS NOT NULL AND s.enrichment_source IS NOT NULL
        RETURN s.id AS id, s.name AS name,
               s.description AS description,
               s.keywords AS keywords,
               s.embedding_hash AS existing_hash
        ORDER BY s.id
    """)

    if not results:
        logger.info("No enriched identifier schemas to embed")
        return stats

    dim = get_embedding_dimension()
    model_name = get_embedding_model()

    # Ensure vector index exists
    client.query(f"""
        CREATE VECTOR INDEX identifier_schema_embedding IF NOT EXISTS
        FOR (n:IdentifierSchema) ON n.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine',
                `vector.quantization.enabled`: true
            }}
        }}
    """)

    # Compute text and hashes, filter to schemas needing re-embedding
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
        logger.info(f"All {len(results)} identifier schema embeddings up to date")
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
        MATCH (s:IdentifierSchema {id: b.id})
        SET s.embedding = b.embedding,
            s.embedding_hash = b.embedding_hash
        """,
        batch=batch_data,
    )
    stats["updated"] = len(to_embed)
    if on_items and batch_data:
        on_items(
            [{"primary_text": item["id"]} for item in batch_data],
            0.0,
        )

    logger.info(
        f"Identifier schema embeddings: {stats['updated']} updated, "
        f"{stats['cached']} cached"
    )
    return stats
