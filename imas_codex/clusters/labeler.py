"""LLM-based cluster labeling.

Generates human-readable labels, descriptions, and enrichment metadata
for clusters using controlled vocabularies from LinkML schemas.
Routes through the canonical ``call_llm`` infrastructure so that LiteLLM
proxy routing, retry logic, and cost tracking are shared with all other
LLM consumers.

Prompts are rendered via the shared Jinja2 template system in
``imas_codex/agentic/prompts/clusters/labeler.md``, with schema context
injected from LinkML-defined controlled vocabularies.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from linkml_runtime.utils.schemaview import SchemaView

from imas_codex.definitions.clusters import (
    CONCEPTS_SCHEMA,
    DATA_TYPES_SCHEMA,
    MAPPING_RELEVANCE_SCHEMA,
    PROPOSED_TERMS_FILE,
    TAGS_SCHEMA,
)
from imas_codex.settings import get_labeling_batch_size, get_model

logger = logging.getLogger(__name__)


@dataclass
class ClusterLabel:
    """Label, description, and enrichment metadata for a cluster."""

    cluster_id: int | str
    label: str
    description: str

    # Enrichment fields
    physics_concepts: list[str] = field(default_factory=list)
    data_type: str | None = None
    tags: list[str] = field(default_factory=list)
    mapping_relevance: str = "medium"

    # Proposed new terms (for human review)
    suggested_concepts: list[str] = field(default_factory=list)


def _load_vocabulary(schema_path: Path, enum_name: str) -> list[str]:
    """Load permissible values from a LinkML enum schema."""
    if not schema_path.exists():
        logger.warning(f"Vocabulary schema not found: {schema_path}")
        return []

    try:
        sv = SchemaView(str(schema_path))
        enum_def = sv.get_enum(enum_name)
        if not enum_def or not enum_def.permissible_values:
            return []
        return list(enum_def.permissible_values.keys())
    except Exception as e:
        logger.warning(f"Failed to load vocabulary from {schema_path}: {e}")
        return []


class ClusterLabeler:
    """Generates labels, descriptions, and enrichment metadata for clusters.

    Uses ``call_llm`` from ``discovery.base.llm`` for all LLM calls,
    inheriting LiteLLM proxy routing, retry logic, and cost tracking.

    Controlled vocabularies are injected into the prompt via Jinja2 templates
    from ``imas_codex/agentic/prompts/clusters/labeler.md``.  Validation of
    LLM responses still uses the vocabularies loaded here.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        """Initialize the labeler.

        Args:
            model: LLM model to use (default from settings)
        """
        self.model = model or get_model("language")

        # Load controlled vocabularies for response validation
        self._concepts = _load_vocabulary(CONCEPTS_SCHEMA, "PhysicsConcept")
        self._data_types = _load_vocabulary(DATA_TYPES_SCHEMA, "DataTypeHint")
        self._tags = _load_vocabulary(TAGS_SCHEMA, "ClusterTag")
        self._relevance_levels = _load_vocabulary(
            MAPPING_RELEVANCE_SCHEMA, "MappingRelevance"
        )

        # Track proposed new terms for human review
        self._proposed_terms: list[dict] = []

    def _build_prompt(self, clusters: list[dict]) -> tuple[str, str]:
        """Build system and user prompts using the Jinja2 template system.

        Returns (system_prompt, user_prompt) matching the canonical pattern
        used by other LLM consumers (scorer, signal enricher, etc.).
        """
        from imas_codex.agentic.prompt_loader import render_prompt

        cluster_data = []
        for cluster in clusters:
            cluster_data.append(
                {
                    "id": cluster["id"],
                    "type": cluster.get(
                        "type",
                        "cross_ids" if cluster.get("is_cross_ids") else "intra_ids",
                    ),
                    "scope": cluster.get("scope", "global"),
                    "scope_detail": cluster.get("scope_detail"),
                    "ids": cluster.get("ids", cluster.get("ids_names", [])),
                    "paths": cluster.get("paths", [])[:20],
                    "path_count": len(cluster.get("paths", [])),
                }
            )

        system_prompt = render_prompt("clusters/labeler")
        user_prompt = json.dumps(cluster_data, indent=2)
        return system_prompt, user_prompt

    def label_clusters(
        self,
        clusters: list[dict],
        batch_size: int | None = None,
    ) -> list[ClusterLabel]:
        """Generate labels for all clusters via LLM.

        Args:
            clusters: List of cluster dictionaries
            batch_size: Number of clusters per API request (default from settings)

        Returns:
            List of ClusterLabel objects

        Raises:
            Exception: If LLM calls fail (no fallback — labels require LLM).
        """
        from imas_codex.clusters.models import ClusterLabelBatch
        from imas_codex.discovery.base.llm import call_llm_structured

        if batch_size is None:
            batch_size = get_labeling_batch_size()

        all_labels = []
        total_clusters = len(clusters)

        logger.info(f"Labeling {total_clusters} clusters using {self.model}")

        for i in range(0, total_clusters, batch_size):
            batch = clusters[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_clusters + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} clusters)"
            )

            system_prompt, user_prompt = self._build_prompt(batch)
            parsed_batch, _cost, _tokens = call_llm_structured(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_model=ClusterLabelBatch,
                max_tokens=65000,
                temperature=0.3,
            )

            batch_labels = self._validate_batch(parsed_batch, batch)
            all_labels.extend(batch_labels)
            logger.info(
                f"Batch {batch_num} complete: {len(batch_labels)} labels generated"
            )

        # Ensure unique labels
        all_labels = self._deduplicate_labels(all_labels)

        logger.info(f"Labeling complete: {len(all_labels)} labels generated")
        return all_labels

    def _validate_batch(
        self, parsed_batch: object, clusters: list[dict]
    ) -> list[ClusterLabel]:
        """Validate parsed LLM results against controlled vocabularies.

        Converts a ``ClusterLabelBatch`` (Pydantic model from structured
        output) into ``ClusterLabel`` dataclass instances, filtering
        enrichment fields against the loaded vocabularies.
        """
        batch_ids = {c["id"] for c in clusters}

        labels = []
        for item in parsed_batch.results:
            cluster_id = item.id
            if cluster_id not in batch_ids:
                continue

            physics_concepts = self._validate_concepts(item.physics_concepts)
            data_type = self._validate_data_type(item.data_type or None)
            tags = self._validate_tags(item.tags)
            mapping_relevance = self._validate_relevance(item.mapping_relevance)

            suggested = item.suggested_concepts
            if suggested:
                self._proposed_terms.append(
                    {
                        "cluster_id": cluster_id,
                        "type": "concept",
                        "terms": suggested,
                    }
                )

            labels.append(
                ClusterLabel(
                    cluster_id=cluster_id,
                    label=item.label or f"Cluster {cluster_id}",
                    description=item.description or "",
                    physics_concepts=physics_concepts,
                    data_type=data_type,
                    tags=tags,
                    mapping_relevance=mapping_relevance,
                    suggested_concepts=suggested,
                )
            )

        return labels

    def _validate_concepts(self, concepts: list) -> list[str]:
        """Validate physics concepts against vocabulary."""
        if not concepts or not self._concepts:
            return []
        return [c for c in concepts if c in self._concepts]

    def _validate_data_type(self, data_type: str | None) -> str | None:
        """Validate data type against vocabulary."""
        if not data_type or not self._data_types:
            return None
        return data_type if data_type in self._data_types else None

    def _validate_tags(self, tags: list) -> list[str]:
        """Validate tags against vocabulary."""
        if not tags or not self._tags:
            return []
        return [t for t in tags if t in self._tags]

    def _validate_relevance(self, relevance: str) -> str:
        """Validate mapping relevance."""
        valid = ["high", "medium", "low"]
        return relevance if relevance in valid else "medium"

    def save_proposed_terms(self) -> int:
        """Save proposed terms to file for human review.

        Returns:
            Number of proposed terms saved
        """
        if not self._proposed_terms:
            return 0

        # Load existing proposals
        existing = []
        if PROPOSED_TERMS_FILE.exists():
            try:
                with open(PROPOSED_TERMS_FILE, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Merge new proposals
        existing.extend(self._proposed_terms)

        # Save
        PROPOSED_TERMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(PROPOSED_TERMS_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        count = len(self._proposed_terms)
        self._proposed_terms = []
        logger.info(f"Saved {count} proposed terms to {PROPOSED_TERMS_FILE}")
        return count

    def _deduplicate_labels(self, labels: list[ClusterLabel]) -> list[ClusterLabel]:
        """Ensure all labels are unique by appending suffixes."""
        seen = {}
        deduplicated = []

        for label in labels:
            base_label = label.label
            if base_label in seen:
                seen[base_label] += 1
                new_label = f"{base_label} {seen[base_label]}"
                deduplicated.append(
                    ClusterLabel(
                        cluster_id=label.cluster_id,
                        label=new_label,
                        description=label.description,
                    )
                )
            else:
                seen[base_label] = 1
                deduplicated.append(label)

        return deduplicated
