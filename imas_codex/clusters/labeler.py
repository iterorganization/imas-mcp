"""
LLM-based cluster labeling using OpenRouter API.

Generates human-readable labels, descriptions, and enrichment metadata
for clusters using controlled vocabularies from LinkML schemas.
"""

import json
import logging
import os
import re
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
from imas_codex.embeddings.openrouter_client import OpenRouterClient
from imas_codex.settings import get_labeling_batch_size, get_language_model

logger = logging.getLogger(__name__)


@dataclass
class ClusterLabel:
    """Label, description, and enrichment metadata for a cluster."""

    cluster_id: int | str
    label: str
    description: str

    # Enrichment fields (NEW)
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

    Uses controlled vocabularies from LinkML schemas for:
    - physics_concepts: Normalized physics quantities
    - data_type: Data structure classification
    - tags: Classification tags
    - mapping_relevance: Usefulness for data mapping
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        enable_enrichment: bool = True,
    ):
        """Initialize the labeler.

        Args:
            model: LLM model to use (default from settings)
            api_key: OpenRouter API key (uses OPENAI_API_KEY env var if not provided)
            base_url: API base URL (uses OPENAI_BASE_URL env var if not provided)
            enable_enrichment: Whether to extract enrichment fields
        """
        self.model = model or get_language_model()
        self.enable_enrichment = enable_enrichment
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        )

        if not api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable."
            )

        # Use shared OpenRouterClient for retry logic and rate limit handling
        self._client = OpenRouterClient(
            model_name=self.model,
            api_key=api_key,
            base_url=base_url,
        )

        # Load controlled vocabularies
        self._concepts = _load_vocabulary(CONCEPTS_SCHEMA, "PhysicsConcept")
        self._data_types = _load_vocabulary(DATA_TYPES_SCHEMA, "DataTypeHint")
        self._tags = _load_vocabulary(TAGS_SCHEMA, "ClusterTag")
        self._relevance_levels = _load_vocabulary(
            MAPPING_RELEVANCE_SCHEMA, "MappingRelevance"
        )

        # Track proposed new terms for human review
        self._proposed_terms: list[dict] = []

    def _build_prompt(self, clusters: list[dict]) -> str:
        """Build the labeling prompt for a batch of clusters."""
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
                    "paths": cluster.get("paths", [])[:20],  # Limit paths for context
                    "path_count": len(cluster.get("paths", [])),
                }
            )

        if self.enable_enrichment:
            return self._build_enriched_prompt(cluster_data)
        return self._build_simple_prompt(cluster_data)

    def _build_simple_prompt(self, cluster_data: list[dict]) -> str:
        """Build a simple labeling-only prompt."""
        return f"""You are an expert in fusion plasma physics and the IMAS data dictionary.

Generate a concise label and description for each cluster of related IMAS data paths.

INSTRUCTIONS:
1. Label: 3-6 words in Title Case, capturing the physics concept
2. Description: 1-2 sentences explaining what physics quantities are grouped and why
3. Labels must be unique across all clusters
4. Focus on the physics meaning, not the data structure

CLUSTERS TO LABEL:
{json.dumps(cluster_data, indent=2)}

RESPOND WITH VALID JSON ONLY - no markdown, no explanation:
[
  {{"id": 0, "label": "Example Physics Label", "description": "Brief physics explanation."}},
  ...
]"""

    def _build_enriched_prompt(self, cluster_data: list[dict]) -> str:
        """Build an enriched prompt with controlled vocabularies."""
        # Format vocabularies for the prompt
        concepts_str = ", ".join(self._concepts[:50])  # Limit for context
        if len(self._concepts) > 50:
            concepts_str += f", ... ({len(self._concepts)} total)"

        data_types_str = ", ".join(self._data_types)
        tags_str = ", ".join(self._tags)

        return f"""You are an expert in fusion plasma physics and the IMAS data dictionary.

Classify and label each cluster of related IMAS data paths.

CONTROLLED VOCABULARIES (use ONLY these values where specified):

PHYSICS CONCEPTS (select 1-3 that best describe the cluster):
{concepts_str}

DATA TYPES (select exactly 1):
{data_types_str}

TAGS (select 1-5 applicable tags):
{tags_str}

MAPPING RELEVANCE (how useful for experimental data mapping):
- high: Core physics quantities commonly measured (Te, ne, Ip, q-profile, etc.)
- medium: Secondary/derived quantities, diagnostic-specific data
- low: Metadata, indices, rarely-populated fields

INSTRUCTIONS:
1. label: 3-6 words in Title Case, capturing the physics concept
2. description: 1-2 sentences explaining the physics grouping
3. physics_concepts: Select 1-3 concepts from the vocabulary above
4. data_type: Select exactly 1 from the vocabulary above
5. tags: Select 1-5 applicable tags from the vocabulary above
6. mapping_relevance: "high", "medium", or "low"
7. suggested_concepts: If a key concept is missing from the vocabulary, suggest it here (optional)

CLUSTERS TO CLASSIFY:
{json.dumps(cluster_data, indent=2)}

RESPOND WITH VALID JSON ONLY - no markdown, no explanation:
[
  {{
    "id": 0,
    "label": "Example Physics Label",
    "description": "Brief physics explanation.",
    "physics_concepts": ["electron_temperature", "ion_temperature"],
    "data_type": "profile_1d",
    "tags": ["core", "measured"],
    "mapping_relevance": "high",
    "suggested_concepts": []
  }},
  ...
]"""

    def label_clusters(
        self,
        clusters: list[dict],
        batch_size: int | None = None,
    ) -> list[ClusterLabel]:
        """Generate labels for all clusters.

        Args:
            clusters: List of cluster dictionaries
            batch_size: Number of clusters per API request (default from settings)

        Returns:
            List of ClusterLabel objects
        """
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

            prompt = self._build_prompt(batch)
            messages = [{"role": "user", "content": prompt}]

            try:
                response = self._client.make_chat_request(
                    messages, model=self.model, max_tokens=100000
                )
                batch_labels = self._parse_response(response, batch)
                all_labels.extend(batch_labels)
                logger.info(
                    f"Batch {batch_num} complete: {len(batch_labels)} labels generated"
                )
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                # Generate fallback labels for failed batch
                for cluster in batch:
                    all_labels.append(self._generate_fallback_label(cluster))

        # Ensure unique labels
        all_labels = self._deduplicate_labels(all_labels)

        logger.info(f"Labeling complete: {len(all_labels)} labels generated")
        return all_labels

    def _parse_response(
        self, response: str, clusters: list[dict]
    ) -> list[ClusterLabel]:
        """Parse the LLM response into ClusterLabel objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\[[\s\S]*\]", response)
        if not json_match:
            raise ValueError(f"No JSON array found in response: {response[:200]}")

        try:
            labels_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        # Create lookup for cluster IDs in this batch
        batch_ids = {c["id"] for c in clusters}

        labels = []
        for item in labels_data:
            cluster_id = item.get("id")
            if cluster_id not in batch_ids:
                continue

            # Validate and filter enrichment fields against vocabularies
            physics_concepts = self._validate_concepts(item.get("physics_concepts", []))
            data_type = self._validate_data_type(item.get("data_type"))
            tags = self._validate_tags(item.get("tags", []))
            mapping_relevance = self._validate_relevance(
                item.get("mapping_relevance", "medium")
            )

            # Track suggested concepts for human review
            suggested = item.get("suggested_concepts", [])
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
                    label=item.get("label", f"Cluster {cluster_id}"),
                    description=item.get("description", ""),
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

    def _generate_fallback_label(self, cluster: dict) -> ClusterLabel:
        """Generate a fallback label when LLM fails."""
        cluster_id = cluster["id"]
        ids = cluster.get("ids", cluster.get("ids_names", []))
        paths = cluster.get("paths", [])

        # Extract common terms from paths
        if paths:
            last_segments = [p.split("/")[-1] for p in paths[:5]]
            common = "_".join(last_segments[:2])
            label = f"{common.replace('_', ' ').title()}"
        else:
            label = f"Cluster {cluster_id}"

        is_cross = cluster.get("is_cross_ids", cluster.get("type") == "cross_ids")
        type_desc = "cross-IDS" if is_cross else "intra-IDS"
        description = (
            f"A {type_desc} cluster containing paths from {', '.join(ids[:3])}."
        )

        # Infer basic enrichment from path patterns
        data_type = self._infer_data_type_from_paths(paths)
        tags = self._infer_tags_from_paths(paths)

        return ClusterLabel(
            cluster_id=cluster_id,
            label=label,
            description=description,
            physics_concepts=[],
            data_type=data_type,
            tags=tags,
            mapping_relevance="medium",
        )

    def _infer_data_type_from_paths(self, paths: list[str]) -> str | None:
        """Infer data type from path patterns."""
        if not paths:
            return None

        path_str = " ".join(paths).lower()

        if "profiles_1d" in path_str or "profile" in path_str:
            return "profile_1d"
        if "profiles_2d" in path_str:
            return "profile_2d"
        if "time" in path_str and "slice" not in path_str:
            return "timeseries"
        if "geometry" in path_str or "boundary" in path_str or "outline" in path_str:
            return "geometry"
        if "coefficient" in path_str or "diffusion" in path_str:
            return "coefficient"
        if "fit" in path_str or "reconstructed" in path_str:
            return "fitting"
        if any(x in path_str for x in ["identifier", "name", "index", "description"]):
            return "metadata"
        if "global" in path_str or "total" in path_str or "average" in path_str:
            return "scalar"

        return None

    def _infer_tags_from_paths(self, paths: list[str]) -> list[str]:
        """Infer tags from path patterns."""
        if not paths:
            return []

        path_str = " ".join(paths).lower()
        tags = []

        # Region tags
        if "core" in path_str:
            tags.append("core")
        if "edge" in path_str:
            tags.append("edge")
        if "separatrix" in path_str:
            tags.append("separatrix")
        if "divertor" in path_str:
            tags.append("divertor")
        if "wall" in path_str:
            tags.append("wall")

        # Species tags
        if "electron" in path_str:
            tags.append("electron")
        if "/ion" in path_str or "ion/" in path_str:
            tags.append("ion")
        if "neutral" in path_str:
            tags.append("neutral")

        # Provenance tags
        if "measured" in path_str:
            tags.append("measured")
        if "fit" in path_str:
            tags.append("fitted")
        if "reconstructed" in path_str:
            tags.append("reconstructed")

        # GGD
        if "/ggd" in path_str or "ggd/" in path_str:
            tags.append("ggd")

        return tags[:5]  # Limit to 5 tags

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
