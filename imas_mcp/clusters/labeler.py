"""
LLM-based cluster labeling using OpenRouter API.

Generates human-readable labels and descriptions for clusters
using Gemini-3-Pro's large context window for batch processing.
"""

import json
import logging
import os
import re
from dataclasses import dataclass

from imas_mcp.embeddings.openrouter_client import OpenRouterClient
from imas_mcp.settings import get_labeling_batch_size, get_language_model

logger = logging.getLogger(__name__)


@dataclass
class ClusterLabel:
    """Label and description for a cluster."""

    cluster_id: int
    label: str
    description: str


class ClusterLabeler:
    """Generates labels and descriptions for clusters using LLM."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize the labeler.

        Args:
            model: LLM model to use (default from settings)
            api_key: OpenRouter API key (uses OPENAI_API_KEY env var if not provided)
            base_url: API base URL (uses OPENAI_BASE_URL env var if not provided)
        """
        self.model = model or get_language_model()
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
                    "ids": cluster.get("ids", cluster.get("ids_names", [])),
                    "paths": cluster.get("paths", [])[:20],  # Limit paths for context
                    "path_count": len(cluster.get("paths", [])),
                }
            )

        return f"""You are an expert in fusion plasma physics and the IMAS (Integrated Modelling & Analysis Suite) data dictionary.

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

            labels.append(
                ClusterLabel(
                    cluster_id=cluster_id,
                    label=item.get("label", f"Cluster {cluster_id}"),
                    description=item.get("description", ""),
                )
            )

        return labels

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

        return ClusterLabel(
            cluster_id=cluster_id,
            label=label,
            description=description,
        )

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
