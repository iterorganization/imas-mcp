"""
LLM-based cluster labeling using OpenRouter API.

Generates human-readable labels and descriptions for clusters
using Gemini-3-Pro's large context window for batch processing.
"""

import json
import logging
import re
from dataclasses import dataclass

import requests

from imas_mcp.settings import get_language_model

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
        import os

        self.model = model or get_language_model()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        )

        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY environment variable."
            )

    def _make_request(self, messages: list[dict], max_tokens: int = 100000) -> str:
        """Make a chat completion request to the API."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "imas-mcp",
            "X-Title": "IMAS MCP Cluster Labeling",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        response = requests.post(url, headers=headers, json=data, timeout=300)

        if response.status_code != 200:
            raise RuntimeError(
                f"API request failed: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

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
        batch_size: int = 500,
    ) -> list[ClusterLabel]:
        """Generate labels for all clusters.

        Args:
            clusters: List of cluster dictionaries
            batch_size: Number of clusters per API request

        Returns:
            List of ClusterLabel objects
        """
        all_labels = []
        total_clusters = len(clusters)

        logger.info(f"Labeling {total_clusters} clusters using {self.model}")

        # Process in batches (Gemini-3-Pro can handle ~500-1000 clusters per request)
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
                response = self._make_request(messages)
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
