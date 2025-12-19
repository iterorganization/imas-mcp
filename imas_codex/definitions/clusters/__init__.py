"""Cluster definitions for semantic path groupings.

This module provides:
- Cached cluster labels (labels.json)
- Controlled vocabularies for cluster enrichment:
  - concepts.yaml: Physics concepts (Te, ne, q, etc.)
  - data_types.yaml: Data structure types (profile_1d, scalar, etc.)
  - tags.yaml: Classification tags (core, edge, measured, etc.)
  - mapping_relevance.yaml: Mapping usefulness (high, medium, low)
"""

from pathlib import Path

# Cached labels from LLM labeling
LABELS_FILE = Path(__file__).parent / "labels.json"

# Controlled vocabularies (LinkML YAML schemas)
CONCEPTS_SCHEMA = Path(__file__).parent / "concepts.yaml"
DATA_TYPES_SCHEMA = Path(__file__).parent / "data_types.yaml"
TAGS_SCHEMA = Path(__file__).parent / "tags.yaml"
MAPPING_RELEVANCE_SCHEMA = Path(__file__).parent / "mapping_relevance.yaml"

# Proposed terms from LLM suggestions (for human review)
PROPOSED_TERMS_FILE = Path(__file__).parent / "proposed_terms.json"

__all__ = [
    "LABELS_FILE",
    "CONCEPTS_SCHEMA",
    "DATA_TYPES_SCHEMA",
    "TAGS_SCHEMA",
    "MAPPING_RELEVANCE_SCHEMA",
    "PROPOSED_TERMS_FILE",
]
