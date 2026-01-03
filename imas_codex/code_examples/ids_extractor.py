"""IMAS IDS extraction transformation for LlamaIndex.

Custom TransformComponent that extracts IMAS IDS references from code chunks
and stores them in node metadata for graph relationship creation.
"""

import functools
import json
import re
from pathlib import Path

from llama_index.core.schema import BaseNode, TransformComponent

# Regex patterns for IMAS IDS detection in code
IDS_PATTERNS = [
    # Python: ids_factory.new("equilibrium")
    r'\.new\(["\'](\w+)["\']\)',
    # Python: factory.equilibrium()
    r"factory\.(\w+)\(\)",
    # String literals that are IDS names
    r'["\'](\w+)["\']',
]


@functools.cache
def get_known_ids() -> frozenset[str]:
    """Get the set of valid IDS names from the bundled catalog.

    Uses the ids_catalog.json bundled with imas-codex to avoid
    importing imas-python (which triggers DD parsing logs).

    Returns:
        Frozen set of lowercase IDS names
    """
    # Use bundled catalog - find latest DD version
    resources_dir = Path(__file__).parent.parent / "resources" / "imas_data_dictionary"

    # Find available DD versions and use the latest
    dd_versions = sorted(
        [d.name for d in resources_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    if not dd_versions:
        # Fallback: return common IDS names
        return frozenset(
            [
                "equilibrium",
                "core_profiles",
                "magnetics",
                "wall",
                "pf_active",
                "core_sources",
                "core_transport",
                "summary",
                "controllers",
            ]
        )

    catalog_path = resources_dir / dd_versions[0] / "schemas" / "ids_catalog.json"

    if not catalog_path.exists():
        return frozenset()

    catalog = json.loads(catalog_path.read_text())
    ids_catalog = catalog.get("ids_catalog", {})

    return frozenset(name.lower() for name in ids_catalog.keys())


class IDSExtractor(TransformComponent):
    """Extract IMAS IDS references from code chunks.

    This LlamaIndex TransformComponent scans code text for patterns
    that indicate IMAS IDS usage. Stores only counts in metadata
    to avoid size limits; full references stored in _related_ids for
    graph linking.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: dict) -> list[BaseNode]:
        """Process nodes and extract IDS references.

        Args:
            nodes: List of LlamaIndex nodes to process

        Returns:
            Nodes with updated metadata containing related_ids_count and line numbers
        """
        for node in nodes:
            content = node.get_content()

            # Extract IDS references
            ids_refs = self._extract_ids(content)
            if ids_refs:
                # Store only count in metadata (avoids size limits)
                node.metadata["related_ids_count"] = len(ids_refs)
                # Store full refs in private metadata for graph linking
                # This is NOT persisted to vector store, only used during ingestion
                node.metadata["_related_ids"] = sorted(ids_refs)

            # Compute absolute line numbers using parent doc text
            full_text = node.metadata.get("_full_doc_text")
            if full_text and node.start_char_idx is not None:
                start_line = full_text[: node.start_char_idx].count("\n") + 1
                end_line = start_line + content.count("\n")
                node.metadata["start_line"] = start_line
                node.metadata["end_line"] = end_line
                # Remove temp field to avoid storing in graph
                del node.metadata["_full_doc_text"]

        return nodes

    def _extract_ids(self, text: str) -> set[str]:
        """Extract IMAS IDS references from text.

        Args:
            text: Code text to scan

        Returns:
            Set of IDS names found in the text
        """
        known_ids = get_known_ids()
        found: set[str] = set()
        for pattern in IDS_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                candidate = match.group(1).lower()
                if candidate in known_ids:
                    found.add(candidate)
        return found


__all__ = ["IDSExtractor", "get_known_ids"]
