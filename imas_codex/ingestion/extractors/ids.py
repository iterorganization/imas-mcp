"""IMAS IDS extraction from text content.

Scans text for IMAS IDS references (equilibrium, core_profiles, etc.)
using regex patterns. Works on code, documents, and wiki pages.

Can be used standalone or as a LlamaIndex TransformComponent.
"""

import functools
import json
import re
from pathlib import Path

from llama_index.core.schema import BaseNode, TransformComponent

# Regex patterns for IMAS IDS detection
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
    resources_dir = (
        Path(__file__).parent.parent.parent / "resources" / "imas_data_dictionary"
    )

    dd_versions = sorted(
        [d.name for d in resources_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    if not dd_versions:
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


def extract_ids_references(text: str) -> set[str]:
    """Extract IMAS IDS references from text.

    Standalone function usable without LlamaIndex dependency.

    Args:
        text: Text to scan for IDS references

    Returns:
        Set of IDS names found
    """
    known_ids = get_known_ids()
    found: set[str] = set()
    for pattern in IDS_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            candidate = match.group(1).lower()
            if candidate in known_ids:
                found.add(candidate)
    return found


class IDSExtractor(TransformComponent):
    """Extract IMAS IDS references from LlamaIndex nodes.

    Scans code/text for patterns indicating IMAS IDS usage.
    Stores counts in metadata; full references in _related_ids for graph linking.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: dict) -> list[BaseNode]:
        """Process nodes and extract IDS references."""
        for node in nodes:
            content = node.get_content()

            ids_refs = extract_ids_references(content)
            if ids_refs:
                node.metadata["related_ids_count"] = len(ids_refs)
                node.metadata["_related_ids"] = sorted(ids_refs)

            # Compute absolute line numbers using parent doc text
            full_text = node.metadata.get("_full_doc_text")
            if full_text and node.start_char_idx is not None:
                start_line = full_text[: node.start_char_idx].count("\n") + 1
                end_line = start_line + content.count("\n")
                node.metadata["start_line"] = start_line
                node.metadata["end_line"] = end_line
                del node.metadata["_full_doc_text"]

        return nodes


__all__ = ["IDSExtractor", "extract_ids_references", "get_known_ids"]
