"""IMAS IDS extraction from text content.

Scans text for IMAS IDS references (equilibrium, core_profiles, etc.)
using regex patterns. Works on code, documents, and wiki pages.

Can be used standalone or as a LlamaIndex TransformComponent.
"""

from llama_index.core.schema import BaseNode, TransformComponent

from imas_codex.discovery.base.imas_patterns import (
    extract_ids_names,
    get_all_ids_names,
)

# Regex patterns for IMAS IDS detection
IDS_PATTERNS = [
    # Python: ids_factory.new("equilibrium")
    r'\.new\(["\'](\w+)["\']\)',
    # Python: factory.equilibrium()
    r"factory\.(\w+)\(\)",
    # String literals that are IDS names
    r'["\'](\w+)["\']',
]


def get_known_ids() -> frozenset[str]:
    """Get the set of valid IDS names from the data dictionary.

    Delegates to the shared ``imas_codex.discovery.base.imas_patterns``
    module for the canonical IDS name list.

    Returns:
        Frozen set of lowercase IDS names
    """
    return frozenset(get_all_ids_names())


def extract_ids_references(text: str) -> set[str]:
    """Extract IMAS IDS references from text.

    Delegates to the shared ``imas_codex.discovery.base.imas_patterns``
    module which uses the same IDS name list across the entire pipeline.

    Args:
        text: Text to scan for IDS references

    Returns:
        Set of IDS names found
    """
    return extract_ids_names(text)


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
