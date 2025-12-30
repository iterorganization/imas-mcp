"""IMAS IDS extraction transformation for LlamaIndex.

Custom TransformComponent that extracts IMAS IDS references from code chunks
and stores them in node metadata for graph relationship creation.
"""

import functools
import re

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
    """Get the set of valid IDS names from IMAS Python.

    Uses IDSFactory to get the authoritative list of IDS names
    for the current data dictionary version.

    Returns:
        Frozen set of lowercase IDS names
    """
    from imas import IDSFactory

    factory = IDSFactory()
    return frozenset(name.lower() for name in factory.ids_names())


class IDSExtractor(TransformComponent):
    """Extract IMAS IDS references from code chunks.

    This LlamaIndex TransformComponent scans code text for patterns
    that indicate IMAS IDS usage and adds them to node metadata.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: dict) -> list[BaseNode]:
        """Process nodes and extract IDS references.

        Args:
            nodes: List of LlamaIndex nodes to process

        Returns:
            Nodes with updated metadata containing related_ids
        """
        for node in nodes:
            ids_refs = self._extract_ids(node.get_content())
            if ids_refs:
                node.metadata["related_ids"] = sorted(ids_refs)
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
