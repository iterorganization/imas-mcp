"""IMAS IDS extraction from text content.

Scans text for IMAS IDS references (equilibrium, core_profiles, etc.)
using regex patterns. Works on code, documents, and wiki pages.
"""

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


__all__ = ["extract_ids_references", "get_known_ids"]
