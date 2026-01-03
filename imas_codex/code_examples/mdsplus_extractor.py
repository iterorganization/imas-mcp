"""MDSplus path extraction transformation for LlamaIndex.

Custom TransformComponent that extracts MDSplus tree paths and TDI function
calls from code chunks, enabling linking to TreeNode metadata.
"""

import re
from typing import NamedTuple

from llama_index.core.schema import BaseNode, TransformComponent


class MDSplusReference(NamedTuple):
    """An extracted MDSplus reference from code."""

    path: str  # Normalized path like \\RESULTS::I_P
    raw: str  # Original match from code
    ref_type: str  # "path" | "tdi_call" | "tdi_quantity"


# Regex patterns for MDSplus path detection
MDSPLUS_PATH_PATTERNS = [
    # Direct path strings: "\\RESULTS::I_P" or '\\results::thomson.profiles.auto:te'
    r'["\']\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)["\']',
    # Python f-string suffixes: f"{tree}:PSI" extracts just the suffix (PSI)
    # Handles both f"{eq_tree}:TIME_PSI" and f"{eq_tree}:I_PL"
    r"\{[^}]+\}:([A-Za-z_][A-Za-z0-9_]+)",
    # Connection.get patterns: conn.get("\\path")
    r'\.get\(["\']\\\\?([^"\']+)["\']\)',
    # MDSplus TdiExecute or TdiCompile
    r'Tdi(?:Execute|Compile)\(["\']([^"\']+)["\']\)',
]

# TDI function call patterns (tcv_eq, tcv_get, etc.)
TDI_FUNCTION_PATTERNS = [
    # tcv_eq("I_P") or tcv_eq('PSI', 'LIUQE')
    r'tcv_eq\(["\']([A-Z_][A-Z0-9_]+)["\']',
    # tcv_get("IP") or similar
    r'tcv_get\(["\']([A-Z_][A-Z0-9_]+)["\']',
    # tcv_psitbx("AREA")
    r'tcv_psitbx\(["\']([A-Z_][A-Z0-9_]+)["\']',
    # Generic TCV TDI: tcv_*(signal_name)
    r'tcv_\w+\(["\']([A-Z_][A-Z0-9_]+)["\']',
]


def normalize_mdsplus_path(path: str) -> str:
    """Normalize an MDSplus path to canonical form.

    - Uppercase
    - Single backslash prefix
    - Consistent :: separator

    Args:
        path: Raw MDSplus path

    Returns:
        Normalized path like \\RESULTS::I_P
    """
    # Remove leading backslashes, then add single
    path = path.lstrip("\\")
    # Uppercase
    path = path.upper()
    # Ensure single backslash prefix
    return f"\\{path}"


def extract_mdsplus_paths(text: str) -> list[MDSplusReference]:
    """Extract MDSplus paths from code text.

    Args:
        text: Code text to scan

    Returns:
        List of MDSplusReference objects with normalized paths
    """
    found: list[MDSplusReference] = []
    seen: set[str] = set()

    # Extract direct paths and f-string suffixes
    for pattern in MDSPLUS_PATH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1)
            # Check if this is a full path (has ::) or just a suffix
            if "::" in raw:
                normalized = normalize_mdsplus_path(raw)
                ref_type = "mdsplus_path"
            else:
                # F-string suffix like "TIME_PSI" from f"{eq_tree}:TIME_PSI"
                # Create a partial path - will match via suffix
                normalized = f"\\RESULTS::{raw.upper()}"
                ref_type = "mdsplus_path"
            if normalized not in seen:
                seen.add(normalized)
                found.append(MDSplusReference(normalized, raw, ref_type))

    # Extract TDI function calls
    for pattern in TDI_FUNCTION_PATTERNS:
        for match in re.finditer(pattern, text):
            quantity = match.group(1).upper()
            # Create a pseudo-path for the quantity
            pseudo_path = f"\\RESULTS::{quantity}"
            if pseudo_path not in seen:
                seen.add(pseudo_path)
                found.append(MDSplusReference(pseudo_path, match.group(0), "tdi_call"))

    return found


class MDSplusExtractor(TransformComponent):
    """Extract MDSplus paths from code chunks.

    This LlamaIndex TransformComponent scans code text for MDSplus
    path references and TDI function calls. Stores only counts in metadata
    to avoid size limits; full references are stored in _mdsplus_refs for
    graph linking.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: dict) -> list[BaseNode]:
        """Process nodes and extract MDSplus references.

        Args:
            nodes: List of LlamaIndex nodes to process

        Returns:
            Nodes with updated metadata containing ref_count and _mdsplus_refs
        """
        for node in nodes:
            content = node.get_content()
            refs = extract_mdsplus_paths(content)

            if refs:
                # Store only count in metadata (avoids size limits)
                node.metadata["mdsplus_ref_count"] = len(refs)
                # Store full refs in private metadata for graph linking
                # This is NOT persisted to vector store, only used during ingestion
                node.metadata["_mdsplus_refs"] = refs

        return nodes


__all__ = [
    "MDSplusExtractor",
    "MDSplusReference",
    "extract_mdsplus_paths",
    "normalize_mdsplus_path",
]
