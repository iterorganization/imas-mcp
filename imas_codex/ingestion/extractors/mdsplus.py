"""MDSplus path extraction from text content.

Extracts MDSplus tree paths and TDI function calls from code/text.
Works on code, documents, and wiki pages.

Can be used standalone or as a LlamaIndex TransformComponent.
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
    # Python f-string suffixes: f"{tree}:PSI" extracts just the suffix
    r"\{[^}]+\}:([A-Za-z_][A-Za-z0-9_]+)",
    # Connection.get patterns: conn.get("\\path")
    r'\.get\(["\']\\\\?([^"\']+)["\']\)',
    # MDSplus TdiExecute or TdiCompile
    r'Tdi(?:Execute|Compile)\(["\']([^"\']+)["\']\)',
    # Python .tdi() method: conn.tdi(r'\results::psi'), tcv.shot(n).tdi(...)
    r'\.tdi\s*\(\s*r?["\']\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)["\']',
    # MATLAB tdi() calls: tdi('\results::thomson:te')
    r"tdi\s*\(\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
    # MATLAB mdsvalue() calls: mdsvalue('\results::ip')
    r"mdsvalue\s*\(\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
    # MATLAB mdsvalue with concatenation: mdsvalue(['\results::ece_lfs:channel_00' int2str(i)])
    r"mdsvalue\s*\(\s*\[\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)",
    # Fortran MDS_OPEN/MDS_GET/MDS_VALUE: call MDS_GET('\results::ip')
    r"MDS_(?:OPEN|GET|VALUE)\s*\(\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
    # IDL mdsvalue: mdsvalue, '\results::ip'
    r"mdsvalue\s*,\s*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
    # Python tree.getNode: self._MDSTree.getNode(self._root+'::psi')
    r"getNode\s*\(\s*[^)]*['\"]\\\\?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*::[A-Za-z0-9_:\.]+)['\"]",
]

# TDI function call patterns (tcv_eq, tcv_get, etc.)
TDI_FUNCTION_PATTERNS = [
    r'tcv_eq\(["\']([A-Z_][A-Z0-9_]+)["\']',
    r'tcv_get\(["\']([A-Z_][A-Z0-9_]+)["\']',
    r'tcv_psitbx\(["\']([A-Z_][A-Z0-9_]+)["\']',
    r'tcv_\w+\(["\']([A-Z_][A-Z0-9_]+)["\']',
    # MATLAB tdi() with TDI function: tdi('tcv_ip()'), tdi('tcv_eq("PSI")')
    r"tdi\s*\(\s*['\"](\w+)\s*\(",
    # mdsvalue with TDI function: mdsvalue('tcv_eq("PSI")')
    r"mdsvalue\s*\(\s*['\"](\w+)\s*\(",
    # Python .tdi() with TDI function: conn.tdi('tcv_ip()')
    r"\.tdi\s*\(\s*['\"](\w+)\s*\(",
]

# MDSplus path pattern for plain text (wiki pages, docs)
MDSPLUS_TEXT_PATTERN = re.compile(
    r"\\\\?[A-Z_][A-Z_0-9]*::[A-Z_][A-Z_0-9:]*",
    re.IGNORECASE,
)


def normalize_mdsplus_path(path: str) -> str:
    """Normalize an MDSplus path to canonical form for graph storage.

    Canonical form: uppercase, single backslash prefix, consistent :: separator.

    Args:
        path: Raw MDSplus path

    Returns:
        Normalized path like \\RESULTS::I_P
    """
    path = path.lstrip("\\")
    path = path.upper()
    path = path.rstrip(":.")
    return f"\\{path}"


def compute_canonical_path(path: str) -> str:
    """Compute canonical path for matching (alias for normalize_mdsplus_path)."""
    return normalize_mdsplus_path(path)


def extract_mdsplus_paths(text: str) -> list[MDSplusReference]:
    """Extract MDSplus paths from code text using code-aware patterns.

    Uses regex patterns tuned for source code (quoted strings, API calls).

    Args:
        text: Code text to scan

    Returns:
        List of MDSplusReference objects with normalized paths
    """
    found: list[MDSplusReference] = []
    seen: set[str] = set()

    for pattern in MDSPLUS_PATH_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            raw = match.group(1)
            if "::" in raw:
                normalized = normalize_mdsplus_path(raw)
                ref_type = "mdsplus_path"
            else:
                normalized = f"\\RESULTS::{raw.upper()}"
                ref_type = "mdsplus_path"
            if normalized not in seen:
                seen.add(normalized)
                found.append(MDSplusReference(normalized, raw, ref_type))

    for pattern in TDI_FUNCTION_PATTERNS:
        for match in re.finditer(pattern, text):
            quantity = match.group(1).upper()
            pseudo_path = f"\\RESULTS::{quantity}"
            if pseudo_path not in seen:
                seen.add(pseudo_path)
                found.append(MDSplusReference(pseudo_path, match.group(0), "tdi_call"))

    return found


def extract_mdsplus_paths_text(text: str) -> list[str]:
    """Extract MDSplus paths from plain text (documents, wiki pages).

    Uses a simpler regex pattern suited for prose rather than code.

    Args:
        text: Plain text to scan

    Returns:
        Deduplicated list of normalized MDSplus paths
    """
    matches = MDSPLUS_TEXT_PATTERN.findall(text)
    normalized = set()
    for m in matches:
        path = m.lstrip("\\")
        path = "\\" + path.upper()
        normalized.add(path)
    return sorted(normalized)


class MDSplusExtractor(TransformComponent):
    """Extract MDSplus paths from LlamaIndex nodes.

    Scans code/text for MDSplus path references and TDI function calls.
    Stores counts in metadata; full paths in mdsplus_paths for graph linking.
    """

    def __call__(self, nodes: list[BaseNode], **kwargs: dict) -> list[BaseNode]:
        """Process nodes and extract MDSplus references."""
        for node in nodes:
            content = node.get_content()
            refs = extract_mdsplus_paths(content)

            if refs:
                node.metadata["mdsplus_ref_count"] = len(refs)
                # Store as flat string list — NamedTuples cause
                # "Collections containing collections" in Neo4j
                node.metadata["mdsplus_paths"] = [r.path for r in refs]

        return nodes


__all__ = [
    "MDSplusExtractor",
    "MDSplusReference",
    "compute_canonical_path",
    "extract_mdsplus_paths",
    "extract_mdsplus_paths_text",
    "normalize_mdsplus_path",
]
