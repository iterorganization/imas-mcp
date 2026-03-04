"""MDSplus path normalization utilities."""

import re


def normalize_mdsplus_path(path: str) -> str:
    """Normalize an MDSplus path to canonical form.

    - Single backslash prefix
    - Uppercase tree and node names
    - Consistent :: separator

    Args:
        path: Raw MDSplus path

    Returns:
        Normalized path like \\RESULTS::TOP.NODE
    """
    # Remove all leading backslashes
    path = path.lstrip("\\")
    # Uppercase the entire path
    path = path.upper()
    # Ensure single backslash prefix
    return f"\\{path}"


def compute_canonical_path(path: str) -> str:
    """Compute canonical path for deduplication and fuzzy matching.

    Builds on normalize_mdsplus_path but additionally:
    - Strips channel indices (CHANNEL_006 -> CHANNEL, _001 -> removed)
    - Strips trailing numeric suffixes for variant matching

    Args:
        path: Raw or normalized MDSplus path

    Returns:
        Canonical path suitable for matching across sources
    """
    # First normalize
    path = normalize_mdsplus_path(path)

    # Strip channel indices and numeric suffixes for fuzzy matching
    # Pattern: _NNN at end (2-3 digits) or CHANNEL_NNN
    path = re.sub(r"[_:]CHANNEL_?\d{2,3}$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"_\d{2,3}$", "", path)

    return path
