"""IMAS path normalization utilities.

Shared between DD build (ingest) and tool queries (search) to ensure
paths are consistently normalized on both sides.
"""

import re

# Matches IMAS DD index annotations: (itime), (i1), (:), (1), etc.
_PAREN_INDEX_RE = re.compile(r"\([^)]*\)")
# Matches bracket index annotations: [1], [:], [0:3], etc.
_BRACKET_INDEX_RE = re.compile(r"\[[^\]]*\]")


def strip_path_annotations(path: str) -> str:
    """Strip index/array annotations from an IMAS path.

    Removes both DD-style parenthesized notation — ``(itime)``, ``(i1)``,
    ``(:)`` — and bracket notation — ``[1]``, ``[:]`` — that users may
    copy from code or documentation.

    Examples::

        >>> strip_path_annotations("flux_loop(i1)/flux/data(:)")
        'flux_loop/flux/data'
        >>> strip_path_annotations("time_slice[1]/profiles_1d[:]/psi")
        'time_slice/profiles_1d/psi'
        >>> strip_path_annotations("equilibrium/time_slice/profiles_1d/psi")
        'equilibrium/time_slice/profiles_1d/psi'
    """
    path = _PAREN_INDEX_RE.sub("", path)
    path = _BRACKET_INDEX_RE.sub("", path)
    return path


def _looks_like_path(text: str) -> bool:
    """Return True if *text* looks like an IMAS path, not natural language.

    IMAS paths are space-free, composed of lowercase ``[a-z0-9_]`` segments
    separated by dots or slashes.  Natural language contains spaces,
    mixed-case tokens like ``eV``, or purely numeric tokens like version
    strings ``3.39.0``.
    """
    if " " in text:
        return False
    # Must have at least one separator
    if "." not in text and "/" not in text:
        return False
    # All segments must be lowercase identifiers (IMAS never uses uppercase)
    segments = re.split(r"[./]", text)
    if not all(re.fullmatch(r"[a-z0-9_]*", seg) for seg in segments):
        return False
    # At least one segment must contain a letter (reject "3.39.0")
    return any(re.search(r"[a-z]", seg) for seg in segments)


def normalize_imas_path(path: str) -> str:
    """Normalize an IMAS path: dot→slash conversion, annotation stripping.

    Handles all common user input formats:
    - Dot notation: ``equilibrium.time_slice.profiles_1d.psi``
    - Mixed: ``equilibrium.time_slice/profiles_1d``
    - Index annotations: ``time_slice(i1)/flux[:]/data``
    - Dot-notation with annotations: ``time_slice(itime).profiles_1d.psi``
    - Clean paths pass through unchanged

    Natural language queries (containing spaces, punctuation like ``e.g.``,
    or sentence-ending dots) are returned stripped but otherwise unchanged
    — dot→slash replacement is only applied to path-like inputs.

    Examples::

        >>> normalize_imas_path("equilibrium.time_slice.profiles_1d.psi")
        'equilibrium/time_slice/profiles_1d/psi'
        >>> normalize_imas_path("equilibrium.time_slice/profiles_1d")
        'equilibrium/time_slice/profiles_1d'
        >>> normalize_imas_path("flux_loop(i1)/flux/data(:)")
        'flux_loop/flux/data'
        >>> normalize_imas_path("  equilibrium/time_slice  ")
        'equilibrium/time_slice'
        >>> normalize_imas_path("electron temperature e.g. in eV")
        'electron temperature e.g. in eV'
        >>> normalize_imas_path("Find B0.")
        'Find B0.'
    """
    path = path.strip()
    # Strip annotations first so _looks_like_path sees clean segments
    path = strip_path_annotations(path)
    # Collapse any double slashes from annotation removal
    while "//" in path:
        path = path.replace("//", "/")
    # Only convert dots when input looks like an IMAS path
    if "." in path and _looks_like_path(path):
        path = path.replace(".", "/")
    return path.strip("/")
