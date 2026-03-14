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
