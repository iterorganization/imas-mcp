"""Central encode/parse/split/merge utilities for StandardName source paths.

StandardName.source_paths stores a union list of URI-prefixed source entity IDs:
  - DD paths:     ``dd:equilibrium/time_slice/profiles_1d/psi``
  - Facility IDs: ``tcv:ip/measured``

The ``dd:`` prefix is the canonical form. Legacy bare paths (no colon)
are treated as DD for backward compatibility but should be normalized.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DD_PREFIX = "dd:"
"""URI prefix for Data Dictionary paths in source_paths."""

KNOWN_FACILITY_NAMESPACES: frozenset[str] = frozenset(
    {
        "tcv",
        "jet",
        "west",
        "iter",
        "aug",
        "mast",
        "east",
        "kstar",
        "d3d",
        "nstx",
    }
)
"""Known facility namespaces for signal source paths."""


# ---------------------------------------------------------------------------
# SourceURI dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceURI:
    """Parsed representation of a source_paths entry."""

    source_type: str  # "dd" or facility namespace
    source_id: str  # bare path (no prefix)

    @property
    def uri(self) -> str:
        """Canonical URI form: ``dd:path`` or ``facility:signal``."""
        return f"{self.source_type}:{self.source_id}"


# ---------------------------------------------------------------------------
# Core encode/parse functions
# ---------------------------------------------------------------------------


def encode_source_path(source_type: str, source_id: str) -> str:
    """Encode a source ID for storage in source_paths.

    DD paths get ``dd:`` prefix; facility signals already have their prefix.

    >>> encode_source_path("dd", "equilibrium/time_slice/profiles_1d/psi")
    'dd:equilibrium/time_slice/profiles_1d/psi'
    >>> encode_source_path("signals", "tcv:ip/measured")
    'tcv:ip/measured'
    """
    if source_type == "dd":
        return f"{DD_PREFIX}{source_id}"
    return source_id  # signals already have facility: prefix


def encode_dd_source(bare_path: str) -> str:
    """Shorthand: add ``dd:`` prefix to a bare DD path.

    >>> encode_dd_source("equilibrium/time_slice/profiles_1d/psi")
    'dd:equilibrium/time_slice/profiles_1d/psi'
    """
    if bare_path.startswith(DD_PREFIX):
        return bare_path
    return f"{DD_PREFIX}{bare_path}"


def parse_source_path(path: str) -> tuple[str, str]:
    """Parse source_paths entry → (source_type, source_id).

    Recognises ``dd:`` prefix and facility namespaces.
    Legacy bare paths (no colon) are treated as DD.

    >>> parse_source_path("dd:equilibrium/time_slice/profiles_1d/psi")
    ('dd', 'equilibrium/time_slice/profiles_1d/psi')
    >>> parse_source_path("tcv:ip/measured")
    ('signals', 'tcv:ip/measured')
    >>> parse_source_path("equilibrium/time_slice/profiles_1d/psi")
    ('dd', 'equilibrium/time_slice/profiles_1d/psi')
    """
    if path.startswith(DD_PREFIX):
        return ("dd", path[len(DD_PREFIX) :])
    if ":" in path:
        return ("signals", path)
    # Legacy bare path → DD
    return ("dd", path)


def parse_source_uri(path: str) -> SourceURI:
    """Parse source_paths entry into a ``SourceURI`` object.

    >>> parse_source_uri("dd:equilibrium/time_slice/profiles_1d/psi")
    SourceURI(source_type='dd', source_id='equilibrium/time_slice/profiles_1d/psi')
    """
    if path.startswith(DD_PREFIX):
        return SourceURI(source_type="dd", source_id=path[len(DD_PREFIX) :])
    if ":" in path:
        ns, _, rest = path.partition(":")
        return SourceURI(source_type=ns, source_id=rest)
    # Legacy bare path → DD
    return SourceURI(source_type="dd", source_id=path)


# ---------------------------------------------------------------------------
# Normalization / stripping
# ---------------------------------------------------------------------------


def normalize_source_path(path: str) -> str:
    """Ensure a source path has the correct URI prefix.

    Adds ``dd:`` to bare DD paths. Already-prefixed paths pass through.

    >>> normalize_source_path("equilibrium/time_slice/profiles_1d/psi")
    'dd:equilibrium/time_slice/profiles_1d/psi'
    >>> normalize_source_path("dd:equilibrium/time_slice/profiles_1d/psi")
    'dd:equilibrium/time_slice/profiles_1d/psi'
    >>> normalize_source_path("tcv:ip/measured")
    'tcv:ip/measured'
    """
    if path.startswith(DD_PREFIX) or ":" in path:
        return path
    return f"{DD_PREFIX}{path}"


def strip_dd_prefix(path: str) -> str:
    """Remove ``dd:`` prefix, returning bare DD path.

    Safe on bare paths (returns unchanged) and non-DD paths (returns unchanged).

    >>> strip_dd_prefix("dd:equilibrium/time_slice/profiles_1d/psi")
    'equilibrium/time_slice/profiles_1d/psi'
    >>> strip_dd_prefix("equilibrium/time_slice/profiles_1d/psi")
    'equilibrium/time_slice/profiles_1d/psi'
    """
    if path.startswith(DD_PREFIX):
        return path[len(DD_PREFIX) :]
    return path


# ---------------------------------------------------------------------------
# IDS filtering helpers
# ---------------------------------------------------------------------------


def ids_prefix_for_source_paths(ids_name: str) -> str:
    """Return the ``STARTS WITH`` prefix for filtering source_paths by IDS.

    >>> ids_prefix_for_source_paths("equilibrium")
    'dd:equilibrium/'
    """
    return f"{DD_PREFIX}{ids_name}/"


# ---------------------------------------------------------------------------
# Split / merge
# ---------------------------------------------------------------------------


def split_source_paths(paths: list[str]) -> dict[str, list[str]]:
    """Split source_paths into typed subsets: ``{'dd': [...], 'signals': [...]}``.

    DD entries are returned as bare paths (prefix stripped).
    """
    result: dict[str, list[str]] = {}
    for path in paths:
        source_type, source_id = parse_source_path(path)
        result.setdefault(source_type, []).append(source_id)
    return result


def merge_source_paths(existing: list[str], new: list[str]) -> list[str]:
    """Deduplicated union of source paths. Deterministic sort for tests/export."""
    return sorted(set(existing) | set(new))


def merge_source_types(existing: list[str], new: list[str]) -> list[str]:
    """Deduplicated union of source types. Sorted for deterministic output."""
    return sorted(set(existing) | set(new))
