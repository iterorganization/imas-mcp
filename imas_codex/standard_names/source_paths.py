"""Central encode/parse/split/merge utilities for StandardName source paths.

StandardName.source_paths stores a union list of all source entity IDs.
DD paths are bare (no colon): "equilibrium/time_slice/profiles_1d/psi"
Signal IDs are facility-prefixed: "tcv:ip/measured"
"""

from __future__ import annotations


def encode_source_path(source_type: str, source_id: str) -> str:
    """Encode a source ID for storage in source_paths.

    DD paths stored bare; all others namespaced with colon.
    """
    if source_type == "dd":
        return source_id
    return f"{source_id}"  # signals already have facility: prefix


def parse_source_path(path: str) -> tuple[str, str]:
    """Parse source_paths entry → (source_type, source_id).

    DD paths never contain colons. All non-DD sources use namespace: prefix.
    """
    if ":" in path:
        return ("signals", path)
    return ("dd", path)


def split_source_paths(paths: list[str]) -> dict[str, list[str]]:
    """Split source_paths into typed subsets: {'dd': [...], 'signals': [...]}."""
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
