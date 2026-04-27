"""Canonical dictionary normalisation for standard name entries.

Ensures that round-trip export→import comparisons are stable by
applying deterministic normalisation rules to entry dictionaries.
Used by both ``export.py`` (pre-write) and ``catalog_import.py``
(pre-diff).

See plan 35 §PR-driven round-trip canonical-dict rules and plan 40
§2 for CANONICAL_KEY_ORDER.
"""

from __future__ import annotations

import copy
import re
from typing import Any

#: List-valued fields that should be sorted alphabetically.
#: ``links`` is excluded — link order is editorial.
SORTED_LIST_FIELDS: frozenset[str] = frozenset({"deprecates", "constraints"})

#: Fields that default to ``None`` when absent.
NULLABLE_SCALAR_FIELDS: frozenset[str] = frozenset(
    {
        "deprecates",
        "superseded_by",
        "validity_domain",
        "cocos_transformation_type",
        "provenance",
    }
)

#: Fields that default to ``[]`` when absent.
LIST_FIELDS: frozenset[str] = frozenset({"links", "constraints"})

#: Multiline string fields that get whitespace-normalised.
MULTILINE_FIELDS: frozenset[str] = frozenset({"description", "documentation"})

#: Exhaustive canonical key order for exported YAML entries.
#: Every allowed top-level key must appear here; unknown keys trigger
#: ``UnknownCatalogKeyError``.  Emission uses this order with
#: ``yaml.safe_dump(sort_keys=False)`` to guarantee byte-stable round-trip.
CANONICAL_KEY_ORDER: tuple[str, ...] = (
    "id",
    "name",
    "kind",
    "status",
    "description",
    "documentation",
    "unit",
    "cocos_transformation_type",
    "cocos",
    "physics_domain",
    "links",
    "validity_domain",
    "constraints",
    "arguments",
    "error_variants",
    "deprecates",
    "superseded_by",
    "provenance",
    "sources",
)

_CANONICAL_KEY_SET: frozenset[str] = frozenset(CANONICAL_KEY_ORDER)


class UnknownCatalogKeyError(ValueError):
    """Raised when an entry contains a key not in CANONICAL_KEY_ORDER."""


def reorder_entry_dict(entry: dict[str, Any]) -> dict[str, Any]:
    """Re-order an entry dict to match CANONICAL_KEY_ORDER.

    Keys present in *entry* are emitted in canonical order.  Keys absent
    from *entry* are omitted (no padding).  Any key not in
    ``CANONICAL_KEY_ORDER`` raises :class:`UnknownCatalogKeyError`.

    Parameters
    ----------
    entry:
        Entry dictionary to reorder.

    Returns
    -------
    New dict with keys in canonical order.

    Raises
    ------
    UnknownCatalogKeyError
        If *entry* contains a key not present in ``CANONICAL_KEY_ORDER``.
    """
    unknown = set(entry.keys()) - _CANONICAL_KEY_SET
    if unknown:
        raise UnknownCatalogKeyError(
            f"Unknown catalog key(s) {sorted(unknown)} not in CANONICAL_KEY_ORDER"
        )
    return {k: entry[k] for k in CANONICAL_KEY_ORDER if k in entry}


def _normalise_string(value: str) -> str:
    """Strip leading/trailing whitespace, collapse ``\\r\\n`` → ``\\n``."""
    if not isinstance(value, str):
        return value
    value = value.replace("\r\n", "\n")
    value = re.sub(r"\n+$", "", value)
    return value.strip()


def canonicalise_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalise a standard name entry dict to canonical form.

    Applies the following transformations (non-mutating — returns a
    new dict):

    1. Missing nullable scalar fields → ``None``.
    2. Missing list fields → ``[]``.
    3. Multiline strings (``description``, ``documentation``):
       strip, collapse ``\\r\\n`` → ``\\n``, strip trailing newlines.
    4. All string values: ``.strip()``.
    5. Sorted list fields (``deprecates``, ``constraints``):
       sorted alphabetically in-place.
    6. Remove keys with ``None`` values for optional fields
       (keeps output YAML clean).

    Parameters
    ----------
    entry:
        Raw entry dictionary (e.g. from graph node properties or
        YAML load).

    Returns
    -------
    Normalised copy of the entry.
    """
    out = copy.deepcopy(entry)

    # 1. Default nullable scalars
    for field in NULLABLE_SCALAR_FIELDS:
        if field not in out:
            out[field] = None

    # 2. Default list fields
    for field in LIST_FIELDS:
        if field not in out:
            out[field] = []

    # 3+4. Normalise strings
    for key, value in out.items():
        if isinstance(value, str):
            if key in MULTILINE_FIELDS:
                out[key] = _normalise_string(value)
            else:
                out[key] = value.strip()

    # 5. Sort unordered list fields
    for field in SORTED_LIST_FIELDS:
        val = out.get(field)
        if isinstance(val, list):
            out[field] = sorted(val)

    return out
