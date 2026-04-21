"""Canonical dictionary normalisation for standard name entries.

Ensures that round-trip export→import comparisons are stable by
applying deterministic normalisation rules to entry dictionaries.
Used by both ``export.py`` (pre-write) and ``catalog_import.py``
(pre-diff).

See plan 35 §PR-driven round-trip canonical-dict rules.
"""

from __future__ import annotations

import copy
import re
from typing import Any

#: List-valued fields that should be sorted alphabetically.
#: ``links`` is excluded — link order is editorial.
SORTED_LIST_FIELDS: frozenset[str] = frozenset({"tags", "deprecates", "constraints"})

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
LIST_FIELDS: frozenset[str] = frozenset({"tags", "links", "constraints"})

#: Multiline string fields that get whitespace-normalised.
MULTILINE_FIELDS: frozenset[str] = frozenset({"description", "documentation"})


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
    5. Sorted list fields (``tags``, ``deprecates``, ``constraints``):
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
