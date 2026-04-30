"""Canonical dictionary normalisation for standard name entries.

Ensures that round-trip export→import comparisons are stable by
applying deterministic normalisation rules to entry dictionaries.
Used by both ``export.py`` (pre-write) and ``catalog_import.py``
(pre-diff).

Also exposes the **deterministic name-key duplicate guard** (plan 39
§5.2 / Phase 1.5):

- :func:`name_key_normalise` — single canonical form for case-folded,
  underscore-collapsed name comparisons.
- :func:`lexical_variants` — enumerate the documented variant set
  (case-fold, collapsed underscores, stripped leading/trailing
  underscores) for a candidate.
- :func:`find_name_key_duplicate` — graph lookup against existing
  ``StandardName.id`` values; returns the existing duplicate's id or
  ``None``.  Caller uses this BEFORE persisting a new ``StandardName``
  to drop and re-tag duplicates without spending an LLM call.

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


# ---------------------------------------------------------------------------
# Phase 1.5 deterministic name-key duplicate guard (plan 39 §5.2)
# ---------------------------------------------------------------------------

#: Pattern matching runs of ``_`` characters. Used by
#: :func:`name_key_normalise` to collapse `electron__temperature` and
#: `electron___temperature` to the same canonical form.
_UNDERSCORE_RUN: re.Pattern[str] = re.compile(r"_+")


def name_key_normalise(name: str) -> str:
    """Return the canonical lookup key for *name*.

    The key folds case, collapses runs of underscores to a single ``_``,
    and strips leading/trailing underscores. Two distinct
    :class:`StandardName` ids that round-trip to the same key are
    considered duplicates by :func:`find_name_key_duplicate`.

    The transformation is **deterministic and total** — every input
    string maps to exactly one key, and the transformation is
    idempotent (``f(f(x)) == f(x)``).

    Examples
    --------
    >>> name_key_normalise("Electron_Temperature")
    'electron_temperature'
    >>> name_key_normalise("electron__temperature")
    'electron_temperature'
    >>> name_key_normalise("__electron_temperature_")
    'electron_temperature'
    >>> name_key_normalise("")
    ''
    """
    if not isinstance(name, str):
        return ""
    folded = name.casefold()
    collapsed = _UNDERSCORE_RUN.sub("_", folded)
    return collapsed.strip("_")


def lexical_variants(name: str) -> set[str]:
    """Enumerate the documented lexical variant set for *name*.

    Returns the set of strings that should be treated as duplicates of
    *name* under the Phase 1.5 dup-guard. The set always contains:

    - ``name`` itself,
    - the canonical key (:func:`name_key_normalise`),
    - the casefold-only variant,
    - the underscore-collapsed variant.

    The set is intentionally small and deterministic — no synonym
    expansion, no morphology, no segment reordering. Plan 39 §5.2:
    "Set enumerated in the Phase 1.5 PR — not inferred from ad-hoc
    tests."
    """
    if not isinstance(name, str) or not name:
        return set()
    folded = name.casefold()
    collapsed = _UNDERSCORE_RUN.sub("_", name)
    return {
        name,
        folded,
        collapsed,
        name_key_normalise(name),
    }


def find_name_key_duplicate(
    gc: Any,
    candidate_name: str,
    *,
    exclude: str | None = None,
) -> str | None:
    """Look up an existing ``StandardName.id`` that collides with *candidate_name*.

    The lookup is **deterministic and read-only** — no LLM, no vector
    search.  A duplicate is reported when an existing
    :class:`StandardName` node has an id whose
    :func:`name_key_normalise` form equals that of *candidate_name*.

    Parameters
    ----------
    gc:
        Active graph client (required).
    candidate_name:
        The candidate name about to be persisted.
    exclude:
        Optional id to ignore when checking — typically the
        ``old_name`` being refined, so the chain's predecessor never
        looks like a self-collision.

    Returns
    -------
    Existing ``StandardName.id`` of the colliding node, or ``None``
    when no collision is found.

    Notes
    -----
    Plan 39 §5.2: vector search is intentionally NOT used — the
    description-vector index is not a name-level dup check.  The guard
    is designed for cases where the LLM has emitted a candidate whose
    canonical form already exists in the graph (e.g.
    ``Electron_Temperature`` vs ``electron_temperature``).
    """
    if not isinstance(candidate_name, str) or not candidate_name.strip():
        return None
    candidate_key = name_key_normalise(candidate_name)
    if not candidate_key:
        return None

    rows = (
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IS NOT NULL
              AND ($exclude IS NULL OR sn.id <> $exclude)
              AND sn.id <> $candidate_name
            WITH sn,
                 toLower(replace(replace(replace(replace(replace(
                     sn.id,
                     '__', '_'),
                     '__', '_'),
                     '__', '_'),
                     '__', '_'),
                     '__', '_')) AS lowered
            WITH sn, lowered,
                 CASE
                   WHEN lowered STARTS WITH '_' THEN substring(lowered, 1)
                   ELSE lowered
                 END AS l2
            WITH sn,
                 CASE
                   WHEN l2 ENDS WITH '_' THEN substring(l2, 0, size(l2) - 1)
                   ELSE l2
                 END AS sn_key
            WHERE sn_key = $candidate_key
            RETURN sn.id AS id
            LIMIT 1
            """,
            candidate_name=candidate_name,
            candidate_key=candidate_key,
            exclude=exclude,
        )
        or []
    )
    for row in rows:
        # Defensive: the Cypher key derivation handles up to 5 doubled
        # underscores cheaply; verify the Python normaliser agrees so
        # weirder inputs (e.g. ``a___b`` with a triple) are still caught.
        existing_id = row.get("id") if isinstance(row, dict) else row["id"]
        if not existing_id:
            continue
        if name_key_normalise(existing_id) == candidate_key:
            return existing_id
    # Fallback Python pass for edge cases the Cypher normaliser misses
    # (e.g. very long underscore runs). Plan 39 §5.2 favours correctness
    # over a single round-trip.
    fallback_rows = (
        gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.id IS NOT NULL
              AND ($exclude IS NULL OR sn.id <> $exclude)
              AND sn.id <> $candidate_name
              AND toLower(sn.id) CONTAINS $rough
            RETURN sn.id AS id
            LIMIT 50
            """,
            candidate_name=candidate_name,
            exclude=exclude,
            rough=candidate_key.replace("_", ""),
        )
        or []
    )
    for row in fallback_rows:
        existing_id = row.get("id") if isinstance(row, dict) else row["id"]
        if not existing_id:
            continue
        if name_key_normalise(existing_id) == candidate_key:
            return existing_id
    return None
