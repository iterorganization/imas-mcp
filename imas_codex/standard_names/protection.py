"""Pipeline protection for catalog-owned editorial fields.

Prevents the codex LLM pipeline from overwriting editorial content
that was manually curated via a catalog PR (origin=catalog_edit).
All writers of protected fields call ``filter_protected()`` before
persisting to the graph.

See plan 35 §Pipeline protection enforcement.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Fields that are catalog-authoritative when origin=catalog_edit.
#: Pipeline writers must not overwrite these without override=True.
PROTECTED_FIELDS: frozenset[str] = frozenset(
    {
        "description",
        "documentation",
        "kind",
        "tags",
        "links",
        "status",
        "deprecates",
        "superseded_by",
        "validity_domain",
        "constraints",
    }
)


def filter_protected(
    items: list[dict[str, Any]],
    *,
    override: bool = False,
    protected_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Strip protected editorial fields from catalog-edited items.

    Parameters
    ----------
    items:
        Dicts to filter. Each must have an ``"id"`` key (the standard name).
    override:
        When ``True``, bypass protection — all fields pass through.
    protected_names:
        Pre-fetched set of standard name IDs whose ``origin`` is
        ``'catalog_edit'``. If ``None``, queries the graph to determine
        protection status. Callers in hot loops should pre-fetch.

    Returns
    -------
    tuple of (filtered_items, skipped_names):
        - ``filtered_items``: new list with protected fields stripped from
          catalog-edited items. Non-protected fields pass through. Items
          without ``origin`` or with ``origin='pipeline'`` pass unchanged.
        - ``skipped_names``: list of item IDs that had fields stripped.

    Notes
    -----
    Does not mutate the input list or its dicts.
    """
    if override:
        return items, []

    if protected_names is None:
        protected_names = _fetch_catalog_edit_names(
            [it["id"] for it in items if "id" in it]
        )

    filtered: list[dict[str, Any]] = []
    skipped: list[str] = []

    for item in items:
        name_id = item.get("id", "")
        if name_id in protected_names:
            stripped = {k: v for k, v in item.items() if k not in PROTECTED_FIELDS}
            if len(stripped) < len(item):
                skipped.append(name_id)
                logger.warning(
                    "Stripped %d protected field(s) from catalog-edited name '%s'",
                    len(item) - len(stripped),
                    name_id,
                )
            filtered.append(stripped)
        else:
            # Shallow copy to avoid mutating caller's dict
            filtered.append(copy.copy(item))

    return filtered, skipped


def _fetch_catalog_edit_names(name_ids: list[str]) -> set[str]:
    """Query graph for names with origin='catalog_edit'."""
    if not name_ids:
        return set()
    try:
        from imas_codex.graph.client import GraphClient

        with GraphClient() as gc:
            rows = gc.query(
                """
                UNWIND $names AS name
                MATCH (sn:StandardName {id: name})
                WHERE sn.origin = 'catalog_edit'
                RETURN sn.id AS id
                """,
                names=name_ids,
            )
            return {r["id"] for r in (rows or [])}
    except Exception:
        logger.warning(
            "Failed to query catalog_edit names — treating all as pipeline",
            exc_info=True,
        )
        return set()
