"""Catalog registry mapping ``fn_id`` to its runner (plan 39 §3, §10).

The catalog is **closed**: only the four entries declared here are
recognised.  Extending the catalog is a deliberate plan-revision step
(see ``README.md``).

Public API:
    - :data:`CATALOG` — ``dict[fn_id_str, CatalogEntry]`` registry.
    - :func:`get_runner` — resolve a parsed :data:`FanoutCall` to its
      runner coroutine.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .runners import (
    run_find_related_dd_paths,
    run_search_dd_clusters,
    run_search_dd_paths,
    run_search_existing_names,
)
from .schemas import (
    FanoutCall,
    FanoutResult,
    _FindRelatedDDPaths,
    _SearchDDClusters,
    _SearchDDPaths,
    _SearchExistingNames,
)

# =====================================================================
# Catalog entry
# =====================================================================


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A single entry in the closed fan-out catalog.

    Attributes:
        fn_id: Human-friendly identifier exposed to the LLM (proposer-
            visible).  Decoupled from the backing symbol so the catalog
            can outlive helper renames.
        variant: Pydantic model class for the discriminated union
            variant.
        runner: Async callable ``(args, *, gc, scope, timeout_s)
            -> FanoutResult``.
    """

    fn_id: str
    variant: type
    runner: Callable[..., Awaitable[FanoutResult]]


CATALOG: dict[str, CatalogEntry] = {
    "search_existing_names": CatalogEntry(
        fn_id="search_existing_names",
        variant=_SearchExistingNames,
        runner=run_search_existing_names,
    ),
    "search_dd_paths": CatalogEntry(
        fn_id="search_dd_paths",
        variant=_SearchDDPaths,
        runner=run_search_dd_paths,
    ),
    "find_related_dd_paths": CatalogEntry(
        fn_id="find_related_dd_paths",
        variant=_FindRelatedDDPaths,
        runner=run_find_related_dd_paths,
    ),
    "search_dd_clusters": CatalogEntry(
        fn_id="search_dd_clusters",
        variant=_SearchDDClusters,
        runner=run_search_dd_clusters,
    ),
}


def get_runner(call: FanoutCall) -> Callable[..., Awaitable[FanoutResult]]:
    """Return the runner coroutine for a parsed catalog call.

    Raises:
        KeyError: ``call.fn_id`` is not in :data:`CATALOG`.  By the
            discriminated-union contract this should be unreachable
            (parse-time validation rejects unknown ``fn_id``s); the
            check is kept as a defensive guard.
    """
    entry = CATALOG.get(call.fn_id)
    if entry is None:
        raise KeyError(f"Unknown fan-out fn_id: {call.fn_id!r}")
    return entry.runner


def normalize_query_or_path(call: FanoutCall) -> str:
    """Return the lowercased + whitespace-collapsed intent string.

    Used by :func:`dispatcher.propose` for query-side dedup (plan 39
    §4.1 S1).
    """
    raw: str
    if hasattr(call, "query"):
        raw = call.query  # type: ignore[union-attr]
    elif hasattr(call, "path"):
        raw = call.path  # type: ignore[union-attr]
    else:  # pragma: no cover — discriminated union exhausts these
        raw = ""
    return " ".join(raw.lower().split())


__all__ = [
    "CATALOG",
    "CatalogEntry",
    "get_runner",
    "normalize_query_or_path",
]


# Type-only re-export for downstream consumers that need ``Any``.
_ = Any
