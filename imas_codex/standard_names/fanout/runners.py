"""Async runners for the fan-out catalog (plan 39 §3.6, §4.2).

Each runner wraps a *sync* helper from
:mod:`imas_codex.graph.dd_search` (and
:mod:`imas_codex.standard_names.search`) with
:func:`asyncio.to_thread` plus a per-call :func:`asyncio.wait_for`,
producing a :class:`FanoutResult`.

Plan-39 contract:

- All four backing helpers are sync (Neo4j blocking I/O + numpy
  embeddings).  A naive ``async def`` would run them on the event loop
  — :func:`asyncio.gather` would not parallelise, and
  :func:`asyncio.wait_for` would not cancel sync code mid-execution.
- :func:`asyncio.to_thread` schedules onto the default thread pool
  (``min(32, cpus + 4)``) shared with existing
  ``asyncio.to_thread(persist_refined_name, …)`` calls.
- The Python-side ``wait_for`` cancels at the gate; the helper itself
  keeps running on the worker thread until completion, but its result
  is discarded.

Lifecycle invariant (plan 39 §10.1):
    Every runner accepts ``gc: GraphClient`` as a keyword argument
    and reuses it.  No runner instantiates its own ``GraphClient``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .schemas import (
    FanoutHit,
    FanoutResult,
    FanoutScope,
    _FindRelatedDDPaths,
    _SearchDDClusters,
    _SearchDDPaths,
    _SearchExistingNames,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Hit normalisers — convert backing-helper output to FanoutHit
# =====================================================================


def _existing_name_hits(rows: list[dict[str, Any]]) -> list[FanoutHit]:
    out = []
    for r in rows or []:
        sn_id = str(r.get("id") or "")
        if not sn_id:
            continue
        desc = r.get("description") or ""
        unit = r.get("unit") or ""
        kind = r.get("kind") or ""
        score = r.get("score")
        label_bits = [sn_id]
        if unit:
            label_bits.append(f"unit={unit}")
        if kind:
            label_bits.append(f"kind={kind}")
        out.append(
            FanoutHit(
                kind="standard_name",
                id=sn_id,
                label="  ".join(label_bits),
                score=float(score) if score is not None else None,
                payload={"description": desc, "unit": unit, "kind": kind},
            )
        )
    return out


def _dd_path_hits(hits: list) -> list[FanoutHit]:
    out = []
    for h in hits or []:
        path = getattr(h, "path", None) or ""
        if not path:
            continue
        ids_name = getattr(h, "ids_name", "") or ""
        score = getattr(h, "score", None)
        units = getattr(h, "units", None) or ""
        bits = [path]
        if ids_name:
            bits.append(f"ids={ids_name}")
        if units:
            bits.append(f"units={units}")
        out.append(
            FanoutHit(
                kind="dd_path",
                id=path,
                label="  ".join(bits),
                score=float(score) if score is not None else None,
                payload={"ids_name": ids_name, "units": units},
            )
        )
    return out


def _related_path_hits(result: Any) -> list[FanoutHit]:
    out = []
    for h in getattr(result, "hits", []) or []:
        path = getattr(h, "path", "") or ""
        if not path:
            continue
        rel = getattr(h, "relationship_type", "") or ""
        via = getattr(h, "via", "") or ""
        bits = [path, f"({rel})"]
        if via:
            bits.append(f"via={via}")
        out.append(
            FanoutHit(
                kind="dd_path",
                id=path,
                label="  ".join(bits),
                score=None,
                payload={"relationship_type": rel, "via": via},
            )
        )
    return out


def _cluster_hits(rows: list) -> list[FanoutHit]:
    out = []
    for r in rows or []:
        cid = getattr(r, "id", "") or ""
        if not cid:
            continue
        label = getattr(r, "label", "") or cid
        desc = getattr(r, "description", "") or ""
        score = getattr(r, "score", None)
        out.append(
            FanoutHit(
                kind="cluster",
                id=cid,
                label=f"{label}: {desc}" if desc else label,
                score=float(score) if score is not None else None,
                payload={"scope": getattr(r, "scope", "")},
            )
        )
    return out


# =====================================================================
# Async wrapper
# =====================================================================


async def _run_with_timeout(
    *,
    fn_id: str,
    args_dict: dict[str, Any],
    sync_callable,
    timeout_s: float,
    normalise,
) -> FanoutResult:
    """Run a sync helper on a worker thread under ``wait_for``.

    Captures both timeouts and arbitrary exceptions as ``ok=False``
    results — runners are expected to never raise (the dispatcher's
    ``executor_partial_fail`` path depends on this).
    """
    t0 = time.monotonic()
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(sync_callable),
            timeout=timeout_s,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return FanoutResult(
            fn_id=fn_id,
            args=args_dict,
            ok=True,
            hits=normalise(raw),
            elapsed_ms=elapsed_ms,
        )
    except TimeoutError:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return FanoutResult(
            fn_id=fn_id,
            args=args_dict,
            ok=False,
            error="timeout",
            elapsed_ms=elapsed_ms,
        )
    except Exception as e:  # pragma: no cover — exact text varies
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        logger.debug("fan-out runner %s raised: %s", fn_id, e, exc_info=True)
        return FanoutResult(
            fn_id=fn_id,
            args=args_dict,
            ok=False,
            error=f"{type(e).__name__}: {e}"[:200],
            elapsed_ms=elapsed_ms,
        )


# =====================================================================
# Runners — one per catalog fn_id
# =====================================================================


async def run_search_existing_names(
    args: _SearchExistingNames,
    *,
    gc: Any,
    scope: FanoutScope,  # noqa: ARG001 — unused; kept for uniform signature
    timeout_s: float,
) -> FanoutResult:
    """Run :func:`search_standard_names_vector` (plan 39 §3.1)."""
    from imas_codex.standard_names.search import search_standard_names_vector

    def _call() -> list[dict[str, Any]]:
        return search_standard_names_vector(
            args.query,
            k=args.k,
            gc=gc,
            include_superseded=False,
        )

    return await _run_with_timeout(
        fn_id=args.fn_id,
        args_dict=args.model_dump(),
        sync_callable=_call,
        timeout_s=timeout_s,
        normalise=_existing_name_hits,
    )


async def run_search_dd_paths(
    args: _SearchDDPaths,
    *,
    gc: Any,
    scope: FanoutScope,
    timeout_s: float,
) -> FanoutResult:
    """Run :func:`hybrid_dd_search` with caller-injected scope."""
    from imas_codex.graph.dd_search import hybrid_dd_search

    def _call():
        return hybrid_dd_search(
            gc,
            args.query,
            ids_filter=scope.ids_filter,
            physics_domain=scope.physics_domain,
            dd_version=scope.dd_version,
            k=args.k,
        )

    return await _run_with_timeout(
        fn_id=args.fn_id,
        args_dict=args.model_dump(),
        sync_callable=_call,
        timeout_s=timeout_s,
        normalise=_dd_path_hits,
    )


async def run_find_related_dd_paths(
    args: _FindRelatedDDPaths,
    *,
    gc: Any,
    scope: FanoutScope,
    timeout_s: float,
) -> FanoutResult:
    """Run :func:`related_dd_search` with caller-injected scope."""
    from imas_codex.graph.dd_search import related_dd_search

    def _call():
        return related_dd_search(
            gc,
            args.path,
            max_results=args.max_results,
            dd_version=scope.dd_version,
        )

    return await _run_with_timeout(
        fn_id=args.fn_id,
        args_dict=args.model_dump(),
        sync_callable=_call,
        timeout_s=timeout_s,
        normalise=_related_path_hits,
    )


async def run_search_dd_clusters(
    args: _SearchDDClusters,
    *,
    gc: Any,
    scope: FanoutScope,
    timeout_s: float,
) -> FanoutResult:
    """Run :func:`cluster_search` with caller-injected scope."""
    from imas_codex.graph.dd_search import cluster_search

    def _call():
        return cluster_search(
            gc,
            args.query,
            k=args.k,
            dd_version=scope.dd_version,
        )

    return await _run_with_timeout(
        fn_id=args.fn_id,
        args_dict=args.model_dump(),
        sync_callable=_call,
        timeout_s=timeout_s,
        normalise=_cluster_hits,
    )


__all__ = [
    "run_search_existing_names",
    "run_search_dd_paths",
    "run_find_related_dd_paths",
    "run_search_dd_clusters",
]
