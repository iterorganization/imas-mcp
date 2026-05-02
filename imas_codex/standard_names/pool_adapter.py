"""Adapter to route --paths / --single-pass / --source signals through the pool path.

Replaces the linear 5-stage DAG (:mod:`pipeline.py`) with a thin adapter
that:

1. Creates ``StandardNameSource`` nodes for explicit paths (idempotent MERGE).
2. Builds batch items matching the shape expected by :func:`compose_batch`.
3. Calls :func:`compose_batch` directly for a one-shot compose pass.

The user-facing CLI surface (``--paths``, ``--single-pass``, ``--source``)
is unchanged.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_explicit_paths(
    state: Any,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Pool-routed replacement for ``run_sn_pipeline``.

    Accepts a :class:`StandardNameBuildState` and processes its
    ``paths_list`` (or the full DD source scope) through the pool
    compose path.

    Steps:

    1. **Seed** — creates ``StandardNameSource`` nodes for each path
       via :func:`merge_standard_name_sources` (idempotent MERGE).
    2. **Compose** — calls :func:`compose_batch` with the batch items
       enriched from the graph.
    3. **Stats** — writes summary stats back to ``state.stats``.

    Args:
        state: Populated ``StandardNameBuildState``.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.workers import compose_batch

    paths = state.paths_list or []
    if not paths and not state.source:
        logger.info("No paths or source specified — nothing to do")
        return

    if state.dry_run:
        state.stats["compose_skipped"] = True
        logger.info("Dry run — skipping composition")
        return

    if stop_event is None:
        stop_event = asyncio.Event()

    # ── Step 1: Seed StandardNameSource nodes ──────────────────────────
    # For explicit paths, create SNS nodes so the compose step has items.
    # For source-based (--source dd without --paths), run extract to
    # discover paths first.
    if paths:
        batch_items = await _seed_explicit_paths(paths, force=state.force)
    else:
        # Full DD extract — creates SNS nodes from graph scan
        batch_items = await _seed_from_source(state)

    if not batch_items:
        logger.info("No items to compose after seeding")
        state.stats["extract_count"] = 0
        return

    state.stats["extract_count"] = len(batch_items)
    logger.info("Pool adapter: seeded %d items for composition", len(batch_items))

    # ── Step 2: Compose via pool compose_batch ─────────────────────────
    mgr = state.budget_manager or BudgetManager(state.cost_limit)

    try:
        composed = await compose_batch(
            batch_items,
            mgr,
            stop_event,
            regen=state.regen,
        )
    except Exception:
        logger.exception("Pool adapter: compose_batch failed")
        composed = 0

    state.stats["generate_name_count"] = composed
    state.stats["compose_cost"] = mgr.total_spent
    logger.info(
        "Pool adapter: composed %d candidates (cost=$%.4f)",
        composed,
        mgr.total_spent,
    )


async def _seed_explicit_paths(
    paths: list[str],
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Create StandardNameSource nodes and return batch items for compose.

    Each path becomes one item in the batch with the same dict shape
    that :func:`claim_generate_name_batch` would return.
    """
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.graph_ops import merge_standard_name_sources

    # Build SNS dicts for merge
    sources = []
    for path in paths:
        ids_name = path.split("/")[0] if "/" in path else ""
        sources.append(
            {
                "id": f"dd:{path}",
                "source_type": "dd",
                "source_id": path,
                "dd_path": path,
                "batch_key": ids_name,
                "status": "extracted",
                "description": "",
            }
        )

    # MERGE into graph
    def _merge():
        return merge_standard_name_sources(sources, force=force)

    await asyncio.to_thread(_merge)

    # Read back enriched items from graph
    def _read_items() -> list[dict]:
        with GraphClient() as gc:
            rows = gc.query(
                """
                UNWIND $paths AS p
                MATCH (n:IMASNode {id: p})
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN n.id AS path,
                       n.description AS description,
                       n.physics_domain AS physics_domain,
                       n.data_type AS data_type,
                       coalesce(u.id, n.unit) AS unit
                """,
                paths=paths,
            )
            return [dict(r) for r in (rows or [])]

    items = await asyncio.to_thread(_read_items)

    # Enrich items with DD version / COCOS from current DDVersion
    def _get_dd_meta() -> dict:
        with GraphClient() as gc:
            row = next(
                iter(
                    gc.query("""
                        MATCH (dv:DDVersion {is_current: true})
                        RETURN dv.id AS dd_version, dv.cocos AS cocos_version
                    """)
                ),
                None,
            )
            return dict(row) if row else {}

    dd_meta = await asyncio.to_thread(_get_dd_meta)
    for item in items:
        item["dd_version"] = dd_meta.get("dd_version")
        item["cocos_version"] = dd_meta.get("cocos_version")

    return items


async def _seed_from_source(state: Any) -> list[dict[str, Any]]:
    """Seed items from DD source scan (non-targeted mode).

    Used when ``--single-pass`` is set without ``--paths``.
    Runs the extract logic to discover DD paths, creates SNS nodes,
    and returns enriched batch items.
    """
    from imas_codex.standard_names.graph_ops import (
        get_existing_standard_names,
        get_named_source_ids,
        merge_standard_name_sources,
    )
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    def _extract():
        existing = get_existing_standard_names()

        named_ids: set[str] = set()
        if not state.force:
            named_ids = get_named_source_ids()

        from imas_codex.standard_names.batching import get_generate_batch_config

        batch_cfg = get_generate_batch_config()
        batches = extract_dd_candidates(
            ids_filter=getattr(state, "ids_filter", None),
            domain_filter=state.domain_filter,
            limit=state.limit,
            existing_names=existing,
            force=state.force,
            name_only=getattr(state, "name_only", False),
            name_only_batch_size=getattr(state, "name_only_batch_size", 50),
            max_batch_size=batch_cfg["batch_size"],
            max_tokens=batch_cfg["max_tokens"],
        )

        if named_ids and not state.force:
            for batch in batches:
                batch.items = [
                    item for item in batch.items if item.get("path") not in named_ids
                ]
            batches = [b for b in batches if b.items]

        # Flatten all batch items into a single list for pool compose
        all_items = []
        for batch in batches:
            all_items.extend(batch.items)

        # Create SNS nodes for all extracted items
        sources = []
        for item in all_items:
            path = item.get("path", "")
            if not path:
                continue
            ids_name = path.split("/")[0] if "/" in path else ""
            sources.append(
                {
                    "id": f"dd:{path}",
                    "source_type": "dd",
                    "source_id": path,
                    "dd_path": path,
                    "batch_key": ids_name,
                    "status": "extracted",
                    "description": item.get("description", ""),
                    "physics_domain": item.get("physics_domain"),
                }
            )

        if sources:
            merge_standard_name_sources(sources, force=state.force)

        return all_items

    return await asyncio.to_thread(_extract)
