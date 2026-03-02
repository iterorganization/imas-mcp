"""Async workers for parallel static tree discovery.

Workers that process static MDSplus trees through the pipeline:
- extract_worker: Claim TreeModelVersion, SSH extract, ingest to graph
- units_worker: Batched unit extraction for NUMERIC/SIGNAL nodes
- enrich_worker: LLM batch description of tree nodes

Workers coordinate through graph_ops claim/mark functions using claimed_at timestamps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from .state import StaticDiscoveryState

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# ============================================================================
# Extract Worker
# ============================================================================


async def extract_worker(
    state: StaticDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Extract worker: claim a TreeModelVersion, SSH extract, ingest to graph.

    Claims TreeModelVersion nodes with status=discovered, runs SSH extraction
    for that version, then immediately ingests the results into the graph.
    Each version is claimed-extracted-ingested as a unit.
    """
    from imas_codex.mdsplus.static import (
        async_discover_static_tree_version,
        ingest_static_tree,
        merge_version_results,
    )

    from .graph_ops import (
        claim_version_for_extraction,
        mark_version_extracted,
        release_version_claim,
    )

    ssh_retry_count = 0
    max_ssh_retries = 5

    while not state.should_stop():
        # Claim a pending version atomically
        claimed = await asyncio.to_thread(
            claim_version_for_extraction,
            state.facility,
            state.tree_name,
        )

        if not claimed:
            state.extract_phase.record_idle()
            if state.extract_phase.done:
                break
            if on_progress:
                on_progress("idle", state.extract_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.extract_phase.record_activity(1)
        version = claimed["version"]
        version_id = claimed["id"]

        if on_progress:
            on_progress(
                f"extracting v{version} {state.facility}:{state.tree_name}",
                state.extract_stats,
                [{"version": version, "phase": "extract"}],
            )

        try:
            # SSH extraction
            data = await async_discover_static_tree_version(
                facility=state.facility,
                tree_name=state.tree_name,
                version=version,
                timeout=state.timeout,
            )
            ssh_retry_count = 0  # Reset on success

            ver_data = data.get("versions", {}).get(str(version), {})
            if "error" in ver_data:
                logger.warning(
                    "Extraction returned error for v%d: %s",
                    version,
                    ver_data["error"][:100],
                )
                await asyncio.to_thread(release_version_claim, version_id)
                state.extract_stats.errors += 1
                if on_progress:
                    on_progress(
                        f"v{version} error: {ver_data['error'][:60]}",
                        state.extract_stats,
                        None,
                    )
                continue

            node_count = ver_data.get("node_count", 0)
            tags = len(ver_data.get("tags", {}))

            # Ingest immediately
            merged = merge_version_results([data])
            from imas_codex.graph import GraphClient

            with GraphClient() as client:
                stats = ingest_static_tree(client, state.facility, merged)

            # Mark version as extracted in graph
            await asyncio.to_thread(mark_version_extracted, version_id, node_count)

            state.extract_stats.processed += 1
            state.extract_stats.record_batch(1)

            if on_progress:
                on_progress(
                    f"v{version} done — {node_count:,} nodes, {tags} tags",
                    state.extract_stats,
                    [
                        {
                            "version": version,
                            "node_count": node_count,
                            "tags": tags,
                            "nodes_created": stats.get("nodes_created", 0),
                        }
                    ],
                )

        except Exception as e:
            ssh_retry_count += 1
            logger.warning(
                "Extract v%d failed (%d/%d): %s",
                version,
                ssh_retry_count,
                max_ssh_retries,
                e,
            )
            state.extract_stats.errors += 1
            await asyncio.to_thread(release_version_claim, version_id)

            if ssh_retry_count >= max_ssh_retries:
                logger.error(
                    "SSH failed after %d attempts — extract worker stopping",
                    max_ssh_retries,
                )
                state.extract_phase.mark_done()
                if on_progress:
                    on_progress(
                        f"SSH failed: {str(e)[:100]}",
                        state.extract_stats,
                        None,
                    )
                break

            backoff = min(2**ssh_retry_count, 32)
            if on_progress:
                on_progress(
                    f"SSH retry {ssh_retry_count} in {backoff}s",
                    state.extract_stats,
                    None,
                )
            await asyncio.sleep(backoff)
            continue

        await asyncio.sleep(0.1)


# ============================================================================
# Units Worker
# ============================================================================


async def units_worker(
    state: StaticDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Units worker: extract units for the latest static tree version.

    Runs batched unit extraction via SSH for NUMERIC/SIGNAL nodes,
    then merges units into TreeNode nodes in the graph.
    """
    from imas_codex.mdsplus.static import async_extract_units_for_version

    # Wait for at least one version to be extracted
    while not state.should_stop():
        if state.extract_phase.done:
            break
        if state.extract_stats.processed > 0:
            break
        if on_progress:
            on_progress("awaiting extract", state.units_stats, None)
        await asyncio.sleep(1.0)

    if state.should_stop():
        return

    # Find the latest version from config
    ver_configs = state.tree_config.get("versions", [])
    if ver_configs:
        latest_version = max(v["version"] for v in ver_configs)
    else:
        latest_version = 1

    if on_progress:
        on_progress(
            f"v{latest_version} — batched extraction",
            state.units_stats,
            None,
        )

    units_batch_size = 500

    def _on_ssh_progress(checked: int, total: int, found: int) -> None:
        state.units_stats.processed = checked
        state.units_stats.record_batch()
        if on_progress:
            batch_num = (checked // units_batch_size) + 1
            total_batches = (total + units_batch_size - 1) // units_batch_size
            on_progress(
                f"{checked:,}/{total:,} checked, {found} with units",
                state.units_stats,
                [
                    {
                        "checked": checked,
                        "total": total,
                        "found": found,
                        "batch": f"{batch_num}/{total_batches}",
                    }
                ],
            )

    try:
        units = await async_extract_units_for_version(
            state.facility,
            state.tree_name,
            latest_version,
            timeout=state.timeout,
            batch_size=units_batch_size,
            on_progress=_on_ssh_progress,
        )

        if units:
            # Merge units into TreeNode nodes in graph
            from imas_codex.graph import GraphClient
            from imas_codex.mdsplus.ingestion import normalize_mdsplus_path

            updates = []
            for path, unit_str in units.items():
                normalized = normalize_mdsplus_path(path)
                node_id = f"{state.facility}:{state.tree_name}:{normalized}"
                updates.append({"id": node_id, "units": unit_str})

            if updates:
                with GraphClient() as gc:
                    gc.query(
                        """
                        UNWIND $updates AS u
                        MATCH (n:TreeNode {id: u.id})
                        SET n.units = u.units
                        """,
                        updates=updates,
                    )

            if on_progress:
                on_progress(
                    f"{len(units)} paths with units — complete",
                    state.units_stats,
                    [{"found": len(units), "status": "complete"}],
                )
        else:
            if on_progress:
                on_progress("no units found", state.units_stats, None)

    except Exception as e:
        logger.exception("Units extraction failed")
        state.units_stats.errors += 1
        if on_progress:
            on_progress(
                f"failed: {str(e)[:80]}",
                state.units_stats,
                None,
            )

    # Units worker runs once then exits
    state.units_phase.mark_done()


# ============================================================================
# Enrich Worker
# ============================================================================


async def enrich_worker(
    state: StaticDiscoveryState,
    on_progress: Callable | None = None,
    **_kwargs,
) -> None:
    """Enrich worker: LLM batch descriptions of tree nodes.

    After extraction completes, detects indexed parameter patterns and
    enriches them first (one LLM call per pattern covers hundreds of
    nodes). Then enriches remaining non-pattern nodes individually.
    """
    from imas_codex.discovery.base.llm import call_llm_structured
    from imas_codex.mdsplus.enrichment import (
        StaticNodeBatch,
        _build_system_prompt,
        _build_user_prompt,
    )
    from imas_codex.settings import get_model

    from .graph_ops import (
        claim_parent_for_enrichment,
        claim_patterns_for_enrichment,
        detect_and_create_patterns,
        fetch_enrichment_context,
        mark_parent_children_enriched,
        mark_patterns_enriched,
        release_parent_claim,
        release_pattern_claims,
    )

    if not state.enrich:
        state.enrich_phase.mark_done()
        return

    # Wait for ALL extraction to complete so sibling/parent context is
    # available in the graph before we start enriching value nodes.
    while not state.should_stop():
        if state.extract_phase.done:
            break
        if on_progress:
            on_progress("awaiting extract", state.enrich_stats, None)
        await asyncio.sleep(1.0)

    if state.should_stop():
        return

    # Detect and create patterns from indexed parameter groups
    if on_progress:
        on_progress("detecting patterns", state.enrich_stats, None)

    patterns_created = await asyncio.to_thread(
        detect_and_create_patterns,
        state.facility,
        state.tree_name,
    )
    if patterns_created and on_progress:
        on_progress(
            f"found {patterns_created} parameter patterns",
            state.enrich_stats,
            [{"patterns_created": patterns_created}],
        )

    model = get_model("language")
    system_prompt = _build_system_prompt(state.facility, state.tree_name)

    # Build version descriptions from config
    version_descs: dict[int, str] = {}
    for vc in state.tree_config.get("versions", []):
        if "description" in vc:
            version_descs[vc["version"]] = vc["description"]

    # --- Phase 1: Enrich patterns (one representative per group) ---
    while not state.should_stop() and not state.budget_exhausted:
        patterns = await asyncio.to_thread(
            claim_patterns_for_enrichment,
            state.facility,
            state.tree_name,
            limit=state.batch_size,
        )
        if not patterns:
            break

        state.enrich_phase.record_activity(len(patterns))
        pattern_ids = [p["id"] for p in patterns]

        # Build prompt from representative nodes
        batch_nodes = []
        for p in patterns:
            batch_nodes.append(
                {
                    "path": p["representative_path"],
                    "node_type": p["node_type"],
                    "tags": p.get("tags"),
                    "units": p.get("units"),
                    # Add pattern context for the LLM
                    "_pattern_leaf": p["leaf_name"],
                    "_pattern_grandparent": p["grandparent_path"],
                    "_pattern_count": p["index_count"],
                }
            )

        # Fetch tree context for the representative nodes
        rep_paths = [p["representative_path"] for p in patterns]
        tree_context = await asyncio.to_thread(
            fetch_enrichment_context,
            state.facility,
            state.tree_name,
            rep_paths,
        )

        if on_progress:
            on_progress(
                f"enriching {len(patterns)} patterns",
                state.enrich_stats,
                [{"pattern_count": len(patterns)}],
            )

        try:
            user_prompt = _build_user_prompt(
                batch_nodes, version_descs, tree_context=tree_context
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            parsed, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=messages,
                response_model=StaticNodeBatch,
            )
            state.enrich_stats.cost += cost

            # Match results to patterns by representative path
            descriptions: dict[str, str] = {}
            metadata: dict[str, dict] = {}
            for r in parsed.results:
                for p in patterns:
                    if p["representative_path"] == r.path:
                        descriptions[p["id"]] = r.description or ""
                        if r.keywords or r.category:
                            metadata[p["id"]] = {
                                "keywords": r.keywords,
                                "category": r.category,
                            }
                        break

            propagated = await asyncio.to_thread(
                mark_patterns_enriched,
                pattern_ids,
                descriptions,
                metadata,
            )

            state.enrich_stats.processed += len(patterns)
            state.enrich_stats.record_batch(len(patterns))

            if on_progress:
                on_progress(
                    f"{len(patterns)} patterns → {propagated} nodes propagated",
                    state.enrich_stats,
                    [
                        {
                            "patterns": len(patterns),
                            "propagated": propagated,
                            "cost": cost,
                        }
                    ],
                )

        except Exception as e:
            logger.error("Pattern enrich batch failed: %s", e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_pattern_claims, pattern_ids)

        await asyncio.sleep(0.1)

    # --- Phase 2: Enrich remaining non-pattern nodes by parent group ---
    while not state.should_stop():
        if state.budget_exhausted:
            if on_progress:
                on_progress("budget exhausted", state.enrich_stats, None)
            break

        # Claim a parent STRUCTURE with un-enriched children
        parent_data = await asyncio.to_thread(
            claim_parent_for_enrichment,
            state.facility,
            state.tree_name,
        )

        if not parent_data:
            state.enrich_phase.record_idle()
            if state.enrich_phase.done:
                break
            if on_progress:
                on_progress("idle", state.enrich_stats, None)
            await asyncio.sleep(2.0)
            continue

        state.enrich_phase.record_activity(1)
        parent_id = parent_data["parent_id"]
        parent_path = parent_data["parent_path"]
        children = parent_data["children"]

        if not children:
            await asyncio.to_thread(release_parent_claim, parent_id)
            continue

        # Fetch tree hierarchy context for the children
        child_paths = [c["path"] for c in children]
        tree_context = await asyncio.to_thread(
            fetch_enrichment_context,
            state.facility,
            state.tree_name,
            child_paths,
        )

        # Build prompt batch from all children of this parent
        batch_nodes = []
        for c in children:
            batch_nodes.append(
                {
                    "path": c["path"],
                    "node_type": c["node_type"],
                    "tags": c.get("tags"),
                    "units": c.get("units"),
                }
            )

        if on_progress:
            on_progress(
                f"{parent_path} ({len(children)} children)",
                state.enrich_stats,
                [{"path": parent_path, "count": len(children)}],
            )

        try:
            user_prompt = _build_user_prompt(
                batch_nodes, version_descs, tree_context=tree_context
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            parsed, cost, _tokens = await asyncio.to_thread(
                call_llm_structured,
                model=model,
                messages=messages,
                response_model=StaticNodeBatch,
            )
            state.enrich_stats.cost += cost

            # Build description and metadata maps keyed by child ID
            descriptions: dict[str, str] = {}
            metadata: dict[str, dict] = {}
            for r in parsed.results:
                matched_id = None
                for c in children:
                    if c["path"] == r.path:
                        matched_id = c["id"]
                        break
                if matched_id:
                    descriptions[matched_id] = r.description or ""
                    if r.keywords or r.category:
                        metadata[matched_id] = {
                            "keywords": r.keywords,
                            "category": r.category,
                        }

            enriched = await asyncio.to_thread(
                mark_parent_children_enriched,
                parent_id,
                descriptions,
                metadata,
            )

            state.enrich_stats.processed += 1
            state.enrich_stats.record_batch(1)

            if on_progress:
                on_progress(
                    f"{parent_path}: {enriched} children enriched",
                    state.enrich_stats,
                    [
                        {
                            "path": parent_path,
                            "description": f"{enriched} children",
                            "cost": cost,
                        }
                    ],
                )

        except Exception as e:
            logger.error("Parent enrich failed for %s: %s", parent_path, e)
            state.enrich_stats.errors += 1
            await asyncio.to_thread(release_parent_claim, parent_id)

        await asyncio.sleep(0.1)
