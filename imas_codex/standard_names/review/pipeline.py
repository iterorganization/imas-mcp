"""SN review pipeline orchestrator.

Wires the EXTRACT → ENRICH → REVIEW → PERSIST workers into the generic
discovery engine for the ``sn review`` command.

Unlike the *generate* pipeline (which creates new names), this pipeline
**reviews existing** StandardName nodes: loading them from the graph,
enriching with cluster/neighborhood context, running LLM quality scoring,
and persisting the review scores back.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from imas_codex.discovery.base.engine import WorkerSpec, run_discovery_engine

if TYPE_CHECKING:
    from imas_codex.standard_names.review.state import SNReviewState

logger = logging.getLogger(__name__)


# =============================================================================
# EXTRACT phase — load names from graph, filter, batch
# =============================================================================


async def extract_review_worker(state: SNReviewState, **_kwargs: Any) -> None:
    """Load StandardNames from graph, apply filters, and form review batches.

    1. Query full StandardName catalog into ``state.all_names``.
    2. Apply CLI filters (--ids, --domain, --status, --unreviewed, --re-review)
       to produce ``state.target_names``.
    3. Reconstruct clusters for targets via batch graph query.
    4. Group into review batches by (cluster × unit) with token budgets.
    5. Store in ``state.review_batches``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.graph.client import GraphClient
    from imas_codex.standard_names.review.enrichment import (
        group_into_review_batches,
        reconstruct_clusters_batch,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_extract")
    wlog.info("Starting review extraction")
    state.extract_stats.status_text = "loading catalog…"

    # --- Step 1: Load full catalog -------------------------------------------

    def _load_all_names() -> list[dict]:
        with GraphClient() as gc:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
                RETURN sn.id AS id,
                       sn.description AS description,
                       sn.documentation AS documentation,
                       sn.kind AS kind,
                       coalesce(u.id, sn.unit) AS unit,
                       sn.tags AS tags,
                       sn.links AS links,
                       sn.imas_paths AS imas_paths,
                       sn.physical_base AS physical_base,
                       sn.subject AS subject,
                       sn.component AS component,
                       sn.coordinate AS coordinate,
                       sn.position AS position,
                       sn.process AS process,
                       sn.cocos_transformation_type AS cocos_transformation_type,
                       sn.physics_domain AS physics_domain,
                       sn.review_status AS review_status,
                       sn.reviewer_score AS reviewer_score,
                       sn.review_input_hash AS review_input_hash,
                       sn.embedding AS embedding,
                       sn.review_tier AS review_tier,
                       sn.reviewer_comments AS reviewer_comments,
                       sn.source_type AS source_type,
                       sn.source_id AS source_id,
                       sn.generated_at AS generated_at,
                       sn.reviewed_at AS reviewed_at,
                       sn.link_status AS link_status
                """
            )
            return [dict(r) for r in rows] if rows else []

    all_names = await asyncio.to_thread(_load_all_names)
    state.all_names = all_names
    wlog.info("Loaded %d StandardNames from graph", len(all_names))

    # --- Step 2: Compute review_input_hash for staleness detection ----------

    _compute_hash = _get_hash_fn()
    if _compute_hash is not None:
        for name in all_names:
            name["_computed_hash"] = _compute_hash(name)

    # --- Step 3: Apply filters -----------------------------------------------

    state.extract_stats.status_text = "filtering targets…"

    # IDS filter: restrict to names linked to IMASNode in that IDS
    ids_eligible: set[str] | None = None
    if state.ids_filter:
        ids_eligible = await asyncio.to_thread(_query_ids_names, state.ids_filter)
        wlog.info(
            "IDS filter %r: %d eligible names", state.ids_filter, len(ids_eligible)
        )

    targets: list[dict] = []
    for name in all_names:
        nid = name.get("id", "")

        # --ids filter
        if ids_eligible is not None and nid not in ids_eligible:
            continue

        # --domain filter
        if state.domain_filter:
            domain = name.get("physics_domain") or ""
            if domain.lower() != state.domain_filter.lower():
                continue

        # --status filter
        if state.status_filter:
            status = name.get("review_status") or "drafted"
            if status != state.status_filter:
                continue

        # --unreviewed: no score OR stale hash
        if state.unreviewed_only:
            has_score = name.get("reviewer_score") is not None
            stored_hash = name.get("review_input_hash")
            computed_hash = name.get("_computed_hash")
            is_stale = (
                computed_hash is not None
                and stored_hash is not None
                and computed_hash != stored_hash
            )
            if has_score and not is_stale:
                continue

        # --re-review: no filtering (include already-reviewed)
        # (default: skip already-reviewed unless --re-review)
        if not state.re_review and not state.unreviewed_only:
            if name.get("reviewer_score") is not None:
                continue

        targets.append(name)

    state.target_names = targets
    wlog.info(
        "Filter result: %d targets from %d total (ids=%s, domain=%s, "
        "status=%s, unreviewed=%s, re_review=%s)",
        len(targets),
        len(all_names),
        state.ids_filter,
        state.domain_filter,
        state.status_filter,
        state.unreviewed_only,
        state.re_review,
    )

    if not targets:
        wlog.info("No names match review filters — nothing to review")
        state.extract_stats.total = 0
        state.extract_stats.processed = 0
        state.extract_stats.freeze_rate()
        state.extract_phase.mark_done()
        return

    # --- Step 4: Reconstruct clusters ----------------------------------------

    state.extract_stats.status_text = "reconstructing clusters…"

    def _batch_clusters() -> dict[str, dict | None]:
        with GraphClient() as gc:
            return reconstruct_clusters_batch(targets, gc)

    clusters = await asyncio.to_thread(_batch_clusters)
    wlog.info(
        "Cluster reconstruction: %d/%d names have clusters",
        sum(1 for v in clusters.values() if v is not None),
        len(targets),
    )

    # --- Step 5: Group into batches ------------------------------------------

    state.extract_stats.status_text = "forming batches…"
    batches = group_into_review_batches(
        targets,
        clusters,
        max_batch_size=state.batch_size,
        token_budget=8000,
    )
    state.review_batches = batches

    total_items = sum(len(b["names"]) for b in batches)
    state.extract_stats.total = total_items
    state.extract_stats.processed = total_items
    state.extract_stats.record_batch(total_items)

    wlog.info(
        "Extraction complete: %d batches, %d names",
        len(batches),
        total_items,
    )
    state.stats["extract_batches"] = len(batches)
    state.stats["extract_count"] = total_items

    state.extract_stats.freeze_rate()
    state.extract_phase.mark_done()
    state.extract_stats.stream_queue.add(
        [
            {
                "primary_text": "extract",
                "description": f"{total_items} names in {len(batches)} batches",
            }
        ]
    )


# =============================================================================
# ENRICH phase — neighborhood context + audit findings
# =============================================================================


async def enrich_review_worker(state: SNReviewState, **_kwargs: Any) -> None:
    """Enrich each batch with neighborhood context and audit findings.

    For each batch:
    1. Build semantic neighborhood via vector search.
    2. Attach relevant audit findings (one-line summary per finding).
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.standard_names.review.enrichment import (
        build_neighborhood_context,
    )

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_enrich")

    batches = state.review_batches
    if not batches:
        wlog.info("No batches to enrich — skipping")
        state.enrich_stats.freeze_rate()
        state.enrich_phase.mark_done()
        return

    state.enrich_stats.total = len(batches)
    wlog.info("Enriching %d batches with neighborhood context", len(batches))

    for i, batch in enumerate(batches):
        if state.should_stop():
            wlog.info("Stop requested during enrichment at batch %d", i)
            break

        # Neighborhood context via vector search
        neighbors = await asyncio.to_thread(
            build_neighborhood_context,
            batch,
            state.all_names,
            k=state.neighborhood_k,
        )
        batch["neighborhood"] = neighbors

        # Audit findings (if audit was run)
        if state.audit_report is not None:
            batch_ids = {n.get("id", "") for n in batch.get("names", [])}
            findings = _extract_audit_findings(state.audit_report, batch_ids)
            batch["audit_findings"] = findings
        else:
            batch["audit_findings"] = []

        state.enrich_stats.processed = i + 1
        state.enrich_stats.record_batch(1)

    wlog.info(
        "Enrichment complete: %d batches enriched",
        state.enrich_stats.processed,
    )
    state.enrich_stats.freeze_rate()
    state.enrich_phase.mark_done()
    state.enrich_stats.stream_queue.add(
        [
            {
                "primary_text": "enrich",
                "description": (f"{state.enrich_stats.processed} batches enriched"),
            }
        ]
    )


# =============================================================================
# REVIEW phase — LLM quality scoring with budget management
# =============================================================================


async def review_review_worker(state: SNReviewState, **_kwargs: Any) -> None:
    """Run LLM quality review on each enriched batch.

    Reuses the existing ``_review_batch()`` pattern from ``workers.py``
    with additions:
    - Budget management (reserve/reconcile)
    - Neighborhood context in prompts
    - Audit findings in prompts
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.settings import get_model

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_review")

    batches = state.review_batches
    if not batches:
        wlog.info("No batches to review — skipping")
        state.review_stats.freeze_rate()
        state.review_phase.mark_done()
        return

    model = state.review_model or get_model("language")
    wlog.info("Reviewing %d batches (model=%s)", len(batches), model)

    # Shared context for prompt rendering
    grammar_enums = _get_grammar_enums()
    compose_ctx = _get_compose_context_for_review()
    calibration_entries = _load_calibration_entries()

    total_items = sum(len(b["names"]) for b in batches)
    state.review_stats.total = total_items

    sem = asyncio.Semaphore(state.concurrency)
    scored: list[dict] = []
    total_cost = 0.0
    total_tokens = 0
    errors = 0
    revised = 0

    async def _process_batch(batch_idx: int, batch: dict) -> list[dict]:
        nonlocal total_cost, total_tokens, errors, revised

        async with sem:
            if state.should_stop():
                return []

            names = batch.get("names", [])
            if not names:
                return []

            # Estimate cost and reserve budget
            estimated_cost = len(names) * 0.002  # ~$0.002 per name heuristic
            reserved = estimated_cost * 1.3
            actual_cost = 0.0

            if state.budget_manager:
                if not state.budget_manager.reserve(estimated_cost):
                    wlog.info(
                        "Budget exhausted at batch %d — stopping review",
                        batch_idx,
                    )
                    state.stop_requested = True
                    return []

            try:
                batch_scored = await _review_single_batch(
                    names=names,
                    model=model,
                    grammar_enums=grammar_enums,
                    compose_ctx=compose_ctx,
                    calibration_entries=calibration_entries,
                    batch_context=batch.get("group_key", ""),
                    neighborhood=batch.get("neighborhood", []),
                    audit_findings=batch.get("audit_findings", []),
                    wlog=wlog,
                )
                batch_cost = batch_scored.pop("_cost", 0.0)
                batch_tokens = batch_scored.pop("_tokens", 0)
                batch_revised = batch_scored.pop("_revised", 0)
                batch_items = batch_scored.pop("_items", [])

                actual_cost = batch_cost
                total_cost += batch_cost
                total_tokens += batch_tokens
                revised += batch_revised

                state.review_stats.cost += batch_cost
                state.review_stats.processed += len(names)
                state.review_stats.record_batch(len(names))

                # Stream progress
                state.review_stats.stream_queue.add(
                    [
                        {
                            "primary_text": f"batch {batch_idx + 1}",
                            "description": (
                                f"{len(batch_items)} scored  ${batch_cost:.3f}"
                            ),
                        }
                    ]
                )
                return batch_items

            except Exception:
                wlog.debug(
                    "Review batch %d failed",
                    batch_idx,
                    exc_info=True,
                )
                errors += len(names)
                state.review_stats.errors += len(names)
                # Pass through unscored on failure
                return list(names)

            finally:
                if state.budget_manager:
                    state.budget_manager.reconcile(reserved, actual_cost)

    tasks = [_process_batch(i, batch) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, list):
            scored.extend(r)
        elif isinstance(r, Exception):
            wlog.warning("Batch task failed: %s", r)

    state.review_results = scored

    wlog.info(
        "Review complete: %d scored, %d revised, %d errors (cost=$%.4f, tokens=%d)",
        len(scored),
        revised,
        errors,
        total_cost,
        total_tokens,
    )
    state.stats["review_scored"] = len(scored)
    state.stats["review_revised"] = revised
    state.stats["review_errors"] = errors
    state.stats["review_cost"] = total_cost

    state.review_stats.freeze_rate()
    state.review_phase.mark_done()


# =============================================================================
# PERSIST phase — write review scores back to graph
# =============================================================================


async def persist_review_worker(state: SNReviewState, **_kwargs: Any) -> None:
    """Persist review results to graph.

    For each reviewed name:
    1. Compute fresh ``review_input_hash``.
    2. Add ``reviewed_at`` timestamp and ``reviewer_model``.
    3. Write via ``write_standard_names()``.
    """
    from imas_codex.cli.logging import WorkerLogAdapter
    from imas_codex.standard_names.graph_ops import write_standard_names

    wlog = WorkerLogAdapter(logger, worker_name="sn_review_persist")

    results = state.review_results
    if not results:
        wlog.info("No review results to persist — skipping")
        state.persist_stats.freeze_rate()
        state.persist_phase.mark_done()
        return

    state.persist_stats.total = len(results)
    wlog.info("Persisting %d review results", len(results))

    # Stamp provenance
    model = state.review_model or "unknown"
    reviewed_at = datetime.now(UTC).isoformat()
    _compute_hash = _get_hash_fn()

    for entry in results:
        entry["reviewer_model"] = model
        entry["reviewed_at"] = reviewed_at

        # Compute fresh review_input_hash
        if _compute_hash is not None:
            entry["review_input_hash"] = _compute_hash(entry)

    # Write to graph
    def _write() -> int:
        return write_standard_names(results)

    written = await asyncio.to_thread(_write)

    state.persist_stats.processed = written
    state.persist_stats.record_batch(written)

    wlog.info("Persisted %d/%d review results", written, len(results))
    state.stats["persist_count"] = written

    state.persist_stats.freeze_rate()
    state.persist_phase.mark_done()
    state.persist_stats.stream_queue.add(
        [
            {
                "primary_text": "persist",
                "description": f"{written} names updated",
            }
        ]
    )


# =============================================================================
# Pipeline entry point
# =============================================================================


async def run_sn_review_engine(
    state: SNReviewState,
    *,
    stop_event: asyncio.Event | None = None,
    on_worker_status: Any | None = None,
) -> None:
    """Run the SN review pipeline.

    Pipeline::

        EXTRACT → ENRICH → REVIEW → PERSIST

    Extract loads and filters StandardNames, reconstructs clusters, batches.
    Enrich adds neighborhood context and audit findings.
    Review runs LLM quality scoring with budget management.
    Persist writes review scores back to the graph.

    Args:
        state: Populated ``SNReviewState`` with filters and config.
        stop_event: Optional asyncio.Event for CLI shutdown signalling.
        on_worker_status: Optional callback for progress display updates.
    """
    workers = [
        WorkerSpec(
            "extract",
            "extract_phase",
            extract_review_worker,
        ),
        WorkerSpec(
            "enrich",
            "enrich_phase",
            enrich_review_worker,
            depends_on=["extract_phase"],
        ),
        WorkerSpec(
            "review",
            "review_phase",
            review_review_worker,
            depends_on=["enrich_phase"],
        ),
        WorkerSpec(
            "persist",
            "persist_phase",
            persist_review_worker,
            depends_on=["review_phase"],
        ),
    ]

    await run_discovery_engine(
        state,
        workers,
        stop_event=stop_event,
        on_worker_status=on_worker_status,
    )


# =============================================================================
# Internal helpers
# =============================================================================


def _get_hash_fn():
    """Return ``compute_review_input_hash`` or ``None`` if unavailable.

    The audits module (Layer 1) provides this function.  If it hasn't
    been created yet, we gracefully degrade — names are still reviewed
    but staleness detection won't work.
    """
    try:
        from imas_codex.standard_names.review.audits import compute_review_input_hash

        return compute_review_input_hash
    except (ImportError, ModuleNotFoundError):
        logger.debug("compute_review_input_hash not available (audits not yet created)")
        return None


def _query_ids_names(ids_prefix: str) -> set[str]:
    """Return StandardName ids linked to IMASNodes in a given IDS."""
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        rows = gc.query(
            """
            MATCH (sn:StandardName)<-[:HAS_STANDARD_NAME]-(node:IMASNode)
            WHERE node.id STARTS WITH $ids_prefix
            RETURN DISTINCT sn.id AS id
            """,
            ids_prefix=ids_prefix + "/",
        )
        return {r["id"] for r in rows} if rows else set()


def _extract_audit_findings(audit_report: Any, batch_ids: set[str]) -> list[str]:
    """Extract one-line audit findings relevant to names in the batch.

    Processes the AuditReport from Layer 1, extracting per-name findings
    and capping verbosity.
    """
    findings: list[str] = []
    if audit_report is None:
        return findings

    # AuditReport has .findings: list of AuditFinding
    for finding in getattr(audit_report, "findings", []):
        affected = getattr(finding, "affected_names", []) or []
        # Check if any affected name is in this batch
        overlap = [nid for nid in affected if nid in batch_ids]
        if overlap:
            severity = getattr(finding, "severity", "info")
            message = getattr(finding, "message", str(finding))
            category = getattr(finding, "category", "general")
            # One-line summary, capped
            summary = f"[{severity}:{category}] {message[:200]}"
            findings.append(summary)

    # Cap total findings per batch
    return findings[:20]


async def _review_single_batch(
    *,
    names: list[dict],
    model: str,
    grammar_enums: dict[str, list[str]],
    compose_ctx: dict[str, Any],
    calibration_entries: list[dict],
    batch_context: str,
    neighborhood: list[dict],
    audit_findings: list[str],
    wlog: logging.LoggerAdapter,
) -> dict[str, Any]:
    """Review a single batch via LLM — mirrors ``_review_batch()`` from workers.py.

    Returns a dict with keys ``_items`` (scored entries), ``_cost``,
    ``_tokens``, ``_revised``.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import SNQualityReviewBatch, SNReviewVerdict

    cal = calibration_entries or []

    # Enrich items with validation issues for reviewer context
    items_with_issues = []
    for item in names:
        item_data = dict(item)
        item_data["validation_issues"] = item.get("validation_issues", [])
        items_with_issues.append(item_data)

    # Merge compose context with review-specific keys
    base_ctx = dict(compose_ctx) if compose_ctx else {}

    # Build context for prompt rendering
    context = {
        **base_ctx,
        "items": items_with_issues,
        "existing_names": [],  # not needed for standalone review
        "calibration_entries": cal,
        "batch_context": batch_context,
        "nearby_existing_names": neighborhood,
        "audit_findings": audit_findings,
        **grammar_enums,
    }

    # System prompt: rubric + calibration (cached across batches)
    system_context = {
        **base_ctx,
        "items": [],
        "existing_names": [],
        "calibration_entries": cal,
        "batch_context": "",
        "nearby_existing_names": [],
        "audit_findings": [],
        **grammar_enums,
    }
    system_prompt = render_prompt("sn/review", system_context)

    # User prompt: actual candidates + context
    user_prompt = render_prompt("sn/review", context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result, cost, tokens = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=SNQualityReviewBatch,
    )

    # Map reviews back to original entries by source_id
    entry_map: dict[str, dict] = {}
    for entry in names:
        sid = entry.get("source_id") or entry.get("id") or ""
        entry_map[sid] = entry

    scored: list[dict] = []
    revised_count = 0

    for review in result.reviews:
        original = entry_map.get(review.source_id)
        if original is None:
            wlog.debug("Review returned unknown source_id: %s", review.source_id)
            continue

        # Store review scores on the entry
        original["reviewer_score"] = review.scores.score
        original["reviewer_scores"] = json.dumps(review.scores.model_dump())
        original["reviewer_comments"] = review.reasoning
        original["review_tier"] = review.scores.tier

        if review.verdict == SNReviewVerdict.revise and review.revised_name:
            revised_entry = dict(original)
            revised_entry["id"] = review.revised_name
            if review.revised_fields:
                for key, value in review.revised_fields.items():
                    if key in revised_entry:
                        revised_entry[key] = value
            scored.append(revised_entry)
            revised_count += 1
            wlog.debug(
                "Revised %r → %r (score %.2f): %s",
                review.standard_name,
                review.revised_name,
                review.scores.score,
                review.reasoning[:120],
            )
        else:
            scored.append(original)
            if review.verdict == SNReviewVerdict.reject:
                wlog.debug(
                    "Low score %r (%.2f, %s): %s",
                    review.standard_name,
                    review.scores.score,
                    review.scores.tier,
                    review.reasoning[:120],
                )

    # Pass through entries not in the review result
    reviewed_ids = {r.source_id for r in result.reviews}
    for entry in names:
        sid = entry.get("source_id") or entry.get("id") or ""
        if sid not in reviewed_ids:
            scored.append(entry)

    return {
        "_items": scored,
        "_cost": cost,
        "_tokens": tokens,
        "_revised": revised_count,
    }


# ---------------------------------------------------------------------------
# Reused helpers from workers.py (avoid circular imports)
# ---------------------------------------------------------------------------


def _get_grammar_enums() -> dict[str, list[str]]:
    """Return grammar enum values for prompt context."""
    try:
        from imas_standard_names.grammar import (
            BinaryOperator,
            Component,
            GeometricBase,
            Object,
            Position,
            Process,
            Subject,
            Transformation,
        )

        return {
            "subjects": [e.value for e in Subject],
            "components": [e.value for e in Component],
            "coordinates": [e.value for e in Component],
            "positions": [e.value for e in Position],
            "processes": [e.value for e in Process],
            "transformations": [e.value for e in Transformation],
            "geometric_bases": [e.value for e in GeometricBase],
            "objects": [e.value for e in Object],
            "binary_operators": [e.value for e in BinaryOperator],
        }
    except (ImportError, ModuleNotFoundError):
        logger.debug("imas_standard_names.grammar not available")
        return {}


def _get_compose_context_for_review() -> dict[str, Any]:
    """Return compose context keys needed by shared prompt includes."""
    try:
        from imas_codex.standard_names.context import build_compose_context

        return build_compose_context()
    except Exception:
        logger.debug("build_compose_context unavailable", exc_info=True)
        return {}


def _load_calibration_entries() -> list[dict]:
    """Load calibration entries from benchmark_calibration.yaml."""
    from pathlib import Path

    import yaml

    cal_path = Path(__file__).parents[1] / "benchmark_calibration.yaml"
    if not cal_path.exists():
        return []
    try:
        with open(cal_path) as f:
            data = yaml.safe_load(f)
        return data.get("entries", [])
    except Exception:
        return []
