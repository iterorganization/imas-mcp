"""SN loop — drives ``sn run`` via concurrent worker pools or legacy
domain rotation.

Primary entry point: :func:`run_sn_pools` (Phase 8 pool-based
orchestrator).  Legacy :func:`run_sn_loop` is retained for backward
compatibility and existing tests.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE

logger = logging.getLogger(__name__)

# Estimated cost of one minimal unit of LLM work (a single compose or
# review API call).  The loop continues scheduling turns while the
# remaining budget exceeds this estimate — no fixed dollar floor.
EST_UNIT_COST: float = 0.05


@dataclass
class RunSummary:
    """Aggregated result of a ``sn run`` invocation."""

    run_id: str
    turn_number: int
    started_at: datetime
    ended_at: datetime | None = None
    cost_spent: float = 0.0
    cost_limit: float = 0.0
    min_score: float | None = None
    names_composed: int = 0
    names_enriched: int = 0
    names_reviewed: int = 0
    names_regenerated: int = 0
    sources_reconciled: int = 0
    links_resolved: int = 0
    domains_touched: set[str] = field(default_factory=set)
    stop_reason: str = "completed"
    pass_records: list[dict[str, Any]] = field(default_factory=list)
    compose_cost: float = 0.0
    review_cost: float = 0.0


def _count_eligible_domains(
    only_domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return domains with extract-eligible DD paths, ordered by backlog size."""
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (n:IMASNode)
        WHERE n.node_category IN $categories
          AND n.node_type IN ['dynamic', 'constant']
          AND trim(coalesce(n.description, '')) <> ''
          AND NOT EXISTS {
              MATCH (sns:StandardNameSource {source_id: n.id, source_type: 'dd'})
              WHERE NOT (sns.status IN ['stale', 'failed', 'extracted'])
          }
        RETURN coalesce(n.physics_domain, 'unclassified') AS domain,
               count(*) AS remaining
        ORDER BY remaining DESC
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, categories=list(SN_SOURCE_CATEGORIES)))
    filtered = [r for r in rows if r["domain"] and r["domain"] != "unclassified"]
    if only_domain:
        filtered = [r for r in filtered if r["domain"] == only_domain]
    return filtered


def _existing_domain_targets(
    only_domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return domains that have un-enriched / un-reviewed names.

    Fallback when no extract-eligible paths remain, or when
    ``--skip-generate`` is set. Returns domains with at least one
    StandardName in an incomplete state.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain IS NOT NULL
          AND (
               sn.pipeline_status IN ['named', 'drafted']
            OR sn.reviewer_score_name IS NULL
          )
        UNWIND sn.physics_domain AS domain
        RETURN domain, count(*) AS remaining
        ORDER BY remaining DESC
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher))
    filtered = [r for r in rows if r["domain"]]
    if only_domain:
        filtered = [r for r in filtered if r["domain"] == only_domain]
    return filtered


def _pick_stalest_domain(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """From eligible domains, pick the one whose last SNRun is oldest.

    Queries the graph for the most recent ``SNRun.ended_at`` per candidate
    domain (via ``domains_touched``).  Returns the candidate with the
    oldest last run.  Tiebreak: domain with more remaining work wins.
    Domains never previously run (no matching SNRun) sort first.
    """
    if len(candidates) == 1:
        return candidates[0]

    from imas_codex.graph.client import GraphClient

    domain_names = [c["domain"] for c in candidates]

    cypher = """
        UNWIND $domains AS dom
        OPTIONAL MATCH (rr:SNRun)
          WHERE dom IN rr.domains_touched
            AND rr.ended_at IS NOT NULL
        WITH dom, max(rr.ended_at) AS last_run
        RETURN dom AS domain, last_run
    """
    with GraphClient() as gc:
        rows = list(gc.query(cypher, domains=domain_names))

    if not rows:
        return candidates[0]

    last_run_map = {r["domain"]: r["last_run"] for r in rows}

    # Sort: null last_run (never run) first, then oldest, then most remaining.
    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    def sort_key(entry: dict[str, Any]) -> tuple[int, datetime, int]:
        lr = last_run_map.get(entry["domain"])
        return (
            0 if lr is None else 1,
            lr if lr is not None else epoch,
            -entry["remaining"],
        )

    candidates_sorted = sorted(candidates, key=sort_key)
    winner = candidates_sorted[0]
    logger.debug(
        "Stale-first rotation: picked %s (last_run=%s, remaining=%d)",
        winner["domain"],
        last_run_map.get(winner["domain"]),
        winner["remaining"],
    )
    return winner


def select_next_domain(
    *,
    skip_generate: bool = False,
    only_domain: str | None = None,
) -> dict[str, Any] | None:
    """Select the next domain for a turn via stale-first rotation.

    Returns ``{"domain": str, "remaining": int}`` for the winning
    domain, or ``None`` if no domain has eligible work.

    When *only_domain* is set, rotation is bypassed — the explicit
    user choice always wins (provided it has work).
    """
    if only_domain:
        # Explicit user choice bypasses rotation
        if skip_generate:
            candidates = _existing_domain_targets(only_domain=only_domain)
        else:
            candidates = _count_eligible_domains(only_domain=only_domain)
            if not candidates:
                candidates = _existing_domain_targets(only_domain=only_domain)
        return candidates[0] if candidates else None

    # Find all eligible domains
    if skip_generate:
        candidates = _existing_domain_targets()
    else:
        candidates = _count_eligible_domains()
        if not candidates:
            candidates = _existing_domain_targets()

    if not candidates:
        return None

    return _pick_stalest_domain(candidates)


# ── Status mapping ────────────────────────────────────────────────────
# Map RunSummary.stop_reason to SNRun.status lifecycle values.
_STOP_TO_STATUS: dict[str, str] = {
    "completed": "completed",
    "budget_exhausted": "completed",
    "stalled": "completed",
    "no_work": "completed",
    "dry_run": "completed",
    "interrupted": "interrupted",
    "failed": "failed",
    "degraded": "degraded",
}


async def run_sn_loop(
    cost_limit: float,
    *,
    turn_number: int = 1,
    min_score: float | None = None,
    per_domain_limit: int | None = None,
    concurrency: int = 2,
    dry_run: bool = False,
    only_domain: str | None = None,
    skip_generate: bool = False,
    skip_enrich: bool = False,
    skip_review: bool = False,
    skip_regen: bool = False,
    source: str = "dd",
    override_edits: list[str] | None = None,
    only: str | None = None,
    loop_state: Any | None = None,
    stop_event: Any | None = None,
) -> RunSummary:
    """Drive the ``sn run`` loop with one-domain-per-turn rotation.

    Each iteration picks ONE domain via stale-first rotation (oldest
    ``SNRun.ended_at`` wins, eligible-source count as tiebreak) and
    runs a full turn on it with the entire remaining budget.  The loop
    stops when the remaining budget can no longer fund a single unit of
    work (< :data:`EST_UNIT_COST`) or no domain has eligible work.

    Soft-limit semantics: the loop continues scheduling new turns while
    ``remaining > EST_UNIT_COST``.  In-flight leases always settle
    naturally — no truncation mid-LLM-call.

    When *only_domain* is set the rotation is bypassed — the explicit
    user choice is used every iteration until budget runs out or the
    domain has no remaining work.

    Args:
        loop_state: Optional :class:`SNLoopState` whose :class:`WorkerStats`
            fields are updated by workers for live progress display.
            When ``None``, progress is logged via the standard logger
            (plain mode).
        stop_event: Optional :class:`asyncio.Event` set by the shutdown
            harness.  Checked between turns for soft shutdown.
    """
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.turn import TurnConfig, run_turn

    started = datetime.now(UTC)
    summary = RunSummary(
        run_id=str(uuid.uuid4()),
        turn_number=turn_number,
        started_at=started,
        cost_limit=cost_limit,
        min_score=min_score,
    )

    if dry_run:
        domains = _count_eligible_domains(only_domain=only_domain)
        summary.pass_records.append(
            {
                "dry_run": True,
                "eligible_domains": [
                    {"domain": d["domain"], "remaining": d["remaining"]}
                    for d in domains
                ],
            }
        )
        summary.ended_at = datetime.now(UTC)
        summary.stop_reason = "dry_run"
        return summary

    # Single shared BudgetManager for the entire run.  All phases (compose,
    # review_names, review_docs, regen) draw from the same pool so the total
    # spend across every phase is gated by the user-specified cost_limit.
    shared_mgr = BudgetManager(cost_limit, run_id=summary.run_id)
    await shared_mgr.start()

    # Pre-create the SNRun node so LLMCost → FOR_RUN edges have a target
    # from the very first LLM call.
    from imas_codex.standard_names.graph_ops import create_sn_run_open

    create_sn_run_open(
        summary.run_id,
        started_at=summary.started_at,
        cost_limit=cost_limit,
        turn_number=turn_number,
        min_score=min_score,
    )

    # Per-domain stall detection: track (last_remaining, stall_count).
    # If a domain is selected with the same `remaining` count as its previous
    # turn AND the turn made no forward progress (compose/review/regen), we
    # count it as a stall.  Two consecutive stalls → stop to avoid infinite
    # turn loops that burn $0.05/turn in extract/enrich overhead.
    MAX_STALLS = 2
    domain_stalls: dict[str, tuple[int, int]] = {}

    # Domain rotation tracking removed from SNLoopState — pending counts
    # now come from graph queries via pending_fn (see cli/sn.py).

    try:
        while True:
            # ── Soft shutdown check ───────────────────────────────
            if stop_event is not None and stop_event.is_set():
                summary.stop_reason = "interrupted"
                logger.info("Stop event set — exiting loop.")
                break

            # ── Budget gate (soft limit) ──────────────────────────
            # Continue while we can afford at least one more LLM call.
            # Use summary.cost_spent (which accumulates via max(mgr.spent,
            # cost_spent + phase_sum)) so tests that mock run_turn work too.
            remaining_budget = cost_limit - summary.cost_spent
            if remaining_budget < EST_UNIT_COST:
                summary.stop_reason = "budget_exhausted"
                logger.info(
                    "Budget exhausted: $%.4f remaining < est unit cost $%.2f",
                    remaining_budget,
                    EST_UNIT_COST,
                )
                break

            # ── Domain selection (stale-first rotation) ───────────
            entry = select_next_domain(
                skip_generate=skip_generate,
                only_domain=only_domain,
            )
            if entry is None:
                summary.stop_reason = "completed"
                logger.info("No eligible domains; nothing to do.")
                break

            dom = entry["domain"]
            logger.info(
                "Turn %d → domain %s (remaining=%d, budget=$%.2f)",
                turn_number,
                dom,
                entry["remaining"],
                remaining_budget,
            )

            # Domain breadcrumb removed; row pending counts come from graph.

            # Snapshot forward-progress counters before the turn so we can
            # detect whether the turn actually advanced anything.
            prev_progress = (
                summary.names_composed
                + summary.names_reviewed
                + summary.names_regenerated
            )
            spent_before = shared_mgr.spent

            # ── Run turn with full remaining budget ───────────────
            cfg = TurnConfig(
                domain=dom,
                cost_limit=remaining_budget,
                limit=per_domain_limit,
                concurrency=concurrency,
                dry_run=False,
                run_id=summary.run_id,
                turn_number=turn_number,
                min_score=min_score,
                skip_generate=skip_generate,
                skip_enrich=skip_enrich,
                skip_review=skip_review,
                skip_regen=skip_regen or min_score is None,
                source=source,
                override_edits=override_edits,
                only=only,
                shared_budget=shared_mgr,
                loop_state=loop_state,
                stop_event=stop_event,
            )
            results = await run_turn(cfg)

            # Measure turn cost: prefer the shared manager's actual spend delta
            # (which captures all LLM calls including L7 Opus revisions) but
            # fall back to summing PhaseResult.cost values for backwards
            # compatibility with tests that mock run_turn without touching the
            # shared BudgetManager.
            mgr_delta = shared_mgr.spent - spent_before
            phase_sum = sum(r.cost for r in results)
            turn_cost = max(mgr_delta, phase_sum)

            # ── Accumulate counters ───────────────────────────────
            for phase in results:
                if phase.name == "generate":
                    summary.names_composed += phase.count
                elif phase.name == "generate_docs":
                    summary.names_enriched += phase.count
                elif phase.name in ("review_names", "review_docs"):
                    summary.names_reviewed += phase.count
                elif phase.name == "refine_name":
                    summary.names_regenerated += phase.count
                elif phase.name == "reconcile":
                    summary.sources_reconciled += phase.count
                elif phase.name == "link":
                    summary.links_resolved += phase.count

            summary.cost_spent = max(shared_mgr.spent, summary.cost_spent + phase_sum)
            summary.domains_touched.add(dom)

            # done_domains counter removed — graph pending counts replace it.

            # Update phase-level cost breakdowns from shared manager.
            phase_spent = shared_mgr.phase_spent
            summary.compose_cost = phase_spent.get("generate", 0.0) + phase_spent.get(
                "refine_name", 0.0
            )
            summary.review_cost = phase_spent.get(
                "review_names", 0.0
            ) + phase_spent.get("review_docs", 0.0)

            summary.pass_records.append(
                {
                    "domain": dom,
                    "remaining_before": entry["remaining"],
                    "budget": remaining_budget,
                    "phases": [
                        {
                            "name": r.name,
                            "count": r.count,
                            "cost": r.cost,
                            "skipped": r.skipped,
                            "error": r.error,
                        }
                        for r in results
                    ],
                }
            )

            # If zero cost was incurred, the domain had no actionable
            # work — avoid an infinite loop by stopping.
            if turn_cost <= 0.0:
                logger.info(
                    "Turn on %s produced zero cost; stopping to avoid loop.", dom
                )
                summary.stop_reason = "completed"
                break

            # ── Per-domain stall detection ────────────────────────
            # Forward-progress check: if compose/review/regen didn't advance
            # AND `remaining` is unchanged vs this domain's previous turn,
            # record a stall.  Two consecutive stalls → stop to prevent
            # budget drain on unrecoverable items (vocab gaps, invariant
            # violations).
            cur_progress = (
                summary.names_composed
                + summary.names_reviewed
                + summary.names_regenerated
            )
            made_progress = cur_progress > prev_progress
            prev_remaining, prev_stalls = domain_stalls.get(dom, (-1, 0))
            if not made_progress and entry["remaining"] == prev_remaining:
                prev_stalls += 1
                domain_stalls[dom] = (entry["remaining"], prev_stalls)
                logger.warning(
                    "Domain %s stalled (remaining=%d unchanged, no progress) "
                    "— stall %d/%d",
                    dom,
                    entry["remaining"],
                    prev_stalls,
                    MAX_STALLS,
                )
                if prev_stalls >= MAX_STALLS:
                    logger.info(
                        "Domain %s hit %d consecutive stalls — "
                        "stopping to preserve budget ($%.2f remaining).",
                        dom,
                        MAX_STALLS,
                        remaining_budget,
                    )
                    summary.stop_reason = "stalled"
                    break
            else:
                domain_stalls[dom] = (entry["remaining"], 0)

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("sn run interrupted by user")
    finally:
        summary.ended_at = datetime.now(UTC)

        # Drain pending LLMCost graph writes.
        cost_is_exact = await shared_mgr.drain_pending()
        if not cost_is_exact and summary.stop_reason not in (
            "interrupted",
            "failed",
        ):
            summary.stop_reason = "degraded"

        # Refresh final cost from graph (includes all drained writes).
        # Use max() to preserve in-loop accumulation for tests that mock
        # run_turn without calling charge_event (where _spent stays 0).
        summary.cost_spent = max(
            summary.cost_spent,
            await shared_mgr.get_total_spent(force_refresh=True),
        )

        # Compute pipeline hash — best-effort, never block finalization.
        _pipeline_hash: str | None = None
        _pipeline_hash_detail: str | None = None
        try:
            from imas_codex.standard_names.pipeline_version import (
                compute_pipeline_hash,
            )

            ph = compute_pipeline_hash()
            _pipeline_hash = ph["_composite"]
            _pipeline_hash_detail = _json.dumps(
                {k: v for k, v in ph.items() if k != "_composite"}
            )
        except Exception:  # noqa: BLE001
            pass

        # Finalize the SNRun node (MATCH + SET on the pre-created node).
        from imas_codex.standard_names.graph_ops import finalize_sn_run

        finalize_sn_run(
            summary.run_id,
            status=_STOP_TO_STATUS.get(summary.stop_reason, "completed"),
            cost_spent=summary.cost_spent,
            cost_is_exact=cost_is_exact,
            ended_at=summary.ended_at,
            turn_number=summary.turn_number,
            cost_limit=round(summary.cost_limit, 6),
            compose_cost=round(summary.compose_cost, 6),
            review_cost=round(summary.review_cost, 6),
            min_score=summary.min_score,
            names_composed=summary.names_composed,
            names_enriched=summary.names_enriched,
            names_reviewed=summary.names_reviewed,
            names_regenerated=summary.names_regenerated,
            domains_touched=sorted(summary.domains_touched),
            stop_reason=summary.stop_reason,
            pipeline_hash=_pipeline_hash,
            pipeline_hash_detail=_pipeline_hash_detail,
        )

    return summary


def summary_table(summary: RunSummary) -> dict[str, Any]:
    """Flatten a :class:`RunSummary` for Rich display / JSON output."""
    return {
        "run_id": summary.run_id,
        "turn_number": summary.turn_number,
        "started_at": summary.started_at.isoformat(),
        "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
        "elapsed_s": (
            (summary.ended_at - summary.started_at).total_seconds()
            if summary.ended_at
            else None
        ),
        "cost_spent": round(summary.cost_spent, 6),
        "cost_limit": summary.cost_limit,
        "min_score": summary.min_score,
        "names_composed": summary.names_composed,
        "names_enriched": summary.names_enriched,
        "names_reviewed": summary.names_reviewed,
        "names_regenerated": summary.names_regenerated,
        "sources_reconciled": summary.sources_reconciled,
        "links_resolved": summary.links_resolved,
        "domains_touched": sorted(summary.domains_touched),
        "stop_reason": summary.stop_reason,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 8 — Pool-based orchestrator (replaces domain rotation)
# ═══════════════════════════════════════════════════════════════════════

# Default regen threshold when min_score is not explicitly provided.
# Imported from defaults.py — do not re-define here.


def _build_pool_specs(
    mgr: Any,
    stop_event: asyncio.Event,
    *,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    only_domain: str | None = None,
) -> list[Any]:
    """Construct 6 :class:`PoolSpec` objects wiring claims → batch processors.

    Each pool gets two adapter closures:

    * **claim adapter** — runs the synchronous ``claim_*_seed_and_expand``
      graph function in a worker thread and returns the result wrapped in
      a dict (``{"items": [...]}``), or ``None`` on empty.
    * **process adapter** — unpacks the claimed batch and delegates to the
      corresponding ``process_*_batch`` async function, forwarding the
      shared :class:`BudgetManager` and ``stop_event``.

    After construction, backlog throttle wrappers are applied to upstream
    pools (generate_name, generate_docs, refine_name, refine_docs) so they
    pause when their downstream review queues exceed the configured cap.
    """
    from collections.abc import Awaitable

    from imas_codex.standard_names.defaults import (
        REVIEW_DOCS_BACKLOG_CAP,
        REVIEW_NAME_BACKLOG_CAP,
    )
    from imas_codex.standard_names.graph_ops import (
        claim_generate_docs_seed_and_expand,
        claim_generate_name_seed_and_expand,
        claim_refine_docs_seed_and_expand,
        claim_refine_name_seed_and_expand,
        claim_review_docs_seed_and_expand,
        claim_review_name_seed_and_expand,
        release_generate_docs_claims,
        release_generate_name_claims,
        release_refine_docs_claims,
        release_refine_name_claims,
        release_review_docs_claims,
        release_review_names_claims,
    )
    from imas_codex.standard_names.pools import PoolSpec
    from imas_codex.standard_names.workers import (
        process_generate_docs_batch,
        process_generate_name_batch,
        process_refine_docs_batch,
        process_refine_name_batch,
        process_review_docs_batch,
        process_review_name_batch,
    )

    regen_score = min_score if min_score is not None else DEFAULT_MIN_SCORE
    _rotation_cap_kwargs: dict[str, Any] = {}
    if rotation_cap is not None:
        _rotation_cap_kwargs["rotation_cap"] = rotation_cap
    _review_name_cap = (
        review_name_backlog_cap
        if review_name_backlog_cap is not None
        else REVIEW_NAME_BACKLOG_CAP
    )
    _review_docs_cap = (
        review_docs_backlog_cap
        if review_docs_backlog_cap is not None
        else REVIEW_DOCS_BACKLOG_CAP
    )

    # ── Adapter factories ─────────────────────────────────────────────

    def _make_claim_adapter(
        claim_fn: Callable[..., list[dict[str, Any]]],
        **kwargs: Any,
    ) -> Callable[[], Awaitable[dict[str, Any] | None]]:
        """Wrap a sync claim function as an async ``ClaimFn``."""

        async def _adapter() -> dict[str, Any] | None:
            items = await asyncio.to_thread(claim_fn, **kwargs)
            if not items:
                return None
            # Alias source_id → path for DD items so compose/grouping helpers
            # that key on `item["path"]` (a legacy convention from the
            # extract-time batch shape) work uniformly with claim-shaped items.
            for it in items:
                if it.get("source_type") == "dd" and "path" not in it:
                    sid = it.get("source_id")
                    if sid:
                        it["path"] = sid
            return {"items": items}

        return _adapter

    def _make_process_adapter(
        process_fn: Callable[
            [list[dict[str, Any]], Any, asyncio.Event],
            Awaitable[int],
        ],
    ) -> Callable[[dict[str, Any]], Awaitable[int]]:
        """Wrap a batch processor as a ``ProcessFn``."""

        async def _adapter(batch: dict[str, Any]) -> int:
            return await process_fn(batch["items"], mgr, stop_event, on_event=on_event)

        return _adapter

    def _make_release_adapter(
        release_fn: Callable[..., int],
        ids_kwarg: str = "sn_ids",
    ) -> Callable[[dict[str, Any]], Awaitable[None]]:
        """Wrap a token-aware release function as an async ``ReleaseFn``.

        Extracts ``id`` and ``claim_token`` from batch items and forwards
        them as keyword arguments to *release_fn*.  All items in a batch
        share the same ``claim_token`` (set atomically at claim time).

        Parameters
        ----------
        release_fn:
            Sync release function accepting keyword arguments
            *<ids_kwarg>* and *claim_token*.
        ids_kwarg:
            Name of the ids keyword argument (``"sn_ids"`` for
            :class:`StandardName` pools; ``"source_ids"`` for
            :class:`StandardNameSource` pools).
        """

        async def _adapter(batch: dict[str, Any]) -> None:
            items = batch.get("items", [])
            if not items:
                return
            ids = [item["id"] for item in items]
            token: str = items[0].get("claim_token") or ""
            await asyncio.to_thread(
                release_fn,
                **{ids_kwarg: ids, "claim_token": token},
            )

        return _adapter

    # ── PoolSpec construction ─────────────────────────────────────────

    specs = [
        PoolSpec(
            name="generate_name",
            claim=_make_claim_adapter(
                claim_generate_name_seed_and_expand,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_generate_name_batch),
            release=_make_release_adapter(
                release_generate_name_claims, ids_kwarg="source_ids"
            ),
            weight=0.25,
        ),
        PoolSpec(
            name="review_name",
            claim=_make_claim_adapter(
                claim_review_name_seed_and_expand,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_review_name_batch),
            release=_make_release_adapter(
                release_review_names_claims, ids_kwarg="sn_ids"
            ),
            weight=0.15,
        ),
        PoolSpec(
            name="refine_name",
            claim=_make_claim_adapter(
                claim_refine_name_seed_and_expand,
                min_score=regen_score,
                **_rotation_cap_kwargs,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_refine_name_batch),
            release=_make_release_adapter(
                release_refine_name_claims, ids_kwarg="sn_ids"
            ),
            weight=0.15,
        ),
        PoolSpec(
            name="generate_docs",
            claim=_make_claim_adapter(
                claim_generate_docs_seed_and_expand,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_generate_docs_batch),
            release=_make_release_adapter(
                release_generate_docs_claims, ids_kwarg="sn_ids"
            ),
            weight=0.20,
        ),
        PoolSpec(
            name="review_docs",
            claim=_make_claim_adapter(
                claim_review_docs_seed_and_expand,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_review_docs_batch),
            release=_make_release_adapter(
                release_review_docs_claims, ids_kwarg="sn_ids"
            ),
            weight=0.15,
        ),
        PoolSpec(
            name="refine_docs",
            claim=_make_claim_adapter(
                claim_refine_docs_seed_and_expand,
                min_score=regen_score,
                **_rotation_cap_kwargs,
                **({"domain": only_domain} if only_domain else {}),
            ),
            process=_make_process_adapter(process_refine_docs_batch),
            release=_make_release_adapter(
                release_refine_docs_claims, ids_kwarg="sn_ids"
            ),
            weight=0.10,
        ),
    ]

    # ── Backlog throttle wiring ───────────────────────────────────────
    # Upstream generators/refiners pause when their downstream review
    # queue exceeds the configured cap.  The throttle wraps the claim
    # adapter to return None (skip) when the downstream pool's
    # PoolHealth.pending_count is over cap, causing the pool to enter
    # its normal exponential backoff.  No blocking, no special yield.
    specs_by_name = {s.name: s for s in specs}

    throttle_rules: list[tuple[str, str, int]] = [
        ("generate_name", "review_name", _review_name_cap),
        ("refine_name", "review_name", _review_name_cap),
        ("generate_docs", "review_docs", _review_docs_cap),
        ("refine_docs", "review_docs", _review_docs_cap),
    ]

    for upstream, downstream, cap in throttle_rules:
        spec = specs_by_name[upstream]
        downstream_health = specs_by_name[downstream].health
        original_claim = spec.claim

        async def _throttled_claim(
            _orig: Callable[[], Awaitable[dict[str, Any] | None]] = original_claim,
            _health: Any = downstream_health,
            _cap: int = cap,
            _up: str = upstream,
            _down: str = downstream,
        ) -> dict[str, Any] | None:
            if _health.pending_count > _cap:
                logger.debug(
                    "throttle: %s paused — %s backlog %d > cap %d",
                    _up,
                    _down,
                    _health.pending_count,
                    _cap,
                )
                return None
            return await _orig()

        spec.claim = _throttled_claim

    return specs


async def _seed_domain_sources(
    domain: str,
    source: str = "dd",
    stop_event: asyncio.Event | None = None,
) -> int:
    """Seed the generate_name pool with StandardNameSource nodes for *domain*.

    Calls :func:`~imas_codex.standard_names.sources.dd.extract_dd_candidates`
    to discover DD paths for *domain* that have no existing StandardNameSource,
    then writes them via
    :func:`~imas_codex.standard_names.graph_ops.merge_standard_name_sources`.

    Returns the number of sources written (0 if none found or source != "dd").
    """
    if source != "dd":
        return 0

    from imas_codex.standard_names.graph_ops import (
        get_existing_standard_names,
        merge_standard_name_sources,
    )
    from imas_codex.standard_names.sources.dd import extract_dd_candidates

    existing = await asyncio.to_thread(get_existing_standard_names)
    batches = await asyncio.to_thread(
        extract_dd_candidates,
        domain_filter=domain,
        existing_names=existing,
        force=False,
        name_only=False,
    )

    sources = []
    for batch in batches:
        for item in batch.items:
            path = item.get("path")
            if not path:
                continue
            sources.append(
                {
                    "id": f"dd:{path}",
                    "source_type": "dd",
                    "source_id": path,
                    "dd_path": path,
                    "batch_key": batch.group_key,
                    "status": "extracted",
                    "description": item.get("description")
                    or item.get("documentation")
                    or "",
                    "physics_domain": item.get("physics_domain"),
                }
            )

    if not sources:
        return 0

    written = await asyncio.to_thread(merge_standard_name_sources, sources, force=False)
    return written


async def run_sn_pools(
    cost_limit: float,
    *,
    min_score: float | None = None,
    rotation_cap: int | None = None,
    escalation_model: str | None = None,
    review_name_backlog_cap: int | None = None,
    review_docs_backlog_cap: int | None = None,
    source: str = "dd",
    only_domain: str | None = None,
    stop_event: asyncio.Event | None = None,
    loop_state: Any | None = None,
    pending_fn: Callable[[], dict[str, int]] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunSummary:
    """Run the pool-based ``sn run`` orchestrator (Phase 8).

    Replaces the per-domain serial :func:`run_sn_loop` with six
    concurrent worker pools that pull work from the graph
    independently and share a single :class:`BudgetManager`.

    Startup sequence:

    1. Create ``SNRun`` node and ``BudgetManager``.
    2. **Reconcile-once (B2)** — ``reconcile_standard_name_sources()``
       runs in a worker thread, completing before any pool issues its
       first claim.  This clears stale claims and revives sources
       whose upstream entities reappeared.
    3. Build 6 :class:`PoolSpec` objects (generate_name, review_name,
       refine_name, generate_docs, review_docs, refine_docs) with
       adapter closures and backlog throttle wiring.
    4. Delegate to :func:`~imas_codex.standard_names.pools.run_pools`
       which runs all pools concurrently with cooperative shutdown.
    5. Finalize ``SNRun`` with the actual stop reason and graph-derived
       cost.

    Args:
        cost_limit: Maximum LLM spend in USD.
        min_score: Review threshold.  Names with
            ``reviewer_score_name < min_score`` are routed to the
            refine_name pool; those above are eligible for review.
            Defaults to ``DEFAULT_MIN_SCORE`` when *None*.
        rotation_cap: Maximum REFINED_FROM chain depth before exhaustion.
            Defaults to ``DEFAULT_REFINE_ROTATIONS`` when *None*.
        escalation_model: Higher-capability model for final refine attempt.
            Defaults to ``DEFAULT_ESCALATION_MODEL`` when *None*.
        review_name_backlog_cap: Max pending review_name items before
            generate_name / refine_name pause.  Defaults to
            ``REVIEW_NAME_BACKLOG_CAP`` when *None*.
        review_docs_backlog_cap: Max pending review_docs items before
            generate_docs / refine_docs pause.  Defaults to
            ``REVIEW_DOCS_BACKLOG_CAP`` when *None*.
        source: ``"dd"`` or ``"signals"`` — scopes reconciliation.
        only_domain: When set, restricts the *extract_phase* seeding of
            ``StandardNameSource`` nodes to this physics domain.  The
            pools themselves are domain-agnostic.
        stop_event: Cooperative shutdown signal (set by the CLI harness).
        loop_state: Optional :class:`SNLoopState` for Rich progress.
        pending_fn: Optional callable ``() → dict[str, int]`` mapping
            pool names to pending counts.  When provided, a background
            watchdog polls this every 5 seconds to keep
            ``PoolHealth.pending_count`` current in headless / ``--quiet``
            mode where the Rich display ticker is absent.
    """
    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.pools import run_pools

    started = datetime.now(UTC)
    run_id = str(uuid.uuid4())
    summary = RunSummary(
        run_id=run_id,
        turn_number=1,
        started_at=started,
        cost_limit=cost_limit,
        min_score=min_score,
    )

    if stop_event is None:
        stop_event = asyncio.Event()

    # Shared BudgetManager — all six pools draw from the same pot.
    shared_mgr = BudgetManager(cost_limit, run_id=run_id)
    await shared_mgr.start()

    # Pre-create the SNRun node so LLMCost → FOR_RUN edges have a target.
    from imas_codex.standard_names.graph_ops import create_sn_run_open

    create_sn_run_open(
        run_id,
        started_at=started,
        cost_limit=cost_limit,
        min_score=min_score,
    )

    cost_is_exact = True

    try:
        # ── B2: Reconcile-once-at-startup ─────────────────────────
        # Must complete BEFORE any pool issues its first claim.
        from imas_codex.standard_names.graph_ops import (
            reconcile_standard_name_sources,
        )

        logger.info("run_sn_pools: reconciling sources (source=%s)…", source)
        recon_result = await asyncio.to_thread(reconcile_standard_name_sources, source)
        recon_total = sum(recon_result.values()) if recon_result else 0
        summary.sources_reconciled = recon_total
        logger.info(
            "run_sn_pools: reconcile complete — %d actions (%s)",
            recon_total,
            recon_result,
        )

        # ── B3: Domain extract (optional) ─────────────────────────
        # When only_domain is set, seed the generate_name pool with
        # StandardNameSource nodes for that domain before the pools
        # start.  The pools themselves are domain-agnostic — they
        # claim whatever status='extracted' items already exist —
        # so without this step, --physics-domain is a no-op.
        if only_domain:
            from imas_codex.standard_names.graph_ops import (
                count_extracted_for_domain,
            )

            n_existing = count_extracted_for_domain(only_domain, source)
            logger.info(
                "run_sn_pools: extract-seed domain=%s — %d items already extracted",
                only_domain,
                n_existing,
            )
            if n_existing == 0:
                logger.info(
                    "run_sn_pools: seeding generate_name pool for domain=%s…",
                    only_domain,
                )
                n_seeded = await _seed_domain_sources(
                    domain=only_domain,
                    source=source,
                    stop_event=stop_event,
                )
                logger.info(
                    "run_sn_pools: seeded %d sources for domain=%s",
                    n_seeded,
                    only_domain,
                )

        # ── Build pool specs ──────────────────────────────────────
        specs = _build_pool_specs(
            shared_mgr,
            stop_event,
            min_score=min_score,
            rotation_cap=rotation_cap,
            escalation_model=escalation_model,
            review_name_backlog_cap=review_name_backlog_cap,
            review_docs_backlog_cap=review_docs_backlog_cap,
            on_event=on_event,
            only_domain=only_domain,
        )

        # ── Wire pool health into display state ───────────────────
        if loop_state is not None and hasattr(loop_state, "set_pool_health"):
            for spec in specs:
                loop_state.set_pool_health(spec.name, spec.health)

        # ── Run pools + orphan sweep ──────────────────────────────
        from imas_codex.standard_names.defaults import (
            DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
            DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
        )
        from imas_codex.standard_names.orphan_sweep import run_orphan_sweep_loop

        sweep_task = asyncio.create_task(
            run_orphan_sweep_loop(
                interval_s=DEFAULT_ORPHAN_SWEEP_INTERVAL_S,
                timeout_s=DEFAULT_ORPHAN_SWEEP_TIMEOUT_S,
                stop_event=stop_event,
            ),
            name="orphan_sweep",
        )
        try:
            health_map = await run_pools(
                specs, shared_mgr, stop_event, pending_fn=pending_fn
            )
        finally:
            if not sweep_task.done():
                sweep_task.cancel()
            await asyncio.gather(sweep_task, return_exceptions=True)
        logger.info("run_sn_pools: all pools exited — %s", health_map)

        # Aggregate per-pool processed counts into RunSummary.
        def _total(name: str) -> int:
            h = health_map.get(name)
            return getattr(h, "total_processed", 0) if h is not None else 0

        summary.names_composed = _total("generate_name")
        summary.names_enriched = _total("generate_docs")
        summary.names_reviewed = _total("review_name") + _total("review_docs")
        summary.names_regenerated = _total("refine_name") + _total("refine_docs")

        # ── Determine stop reason ─────────────────────────────────
        # Check exhaustion before stop_event: the budget watchdog sets
        # stop_event when exhausted, so checking stop_event first would
        # misclassify budget-exhausted runs as "interrupted".
        if shared_mgr.hard_exhausted():
            summary.stop_reason = "budget_exhausted"
        elif stop_event.is_set():
            summary.stop_reason = "interrupted"
        else:
            summary.stop_reason = "completed"

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("run_sn_pools interrupted by user")
    except Exception as exc:
        summary.stop_reason = "failed"
        logger.error("run_sn_pools failed: %s", exc, exc_info=True)
    finally:
        summary.ended_at = datetime.now(UTC)

        # Release any orphaned claims left by batches in flight at shutdown.
        try:
            from imas_codex.standard_names.graph_ops import release_all_orphan_claims

            orphan_counts = release_all_orphan_claims()
            if orphan_counts.get("sn", 0) or orphan_counts.get("sns", 0):
                logger.info(
                    "run_sn_pools: orphan sweep released %d SN + %d SNS",
                    orphan_counts.get("sn", 0),
                    orphan_counts.get("sns", 0),
                )
        except Exception as _orphan_exc:  # noqa: BLE001
            logger.warning(
                "run_sn_pools: orphan sweep failed (non-fatal): %s", _orphan_exc
            )

        # Drain pending LLMCost graph writes.
        cost_is_exact = await shared_mgr.drain_pending()
        if not cost_is_exact and summary.stop_reason not in (
            "interrupted",
            "failed",
        ):
            summary.stop_reason = "degraded"

        # Refresh final cost from graph.
        summary.cost_spent = max(
            summary.cost_spent,
            await shared_mgr.get_total_spent(force_refresh=True),
        )

        # Phase-level cost breakdowns.
        phase_spent = shared_mgr.phase_spent
        summary.compose_cost = phase_spent.get("generate_name", 0.0) + phase_spent.get(
            "refine_name", 0.0
        )
        summary.review_cost = phase_spent.get("review_name", 0.0) + phase_spent.get(
            "review_docs", 0.0
        )

        # Compute pipeline hash — best-effort.
        _pipeline_hash: str | None = None
        _pipeline_hash_detail: str | None = None
        try:
            from imas_codex.standard_names.pipeline_version import (
                compute_pipeline_hash,
            )

            ph = compute_pipeline_hash()
            _pipeline_hash = ph["_composite"]
            _pipeline_hash_detail = _json.dumps(
                {k: v for k, v in ph.items() if k != "_composite"}
            )
        except Exception:  # noqa: BLE001
            pass

        # Finalize the SNRun node.
        from imas_codex.standard_names.graph_ops import finalize_sn_run

        finalize_sn_run(
            run_id,
            status=_STOP_TO_STATUS.get(summary.stop_reason, "completed"),
            cost_spent=summary.cost_spent,
            cost_is_exact=cost_is_exact,
            ended_at=summary.ended_at,
            cost_limit=round(summary.cost_limit, 6),
            compose_cost=round(summary.compose_cost, 6),
            review_cost=round(summary.review_cost, 6),
            min_score=summary.min_score,
            names_composed=summary.names_composed,
            names_enriched=summary.names_enriched,
            names_reviewed=summary.names_reviewed,
            names_regenerated=summary.names_regenerated,
            stop_reason=summary.stop_reason,
            pipeline_hash=_pipeline_hash,
            pipeline_hash_detail=_pipeline_hash_detail,
        )

    return summary
