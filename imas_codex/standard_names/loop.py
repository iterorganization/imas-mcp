"""SN loop — drives ``sn run`` with one-domain-per-turn rotation.

Picks the stalest extract-eligible physics domain via stale-first
rotation (oldest ``SNRun.ended_at`` wins, with eligible-source count as
tiebreak), runs one :func:`run_turn` on it with the full remaining
budget, and repeats until the remaining budget can no longer fund a
single unit of work or no domain has eligible work.

Entry point: :func:`run_sn_loop`.
"""

from __future__ import annotations

import json as _json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

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
        RETURN sn.physics_domain AS domain, count(*) AS remaining
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
    progress_display: Any | None = None,
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
        progress_display: Optional :class:`SNLoopProgressDisplay` for
            Rich live monitoring.  When ``None``, progress is logged
            via the standard logger (plain mode).
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

    # ── Display: signal run start ─────────────────────────────────
    if progress_display is not None:
        # Count eligible domains for the domains tracker.
        # Best-effort: don't let display setup block the pipeline.
        try:
            _eligible = _count_eligible_domains(only_domain=only_domain)
            progress_display.start_run(total_domains=len(_eligible))
        except Exception:
            progress_display.start_run(total_domains=0)

    try:
        while True:
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

            # ── Display: signal turn start ────────────────────────
            if progress_display is not None:
                # Build phase plan from skip flags
                _phase_plan = []
                for _pname, _skip in [
                    ("reconcile", False),
                    ("generate", skip_generate),
                    ("enrich", skip_enrich),
                    ("link", False),
                    ("review_names", skip_review),
                    ("review_docs", skip_review),
                    ("regen", skip_regen or min_score is None),
                ]:
                    _phase_plan.append(_pname)
                progress_display.start_turn(domain=dom, phase_plan=_phase_plan)

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
                progress_display=progress_display,
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
                elif phase.name == "enrich":
                    summary.names_enriched += phase.count
                elif phase.name in ("review_names", "review_docs"):
                    summary.names_reviewed += phase.count
                elif phase.name == "regen":
                    summary.names_regenerated += phase.count
                elif phase.name == "reconcile":
                    summary.sources_reconciled += phase.count
                elif phase.name == "link":
                    summary.links_resolved += phase.count

            summary.cost_spent = max(shared_mgr.spent, summary.cost_spent + phase_sum)
            summary.domains_touched.add(dom)

            # ── Display: update phase results and signal turn end ──
            if progress_display is not None:
                for phase in results:
                    if phase.skipped:
                        progress_display.end_phase(phase.name, status="skipped")
                    else:
                        progress_display.update_phase(
                            phase.name,
                            completed=phase.count,
                            cost=phase.cost,
                        )
                        progress_display.end_phase(phase.name, status="completed")
                progress_display.end_turn(domain=dom)

            # Update phase-level cost breakdowns from shared manager.
            phase_spent = shared_mgr.phase_spent
            summary.compose_cost = phase_spent.get("generate", 0.0) + phase_spent.get(
                "regen", 0.0
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
