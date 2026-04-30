"""Dispatcher for structured fan-out (plan 39 §4, §10.2).

Public entry point:
    :func:`run_fanout` — ``async`` orchestrator.  Returns the rendered
    markdown evidence block (or ``""`` for any true-no-op outcome).

Three internal stages:
    1. :func:`propose` — Stage A LLM call → :class:`FanoutPlan`.
    2. :func:`execute` — pure-Python parallel runner over Stage A's
       calls.
    3. :func:`render.format_results` — markdown evidence block.

Cost ownership (plan 39 §7.3 I1):
    The Stage A proposer call is charged to the **caller's**
    :class:`BudgetLease` via
    ``parent_lease.charge_event(cost, LLMCostEvent(batch_id=fanout_run_id, …))``.
    The caller's existing Stage C (synthesizer) call charges as today;
    the caller is responsible for stamping ``batch_id=fanout_run_id``
    onto its synthesizer's :class:`LLMCostEvent` so the
    ``Fanout`` ↔ ``LLMCost`` join works (plan 39 §8.3).

Failure modes (plan 39 §7.2):
    Every code path returns ``""`` on failure so the call-site's
    ``{{ fanout_evidence }}`` placeholder collapses to an empty line —
    the "true no-op" semantics.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from imas_codex.standard_names.budget import BudgetLease, LLMCostEvent

from .catalog import get_runner, normalize_query_or_path
from .config import (
    FanoutSettings,
    render_proposer_system_prompt,
)
from .render import format_results
from .schemas import (
    CandidateContext,
    FanoutOutcome,
    FanoutPlan,
    FanoutResult,
    FanoutScope,
)
from .telemetry import write_fanout_node

logger = logging.getLogger(__name__)


# =====================================================================
# Stage A — propose
# =====================================================================


def _build_proposer_user_prompt(
    candidate: CandidateContext,
    reviewer_excerpt: str,
    scope: FanoutScope,
) -> str:
    """Build the dynamic Stage A user prompt body.

    The system prompt (with the ``catalog_version`` line) is rendered
    by :func:`render_proposer_system_prompt`; everything else lives
    here so it stays out of the cached prefix.
    """
    parts = [
        f"Candidate name: {candidate.name}",
        f"DD path: {candidate.path}",
    ]
    if candidate.description:
        parts.append(f"Description: {candidate.description}")
    if candidate.physics_domain:
        parts.append(f"Physics domain: {candidate.physics_domain}")
    if scope.ids_filter:
        parts.append(f"Caller scope — ids_filter: {scope.ids_filter}")
    if scope.physics_domain:
        parts.append(f"Caller scope — physics_domain: {scope.physics_domain}")
    parts.append("")
    parts.append("Recent reviewer feedback (clarity / disambiguation only):")
    parts.append(reviewer_excerpt or "(none)")
    return "\n".join(parts)


async def propose(
    *,
    candidate: CandidateContext,
    reviewer_excerpt: str,
    scope: FanoutScope,
    settings: FanoutSettings,
    parent_lease: BudgetLease,
    fanout_run_id: str,
) -> tuple[FanoutPlan | None, FanoutOutcome | None]:
    """Run Stage A.  Charges proposer cost to ``parent_lease``.

    Returns:
        ``(plan, None)`` on a successfully parsed, non-empty (after
        S1 dedup) plan;
        ``(None, "planner_schema_fail")`` on parse failure;
        ``(None, "planner_all_invalid")`` on empty plan (incl. post-
        dedup).

    The caller turns either ``planner_*`` outcome into ``""`` evidence
    and a Fanout-node write with the matching ``outcome``.
    """
    from imas_codex.discovery.base.llm import acall_llm_structured

    system_prompt = render_proposer_system_prompt()
    user_prompt = _build_proposer_user_prompt(candidate, reviewer_excerpt, scope)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await acall_llm_structured(
            model=settings.proposer_model,
            messages=messages,
            response_model=FanoutPlan,
            temperature=settings.proposer_temperature,
            service="standard-names",
        )
    except Exception as e:
        logger.info(
            "fan-out planner_schema_fail (run_id=%s): %s",
            fanout_run_id,
            e,
            exc_info=True,
        )
        return None, "planner_schema_fail"

    plan: FanoutPlan = result.parsed  # type: ignore[attr-defined]
    cost: float = float(getattr(result, "cost", 0.0))
    tokens_in: int = int(getattr(result, "input_tokens", 0) or 0)
    tokens_out: int = int(getattr(result, "output_tokens", 0) or 0)

    # Charge the proposer cost to the caller's lease.  Sub-event so
    # the analytics query can isolate fan-out spend via batch_id.
    if cost > 0:
        parent_lease.charge_event(
            cost,
            LLMCostEvent(
                model=settings.proposer_model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                sn_ids=(candidate.sn_id,) if candidate.sn_id else (),
                batch_id=fanout_run_id,
                phase="sn_fanout_refine_proposer",
                service="standard-names",
            ),
        )

    # S1 query-side dedup — collapse calls with identical
    # ``(fn_id, normalized_query_or_path)`` to the first occurrence.
    deduped: list = []
    seen: set[tuple[str, str]] = set()
    for call in plan.queries:
        key = (call.fn_id, normalize_query_or_path(call))
        if key in seen:
            logger.debug(
                "fan-out duplicate_query_collapsed: %s (run_id=%s)",
                key,
                fanout_run_id,
            )
            continue
        seen.add(key)
        deduped.append(call)
    if len(deduped) != len(plan.queries):
        plan = FanoutPlan(queries=deduped, notes=plan.notes)

    if not plan.queries:
        return None, "planner_all_invalid"

    return plan, None


# =====================================================================
# Stage B — execute
# =====================================================================


async def execute(
    plan: FanoutPlan,
    *,
    gc: Any,
    scope: FanoutScope,
    settings: FanoutSettings,
) -> list[FanoutResult]:
    """Run Stage B — parallel executor with per-call + total timeouts.

    Wraps the gather in :func:`asyncio.wait_for(total_timeout_s)` so
    even sync helpers that ignore their per-call timeout are bounded
    at the gate (the helper keeps running on the worker thread until
    completion, but its result is discarded — see plan 39 §4.2).
    """
    coros = [
        get_runner(call)(
            call,
            gc=gc,
            scope=scope,
            timeout_s=settings.function_timeout_s,
        )
        for call in plan.queries
    ]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=settings.total_timeout_s,
        )
        return list(results)
    except TimeoutError:
        logger.info("fan-out total_timeout_s tripped (%.1fs)", settings.total_timeout_s)
        # Synthesise per-call failure results so render gets a consistent shape.
        return [
            FanoutResult(
                fn_id=call.fn_id,
                args=call.model_dump(),
                ok=False,
                error="total_timeout",
            )
            for call in plan.queries
        ]


def _classify_outcome(results: list[FanoutResult]) -> FanoutOutcome:
    """Classify a list of executor results into a :data:`FanoutOutcome`."""
    successful = [r for r in results if r.ok]
    if not successful:
        # All runners failed — treat as partial_fail with zero successes.
        return "executor_partial_fail"
    if all(not r.hits for r in successful):
        return "executor_all_empty"
    if len(successful) < len(results):
        return "executor_partial_fail"
    return "ok"


# =====================================================================
# Public entry point — run_fanout
# =====================================================================


async def run_fanout(
    *,
    site: str,
    candidate: CandidateContext,
    reviewer_excerpt: str,
    scope: FanoutScope,
    gc: Any,
    parent_lease: BudgetLease,
    settings: FanoutSettings,
    arm: str = "on",
    escalate: bool = False,
    fanout_run_id: str | None = None,
) -> str:
    """Run fan-out for a single refine cycle.  Returns evidence string.

    Args:
        site: Plug-in site name (MVP: ``"refine_name"``).  Recorded on
            the Fanout telemetry node.
        candidate: Refine-site candidate metadata.
        reviewer_excerpt: Pre-truncated reviewer-comment slice (caller
            applies the dim allow-list and char cap from plan 39 §5.1
            before passing it in — fan-out itself is site-agnostic).
        scope: Caller-injected scope (never LLM-supplied).
        gc: Refine cycle's :class:`GraphClient` (one per cycle, plan
            39 §10.1).
        parent_lease: Caller's :class:`BudgetLease`; proposer cost is
            charged here as a sub-event.
        settings: Loaded :class:`FanoutSettings`.
        arm: ``"on"`` or ``"off"`` for the within-cohort A/B (plan 39
            §8.4).  ``"off"`` is a true no-op that still writes a
            ``Fanout`` node with ``outcome="off_arm"`` so the
            denominator is queryable.
        escalate: Whether the caller's cycle is on the escalation
            tier (plan 39 §7.3).  Selects spend cap + evidence token
            cap.
        fanout_run_id: Override for testing / reproducibility.  When
            ``None`` a uuid4 is generated.

    Returns:
        Markdown evidence block, or ``""`` for any true-no-op outcome.
    """
    if fanout_run_id is None:
        fanout_run_id = str(uuid.uuid4())

    # ── Master switch + per-site enable flag ─────────────────────────
    if not settings.enabled:
        return ""
    if not settings.sites.get(site, False):
        return ""

    # ── Within-cohort A/B: off-arm true no-op + telemetry node ──────
    if arm == "off":
        write_fanout_node(
            gc,
            run_id=fanout_run_id,
            sn_id=candidate.sn_id,
            site=site,
            outcome="off_arm",
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
            arm="off",
            escalate=escalate,
        )
        return ""

    # ── Pre-flight budget check (plan 39 §7.2 no_budget) ─────────────
    cap = settings.cap_for_charge(escalate=escalate)
    # Snapshot how much has already been charged to the parent lease
    # before fan-out starts; subsequent fan-out charges are reflected
    # in the lease's running ``charged`` figure.  This is a soft
    # pre-flight gate, not a hard ceiling.
    pre_charged = parent_lease.charged
    if cap <= 0:
        write_fanout_node(
            gc,
            run_id=fanout_run_id,
            sn_id=candidate.sn_id,
            site=site,
            outcome="no_budget",
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
            arm=arm,
            escalate=escalate,
        )
        return ""

    # ── Stage A: proposer ────────────────────────────────────────────
    plan, planner_failure = await propose(
        candidate=candidate,
        reviewer_excerpt=reviewer_excerpt,
        scope=scope,
        settings=settings,
        parent_lease=parent_lease,
        fanout_run_id=fanout_run_id,
    )
    if planner_failure is not None:
        write_fanout_node(
            gc,
            run_id=fanout_run_id,
            sn_id=candidate.sn_id,
            site=site,
            outcome=planner_failure,
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
            arm=arm,
            escalate=escalate,
        )
        return ""

    assert plan is not None  # narrow for type-checkers

    # ── Post-proposer budget check ──────────────────────────────────
    fanout_charge = parent_lease.charged - pre_charged
    if fanout_charge > cap:
        write_fanout_node(
            gc,
            run_id=fanout_run_id,
            sn_id=candidate.sn_id,
            site=site,
            outcome="no_budget",
            plan_size=len(plan.queries),
            hits_count=0,
            evidence_tokens=0,
            arm=arm,
            escalate=escalate,
        )
        return ""

    # ── Stage B: execute ────────────────────────────────────────────
    results = await execute(plan, gc=gc, scope=scope, settings=settings)
    outcome = _classify_outcome(results)

    # ── Render ──────────────────────────────────────────────────────
    evidence = ""
    if outcome != "executor_all_empty":
        evidence = format_results(
            results,
            result_hit_cap=settings.result_hit_cap,
            evidence_token_cap=settings.evidence_token_cap_for(escalate=escalate),
        )

    hits_count = sum(len(r.hits) for r in results if r.ok)
    evidence_tokens = max(1, len(evidence) // 4) if evidence else 0
    write_fanout_node(
        gc,
        run_id=fanout_run_id,
        sn_id=candidate.sn_id,
        site=site,
        outcome=outcome,
        plan_size=len(plan.queries),
        hits_count=hits_count,
        evidence_tokens=evidence_tokens,
        arm=arm,
        escalate=escalate,
    )

    return evidence


__all__ = [
    "run_fanout",
    "propose",
    "execute",
]
