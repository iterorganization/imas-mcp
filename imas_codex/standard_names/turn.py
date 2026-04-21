"""Per-domain turn orchestrator for ``sn run``.

Chains one full quality-improvement cycle for a single physics domain::

    generate → enrich → review → regen

Each phase delegates to the same library functions used by the individual
``sn run``, ``sn enrich``, and ``sn review`` CLI commands — no pipeline
logic is duplicated.

The turn is identified by a UUID (``run_id``) stamped onto every
StandardName node produced or regenerated during the cycle, along with a
``turn_number`` that increments across successive ``sn run
--turn-number N`` invocations.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default cost-budget split across the four phases.
# Values must sum to 1.0.
TURN_SPLIT: tuple[float, float, float, float] = (0.40, 0.20, 0.20, 0.20)


@dataclass
class PhaseResult:
    """Outcome of a single phase within one turn."""

    name: str
    exit_code: int = 0
    cost: float = 0.0
    elapsed: float = 0.0
    count: int = 0
    error: str | None = None
    skipped: bool = False


@dataclass
class TurnConfig:
    """Configuration for a single turn (one domain × four phases)."""

    domain: str
    cost_limit: float = 5.0
    limit: int | None = None
    concurrency: int = 2
    dry_run: bool = False
    fail_fast: bool = False
    skip_generate: bool = False
    skip_enrich: bool = False
    skip_review: bool = False
    skip_regen: bool = False
    split: tuple[float, float, float, float] = TURN_SPLIT
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int = 1
    min_score: float | None = None

    def phase_budget(self, index: int) -> float:
        """Return the cost budget allocated to phase *index* (0–3)."""
        return self.cost_limit * self.split[index]


async def _run_generate_phase(
    cfg: TurnConfig,
    *,
    regen: bool = False,
    force: bool = False,
) -> PhaseResult:
    """Run the generate (or regen) pipeline phase.

    Constructs a :class:`StandardNameBuildState` and invokes
    :func:`run_sn_pipeline` — the same function that ``sn run``
    delegates to. When ``regen=True`` the state is configured to
    select existing reviewed names whose ``reviewer_score`` is below
    ``cfg.min_score``.
    """
    phase_name = "regen" if regen else "generate"
    phase_idx = 3 if regen else 0
    budget = cfg.phase_budget(phase_idx)

    if cfg.dry_run:
        return PhaseResult(
            name=phase_name,
            skipped=False,
            cost=0.0,
            count=0,
        )

    from imas_codex.standard_names.pipeline import run_sn_pipeline
    from imas_codex.standard_names.state import StandardNameBuildState

    state = StandardNameBuildState(
        facility="dd",
        source="dd",
        domain_filter=cfg.domain,
        cost_limit=budget,
        dry_run=False,
        force=force,
        regen=regen,
        min_score=cfg.min_score if regen else None,
        turn_number=cfg.turn_number,
        run_id=cfg.run_id,
        limit=cfg.limit,
    )

    t0 = time.monotonic()
    try:
        await run_sn_pipeline(state)
    except Exception as exc:
        logger.error("Phase %s failed: %s", phase_name, exc, exc_info=True)
        return PhaseResult(
            name=phase_name,
            exit_code=1,
            cost=state.total_cost,
            elapsed=time.monotonic() - t0,
            count=state.stats.get("compose_count", 0),
            error=str(exc),
        )
    elapsed = time.monotonic() - t0

    compose_count = state.stats.get("compose_count", 0)
    compose_cost = state.stats.get("compose_cost", 0.0)

    # Stamp run provenance on produced names
    persisted_ids: list[str] = [
        n.get("id", "") for n in state.consolidated if n.get("id")
    ]
    if persisted_ids:
        from imas_codex.standard_names.graph_ops import write_run_provenance

        await asyncio.to_thread(
            write_run_provenance, persisted_ids, cfg.run_id, cfg.turn_number
        )

    return PhaseResult(
        name=phase_name,
        cost=compose_cost,
        elapsed=elapsed,
        count=compose_count,
    )


async def _run_enrich_phase(cfg: TurnConfig) -> PhaseResult:
    """Run the enrich pipeline phase.

    Constructs a :class:`StandardNameEnrichState` targeting
    ``validation_status='valid'`` names in the given domain that are
    still at ``review_status='named'`` (missing enrichment).
    """
    budget = cfg.phase_budget(1)

    if cfg.dry_run:
        return PhaseResult(name="enrich", skipped=False, cost=0.0, count=0)

    from imas_codex.standard_names.enrich_pipeline import run_sn_enrich_engine
    from imas_codex.standard_names.enrich_state import StandardNameEnrichState

    state = StandardNameEnrichState(
        facility="dd",
        domain=[cfg.domain],
        status_filter=["named"],
        cost_limit=budget,
        limit=cfg.limit,
        dry_run=False,
        force=False,
    )

    t0 = time.monotonic()
    try:
        await run_sn_enrich_engine(state)
    except Exception as exc:
        logger.error("Phase enrich failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="enrich",
            exit_code=1,
            cost=state.total_cost,
            elapsed=time.monotonic() - t0,
            count=state.stats.get("persist_written", 0),
            error=str(exc),
        )
    elapsed = time.monotonic() - t0

    return PhaseResult(
        name="enrich",
        cost=state.total_cost,
        elapsed=elapsed,
        count=state.stats.get("persist_written", 0),
    )


async def _run_review_phase(cfg: TurnConfig) -> PhaseResult:
    """Run the review pipeline phase.

    Reviews ``valid`` names in the given domain that are unreviewed or
    have stale review hashes. Writes reviewer scores and comments onto
    each name without status demotion — low-score selection for regen
    is handled by the regen phase via ``--min-score``.
    """
    budget = cfg.phase_budget(2)

    if cfg.dry_run:
        return PhaseResult(name="review", skipped=False, cost=0.0, count=0)

    from imas_codex.standard_names.review.budget import ReviewBudgetManager
    from imas_codex.standard_names.review.consolidation import run_consolidation
    from imas_codex.standard_names.review.pipeline import run_sn_review_engine
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(
        facility="dd",
        cost_limit=budget,
        domain_filter=cfg.domain,
        unreviewed_only=True,
        skip_audit=True,
        concurrency=cfg.concurrency,
        dry_run=False,
        budget_manager=ReviewBudgetManager(budget),
    )

    t0 = time.monotonic()
    try:
        stop_event = asyncio.Event()
        await run_sn_review_engine(state, stop_event=stop_event)
        run_consolidation(state)
    except Exception as exc:
        logger.error("Phase review failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="review",
            exit_code=1,
            cost=state.total_cost,
            elapsed=time.monotonic() - t0,
            count=state.stats.get("persist_count", 0),
            error=str(exc),
        )
    elapsed = time.monotonic() - t0

    return PhaseResult(
        name="review",
        cost=state.total_cost,
        elapsed=elapsed,
        count=state.stats.get("persist_count", 0),
    )


async def run_turn(cfg: TurnConfig) -> list[PhaseResult]:
    """Execute one full turn (generate → enrich → review → regen).

    Runs the four phases in sequence, respecting skip flags. Returns
    a list of :class:`PhaseResult` for every phase (including skipped
    ones).
    """
    results: list[PhaseResult] = []
    phases: list[tuple[str, bool, Any]] = [
        ("generate", cfg.skip_generate, lambda: _run_generate_phase(cfg)),
        ("enrich", cfg.skip_enrich, lambda: _run_enrich_phase(cfg)),
        ("review", cfg.skip_review, lambda: _run_review_phase(cfg)),
        (
            "regen",
            cfg.skip_regen,
            lambda: _run_generate_phase(cfg, regen=True, force=True),
        ),
    ]

    for name, skip, fn in phases:
        if skip:
            results.append(PhaseResult(name=name, skipped=True))
            logger.info("Skipping phase: %s", name)
            continue

        logger.info("Starting phase: %s", name)
        result = await fn()
        results.append(result)

        if result.exit_code != 0 and cfg.fail_fast:
            logger.error(
                "Phase %s failed (exit_code=%d), --fail-fast: aborting turn",
                name,
                result.exit_code,
            )
            # Mark remaining phases as skipped
            for remaining_name, _, _ in phases[phases.index((name, skip, fn)) + 1 :]:
                results.append(PhaseResult(name=remaining_name, skipped=True))
            break

    return results


def turn_summary(
    results: list[PhaseResult],
    cfg: TurnConfig,
) -> dict[str, Any]:
    """Build a summary dict suitable for rich printing.

    Returns:
        Dict with per-phase and aggregate metrics.
    """
    total_cost = sum(r.cost for r in results)
    total_elapsed = sum(r.elapsed for r in results)
    total_count = sum(r.count for r in results if not r.skipped)
    max_exit = max((r.exit_code for r in results), default=0)
    errors = [r for r in results if r.error]

    return {
        "run_id": cfg.run_id,
        "turn_number": cfg.turn_number,
        "domain": cfg.domain,
        "phases": [
            {
                "name": r.name,
                "skipped": r.skipped,
                "exit_code": r.exit_code,
                "cost": r.cost,
                "elapsed": r.elapsed,
                "count": r.count,
                "error": r.error,
                "budget": cfg.phase_budget(i),
            }
            for i, r in enumerate(results)
        ],
        "total_cost": total_cost,
        "total_elapsed": total_elapsed,
        "total_count": total_count,
        "cost_limit": cfg.cost_limit,
        "exit_code": max_exit,
        "errors": [{"phase": e.name, "error": e.error} for e in errors],
        "dry_run": cfg.dry_run,
    }
