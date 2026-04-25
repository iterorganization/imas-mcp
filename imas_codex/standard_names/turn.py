"""Per-domain turn orchestrator for ``sn run``.

Chains one full quality-improvement cycle for a single physics domain::

    reconcile → generate → enrich → link → review_names → review_docs → regen

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

# Default cost-budget split across the five LLM phases.
# Values must sum to 1.0.
# Phases: generate (30%), enrich (25%), review_names (15%), review_docs (15%), regen (15%).
#
# TURN_SPLIT_LEGACY — original allocation; preserved for back-compat and as
#   the active default until Phase A (lean compose prompt) lands.
# TURN_SPLIT_LEAN — review-heavy allocation activated by TurnConfig.compose_lean=True.
#   Shifts budget from over-provisioned compose/enrich toward starved review phases.
#   Phases: generate (15%), enrich (10%), review_names (30%), review_docs (30%), regen (15%).
TURN_SPLIT_LEGACY: tuple[float, float, float, float, float] = (
    0.30,
    0.25,
    0.15,
    0.15,
    0.15,
)
TURN_SPLIT_LEAN: tuple[float, float, float, float, float] = (
    0.15,
    0.10,
    0.30,
    0.30,
    0.15,
)
# Backward-compat alias — always points to the legacy split.  Callers that
# construct TurnConfig should prefer TurnConfig.split (or compose_lean).
TURN_SPLIT: tuple[float, float, float, float, float] = TURN_SPLIT_LEGACY

# Valid --only phase choices (CLI enforces this set).
TURN_PHASES: tuple[str, ...] = (
    "reconcile",
    "extract",
    "compose",
    "validate",
    "consolidate",
    "persist",
    "review",
    "review_names",
    "review_docs",
    "link",
)

# Maps an --only value to the set of turn-level phases to keep running.
# Everything outside the set is skipped.
_ONLY_TO_ACTIVE: dict[str, set[str]] = {
    "reconcile": {"reconcile"},
    "extract": {"generate"},
    "compose": {"generate"},
    "validate": {"generate"},
    "consolidate": {"generate"},
    "persist": {"generate"},
    "review": {"review_names", "review_docs"},
    "review_names": {"review_names"},
    "review_docs": {"review_docs"},
    "link": {"link"},
}

# Max link-resolution iterations per turn.
_MAX_RESOLVE_ROUNDS = 3
_RESOLVE_BATCH_LIMIT = 50


def _active_split(
    config: TurnConfig,
) -> tuple[float, float, float, float, float]:
    """Return the TURN_SPLIT tuple appropriate for *config*.

    When ``config.compose_lean`` is ``True`` (Phase A lean-prompt flag), the
    review-heavy ``TURN_SPLIT_LEAN`` is used.  Otherwise ``TURN_SPLIT_LEGACY``
    is returned unchanged, preserving production behaviour until Phase A lands.
    """
    return TURN_SPLIT_LEAN if config.compose_lean else TURN_SPLIT_LEGACY


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
    touched_names: list[str] = field(default_factory=list)


@dataclass
class TurnConfig:
    """Configuration for a single turn (one domain × seven phases).

    Phase budget split (5 LLM phases sharing ``cost_limit``):

    Legacy split (``compose_lean=False``, default):
      - generate: 30%
      - enrich: 25%
      - review_names: 15%
      - review_docs: 15%
      - regen: 15%

    Lean split (``compose_lean=True``, activated with Phase A):
      - generate: 15%
      - enrich: 10%
      - review_names: 30%
      - review_docs: 30%
      - regen: 15%

    Non-LLM phases (reconcile, link) have zero cost.

    ``compose_lean`` is the Phase A/B coordination flag.  Set it to
    ``True`` only after the lean compose prompt lands; ``False`` preserves
    legacy behaviour so Phase B merges safely before Phase A.
    """

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
    only: str | None = None
    compose_lean: bool = False
    split: tuple[float, float, float, float, float] = field(
        default_factory=lambda: TURN_SPLIT_LEGACY
    )
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turn_number: int = 1
    min_score: float | None = None
    source: str = "dd"
    override_edits: list[str] | None = None

    def __post_init__(self) -> None:
        # Always derive ``split`` from ``compose_lean`` so the two stay in sync.
        # This ensures production behaviour is unchanged (compose_lean=False)
        # until Phase A enables lean prompts.
        self.split = _active_split(self)

    def phase_budget(self, index: int) -> float:
        """Return the cost budget allocated to LLM phase *index* (0–4).

        The five LLM phases (generate, enrich, review_names, review_docs,
        regen) share the cost budget.  Non-LLM phases (reconcile, link)
        have zero cost.
        """
        return self.cost_limit * self.split[index]


# ── Phase implementations ─────────────────────────────────────────────


async def _run_reconcile_phase(cfg: TurnConfig) -> PhaseResult:
    """Reconcile StandardNameSource nodes after upstream DD/signal rebuild.

    Re-links sources, marks stale, and revives previously-stale sources
    that reappear.  Scoped by ``cfg.source`` (``dd`` or ``signals``).
    """
    if cfg.dry_run:
        return PhaseResult(name="reconcile", count=0)

    from imas_codex.standard_names.graph_ops import reconcile_standard_name_sources

    t0 = time.monotonic()
    try:
        result = await asyncio.to_thread(reconcile_standard_name_sources, cfg.source)
    except Exception as exc:
        logger.error("Phase reconcile failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="reconcile",
            exit_code=1,
            elapsed=time.monotonic() - t0,
            error=str(exc),
        )

    total = sum(result.values())
    return PhaseResult(
        name="reconcile",
        elapsed=time.monotonic() - t0,
        count=total,
    )


async def _run_generate_phase(
    cfg: TurnConfig,
    *,
    regen: bool = False,
    force: bool = False,
    budget_override: float | None = None,
) -> PhaseResult:
    """Run the generate (or regen) pipeline phase.

    Constructs a :class:`StandardNameBuildState` and invokes
    :func:`run_sn_pipeline` — the same function that ``sn run``
    delegates to. When ``regen=True`` the state is configured to
    select existing reviewed names whose ``reviewer_score`` is below
    ``cfg.min_score``.

    Args:
        cfg: Turn configuration.
        regen: If True, regenerate low-scoring names.
        force: If True, regenerate even without stale sources.
        budget_override: When set, use this budget instead of the fixed
            phase split.  Used by :func:`run_turn` for adaptive regen budget.
    """
    phase_name = "regen" if regen else "generate"
    phase_idx = 4 if regen else 0
    budget = (
        budget_override if budget_override is not None else cfg.phase_budget(phase_idx)
    )

    if cfg.dry_run:
        return PhaseResult(
            name=phase_name,
            skipped=False,
            cost=0.0,
            count=0,
        )

    from imas_codex.standard_names.budget import BudgetManager
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
        budget_manager=BudgetManager(budget),
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
        touched_names=persisted_ids,
    )


async def _run_enrich_phase(cfg: TurnConfig) -> PhaseResult:
    """Run the enrich pipeline phase.

    Constructs a :class:`StandardNameEnrichState` targeting
    ``validation_status='valid'`` names in the given domain that are
    still at ``pipeline_status='named'`` (missing enrichment).
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


async def _run_link_phase(
    cfg: TurnConfig,
    touched_names: set[str],
) -> PhaseResult:
    """Resolve ``dd:`` links to ``name:`` links on touched names.

    Multi-round: loops until no unresolved links remain or
    ``_MAX_RESOLVE_ROUNDS`` iterations elapse.  When *touched_names*
    is empty (e.g. ``--only link``), falls back to a global
    sweep of all unresolved names.
    """
    if cfg.dry_run:
        return PhaseResult(name="link", count=0)

    from imas_codex.standard_names.graph_ops import resolve_links_batch

    override_names = set(cfg.override_edits) if cfg.override_edits else None

    total_resolved = 0
    total_unresolved = 0
    total_failed = 0

    t0 = time.monotonic()
    try:
        for round_num in range(1, _MAX_RESOLVE_ROUNDS + 1):
            items = await asyncio.to_thread(
                _fetch_unresolved_links,
                touched_names if touched_names else None,
                _RESOLVE_BATCH_LIMIT,
            )
            if not items:
                break

            result = await asyncio.to_thread(
                resolve_links_batch,
                items,
                override_names=override_names,
            )
            total_resolved += result["resolved"]
            total_unresolved += result["unresolved"]
            total_failed += result["failed"]

            logger.info(
                "link round %d: %d resolved, %d unresolved, %d failed",
                round_num,
                result["resolved"],
                result["unresolved"],
                result["failed"],
            )

            if result["unresolved"] == 0:
                break
    except Exception as exc:
        logger.error("Phase link failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="link",
            exit_code=1,
            elapsed=time.monotonic() - t0,
            count=total_resolved,
            error=str(exc),
        )

    return PhaseResult(
        name="link",
        elapsed=time.monotonic() - t0,
        count=total_resolved,
    )


async def _run_review_names_phase(
    cfg: TurnConfig,
    budget_override: float | None = None,
) -> PhaseResult:
    """Run the name-review pipeline phase.

    Reviews ``valid`` names in the given domain using the 4-dim name rubric.
    Writes ``reviewer_score_name`` + ``reviewed_name_at``.  Bootstraps
    ``reviewer_score`` from ``reviewer_score_name`` when null.

    Args:
        cfg: Turn configuration.
        budget_override: When set, use this budget instead of the fixed
            phase split.  Used by :func:`run_turn` for adaptive budgets.
    """
    budget = budget_override if budget_override is not None else cfg.phase_budget(2)

    if cfg.dry_run:
        return PhaseResult(name="review_names", skipped=False, cost=0.0, count=0)

    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.review.consolidation import run_consolidation
    from imas_codex.standard_names.review.pipeline import run_sn_review_engine
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(
        facility="dd",
        cost_limit=budget,
        domain_filter=cfg.domain,
        status_filter=None,
        unreviewed_only=True,
        skip_audit=True,
        concurrency=cfg.concurrency,
        dry_run=False,
        target="names",
        budget_manager=BudgetManager(budget),
    )

    t0 = time.monotonic()
    try:
        stop_event = asyncio.Event()
        await run_sn_review_engine(state, stop_event=stop_event)
        run_consolidation(state)
    except Exception as exc:
        logger.error("Phase review_names failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="review_names",
            exit_code=1,
            cost=state.total_cost,
            elapsed=time.monotonic() - t0,
            count=state.stats.get("persist_count", 0),
            error=str(exc),
        )
    elapsed = time.monotonic() - t0

    persist_count = state.stats.get("persist_count", 0)

    # Invariant: if eligible names were identified but nothing was persisted
    # and we are not budget-exhausted, something silently failed.
    budget_blocked = state.stats.get("budget_reservation_blocked", 0) > 0
    if (
        len(state.target_names) > 0
        and persist_count == 0
        and state.total_cost < budget * 0.5
        and not budget_blocked
    ):
        msg = (
            f"invariant violated: {len(state.target_names)} eligible names "
            "identified but zero persisted (not budget-exhausted)"
        )
        logger.error("Phase review_names %s", msg)
        return PhaseResult(
            name="review_names",
            exit_code=1,
            cost=state.total_cost,
            elapsed=elapsed,
            count=0,
            error=msg,
        )

    return PhaseResult(
        name="review_names",
        cost=state.total_cost,
        elapsed=elapsed,
        count=persist_count,
    )


async def _run_review_docs_phase(
    cfg: TurnConfig,
    budget_override: float | None = None,
) -> PhaseResult:
    """Run the docs-review pipeline phase.

    Reviews ``valid`` names whose ``reviewed_name_at IS NOT NULL`` using
    the 4-dim docs rubric.  Writes ``reviewer_score_docs`` +
    ``reviewed_docs_at``.  Never touches ``reviewer_score``.

    Args:
        cfg: Turn configuration.
        budget_override: When set, use this budget instead of the fixed
            phase split.  Used by :func:`run_turn` for adaptive budgets.
    """
    budget = budget_override if budget_override is not None else cfg.phase_budget(3)

    if cfg.dry_run:
        return PhaseResult(name="review_docs", skipped=False, cost=0.0, count=0)

    from imas_codex.standard_names.budget import BudgetManager
    from imas_codex.standard_names.review.consolidation import run_consolidation
    from imas_codex.standard_names.review.pipeline import run_sn_review_engine
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(
        facility="dd",
        cost_limit=budget,
        domain_filter=cfg.domain,
        status_filter=None,
        unreviewed_only=True,
        skip_audit=True,
        concurrency=cfg.concurrency,
        dry_run=False,
        target="docs",
        budget_manager=BudgetManager(budget),
    )

    t0 = time.monotonic()
    try:
        stop_event = asyncio.Event()
        await run_sn_review_engine(state, stop_event=stop_event)
        run_consolidation(state)
    except Exception as exc:
        logger.error("Phase review_docs failed: %s", exc, exc_info=True)
        return PhaseResult(
            name="review_docs",
            exit_code=1,
            cost=state.total_cost,
            elapsed=time.monotonic() - t0,
            count=state.stats.get("persist_count", 0),
            error=str(exc),
        )
    elapsed = time.monotonic() - t0

    persist_count = state.stats.get("persist_count", 0)
    docs_skipped = state.stats.get("docs_skipped_missing_name", 0)

    if docs_skipped > 0:
        logger.info(
            "Phase review_docs: %d names skipped (reviewed_name_at IS NULL)",
            docs_skipped,
        )

    # Invariant: if eligible names were identified but nothing was persisted
    # and we are not budget-exhausted, something silently failed.
    budget_blocked = state.stats.get("budget_reservation_blocked", 0) > 0
    if (
        len(state.target_names) > 0
        and persist_count == 0
        and state.total_cost < budget * 0.5
        and not budget_blocked
    ):
        msg = (
            f"invariant violated: {len(state.target_names)} eligible names "
            "identified but zero persisted (not budget-exhausted)"
        )
        logger.error("Phase review_docs %s", msg)
        return PhaseResult(
            name="review_docs",
            exit_code=1,
            cost=state.total_cost,
            elapsed=elapsed,
            count=0,
            error=msg,
        )

    return PhaseResult(
        name="review_docs",
        cost=state.total_cost,
        elapsed=elapsed,
        count=persist_count,
    )


# Keep _run_review_phase as a back-compat alias so existing callers
# (including test_turn_review_gate.py and test_run_provenance.py) still work.
_run_review_phase = _run_review_names_phase


# ── Helpers ────────────────────────────────────────────────────────────

# Budget shares for trailing phases, keyed by phase name.
# These are fractions of the *remaining* budget after prior phases,
# not fractions of the total cost_limit.
_ADAPTIVE_SHARES: dict[str, float] = {
    "review_names": 0.45,
    "review_docs": 0.35,
    "regen": 1.00,  # regen gets everything that's left
}


def _adaptive_review_budget(
    cost_limit: float,
    prior_results: list[PhaseResult],
    phase_name: str,
) -> float:
    """Compute an adaptive budget for a trailing phase.

    Instead of a fixed percentage split, trailing phases (review_names,
    review_docs, regen) get a share of the *remaining* budget after
    subtracting actual spend from prior phases.

    This prevents the common scenario where generate spends $0 (all
    names already composed) but its 30% allocation is wasted, leaving
    review with only 15% of the total.

    Args:
        cost_limit: Total turn budget.
        prior_results: Results from phases that have already run.
        phase_name: Which trailing phase to compute budget for.

    Returns:
        Dollar budget for the phase.
    """
    prior_spend = sum(r.cost for r in prior_results if not r.skipped)
    remaining = max(cost_limit - prior_spend, 0.0)
    share = _ADAPTIVE_SHARES.get(phase_name, 0.15)
    return remaining * share


def _fetch_unresolved_links(
    name_ids: set[str] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch StandardName nodes with unresolved links.

    When *name_ids* is provided, scopes to only those names.
    When ``None``, performs a global sweep.
    """
    from imas_codex.graph.client import GraphClient

    with GraphClient() as gc:
        if name_ids:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.id IN $names
                  AND sn.link_status = 'unresolved'
                RETURN sn.id AS id, sn.links AS links,
                       coalesce(sn.link_retry_count, 0) AS retry_count
                LIMIT $limit
                """,
                names=list(name_ids),
                limit=limit,
            )
        else:
            rows = gc.query(
                """
                MATCH (sn:StandardName)
                WHERE sn.link_status = 'unresolved'
                RETURN sn.id AS id, sn.links AS links,
                       coalesce(sn.link_retry_count, 0) AS retry_count
                LIMIT $limit
                """,
                limit=limit,
            )
    return [dict(r) for r in rows]


def skip_flags_from_only(only_phase: str | None) -> dict[str, bool]:
    """Derive per-phase skip flags from an ``--only`` selection.

    Returns a dict of ``skip_*`` keys (matching :class:`TurnConfig`
    field names) that should be set to ``True`` when *only_phase* is
    active.  When *only_phase* is ``None``, returns an empty dict
    (no overrides).
    """
    if only_phase is None:
        return {}

    active = _ONLY_TO_ACTIVE.get(only_phase, set())
    return {
        "skip_generate": "generate" not in active,
        "skip_enrich": "generate" not in active,  # enrich follows generate
        "skip_review": "review_names" not in active and "review_docs" not in active,
        "skip_regen": "generate" not in active,
    }


# ── Turn runner ────────────────────────────────────────────────────────


async def run_turn(cfg: TurnConfig) -> list[PhaseResult]:
    """Execute one full turn (reconcile → generate → enrich → link → review_names → review_docs → regen).

    Runs the seven phases in sequence, respecting skip flags.  Returns
    a list of :class:`PhaseResult` for every phase (including skipped
    ones).  Names created/updated by generate are tracked and passed
    to the link phase to scope link resolution.
    """
    results: list[PhaseResult] = []
    touched_names: set[str] = set()

    # Determine which phases to skip.  reconcile and link are
    # always active unless ``--only`` restricts to a different phase.
    _only_active = _ONLY_TO_ACTIVE.get(cfg.only, set()) if cfg.only else None
    _skip_reconcile = _only_active is not None and "reconcile" not in _only_active
    _skip_link = _only_active is not None and "link" not in _only_active
    _skip_review_names = cfg.skip_review or (
        _only_active is not None and "review_names" not in _only_active
    )
    _skip_review_docs = cfg.skip_review or (
        _only_active is not None and "review_docs" not in _only_active
    )

    phases: list[tuple[str, bool, Any]] = [
        ("reconcile", _skip_reconcile, lambda: _run_reconcile_phase(cfg)),
        ("generate", cfg.skip_generate, lambda: _run_generate_phase(cfg)),
        ("enrich", cfg.skip_enrich, lambda: _run_enrich_phase(cfg)),
        (
            "link",
            _skip_link,
            lambda: _run_link_phase(cfg, touched_names),
        ),
        # Trailing phases use adaptive budgets computed from actual prior spend.
        # The lambda captures `results` by reference, so it sees the current
        # state when called (after prior phases have completed).
        (
            "review_names",
            _skip_review_names,
            lambda: _run_review_names_phase(
                cfg,
                budget_override=_adaptive_review_budget(
                    cfg.cost_limit, results, "review_names"
                ),
            ),
        ),
        (
            "review_docs",
            _skip_review_docs,
            lambda: _run_review_docs_phase(
                cfg,
                budget_override=_adaptive_review_budget(
                    cfg.cost_limit, results, "review_docs"
                ),
            ),
        ),
        (
            "regen",
            cfg.skip_regen,
            lambda: _run_generate_phase(
                cfg,
                regen=True,
                force=True,
                budget_override=_adaptive_review_budget(
                    cfg.cost_limit, results, "regen"
                ),
            ),
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

        # Accumulate touched names for downstream scoping
        if result.touched_names:
            touched_names.update(result.touched_names)

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
            }
            for r in results
        ],
        "total_cost": total_cost,
        "total_elapsed": total_elapsed,
        "total_count": total_count,
        "cost_limit": cfg.cost_limit,
        "exit_code": max_exit,
        "errors": [{"phase": e.name, "error": e.error} for e in errors],
        "dry_run": cfg.dry_run,
    }
