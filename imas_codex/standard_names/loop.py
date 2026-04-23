"""SN loop — drives ``sn run`` with one-domain-per-turn rotation.

Picks the stalest extract-eligible physics domain via stale-first
rotation (oldest ``SNRun.ended_at`` wins, with eligible-source count as
tiebreak), runs one :func:`run_turn` on it with the full remaining
budget, and repeats until the budget drops below the turn-entry floor or
no domain has eligible work.

Entry point: :func:`run_sn_loop`.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES

logger = logging.getLogger(__name__)

# Minimum remaining budget required to start a new turn.  Below this
# threshold the loop stops cleanly — a viable turn needs enough budget
# to fund at least one compose batch plus a review pass (~$0.75).
MIN_VIABLE_TURN: float = 0.75


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
              WHERE NOT (sns.status IN ['stale', 'failed'])
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


def _write_sn_run(summary: RunSummary) -> None:
    """Persist an SNRun node using the generated Pydantic model."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.models import SNRun

    rr = SNRun(
        id=summary.run_id,
        turn_number=summary.turn_number,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        cost_spent=round(summary.cost_spent, 6),
        cost_limit=round(summary.cost_limit, 6),
        min_score=summary.min_score,
        names_composed=summary.names_composed,
        names_enriched=summary.names_enriched,
        names_reviewed=summary.names_reviewed,
        names_regenerated=summary.names_regenerated,
        domains_touched=sorted(summary.domains_touched),
        stop_reason=summary.stop_reason,
    )
    try:
        props = rr.model_dump(mode="json")
        with GraphClient() as gc:
            gc.create_nodes("SNRun", [props])
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to persist SNRun %s: %s", summary.run_id, exc)


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
) -> RunSummary:
    """Drive the ``sn run`` loop with one-domain-per-turn rotation.

    Each iteration picks ONE domain via stale-first rotation (oldest
    ``SNRun.ended_at`` wins, eligible-source count as tiebreak) and
    runs a full turn on it with the entire remaining budget.  The loop
    stops when the remaining budget drops below :data:`MIN_VIABLE_TURN`
    or no domain has eligible work.

    When *only_domain* is set the rotation is bypassed — the explicit
    user choice is used every iteration until budget runs out or the
    domain has no remaining work.
    """
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

    remaining_budget = cost_limit

    try:
        while True:
            # ── Turn-entry floor ──────────────────────────────────
            if remaining_budget < MIN_VIABLE_TURN:
                summary.stop_reason = "budget_exhausted"
                logger.info(
                    "Budget exhausted: $%.2f remaining < floor $%.2f",
                    remaining_budget,
                    MIN_VIABLE_TURN,
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
            )
            results = await run_turn(cfg)

            # ── Accumulate counters ───────────────────────────────
            turn_cost = 0.0
            for phase in results:
                turn_cost += phase.cost
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

            summary.cost_spent += turn_cost
            remaining_budget -= turn_cost
            summary.domains_touched.add(dom)

            summary.pass_records.append(
                {
                    "domain": dom,
                    "remaining_before": entry["remaining"],
                    "budget": remaining_budget + turn_cost,
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

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("sn run interrupted by user")
    finally:
        summary.ended_at = datetime.now(UTC)
        _write_sn_run(summary)

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
