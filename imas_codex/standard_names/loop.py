"""SN loop — drives ``sn run`` across physics domains.

Picks extract-eligible physics domains (ordered by backlog), allocates a
fair-share budget, runs one :func:`run_turn` for each, persists an
:class:`~imas_codex.graph.models.SNRun` node, and stops when the total
cost budget is exhausted or the user interrupts.

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
) -> RunSummary:
    """Drive the ``sn run`` loop across physics domains.

    Single-pass loop (no plateau detection): iterates extract-eligible
    domains once in descending backlog order, runs one turn per domain,
    stops when the total cost budget is exhausted. The caller drives
    multi-turn iteration by re-invoking with ``turn_number+1``.
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

    try:
        if skip_generate:
            domains = _existing_domain_targets(only_domain=only_domain)
        else:
            domains = _count_eligible_domains(only_domain=only_domain)
            if not domains:
                domains = _existing_domain_targets(only_domain=only_domain)

        if not domains:
            summary.stop_reason = "completed"
            logger.info("No eligible domains; nothing to do.")
            return summary

        n_domains = len(domains)
        per_domain_budget = max(0.01, cost_limit / max(n_domains, 1))

        logger.info(
            "Turn %d: %d eligible domain(s), budget $%.2f, per-domain $%.2f",
            turn_number,
            n_domains,
            cost_limit,
            per_domain_budget,
        )

        for entry in domains:
            dom = entry["domain"]
            if summary.cost_spent >= cost_limit:
                summary.stop_reason = "budget_exhausted"
                break

            dom_budget = min(per_domain_budget, cost_limit - summary.cost_spent)
            if dom_budget <= 0.01:
                summary.stop_reason = "budget_exhausted"
                break

            cfg = TurnConfig(
                domain=dom,
                cost_limit=dom_budget,
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
            )
            results = await run_turn(cfg)

            for phase in results:
                summary.cost_spent += phase.cost
                if phase.name == "generate":
                    summary.names_composed += phase.count
                elif phase.name == "enrich":
                    summary.names_enriched += phase.count
                elif phase.name == "review":
                    summary.names_reviewed += phase.count
                elif phase.name == "regen":
                    summary.names_regenerated += phase.count
            summary.domains_touched.add(dom)

            summary.pass_records.append(
                {
                    "domain": dom,
                    "remaining_before": entry["remaining"],
                    "budget": dom_budget,
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

        if summary.cost_spent >= cost_limit:
            summary.stop_reason = "budget_exhausted"

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("sn run interrupted by user")
    finally:
        summary.ended_at = datetime.now(UTC)
        _write_sn_run(summary)

    return summary


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
               sn.review_status IN ['named', 'drafted']
            OR sn.reviewer_score IS NULL
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
        "domains_touched": sorted(summary.domains_touched),
        "stop_reason": summary.stop_reason,
    }
