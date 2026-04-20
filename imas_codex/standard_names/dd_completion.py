"""DD completion loop — drives the ``sn generate`` rotator.

Iterates rotation cycles (generate → enrich → review → regen) across
every physics_domain that still has eligible DD paths, persists a
:class:`~imas_codex.graph.models.RotationRun` node for each full
invocation, and stops when budget is exhausted, consecutive passes
produce no net change (plateau), or the user interrupts.

Entry point: :func:`run_dd_completion`.
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
class RotationSummary:
    """Aggregated result of a ``sn generate`` rotator run."""

    rotation_id: str
    started_at: datetime
    ended_at: datetime | None = None
    cost_spent: float = 0.0
    cost_limit: float = 0.0
    names_composed: int = 0
    names_enriched: int = 0
    names_reviewed: int = 0
    names_regenerated: int = 0
    domains_touched: set[str] = field(default_factory=set)
    stop_reason: str = "completed"
    passes: int = 0
    pass_records: list[dict[str, Any]] = field(default_factory=list)


def _count_eligible_domains(
    only_domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return domains with extract-eligible DD paths, ordered by backlog size.

    An "eligible" path is one that:
      * has ``node_category IN SN_SOURCE_CATEGORIES`` ({quantity, geometry})
      * has ``node_type IN ['dynamic', 'constant']`` and a non-empty description
      * is NOT already covered by a StandardNameSource whose status is not
        ``stale`` or ``failed`` (which matches the "skip if not force" guard
        in :func:`imas_codex.standard_names.sources.dd.extract_from_dd`).

    Returns a list of ``{"domain": str, "remaining": int}`` dicts.
    """
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
    # Drop empty/unclassified domain (the generate pipeline requires a
    # physics_domain filter; unclassified paths are named separately).
    filtered = [r for r in rows if r["domain"] and r["domain"] != "unclassified"]
    if only_domain:
        filtered = [r for r in filtered if r["domain"] == only_domain]
    return filtered


def _write_rotation_run(summary: RotationSummary) -> None:
    """Persist a RotationRun node using the generated Pydantic model."""
    from imas_codex.graph.client import GraphClient
    from imas_codex.graph.models import RotationRun

    rr = RotationRun(
        id=summary.rotation_id,
        started_at=summary.started_at,
        ended_at=summary.ended_at,
        cost_spent=round(summary.cost_spent, 6),
        cost_limit=round(summary.cost_limit, 6),
        names_composed=summary.names_composed,
        names_enriched=summary.names_enriched,
        names_reviewed=summary.names_reviewed,
        names_regenerated=summary.names_regenerated,
        domains_touched=sorted(summary.domains_touched),
        stop_reason=summary.stop_reason,
        passes=summary.passes,
    )
    try:
        # Serialize datetimes to ISO strings; Neo4j driver handles native
        # datetime too, but create_nodes stores dict values via SET n += $props
        # and ISO strings keep round-trip behaviour identical to other SN writes.
        props = rr.model_dump(mode="json")
        with GraphClient() as gc:
            gc.create_nodes("RotationRun", [props])
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Failed to persist RotationRun %s: %s", summary.rotation_id, exc)


async def run_dd_completion(
    cost_limit: float,
    *,
    plateau_passes: int = 1,
    per_domain_limit: int | None = None,
    concurrency: int = 2,
    dry_run: bool = False,
    only_domain: str | None = None,
) -> RotationSummary:
    """Drive the DD naming exercise to completion.

    Loop::

        repeat:
          domains = eligible_domains()    # live query
          if not domains and plateau_count >= plateau_passes: stop
          for d in domains:
              run_rotation(domain=d, budget=fair_share_of(remaining))
              accumulate counts / cost
              if cost_spent >= cost_limit: stop

    Args:
        cost_limit: Total budget in USD across all passes.
        plateau_passes: Consecutive zero-change passes before plateau exit.
        per_domain_limit: Max paths per domain per rotation (forwarded).
        concurrency: Parallel workers per phase.
        dry_run: Plan domains without running any LLM calls.

    Returns:
        RotationSummary describing the end-to-end run.
    """
    from imas_codex.standard_names.rotation import RotationConfig, run_rotation

    started = datetime.now(UTC)
    summary = RotationSummary(
        rotation_id=str(uuid.uuid4()),
        started_at=started,
        cost_limit=cost_limit,
    )

    if dry_run:
        # No LLM calls — just report what would run.
        domains = _count_eligible_domains(only_domain=only_domain)
        summary.pass_records.append(
            {
                "pass": 0,
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

    plateau_count = 0
    pass_idx = 0

    try:
        while True:
            pass_idx += 1
            summary.passes = pass_idx

            domains = _count_eligible_domains(only_domain=only_domain)
            remaining_budget = cost_limit - summary.cost_spent
            if remaining_budget <= 0:
                summary.stop_reason = "budget_exhausted"
                break

            pass_before = {
                "composed": summary.names_composed,
                "enriched": summary.names_enriched,
                "reviewed": summary.names_reviewed,
                "regenerated": summary.names_regenerated,
            }

            if not domains:
                # Nothing extract-eligible remains — we still run an
                # enrich/review/regen pass to catch orphaned named /
                # needs_revision entries, then check plateau.
                logger.info(
                    "Pass %d: no extract-eligible domains; running "
                    "maintenance pass (enrich/review/regen)",
                    pass_idx,
                )
                # Pick currently-covered domains from the graph instead.
                domains = _existing_domain_targets(only_domain=only_domain)

            if not domains:
                plateau_count += 1
                logger.info(
                    "Pass %d: no domains to process (plateau %d/%d)",
                    pass_idx,
                    plateau_count,
                    plateau_passes,
                )
                if plateau_count >= plateau_passes:
                    summary.stop_reason = "plateau"
                    break
                continue

            # Fair-share budget across domains for this pass.
            n_domains = len(domains)
            per_domain_budget = max(0.01, remaining_budget / max(n_domains, 1))

            logger.info(
                "Pass %d: %d eligible domain(s), remaining budget $%.2f, "
                "per-domain $%.2f",
                pass_idx,
                n_domains,
                remaining_budget,
                per_domain_budget,
            )

            for entry in domains:
                dom = entry["domain"]
                if summary.cost_spent >= cost_limit:
                    summary.stop_reason = "budget_exhausted"
                    break

                # Cap each domain rotation at whatever budget remains.
                dom_budget = min(per_domain_budget, cost_limit - summary.cost_spent)
                if dom_budget <= 0.01:
                    summary.stop_reason = "budget_exhausted"
                    break

                cfg = RotationConfig(
                    domain=dom,
                    cost_limit=dom_budget,
                    limit=per_domain_limit,
                    concurrency=concurrency,
                    dry_run=False,
                    # Same rotation_id for every per-domain cycle in this
                    # DD-completion run, so RotationRun ↔ StandardName
                    # traceability is coherent.
                    rotation_id=summary.rotation_id,
                )
                results = await run_rotation(cfg)

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
                        "pass": pass_idx,
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

            pass_after = {
                "composed": summary.names_composed,
                "enriched": summary.names_enriched,
                "reviewed": summary.names_reviewed,
                "regenerated": summary.names_regenerated,
            }
            delta = sum(pass_after[k] - pass_before[k] for k in pass_before)
            if delta == 0:
                plateau_count += 1
                logger.info(
                    "Pass %d produced 0 net change (plateau %d/%d)",
                    pass_idx,
                    plateau_count,
                    plateau_passes,
                )
                if plateau_count >= plateau_passes:
                    summary.stop_reason = "plateau"
                    break
            else:
                plateau_count = 0

            if summary.cost_spent >= cost_limit:
                summary.stop_reason = "budget_exhausted"
                break

    except KeyboardInterrupt:
        summary.stop_reason = "interrupted"
        logger.warning("DD completion interrupted by user after pass %d", pass_idx)
    finally:
        summary.ended_at = datetime.now(UTC)
        _write_rotation_run(summary)

    return summary


def _existing_domain_targets(
    only_domain: str | None = None,
) -> list[dict[str, Any]]:
    """Return domains that have un-enriched / un-reviewed / needs_revision names.

    Used when no extract-eligible paths remain: we still want to run
    enrich/review/regen on whatever leftover names are in an incomplete
    state, to maximize coverage before reporting plateau.
    """
    from imas_codex.graph.client import GraphClient

    cypher = """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain IS NOT NULL
          AND (
               sn.review_status IN ['named', 'drafted']
            OR sn.reviewer_score IS NULL
            OR (sn.validation_status = 'needs_revision'
                AND coalesce(sn.regen_count, 0) < 1)
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


def summary_table(summary: RotationSummary) -> dict[str, Any]:
    """Flatten a :class:`RotationSummary` for Rich display / JSON output."""
    return {
        "rotation_id": summary.rotation_id,
        "started_at": summary.started_at.isoformat(),
        "ended_at": summary.ended_at.isoformat() if summary.ended_at else None,
        "elapsed_s": (
            (summary.ended_at - summary.started_at).total_seconds()
            if summary.ended_at
            else None
        ),
        "cost_spent": round(summary.cost_spent, 6),
        "cost_limit": summary.cost_limit,
        "passes": summary.passes,
        "names_composed": summary.names_composed,
        "names_enriched": summary.names_enriched,
        "names_reviewed": summary.names_reviewed,
        "names_regenerated": summary.names_regenerated,
        "domains_touched": sorted(summary.domains_touched),
        "stop_reason": summary.stop_reason,
    }
