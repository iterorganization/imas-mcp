"""Per-domain SN generation cycle runner with rich score + theme reporting.

Iterates physics_domains with extract-eligible backlog, runs ``sn run``
sequentially at a per-domain cost cap, and emits a markdown report with
score distribution, quarantine rate, anti-pattern themes, and VocabGap
harvest per domain.

Usage::

    uv run python scripts/run_domain_cycles.py \\
        --cost-limit 5.00 \\
        --turn-number 1 \\
        --min-score 0.0 \\
        --source dd \\
        --target names \\
        --skip-domains general \\
        --report /tmp/cycle_run_report.md \\
        --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

try:
    from imas_codex.graph.client import GraphClient
except ImportError:  # pragma: no cover
    GraphClient = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Anti-pattern theme keywords for substring matching on reviewer comments
# ---------------------------------------------------------------------------

ANTI_PATTERN_THEMES: dict[str, list[str]] = {
    "instrument_identity_in_name": [
        "instrument",
        "device name",
        "sensor name",
        "probe name",
        "diagnostic name",
    ],
    "initial_as_state_prefix": [
        "initial",
        "initial_",
        "_initial",
        "initial value",
        "initial state",
    ],
    "closed_vocab_missing": [
        "closed vocab",
        "closed vocabulary",
        "not in grammar",
        "not a valid token",
        "unrecognized token",
    ],
    "missing_optical_bases": [
        "optical",
        "wavelength",
        "photon",
        "emission line",
        "spectral",
    ],
    "unit_mismatch": [
        "unit mismatch",
        "wrong unit",
        "incorrect unit",
        "unit inconsistency",
        "unit conflict",
    ],
    "vector_component_grammar": [
        "vector component",
        "component grammar",
        "r_component",
        "z_component",
        "phi_component",
        "radial component",
        "toroidal component",
        "poloidal component",
    ],
    "canonical_ordering_issue": [
        "canonical order",
        "ordering issue",
        "segment order",
        "wrong order",
        "reorder",
    ],
}


# ---------------------------------------------------------------------------
# Graph queries
# ---------------------------------------------------------------------------


def _query_domains_with_backlog(
    gc: Any,
    source: str = "dd",
    skip_domains: set[str] | None = None,
    include_domains: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return domains ordered by backlog size (largest first).

    Queries StandardNameSource nodes with ``status='extracted'`` and
    ``claimed_at IS NULL``, joined to IMASNode→physics_domain.

    Args:
        gc: GraphClient instance.
        source: Source type filter (``dd`` or ``signals``).
        skip_domains: Set of domain names to exclude.
        include_domains: If non-empty, restrict to only these domains.

    Returns:
        List of dicts with ``domain`` and ``backlog`` keys.
    """
    cypher = """
        MATCH (sns:StandardNameSource)
        WHERE sns.status = 'extracted'
          AND sns.claimed_at IS NULL
          AND sns.source_type = $source_type
        OPTIONAL MATCH (n:IMASNode {id: sns.source_id})
        WITH coalesce(n.physics_domain, sns.physics_domain) AS domain,
             count(*) AS backlog
        WHERE domain IS NOT NULL AND domain <> 'unclassified'
        RETURN domain, backlog
        ORDER BY backlog DESC
    """
    rows = list(gc.query(cypher, source_type=source))
    result = []
    for row in rows:
        domain = row.get("domain")
        if not domain:
            continue
        if skip_domains and domain in skip_domains:
            continue
        if include_domains and domain not in include_domains:
            continue
        result.append({"domain": domain, "backlog": row.get("backlog", 0)})
    return result


def _query_domain_stats(gc: Any, domain: str, since_ts: datetime) -> dict[str, Any]:
    """Query rich stats for a domain after running the cycle.

    Args:
        gc: GraphClient instance.
        domain: Physics domain to query.
        since_ts: Start timestamp for cost calculation.

    Returns:
        Dict with generated, reviewed, scores, quarantine, themes, vocab_gaps, spend.
    """
    # 1. Pipeline status breakdown
    pipeline_rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain = $domain
        RETURN sn.pipeline_status AS status, count(*) AS n
        """,
        domain=domain,
    )
    pipeline_counts: dict[str, int] = {}
    for row in pipeline_rows:
        s = row.get("status") or "unknown"
        pipeline_counts[str(s)] = int(row.get("n", 0))

    total_generated = sum(pipeline_counts.values())

    # 2. Reviewed names with scores
    score_rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain = $domain
          AND sn.reviewer_score_name IS NOT NULL
        RETURN sn.reviewer_score_name AS score
        """,
        domain=domain,
    )
    scores = [float(r["score"]) for r in score_rows if r.get("score") is not None]
    n_reviewed = len(scores)

    score_stats: dict[str, Any] = {
        "n": n_reviewed,
        "min": None,
        "mean": None,
        "median": None,
        "p75": None,
        "max": None,
    }
    if scores:
        scores_sorted = sorted(scores)
        n = len(scores_sorted)
        score_stats["min"] = round(scores_sorted[0], 3)
        score_stats["max"] = round(scores_sorted[-1], 3)
        score_stats["mean"] = round(sum(scores_sorted) / n, 3)
        mid = n // 2
        score_stats["median"] = round(
            scores_sorted[mid]
            if n % 2
            else (scores_sorted[mid - 1] + scores_sorted[mid]) / 2,
            3,
        )
        p75_idx = int(n * 0.75)
        score_stats["p75"] = round(scores_sorted[min(p75_idx, n - 1)], 3)

    # 3. Quarantine count and top issues
    quarantine_rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain = $domain
          AND sn.validation_status = 'quarantined'
        RETURN sn.validation_issues AS issues
        """,
        domain=domain,
    )
    quarantine_count = len(quarantine_rows)
    issue_prefix_counter: dict[str, int] = {}
    for row in quarantine_rows:
        issues = row.get("issues") or []
        if isinstance(issues, str):
            issues = [issues]
        for issue in issues[:3]:  # top 3 issues per name
            prefix = str(issue)[:60].strip()
            issue_prefix_counter[prefix] = issue_prefix_counter.get(prefix, 0) + 1
    top_issues = sorted(issue_prefix_counter, key=lambda k: -issue_prefix_counter[k])[
        :3
    ]

    # 4. Anti-pattern themes from reviewer comments (substring matching)
    theme_rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain = $domain
          AND sn.reviewer_comments_name IS NOT NULL
        RETURN sn.reviewer_comments_name AS comments
        LIMIT 200
        """,
        domain=domain,
    )
    theme_counts: dict[str, int] = dict.fromkeys(ANTI_PATTERN_THEMES, 0)
    for row in theme_rows:
        text = (row.get("comments") or "").lower()
        for theme, keywords in ANTI_PATTERN_THEMES.items():
            if any(kw in text for kw in keywords):
                theme_counts[theme] += 1
    top_themes = sorted(
        [(t, c) for t, c in theme_counts.items() if c > 0],
        key=lambda x: -x[1],
    )[:5]

    # 5. VocabGap count by segment for this domain's sources
    gap_rows = gc.query(
        """
        MATCH (sns:StandardNameSource)
        WHERE sns.source_type = 'dd' OR sns.source_type = 'signals'
        MATCH (n:IMASNode {id: sns.source_id})
        WHERE n.physics_domain = $domain
        MATCH (n)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(vg:VocabGap)
        RETURN vg.segment AS segment, count(DISTINCT vg.id) AS gap_count
        ORDER BY gap_count DESC
        """,
        domain=domain,
    )
    vocab_gaps: dict[str, int] = {}
    for row in gap_rows:
        seg = row.get("segment") or "unknown"
        vocab_gaps[str(seg)] = int(row.get("gap_count", 0))

    # 6. Total spend since cycle start
    spend_rows = gc.query(
        """
        MATCH (sn:StandardName)
        WHERE sn.physics_domain = $domain
          AND sn.generated_at >= $since_ts
        RETURN
            coalesce(sum(sn.llm_cost_compose), 0.0) +
            coalesce(sum(sn.llm_cost_review), 0.0) +
            coalesce(sum(sn.llm_cost_enrich), 0.0) AS total_spend
        """,
        domain=domain,
        since_ts=since_ts.isoformat(),
    )
    cycle_spend = 0.0
    if spend_rows and spend_rows[0].get("total_spend") is not None:
        cycle_spend = float(spend_rows[0]["total_spend"])

    return {
        "pipeline_counts": pipeline_counts,
        "total_generated": total_generated,
        "scores": score_stats,
        "n_reviewed": n_reviewed,
        "quarantine_count": quarantine_count,
        "top_issues": top_issues,
        "top_themes": top_themes,
        "vocab_gaps": vocab_gaps,
        "cycle_spend": cycle_spend,
    }


# ---------------------------------------------------------------------------
# Markdown report helpers
# ---------------------------------------------------------------------------


def _format_domain_section(
    domain: str,
    stats: dict[str, Any],
    cost: float,
    turn_number: int,
    duration_s: float,
) -> str:
    """Format a markdown section for one domain."""
    pc = stats["pipeline_counts"]
    sc = stats["scores"]
    qc = stats["quarantine_count"]
    total = stats["total_generated"]

    named = pc.get("named", 0)
    valid_count = pc.get("valid", named)  # fallback to named if valid not separate
    quarantined = qc

    quarantine_pct = (quarantined / total * 100) if total > 0 else 0.0

    reviewed_line = f"reviewed: {sc['n']}"
    if sc["mean"] is not None:
        reviewed_line += (
            f"  mean={sc['mean']}  median={sc['median']}"
            f"  min={sc['min']}  max={sc['max']}"
        )

    top_issues_str = ", ".join(stats["top_issues"]) if stats["top_issues"] else "none"

    theme_parts = [f"{t} ({c})" for t, c in stats["top_themes"]]
    top_themes_str = ", ".join(theme_parts) if theme_parts else "none"

    gap_parts = [f"{seg}={cnt}" for seg, cnt in stats["vocab_gaps"].items()]
    vocab_gaps_str = " ".join(gap_parts) if gap_parts else "none"

    lines = [
        f"## {domain}  (${cost:.4f}, turn {turn_number}, {duration_s:.1f}s)",
        f"- generated: {total} (named={named}, valid={valid_count}, quarantined={quarantined})",
        f"- {reviewed_line}",
        f"- quarantine rate: {quarantine_pct:.1f}%   top issues: {top_issues_str}",
        f"- top themes: {top_themes_str}",
        f"- vocab gaps: {vocab_gaps_str}",
        "",
    ]
    return "\n".join(lines)


def _format_global_summary(
    domain_results: list[dict[str, Any]],
    turn_number: int,
    total_cost: float,
) -> str:
    """Format the global summary section."""
    total_names = sum(r["stats"]["total_generated"] for r in domain_results)
    all_scores = []
    for r in domain_results:
        sc = r["stats"]["scores"]
        if sc["mean"] is not None:
            # Approximate: weight by n_reviewed
            n = sc["n"]
            all_scores.extend([sc["mean"]] * max(n, 1))

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_gaps = sum(sum(r["stats"]["vocab_gaps"].values()) for r in domain_results)
    n_domains = len(domain_results)

    lines = [
        f"# Run summary — turn {turn_number}, total ${total_cost:.4f}",
        "",
        "| Domain | Generated | Reviewed | Mean Score | Quarantine | Spend |",
        "|--------|-----------|----------|------------|------------|-------|",
    ]
    for r in domain_results:
        sc = r["stats"]["scores"]
        ms = f"{sc['mean']:.3f}" if sc["mean"] is not None else "-"
        qc = r["stats"]["quarantine_count"]
        lines.append(
            f"| {r['domain']} | {r['stats']['total_generated']} "
            f"| {r['stats']['n_reviewed']} | {ms} "
            f"| {qc} | ${r['cost']:.4f} |"
        )
    lines += [
        "",
        f"**Totals:** {n_domains} domains, {total_names} names, "
        f"mean score {mean_score:.3f}, vocab gaps {total_gaps}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _build_sn_run_cmd(
    domain: str,
    source: str,
    target: str,
    cost_limit: float,
    turn_number: int,
    min_score: float | None,
    dry_run: bool,
) -> list[str]:
    """Build the ``uv run imas-codex sn run`` subprocess command."""
    cmd = [
        sys.executable,
        "-m",
        "imas_codex",
        "sn",
        "run",
        "--source",
        source,
        "--target",
        target,
        "--physics-domain",
        domain,
        "--cost-limit",
        str(cost_limit),
        "--turn-number",
        str(turn_number),
        "--single-pass",
    ]
    if min_score is not None and min_score > 0.0:
        cmd += ["--min-score", str(min_score)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main(argv: list[str] | None = None) -> int:
    """Entry point for the per-domain cycle runner."""
    parser = argparse.ArgumentParser(
        description="Per-domain SN generation cycle runner with rich reporting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=5.00,
        metavar="USD",
        help="Maximum LLM cost per domain in USD.",
    )
    parser.add_argument(
        "--turn-number",
        type=int,
        default=1,
        metavar="N",
        help="Turn number to stamp on generated names.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="F",
        help="Reviewer-score threshold for regen phase (omit to skip regen).",
    )
    parser.add_argument(
        "--source",
        choices=["dd", "signals"],
        default="dd",
        help="Source to extract candidates from.",
    )
    parser.add_argument(
        "--target",
        choices=["names", "docs"],
        default="names",
        help="Which generation pass to run.",
    )
    parser.add_argument(
        "--skip-domains",
        default="",
        metavar="DOMAINS",
        help="Comma-separated domain names to skip.",
    )
    parser.add_argument(
        "--include-domains",
        default="",
        metavar="DOMAINS",
        help="If non-empty, only run these comma-separated domains.",
    )
    parser.add_argument(
        "--report",
        default="/tmp/cycle_run_report.md",
        metavar="PATH",
        help="File path for the markdown report (appended, not overwritten).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each sn run invocation.",
    )

    args = parser.parse_args(argv)

    skip_domains: set[str] = {
        d.strip() for d in args.skip_domains.split(",") if d.strip()
    }
    include_domains: set[str] = {
        d.strip() for d in args.include_domains.split(",") if d.strip()
    }

    start_ts = datetime.now(UTC)

    # ------------------------------------------------------------------
    # Discover domains
    # ------------------------------------------------------------------
    try:
        with GraphClient() as gc:
            domains = _query_domains_with_backlog(
                gc,
                source=args.source,
                skip_domains=skip_domains,
                include_domains=include_domains,
            )
    except Exception as exc:
        print(
            f"[cycle-runner] ERROR: Failed to connect to graph: {exc}", file=sys.stderr
        )
        return 1

    if not domains:
        print(
            "[cycle-runner] No domains with extract-eligible backlog found. "
            "Nothing to do — exiting.",
        )
        return 0

    print(
        f"[cycle-runner] Found {len(domains)} domain(s): "
        + ", ".join(f"{d['domain']} (backlog={d['backlog']})" for d in domains)
    )

    # ------------------------------------------------------------------
    # Per-domain loop
    # ------------------------------------------------------------------
    domain_results: list[dict[str, Any]] = []
    report_lines: list[str] = []
    total_cost = 0.0

    for entry in domains:
        domain = entry["domain"]
        print(f"\n[cycle-runner] ── Running domain: {domain} ──")

        cmd = _build_sn_run_cmd(
            domain=domain,
            source=args.source,
            target=args.target,
            cost_limit=args.cost_limit,
            turn_number=args.turn_number,
            min_score=args.min_score,
            dry_run=args.dry_run,
        )

        domain_start = time.monotonic()
        returncode = 0
        try:
            result = subprocess.run(cmd, check=True)
            returncode = result.returncode
        except subprocess.CalledProcessError as exc:
            returncode = exc.returncode
            print(
                f"[cycle-runner] WARNING: sn run failed for domain '{domain}' "
                f"(returncode={returncode}). Continuing to next domain.",
                file=sys.stderr,
            )
        except Exception as exc:
            print(
                f"[cycle-runner] WARNING: Unexpected error for domain '{domain}': {exc}. "
                "Continuing.",
                file=sys.stderr,
            )
            returncode = -1

        duration_s = time.monotonic() - domain_start

        # Query stats from graph
        try:
            with GraphClient() as gc:
                stats = _query_domain_stats(gc, domain, start_ts)
        except Exception as exc:
            print(
                f"[cycle-runner] WARNING: Failed to query stats for '{domain}': {exc}",
                file=sys.stderr,
            )
            stats = {
                "pipeline_counts": {},
                "total_generated": 0,
                "scores": {
                    "n": 0,
                    "min": None,
                    "mean": None,
                    "median": None,
                    "p75": None,
                    "max": None,
                },
                "n_reviewed": 0,
                "quarantine_count": 0,
                "top_issues": [],
                "top_themes": [],
                "vocab_gaps": {},
                "cycle_spend": 0.0,
            }

        domain_cost = stats["cycle_spend"]
        total_cost += domain_cost

        section = _format_domain_section(
            domain=domain,
            stats=stats,
            cost=domain_cost,
            turn_number=args.turn_number,
            duration_s=duration_s,
        )
        print(section)
        report_lines.append(section)

        domain_results.append(
            {
                "domain": domain,
                "stats": stats,
                "cost": domain_cost,
                "returncode": returncode,
                "duration_s": duration_s,
            }
        )

    # ------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------
    summary = _format_global_summary(domain_results, args.turn_number, total_cost)
    print(summary)
    report_lines.append(summary)

    # Aggregate for final one-liner
    total_names = sum(r["stats"]["total_generated"] for r in domain_results)
    all_scores: list[float] = []
    for r in domain_results:
        sc = r["stats"]["scores"]
        if sc["mean"] is not None:
            all_scores.extend([sc["mean"]] * max(sc["n"], 1))
    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total_gaps = sum(sum(r["stats"]["vocab_gaps"].values()) for r in domain_results)

    print(
        f"✓ {len(domain_results)} domains, {total_names} names, "
        f"mean score {mean_score:.3f}, vocab gaps {total_gaps}"
    )

    # ------------------------------------------------------------------
    # Write report file
    # ------------------------------------------------------------------
    run_header = (
        f"<!-- cycle-runner run: {start_ts.isoformat()} "
        f"turn={args.turn_number} cost_limit={args.cost_limit} -->\n\n"
    )
    full_report = run_header + "\n".join(report_lines)
    try:
        with open(args.report, "a", encoding="utf-8") as fh:
            fh.write(full_report + "\n")
        print(f"[cycle-runner] Report appended to {args.report}")
    except OSError as exc:
        print(
            f"[cycle-runner] WARNING: Could not write report to {args.report}: {exc}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
