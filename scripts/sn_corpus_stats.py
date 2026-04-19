#!/usr/bin/env python3
"""Standard-name corpus statistics dashboard.

Plan 31 §G.3 — prints five reporting sections using Cypher aggregations
against the live Neo4j graph.  Pure read-only; exits 0 always.

Usage:
    uv run python scripts/sn_corpus_stats.py
"""

from __future__ import annotations

import sys


def _section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def run() -> None:  # noqa: C901 – single reporting function, intentionally flat
    try:
        from imas_codex.graph.client import GraphClient

        gc = GraphClient()
    except Exception as exc:
        print(f"⚠  Cannot connect to Neo4j: {exc}")
        print("   Dashboard requires a live graph.  Exiting cleanly.")
        return

    # ── Section 1: Total / valid / quarantined ──────────────────────
    _section("1. Corpus summary")
    rows = gc.query("""
        MATCH (sn:StandardName)
        WITH count(sn) AS total,
             sum(CASE WHEN sn.validation_status = 'valid' THEN 1 ELSE 0 END) AS valid,
             sum(CASE WHEN sn.validation_status = 'quarantined' THEN 1 ELSE 0 END) AS quarantined,
             sum(CASE WHEN sn.validation_status = 'pending'
                        OR sn.validation_status IS NULL THEN 1 ELSE 0 END) AS pending
        RETURN total, valid, quarantined, pending
    """)
    r = rows[0] if rows else {"total": 0, "valid": 0, "quarantined": 0, "pending": 0}
    total = r["total"]
    q_pct = (r["quarantined"] / total * 100) if total else 0
    v_pct = (r["valid"] / total * 100) if total else 0
    p_pct = (r["pending"] / total * 100) if total else 0
    print(f"  Total standard names:  {total}")
    print(f"  Valid:                 {r['valid']:>5}  ({v_pct:5.1f}%)")
    print(f"  Quarantined:           {r['quarantined']:>5}  ({q_pct:5.1f}%)")
    print(f"  Pending:               {r['pending']:>5}  ({p_pct:5.1f}%)")

    # ── Section 2: Per-domain quarantine rate ───────────────────────
    _section("2. Per-domain quarantine rate")
    rows = gc.query("""
        MATCH (sn:StandardName)
        WHERE sn.physics_domain IS NOT NULL
        WITH sn.physics_domain AS domain,
             count(sn) AS total,
             sum(CASE WHEN sn.validation_status = 'quarantined'
                      THEN 1 ELSE 0 END) AS quarantined
        RETURN domain, total, quarantined,
               CASE WHEN total = 0 THEN 0.0
                    ELSE toFloat(quarantined) / toFloat(total) * 100
               END AS q_pct
        ORDER BY q_pct DESC
    """)
    if rows:
        print(f"  {'Domain':<25} {'Total':>6} {'Quar':>6} {'Rate':>7}")
        print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 7}")
        for r in rows:
            bar = "█" * int(r["q_pct"] / 5) if r["q_pct"] > 0 else ""
            print(
                f"  {r['domain']:<25} {r['total']:>6} "
                f"{r['quarantined']:>6} {r['q_pct']:>6.1f}% {bar}"
            )
    else:
        print("  (no domain data)")

    # ── Section 3: Audit-issue histogram ────────────────────────────
    _section("3. Validation issues (audit histogram)")
    rows = gc.query("""
        MATCH (sn:StandardName)
        WHERE sn.validation_issues IS NOT NULL
          AND size(sn.validation_issues) > 0
        UNWIND sn.validation_issues AS issue
        WITH CASE
               WHEN issue CONTAINS ':' THEN split(issue, ':')[0]
               ELSE issue
             END AS issue_type
        RETURN issue_type, count(*) AS cnt
        ORDER BY cnt DESC
        LIMIT 20
    """)
    if rows:
        max_cnt = max(r["cnt"] for r in rows)
        for r in rows:
            bar_len = int(r["cnt"] / max(max_cnt, 1) * 30)
            print(f"  {r['issue_type']:<35} {r['cnt']:>5}  {'█' * bar_len}")
    else:
        print("  (no validation issues)")

    # ── Section 4: Description coverage + mean length ───────────────
    _section("4. Description coverage")
    rows = gc.query("""
        MATCH (sn:StandardName)
        WITH count(sn) AS total,
             sum(CASE WHEN nullIf(coalesce(sn.description, ''), '') IS NOT NULL
                      THEN 1 ELSE 0 END) AS has_desc,
             collect(CASE WHEN nullIf(coalesce(sn.description, ''), '') IS NOT NULL
                          THEN size(sn.description) END) AS lengths
        RETURN total, has_desc,
               CASE WHEN total = 0 THEN 0.0
                    ELSE toFloat(has_desc) / toFloat(total) * 100
               END AS coverage_pct,
               CASE WHEN size(lengths) = 0 THEN 0
                    ELSE reduce(s = 0, x IN lengths | s + x) / size(lengths)
               END AS mean_len
    """)
    r = (
        rows[0]
        if rows
        else {"total": 0, "has_desc": 0, "coverage_pct": 0, "mean_len": 0}
    )
    print(
        f"  With description:  {r['has_desc']}/{r['total']}  ({r['coverage_pct']:.1f}%)"
    )
    print(f"  Mean length:       {r['mean_len']:.0f} chars")

    # ── Section 5: Reviewer score distribution ──────────────────────
    _section("5. Reviewer score distribution")
    rows = gc.query("""
        MATCH (sn:StandardName)
        WHERE sn.reviewer_score IS NOT NULL
        WITH count(sn) AS reviewed,
             avg(sn.reviewer_score) AS mean_score,
             percentileCont(sn.reviewer_score, 0.25) AS p25,
             percentileCont(sn.reviewer_score, 0.50) AS median,
             percentileCont(sn.reviewer_score, 0.75) AS p75,
             min(sn.reviewer_score) AS min_score,
             max(sn.reviewer_score) AS max_score
        RETURN reviewed, mean_score, p25, median, p75, min_score, max_score
    """)
    r = rows[0] if rows else None
    if r and r["reviewed"]:
        print(f"  Reviewed names:    {r['reviewed']}")
        print(f"  Mean score:        {r['mean_score']:.3f}")
        print(f"  Median:            {r['median']:.3f}")
        print(f"  P25 / P75:         {r['p25']:.3f} / {r['p75']:.3f}")
        print(f"  Min / Max:         {r['min_score']:.3f} / {r['max_score']:.3f}")

        # Bucket histogram
        buckets = gc.query("""
            MATCH (sn:StandardName)
            WHERE sn.reviewer_score IS NOT NULL
            WITH CASE
                   WHEN sn.reviewer_score < 0.3 THEN '[0.0, 0.3)'
                   WHEN sn.reviewer_score < 0.5 THEN '[0.3, 0.5)'
                   WHEN sn.reviewer_score < 0.7 THEN '[0.5, 0.7)'
                   WHEN sn.reviewer_score < 0.8 THEN '[0.7, 0.8)'
                   WHEN sn.reviewer_score < 0.9 THEN '[0.8, 0.9)'
                   ELSE '[0.9, 1.0]'
                 END AS bucket,
                 count(sn) AS cnt
            RETURN bucket, cnt
            ORDER BY bucket
        """)
        if buckets:
            max_cnt = max(b["cnt"] for b in buckets)
            print()
            for b in buckets:
                bar_len = int(b["cnt"] / max(max_cnt, 1) * 30)
                print(f"  {b['bucket']:<12} {b['cnt']:>5}  {'█' * bar_len}")
    else:
        print("  (no reviewer scores — run `sn review` first)")

    print()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    sys.exit(0)
