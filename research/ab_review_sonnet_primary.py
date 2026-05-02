"""A/B test: quality-cost-balanced (Sonnet primary) vs default (Opus primary).

Compares Sonnet 4.6 cycle-0 review scores against the stored Opus 4.6 reviews
on a fixed sample of accepted standard names. NO graph writes — strictly
read-only sampling + parallel LLM calls + console report.

Output: per-dim score divergence, tier agreement, mean cost, dispute rate.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time

from imas_codex.discovery.base.llm import acall_llm_structured
from imas_codex.graph.client import GraphClient
from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names.context import _build_enum_lists
from imas_codex.standard_names.models import StandardNameQualityReviewNameOnly

CANDIDATE_MODEL = "openrouter/anthropic/claude-sonnet-4.6"
SYSTEM = "You are a quality reviewer for IMAS standard names in fusion plasma physics."
TIER_THRESHOLDS = [(0.85, "A"), (0.70, "B"), (0.50, "C"), (0.0, "D")]
DIMS = ("grammar", "semantic", "convention", "completeness")
DOMAIN = os.environ.get("AB_DOMAIN", "magnetic_field_diagnostics")
SAMPLE_SIZE = int(os.environ.get("AB_SAMPLE", "30"))
MAX_CONCURRENCY = int(os.environ.get("AB_CONCURRENCY", "6"))


def tier(score: float) -> str:
    for th, label in TIER_THRESHOLDS:
        if score >= th:
            return label
    return "D"


def fetch_sample(domain: str, n: int) -> list[dict]:
    cypher = """
        MATCH (sn:StandardName)
        WHERE sn.name_stage = 'accepted'
          AND sn.reviewer_score_name IS NOT NULL
          AND sn.reviewer_model_name CONTAINS 'opus'
          AND $domain IN coalesce(sn.physics_domain, [])
        OPTIONAL MATCH (sn)-[:HAS_UNIT]->(u:Unit)
        RETURN sn.id AS id, sn.name AS name,
               sn.description AS description,
               sn.documentation AS documentation,
               sn.kind AS kind,
               coalesce(u.id, sn.unit) AS unit,
               sn.tags AS tags,
               sn.physics_domain AS physics_domain,
               coalesce(sn.chain_length, 0) AS chain_length,
               sn.reviewer_score_name AS opus_score,
               sn.reviewer_scores_name AS opus_scores
        ORDER BY sn.id
        LIMIT $n
    """
    with GraphClient() as g, g.session() as s:
        return [dict(r) for r in s.run(cypher, domain=domain, n=n)]


async def review_one(item: dict, sem: asyncio.Semaphore) -> dict:
    ctx = {"items": [item], **_build_enum_lists()}
    user = render_prompt("sn/review_name_only", ctx)
    async with sem:
        t0 = time.time()
        try:
            res = await acall_llm_structured(
                model=CANDIDATE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_model=StandardNameQualityReviewNameOnly,
                service="standard-names",
            )
            parsed, cost, _tokens = res
            elapsed = time.time() - t0
            return {
                "id": item["id"],
                "ok": True,
                "cost": cost,
                "elapsed": elapsed,
                "scores": {d: getattr(parsed.scores, d) for d in DIMS},
                "score": parsed.scores.score,
            }
        except Exception as e:
            return {
                "id": item["id"],
                "ok": False,
                "error": f"{type(e).__name__}: {e!s}"[:200],
                "cost": 0.0,
            }


def parse_opus_scores(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, dict):
        return {d: int(raw.get(d, 0)) for d in DIMS}
    return None


def normalize(scores: dict) -> float:
    return sum(scores.values()) / 80.0


async def main():
    sample = fetch_sample(DOMAIN, SAMPLE_SIZE)
    print(f"Sampled {len(sample)} accepted names from domain={DOMAIN}", flush=True)
    if not sample:
        print("No sample names — aborting", file=sys.stderr)
        sys.exit(2)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    results = await asyncio.gather(*[review_one(it, sem) for it in sample])

    # Index existing Opus scores
    opus_by_id = {it["id"]: parse_opus_scores(it.get("opus_scores")) for it in sample}
    opus_score_by_id = {it["id"]: it.get("opus_score") for it in sample}

    # Compare
    per_dim_diff = {d: [] for d in DIMS}
    overall_diff = []
    tier_agree = 0
    n_compared = 0
    sonnet_costs = []
    sonnet_errors = []
    sonnet_below_thresh = 0  # would trigger cycle 2 (escalation) at 0.20

    for r in results:
        if not r.get("ok"):
            sonnet_errors.append(r)
            continue
        sonnet_costs.append(r["cost"])
        sid = r["id"]
        opus = opus_by_id.get(sid)
        opus_total = opus_score_by_id.get(sid)
        if opus is None or opus_total is None:
            continue
        n_compared += 1
        sonnet_norm = r["score"]
        opus_norm = float(opus_total)
        overall_diff.append(abs(sonnet_norm - opus_norm))
        if tier(sonnet_norm) == tier(opus_norm):
            tier_agree += 1
        for d in DIMS:
            per_dim_diff[d].append(abs(r["scores"][d] - opus[d]) / 20.0)
        # Disagreement detection mirrors review pipeline (max-min diff across reviewers)
        # We only have 1 vs 1 here, so just flag if abs diff >= 0.20 normalized.
        if abs(sonnet_norm - opus_norm) >= 0.20:
            sonnet_below_thresh += 1

    # ── Report ─────────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print(f"A/B test: candidate model = {CANDIDATE_MODEL}")
    print("Reference: openrouter/anthropic/claude-opus-4.6 (cycle 0 of default)")
    print("─" * 70)
    print(f"Sample size:           {len(sample)}")
    print(f"Sonnet calls OK:       {len(sonnet_costs)}")
    print(f"Sonnet errors:         {len(sonnet_errors)}")
    print(f"Compared with Opus:    {n_compared}")
    print()
    if n_compared:
        print("Per-dim |Δ| (normalized to 0-1, lower=better):")
        for d in DIMS:
            vals = per_dim_diff[d]
            print(
                f"  {d:14s}  mean={statistics.mean(vals):.3f}  "
                f"median={statistics.median(vals):.3f}  max={max(vals):.3f}"
            )
        flat = [v for vs in per_dim_diff.values() for v in vs]
        print(
            f"  ALL DIMS:       mean={statistics.mean(flat):.3f}  max={max(flat):.3f}"
        )
        print()
        print(
            f"Overall score |Δ|:   mean={statistics.mean(overall_diff):.3f}  "
            f"max={max(overall_diff):.3f}"
        )
        print(
            f"Tier agreement:      {tier_agree}/{n_compared} = "
            f"{tier_agree * 100 / n_compared:.1f}%"
        )
        print(
            f"Disagreement >=0.20: {sonnet_below_thresh}/{n_compared} "
            f"({sonnet_below_thresh * 100 / n_compared:.1f}%) — escalates to Opus"
        )
    print()
    if sonnet_costs:
        sonnet_total = sum(sonnet_costs)
        sonnet_mean = statistics.mean(sonnet_costs)
        print(
            f"Sonnet cost/call:    mean=${sonnet_mean:.4f}  total=${sonnet_total:.4f}"
        )
        # Simulated quality-cost-balanced cost: cycle0 always, cycle1 always,
        # cycle2 only on disagreement. Assume Opus arbiter ≈ $0.044/call,
        # GPT-5.4 ≈ $0.020/call.
        n_ok = len(sonnet_costs)
        gpt_est = n_ok * 0.020
        opus_est = sonnet_below_thresh * 0.044
        sim_total = sonnet_total + gpt_est + opus_est
        sim_per_name = sim_total / n_ok if n_ok else 0.0
        # default-profile reference: ~$0.044 + ~$0.020 + (disagreement rate)*0.043
        # Approximate observed default avg = $0.044/name (review_name pool avg).
        default_per_name = 0.044
        ratio = sim_per_name / default_per_name if default_per_name else 0.0
        print("Quality-cost-balanced (simulated full-chain):")
        print(f"  Sonnet primary:     ${sonnet_total:.3f}")
        print(f"  GPT-5.4 secondary:  ${gpt_est:.3f}  (estimate)")
        print(
            f"  Opus arbiter:       ${opus_est:.3f}  (only on {sonnet_below_thresh} disputes)"
        )
        print(f"  Total:              ${sim_total:.3f}  → ${sim_per_name:.4f}/name")
        print(f"  vs default:         ${default_per_name:.4f}/name → ratio {ratio:.2f}")
        print(f"  Estimated reduction: {(1 - ratio) * 100:.0f}%")
    if sonnet_errors:
        print(f"\nErrors ({len(sonnet_errors)}):")
        for e in sonnet_errors[:5]:
            print(f"  {e['id']}: {e.get('error')}")

    # ── Acceptance criteria summary ────────────────────────────────────
    print()
    print("─" * 70)
    if n_compared:
        flat = [v for vs in per_dim_diff.values() for v in vs]
        per_dim_mean = statistics.mean(flat)
        tier_pct = tier_agree * 100 / n_compared
        ratio = sim_per_name / default_per_name if default_per_name else 1.0
        crit = []
        crit.append(
            ("per-dim mean |Δ| ≤ 0.05", per_dim_mean <= 0.05, f"{per_dim_mean:.3f}")
        )
        crit.append(("tier agreement ≥ 90%", tier_pct >= 90.0, f"{tier_pct:.1f}%"))
        crit.append(("net cost ≤ 40% of default", ratio <= 0.40, f"{ratio * 100:.0f}%"))
        for label, ok, val in crit:
            mark = "✓" if ok else "✗"
            print(f"  {mark} {label:35s}  observed: {val}")
        all_ok = all(c[1] for c in crit)
        ambiguous = (
            (0.05 < per_dim_mean <= 0.08)
            or (85 <= tier_pct < 90)
            or (0.40 < ratio <= 0.55)
        )
        print()
        if all_ok:
            print("VERDICT: All criteria met → SAFE to adopt as default.")
        elif ambiguous:
            print("VERDICT: Ambiguous → DO NOT change default; report for review.")
        else:
            print("VERDICT: Fails one or more hard criteria → KEEP default.")


if __name__ == "__main__":
    asyncio.run(main())
