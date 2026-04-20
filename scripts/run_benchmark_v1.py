"""Plan 34 — Benchmark v1 runner for Standard Name quality gating.

Loads ``tests/standard_names/eval_sets/benchmark_v1.json`` (or a
compatible file) and evaluates:

* **Positives** — invoke the compose prompt per DD path, compare
  generated name to ``expected_name`` (exact match), then score with
  the reviewer prompt.
* **Negatives** — hand the candidate name + DD context to the reviewer,
  confirm rejection (score < 0.5 OR validation flag).

Emits, under ``--output``:

- ``benchmark-v1.positives.jsonl``
- ``benchmark-v1.negatives.jsonl``
- ``benchmark-v1.summary.json``
- ``benchmark-v1-summary.md``

Gating thresholds (from plan 34 §3.4):

- pass@1   ≥ 0.80     (positives, exact match)
- mean_score ≥ 0.75   (positives, reviewer score)
- reject_rate ≥ 0.90  (negatives, score < 0.5 OR flagged)

Usage::

    uv run python scripts/run_benchmark_v1.py \\
        --eval-set tests/standard_names/eval_sets/benchmark_v1.json \\
        --output plans/research/data \\
        --cost-cap 5.00

Options:

- ``--limit N``        cap positives+negatives to first N each (dry-run)
- ``--mock``           skip LLM calls; fabricate canned outputs for
                        schema / pipeline validation.  Cost = $0.
- ``--model MODEL``    override compose+review model (default:
                        ``get_sn_benchmark_reviewer_model()``).

The runner is deliberately script-level (no CLI subcommand) to match
``scripts/prompt_ab_run.py``.  Promote to ``imas-codex sn benchmark
--set v1`` once the eval set is full and gating is wired into CI.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_sn_benchmark_reviewer_model

logger = logging.getLogger(__name__)


# ── Pydantic response models ───────────────────────────────────────────


class ComposeCandidate(BaseModel):
    path: str
    standard_name: str
    rationale: str = Field(..., max_length=400)


class ComposeBatch(BaseModel):
    candidates: list[ComposeCandidate]


class ReviewScore(BaseModel):
    identifier: str = Field(
        ...,
        description="Stable identifier for the item (DD path for positives, "
        "candidate name for negatives).",
    )
    standard_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["pass", "revise", "fail"]
    comment: str = Field(..., max_length=300)


class ReviewBatch(BaseModel):
    scores: list[ReviewScore]


# ── Prompts (condensed, mirror prompt_ab_run.py) ───────────────────────

GRAMMAR = """\
An IMAS Standard Name is a short, snake_case identifier for a
physically-meaningful quantity.

Grammar:
- Name = physical_base[+ modifiers].
- physical_base is a CONTROLLED vocabulary term (electron_temperature,
  poloidal_magnetic_flux, plasma_current, safety_factor, ...).
- Component/axis tokens use <axis>_component_of_<quantity> or go AFTER
  physical_base when a bare 'radial/toroidal/poloidal/vertical' prefix
  is the canonical form.
- Boundary/object-scoped quantities take a 'of_<object>' suffix
  (elongation_of_plasma_boundary, NOT bare 'elongation').
- NEVER include: units, abbreviations, symbols, IDS names, measurement
  methods, processing adjectives (filtered_, reconstructed_).
"""

COMPOSE_SYSTEM = (
    "You are a fusion-physics data architect composing IMAS Standard Names "
    "for Data Dictionary paths.  Unit is authoritative and injected at "
    "persistence time — never put unit, IDS name, or symbol in the "
    "output.  Prefer the controlled vocabulary.\n\n" + GRAMMAR
)


def _compose_user_message(items: list[dict]) -> str:
    lines = [
        "Generate one Standard Name per DD path.  Return JSON "
        "`{candidates: [{path, standard_name, rationale}]}`.  "
        "Rationale: one sentence citing the physical quantity.\n",
        "## Batch\n",
    ]
    for i, it in enumerate(items, 1):
        desc = (it.get("description") or "").strip()[:240] or "(no description)"
        lines.append(
            f"{i}. path: {it['path']}\n"
            f"   ids: {it['ids']}    physics_domain: {it.get('physics_domain', '—')}    "
            f"node_category: {it.get('node_category', '—')}    unit: {it.get('unit', 'dimensionless')}\n"
            f"   description: {desc}"
        )
        if it.get("cluster_siblings"):
            sibs = ", ".join(it["cluster_siblings"][:6])
            lines.append(f"   cluster_siblings: {sibs}")
        lines.append("")
    return "\n".join(lines)


REVIEW_SYSTEM = (
    "You are a Standard Name reviewer.  For each (DD path, proposed name) "
    "pair, score on a [0, 1] continuous rubric:\n"
    "  1.0 — grammar-compliant, controlled vocab, correct quantity.\n"
    "  0.8 — correct quantity, minor vocab slip.\n"
    "  0.5 — correct quantity but grammar/style violations (abbreviation, "
    "symbol, misplaced component, unit in name).\n"
    "  0.2 — wrong quantity/identity.\n"
    "  0.0 — nonsense, IDS leakage, empty.\n"
    "Emit verdict: pass (≥0.8), revise (0.5–0.79), fail (<0.5).\n"
    "Use the DD path identifier as the 'identifier' field."
)


def _review_user_positive(items: list[dict], candidates: list[ComposeCandidate]) -> str:
    by_path = {c.path: c for c in candidates}
    lines = [
        "Score each proposed Standard Name.  Return `{scores: [{identifier, "
        "standard_name, score, verdict, comment}]}`.  Comment ≤ 30 words.\n"
    ]
    for i, it in enumerate(items, 1):
        c = by_path.get(it["path"])
        proposed = c.standard_name if c else "(missing)"
        desc = (it.get("description") or "").strip()[:200] or "(no description)"
        lines.append(
            f"{i}. identifier: {it['path']}\n"
            f"   unit: {it.get('unit', '—')}    physics_domain: {it.get('physics_domain', '—')}\n"
            f"   description: {desc}\n"
            f"   proposed_name: {proposed}\n"
        )
    return "\n".join(lines)


def _review_user_negative(items: list[dict]) -> str:
    """Items carry candidate_name instead of being freshly generated."""
    lines = [
        "Score each proposed Standard Name.  These are candidate names "
        "proposed for the given DD path — reject those violating the "
        "grammar or controlled vocabulary.  Return `{scores: [{identifier, "
        "standard_name, score, verdict, comment}]}`.\n"
    ]
    for i, it in enumerate(items, 1):
        desc = (it.get("description") or "").strip()[:200] or "(no description)"
        lines.append(
            f"{i}. identifier: {it['candidate_name']}\n"
            f"   dd_path: {it.get('path', '—')}    unit: {it.get('unit', '—')}    "
            f"physics_domain: {it.get('physics_domain', '—')}\n"
            f"   description: {desc}\n"
            f"   proposed_name: {it['candidate_name']}\n"
        )
    return "\n".join(lines)


# ── Graph enrichment ───────────────────────────────────────────────────


def _enrich_positive(gc: GraphClient, item: dict) -> dict:
    path = item["dd_path"]
    rows = list(
        gc.query(
            """
            MATCH (n:IMASNode {id: $pid})-[:IN_IDS]->(ids:IDS)
            OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
            RETURN n.id AS path,
                   ids.id AS ids,
                   coalesce(n.description, '') AS description,
                   coalesce(u.id, 'dimensionless') AS unit,
                   coalesce(n.physics_domain, '') AS physics_domain,
                   coalesce(n.node_category, '') AS node_category
            """,
            pid=path,
        )
    )
    if rows:
        base = dict(rows[0])
    else:
        base = {
            "path": path,
            "ids": path.split("/", 1)[0],
            "description": "",
            "unit": item.get("expected_unit", "dimensionless"),
            "physics_domain": item.get("physics_domain", ""),
            "node_category": "",
        }
    # cluster siblings
    sib_rows = list(
        gc.query(
            """
            MATCH (n:IMASNode {id: $pid})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                  <-[:IN_CLUSTER]-(peer:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            WHERE peer.id <> n.id
            RETURN DISTINCT sn.id AS name LIMIT 6
            """,
            pid=path,
        )
    )
    base["cluster_siblings"] = [r["name"] for r in sib_rows]
    base["_expected"] = item
    return base


def _enrich_negative(gc: GraphClient, item: dict) -> dict:
    path = item.get("dd_path")
    base: dict = {
        "candidate_name": item["candidate_name"],
        "path": path,
        "ids": path.split("/", 1)[0] if path else "—",
        "physics_domain": item.get("physics_domain", ""),
        "anti_pattern_category": item.get("anti_pattern_category", ""),
        "rejection_reason": item.get("rejection_reason", ""),
        "description": "",
        "unit": "—",
    }
    if path:
        rows = list(
            gc.query(
                """
                MATCH (n:IMASNode {id: $pid})
                OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
                RETURN coalesce(n.description, '') AS description,
                       coalesce(u.id, 'dimensionless') AS unit
                """,
                pid=path,
            )
        )
        if rows:
            base["description"] = rows[0]["description"]
            base["unit"] = rows[0]["unit"]
    return base


# ── Mock helpers (for --mock) ──────────────────────────────────────────


def _mock_compose(items: list[dict]) -> ComposeBatch:
    return ComposeBatch(
        candidates=[
            ComposeCandidate(
                path=it["path"],
                standard_name=it["_expected"]["expected_name"],
                rationale="[mock] returned expected_name verbatim",
            )
            for it in items
        ]
    )


def _mock_review_positive(
    items: list[dict], candidates: list[ComposeCandidate]
) -> ReviewBatch:
    return ReviewBatch(
        scores=[
            ReviewScore(
                identifier=c.path,
                standard_name=c.standard_name,
                score=0.9,
                verdict="pass",
                comment="[mock] canned pass",
            )
            for c in candidates
        ]
    )


def _mock_review_negative(items: list[dict]) -> ReviewBatch:
    return ReviewBatch(
        scores=[
            ReviewScore(
                identifier=it["candidate_name"],
                standard_name=it["candidate_name"],
                score=0.15,
                verdict="fail",
                comment="[mock] canned rejection",
            )
            for it in items
        ]
    )


# ── Main ───────────────────────────────────────────────────────────────


def _filter_todo(positives: list[dict]) -> tuple[list[dict], int]:
    filled = [p for p in positives if p.get("dd_path") not in ("TODO", None, "")]
    skipped = len(positives) - len(filled)
    return filled, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-set",
        type=Path,
        default=Path("tests/standard_names/eval_sets/benchmark_v1.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("plans/research/data"),
    )
    ap.add_argument("--cost-cap", type=float, default=5.0)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument(
        "--limit", type=int, default=None, help="Cap positives/negatives for dry-run"
    )
    ap.add_argument("--mock", action="store_true", help="Skip LLM calls")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)
    model = args.model or get_sn_benchmark_reviewer_model()

    eval_data = json.loads(args.eval_set.read_text())
    positives_raw, todo_skipped = _filter_todo(eval_data.get("positives", []))
    negatives_raw = eval_data.get("negatives", [])

    if args.limit:
        positives_raw = positives_raw[: args.limit]
        negatives_raw = negatives_raw[: args.limit]

    logger.info(
        "Loaded eval set: %d filled positives (%d TODO skipped), %d negatives; mock=%s",
        len(positives_raw),
        todo_skipped,
        len(negatives_raw),
        args.mock,
    )

    # ── Enrich from graph (or synthesise if --mock and graph unreachable)
    total_cost = 0.0
    positives: list[dict] = []
    negatives: list[dict] = []

    try:
        with GraphClient() as gc:
            positives = [_enrich_positive(gc, it) for it in positives_raw]
            negatives = [_enrich_negative(gc, it) for it in negatives_raw]
    except Exception as exc:
        if not args.mock:
            raise
        logger.warning("Graph unavailable (%s); synthesising minimal enrichment.", exc)
        for it in positives_raw:
            positives.append(
                {
                    "path": it["dd_path"],
                    "ids": it["dd_path"].split("/", 1)[0],
                    "description": "",
                    "unit": it.get("expected_unit", "dimensionless"),
                    "physics_domain": it.get("physics_domain", ""),
                    "node_category": "",
                    "cluster_siblings": [],
                    "_expected": it,
                }
            )
        for it in negatives_raw:
            negatives.append(
                {
                    "candidate_name": it["candidate_name"],
                    "path": it.get("dd_path"),
                    "ids": (it.get("dd_path", "") or "").split("/", 1)[0] or "—",
                    "physics_domain": it.get("physics_domain", ""),
                    "anti_pattern_category": it.get("anti_pattern_category", ""),
                    "rejection_reason": it.get("rejection_reason", ""),
                    "description": "",
                    "unit": "—",
                }
            )

    # ── Compose (positives only) ───────────────────────────────────────
    if positives:
        if args.mock:
            compose_parsed = _mock_compose(positives)
            compose_cost = 0.0
            compose_tokens = 0
            compose_latency = 0.0
        else:
            if total_cost >= args.cost_cap:
                logger.error("Cost cap reached before compose step")
                return 2
            t0 = time.time()
            compose_parsed, compose_cost, compose_tokens = call_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": COMPOSE_SYSTEM},
                    {"role": "user", "content": _compose_user_message(positives)},
                ],
                response_model=ComposeBatch,
                service="standard-names",
                max_tokens=4000,
                temperature=0.0,
            )
            compose_latency = time.time() - t0
            total_cost += compose_cost
        logger.info(
            "compose: %d candidates | cost=$%.4f tokens=%d latency=%.1fs (total=$%.4f)",
            len(compose_parsed.candidates),
            compose_cost,
            compose_tokens,
            compose_latency,
            total_cost,
        )
    else:
        compose_parsed = ComposeBatch(candidates=[])
        compose_cost = 0.0

    # ── Review positives ───────────────────────────────────────────────
    if positives:
        if args.mock:
            pos_review = _mock_review_positive(positives, compose_parsed.candidates)
            pos_review_cost = 0.0
        elif total_cost < args.cost_cap:
            t0 = time.time()
            pos_review, pos_review_cost, _ = call_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM},
                    {
                        "role": "user",
                        "content": _review_user_positive(
                            positives, compose_parsed.candidates
                        ),
                    },
                ],
                response_model=ReviewBatch,
                service="standard-names",
                max_tokens=4000,
                temperature=0.0,
            )
            total_cost += pos_review_cost
            logger.info(
                "review positives: cost=$%.4f latency=%.1fs (total=$%.4f)",
                pos_review_cost,
                time.time() - t0,
                total_cost,
            )
        else:
            logger.warning("Cost cap hit; skipping positive review")
            pos_review = ReviewBatch(scores=[])
            pos_review_cost = 0.0
    else:
        pos_review = ReviewBatch(scores=[])
        pos_review_cost = 0.0

    # ── Review negatives ───────────────────────────────────────────────
    if negatives:
        if args.mock:
            neg_review = _mock_review_negative(negatives)
            neg_review_cost = 0.0
        elif total_cost < args.cost_cap:
            t0 = time.time()
            neg_review, neg_review_cost, _ = call_llm_structured(
                model=model,
                messages=[
                    {"role": "system", "content": REVIEW_SYSTEM},
                    {"role": "user", "content": _review_user_negative(negatives)},
                ],
                response_model=ReviewBatch,
                service="standard-names",
                max_tokens=4000,
                temperature=0.0,
            )
            total_cost += neg_review_cost
            logger.info(
                "review negatives: cost=$%.4f latency=%.1fs (total=$%.4f)",
                neg_review_cost,
                time.time() - t0,
                total_cost,
            )
        else:
            logger.warning("Cost cap hit; skipping negative review")
            neg_review = ReviewBatch(scores=[])
            neg_review_cost = 0.0
    else:
        neg_review = ReviewBatch(scores=[])
        neg_review_cost = 0.0

    # ── Aggregate per-positive results ─────────────────────────────────
    compose_by_path = {c.path: c for c in compose_parsed.candidates}
    review_by_path = {s.identifier: s for s in pos_review.scores}

    pos_rows: list[dict] = []
    for it in positives:
        path = it["path"]
        expected = it["_expected"]
        cand = compose_by_path.get(path)
        score = review_by_path.get(path)
        generated = cand.standard_name if cand else ""
        exact = generated == expected["expected_name"]
        pos_rows.append(
            {
                "path": path,
                "physics_domain": expected.get("physics_domain"),
                "expected_name": expected.get("expected_name"),
                "generated_name": generated,
                "exact_match": exact,
                "reviewer_score": score.score if score else None,
                "reviewer_verdict": score.verdict if score else None,
                "reviewer_comment": score.comment if score else None,
                "source": expected.get("source"),
            }
        )

    # ── Aggregate per-negative results ─────────────────────────────────
    review_by_name = {s.identifier: s for s in neg_review.scores}
    neg_rows: list[dict] = []
    for it in negatives:
        name = it["candidate_name"]
        score = review_by_name.get(name)
        rejected = bool(score and (score.score < 0.5 or score.verdict == "fail"))
        neg_rows.append(
            {
                "candidate_name": name,
                "anti_pattern_category": it.get("anti_pattern_category"),
                "physics_domain": it.get("physics_domain"),
                "reviewer_score": score.score if score else None,
                "reviewer_verdict": score.verdict if score else None,
                "reviewer_comment": score.comment if score else None,
                "rejected": rejected,
                "rejection_reason_expected": it.get("rejection_reason"),
            }
        )

    # ── Write JSONL ────────────────────────────────────────────────────
    pos_jsonl = args.output / "benchmark-v1.positives.jsonl"
    with pos_jsonl.open("w") as f:
        for r in pos_rows:
            f.write(json.dumps(r) + "\n")
    neg_jsonl = args.output / "benchmark-v1.negatives.jsonl"
    with neg_jsonl.open("w") as f:
        for r in neg_rows:
            f.write(json.dumps(r) + "\n")

    # ── Gating ─────────────────────────────────────────────────────────
    scored_positives = [r for r in pos_rows if r["reviewer_score"] is not None]
    n_pos = len(pos_rows) or 1
    exact_rate = sum(1 for r in pos_rows if r["exact_match"]) / n_pos
    mean_score = (
        sum(r["reviewer_score"] for r in scored_positives) / len(scored_positives)
        if scored_positives
        else 0.0
    )
    n_neg = len(neg_rows) or 1
    reject_rate = sum(1 for r in neg_rows if r["rejected"]) / n_neg

    thresholds = eval_data.get("_meta", {}).get("gating", {})
    gate_pass_at_1 = thresholds.get("positives_pass_at_1", 0.80)
    gate_mean_score = thresholds.get("positives_mean_reviewer_score", 0.75)
    gate_reject_rate = thresholds.get("negatives_rejection_rate", 0.90)

    gates = {
        "pass_at_1": {
            "actual": round(exact_rate, 3),
            "threshold": gate_pass_at_1,
            "passed": exact_rate >= gate_pass_at_1,
        },
        "mean_reviewer_score": {
            "actual": round(mean_score, 3),
            "threshold": gate_mean_score,
            "passed": mean_score >= gate_mean_score,
        },
        "negatives_rejection_rate": {
            "actual": round(reject_rate, 3),
            "threshold": gate_reject_rate,
            "passed": reject_rate >= gate_reject_rate,
        },
    }
    all_passed = all(g["passed"] for g in gates.values())

    # Per-domain breakdown
    per_domain: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "exact": 0, "score_sum": 0.0, "score_n": 0}
    )
    for r in pos_rows:
        d = r["physics_domain"] or "?"
        per_domain[d]["n"] += 1
        per_domain[d]["exact"] += int(r["exact_match"])
        if r["reviewer_score"] is not None:
            per_domain[d]["score_sum"] += r["reviewer_score"]
            per_domain[d]["score_n"] += 1
    per_domain_out = {
        d: {
            "n": v["n"],
            "exact_rate": round(v["exact"] / v["n"], 3) if v["n"] else None,
            "mean_score": round(v["score_sum"] / v["score_n"], 3)
            if v["score_n"]
            else None,
        }
        for d, v in per_domain.items()
    }

    summary = {
        "model": model,
        "mock": args.mock,
        "cost_usd": round(total_cost, 4),
        "positives_total": len(pos_rows),
        "positives_todo_skipped": todo_skipped,
        "negatives_total": len(neg_rows),
        "gates": gates,
        "all_gates_passed": all_passed,
        "per_domain": per_domain_out,
    }
    (args.output / "benchmark-v1.summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # Markdown summary
    md = ["# Benchmark v1 — run summary", ""]
    md.append(f"- model: `{model}`  mock: `{args.mock}`  cost: `${total_cost:.4f}`")
    md.append(
        f"- positives: {len(pos_rows)} evaluated ({todo_skipped} TODO stubs skipped); "
        f"negatives: {len(neg_rows)}"
    )
    md.append("")
    md.append("## Gates")
    md.append("")
    md.append("| Gate | Actual | Threshold | Pass |")
    md.append("|------|-------:|----------:|:----:|")
    for name, g in gates.items():
        md.append(
            f"| {name} | {g['actual']} | {g['threshold']} | {'✅' if g['passed'] else '❌'} |"
        )
    md.append("")
    md.append(
        f"**Overall:** {'✅ all gates passed' if all_passed else '❌ one or more gates failed'}"
    )
    md.append("")
    md.append("## Per-domain breakdown (positives)")
    md.append("")
    md.append("| Domain | n | exact_rate | mean_score |")
    md.append("|--------|--:|-----------:|-----------:|")
    for d, v in sorted(per_domain_out.items()):
        md.append(f"| {d} | {v['n']} | {v['exact_rate']} | {v['mean_score']} |")
    md.append("")
    md.append("## Negative rejections")
    md.append("")
    md.append("| Category | Candidate | Score | Rejected |")
    md.append("|----------|-----------|------:|:--------:|")
    for r in neg_rows:
        md.append(
            f"| {r['anti_pattern_category']} | `{r['candidate_name']}` | "
            f"{r['reviewer_score']} | {'✅' if r['rejected'] else '❌'} |"
        )
    md.append("")
    md.append(f"Outputs: `{pos_jsonl}`, `{neg_jsonl}`, `benchmark-v1.summary.json`.")
    (args.output / "benchmark-v1-summary.md").write_text("\n".join(md))

    logger.info("Wrote %s", args.output / "benchmark-v1-summary.md")
    logger.info("Gates: %s", json.dumps(gates, indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
