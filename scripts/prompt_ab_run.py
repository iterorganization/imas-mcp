"""Plan 32 Phase 2 — A/B/C compose prompt bake-off harness (implementation).

See ``scripts/prompt_ab.py`` for the original scaffold.  This script is
the focused, budget-capped runner that:

1. Pulls DD metadata for each eval-set path from the graph.
2. Builds three variants of the compose prompt:
   - **A (baseline, rich)**: full grammar + exemplars + cluster siblings.
   - **B (name-only, lean)**: grammar + unit policy only.
   - **C (tool-calling prompt, no tool loop)**: variant-C prompt
     rendered as-is; tool-calling is declared in the prompt but *not*
     wired (the loop is out of scope here and the quality signal of
     the prompt design itself is what we measure).
3. Calls ``anthropic/claude-opus-4.6`` once per variant (all 20 paths
   batched in a single request) via ``call_llm_structured``.
4. Runs a review pass with ``sn/review_name_only.md`` content
   (embedded) over all 60 names, also in one opus call per variant.
5. Writes per-variant JSONL + aggregate CSV + summary markdown.

Output files::

    plans/research/data/prompt-ab-v1.{A,B,C}.jsonl
    plans/research/data/prompt-ab-v1.csv
    plans/research/prompt-ab-results.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.graph.client import GraphClient
from imas_codex.settings import get_sn_benchmark_reviewer_model

logger = logging.getLogger(__name__)


# ── Pydantic models ────────────────────────────────────────────────────


class Candidate(BaseModel):
    path: str
    standard_name: str
    rationale: str = Field(..., max_length=400)


class ComposeBatch(BaseModel):
    candidates: list[Candidate]


class ReviewScore(BaseModel):
    path: str
    standard_name: str
    score: float = Field(..., ge=0.0, le=1.0)
    verdict: Literal["pass", "revise", "fail"]
    comment: str = Field(..., max_length=300)


class ReviewBatch(BaseModel):
    scores: list[ReviewScore]


# ── Prompts (compressed, self-contained) ───────────────────────────────

GRAMMAR = """\
An IMAS Standard Name is a short, snake_case identifier for a
physically-meaningful quantity.

Grammar:
- Name = physical_base[+ modifiers].
- physical_base is a CONTROLLED vocabulary term; examples: electron_temperature,
  electron_density, ion_temperature, poloidal_magnetic_flux, plasma_current,
  safety_factor, radial_electric_field, particle_flux, major_radius,
  minor_radius, elongation, triangularity_upper, triangularity_lower,
  magnetic_axis_radius, magnetic_axis_height.
- Positional/species/orientation tokens go AFTER physical_base:
  e.g. electron_temperature_core, particle_flux_radial,
  magnetic_field_toroidal, magnetic_axis_radius.
- NEVER include in a Standard Name: units, abbreviations, symbols,
  IDS names, measurement methods, processing adjectives
  (filtered_, reconstructed_, averaged_), or DD section names.
- Geometric coordinates of named entities use the pattern
  <entity>_<axis>_coordinate (e.g. plasma_boundary_r_coordinate,
  magnetics_probe_r_coordinate).
- Constant/metadata fields (pulse numbers, toroidal reference radius R0,
  code version) should be named descriptively
  (e.g. reference_major_radius for vacuum_toroidal_field/r0).
"""

SYSTEM_PROMPT = (
    "You are a fusion-physics data architect composing Standard Names "
    "for IMAS Data Dictionary paths.  Unit is authoritative and injected "
    "at persistence time — never put unit or IDS name in the output.  "
    "Prefer controlled vocabulary; reuse existing names when the path "
    "measures the same quantity.\n\n" + GRAMMAR
)


def _path_block(items: list[dict], *, include_siblings: bool) -> str:
    """Render the batch of paths for the user message."""
    lines: list[str] = []
    for i, it in enumerate(items, 1):
        desc = (it.get("description") or "").strip() or "(no description)"
        unit = it.get("unit") or "dimensionless"
        lines.append(
            f"{i}. path: {it['path']}\n"
            f"   ids: {it['ids']}    physics_domain: {it.get('physics_domain', '—')}    "
            f"node_category: {it.get('node_category', '—')}    unit: {unit}\n"
            f"   description: {desc[:280]}"
        )
        if include_siblings and it.get("cluster_siblings"):
            sibs = ", ".join(it["cluster_siblings"][:6])
            lines.append(f"   cluster_siblings: {sibs}")
        lines.append("")
    return "\n".join(lines)


def _user_message_A(items: list[dict]) -> str:
    return (
        "## Variant A — full compose context\n\n"
        "Generate one Standard Name per path.  The batch shares an "
        "authoritative unit family; consult cluster_siblings as evidence "
        "for the controlled vocabulary.  Typical anti-patterns to avoid:\n"
        " - ❌ electron_temp   ✅ electron_temperature\n"
        " - ❌ core_electron_temperature   ✅ electron_temperature_core\n"
        " - ❌ Te / B_t / q   (no symbols)\n"
        " - ❌ poloidal_flux   ✅ poloidal_magnetic_flux\n"
        " - ❌ reconstructed_<x>, filtered_<x>   (processing metadata)\n"
        " - ❌ plasma_current_IP   ✅ plasma_current\n\n"
        "Return one `{path, standard_name, rationale}` per input path. "
        "Rationale must cite the physical quantity in one sentence.\n\n"
        "## Batch\n\n" + _path_block(items, include_siblings=True)
    )


def _user_message_B(items: list[dict]) -> str:
    return (
        "## Variant B — name-only lean context\n\n"
        "Generate one Standard Name per path.  Unit is authoritative "
        "and already pre-populated.  Keep names grammar-compliant: "
        "lowercase snake_case, no units/symbols/IDS names, position "
        "tokens after physical base.\n\n"
        "Return one `{path, standard_name, rationale}` per input path. "
        "Rationale: one sentence, naming the physical quantity.\n\n"
        "## Batch\n\n" + _path_block(items, include_siblings=False)
    )


def _user_message_C(items: list[dict]) -> str:
    # Mirrors the variant-C prompt but inlined; tool calls declared
    # rhetorically only (no tool loop).
    return (
        "## Variant C — lean prompt, tool-calling policy declared\n\n"
        "You have access to three *hypothetical* tools:\n"
        " - fetch_cluster_siblings(cluster_id)\n"
        " - fetch_reference_exemplar(concept)\n"
        " - fetch_version_history(path)\n\n"
        "Use them sparingly and prefer direct naming when the physical "
        "identity is obvious.  (In this harness run tool calls are not "
        "executed; emit a name directly and, if you *would* have "
        "called a tool, record that in rationale — one tool per item max.)\n\n"
        "Return one `{path, standard_name, rationale}` per input path. "
        "Rationale must cite the physical quantity in one sentence.\n\n"
        "## Batch\n\n" + _path_block(items, include_siblings=False)
    )


USER_BUILDERS = {"A": _user_message_A, "B": _user_message_B, "C": _user_message_C}


# ── Review prompt ──────────────────────────────────────────────────────

REVIEW_SYSTEM = (
    "You are a Standard Name reviewer.  For each (path, proposed name) "
    "pair, score the proposed Standard Name on a [0, 1] continuous "
    "rubric:\n"
    "  1.0 — grammar-compliant, uses controlled vocabulary, correct "
    "physical quantity, no DD/IDS/method leakage.\n"
    "  0.8 — correct quantity, minor controlled-vocab slip "
    "(e.g. poloidal_flux instead of poloidal_magnetic_flux).\n"
    "  0.5 — correct quantity but clear grammar/style violations "
    "(abbreviation, symbol, position before base, unit in name).\n"
    "  0.2 — wrong quantity/identity, only superficially similar.\n"
    "  0.0 — nonsense, includes IDS name, or empty.\n\n"
    "Also emit a verdict: pass (≥0.8), revise (0.5–0.79), fail (<0.5)."
)


def review_user(items: list[dict], candidates: list[Candidate]) -> str:
    by_path = {c.path: c for c in candidates}
    lines = [
        "Score each proposed Standard Name.  Return `{path, standard_name, "
        "score, verdict, comment}` per entry.  Comment: ≤ 30 words.\n"
    ]
    for i, it in enumerate(items, 1):
        c = by_path.get(it["path"])
        proposed = c.standard_name if c else "(missing)"
        desc = (it.get("description") or "").strip()[:200] or "(no description)"
        lines.append(
            f"{i}. path: {it['path']}\n"
            f"   unit: {it.get('unit', '—')}    ids: {it['ids']}\n"
            f"   description: {desc}\n"
            f"   proposed_name: {proposed}\n"
        )
    return "\n".join(lines)


# ── Graph metadata fetch ───────────────────────────────────────────────


def _enrich_items(gc: GraphClient, eval_items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in eval_items:
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
                pid=it["path"],
            )
        )
        base = (
            rows[0]
            if rows
            else {
                "path": it["path"],
                "ids": it["path"].split("/", 1)[0],
                "description": "",
                "unit": "dimensionless",
                "physics_domain": it.get("physics_domain", ""),
                "node_category": "",
            }
        )
        # pull siblings from the first cluster the node participates in
        sib_rows = list(
            gc.query(
                """
                MATCH (n:IMASNode {id: $pid})-[:IN_CLUSTER]->(c:IMASSemanticCluster)
                      <-[:IN_CLUSTER]-(peer:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
                WHERE peer.id <> n.id
                RETURN DISTINCT sn.id AS name LIMIT 6
                """,
                pid=it["path"],
            )
        )
        base["cluster_siblings"] = [r["name"] for r in sib_rows]
        base["path_kind"] = it.get("path_kind", "")
        out.append(base)
    return out


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-set",
        type=Path,
        default=Path("tests/standard_names/eval_sets/prompt_ab_v1.json"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("plans/research/data"),
    )
    ap.add_argument("--cost-cap", type=float, default=3.0)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model = args.model or get_sn_benchmark_reviewer_model()
    args.output.mkdir(parents=True, exist_ok=True)

    # 1. Load + enrich
    eval_data = json.loads(args.eval_set.read_text())
    with GraphClient() as gc:
        items = _enrich_items(gc, eval_data["items"])
    logger.info("Enriched %d items", len(items))

    total_cost = 0.0
    results: dict[str, dict] = {}

    # 2. Compose per variant
    for variant in ["A", "B", "C"]:
        if total_cost >= args.cost_cap:
            logger.warning(
                "Cost cap %.2f hit; skipping variant %s", args.cost_cap, variant
            )
            continue
        user = USER_BUILDERS[variant](items)
        t0 = time.time()
        parsed, cost, tokens = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            response_model=ComposeBatch,
            service="standard-names",
            max_tokens=4000,
            temperature=0.0,
        )
        latency = time.time() - t0
        total_cost += cost
        logger.info(
            "variant=%s compose  cost=$%.4f tokens=%d latency=%.1fs  total=$%.4f",
            variant,
            cost,
            tokens,
            latency,
            total_cost,
        )
        results[variant] = {
            "candidates": parsed.candidates,
            "compose_cost": cost,
            "compose_tokens": tokens,
            "compose_latency": latency,
        }

    # 3. Review per variant
    for variant, res in results.items():
        if total_cost >= args.cost_cap:
            logger.warning("Cost cap hit; skipping review for %s", variant)
            res["scores"] = []
            res["review_cost"] = 0.0
            continue
        r_user = review_user(items, res["candidates"])
        t0 = time.time()
        parsed, cost, _ = call_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM},
                {"role": "user", "content": r_user},
            ],
            response_model=ReviewBatch,
            service="standard-names",
            max_tokens=4000,
            temperature=0.0,
        )
        total_cost += cost
        latency = time.time() - t0
        logger.info(
            "variant=%s review   cost=$%.4f latency=%.1fs  total=$%.4f",
            variant,
            cost,
            latency,
            total_cost,
        )
        res["scores"] = parsed.scores
        res["review_cost"] = cost

    # 4. Emit JSONL + CSV
    csv_rows: list[dict] = []
    for variant, res in results.items():
        by_path = {s.path: s for s in res.get("scores", [])}
        out_jsonl = args.output / f"prompt-ab-v1.{variant}.jsonl"
        with out_jsonl.open("w") as f:
            for c in res["candidates"]:
                s = by_path.get(c.path)
                row = {
                    "variant": variant,
                    "path": c.path,
                    "standard_name": c.standard_name,
                    "rationale": c.rationale,
                    "reviewer_score": s.score if s else None,
                    "reviewer_verdict": s.verdict if s else None,
                    "reviewer_comment": s.comment if s else None,
                }
                f.write(json.dumps(row) + "\n")
                csv_rows.append(row)

    csv_path = args.output / "prompt-ab-v1.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = [
            "variant",
            "path",
            "standard_name",
            "reviewer_score",
            "reviewer_verdict",
            "reviewer_comment",
            "rationale",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)

    # 5. Summary
    summary = {
        "model": model,
        "total_cost_usd": round(total_cost, 4),
        "per_variant": {},
    }
    for variant, res in results.items():
        scores = [s.score for s in res.get("scores", [])]
        verdicts = [s.verdict for s in res.get("scores", [])]
        summary["per_variant"][variant] = {
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "pass_at_1": round(
                sum(1 for v in verdicts if v == "pass") / len(verdicts), 3
            )
            if verdicts
            else None,
            "compose_cost": round(res["compose_cost"], 4),
            "compose_tokens": res["compose_tokens"],
            "compose_latency_s": round(res["compose_latency"], 2),
            "review_cost": round(res.get("review_cost", 0.0), 4),
            "n_candidates": len(res["candidates"]),
        }
    summary_path = args.output / "prompt-ab-v1.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary: %s", json.dumps(summary["per_variant"], indent=2))
    logger.info("Total cost: $%.4f", total_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
