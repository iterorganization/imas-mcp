"""Plan 32 Phase 1 — LLM-score node_category audit samples.

Takes the JSON produced by ``sample_node_category_audit.py`` and asks an
LLM reviewer (opus-4.6 by default) whether each DD path is a plausible
Standard Name candidate.  Fills ``reviewer_verdict`` in-place.

The reviewer returns one of:
    - ``keep``: path *should* be an SN candidate (currently excluded ⇒ FN)
    - ``drop``: path correctly excluded (not a quantity worth a name)
    - ``borderline``: ambiguous / convention-dependent

Cost is accumulated across batches; aborts if the running total exceeds
``--budget`` (default $1.00).

Run::

    uv run python scripts/one_offs/score_node_category_audit.py \\
        --input  plans/research/data/node-category-samples.json \\
        --output plans/research/data/node-category-samples-scored.json \\
        --batch-size 20 --budget 1.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import call_llm_structured
from imas_codex.settings import get_sn_benchmark_reviewer_model

logger = logging.getLogger(__name__)


class PathVerdict(BaseModel):
    path: str
    verdict: Literal["keep", "drop", "borderline"]
    rationale: str = Field(..., max_length=400)


class BatchVerdicts(BaseModel):
    verdicts: list[PathVerdict]


SYSTEM = (
    "You are a senior plasma-physics data architect auditing the IMAS "
    "Data Dictionary for Standard-Name coverage.  A Standard Name is a "
    "short, physically-meaningful identifier for a measurable or "
    "computable physical quantity (e.g. `electron_temperature`, "
    "`plasma_current`, `magnetic_axis_radius`).  Standard Names are "
    "NOT assigned to: error/uncertainty fields, provenance/metadata "
    "(code version, comment strings, timestamps), bare coordinate "
    "arrays, identifier enumerations, or purely structural container "
    "nodes.  For each DD path, decide whether it SHOULD have a "
    "Standard Name.  Return `keep` if it should, `drop` if it should "
    "not, `borderline` if ambiguous or convention-dependent."
)


def score_batch(batch: list[dict], model: str) -> tuple[list[dict], float]:
    """Score one batch of paths; returns verdicts + cost."""
    user = (
        "Classify the following IMAS DD paths.  Reply with one entry "
        "per path, preserving input order.\n\n"
    )
    for i, row in enumerate(batch, 1):
        desc = (row.get("description") or "").strip() or "(no description)"
        unit = row.get("unit") or "—"
        user += (
            f"{i}. path: {row['path']}\n"
            f"   ids: {row['ids']}    node_category: {row['node_category']}    "
            f"node_type: {row.get('node_type', '—')}    unit: {unit}\n"
            f"   description: {desc[:300]}\n\n"
        )
    result = call_llm_structured(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        response_model=BatchVerdicts,
        service="standard-names",
        max_tokens=4000,
        temperature=0.0,
    )
    parsed, cost, _ = result
    by_path = {v.path: v for v in parsed.verdicts}
    out = []
    for row in batch:
        v = by_path.get(row["path"])
        if v is None:
            out.append({**row, "reviewer_verdict": "", "rationale": "(missing)"})
        else:
            out.append({**row, "reviewer_verdict": v.verdict, "rationale": v.rationale})
    return out, cost


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = json.loads(args.input.read_text())
    model = args.model or get_sn_benchmark_reviewer_model()
    logger.info("Using reviewer model: %s", model)

    samples = data["samples"]
    scored: list[dict] = []
    total_cost = 0.0
    for i in range(0, len(samples), args.batch_size):
        if total_cost >= args.budget:
            logger.warning(
                "Budget %.2f reached (cost=%.4f); stopping at batch %d",
                args.budget,
                total_cost,
                i // args.batch_size,
            )
            # Append un-scored remainder verbatim
            for row in samples[i:]:
                scored.append(
                    {**row, "reviewer_verdict": "", "rationale": "(unbudgeted)"}
                )
            break
        batch = samples[i : i + args.batch_size]
        out, cost = score_batch(batch, model)
        total_cost += cost
        scored.extend(out)
        logger.info(
            "Batch %d: %d paths, $%.4f (total $%.4f)",
            i // args.batch_size,
            len(batch),
            cost,
            total_cost,
        )

    data["samples"] = scored
    data["reviewer_model"] = model
    data["reviewer_cost_usd"] = round(total_cost, 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    logger.info(
        "Wrote %d scored samples → %s (cost $%.4f)",
        len(scored),
        args.output,
        total_cost,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
