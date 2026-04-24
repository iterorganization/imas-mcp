#!/usr/bin/env python3
"""Reviewer alignment verification script.

Invokes the review pipeline once with a synthetic SN candidate and prints
a markdown table of model × score × cost.  Designed as a post-deployment
smoke-test.

Usage:
    uv run python scripts/verify_reviewer_alignment.py

Budget ceiling: $0.05 per run (hard-coded guard).
"""

from __future__ import annotations

import asyncio
import sys
import time

# Hard budget guard — abort before making any calls if budget exceeded.
_BUDGET_CEILING_USD = 0.25


async def _call_single_reviewer(
    model: str,
    candidate_name: str,
    candidate_description: str,
    candidate_unit: str,
) -> dict:
    """Invoke one reviewer model on a single synthetic SN candidate."""
    from imas_codex.discovery.base.llm import acall_llm_structured
    from imas_codex.llm.prompt_loader import render_prompt
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnlyBatch,
    )

    item = {
        "id": candidate_name,
        "source_id": "synthetic:test",
        "standard_name": candidate_name,
        "description": candidate_description,
        "documentation": "",
        "unit": candidate_unit,
        "kind": "scalar",
        "physics_domain": "magnetics",
        "tags": ["magnetics"],
        "links": [],
        "validation_issues": [],
    }

    grammar_enums: dict = {}

    context = {
        "items": [item],
        "existing_names": [],
        "review_scored_examples": [],
        "batch_context": "verification run — single synthetic candidate",
        "nearby_existing_names": [],
        "audit_findings": [],
        "prior_reviews": [],
        **grammar_enums,
    }

    system_prompt = render_prompt(
        "sn/review_names",
        {
            **context,
            "items": [],
        },
    )
    user_prompt = render_prompt("sn/review_names", context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.monotonic()
    result = await acall_llm_structured(
        model=model,
        messages=messages,
        response_model=StandardNameQualityReviewNameOnlyBatch,
        service="verify-reviewer-alignment",
    )
    elapsed = time.monotonic() - t0

    parsed, cost, tokens = result
    review = parsed.reviews[0] if parsed.reviews else None

    return {
        "model": model.split("/")[-1],  # short name for display
        "model_full": model,
        "score": review.scores.score if review else None,
        "comment": review.reasoning[:80] if review else "(no review)",
        "tokens": tokens,
        "cost_usd": cost,
        "elapsed_s": round(elapsed, 1),
        "error": None,
    }


async def main() -> None:
    from imas_codex.settings import get_sn_review_names_models

    models = get_sn_review_names_models()
    if not models:
        print("ERROR: no reviewer models configured in pyproject.toml", file=sys.stderr)
        sys.exit(1)

    # Synthetic candidate
    candidate_name = "plasma_current"
    candidate_description = (
        "Total plasma current flowing in the toroidal direction, "
        "positive in the direction of increasing toroidal angle phi."
    )
    candidate_unit = "A"

    print(f"\nVerifying reviewer alignment on '{candidate_name}'")
    print(f"Models: {models}\n")

    results = []
    total_cost = 0.0

    for model in models:
        if total_cost >= _BUDGET_CEILING_USD:
            print(
                f"BUDGET CEILING ${_BUDGET_CEILING_USD:.2f} reached — skipping {model}"
            )
            results.append(
                {
                    "model": model.split("/")[-1],
                    "model_full": model,
                    "score": None,
                    "comment": "skipped — budget ceiling",
                    "tokens": 0,
                    "cost_usd": 0.0,
                    "elapsed_s": 0.0,
                    "error": "budget_ceiling",
                }
            )
            continue

        print(f"  Calling {model}...", flush=True)
        try:
            row = await _call_single_reviewer(
                model=model,
                candidate_name=candidate_name,
                candidate_description=candidate_description,
                candidate_unit=candidate_unit,
            )
        except Exception as exc:
            row = {
                "model": model.split("/")[-1],
                "model_full": model,
                "score": None,
                "comment": str(exc)[:80],
                "tokens": 0,
                "cost_usd": 0.0,
                "elapsed_s": 0.0,
                "error": type(exc).__name__,
            }

        total_cost += row["cost_usd"]
        results.append(row)
        status = "✓" if row["error"] is None else f"✗ {row['error']}"
        print(
            f"    {status}  score={row['score']:.3f}  cost=${row['cost_usd']:.4f}"
            if row["score"] is not None
            else f"    {status}"
        )

    # -----------------------------------------------------------------------
    # Print markdown table
    # -----------------------------------------------------------------------
    print("\n## Reviewer Alignment Matrix\n")
    print(f"Candidate: `{candidate_name}`\n")
    print("| model | score | comment_len | tokens | cost ($) | elapsed (s) | status |")
    print("|-------|-------|-------------|--------|----------|-------------|--------|")

    for row in results:
        score_str = f"{row['score']:.3f}" if row["score"] is not None else "—"
        comment_len = len(row["comment"]) if row["comment"] else 0
        status = "✓" if row["error"] is None else f"✗ {row['error']}"
        print(
            f"| {row['model']} | {score_str} | {comment_len} | "
            f"{row['tokens']} | {row['cost_usd']:.4f} | {row['elapsed_s']}s | {status} |"
        )

    print(f"\n**Total cost:** ${total_cost:.4f}")

    # Assertions
    errors = [r for r in results if r["error"] is not None]
    if errors:
        print(f"\nFAILED: {len(errors)} model(s) returned errors:", file=sys.stderr)
        for e in errors:
            print(f"  {e['model_full']}: {e['error']}", file=sys.stderr)
        sys.exit(1)

    for row in results:
        score = row["score"]
        assert score is not None and 0.0 <= score <= 1.0, (
            f"Model {row['model_full']} returned invalid score: {score!r}"
        )
        assert row["comment"], f"Model {row['model_full']} returned empty comment"

    print("\n✓ All assertions passed.")


if __name__ == "__main__":
    asyncio.run(main())
