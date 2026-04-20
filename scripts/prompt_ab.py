"""Plan 32 Phase 2 — prompt A/B/C research harness.

Runs the 20-path eval set from
``tests/standard_names/eval_sets/prompt_ab_v1.json`` through three
compose prompt variants and writes a comparison report.

Variants:
- **A (baseline):** production ``sn/compose_dd`` prompt (full context
  front-loaded: cluster siblings, exemplars, review feedback).
- **B (name-only):** production ``sn/compose_dd_name_only`` prompt (lean
  per-batch, no context front-loaded).
- **C (tool-calling):** ``sn/compose_dd_tool_calling`` prompt + tool
  defs from ``imas_codex.standard_names.prompt_tools``; LLM fetches
  context on-demand.

The harness uses :func:`imas_codex.discovery.base.llm.call_llm_structured`
with the model from ``get_model('sn-generate')``. Output is one JSON
file per variant plus a summary diff table.

Run::

    uv run python scripts/prompt_ab.py \\
        --eval-set tests/standard_names/eval_sets/prompt_ab_v1.json \\
        --output plans/research/standard-names/prompt_ab_results/ \\
        --cost-limit 2.0

**Status:** scaffolded. The concrete compose-worker call wiring is left
as a focused follow-up so the empirical sweep can run without
re-engineering. Once populated, each output file will contain::

    [
      {
        "variant": "A" | "B" | "C",
        "path": "...",
        "standard_name": "electron_temperature",
        "rationale": "...",
        "cost_usd": 0.0123,
        "latency_s": 4.2,
        "tool_calls": [{"name": "fetch_cluster_siblings", "args": {...}}]
      },
      ...
    ]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def load_eval_set(path: Path) -> list[dict]:
    """Load the stratified eval set and return the items list."""
    data = json.loads(path.read_text())
    return data["items"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=Path("tests/standard_names/eval_sets/prompt_ab_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plans/research/standard-names/prompt_ab_results"),
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=2.0,
        help="Total cost cap across all three variants (USD).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["A", "B", "C"],
        default=["A", "B", "C"],
        help="Which variants to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without calling the LLM.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    items = load_eval_set(args.eval_set)
    args.output.mkdir(parents=True, exist_ok=True)

    plan = {
        "generated_at": datetime.now(UTC).isoformat(),
        "eval_set": str(args.eval_set),
        "n_items": len(items),
        "variants": args.variants,
        "cost_limit": args.cost_limit,
        "output_dir": str(args.output),
    }
    plan_path = args.output / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    logger.info("Wrote plan to %s", plan_path)

    if args.dry_run:
        logger.info("Dry run — skipping LLM calls.")
        for item in items[:3]:
            logger.info("  would compose: %s", item["path"])
        return 0

    # NOTE: concrete variant runners are intentionally left as focused
    # follow-ups. Each runner needs to:
    #   1. Render the appropriate compose prompt with the item's context.
    #   2. For variant C, pass TOOLS from prompt_tools and handle the
    #      tool-call loop via dispatch_tool_call.
    #   3. Record cost/latency/tool_calls per item.
    #   4. Write <variant>.json to args.output.
    #
    # Runner skeleton for reference:
    #
    #   from imas_codex.discovery.base.llm import call_llm_structured
    #   from imas_codex.llm.prompt_loader import render_prompt
    #   from imas_codex.settings import get_model
    #   from imas_codex.standard_names.prompt_tools import TOOLS, dispatch_tool_call
    #
    #   model = get_model("sn-generate")
    #   for item in items:
    #       user = render_prompt("sn/compose_dd_tool_calling", paths_block=...)
    #       result, cost, _ = call_llm_structured(
    #           model=model,
    #           messages=[{"role": "system", "content": system},
    #                     {"role": "user", "content": user}],
    #           response_model=ComposeResponse,
    #           tools=TOOLS,  # variant C only
    #           tool_dispatcher=dispatch_tool_call,
    #       )

    logger.warning(
        "Variant runners not implemented yet (plan 32 Phase 2 scaffold). "
        "Wire up using the runner skeleton in scripts/prompt_ab.py."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
