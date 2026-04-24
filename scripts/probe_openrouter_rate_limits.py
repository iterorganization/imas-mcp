"""Probe OpenRouter concurrent-request ceilings per model family.

Sends N simultaneous requests for each model and records the highest N that
produces zero 429 responses.  Sweeps N = [32, 48, 64, 96, 128] (extended from
Phase 7 which only swept to 32) and stops each model at the first N that
returns ≥1 429, ≥1 connection-reset, or ≥1 structural parse failure.

Usage::

    uv run python scripts/probe_openrouter_rate_limits.py

Output:
    - Clean table to stdout
    - Appends a dated section to ``docs/ops/openrouter-rate-ceilings.md``

Budget guard: aborts if cumulative spend exceeds BUDGET_HARD_STOP USD.

All LLM calls go through :func:`imas_codex.discovery.base.llm.acall_llm_structured`
per project policy.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODELS: list[str] = [
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/anthropic/claude-opus-4.6",
    "openrouter/openai/gpt-5.4",
    "openrouter/google/gemini-3.1-pro-preview",
]

N_SWEEP: list[int] = [32, 48, 64, 96, 128]

# Cooldown between N-levels (seconds) to let OpenRouter rate-limit bucket refill
COOLDOWN_S: float = 10.0

# Hard budget stop in USD — script aborts if exceeded
BUDGET_HARD_STOP: float = 1.50

SERVICE_TAG: str = "rate-probe"

DOCS_OUTPUT: Path = (
    Path(__file__).parent.parent / "docs" / "ops" / "openrouter-rate-ceilings.md"
)

# ---------------------------------------------------------------------------
# Minimal compose-shaped payload
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an IMAS standard name generator. Given a data path and description, \
emit a compact standard name: lowercase words joined by underscores, \
2-5 words, physical quantity first.

Respond with valid JSON only."""

USER_PROMPT = """\
Path: core_profiles/profiles_1d/electrons/temperature
Description: Electron temperature profile on the 1D radial grid.
IDS: core_profiles  Units: eV

Generate a standard name for this quantity."""


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class ProbeResult(BaseModel):
    standard_name: str


# ---------------------------------------------------------------------------
# Per-call probe wrapper
# ---------------------------------------------------------------------------


@dataclass
class CallOutcome:
    success: bool
    is_429: bool
    cost: float
    latency: float  # seconds
    error: str = ""


async def _probe_single_call(
    model: str,
    messages: list[dict],
    call_idx: int,
) -> CallOutcome:
    """Issue one structured LLM call; classify outcome."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    t0 = time.monotonic()
    try:
        result = await acall_llm_structured(
            model=model,
            messages=messages,
            response_model=ProbeResult,
            max_tokens=50,
            temperature=0.0,
            max_retries=1,  # fail fast — we want to *see* 429s
            retry_base_delay=0.0,
            service=SERVICE_TAG,
        )
        latency = time.monotonic() - t0
        return CallOutcome(
            success=True, is_429=False, cost=result.cost, latency=latency
        )

    except Exception as exc:
        latency = time.monotonic() - t0
        err_str = str(exc).lower()
        is_429 = (
            "429" in err_str
            or "too many requests" in err_str
            or ("rate" in err_str and ("limit" in err_str or "exceed" in err_str))
        )
        return CallOutcome(
            success=False,
            is_429=is_429,
            cost=0.0,
            latency=latency,
            error=str(exc)[:120],
        )


# ---------------------------------------------------------------------------
# N-level sweep for a single model
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    n: int
    successes: int
    rate_429s: int
    other_errors: int
    wall_time: float
    rpm: float
    cost: float
    ttfb_median: float = 0.0  # median latency in seconds across all calls


@dataclass
class ModelResult:
    model: str
    ceiling: int  # last N with zero 429s; 0 if even N=4 429d
    levels: list[LevelResult] = field(default_factory=list)
    total_cost: float = 0.0
    aborted_budget: bool = False


async def probe_model(
    model: str,
    budget_tracker: list[float],  # mutable single-element list for shared state
) -> ModelResult:
    """Sweep N-levels for one model; return ceiling and per-level metrics."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    result = ModelResult(model=model, ceiling=0)
    last_clean_n = 0
    first_level = True

    for n in N_SWEEP:
        # Budget guard
        if budget_tracker[0] >= BUDGET_HARD_STOP:
            print(
                f"  [BUDGET STOP] cumulative spend ${budget_tracker[0]:.3f} ≥ ${BUDGET_HARD_STOP}",
                flush=True,
            )
            result.aborted_budget = True
            break

        # Inter-level cooldown so OpenRouter token bucket partially refills
        if not first_level:
            print(f"  [cooldown {COOLDOWN_S:.0f}s] … ", end="", flush=True)
            await asyncio.sleep(COOLDOWN_S)
            print("done", flush=True)
        first_level = False

        print(f"  N={n:2d} concurrent … ", end="", flush=True)

        t_start = time.monotonic()
        tasks = [_probe_single_call(model, messages, i) for i in range(n)]
        outcomes: list[CallOutcome] = await asyncio.gather(*tasks)
        wall = time.monotonic() - t_start

        successes = sum(1 for o in outcomes if o.success)
        rate_429s = sum(1 for o in outcomes if o.is_429)
        other_errors = sum(1 for o in outcomes if not o.success and not o.is_429)
        level_cost = sum(o.cost for o in outcomes)
        rpm = (n / wall) * 60 if wall > 0 else 0.0

        # TTFB proxy: median latency across all calls (non-streaming; latency = TTFB)
        all_latencies = [o.latency for o in outcomes]
        ttfb_med = statistics.median(all_latencies) if all_latencies else 0.0

        budget_tracker[0] += level_cost
        result.total_cost += level_cost

        lvl = LevelResult(
            n=n,
            successes=successes,
            rate_429s=rate_429s,
            other_errors=other_errors,
            wall_time=wall,
            rpm=rpm,
            cost=level_cost,
            ttfb_median=ttfb_med,
        )
        result.levels.append(lvl)

        # Check for connection-reset errors (stop condition)
        conn_resets = sum(
            1
            for o in outcomes
            if not o.success
            and not o.is_429
            and any(
                k in o.error.lower()
                for k in ("connection reset", "connectionreset", "econnreset")
            )
        )

        status = (
            f"ok={successes}/{n}  429s={rate_429s}  errs={other_errors}"
            f"  ttfb_med={ttfb_med:.1f}s  wall={wall:.1f}s  rpm={rpm:.0f}  cost=${level_cost:.4f}"
        )
        print(status, flush=True)

        if rate_429s >= 1:
            # Hit rate limit — stop sweep for this model
            print(f"  → 429 at N={n}; ceiling = {last_clean_n}", flush=True)
            result.ceiling = last_clean_n
            break

        if conn_resets >= 1:
            # Connection reset — treat as rate limit proxy
            print(
                f"  → connection-reset at N={n}; ceiling = {last_clean_n}", flush=True
            )
            result.ceiling = last_clean_n
            break

        if other_errors == n:
            # Total failure at this N — structural parse failure or incompatibility
            print(
                f"  → 100% errors at N={n}; treating as incompatible, stopping",
                flush=True,
            )
            result.ceiling = last_clean_n
            break

        last_clean_n = n
    else:
        # Completed all N levels without a 429
        result.ceiling = N_SWEEP[-1]
        print(f"  → No 429 up to N={N_SWEEP[-1]}; ceiling ≥ {N_SWEEP[-1]}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_TABLE_COLS = ("Model", "Ceiling", "Cost ($)", "Notes")


def _short_model(model: str) -> str:
    """Strip openrouter/ prefix for compact display."""
    return model.removeprefix("openrouter/")


def _build_table_rows(results: list[ModelResult]) -> list[tuple[str, str, str, str]]:
    rows = []
    for r in results:
        ceiling_str = (
            f"≥{r.ceiling}"
            if r.ceiling == N_SWEEP[-1] and not r.aborted_budget
            else str(r.ceiling)
        )
        if r.aborted_budget:
            notes = "budget abort"
        elif r.ceiling == N_SWEEP[-1]:
            notes = "no 429 in sweep"
        else:
            fail_n = _next_n(r)
            if fail_n == -1:
                # Stopped on non-429 errors (parse failures / compat)
                first_fail = next(
                    (
                        lvl
                        for lvl in r.levels
                        if lvl.rate_429s == 0 and lvl.other_errors == lvl.n
                    ),
                    None,
                )
                notes = (
                    f"100% parse/compat errors at N={first_fail.n}"
                    if first_fail
                    else "incompatible (no JSON)"
                )
            else:
                notes = f"429 at N={fail_n}"
        rows.append(
            (
                _short_model(r.model),
                ceiling_str,
                f"{r.total_cost:.4f}",
                notes,
            )
        )
    return rows


def _next_n(r: ModelResult) -> int:
    """Return the N that triggered the 429."""
    for lvl in r.levels:
        if lvl.rate_429s >= 1:
            return lvl.n
    return -1


def _print_table(results: list[ModelResult], total_cost: float) -> None:
    rows = _build_table_rows(results)
    col_widths = [
        max(len(_TABLE_COLS[i]), max(len(r[i]) for r in rows))
        for i in range(len(_TABLE_COLS))
    ]

    def _row(cells: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))
            + " |"
        )

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"

    print()
    print("=" * 60)
    print("  OpenRouter Rate-Limit Probe Results")
    print("=" * 60)
    print(sep)
    print(_row(_TABLE_COLS))
    print(sep)
    for r in rows:
        print(_row(r))
    print(sep)
    print(f"  Total spend: ${total_cost:.4f}")
    print()


def _build_markdown(results: list[ModelResult], total_cost: float, ts: str) -> str:
    rows = _build_table_rows(results)
    col_widths = [
        max(len(_TABLE_COLS[i]), max(len(r[i]) for r in rows))
        for i in range(len(_TABLE_COLS))
    ]

    def _md_row(cells: tuple[str, ...]) -> str:
        return (
            "| "
            + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths, strict=True))
            + " |"
        )

    sep_row = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [
        f"## {ts}",
        "",
        f"N sweep: {N_SWEEP}",
        f"Total spend: ${total_cost:.4f}",
        "",
        _md_row(_TABLE_COLS),
        sep_row,
    ]
    for r in rows:
        lines.append(_md_row(r))

    lines += [
        "",
        "### Per-model level detail",
        "",
    ]
    for r in results:
        lines.append(f"#### {_short_model(r.model)}")
        lines.append("")
        lines.append("| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for lvl in r.levels:
            lines.append(
                f"| {lvl.n} | {lvl.successes} | {lvl.rate_429s} | {lvl.other_errors}"
                f" | {lvl.ttfb_median:.1f} | {lvl.wall_time:.1f} | {lvl.rpm:.0f} | {lvl.cost:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def _write_markdown(md: str) -> None:
    DOCS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = "# OpenRouter Rate-Limit Ceilings\n\n"
    if DOCS_OUTPUT.exists():
        existing = DOCS_OUTPUT.read_text()
        # Remove duplicate header if file already has one
        body = existing.removeprefix(header)
        DOCS_OUTPUT.write_text(header + md + "\n\n---\n\n" + body)
    else:
        DOCS_OUTPUT.write_text(header + md + "\n")
    print(f"Results written to {DOCS_OUTPUT}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\nOpenRouter Rate-Limit Probe — {ts}")
    print(
        f"Models: {len(MODELS)}  N-sweep: {N_SWEEP}  Budget cap: ${BUDGET_HARD_STOP:.2f}\n"
    )

    budget_tracker: list[float] = [0.0]
    results: list[ModelResult] = []

    for model in MODELS:
        short = _short_model(model)
        print(f"── {short} ──", flush=True)
        r = await probe_model(model, budget_tracker)
        results.append(r)
        print(
            f"  subtotal=${r.total_cost:.4f}  running=${budget_tracker[0]:.4f}\n",
            flush=True,
        )

        if budget_tracker[0] >= BUDGET_HARD_STOP:
            print(
                f"HARD BUDGET STOP: ${budget_tracker[0]:.3f} ≥ ${BUDGET_HARD_STOP:.2f}. "
                "Remaining models skipped.",
                flush=True,
            )
            break

    total_cost = budget_tracker[0]
    _print_table(results, total_cost)

    md = _build_markdown(results, total_cost, ts)
    _write_markdown(md)

    # Summary for caller: ceilings dict
    ceilings = {_short_model(r.model): r.ceiling for r in results}
    print("Ceilings (last clean N):")
    for model, ceiling in ceilings.items():
        print(f"  {model}: {ceiling}")

    # Exit non-zero if every model got 429d at N=4
    if all(c == 0 for c in ceilings.values()):
        print(
            "\nERROR: All models 429d at N=4 — check API key / account limits.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
