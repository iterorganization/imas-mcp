"""Markdown evidence renderer for structured fan-out (plan 39 §6.2).

Renders the list of :class:`FanoutResult` returned by the executor
into a compact markdown block injected into the call-site's existing
user prompt via the ``{{ fanout_evidence }}`` placeholder.

Bounds applied here:

- ``result_hit_cap`` — truncate each :class:`FanoutResult.hits` list.
- ``evidence_token_cap`` — total cap on the rendered string, sized
  using the cheap ``len // 4`` token approximation (good enough for
  budget gating; never undercounts).

When all inputs are empty (no successful runners, or all runners
returned zero hits) the function returns ``""`` so the call-site's
``{{ fanout_evidence }}`` placeholder collapses to an empty line —
the "true no-op" semantics of plan 39 §7.2.
"""

from __future__ import annotations

from .schemas import FanoutResult


def _approx_tokens(text: str) -> int:
    """Cheap len/4 token estimate (always >= true count for ASCII)."""
    return max(1, len(text) // 4) if text else 0


def _format_args(args: dict) -> str:
    """One-line repr of runner args, dropping ``fn_id``."""
    parts = [f"{k}={v!r}" for k, v in args.items() if k != "fn_id"]
    return ", ".join(parts)


def _format_hit(hit) -> str:
    parts = [f"- {hit.label}"]
    if hit.score is not None:
        parts.append(f"(score={hit.score:.2f})")
    return "  ".join(parts)


def format_results(
    results: list[FanoutResult],
    *,
    result_hit_cap: int,
    evidence_token_cap: int,
) -> str:
    """Render a list of :class:`FanoutResult` to a markdown evidence block.

    Empty inputs (or all-empty hits) return ``""``.

    Args:
        results: Per-runner outcomes from the executor.
        result_hit_cap: Maximum hits rendered per result.
        evidence_token_cap: Soft cap on total tokens in the returned
            block.  Once the running approximate token count exceeds
            this cap the renderer truncates and appends ``"..."``.

    Returns:
        Markdown block, or ``""`` if no successful runner produced any
        hits.
    """
    successful = [r for r in results if r.ok and r.hits]
    if not successful:
        return ""

    n_queries = len(results)
    n_errors = sum(1 for r in results if not r.ok)

    out: list[str] = []
    out.append(f"## Fan-out evidence (queries={n_queries}, errors={n_errors})")
    out.append("")

    running_tokens = _approx_tokens(out[0])
    truncated = False

    for result in successful:
        if truncated:
            break
        header = f"### {result.fn_id}({_format_args(result.args)})"
        out.append(header)
        running_tokens += _approx_tokens(header)

        hits = result.hits[:result_hit_cap]
        for hit in hits:
            line = _format_hit(hit)
            line_tokens = _approx_tokens(line)
            if running_tokens + line_tokens > evidence_token_cap:
                out.append("...")
                truncated = True
                break
            out.append(line)
            running_tokens += line_tokens

        out.append("")

    return "\n".join(out).rstrip() + "\n"


__all__ = ["format_results"]
