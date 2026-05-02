"""Anthropic prompt-cache fire/no-fire probe for OpenRouter models.

Background
----------
On 2026-05-02 we discovered that ``openrouter/anthropic/claude-opus-4.6``
silently does NOT honour ``cache_control`` blocks: across all upstream
providers (Anthropic native, Google Vertex, Amazon Bedrock) the response
returns ``prompt_tokens_details.cached_tokens=0`` and
``cache_write_tokens=0`` even on the second back-to-back call with an
identical static system block of 3618 tokens (well above the 1024-token
floor). Sonnet 4.6 and Haiku 3.5 cache normally on the same request shape.

This drove cost-per-review up to ~$0.052 (Opus full price) instead of
the targeted ~$0.012 (cache-warm). The reviewer primary model has been
swapped to ``claude-sonnet-4.6`` as a workaround (see commit log).

When upstream OpenRouter Opus caching is fixed, re-run this probe — the
two ``opus-4.6`` calls should report ``cache_creation_tokens > 1000`` on
the first call and ``cache_read_tokens > 1000`` on the second. At that
point the reviewer ``models[0]`` can be reverted to Opus 4.6 in
``pyproject.toml`` under ``[tool.imas-codex.sn.review.names]``.

Run::

    uv run python research/cache_probe.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from imas_codex.discovery.base.llm import acall_llm_structured
from imas_codex.llm.prompt_loader import render_prompt


class _Tiny(BaseModel):
    answer: str


_PROBE_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/anthropic/claude-opus-4.6",
    "openrouter/anthropic/claude-haiku-4.5",
]


def _build_static_system() -> str:
    """Render the reviewer system prompt with empty siblings — fully static."""
    ctx = {
        "items": [{"id": "foo", "description": "a"}],
        "vector_neighbours": [],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
    }
    return render_prompt("sn/review_names_system", ctx)


async def _probe_model(model: str, sys_prompt: str) -> None:
    print(f"\n=== {model} ===")
    print(f"system prompt: {len(sys_prompt)} chars (~{len(sys_prompt) // 4} tokens)")
    for i in range(2):
        out = await acall_llm_structured(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f'Probe call {i}. Reply with JSON {{"answer":"ok{i}"}}.'
                    ),
                },
            ],
            response_model=_Tiny,
            service="cache-probe",
        )
        print(
            f"  call#{i}: input={out.input_tokens} output={out.output_tokens} "
            f"cache_read={out.cache_read_tokens} "
            f"cache_creation={out.cache_creation_tokens} cost=${out.cost:.4f}"
        )


async def _main() -> None:
    sys_prompt = _build_static_system()
    for model in _PROBE_MODELS:
        try:
            await _probe_model(model, sys_prompt)
        except Exception as exc:  # noqa: BLE001 — probe should keep going
            print(f"\n=== {model} ===\n  ERROR: {exc!r}")


if __name__ == "__main__":
    asyncio.run(_main())
