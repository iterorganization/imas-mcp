"""Test prompt caching through LiteLLM proxy → OpenRouter → Anthropic.

Validates that the proxy chain preserves cache_control directives and that
cached_tokens metrics appear in the response — critical for cost efficiency
with agent teams ($1.50/M cached vs $15/M uncached for Opus input tokens).

Usage:
    # Via proxy (preferred):
    LITELLM_PROXY_URL=http://localhost:4000 uv run python scripts/test_prompt_caching.py

    # Direct OpenRouter (for comparison):
    uv run python scripts/test_prompt_caching.py --direct

    # Pin to Anthropic provider (if cache hit rates are low):
    uv run python scripts/test_prompt_caching.py --pin-provider

Requirements:
    uv sync --extra serve  (or: pip install litellm)
"""

from __future__ import annotations

import os
import sys
import time

import click

# Generate a large system prompt (~1500 tokens, above Anthropic's 1024-token min)
LARGE_SYSTEM = (
    "You are an expert in IMAS (Integrated Modelling & Analysis Suite) "
    "for fusion research. You understand data dictionaries, IDS structures, "
    "COCOS conventions, MDSplus trees, and facility-specific data systems "
    "across TCV, JET, ITER, and JT60SA. "
) * 50  # ~1500 tokens


def test_caching(
    model: str,
    via_proxy: bool = True,
    pin_provider: bool = False,
) -> bool:
    """Run two identical requests and check for cache hits on the second.

    Returns True if prompt caching is working (cached_tokens > 0 on call 2).
    """
    import litellm

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": LARGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "What is an IDS in IMAS? Answer in one sentence."},
    ]

    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": 50,
    }

    if via_proxy:
        proxy_url = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
        master_key = os.environ.get("LITELLM_MASTER_KEY", "sk-litellm-imas-codex")
        kwargs["api_base"] = proxy_url
        kwargs["api_key"] = master_key

    if pin_provider:
        kwargs["extra_body"] = {"provider": {"order": ["Anthropic"]}}

    # --- Call 1: cache write ---
    click.echo("Call 1 (cache write)...")
    r1 = litellm.completion(**kwargs)
    click.echo(f"  Usage: {r1.usage}")
    cached_write = getattr(
        getattr(r1.usage, "prompt_tokens_details", None), "cached_tokens", 0
    )
    click.echo(f"  Cached tokens: {cached_write}")

    # Wait for cache propagation (Anthropic needs ~1-2s)
    click.echo("Waiting 3s for cache propagation...")
    time.sleep(3)

    # --- Call 2: should be cache hit ---
    click.echo("Call 2 (cache read)...")
    r2 = litellm.completion(**kwargs)
    click.echo(f"  Usage: {r2.usage}")
    cached_read = getattr(
        getattr(r2.usage, "prompt_tokens_details", None), "cached_tokens", 0
    )
    click.echo(f"  Cached tokens: {cached_read}")

    click.echo()
    if cached_read > 0:
        savings_pct = (1 - 1.5 / 15) * 100  # Opus input pricing
        click.echo(f"✅ Prompt caching WORKING ({cached_read} tokens cached)")
        click.echo(f"   Estimated savings: {savings_pct:.0f}% on cached input tokens")
        return True
    elif cached_write > 0:
        click.echo("⚠️  Cache write detected on call 1 but not read on call 2")
        click.echo("   OpenRouter may have routed to a different provider instance")
        click.echo("   Try: --pin-provider to force Anthropic direct")
        return False
    else:
        click.echo("❌ Prompt caching NOT detected")
        click.echo("   Check:")
        click.echo("   1. Langfuse dashboard for cached_tokens in usage")
        click.echo("   2. OpenRouter activity page for cache_discount field")
        click.echo("   3. Try --pin-provider to force Anthropic provider")
        return False


@click.command()
@click.option("--direct", is_flag=True, help="Bypass proxy, call OpenRouter directly")
@click.option(
    "--pin-provider",
    is_flag=True,
    help="Pin to Anthropic provider (avoid OpenRouter routing variance)",
)
@click.option(
    "--model",
    default=None,
    help="Model name (default: 'sonnet' via proxy, 'openrouter/anthropic/claude-sonnet-4-20250514' direct)",
)
def main(direct: bool, pin_provider: bool, model: str | None) -> None:
    """Test prompt caching through the LiteLLM proxy chain."""
    if model is None:
        model = "openrouter/anthropic/claude-sonnet-4-20250514" if direct else "sonnet"

    click.echo(f"Model: {model}")
    click.echo(f"Via proxy: {not direct}")
    click.echo(f"Pin provider: {pin_provider}")
    click.echo()

    success = test_caching(model, via_proxy=not direct, pin_provider=pin_provider)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
