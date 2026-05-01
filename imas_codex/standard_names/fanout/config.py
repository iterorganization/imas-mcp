"""Configuration for structured fan-out (plan 39 §9).

Reads ``[tool.imas-codex.sn.fanout]`` from ``pyproject.toml`` into a
:class:`FanoutSettings` Pydantic model and exposes :data:`CATALOG_VERSION`
— a sha256 over the **fully rendered proposer prompt body** (plan 39
§6.1, I4) — for the literal ``catalog_version=<hex>`` line at the top
of the Stage A system prompt.

Hash semantics
--------------

The hash deliberately covers the rendered prompt body, not the schema
dict, because:

- ``Pydantic.model_json_schema()`` output is not stable across minor
  versions (e.g. ``definitions`` -> ``$defs``); a dependency bump would
  silently invalidate caches on the wrong axis.
- The proposer prompt's help text is not in the schema dict; help-text
  edits would change the actual prompt prefix without flipping a
  schema-only hash.

Sampling parameters (e.g. ``proposer-temperature``) are *not* part of
the hash — caching covers the prompt, not the sampling distribution
(plan 39 §6.1 S6).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from imas_codex.settings import _load_pyproject_settings

# =====================================================================
# Settings model
# =====================================================================


class FanoutSettings(BaseModel):
    """Pydantic-validated settings for structured fan-out.

    All fields default to their plan-§9 values so an empty
    ``[tool.imas-codex.sn.fanout]`` block yields a fully usable
    (default-disabled) config.
    """

    enabled: bool = False
    """Master switch.  Default off until rolled out."""

    max_fan_degree: int = 3
    """Hard cap on Stage A query count (parse-time bound)."""

    function_timeout_s: float = 5.0
    """Per-runner ``asyncio.wait_for`` timeout."""

    total_timeout_s: float = 12.0
    """Whole-gather ``asyncio.wait_for`` timeout."""

    result_hit_cap: int = 8
    """Per-result hit cap applied by the renderer."""

    evidence_token_cap_baseline: int = 2000
    """Total evidence-block token cap on baseline (Sonnet/Haiku) cycles."""

    evidence_token_cap_escalation: int = 800
    """Total evidence-block token cap on Opus-escalation cycles."""

    proposer_model: str = "openrouter/anthropic/claude-haiku-4.5"
    """Stage A LLM model identifier."""

    proposer_temperature: float = 0.1
    """Stage A sampling temperature (low for plan stability)."""

    fanout_cost_estimate_baseline: float = 0.005
    """Baseline parent-lease pad ($) for fan-out cycles (plan 39 §7.3 I1)."""

    fanout_cost_estimate_escalation: float = 0.05
    """Escalation parent-lease pad ($) for fan-out cycles (plan 39 §7.3 I1)."""

    fanout_max_charge_per_cycle_baseline: float = 0.02
    """Hard cap on cumulative fan-out sub-event spend (baseline)."""

    fanout_max_charge_per_cycle_escalation: float = 0.10
    """Hard cap on cumulative fan-out sub-event spend (escalation)."""

    refine_trigger_keywords: tuple[str, ...] = (
        "unclear",
        "ambiguous",
        "duplicate",
        "consider",
        "compare",
        "decomposition",
        "absorbed",
        "compound",
        "awkward",
    )
    """Trigger keywords for refine-site fan-out.

    Matched case-insensitively against the reviewer-comment excerpt
    extracted from :attr:`refine_trigger_comment_dims`.  The default
    list combines disambiguation cues (``unclear``, ``ambiguous``,
    ``duplicate``) with grammar-decomposition cues actually written
    by the rubric judge (``decomposition``, ``absorbed``, ``compound``,
    ``awkward``)."""

    refine_trigger_comment_dims: tuple[str, ...] = (
        "grammar",
        "semantic",
    )
    """Reviewer-comment dims used by the trigger predicate (plan 39 I3).

    Must match the actual ``reviewer_comments_per_dim_name`` keys
    written by ``review_names`` (rubric: grammar / semantic /
    convention / completeness).  Defaults to the two dims most
    likely to surface disambiguation/decomposition signal."""

    refine_trigger_comment_chars: int = 800
    """Total excerpt length cap (plan 39 §5.1 S3)."""

    refine_fanout_arm_percent: int = 50
    """Within-cohort A/B on-arm percentage (plan 39 §8.4 I2).

    ``50`` (default) is a balanced 50/50 split; ``0`` forces every
    trigger-eligible item to the off arm (kill switch); ``100`` forces
    every item to the on arm (full rollout).  Used by
    :func:`imas_codex.standard_names.fanout.trigger.assign_arm`.
    """

    sites: dict[str, bool] = Field(
        default_factory=lambda: {"refine_name": False},
        description="Per-site enable flags (under [tool.imas-codex.sn.fanout.sites]).",
    )

    def cap_for_charge(self, *, escalate: bool) -> float:
        """Return the per-cycle fan-out spend cap for the cycle's tier."""
        return (
            self.fanout_max_charge_per_cycle_escalation
            if escalate
            else self.fanout_max_charge_per_cycle_baseline
        )

    def evidence_token_cap_for(self, *, escalate: bool) -> int:
        """Return the renderer evidence-token cap for the cycle's tier."""
        return (
            self.evidence_token_cap_escalation
            if escalate
            else self.evidence_token_cap_baseline
        )

    def cost_estimate_for(self, *, escalate: bool) -> float:
        """Return the parent-lease pad amount for the cycle's tier."""
        return (
            self.fanout_cost_estimate_escalation
            if escalate
            else self.fanout_cost_estimate_baseline
        )


# =====================================================================
# Settings loader
# =====================================================================


def _kebab_to_snake(key: str) -> str:
    """Convert pyproject TOML kebab-case keys to Pydantic snake_case."""
    return key.replace("-", "_")


def load_settings() -> FanoutSettings:
    """Load :class:`FanoutSettings` from ``[tool.imas-codex.sn.fanout]``.

    Missing or empty section -> all defaults (fan-out disabled).  Unknown
    keys are silently ignored so plan revisions can add new fields
    without breaking older configs.
    """
    raw = _load_pyproject_settings()
    section = raw.get("sn", {}).get("fanout", {})
    if not isinstance(section, dict):
        return FanoutSettings()

    sites = section.get("sites", {}) or {}
    if not isinstance(sites, dict):
        sites = {}

    converted: dict[str, object] = {}
    for k, v in section.items():
        if k == "sites":
            continue
        snake = _kebab_to_snake(k)
        if snake in FanoutSettings.model_fields:
            converted[snake] = v

    converted["sites"] = {str(k): bool(v) for k, v in sites.items()}
    return FanoutSettings(**converted)


# =====================================================================
# CATALOG_VERSION — sha256 over the rendered proposer-prompt body
# =====================================================================


_PROMPT_NAME = "sn/fanout_propose"


def _read_proposer_prompt_body() -> str:
    """Return the proposer prompt body (everything below the version line).

    The body is the post-frontmatter Stage A system prompt **excluding**
    the literal ``catalog_version=<hex>`` line that we prepend at
    runtime.  Hashing the body (not the schema dict) is the I4
    correction in plan 39 §6.1.

    Implementation note: we read directly from
    :func:`imas_codex.llm.prompt_loader.load_prompts` (which strips
    frontmatter) but do *not* call :func:`render_prompt`, so the hash
    is independent of unrelated schema-context provider changes.
    """
    from imas_codex.llm.prompt_loader import load_prompts

    prompts = load_prompts()
    if _PROMPT_NAME not in prompts:
        raise KeyError(
            f"Fan-out proposer prompt {_PROMPT_NAME!r} missing from prompts dir."
        )
    return prompts[_PROMPT_NAME].content


@lru_cache(maxsize=1)
def _compute_catalog_version() -> str:
    """Compute the sha256 of the proposer-prompt body."""
    body = _read_proposer_prompt_body()
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


CATALOG_VERSION: str = _compute_catalog_version()
"""Sha256 of the proposer-prompt body (plan 39 §6.1 I4).

Computed once at module import via :func:`_compute_catalog_version`;
the literal first line of the rendered system prompt is::

    catalog_version=<CATALOG_VERSION>

so any change to the body flips this hash and busts every cached
prompt prefix.
"""


def render_proposer_system_prompt() -> str:
    """Return the Stage A system prompt with the version line prepended.

    Output shape::

        catalog_version=<sha256>
        <body>

    Used by :func:`imas_codex.standard_names.fanout.dispatcher.propose`.
    """
    body = _read_proposer_prompt_body()
    return f"catalog_version={_compute_catalog_version()}\n{body}"


def _reset_catalog_version_cache() -> None:
    """Clear the cached catalog version (test-only).

    The proposer prompt is static at runtime; this hook exists solely
    so ``test_catalog_version_hash_covers_body`` can mutate the prompt
    file and observe the hash flipping.
    """
    _compute_catalog_version.cache_clear()
    # Also clear the prompt-loader's own cache so the test sees the
    # newly-written prompt body on the next read.
    from imas_codex.llm.prompt_loader import load_prompts

    if hasattr(load_prompts, "cache_clear"):
        load_prompts.cache_clear()  # type: ignore[attr-defined]
    global CATALOG_VERSION
    CATALOG_VERSION = _compute_catalog_version()


def _prompt_path(name: str = _PROMPT_NAME) -> Path:
    """Return the on-disk path for a prompt (test helper)."""
    from imas_codex.llm.prompt_loader import PROMPTS_DIR

    return Path(PROMPTS_DIR) / f"{name}.md"


__all__ = [
    "CATALOG_VERSION",
    "FanoutSettings",
    "load_settings",
    "render_proposer_system_prompt",
    "_compute_catalog_version",
    "_reset_catalog_version_cache",
]
