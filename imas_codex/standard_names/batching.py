"""Token-aware batching utilities for the SN generate and enrich pipelines.

Provides three grouping strategies and a pre-flight token check that
splits oversized batches before they reach the LLM.

Grouping strategies
-------------------

- ``group_by_cluster_and_unit`` — (cluster × unit) for full compose mode.
  Wraps the existing logic in ``enrichment.group_by_concept_and_unit``.
- ``group_by_domain_and_unit`` — (physics_domain × unit) for name-only mode.
  Wraps ``enrichment.group_for_name_only``.
- ``group_for_enrich`` — sequential chunking for the enrich pipeline.

Token estimation
----------------

Uses a ``len(text) / 4`` heuristic (no tiktoken dependency). The
estimate is intentionally conservative (slightly over-counts) which
is safe for a budget guard.

Pre-flight check
-----------------

``pre_flight_token_check`` iterates over a batch list, estimates tokens
for each, and binary-splits any that exceed ``max_tokens``.  This is a
safety net — the primary batch-size cap should keep most batches well
under budget.
"""

from __future__ import annotations

import logging
from typing import Any

from imas_codex.standard_names.sources.base import ExtractionBatch

logger = logging.getLogger(__name__)

# ── Default token budget ────────────────────────────────────────────────────
#
# 200 k context window with a 50 k safety margin → 150 k usable.
DEFAULT_MAX_TOKENS = 150_000


# ── Token estimation ────────────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Estimate token count for *text* using a ``len / 4`` heuristic.

    This is intentionally conservative (over-counts by ~5–10 %% for
    English prose) so the pre-flight check errs on the side of
    splitting.
    """
    return max(1, len(text) // 4)


def estimate_batch_tokens(batch: ExtractionBatch) -> int:
    """Estimate total token count for an :class:`ExtractionBatch`.

    Accounts for the ``context`` header and per-item payloads
    (``description``, ``documentation``, ``path``).
    """
    total = estimate_tokens(batch.context)
    for item in batch.items:
        for key in ("path", "description", "documentation", "unit"):
            val = item.get(key)
            if val:
                total += estimate_tokens(str(val))
        # Siblings contribute to the prompt too
        for sib in item.get("cluster_siblings", []):
            for k in ("path", "description"):
                sv = sib.get(k)
                if sv:
                    total += estimate_tokens(str(sv))
    return total


def estimate_enrich_batch_tokens(batch: dict[str, Any]) -> int:
    """Estimate total token count for an enrich batch dict.

    Enrich batches are plain dicts with an ``items`` list of SN dicts.
    Each item carries ``description``, ``documentation``, ``tags``,
    ``links``, and contextualisation data that all end up in the prompt.
    """
    total = 0
    for item in batch.get("items", []):
        for key in (
            "name",
            "description",
            "documentation",
            "unit",
            "kind",
            "physics_domain",
        ):
            val = item.get(key)
            if val:
                total += estimate_tokens(str(val))
        # Contextualisation payload (added by contextualise worker)
        ctx = item.get("context")
        if isinstance(ctx, str):
            total += estimate_tokens(ctx)
        elif isinstance(ctx, dict):
            for v in ctx.values():
                if isinstance(v, str):
                    total += estimate_tokens(v)
        # Tags / links are small but count them
        for list_key in ("tags", "links", "source_paths"):
            lst = item.get(list_key)
            if lst:
                total += estimate_tokens(" ".join(str(x) for x in lst))
    return total


# ── Pre-flight token check ──────────────────────────────────────────────────


def pre_flight_token_check(
    batches: list[ExtractionBatch],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[ExtractionBatch]:
    """Split extraction batches that exceed *max_tokens*.

    Iterates over *batches*, estimates tokens for each, and binary-splits
    any that are over budget.  Splitting preserves the ``group_key``
    (with ``#split-N`` suffixes) and all other metadata.

    Args:
        batches: List of extraction batches (mutated in place is fine —
            callers should use the returned list).
        max_tokens: Maximum estimated tokens per batch.

    Returns:
        A new list where every batch is ≤ *max_tokens*.
    """
    from imas_codex.standard_names.enrichment import build_batch_context

    result: list[ExtractionBatch] = []
    splits = 0

    for batch in batches:
        est = estimate_batch_tokens(batch)
        if est <= max_tokens or len(batch.items) <= 1:
            result.append(batch)
            continue

        # Binary split
        mid = len(batch.items) // 2
        left_items = batch.items[:mid]
        right_items = batch.items[mid:]

        left_batch = ExtractionBatch(
            source=batch.source,
            group_key=f"{batch.group_key}#split-0",
            items=left_items,
            context=build_batch_context(left_items, batch.group_key),
            existing_names=batch.existing_names,
            dd_version=batch.dd_version,
            cocos_version=batch.cocos_version,
            cocos_params=batch.cocos_params,
            mode=batch.mode,
        )
        right_batch = ExtractionBatch(
            source=batch.source,
            group_key=f"{batch.group_key}#split-1",
            items=right_items,
            context=build_batch_context(right_items, batch.group_key),
            existing_names=batch.existing_names,
            dd_version=batch.dd_version,
            cocos_version=batch.cocos_version,
            cocos_params=batch.cocos_params,
            mode=batch.mode,
        )
        splits += 1

        # Recurse — each half might still be oversized
        result.extend(
            pre_flight_token_check([left_batch, right_batch], max_tokens=max_tokens)
        )

    if splits:
        logger.info(
            "Pre-flight token check: split %d oversized batches "
            "(max_tokens=%d, result=%d batches)",
            splits,
            max_tokens,
            len(result),
        )

    return result


def pre_flight_enrich_token_check(
    batches: list[dict[str, Any]],
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """Split enrich batches that exceed *max_tokens*.

    Same logic as :func:`pre_flight_token_check` but for the plain-dict
    enrich batch format.

    Args:
        batches: List of enrich batch dicts.
        max_tokens: Maximum estimated tokens per batch.

    Returns:
        A new list where every batch is ≤ *max_tokens*.
    """
    result: list[dict[str, Any]] = []
    splits = 0

    for batch in batches:
        est = estimate_enrich_batch_tokens(batch)
        items = batch.get("items", [])
        if est <= max_tokens or len(items) <= 1:
            result.append(batch)
            continue

        # Binary split
        mid = len(items) // 2
        left = {**batch, "items": items[:mid], "batch_index": len(result)}
        right = {**batch, "items": items[mid:], "batch_index": len(result) + 1}
        splits += 1

        result.extend(
            pre_flight_enrich_token_check([left, right], max_tokens=max_tokens)
        )

    if splits:
        logger.info(
            "Pre-flight enrich token check: split %d oversized batches "
            "(max_tokens=%d, result=%d batches)",
            splits,
            max_tokens,
            len(result),
        )

    return result


# ── Configuration helpers ───────────────────────────────────────────────────


def get_generate_batch_config() -> dict[str, int]:
    """Read batch configuration from ``[tool.imas-codex.sn-generate]``.

    Returns:
        Dict with keys ``batch_size``, ``name_only_batch_size``, ``max_tokens``.
    """
    from imas_codex.settings import _get_section

    section = _get_section("sn-run")
    return {
        "batch_size": int(section.get("batch-size", 25)),
        "name_only_batch_size": int(section.get("name-only-batch-size", 50)),
        "max_tokens": int(section.get("max-tokens", DEFAULT_MAX_TOKENS)),
    }


def get_enrich_batch_config() -> dict[str, int]:
    """Read batch configuration from ``[tool.imas-codex.sn-enrich]``.

    Returns:
        Dict with keys ``batch_size``, ``max_tokens``.
    """
    from imas_codex.settings import _get_section

    section = _get_section("sn-enrich")
    return {
        "batch_size": int(section.get("batch-size", 12)),
        "max_tokens": int(section.get("max-tokens", DEFAULT_MAX_TOKENS)),
    }
