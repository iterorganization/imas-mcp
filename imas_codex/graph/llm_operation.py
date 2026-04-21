"""LLMUsage dataclass and apply_llm_operation helper.

Provides a structured representation of LLM operation metadata and a
helper to stamp those fields onto a node property dict before graph
persistence.  Every field maps 1:1 to the ``LLMOperation`` mixin
defined in ``schemas/common.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class LLMUsage:
    """Structured container for a single LLM call's cost/token metadata.

    Mirrors the eight ``llm_*`` fields of the ``LLMOperation`` schema mixin.
    """

    model: str
    cost: float
    tokens_in: int
    tokens_out: int
    tokens_cached_read: int = 0
    tokens_cached_write: int = 0
    service: str | None = None
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


def apply_llm_operation(
    node_props: dict[str, Any],
    usage: LLMUsage,
) -> dict[str, Any]:
    """Attach ``llm_*`` fields onto a node property dict in-place.

    Returns the dict for chaining.
    """
    node_props.update(
        {
            "llm_model": usage.model,
            "llm_cost": usage.cost,
            "llm_tokens_in": usage.tokens_in,
            "llm_tokens_out": usage.tokens_out,
            "llm_tokens_cached_read": usage.tokens_cached_read,
            "llm_tokens_cached_write": usage.tokens_cached_write,
            "llm_service": usage.service,
            "llm_at": (
                usage.at.isoformat() if isinstance(usage.at, datetime) else usage.at
            ),
        }
    )
    return node_props
