"""
Composable LLM session management for unified access across CLI tools.

Provides:
- LLMSession: Unified interface for all LLM-using commands
- CostTracker: Shared cost tracking with budget enforcement
- Environment variable overrides for production configuration

Usage:
    session = LLMSession(task="language", cost_limit_usd=10.0)
    llm = session.get_llm()

    # Check budget before expensive operations
    if session.budget_exhausted:
        raise BudgetExhaustedError(session.cost_tracker.summary())
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from imas_codex.agentic.llm import (
    get_llm as _get_llm_factory,
    get_model_id,
)
from imas_codex.settings import get_model

if TYPE_CHECKING:
    from llama_index.llms.openrouter import OpenRouter

# Environment variable names for configuration overrides
ENV_MODEL = "IMAS_CODEX_MODEL"
ENV_COST_LIMIT = "IMAS_CODEX_COST_LIMIT"

# Approximate costs per 1M tokens (USD) - OpenRouter pricing
# Updated 2025-01 from https://openrouter.ai/models
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # (input_cost_per_1M, output_cost_per_1M)
    "google/gemini-3-flash-preview": (0.10, 0.40),
    "google/gemini-3-pro-preview": (1.25, 10.00),
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "anthropic/claude-opus-4.5": (15.00, 75.00),
}

# Default fallback cost estimate
DEFAULT_COST_PER_1M = (1.00, 5.00)


class BudgetExhaustedError(Exception):
    """Raised when cost limit is exceeded."""

    def __init__(self, summary: str):
        self.summary = summary
        super().__init__(f"Budget exhausted: {summary}")


@dataclass
class CostTracker:
    """Track LLM costs across operations with optional budget enforcement.

    Attributes:
        limit_usd: Maximum cost in USD (None = no limit)
        total_cost_usd: Running total of costs incurred
        input_tokens: Total input tokens processed
        output_tokens: Total output tokens generated
        request_count: Number of LLM requests made
    """

    limit_usd: float | None = None
    total_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
    ) -> float:
        """Record a request and return its cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model identifier for pricing lookup

        Returns:
            Cost of this request in USD
        """
        cost = estimate_cost(input_tokens, output_tokens, model)
        self.total_cost_usd += cost
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.request_count += 1
        return cost

    def is_exhausted(self) -> bool:
        """Check if budget limit has been reached."""
        if self.limit_usd is None:
            return False
        return self.total_cost_usd >= self.limit_usd

    def remaining_usd(self) -> float | None:
        """Return remaining budget in USD, or None if unlimited."""
        if self.limit_usd is None:
            return None
        return max(0.0, self.limit_usd - self.total_cost_usd)

    def summary(self) -> str:
        """Return human-readable cost summary."""
        parts = [f"${self.total_cost_usd:.4f} spent"]
        if self.limit_usd is not None:
            parts.append(f"(limit: ${self.limit_usd:.2f})")
        parts.append(f"| {self.request_count} requests")
        parts.append(f"| {self.input_tokens:,} in / {self.output_tokens:,} out tokens")
        return " ".join(parts)


@dataclass
class LLMSession:
    """Composable session for unified LLM access across all CLI tools.

    Provides consistent interface for:
    - Model selection (task-based config, env override, explicit override)
    - Cost tracking with optional budget limits
    - Dry-run cost estimation

    Priority for model selection:
    1. Explicit `model` parameter
    2. Environment variable IMAS_CODEX_MODEL
    3. Task-based configuration from pyproject.toml
    4. Default model

    Priority for cost limit:
    1. Explicit `cost_limit_usd` parameter
    2. Environment variable IMAS_CODEX_COST_LIMIT
    3. None (no limit)

    Attributes:
        task: pyproject.toml section for model selection ('language', 'vision', 'agent', 'compaction')
        model: Explicit model override (None = use config/env)
        cost_limit_usd: Maximum cost in USD (None = no limit)
        temperature: LLM temperature (0.0-1.0)
        dry_run: If True, estimate costs without executing
        cost_tracker: Shared cost tracker instance
    """

    task: str = "agent"
    model: str | None = None
    cost_limit_usd: float | None = None
    temperature: float = 0.3
    dry_run: bool = False
    cost_tracker: CostTracker = field(default_factory=CostTracker)

    def __post_init__(self):
        """Apply environment variable overrides and initialize tracker."""
        # Model override from env
        if self.model is None:
            env_model = os.environ.get(ENV_MODEL)
            if env_model:
                self.model = env_model

        # Cost limit from env (if not explicitly set)
        if self.cost_limit_usd is None:
            env_limit = os.environ.get(ENV_COST_LIMIT)
            if env_limit:
                try:
                    self.cost_limit_usd = float(env_limit)
                except ValueError:
                    pass  # Ignore invalid values

        # Set limit on tracker
        if self.cost_limit_usd is not None:
            self.cost_tracker.limit_usd = self.cost_limit_usd

    @property
    def resolved_model(self) -> str:
        """Get the resolved model ID (applying all override rules)."""
        if self.model:
            return get_model_id(self.model)
        return get_model(self.task)

    def get_llm(self, **kwargs) -> OpenRouter:
        """Get configured LLM instance.

        Args:
            **kwargs: Additional arguments passed to get_llm factory

        Returns:
            Configured OpenRouter LLM instance

        Raises:
            BudgetExhaustedError: If budget is already exhausted
        """
        if self.budget_exhausted:
            raise BudgetExhaustedError(self.cost_tracker.summary())

        return _get_llm_factory(
            model=self.resolved_model,
            temperature=self.temperature,
            **kwargs,
        )

    @property
    def budget_exhausted(self) -> bool:
        """Check if budget limit has been reached."""
        return self.cost_tracker.is_exhausted()

    def check_budget(self) -> None:
        """Raise if budget is exhausted.

        Raises:
            BudgetExhaustedError: If budget limit has been reached
        """
        if self.budget_exhausted:
            raise BudgetExhaustedError(self.cost_tracker.summary())

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """Record token usage and return cost.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model used (defaults to session's resolved model)

        Returns:
            Cost of this request in USD
        """
        return self.cost_tracker.record(
            input_tokens,
            output_tokens,
            model or self.resolved_model,
        )

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str | None = None,
    ) -> float:
        """Estimate cost without recording.

        Args:
            input_tokens: Estimated input tokens
            output_tokens: Estimated output tokens
            model: Model to estimate for (defaults to session's resolved model)

        Returns:
            Estimated cost in USD
        """
        return estimate_cost(
            input_tokens,
            output_tokens,
            model or self.resolved_model,
        )

    def summary(self) -> str:
        """Return human-readable session summary."""
        parts = [
            f"Model: {self.resolved_model}",
            f"Section: {self.task}",
            self.cost_tracker.summary(),
        ]
        if self.dry_run:
            parts.insert(0, "[DRY RUN]")
        return " | ".join(parts)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost for a request.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier for pricing lookup

    Returns:
        Estimated cost in USD
    """
    input_cost, output_cost = MODEL_COSTS.get(model, DEFAULT_COST_PER_1M)
    return (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000


def create_session(
    task: str = "agent",
    model: str | None = None,
    cost_limit_usd: float | None = None,
    temperature: float = 0.3,
    dry_run: bool = False,
) -> LLMSession:
    """Factory function to create an LLM session.

    This is the primary entry point for CLI commands to create sessions.

    Args:
        task: pyproject.toml section for model selection ('language', 'agent', etc.)
        model: Explicit model override
        cost_limit_usd: Maximum cost in USD
        temperature: LLM temperature
        dry_run: If True, estimate costs without executing

    Returns:
        Configured LLMSession instance
    """
    return LLMSession(
        task=task,
        model=model,
        cost_limit_usd=cost_limit_usd,
        temperature=temperature,
        dry_run=dry_run,
    )
