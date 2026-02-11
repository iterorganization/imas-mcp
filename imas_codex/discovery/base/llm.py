"""
Common LLM infrastructure for discovery operations.

Provides a uniform interface for calling LLMs across discovery domains
(paths, wiki, signals). Handles:
- LiteLLM noise suppression (stdout/stderr print messages)
- API key management
- Retry with exponential backoff (wraps both LLM call + response parsing)
- Cost extraction and accumulation across retries
- JSON sanitization and Pydantic structured output parsing
- Model-aware token limits

Both sync and async entry points share identical retry/parse/cost logic.
The retry loop wraps both the API call and the response parsing so that
truncated JSON or Pydantic validation errors from malformed responses
trigger a fresh LLM attempt rather than an immediate failure.

Usage:
    from imas_codex.discovery.base.llm import (
        call_llm_structured,
        acall_llm_structured,
        call_llm,
        acall_llm,
    )

    # Structured output with retry+parse (preferred for scoring)
    batch, cost, tokens = call_llm_structured(
        model="google/gemini-3-flash-preview",
        messages=[...],
        response_model=ScoreBatch,
    )

    # Async structured output
    batch, cost, tokens = await acall_llm_structured(
        model="google/gemini-3-flash-preview",
        messages=[...],
        response_model=WikiScoreBatch,
    )

    # Raw response (when caller needs custom parsing)
    response, cost = call_llm(
        model="google/gemini-3-flash-preview",
        messages=[...],
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry

# Error patterns that warrant retry (all lowercase for matching).
# Proven in wiki pipeline — shared across all discovery domains.
RETRYABLE_PATTERNS = frozenset(
    {
        "overloaded",
        "rate",
        "429",
        "503",
        "timeout",
        "eof",  # EOF while parsing JSON (truncated response)
        "json",  # JSON parsing errors
        "truncated",
        "validation",  # Pydantic validation errors from malformed responses
    }
)

# ---------------------------------------------------------------------------
# Model-aware token limits
# ---------------------------------------------------------------------------
# Gemini 3 Flash: 1M context, 65k output, $0.10/$0.40 per 1M tokens
# Claude Sonnet: 200k context, ~8k output default
# Claude Haiku: 200k context, ~4k output default
#
# These limits are intentionally generous for Gemini Flash since it's
# the primary scoring model and large batches (50+ items) need room.
MODEL_TOKEN_LIMITS: dict[str, dict[str, int]] = {
    "gemini": {
        "max_tokens": 65000,  # Gemini 3 Flash supports up to 65k output
        "timeout": 120,  # 2 min — large batches take time
    },
    "claude": {
        "max_tokens": 32000,
        "timeout": 120,
    },
    "default": {
        "max_tokens": 32000,
        "timeout": 120,
    },
}


def get_model_limits(model: str) -> dict[str, int]:
    """Get token limits for a model based on its family.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview")

    Returns:
        Dict with max_tokens and timeout values.
    """
    model_lower = model.lower()
    for family, limits in MODEL_TOKEN_LIMITS.items():
        if family in model_lower:
            return limits
    return MODEL_TOKEN_LIMITS["default"]


# ---------------------------------------------------------------------------
# LiteLLM noise suppression
# ---------------------------------------------------------------------------


def suppress_litellm_noise() -> None:
    """Suppress all LiteLLM diagnostic output.

    LiteLLM prints "Give Feedback", "Provider List", and debug info
    directly to stdout/stderr via print() calls, bypassing Python's
    logging system. This function suppresses both:
    1. Logger-based output (via logging levels)
    2. Print-based output (via litellm.suppress_debug_info)

    Call this once at module load time in any module that uses LiteLLM.
    """
    import litellm

    # Suppress print-based diagnostic messages
    litellm.suppress_debug_info = True

    # Suppress all litellm logging to ERROR level
    for logger_name in ("LiteLLM", "LiteLLM Proxy", "LiteLLM Router", "httpx"):
        level = logging.WARNING if logger_name == "httpx" else logging.ERROR
        logging.getLogger(logger_name).setLevel(level)

    # Environment variables for litellm internals
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


# ---------------------------------------------------------------------------
# API key / model helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Get OpenRouter API key from environment.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Set it in .env or export it."
        )
    return api_key


def ensure_openrouter_prefix(model: str) -> str:
    """Ensure model ID has openrouter/ prefix for LiteLLM routing."""
    if not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


# ---------------------------------------------------------------------------
# Retry / cost helpers
# ---------------------------------------------------------------------------


def _is_retryable(error_msg: str) -> bool:
    """Check if an error message indicates a retryable condition."""
    msg_lower = error_msg.lower()
    return any(pattern in msg_lower for pattern in RETRYABLE_PATTERNS)


def extract_cost(response: Any) -> float:
    """Extract actual LLM cost from a LiteLLM response.

    Priority:
    1. OpenRouter response_cost from _hidden_params (most accurate)
    2. Fallback: Claude Sonnet rates ($3/$15 per 1M tokens)

    Args:
        response: LiteLLM completion response object

    Returns:
        Cost in USD.
    """
    if (
        hasattr(response, "_hidden_params")
        and "response_cost" in response._hidden_params
    ):
        return response._hidden_params["response_cost"]

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return (input_tokens * 3 + output_tokens * 15) / 1_000_000


def _sanitize_content(content: str) -> str:
    """Sanitize LLM response content for JSON parsing.

    Removes control characters and fixes surrogate encoding issues
    that LLMs sometimes produce from file paths or Unicode content.

    Args:
        content: Raw LLM response content string.

    Returns:
        Cleaned string safe for JSON/Pydantic parsing.
    """
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)
    content = content.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="replace"
    )
    return content


def _build_kwargs(
    model: str,
    api_key: str,
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout: int | None,
) -> dict[str, Any]:
    """Build litellm completion kwargs with model-aware defaults.

    When max_tokens or timeout are not explicitly set, uses
    model-family defaults from MODEL_TOKEN_LIMITS.
    """
    model_id = ensure_openrouter_prefix(model)
    limits = get_model_limits(model)

    kwargs: dict[str, Any] = {
        "model": model_id,
        "api_key": api_key,
        "max_tokens": max_tokens if max_tokens is not None else limits["max_tokens"],
        "timeout": timeout if timeout is not None else limits["timeout"],
        "messages": messages,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    return kwargs


# ---------------------------------------------------------------------------
# Structured output: call + parse in shared retry loop
# ---------------------------------------------------------------------------


def call_llm_structured(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> tuple[BaseModel, float, int]:
    """Call LLM and parse structured output, retrying on both API and parse errors.

    Wraps the LLM call and Pydantic parsing in a single retry loop so that
    truncated JSON or validation errors trigger a fresh attempt. Cost is
    accumulated across retries since API calls are billed regardless.

    This pattern was proven in the wiki scoring pipeline and is shared
    across all discovery domains.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_model: Pydantic model for structured output parsing.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (parsed_model, total_cost_usd, total_tokens).

    Raises:
        ValueError: If response parsing fails after all retries.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_model,
        max_tokens,
        temperature,
        timeout,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            total_cost += extract_cost(response)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = response_model.model_validate_json(content)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            return parsed, total_cost, total_tokens

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                time.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]


async def acall_llm_structured(
    model: str,
    messages: list[dict[str, str]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> tuple[BaseModel, float, int]:
    """Async version of call_llm_structured.

    Identical retry+parse semantics using litellm.acompletion() and
    asyncio.sleep() for non-blocking backoff.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_model: Pydantic model for structured output parsing.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (parsed_model, total_cost_usd, total_tokens).

    Raises:
        ValueError: If response parsing fails after all retries.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_model,
        max_tokens,
        temperature,
        timeout,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            total_cost += extract_cost(response)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = response_model.model_validate_json(content)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            return parsed, total_cost, total_tokens

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM error (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Raw LLM calls (when caller needs custom response handling)
# ---------------------------------------------------------------------------


def call_llm(
    model: str,
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> tuple[Any, float]:
    """Call LLM synchronously with retry logic and cost tracking.

    Returns the raw LiteLLM response for callers that need custom
    parsing. Prefer call_llm_structured() for Pydantic models.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_format: Optional Pydantic model for structured output.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (litellm_response, cost_usd).

    Raises:
        ValueError: If API key is not set.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_format,
        max_tokens,
        temperature,
        timeout,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            cost = extract_cost(response)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM rate limited (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                time.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise
            else:
                raise

    raise last_error  # type: ignore[misc]


async def acall_llm(
    model: str,
    messages: list[dict[str, str]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> tuple[Any, float]:
    """Call LLM asynchronously with retry logic and cost tracking.

    Returns the raw LiteLLM response for callers that need custom
    parsing. Prefer acall_llm_structured() for Pydantic models.

    Args:
        model: Model identifier (e.g., "google/gemini-3-flash-preview").
        messages: Chat messages [{"role": ..., "content": ...}].
        response_format: Optional Pydantic model for structured output.
        max_tokens: Max output tokens (None = model-family default).
        temperature: Sampling temperature (None = model default).
        timeout: Request timeout seconds (None = model-family default).
        max_retries: Maximum retry attempts.
        retry_base_delay: Base delay for exponential backoff (seconds).

    Returns:
        Tuple of (litellm_response, cost_usd).

    Raises:
        ValueError: If API key is not set.
        Exception: Non-retryable errors from LLM provider.
    """
    import litellm

    suppress_litellm_noise()

    api_key = get_api_key()
    kwargs = _build_kwargs(
        model,
        api_key,
        messages,
        response_format,
        max_tokens,
        temperature,
        timeout,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            cost = extract_cost(response)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_retryable(error_msg) and attempt < max_retries - 1:
                delay = retry_base_delay * (2**attempt)
                logger.debug(
                    "LLM rate limited (attempt %d/%d): %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    error_msg[:100],
                    delay,
                )
                await asyncio.sleep(delay)
            elif attempt == max_retries - 1:
                logger.error(
                    "LLM failed after %d attempts: %s",
                    max_retries,
                    error_msg[:200],
                )
                raise ValueError(
                    f"LLM failed after {max_retries} attempts: {error_msg[:200]}"
                ) from e
            else:
                raise

    raise last_error  # type: ignore[misc]
