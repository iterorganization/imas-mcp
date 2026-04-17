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

    # Async structured output (also returns LLMResult with cache info)
    llm_out = await acall_llm_structured(
        model="google/gemini-3-flash-preview",
        messages=[...],
        response_model=WikiScoreBatch,
    )
    batch, cost, tokens = llm_out          # backward-compatible
    cache_read = llm_out.cache_read_tokens  # new: cache metrics

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
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import yaml

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

LLM_SERVICE = Literal[
    "facility-discovery",
    "standard-names",
    "data-dictionary",
    "imas-mapping",
    "embedding",
    "untagged",
]

_VALID_SERVICES: set[str] = set(LLM_SERVICE.__args__)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fatal error: provider budget/credit exhaustion
# ---------------------------------------------------------------------------


class ProviderBudgetExhausted(Exception):
    """Raised when the LLM provider rejects requests due to credit/budget limits.

    This is a fatal, non-retryable error. Callers should halt all LLM-dependent
    work immediately rather than retrying — the key/account needs manual
    intervention (top-up credits or raise the spending limit).
    """


class LLMResult:
    """Return type for call_llm_structured / acall_llm_structured.

    Backward-compatible with 3-tuple unpacking::

        result, cost, tokens = call_llm_structured(...)  # still works

    Also carries prompt-cache token counts for callers that need them::

        llm_out = await acall_llm_structured(...)
        result, cost, tokens = llm_out
        cache_read = llm_out.cache_read_tokens
        cache_creation = llm_out.cache_creation_tokens

    Attributes:
        parsed: The Pydantic model instance returned by the LLM.
        cost: Total cost in USD (accumulated across retries).
        tokens: Total tokens (prompt + completion).
        cache_read_tokens: Tokens served from provider prompt cache (0 if
            the provider doesn't report caching or the prompt wasn't cached).
        cache_creation_tokens: Tokens written to the provider prompt cache.
    """

    __slots__ = (
        "parsed",
        "cost",
        "tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
    )

    def __init__(
        self,
        parsed: Any,
        cost: float,
        tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        self.parsed = parsed
        self.cost = cost
        self.tokens = tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens

    # Allow ``result, cost, tokens = call_llm_structured(...)``
    def __iter__(self):
        return iter((self.parsed, self.cost, self.tokens))

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (
            f"LLMResult(cost={self.cost:.4f}, tokens={self.tokens}, "
            f"cache_read={self.cache_read_tokens}, "
            f"cache_creation={self.cache_creation_tokens})"
        )


# Patterns indicating the API key or account has hit a hard spending cap.
# Matched case-insensitively against the full error message.
_BUDGET_EXHAUSTED_PATTERNS = (
    "requires more credits",
    "insufficient_quota",
    "billing_hard_limit_reached",
    "exceeded your current quota",
    "payment required",
)


def _is_budget_exhausted(error_msg: str) -> bool:
    """Return True if *error_msg* indicates a hard credit/budget limit."""
    msg = error_msg.lower()
    # HTTP 402 Payment Required is the canonical signal from OpenRouter
    if "402" in msg and ("credit" in msg or "payment" in msg or "afford" in msg):
        return True
    return any(p in msg for p in _BUDGET_EXHAUSTED_PATTERNS)


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
        "max_tokens": 16000,  # Observed completions are ~9-10K; 16K gives ample headroom
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


@lru_cache(maxsize=1)
def _can_reach_github(timeout: float = 1.5) -> bool:
    """Fast cached check for GitHub raw content connectivity.

    Used to decide whether LiteLLM should fetch remote model pricing
    and Anthropic beta headers, or use bundled local copies.

    Returns True when GitHub is reachable (e.g. login nodes with
    internet), False on air-gapped compute nodes (e.g. Titan).
    """
    import socket

    try:
        socket.create_connection(
            ("raw.githubusercontent.com", 443), timeout=timeout
        ).close()
        return True
    except OSError:
        return False


def set_litellm_offline_env() -> None:
    """Set env vars to prevent LiteLLM import-time remote fetches.

    **Must be called before** ``import litellm`` to be effective.
    Only sets the vars when GitHub is unreachable (air-gapped nodes).
    When GitHub is reachable, lets LiteLLM fetch the latest data.
    """
    if not _can_reach_github():
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        os.environ.setdefault("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
        logger.debug("Air-gapped: using local LiteLLM model cost map")


def suppress_litellm_noise() -> None:
    """Suppress all LiteLLM and HuggingFace diagnostic output.

    LiteLLM prints "Give Feedback", "Provider List", and debug info
    directly to stdout/stderr via print() calls, bypassing Python's
    logging system. This function suppresses both:
    1. Logger-based output (via logging levels)
    2. Print-based output (via litellm.suppress_debug_info)

    Also suppresses huggingface_hub/transformers/sentence_transformers
    logging which pollutes discovery output when these are installed.

    Call this once at module load time in any module that uses LiteLLM.
    """
    set_litellm_offline_env()
    try:
        import litellm
    except ModuleNotFoundError:
        # litellm not installed — nothing to suppress.  This can happen
        # when the CLI is imported outside the managed venv (e.g. bare
        # ``python`` on WSL) and litellm is not on sys.path.
        return

    # Suppress print-based diagnostic messages
    litellm.suppress_debug_info = True

    # Suppress all litellm logging to ERROR level
    for logger_name in (
        "LiteLLM",
        "LiteLLM Proxy",
        "LiteLLM Router",
        "httpx",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        level = logging.WARNING if logger_name == "httpx" else logging.ERROR
        logging.getLogger(logger_name).setLevel(level)

    # Environment variables for litellm internals
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    # Disable hf_xet native bindings (set early in __init__.py too,
    # but reinforce here in case suppress_litellm_noise is called
    # before package __init__ in some import path)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


# ---------------------------------------------------------------------------
# API key / model helpers
# ---------------------------------------------------------------------------


def get_api_key() -> str:
    """Get OpenRouter API key from environment.

    Raises:
        ValueError: If OPENROUTER_API_KEY_IMAS_CODEX is not set.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY_IMAS_CODEX")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY_IMAS_CODEX not set. Add to .env or export."
        )
    return api_key


def get_api_key_for_service(service: str) -> str:
    """Get OpenRouter API key, with per-service override support.

    Checks OPENROUTER_API_KEY_<SERVICE_UPPER> first (hyphens → underscores),
    then falls back to OPENROUTER_API_KEY_IMAS_CODEX.

    Examples:
        service="facility-discovery" → checks OPENROUTER_API_KEY_FACILITY_DISCOVERY
        service="standard-names" → checks OPENROUTER_API_KEY_STANDARD_NAMES
        service="untagged" → checks OPENROUTER_API_KEY_UNTAGGED (unlikely set)
    """
    if service and service != "untagged":
        env_var = f"OPENROUTER_API_KEY_{service.upper().replace('-', '_')}"
        per_service_key = os.environ.get(env_var)
        if per_service_key:
            return per_service_key
    return get_api_key()


_LOCAL_MODEL_PREFIXES = ("ollama/", "hosted_vllm/", "openai/localhost")


def ensure_model_prefix(model: str) -> str:
    """Ensure model ID has the correct provider prefix for LiteLLM routing.

    OpenRouter models get the ``openrouter/`` prefix to preserve
    ``cache_control`` blocks. Local models (ollama, vLLM) are passed
    through without modification.
    """
    if any(model.startswith(p) for p in _LOCAL_MODEL_PREFIXES):
        return model
    if not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


# ---------------------------------------------------------------------------
# JSON schema format — always convert Pydantic models to json_schema dicts
# ---------------------------------------------------------------------------


def _is_pydantic_model(obj: Any) -> bool:
    """Check if obj is a Pydantic BaseModel class (not instance)."""
    try:
        from pydantic import BaseModel

        return isinstance(obj, type) and issubclass(obj, BaseModel)
    except ImportError:
        return False


def _strip_unsupported_schema_props(schema: dict) -> dict:
    """Recursively strip JSON Schema properties unsupported by some providers.

    Anthropic (via Azure) rejects ``maxItems``, ``minItems``, ``minLength``,
    ``maxLength`` (on arrays), ``pattern``, and ``minimum``/``maximum``
    (on numbers) in structured output schemas even with ``strict: false``.
    We strip them so the schema works across all providers; our Pydantic
    parsing still validates these constraints.
    """
    unsupported = {
        "maxItems",
        "minItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
    }
    cleaned: dict = {}
    for key, value in schema.items():
        if key in unsupported:
            continue
        if isinstance(value, dict):
            cleaned[key] = _strip_unsupported_schema_props(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_unsupported_schema_props(item)
                if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _to_json_schema_format(model_cls: type) -> dict:
    """Convert a Pydantic model class to a non-strict json_schema format.

    Uses ``strict: false`` so that freeform dicts (``dict[str, str]``)
    are accepted by the API.  Strips provider-unsupported constraints
    (``maxItems``, etc.) — Pydantic parsing validates them instead.
    """
    raw_schema = model_cls.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model_cls.__name__,
            "strict": False,
            "schema": _strip_unsupported_schema_props(raw_schema),
        },
    }


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
    if hasattr(response, "_hidden_params"):
        cost = response._hidden_params.get("response_cost")
        if cost is not None:
            return float(cost)

    usage = getattr(response, "usage", None)
    if usage:
        input_tokens = usage.prompt_tokens or 0
        output_tokens = usage.completion_tokens or 0
        return (input_tokens * 3 + output_tokens * 15) / 1_000_000
    return 0.0


def _extract_cache_fields(ptd: Any) -> tuple[int, int]:
    """Extract cache read/write token counts from prompt_tokens_details.

    Different providers use different field names:
    - litellm formal: ``cache_creation_tokens`` (None by default)
    - OpenRouter extra: ``cache_write_tokens``
    - Both use: ``cached_tokens`` for reads

    Returns ``(cached_read, cache_write)`` with 0 defaults.
    """
    if ptd is None:
        return 0, 0
    cached = getattr(ptd, "cached_tokens", 0) or 0
    # Check both litellm formal field and OpenRouter's extra field
    cache_write = getattr(ptd, "cache_creation_tokens", 0) or 0
    if cache_write == 0:
        cache_write = getattr(ptd, "cache_write_tokens", 0) or 0
    return cached, cache_write


def _log_cache_metrics(response: Any, model: str) -> None:
    """Log prompt cache hit/miss metrics from LLM response usage.

    Providers report cached token counts in ``usage.prompt_tokens_details``.
    This logs at DEBUG level for post-hoc analysis of cache effectiveness
    via the auto-rotating CLI log files.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached, cache_write = _extract_cache_fields(ptd)
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    cost = extract_cost(response)

    if cached > 0:
        pct = cached / prompt * 100 if prompt > 0 else 0
        logger.debug(
            "LLM cache HIT: %d/%d prompt tokens cached (%.0f%%), "
            "completion=%d, cost=$%.4f, model=%s",
            cached,
            prompt,
            pct,
            completion,
            cost,
            model,
        )
    elif cache_write > 0:
        logger.debug(
            "LLM cache WRITE: %d tokens written, prompt=%d, "
            "completion=%d, cost=$%.4f, model=%s",
            cache_write,
            prompt,
            completion,
            cost,
            model,
        )
    else:
        logger.debug(
            "LLM cache MISS: prompt=%d, completion=%d, cost=$%.4f, model=%s",
            prompt,
            completion,
            cost,
            model,
        )


def extract_cache_tokens(response: Any) -> tuple[int, int]:
    """Extract prompt-cache token counts from an LLM response.

    Providers (OpenRouter, Anthropic, OpenAI) report cached token counts
    in ``usage.prompt_tokens_details``.  This helper mirrors the extraction
    logic of :func:`_log_cache_metrics` but returns the values instead of
    logging them, so callers can accumulate cache statistics.

    Args:
        response: Raw litellm response object.

    Returns:
        ``(cache_read_tokens, cache_creation_tokens)`` — both default to 0
        when the provider does not report caching information.
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    return _extract_cache_fields(ptd)


def _sanitize_content(content: str) -> str:
    """Sanitize LLM response content for JSON parsing.

    Removes control characters, strips markdown code fences, extracts JSON
    from prose wrappers, and fixes surrogate encoding issues.

    Args:
        content: Raw LLM response content string.

    Returns:
        Cleaned string safe for JSON/Pydantic parsing.
    """
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    content = re.sub(r"^```(?:json)?\s*\n?", "", content.strip())
    content = re.sub(r"\n?```\s*$", "", content)

    # Extract JSON from prose wrappers — LLMs sometimes "think aloud" before
    # or after the JSON object (e.g. "Looking at these paths...\n{...}")
    stripped = content.strip()
    if stripped and stripped[0] not in ("{", "["):
        json_start = _find_json_start(stripped)
        if json_start >= 0:
            content = _extract_balanced_json(stripped, json_start)

    # Remove control characters
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

    # Fix surrogate encoding issues
    content = content.encode("utf-8", errors="surrogateescape").decode(
        "utf-8", errors="replace"
    )
    return content


def _find_json_start(text: str) -> int:
    """Find the start of a JSON object or array in text.

    Looks for ``{`` or ``[`` that is likely the start of a JSON value,
    skipping occurrences inside obvious prose (e.g. ``{variable}``
    template markers).
    """
    for i, ch in enumerate(text):
        if ch == "{":
            # Skip Jinja/template-like markers: single word in braces
            # e.g. {variable} but not {"key": ...}
            rest = text[i + 1 : i + 50]
            if rest.lstrip().startswith('"') or rest.lstrip().startswith("'"):
                return i
            # Also accept if it looks like the start of JSON with newlines
            if "\n" in text[i : i + 200]:
                return i
            # Fallback: accept any { that's followed by another { or "
            for ch2 in rest:
                if ch2 in ('"', "'", "{", "["):
                    return i
                if ch2 in (" ", "\t", "\n", "\r"):
                    continue
                break
        elif ch == "[":
            return i
    return -1


def _extract_balanced_json(text: str, start: int) -> str:
    """Extract a balanced JSON object/array starting at *start*.

    Uses a simple brace/bracket counter that respects JSON strings
    (skipping escaped characters inside double quotes).
    Falls back to ``text[start:]`` if no balanced close is found.
    """
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        i += 1
    # No balanced close found — return from start to end
    return text[start:]


# ---------------------------------------------------------------------------
# Prompt caching (data-driven from config/prompt_caching.yaml)
# ---------------------------------------------------------------------------

_PROMPT_CACHING_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "prompt_caching.yaml"
)


@lru_cache(maxsize=1)
def _cache_control_patterns() -> tuple[str, ...]:
    """Match patterns for providers needing explicit cache_control breakpoints."""
    with _PROMPT_CACHING_CONFIG.open() as f:
        config = yaml.safe_load(f)
    patterns: list[str] = []
    for provider in config.get("providers", {}).values():
        patterns.extend(provider.get("match", []))
    return tuple(patterns)


def _supports_cache_control(model: str) -> bool:
    """Check if a model needs explicit cache_control breakpoints."""
    model_lower = model.lower()
    return any(p in model_lower for p in _cache_control_patterns())


def inject_cache_control(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add cache_control breakpoints to system messages for prompt caching.

    Converts the last system message's content to a content-block list
    with ``cache_control: {"type": "ephemeral"}`` on the last block.
    Provider support is configured in ``config/prompt_caching.yaml``.

    The default 5-minute ``ephemeral`` TTL is optimal for discovery
    workers that fire calls every 1-3 seconds, keeping the cache warm
    continuously throughout a CLI run.

    Args:
        messages: Chat messages (not mutated — a shallow copy is returned).

    Returns:
        New message list with cache_control injected on the last system message.
    """
    messages = [m.copy() for m in messages]

    # Walk backwards to find the last system message
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "system":
            content = messages[i]["content"]
            if isinstance(content, str):
                messages[i]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list) and content:
                # Already in block format — add cache_control to last block
                content = [block.copy() for block in content]
                content[-1]["cache_control"] = {"type": "ephemeral"}
                messages[i]["content"] = content
            break

    return messages


def _build_kwargs(
    model: str,
    api_key: str,
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None,
    max_tokens: int | None,
    temperature: float | None,
    timeout: int | None,
    *,
    service: str = "untagged",
) -> dict[str, Any]:
    """Build litellm completion kwargs with model-aware defaults.

    When max_tokens or timeout are not explicitly set, uses
    model-family defaults from MODEL_TOKEN_LIMITS.

    Routes through the LiteLLM proxy when ``[llm].location`` is configured
    or ``LITELLM_PROXY_URL`` is set (essential on air-gapped clusters where
    direct outbound HTTPS is unavailable).  The proxy handles model routing
    via its own ``model_list`` configuration.

    For models with caching support (per ``config/prompt_caching.yaml``),
    ``cache_control`` breakpoints are injected on the system prompt to
    enable prompt caching via OpenRouter.
    """
    from imas_codex.settings import get_llm_location, get_llm_proxy_url

    limits = get_model_limits(model)

    # Route through LiteLLM proxy when configured (location != "local" or env override)
    llm_location = get_llm_location()

    # Inject cache_control for models that support explicit breakpoints.
    # This must happen before building kwargs so both proxy and direct
    # paths benefit from prompt caching.
    supports_cache = _supports_cache_control(model)
    if supports_cache:
        messages = inject_cache_control(messages)

    # Decide routing: proxy vs direct.
    # The LiteLLM proxy strips cache_control blocks and response_cost from
    # responses, breaking prompt caching and actual cost reporting.  When
    # the model supports caching and a direct API key is available, bypass
    # the proxy to preserve both.
    use_proxy = llm_location != "local" or bool(os.getenv("LITELLM_PROXY_URL"))
    bypass_proxy = supports_cache and bool(os.getenv("OPENROUTER_API_KEY_IMAS_CODEX"))

    if use_proxy and not bypass_proxy:
        proxy_url = get_llm_proxy_url()
        # The proxy model names use the openrouter/ prefix (e.g.
        # openrouter/google/gemini-3.1-flash-lite-preview).  When calling
        # the proxy, wrap in openai/ so the *client* litellm treats the
        # proxy as an OpenAI-compatible endpoint and sends the model name
        # verbatim instead of interpreting the openrouter/ prefix as a
        # provider hint and bypassing the proxy.
        model_id = f"openai/{ensure_model_prefix(model)}"
        proxy_key = os.getenv("LITELLM_API_KEY") or os.getenv(
            "LITELLM_MASTER_KEY", api_key
        )
        kwargs: dict[str, Any] = {
            "model": model_id,
            "api_key": proxy_key,
            "api_base": proxy_url,
            "max_tokens": max_tokens
            if max_tokens is not None
            else limits["max_tokens"],
            "timeout": timeout if timeout is not None else limits["timeout"],
            "messages": messages,
        }
    else:
        model_id = ensure_model_prefix(model)
        # When bypassing the proxy, always use the OpenRouter API key
        # (not the proxy key that might have been passed in).
        direct_key = get_api_key_for_service(service) if bypass_proxy else api_key

        kwargs = {
            "model": model_id,
            "api_key": direct_key,
            "max_tokens": max_tokens
            if max_tokens is not None
            else limits["max_tokens"],
            "timeout": timeout if timeout is not None else limits["timeout"],
            "messages": messages,
        }

        if bypass_proxy:
            logger.debug("Bypassing proxy for %s (cache_control preserved)", model_id)

    if response_format is not None:
        # Always convert Pydantic models to explicit json_schema dicts.
        # Passing raw Pydantic classes through LiteLLM proxy → OpenRouter
        # does not reliably enforce structured output for all providers.
        # The explicit schema dict is provider-agnostic; strict=false lets
        # the model use it as guidance while our Pydantic parsing validates.
        if _is_pydantic_model(response_format):
            kwargs["response_format"] = _to_json_schema_format(response_format)
        else:
            kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Per-service X-Title for OpenRouter dashboard visibility.
    # Client extra_headers shallow-replaces proxy config extra_headers,
    # so per-service titles override the proxy fallback "imas-codex".
    title = f"imas-codex:{service}"
    kwargs["extra_headers"] = {
        "X-Title": title,
        "HTTP-Referer": "https://github.com/iterorganization/imas-codex",
    }

    kwargs["metadata"] = {
        "service": service,
    }

    return kwargs


# ---------------------------------------------------------------------------
# Structured output: call + parse in shared retry loop
# ---------------------------------------------------------------------------


def call_llm_structured(
    model: str,
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
) -> LLMResult:
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
        :class:`LLMResult` — backward-compatible with 3-tuple unpacking
        ``(parsed_model, total_cost_usd, total_tokens)`` and also carries
        ``cache_read_tokens`` / ``cache_creation_tokens``.

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
        service=service,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            total_cost += extract_cost(response)
            _log_cache_metrics(response, model)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = response_model.model_validate_json(content)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            cache_read, cache_creation = extract_cache_tokens(response)
            return LLMResult(
                parsed, total_cost, total_tokens, cache_read, cache_creation
            )

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
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
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
) -> LLMResult:
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
        :class:`LLMResult` — backward-compatible with 3-tuple unpacking
        ``(parsed_model, total_cost_usd, total_tokens)`` and also carries
        ``cache_read_tokens`` / ``cache_creation_tokens``.

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
        service=service,
    )

    last_error: Exception | None = None
    total_cost = 0.0

    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            total_cost += extract_cost(response)
            _log_cache_metrics(response, model)

            # Parse response content through Pydantic
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty response content")

            content = _sanitize_content(content)
            parsed = response_model.model_validate_json(content)

            total_tokens = (
                response.usage.prompt_tokens + response.usage.completion_tokens
            )
            cache_read, cache_creation = extract_cache_tokens(response)
            return LLMResult(
                parsed, total_cost, total_tokens, cache_read, cache_creation
            )

        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
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
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
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
        service=service,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = litellm.completion(**kwargs)
            cost = extract_cost(response)
            _log_cache_metrics(response, model)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
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
    messages: list[dict[str, Any]],
    response_format: type[BaseModel] | None = None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    service: str = "untagged",
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
        service=service,
    )

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = await litellm.acompletion(**kwargs)
            cost = extract_cost(response)
            _log_cache_metrics(response, model)
            return response, cost
        except Exception as e:
            last_error = e
            error_msg = str(e)
            if _is_budget_exhausted(error_msg):
                raise ProviderBudgetExhausted(
                    f"LLM provider budget exhausted: {error_msg[:200]}"
                ) from e
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
