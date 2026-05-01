"""Tests for reviewer model compatibility — GPT-5.x temperature and Gemini JSON.

Covers:
  1. GPT-5.4 temperature guard in _build_kwargs (no temperature=0.0 sent).
  2. Gemini prose-wrapped JSON extraction via _sanitize_content.
  3. Review pipeline passes no explicit temperature (uses provider default).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. GPT-5.x temperature guard in _build_kwargs
# ---------------------------------------------------------------------------


def _make_build_kwargs(model: str, temperature: float | None) -> dict:
    """Call _build_kwargs with minimal mocks and return the resulting kwargs."""
    from imas_codex.discovery.base.llm import _build_kwargs

    messages = [{"role": "user", "content": "hello"}]
    # get_llm_location is imported lazily inside _build_kwargs from settings
    with (
        patch("imas_codex.settings.get_llm_location", return_value="local"),
        patch(
            "imas_codex.settings.get_llm_proxy_url",
            return_value="http://localhost:18400",
        ),
        patch(
            "imas_codex.discovery.base.llm.get_api_key_for_service",
            return_value="sk-test",
        ),
        patch(
            "imas_codex.discovery.base.llm._supports_cache_control",
            return_value=False,
        ),
    ):
        return _build_kwargs(
            model=model,
            api_key="sk-test",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=temperature,
            timeout=None,
        )


@pytest.mark.parametrize(
    "model,temperature,expect_temp_key",
    [
        # GPT-5.4 with temperature=0.0 → should NOT include temperature in kwargs
        ("openrouter/openai/gpt-5.4", 0.0, False),
        ("openai/gpt-5.4", 0.0, False),
        # GPT-5.4 with temperature=None → no temperature kwarg (already default)
        ("openrouter/openai/gpt-5.4", None, False),
        # GPT-5.4 with temperature=1.0 → should include (non-zero, non-None)
        ("openrouter/openai/gpt-5.4", 1.0, True),
        # Opus with temperature=0.0 → should include
        ("openrouter/anthropic/claude-opus-4.6", 0.0, True),
        # Gemini with temperature=0.0 → should include
        ("openrouter/google/gemini-3.1-pro-preview", 0.0, True),
        # None for any model → no temperature key
        ("openrouter/anthropic/claude-opus-4.6", None, False),
    ],
)
def test_build_kwargs_temperature(
    model: str, temperature: float | None, expect_temp_key: bool
) -> None:
    """_build_kwargs must not pass temperature=0.0 to GPT-5.x models."""
    kwargs = _make_build_kwargs(model, temperature)
    if expect_temp_key:
        assert "temperature" in kwargs, (
            f"Expected 'temperature' in kwargs for model={model!r}, "
            f"temperature={temperature!r}"
        )
    else:
        assert "temperature" not in kwargs, (
            f"Expected 'temperature' NOT in kwargs for model={model!r}, "
            f"temperature={temperature!r} — got {kwargs.get('temperature')!r}"
        )


# ---------------------------------------------------------------------------
# 2. Gemini prose-wrapped JSON extraction via _sanitize_content
# ---------------------------------------------------------------------------

_PROSE_WRAPPED_SAMPLES = [
    # Markdown code fence
    '```json\n{"score": 0.8, "comment": "looks good"}\n```',
    # Prose before JSON
    'Here is the review:\n{"score": 0.8, "comment": "looks good"}',
    # Prose before and after JSON
    'Sure! {"score": 0.8, "comment": "looks good"}\nDone.',
    # JSON only (clean)
    '{"score": 0.8, "comment": "looks good"}',
    # Markdown fence without language tag
    '```\n{"score": 0.8, "comment": "looks good"}\n```',
]


@pytest.mark.parametrize("raw_content", _PROSE_WRAPPED_SAMPLES)
def test_sanitize_content_extracts_json(raw_content: str) -> None:
    """_sanitize_content must extract valid JSON regardless of prose wrapping."""
    from imas_codex.discovery.base.llm import _sanitize_content

    sanitized = _sanitize_content(raw_content)
    parsed = json.loads(sanitized)
    assert parsed["score"] == pytest.approx(0.8)
    assert "looks good" in parsed["comment"]


def test_sanitize_content_large_review_batch() -> None:
    """_sanitize_content handles a full ReviewBatch-shaped JSON wrapped in prose."""
    from imas_codex.discovery.base.llm import _sanitize_content

    batch = {
        "reviews": [
            {
                "id": "plasma_current",
                "score": 0.9,
                "grammar_ok": True,
                "comment": "Well formed",
            }
        ]
    }
    prose_wrapped = (
        f"I have reviewed the names carefully.\n\n"
        f"```json\n{json.dumps(batch)}\n```\n\nHope this helps."
    )
    sanitized = _sanitize_content(prose_wrapped)
    parsed = json.loads(sanitized)
    assert parsed["reviews"][0]["id"] == "plasma_current"


# ---------------------------------------------------------------------------
# 3. Review pipeline acall_llm_structured invocation — no explicit temperature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_review_single_batch_no_explicit_temperature() -> None:
    """_review_single_batch must not pass explicit temperature to acall_llm_structured.

    Captures the kwargs passed to acall_llm_structured and asserts
    'temperature' is not in the call kwargs (so GPT-5.4 gets the provider
    default rather than 0.0).
    """
    from imas_codex.discovery.base.llm import LLMResult
    from imas_codex.standard_names.models import (
        StandardNameQualityReviewNameOnly,
        StandardNameQualityReviewNameOnlyBatch,
        StandardNameQualityScoreNameOnly,
    )

    fake_review = StandardNameQualityReviewNameOnly(
        source_id="plasma_current",
        standard_name="plasma_current",
        scores=StandardNameQualityScoreNameOnly(
            grammar=18, semantic=18, convention=18, completeness=16
        ),
        reasoning="Well formed name following CF conventions.",
    )
    fake_review_result = StandardNameQualityReviewNameOnlyBatch(reviews=[fake_review])
    fake_llm_result = LLMResult(fake_review_result, 0.001, 100)

    captured_kwargs: dict = {}

    async def _mock_acall(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_llm_result

    with (
        patch(
            "imas_codex.discovery.base.llm.acall_llm_structured",
            side_effect=_mock_acall,
        ),
        patch(
            "imas_codex.llm.prompt_loader.render_prompt",
            return_value="mock prompt",
        ),
    ):
        from imas_codex.standard_names.review.pipeline import _review_single_batch

        fake_item = {
            "id": "plasma_current",
            "source_id": "tcv:ip/measured",
            "standard_name": "plasma_current",
            "description": "Plasma current",
            "unit": "A",
            "kind": "scalar",
            "validation_issues": [],
        }

        await _review_single_batch(
            names=[fake_item],
            model="openrouter/openai/gpt-5.4",
            grammar_enums={},
            compose_ctx={},
            batch_context="test",
            neighborhood=[],
            audit_findings=[],
            wlog=MagicMock(),
            name_only=True,
            target="names",
        )

    assert "temperature" not in captured_kwargs, (
        f"pipeline must not pass explicit temperature; got: {captured_kwargs.get('temperature')!r}"
    )
