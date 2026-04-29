"""Smoke tests for the MockLLM fixture (P6.1).

Verifies that the fixture intercepts ``acall_llm_structured``, returns
scripted responses in FIFO order, and raises ``RuntimeError`` on unscripted
calls.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel


class _DummyResponse(BaseModel):
    value: str


@pytest.mark.asyncio
async def test_mock_llm_basic(mock_llm):
    """Scripted response is returned and call is recorded."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    mock_llm.add_response("generate_name", response=_DummyResponse(value="ok"))

    result, cost, tokens = await acall_llm_structured(
        model="openrouter/test/model",
        messages=[
            {"role": "system", "content": "You compose standard names."},
            {"role": "user", "content": "go"},
        ],
        response_model=_DummyResponse,
        service="facility-discovery",
    )

    assert result.value == "ok"
    assert cost == 0.0
    assert tokens == {"input": 0, "output": 0, "cached": 0}
    assert mock_llm.calls_for("generate_name") == 1


@pytest.mark.asyncio
async def test_mock_llm_unscripted_raises(mock_llm):
    """Calls without a scripted response raise RuntimeError immediately."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    with pytest.raises(RuntimeError, match="no scripted response"):
        await acall_llm_structured(
            model="openrouter/test/model",
            messages=[{"role": "system", "content": "You compose standard names."}],
            response_model=_DummyResponse,
            service="facility-discovery",
        )


@pytest.mark.asyncio
async def test_mock_llm_dict_coercion(mock_llm):
    """A dict scripted response is coerced to the expected Pydantic model."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    mock_llm.add_response("generate_name", response={"value": "coerced"})

    result, _, _ = await acall_llm_structured(
        model="openrouter/test/model",
        messages=[
            {"role": "system", "content": "You compose standard names."},
            {"role": "user", "content": "go"},
        ],
        response_model=_DummyResponse,
    )

    assert isinstance(result, _DummyResponse)
    assert result.value == "coerced"


@pytest.mark.asyncio
async def test_mock_llm_stage_inference_review(mock_llm):
    """Stage is inferred from 'review' keyword in system prompt."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    mock_llm.add_response("review_name", response=_DummyResponse(value="reviewed"))

    result, _, _ = await acall_llm_structured(
        model="openrouter/test/model",
        messages=[
            {"role": "system", "content": "You review standard names for quality."},
        ],
        response_model=_DummyResponse,
    )

    assert result.value == "reviewed"
    assert mock_llm.calls_for("review_name") == 1
    assert mock_llm.calls_for("generate_name") == 0


@pytest.mark.asyncio
async def test_mock_llm_fifo_ordering(mock_llm):
    """Multiple responses for the same stage are returned in FIFO order."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    mock_llm.add_response("generate_name", response=_DummyResponse(value="first"))
    mock_llm.add_response("generate_name", response=_DummyResponse(value="second"))

    msgs = [{"role": "system", "content": "You compose standard names."}]

    r1, _, _ = await acall_llm_structured(
        model="openrouter/test/model",
        messages=msgs,
        response_model=_DummyResponse,
    )
    r2, _, _ = await acall_llm_structured(
        model="openrouter/test/model",
        messages=msgs,
        response_model=_DummyResponse,
    )

    assert r1.value == "first"
    assert r2.value == "second"
    assert mock_llm.calls_for("generate_name") == 2


@pytest.mark.asyncio
async def test_mock_llm_model_filter(mock_llm):
    """Responses pinned to a specific model are skipped for other models."""
    from imas_codex.discovery.base.llm import acall_llm_structured

    mock_llm.add_response(
        "generate_name",
        response=_DummyResponse(value="pinned"),
        model="openrouter/specific/model",
    )
    mock_llm.add_response(
        "generate_name",
        response=_DummyResponse(value="any"),
    )

    # First call uses a different model — should skip the pinned entry and
    # consume the wildcard entry.
    result, _, _ = await acall_llm_structured(
        model="openrouter/other/model",
        messages=[{"role": "system", "content": "You compose standard names."}],
        response_model=_DummyResponse,
    )
    assert result.value == "any"
