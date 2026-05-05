"""Shared fixtures for standard name tests.

The ``mock_llm`` fixture replaces the real LLM call (``acall_llm_structured``)
with a deterministic, scripted response provider. Tests register expected
responses per (stage, model) and the mock returns them in registration order.

Stages map to worker pool names: ``'generate_name'``, ``'review_name'``,
``'refine_name'``, ``'generate_docs'``, ``'review_docs'``, ``'refine_docs'``.
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import patch

import pytest


@pytest.fixture()
def sample_standard_names() -> list[dict]:
    """Sample standard name dicts for write_standard_names testing."""
    return [
        {
            "id": "electron_temperature",
            "source_types": ["dd"],
            "source_id": "core_profiles/profiles_1d/electrons/temperature",
            "physical_base": "temperature",
            "subject": "electron",
            "description": "Electron temperature profile",
            "documentation": "The electron temperature $T_e$ is measured by Thomson scattering.",
            "kind": "scalar",
            "links": ["ion_temperature", "electron_density"],
            "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "core plasma",
            "constraints": ["T_e > 0"],
            "unit": "eV",
            "model": "test/model",
            "pipeline_status": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "plasma_current",
            "source_types": ["signals"],
            "source_id": "tcv:ip/measured",
            "physical_base": "current",
            "description": "Plasma current",
            "unit": "A",
            "kind": "scalar",
            "model": "test/model",
            "pipeline_status": "drafted",
            "generated_at": "2024-01-01T00:00:00Z",
        },
    ]


@pytest.fixture()
def mock_graph_client():
    """A mock GraphClient that records query calls."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.query = MagicMock(return_value=[])
    return client


# ---------------------------------------------------------------------------
# MockLLM — scripted LLM response fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ScriptedResponse:
    stage: str
    model: str | None  # None = match any model
    response: Any  # parsed pydantic instance OR dict that will be coerced
    cost: float = 0.0
    tokens: dict[str, int] | None = None


class MockLLM:
    """Scripted LLM mock for SN pipeline tests.

    Usage::

        def test_generate_then_review(mock_llm):
            mock_llm.add_response('generate_name', response=GeneratedName(name=...))
            mock_llm.add_response('review_name',   response=ReviewResult(score=0.9))
            await some_pipeline_step(...)
            assert mock_llm.calls_for('generate_name') == 1

    Fallthrough behaviour: if no scripted response matches a call, the mock
    raises ``RuntimeError`` so tests fail loudly rather than silently using
    the real LLM.
    """

    def __init__(self) -> None:
        self._queue: list[_ScriptedResponse] = []
        self._calls: list[dict[str, Any]] = []

    def add_response(
        self,
        stage: str,
        *,
        response: Any,
        model: str | None = None,
        cost: float = 0.0,
        tokens: dict[str, int] | None = None,
    ) -> None:
        """Register a scripted response. FIFO per stage."""
        self._queue.append(
            _ScriptedResponse(
                stage=stage,
                model=model,
                response=response,
                cost=cost,
                tokens=tokens or {"input": 0, "output": 0, "cached": 0},
            )
        )

    def calls_for(self, stage: str) -> int:
        """Return number of calls dispatched for *stage*."""
        return sum(1 for c in self._calls if c["stage"] == stage)

    @property
    def calls(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded call records."""
        return list(self._calls)

    async def _dispatch(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        service: str | None = None,
        stage: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, float, dict[str, int]]:
        """Replacement coroutine for ``acall_llm_structured``."""
        # Stage priority: explicit kwarg > inferred from messages > 'unknown'
        resolved_stage = stage or _infer_stage(messages, service) or "unknown"
        self._calls.append(
            {
                "stage": resolved_stage,
                "model": model,
                "service": service,
                "messages": messages,
                "response_model": response_model.__name__,
            }
        )

        for i, scripted in enumerate(self._queue):
            if scripted.stage != resolved_stage:
                continue
            if scripted.model is not None and scripted.model != model:
                continue
            self._queue.pop(i)
            payload = scripted.response
            if isinstance(payload, dict):
                payload = response_model(**payload)
            return payload, scripted.cost, scripted.tokens

        raise RuntimeError(
            f"MockLLM: no scripted response for stage={resolved_stage!r} "
            f"model={model!r}. Registered: "
            f"{[(s.stage, s.model) for s in self._queue]}"
        )


def _infer_stage(
    messages: list[dict[str, str]],
    service: str | None,  # noqa: ARG001
) -> str | None:
    """Best-effort stage inference from system-prompt keywords.

    Workers that need precision should pass ``stage=`` explicitly; this
    function is a fallback for callers that don't.
    """
    if not messages:
        return None
    sys_content = next(
        (m.get("content", "") for m in messages if m.get("role") == "system"), ""
    )
    s = sys_content.lower()
    if "review" in s:
        # Disambiguate review_name vs review_docs.  The name-axis review
        # prompt says "name-only mode" / "name-axis review"; the docs-axis
        # prompt says "evaluating…documentation" / "documentation text quality".
        # A naïve "documentation" check false-matches on the name prompt
        # because it mentions "documentation is filled in by a later pass".
        if "name-only mode" in s or "name-axis" in s:
            return "review_name"
        if (
            "documentation text quality" in s
            or "evaluating" in s
            and "documentation" in s
        ):
            return "review_docs"
        # Fallback: presence of "documentation quality" signals docs axis
        if "documentation quality" in s:
            return "review_docs"
        return "review_name"
    if "refine" in s or "regenerate" in s:
        if "documentation" in s or "description" in s:
            return "refine_docs"
        return "refine_name"
    if "documentation" in s or "enrich" in s:
        return "generate_docs"
    if "compose" in s or "standard name" in s:
        return "generate_name"
    return None


@pytest.fixture()
def mock_llm():
    """Patch ``acall_llm_structured`` globally and yield a :class:`MockLLM`.

    All SN workers import ``acall_llm_structured`` lazily (function-local
    ``from imas_codex.discovery.base.llm import acall_llm_structured``), so
    patching the canonical module attribute is sufficient to intercept every
    call.
    """
    mock = MockLLM()

    # Patch at every known import site.  Function-local imports all resolve
    # to the same object in imas_codex.discovery.base.llm, so one target
    # covers the standard_names workers.  Additional sites (discovery, graph)
    # are included so the fixture is safe to use in cross-module tests.
    targets = [
        # Canonical definition — covers all function-local SN imports:
        #   workers.py, enrich_workers.py, benchmark.py, review/pipeline.py
        "imas_codex.discovery.base.llm.acall_llm_structured",
        # Re-export in discovery.base.__init__ (used by some discovery tests)
        "imas_codex.discovery.base.acall_llm_structured",
    ]

    patches = []
    for target in targets:
        try:
            p = patch(target, side_effect=mock._dispatch)
            patches.append(p)
            p.start()
        except (ImportError, AttributeError):
            # Import site may not exist yet (e.g. Phase 2 code not landed).
            pass

    try:
        yield mock
    finally:
        for p in patches:
            p.stop()
