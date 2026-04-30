"""B12: bounded full-batch compose retry with re-enrichment in pool path.

Tests that ``compose_batch`` retries the LLM call when grammar validation
fails on the first attempt, re-enriches items with expanded DD context,
and only falls through to L6 per-candidate retry on the final attempt.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch_item(
    path: str = "equilibrium/time_slice/profiles_1d/psi", **kw
) -> dict[str, Any]:
    base = {
        "path": path,
        "description": "Poloidal flux",
        "physics_domain": "equilibrium",
        "unit": "Wb",
        "data_type": "FLT_1D",
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }
    base.update(kw)
    return base


class _FakeCandidate:
    """Mimics a StandardNameComposeBatch candidate."""

    def __init__(self, name: str, source_id: str, **kw):
        self.standard_name = name
        self.source_id = source_id
        self.description = kw.get("description", "desc")
        self.kind = kw.get("kind", "scalar")
        self.dd_paths = kw.get("dd_paths", [source_id])
        self.grammar_fields = kw.get("grammar_fields", {})
        self.reason = kw.get("reason", "")


class _FakeBatchResult:
    """Mimics StandardNameComposeBatch LLM response."""

    def __init__(
        self,
        candidates: list[_FakeCandidate],
        vocab_gaps=None,
        attachments=None,
        skipped=None,
    ):
        self.candidates = candidates
        self.vocab_gaps = vocab_gaps or []
        self.attachments = attachments or []
        self.skipped = skipped or []


class _FakeLLMOut:
    """Mimics the LLMResult triple from acall_llm_structured."""

    def __init__(self, result, cost=0.01, tokens=100):
        self._result = result
        self._cost = cost
        self._tokens = tokens
        self.input_tokens = tokens
        self.output_tokens = tokens // 2
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0

    def __iter__(self):
        return iter((self._result, self._cost, self._tokens))


class _FakeBudgetManager:
    """Minimal BudgetManager stub."""

    def __init__(self):
        self.reserved = 0.0

    def reserve(self, amount, phase=""):
        self.reserved += amount
        return _FakeLease()


class _FakeLease:
    def charge_event(self, cost, event):
        return MagicMock(overspend=0)

    def release_unused(self):
        pass


# Names that will pass/fail grammar parsing
_GOOD_NAME = "electron_temperature"
_BAD_NAME = "bad!!!name"


# ---------------------------------------------------------------------------
# Fixture — deep-patch compose_batch's local imports
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_compose_deps():
    """Patch all local imports inside compose_batch so it can run in isolation."""
    mock_gc = MagicMock()
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)
    mock_gc.query = MagicMock(return_value=[])

    patches = [
        # Context / settings
        patch(
            "imas_codex.standard_names.context.build_compose_context",
            return_value={},
        ),
        patch("imas_codex.settings.get_compose_lean", return_value=False),
        patch("imas_codex.settings.get_model", return_value="test-model"),
        # Prompt rendering
        patch("imas_codex.llm.prompt_loader.render_prompt", return_value="prompt"),
        # Enrichment
        patch(
            "imas_codex.standard_names.workers._enrich_batch_items",
            side_effect=lambda items: None,
        ),
        patch(
            "imas_codex.standard_names.workers._search_nearby_names",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.workers._enrich_ids_context",
            return_value=None,
        ),
        patch(
            "imas_codex.standard_names.context.build_domain_vocabulary_preseed",
            return_value="",
        ),
        patch(
            "imas_codex.standard_names.review.themes.extract_reviewer_themes",
            return_value=[],
        ),
        patch(
            "imas_codex.standard_names.example_loader.load_compose_examples",
            return_value=[],
        ),
        # Graph client
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc),
        # Persist
        patch(
            "imas_codex.standard_names.graph_ops.persist_generated_name_batch",
            return_value=1,
        ),
        # Audits
        patch(
            "imas_codex.standard_names.audits.run_audits",
            return_value=[],
        ),
        # Misc
        patch(
            "imas_codex.standard_names.workers._auto_detect_physical_base_gaps",
            return_value=[],
        ),
    ]
    for p in patches:
        p.start()
    yield
    patch.stopall()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPoolComposeRetry:
    """B12: full-batch retry with re-enrichment in pool compose_batch."""

    @pytest.mark.asyncio
    async def test_all_pass_no_retry(self, _patch_compose_deps) -> None:
        """When all candidates parse on first try, no retry occurs."""
        from imas_codex.standard_names.workers import compose_batch

        good_result = _FakeBatchResult(
            [
                _FakeCandidate(_GOOD_NAME, "equilibrium/time_slice/profiles_1d/psi"),
            ]
        )
        llm_out = _FakeLLMOut(good_result)

        call_count = 0

        async def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return llm_out

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("imas_codex.standard_names.workers._retry_attempts", return_value=1),
        ):
            batch = [_make_batch_item()]
            stop = asyncio.Event()
            mgr = _FakeBudgetManager()
            await compose_batch(batch, mgr, stop)

        assert call_count == 1, "Should not retry when all candidates pass"

    @pytest.mark.asyncio
    async def test_grammar_fail_triggers_retry(self, _patch_compose_deps) -> None:
        """Grammar failure on first attempt triggers re-enrich + re-LLM."""
        import sys

        from imas_codex.standard_names.workers import compose_batch

        _PARSEABLE_BAD = "xyzzy_garbage_invalid_token"
        bad_result = _FakeBatchResult(
            [
                _FakeCandidate(
                    _PARSEABLE_BAD,
                    "equilibrium/time_slice/profiles_1d/psi",
                ),
            ]
        )
        good_result = _FakeBatchResult(
            [
                _FakeCandidate(_GOOD_NAME, "equilibrium/time_slice/profiles_1d/psi"),
            ]
        )

        call_count = 0
        re_enrich_called = False

        async def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _FakeLLMOut(bad_result)
            return _FakeLLMOut(good_result)

        original_to_thread = asyncio.to_thread

        async def patched_to_thread(fn, *args, **kwargs):
            nonlocal re_enrich_called
            if hasattr(fn, "__name__") and fn.__name__ == "_re_enrich_expanded":
                re_enrich_called = True
                return None
            return await original_to_thread(fn, *args, **kwargs)

        # Mock grammar so bad name fails, good name passes
        mock_grammar = MagicMock()

        def _mock_parse(name):
            if name == _GOOD_NAME:
                return MagicMock(physical_base="temperature")
            raise ValueError(f"Grammar parse failed for {name}")

        def _mock_compose(parsed):
            return _GOOD_NAME

        mock_grammar.parse_standard_name = _mock_parse
        mock_grammar.compose_standard_name = _mock_compose

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("imas_codex.standard_names.workers._retry_attempts", return_value=1),
            patch("asyncio.to_thread", patched_to_thread),
            patch.dict(
                sys.modules,
                {"imas_standard_names.grammar": mock_grammar},
            ),
        ):
            batch = [_make_batch_item()]
            stop = asyncio.Event()
            mgr = _FakeBudgetManager()
            await compose_batch(batch, mgr, stop)

        assert call_count == 2, "Should retry once on grammar failure"
        assert re_enrich_called, "Should re-enrich items before retry"

    @pytest.mark.asyncio
    async def test_retry_exhausted_falls_through(self, _patch_compose_deps) -> None:
        """When retries are exhausted, falls through to L6 per-candidate."""
        import sys

        from imas_codex.standard_names.workers import compose_batch

        # Use a snake_case name that passes pre-validation but fails grammar
        _PARSEABLE_BAD = "xyzzy_garbage_invalid_token"
        bad_result = _FakeBatchResult(
            [
                _FakeCandidate(
                    _PARSEABLE_BAD,
                    "equilibrium/time_slice/profiles_1d/psi",
                ),
            ]
        )

        call_count = 0

        async def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _FakeLLMOut(bad_result)

        grammar_retry_called = False

        async def mock_grammar_retry(name, error, model, acall_fn):
            nonlocal grammar_retry_called
            grammar_retry_called = True
            return (_GOOD_NAME, 0.001, 10, 5)

        # Mock imas_standard_names.grammar so parse raises for bad names
        mock_grammar_module = MagicMock()

        def _mock_parse(name):
            if name == _GOOD_NAME:
                return MagicMock(physical_base="temperature")
            raise ValueError(f"Grammar parse failed for {name}")

        def _mock_compose(parsed):
            return _GOOD_NAME

        mock_grammar_module.parse_standard_name = _mock_parse
        mock_grammar_module.compose_standard_name = _mock_compose

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("imas_codex.standard_names.workers._retry_attempts", return_value=1),
            patch(
                "imas_codex.standard_names.workers._grammar_retry",
                mock_grammar_retry,
            ),
            patch.dict(
                sys.modules,
                {"imas_standard_names.grammar": mock_grammar_module},
            ),
        ):
            batch = [_make_batch_item()]
            stop = asyncio.Event()
            mgr = _FakeBudgetManager()
            await compose_batch(batch, mgr, stop)

        # 1 initial + 1 retry = 2 LLM calls
        assert call_count == 2
        assert grammar_retry_called, "L6 grammar retry should run after B12 exhausted"

    @pytest.mark.asyncio
    async def test_retry_disabled_when_zero(self, _patch_compose_deps) -> None:
        """retry_attempts=0 disables the retry loop entirely."""
        from imas_codex.standard_names.workers import compose_batch

        bad_result = _FakeBatchResult(
            [
                _FakeCandidate(_BAD_NAME, "equilibrium/time_slice/profiles_1d/psi"),
            ]
        )

        call_count = 0

        async def mock_llm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _FakeLLMOut(bad_result)

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("imas_codex.standard_names.workers._retry_attempts", return_value=0),
            patch(
                "imas_codex.standard_names.workers._grammar_retry",
                AsyncMock(return_value=(None, 0.0, 0, 0)),
            ),
        ):
            batch = [_make_batch_item()]
            stop = asyncio.Event()
            mgr = _FakeBudgetManager()
            await compose_batch(batch, mgr, stop)

        assert call_count == 1, "Should NOT retry when retry_attempts=0"

    @pytest.mark.asyncio
    async def test_budget_reserves_for_retries(self, _patch_compose_deps) -> None:
        """Budget reservation accounts for retry_attempts."""
        from imas_codex.standard_names.workers import compose_batch

        good_result = _FakeBatchResult(
            [
                _FakeCandidate(_GOOD_NAME, "equilibrium/time_slice/profiles_1d/psi"),
            ]
        )

        async def mock_llm(*args, **kwargs):
            return _FakeLLMOut(good_result)

        with (
            patch("imas_codex.discovery.base.llm.acall_llm_structured", mock_llm),
            patch("imas_codex.standard_names.workers._retry_attempts", return_value=2),
        ):
            batch = [_make_batch_item()]
            stop = asyncio.Event()
            mgr = _FakeBudgetManager()
            await compose_batch(batch, mgr, stop)

        # 1 item × $0.20 × (2 retries + 1) = $0.60
        assert mgr.reserved == pytest.approx(0.60, abs=0.01)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
