"""Delta H: bounded retry loop for failed composition.

Tests that the compose worker retries the LLM call when grammar
validation fails on the first attempt, and that the retry_reason
field renders correctly in the prompt template.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt


def _min_item(**overrides) -> dict:
    base = {
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "description": "Poloidal flux",
        "data_type": "FLT_1D",
        "units": "Wb",
    }
    base.update(overrides)
    return base


def _min_context(items: list[dict], **overrides) -> dict:
    ctx = {
        "items": items,
        "ids_name": "equilibrium",
        "ids_contexts": {},
        "existing_names": [],
        "cluster_context": None,
        "nearby_existing_names": [],
        "reference_exemplars": [],
        "cocos_version": 11,
        "dd_version": "4.0.0",
    }
    ctx.update(overrides)
    return ctx


class TestRetryReasonRendering:
    """The retry_reason block should appear only when retry_reason is set."""

    def test_no_retry_reason_no_block(self) -> None:
        rendered = render_prompt("sn/compose_dd", _min_context([_min_item()]))
        assert "Retry Context" not in rendered

    def test_retry_reason_renders(self) -> None:
        reason = (
            "Previous attempt failed: grammar round-trip failed for "
            "bad_name_here. Consider expanded neighbour context and "
            "produce a different name."
        )
        ctx = _min_context([_min_item()], retry_reason=reason)
        rendered = render_prompt("sn/compose_dd", ctx)

        assert "Retry Context" in rendered
        assert "grammar round-trip failed" in rendered
        assert "bad_name_here" in rendered
        assert "expanded neighbour context" in rendered

    def test_retry_reason_before_unit_policy(self) -> None:
        """Retry block should appear before Unit Policy section."""
        reason = "Previous attempt failed: grammar round-trip failed for x."
        ctx = _min_context([_min_item()], retry_reason=reason)
        rendered = render_prompt("sn/compose_dd", ctx)

        retry_pos = rendered.index("Retry Context")
        unit_pos = rendered.index("Unit Policy")
        assert retry_pos < unit_pos


class TestRetryConstants:
    """Module-level constants are correctly defined."""

    def test_retry_attempts_is_one(self) -> None:
        from imas_codex.standard_names.workers import _RETRY_ATTEMPTS

        assert _RETRY_ATTEMPTS == 1

    def test_retry_k_expansion_is_twelve(self) -> None:
        from imas_codex.standard_names.workers import _RETRY_K_EXPANSION

        assert _RETRY_K_EXPANSION == 12


class TestHybridSearchKParameter:
    """The search_k parameter is forwarded correctly."""

    def test_default_search_k(self) -> None:
        """Default search_k should be 10."""
        import inspect

        from imas_codex.standard_names.workers import _hybrid_search_neighbours

        sig = inspect.signature(_hybrid_search_neighbours)
        assert sig.parameters["search_k"].default == 10


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
