"""W4c: LLMCost recording — pool normalisation and event_type tagging."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.graph_ops import _normalize_pool


class TestNormalizePool:
    """Verify phase → pool mapping covers all known phase labels."""

    @pytest.mark.parametrize(
        "phase,expected_pool",
        [
            ("generate", "compose"),
            ("generate_name", "compose"),
            ("compose", "compose"),
            ("regen", "refine_name"),
            ("refine_name", "refine_name"),
            ("refine_docs", "refine_docs"),
            ("validate", "validate"),
            ("validate_name", "validate"),
            ("review_names", "review"),
            ("review_docs", "review"),
            ("review", "review"),
            ("enrich", "enrich"),
            ("enrich_links", "enrich"),
        ],
    )
    def test_known_phase_maps_correctly(self, phase: str, expected_pool: str):
        assert _normalize_pool(phase) == expected_pool

    def test_unknown_phase_passes_through(self):
        """Unknown phases should pass through as-is (no KeyError)."""
        assert _normalize_pool("future_phase") == "future_phase"

    def test_canonical_pools_are_six(self):
        """The set of output pool values should be exactly six."""
        from imas_codex.standard_names.graph_ops import _PHASE_TO_POOL

        pools = set(_PHASE_TO_POOL.values())
        assert pools == {
            "compose",
            "refine_name",
            "refine_docs",
            "validate",
            "review",
            "enrich",
        }
