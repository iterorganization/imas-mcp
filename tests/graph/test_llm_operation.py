"""Tests for LLMUsage dataclass and apply_llm_operation helper."""

from __future__ import annotations

from datetime import UTC, datetime

from imas_codex.graph.llm_operation import LLMUsage, apply_llm_operation


class TestLLMUsage:
    """LLMUsage dataclass defaults and field types."""

    def test_defaults(self):
        usage = LLMUsage(
            model="openrouter/anthropic/claude-sonnet-4.5",
            cost=0.0042,
            tokens_in=1200,
            tokens_out=300,
        )
        assert usage.tokens_cached_read == 0
        assert usage.tokens_cached_write == 0
        assert usage.service is None
        assert isinstance(usage.at, datetime)
        assert usage.at.tzinfo is not None  # timezone-aware

    def test_all_fields(self):
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=UTC)
        usage = LLMUsage(
            model="google/gemini-3-flash",
            cost=0.001,
            tokens_in=500,
            tokens_out=100,
            tokens_cached_read=200,
            tokens_cached_write=50,
            service="standard-names",
            at=ts,
        )
        assert usage.model == "google/gemini-3-flash"
        assert usage.cost == 0.001
        assert usage.tokens_in == 500
        assert usage.tokens_out == 100
        assert usage.tokens_cached_read == 200
        assert usage.tokens_cached_write == 50
        assert usage.service == "standard-names"
        assert usage.at == ts


class TestApplyLLMOperation:
    """apply_llm_operation round-trip."""

    def test_stamps_all_fields(self):
        usage = LLMUsage(
            model="test/model",
            cost=0.01,
            tokens_in=100,
            tokens_out=50,
            tokens_cached_read=30,
            tokens_cached_write=10,
            service="test-service",
        )
        props: dict = {"id": "my-node", "name": "test"}
        result = apply_llm_operation(props, usage)

        assert result is props  # mutates in-place and returns
        assert props["llm_model"] == "test/model"
        assert props["llm_cost"] == 0.01
        assert props["llm_tokens_in"] == 100
        assert props["llm_tokens_out"] == 50
        assert props["llm_tokens_cached_read"] == 30
        assert props["llm_tokens_cached_write"] == 10
        assert props["llm_service"] == "test-service"
        # llm_at should be an ISO string
        assert isinstance(props["llm_at"], str)
        assert "T" in props["llm_at"]

    def test_preserves_existing_keys(self):
        usage = LLMUsage(model="m", cost=0, tokens_in=0, tokens_out=0)
        props = {"id": "n1", "score": 0.95}
        apply_llm_operation(props, usage)
        assert props["id"] == "n1"
        assert props["score"] == 0.95

    def test_iso_string_passthrough(self):
        """If .at is already a string, pass it through unchanged."""
        usage = LLMUsage(model="m", cost=0, tokens_in=0, tokens_out=0)
        usage.at = "2025-01-15T10:30:00+00:00"  # type: ignore[assignment]
        props: dict = {}
        apply_llm_operation(props, usage)
        assert props["llm_at"] == "2025-01-15T10:30:00+00:00"
