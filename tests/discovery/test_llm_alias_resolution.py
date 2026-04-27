"""Tests for proxy alias resolution in _build_kwargs.

The LiteLLM proxy config defines ``cc:*`` aliases (e.g. ``cc:haiku``) for
billing-isolated access to Anthropic models.  When the imas-codex Python
client calls ``call_llm_structured(model="cc:haiku", ...)``, the alias
must be resolved to the real model name (``anthropic/claude-haiku-4.5``)
*before* routing decisions, so that:

1. ``_supports_cache_control()`` matches → ``cache_control`` injected
2. ``ensure_model_prefix()`` adds ``openrouter/`` → cost tracking works
3. ``get_model_limits()`` returns ``claude`` family limits
"""

from __future__ import annotations

import pytest

from imas_codex.discovery.base import llm


def _make_stub_settings(
    monkeypatch, location="iter", proxy_url="http://127.0.0.1:18400"
):
    """Stub get_llm_location and get_llm_proxy_url (late imports in _build_kwargs)."""
    monkeypatch.setattr("imas_codex.settings.get_llm_location", lambda: location)
    monkeypatch.setattr("imas_codex.settings.get_llm_proxy_url", lambda: proxy_url)


MESSAGES = [{"role": "user", "content": "test"}]


class TestResolveModelAlias:
    """Unit tests for resolve_model_alias()."""

    def test_resolves_cc_haiku(self):
        result = llm.resolve_model_alias("cc:haiku")
        assert result == "anthropic/claude-haiku-4.5"

    def test_resolves_cc_sonnet(self):
        result = llm.resolve_model_alias("cc:sonnet")
        assert result == "anthropic/claude-sonnet-4.6"

    def test_resolves_cc_opus(self):
        result = llm.resolve_model_alias("cc:opus")
        assert result == "anthropic/claude-opus-4.6"

    def test_passthrough_regular_model(self):
        assert (
            llm.resolve_model_alias("anthropic/claude-haiku-4.5")
            == "anthropic/claude-haiku-4.5"
        )

    def test_passthrough_openrouter_prefixed(self):
        assert (
            llm.resolve_model_alias("openrouter/anthropic/claude-haiku-4.5")
            == "openrouter/anthropic/claude-haiku-4.5"
        )

    def test_passthrough_google_model(self):
        assert (
            llm.resolve_model_alias("google/gemini-3-flash-preview")
            == "google/gemini-3-flash-preview"
        )

    def test_unknown_cc_alias_passes_through(self):
        """Unknown cc:* aliases pass through with a warning."""
        result = llm.resolve_model_alias("cc:nonexistent")
        assert result == "cc:nonexistent"


class TestProxyAliasMap:
    """Tests for _proxy_alias_map() config loading."""

    def test_loads_all_cc_aliases(self):
        aliases = llm._proxy_alias_map()
        assert "cc:haiku" in aliases
        assert "cc:sonnet" in aliases
        assert "cc:opus" in aliases

    def test_no_non_cc_entries(self):
        """Only cc:* entries should be in the alias map."""
        aliases = llm._proxy_alias_map()
        for key in aliases:
            assert key.startswith("cc:"), f"Unexpected alias: {key}"


class TestBuildKwargsAliasResolution:
    """_build_kwargs resolves cc:* aliases before routing decisions."""

    def test_cc_haiku_gets_openrouter_prefix(self, monkeypatch):
        """cc:haiku resolves to anthropic/claude-haiku-4.5 and gets openrouter/ prefix."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        kwargs = llm._build_kwargs(
            model="cc:haiku",
            api_key="test-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert kwargs["model"] == "openrouter/anthropic/claude-haiku-4.5"

    def test_cc_haiku_gets_cache_control_injected(self, monkeypatch):
        """cc:haiku resolves to a claude model → cache_control injected on system msg."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        messages = [
            {"role": "system", "content": "You are a physics expert."},
            {"role": "user", "content": "Describe psi."},
        ]
        kwargs = llm._build_kwargs(
            model="cc:haiku",
            api_key="test-key",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        sys_msg = kwargs["messages"][0]
        # inject_cache_control converts string content to content blocks
        assert isinstance(sys_msg["content"], list)
        last_block = sys_msg["content"][-1]
        assert "cache_control" in last_block

    def test_cc_haiku_bypasses_proxy_with_direct_key(self, monkeypatch):
        """cc:haiku with OPENROUTER_API_KEY_IMAS_CODEX → bypass proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-key")

        kwargs = llm._build_kwargs(
            model="cc:haiku",
            api_key="test-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # Bypass path: no api_base, direct OpenRouter model name
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/anthropic/claude-haiku-4.5"
        assert kwargs["api_key"] == "direct-key"

    def test_cc_haiku_uses_proxy_without_direct_key(self, monkeypatch):
        """Without direct key, cc:haiku goes through proxy with resolved name."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        kwargs = llm._build_kwargs(
            model="cc:haiku",
            api_key="test-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # Proxy path: has api_base, model wrapped in openai/
        assert kwargs["api_base"] == "http://127.0.0.1:18400"
        # Resolved name gets openrouter/ prefix, then wrapped in openai/ for proxy
        assert kwargs["model"] == "openai/openrouter/anthropic/claude-haiku-4.5"

    def test_cc_haiku_gets_claude_token_limits(self, monkeypatch):
        """After alias resolution, claude family limits should apply."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        kwargs = llm._build_kwargs(
            model="cc:haiku",
            api_key="test-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # Claude family: max_tokens=32000
        assert kwargs["max_tokens"] == 32000


class TestCostExtractionWithAlias:
    """extract_cost works the same regardless of alias usage."""

    def test_extract_cost_from_hidden_params(self):
        """response_cost in _hidden_params takes priority."""

        class FakeResponse:
            _hidden_params = {"response_cost": 0.00123}
            usage = None

        assert llm.extract_cost(FakeResponse()) == 0.00123

    def test_extract_cost_fallback_to_usage(self):
        """Without response_cost, falls back to usage-based estimate."""

        class FakeUsage:
            prompt_tokens = 1000
            completion_tokens = 100

        class FakeResponse:
            _hidden_params = {}
            usage = FakeUsage()

        cost = llm.extract_cost(FakeResponse())
        # (1000 * 3 + 100 * 15) / 1_000_000 = 0.0045
        assert abs(cost - 0.0045) < 1e-8
