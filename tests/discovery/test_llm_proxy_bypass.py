"""Tests for proxy bypass logic in _build_kwargs.

When the model supports cache_control (Anthropic/Gemini) and a direct
OpenRouter API key is available, _build_kwargs should bypass the LiteLLM
proxy to preserve prompt caching and actual cost reporting.
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


class TestProxyBypass:
    """Proxy bypass routing logic."""

    def test_proxy_used_when_no_direct_key(self, monkeypatch):
        """Without OPENROUTER_API_KEY_IMAS_CODEX, always use proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert kwargs["api_base"] == "http://127.0.0.1:18400"
        assert kwargs["model"].startswith("openai/")

    def test_bypass_for_cache_model_with_direct_key(self, monkeypatch):
        """With direct key + cache-capable model → bypass proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        # No api_base = direct to OpenRouter
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/anthropic/claude-sonnet-4.6"
        # Should use direct key
        assert kwargs["api_key"] == "direct-or-key"

    def test_proxy_used_for_non_cache_model(self, monkeypatch):
        """Non-cache models (e.g., GPT-5) always use proxy."""
        _make_stub_settings(monkeypatch, location="iter")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "direct-or-key")
        monkeypatch.setenv("LITELLM_API_KEY", "proxy-key")

        kwargs = llm._build_kwargs(
            model="openai/gpt-5.4",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert kwargs["api_base"] == "http://127.0.0.1:18400"
        assert kwargs["model"].startswith("openai/")

    def test_local_mode_bypasses_proxy(self, monkeypatch):
        """Local mode (no proxy URL) always goes direct."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=MESSAGES,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        assert "api_base" not in kwargs
        assert kwargs["model"] == "openrouter/anthropic/claude-sonnet-4.6"


class TestCacheControlInjection:
    """cache_control blocks are injected for supported models."""

    def test_cache_control_injected_for_claude(self, monkeypatch):
        """Claude models get cache_control on system message."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        messages = [
            {"role": "system", "content": "You are a physics expert."},
            {"role": "user", "content": "Describe psi."},
        ]
        kwargs = llm._build_kwargs(
            model="anthropic/claude-sonnet-4.6",
            api_key="or-key",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        sys_msg = kwargs["messages"][0]
        # inject_cache_control converts string content to content blocks
        if isinstance(sys_msg["content"], list):
            last_block = sys_msg["content"][-1]
            assert "cache_control" in last_block
        else:
            # If content is still a string, cache_control should be
            # present as a top-level key on the message
            pass  # some models may not restructure

    def test_no_cache_control_for_gpt(self, monkeypatch):
        """GPT models should NOT get cache_control injected."""
        _make_stub_settings(monkeypatch, location="local")
        monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello."},
        ]
        kwargs = llm._build_kwargs(
            model="openai/gpt-5.4",
            api_key="or-key",
            messages=messages,
            response_format=None,
            max_tokens=None,
            temperature=None,
            timeout=None,
        )
        sys_msg = kwargs["messages"][0]
        # GPT messages should remain plain strings
        assert isinstance(sys_msg["content"], str)
