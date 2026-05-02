"""Tests for the missing-openrouter-prefix warning in llm._build_kwargs.

When a caller passes an unprefixed model id (e.g. ``anthropic/claude-sonnet-4.6``)
and the direct-path key ``OPENROUTER_API_KEY_IMAS_CODEX`` is set,
``call_llm_structured`` must emit a runtime warning. This catches future
regressions where pyproject sections silently lose the ``openrouter/`` prefix
and start routing through the LiteLLM proxy (which strips cache_control and
zeroes response_cost).
"""

from __future__ import annotations

import logging

import pytest

from imas_codex.discovery.base import llm

MESSAGES = [{"role": "user", "content": "x"}]


def _stub_settings(monkeypatch, location="local", proxy_url=""):
    monkeypatch.setattr("imas_codex.settings.get_llm_location", lambda: location)
    monkeypatch.setattr("imas_codex.settings.get_llm_proxy_url", lambda: proxy_url)


class TestPrefixWarning:
    def setup_method(self):
        # Clear the dedup cache so each test sees a fresh warning state.
        llm._PREFIX_WARNED.clear()

    def test_warns_when_unprefixed_and_direct_key_set(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-or-test")
        _stub_settings(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="imas_codex.discovery.base.llm"):
            llm._build_kwargs(
                "anthropic/claude-sonnet-4.6",
                "sk-or-test",
                MESSAGES,
                None,
                None,
                None,
                None,
            )

        assert any(
            "missing openrouter/ prefix" in rec.message for rec in caplog.records
        ), f"expected warning, got: {[r.message for r in caplog.records]}"

    def test_no_warn_when_prefix_present(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-or-test")
        _stub_settings(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="imas_codex.discovery.base.llm"):
            llm._build_kwargs(
                "openrouter/anthropic/claude-sonnet-4.6",
                "sk-or-test",
                MESSAGES,
                None,
                None,
                None,
                None,
            )

        assert not any(
            "missing openrouter/ prefix" in rec.message for rec in caplog.records
        )

    def test_no_warn_without_direct_key(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
        _stub_settings(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="imas_codex.discovery.base.llm"):
            llm._build_kwargs(
                "anthropic/claude-sonnet-4.6",
                "sk-or-test",
                MESSAGES,
                None,
                None,
                None,
                None,
            )

        assert not any(
            "missing openrouter/ prefix" in rec.message for rec in caplog.records
        )

    def test_no_warn_for_local_models(self, monkeypatch, caplog):
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-or-test")
        _stub_settings(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="imas_codex.discovery.base.llm"):
            llm._build_kwargs(
                "ollama/llama3",
                "sk-or-test",
                MESSAGES,
                None,
                None,
                None,
                None,
            )

        assert not any(
            "missing openrouter/ prefix" in rec.message for rec in caplog.records
        )

    def test_warning_dedup(self, monkeypatch, caplog):
        """Warning fires once per unique unprefixed model id."""
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-or-test")
        _stub_settings(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="imas_codex.discovery.base.llm"):
            for _ in range(3):
                llm._build_kwargs(
                    "anthropic/claude-sonnet-4.6",
                    "sk-or-test",
                    MESSAGES,
                    None,
                    None,
                    None,
                    None,
                )

        warnings = [
            r for r in caplog.records if "missing openrouter/ prefix" in r.message
        ]
        assert len(warnings) == 1
