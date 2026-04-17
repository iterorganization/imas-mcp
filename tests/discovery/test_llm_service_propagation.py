"""Tests for service= propagation through public LLM API functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.base.llm import LLM_SERVICE


class TestServiceIsKeywordOnly:
    """service= must be keyword-only — positional call must raise TypeError."""

    def test_call_llm_structured_rejects_positional_service(self):
        import inspect

        from imas_codex.discovery.base.llm import call_llm_structured

        sig = inspect.signature(call_llm_structured)
        # service should be keyword-only
        param = sig.parameters.get("service")
        assert param is not None, "service parameter not found"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_acall_llm_structured_rejects_positional_service(self):
        import inspect

        from imas_codex.discovery.base.llm import acall_llm_structured

        sig = inspect.signature(acall_llm_structured)
        param = sig.parameters.get("service")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_call_llm_rejects_positional_service(self):
        import inspect

        from imas_codex.discovery.base.llm import call_llm

        sig = inspect.signature(call_llm)
        param = sig.parameters.get("service")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_acall_llm_rejects_positional_service(self):
        import inspect

        from imas_codex.discovery.base.llm import acall_llm

        sig = inspect.signature(acall_llm)
        param = sig.parameters.get("service")
        assert param is not None
        assert param.kind == inspect.Parameter.KEYWORD_ONLY


class TestServicePropagation:
    """Verify service= is forwarded to _build_kwargs in all public functions."""

    @patch("imas_codex.discovery.base.llm._build_kwargs")
    @patch("imas_codex.discovery.base.llm.get_api_key", return_value="test-key")
    def test_call_llm_structured_forwards_service(
        self, mock_key, mock_build, monkeypatch
    ):
        from pydantic import BaseModel

        from imas_codex.discovery.base.llm import call_llm_structured

        class Dummy(BaseModel):
            x: int

        # Mock _build_kwargs to return valid kwargs
        mock_build.return_value = {
            "model": "test",
            "messages": [],
            "api_key": "k",
            "max_tokens": 100,
            "timeout": 30,
        }
        # Mock litellm.completion
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"x": 1}'
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response._hidden_params = {}

        with patch("litellm.completion", return_value=mock_response):
            try:
                call_llm_structured(
                    model="test",
                    messages=[],
                    response_model=Dummy,
                    service="data-dictionary",
                )
            except Exception:
                pass  # May fail on parsing, that's OK

        # Verify service= was passed through to _build_kwargs
        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs.get("service") == "data-dictionary"


class TestPerServiceApiKey:
    """Test get_api_key_for_service per-service key resolution."""

    def test_falls_back_to_imas_codex_key(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.delenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", raising=False)
        key = get_api_key_for_service("facility-discovery")
        assert key == "sk-default"

    def test_uses_per_service_key_when_set(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.setenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", "sk-discovery")
        key = get_api_key_for_service("facility-discovery")
        assert key == "sk-discovery"

    def test_untagged_skips_per_service_lookup(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.setenv("OPENROUTER_API_KEY_UNTAGGED", "sk-should-not-use")
        key = get_api_key_for_service("untagged")
        assert key == "sk-default"

    def test_hyphenated_service_maps_to_underscore_env(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.setenv("OPENROUTER_API_KEY_STANDARD_NAMES", "sk-sn")
        key = get_api_key_for_service("standard-names")
        assert key == "sk-sn"

    def test_data_dictionary_key(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.setenv("OPENROUTER_API_KEY_DATA_DICTIONARY", "sk-dd")
        key = get_api_key_for_service("data-dictionary")
        assert key == "sk-dd"

    def test_imas_mapping_key(self, monkeypatch):
        from imas_codex.discovery.base.llm import get_api_key_for_service

        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "sk-default")
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_MAPPING", "sk-mapping")
        key = get_api_key_for_service("imas-mapping")
        assert key == "sk-mapping"
