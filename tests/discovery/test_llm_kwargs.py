"""Tests for _build_kwargs JSON schema conversion and _sanitize_content in llm.py."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from imas_codex.discovery.base.llm import (
    _extract_balanced_json,
    _find_json_start,
    _is_pydantic_model,
    _sanitize_content,
    _strip_unsupported_schema_props,
    _to_json_schema_format,
)


class DummyResponse(BaseModel):
    """Simple model for testing."""

    name: str
    score: float


class NestedResponse(BaseModel):
    """Model with nested types for testing."""

    items: list[str]
    metadata: dict[str, str]


def test_is_pydantic_model_class():
    assert _is_pydantic_model(DummyResponse) is True


def test_is_pydantic_model_rejects_instance():
    assert _is_pydantic_model(DummyResponse(name="x", score=1.0)) is False


def test_is_pydantic_model_rejects_dict():
    assert _is_pydantic_model({"type": "json_object"}) is False


def test_is_pydantic_model_rejects_none():
    assert _is_pydantic_model(None) is False


def test_to_json_schema_format_structure():
    result = _to_json_schema_format(DummyResponse)
    assert result["type"] == "json_schema"
    assert result["json_schema"]["name"] == "DummyResponse"
    assert result["json_schema"]["strict"] is False
    schema = result["json_schema"]["schema"]
    assert "properties" in schema
    assert "name" in schema["properties"]
    assert "score" in schema["properties"]


def test_to_json_schema_format_nested():
    result = _to_json_schema_format(NestedResponse)
    schema = result["json_schema"]["schema"]
    assert "items" in schema["properties"]
    assert "metadata" in schema["properties"]


class ModelWithConstraints(BaseModel):
    """Model with array/string constraints that some providers reject."""

    keywords: list[str] = Field(default_factory=list, max_length=8)
    name: str = Field(min_length=1, max_length=100)


def test_strip_unsupported_schema_props_removes_maxItems():
    """maxItems from max_length on list fields must be stripped."""
    schema = ModelWithConstraints.model_json_schema()
    cleaned = _strip_unsupported_schema_props(schema)
    kw_prop = cleaned["properties"]["keywords"]
    assert "maxItems" not in kw_prop


def test_to_json_schema_format_strips_constraints():
    """_to_json_schema_format should produce a provider-safe schema."""
    result = _to_json_schema_format(ModelWithConstraints)
    schema = result["json_schema"]["schema"]
    kw_prop = schema["properties"]["keywords"]
    assert "maxItems" not in kw_prop
    # Pydantic validation still enforces it — just not in the API schema


@pytest.mark.parametrize(
    "model_name",
    [
        "anthropic/claude-sonnet-4.6",
        "openrouter/anthropic/claude-sonnet-4.6",
        "google/gemini-3-flash-preview",
        "openai/gpt-5.4",
        "gpt-5.2-codex",
    ],
)
def test_build_kwargs_wraps_pydantic_for_all_models(model_name, monkeypatch):
    """All models get Pydantic→json_schema conversion, not just GPT-5."""
    from imas_codex.discovery.base import llm

    # Stub out settings/proxy lookups
    monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
    monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)

    kwargs = llm._build_kwargs(
        model=model_name,
        api_key="test-key",
        messages=[{"role": "user", "content": "test"}],
        response_format=DummyResponse,
        max_tokens=None,
        temperature=None,
        timeout=None,
    )
    rf = kwargs["response_format"]
    assert isinstance(rf, dict), f"Expected dict, got {type(rf)} for model {model_name}"
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "DummyResponse"
    assert rf["json_schema"]["strict"] is False


def test_build_kwargs_passes_dict_response_format_unchanged(monkeypatch):
    """Non-Pydantic response_format (e.g., raw dict) is passed through."""
    from imas_codex.discovery.base import llm

    monkeypatch.setattr(llm, "get_llm_location", lambda: "local", raising=False)
    monkeypatch.setattr(llm, "get_llm_proxy_url", lambda: None, raising=False)

    raw_format = {"type": "json_object"}
    kwargs = llm._build_kwargs(
        model="anthropic/claude-sonnet-4.6",
        api_key="test-key",
        messages=[{"role": "user", "content": "test"}],
        response_format=raw_format,
        max_tokens=None,
        temperature=None,
        timeout=None,
    )
    assert kwargs["response_format"] is raw_format


# ---------------------------------------------------------------------------
# _sanitize_content tests — prose extraction, code fences, edge cases
# ---------------------------------------------------------------------------


class TestSanitizeContentBasic:
    """Basic sanitization: code fences, control chars, surrogates."""

    def test_strips_json_code_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_strips_bare_code_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_removes_control_chars(self):
        raw = '{"a":\x00 1}'
        assert _sanitize_content(raw) == '{"a": 1}'

    def test_passthrough_clean_json(self):
        raw = '{"candidates": [], "vocab_gaps": []}'
        assert _sanitize_content(raw) == raw


class TestSanitizeContentProseExtraction:
    """Extract JSON from LLM prose wrappers."""

    def test_extracts_json_after_thinking_text(self):
        raw = 'Looking at these paths, I need to analyze them.\n\n{"candidates": [], "vocab_gaps": []}'
        result = _sanitize_content(raw)
        assert result == '{"candidates": [], "vocab_gaps": []}'

    def test_extracts_json_with_multiline_preamble(self):
        raw = (
            "I'll analyze the DD paths for standard names.\n"
            "First, let me consider the physics.\n"
            "Here are the results:\n"
            '{"candidates": [{"name": "electron_temperature"}], "vocab_gaps": []}'
        )
        result = _sanitize_content(raw)
        assert '"candidates"' in result
        assert result.startswith("{")

    def test_extracts_json_with_trailing_text(self):
        raw = 'Here is the output:\n{"a": 1, "b": 2}\nI hope this helps!'
        result = _sanitize_content(raw)
        assert result == '{"a": 1, "b": 2}'

    def test_extracts_nested_json(self):
        raw = 'Analysis:\n{"outer": {"inner": [1, 2, 3]}, "x": "y"}'
        result = _sanitize_content(raw)
        assert result == '{"outer": {"inner": [1, 2, 3]}, "x": "y"}'

    def test_handles_json_with_strings_containing_braces(self):
        raw = 'Result:\n{"desc": "function f(x) { return x; }", "n": 1}'
        result = _sanitize_content(raw)
        assert result == '{"desc": "function f(x) { return x; }", "n": 1}'

    def test_extracts_array_from_prose(self):
        raw = "Here are the items:\n[1, 2, 3]"
        result = _sanitize_content(raw)
        assert result == "[1, 2, 3]"

    def test_real_world_compose_error_pattern(self):
        """Reproduce the actual error pattern from SN compose failures."""
        raw = (
            "Looking at these two paths from the mhd IDS, I need to generate "
            "standard names following the grammar rules.\n\n"
            "{\n"
            '  "candidates": [\n'
            "    {\n"
            '      "name": "mhd_frequency_linear_toroidal_mode_number",\n'
            '      "description": "Toroidal mode number for MHD instability"\n'
            "    }\n"
            "  ],\n"
            '  "attachments": [],\n'
            '  "skipped": [],\n'
            '  "vocab_gaps": []\n'
            "}"
        )
        result = _sanitize_content(raw)
        assert result.startswith("{")
        assert result.endswith("}")
        import json

        parsed = json.loads(result)
        assert "candidates" in parsed


class TestFindJsonStart:
    """Unit tests for _find_json_start."""

    def test_finds_object_start(self):
        assert _find_json_start('hello\n{"a": 1}') >= 0

    def test_finds_array_start(self):
        assert _find_json_start("text [1,2]") == 5

    def test_returns_negative_for_no_json(self):
        assert _find_json_start("just plain text") == -1

    def test_skips_to_first_brace(self):
        text = 'preamble {"key": "value"}'
        idx = _find_json_start(text)
        assert text[idx] == "{"


class TestExtractBalancedJson:
    """Unit tests for _extract_balanced_json."""

    def test_simple_object(self):
        text = '{"a": 1} trailing'
        assert _extract_balanced_json(text, 0) == '{"a": 1}'

    def test_nested_object(self):
        text = '{"a": {"b": 2}} end'
        assert _extract_balanced_json(text, 0) == '{"a": {"b": 2}}'

    def test_with_string_containing_braces(self):
        text = '{"code": "if (x) { y; }"} extra'
        assert _extract_balanced_json(text, 0) == '{"code": "if (x) { y; }"}'

    def test_escaped_quotes_in_string(self):
        text = r'{"a": "say \"hello\""} more'
        assert _extract_balanced_json(text, 0) == r'{"a": "say \"hello\""}'

    def test_array(self):
        text = "[1, [2, 3], 4] rest"
        assert _extract_balanced_json(text, 0) == "[1, [2, 3], 4]"

    def test_no_close_returns_to_end(self):
        text = '{"unbalanced'
        assert _extract_balanced_json(text, 0) == '{"unbalanced'
