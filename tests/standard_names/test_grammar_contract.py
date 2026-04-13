"""Contract tests for ISN grammar API stability."""

from __future__ import annotations


def test_grammar_context_contract():
    """Verify get_grammar_context() returns all keys codex depends on."""
    from imas_standard_names.grammar.context import get_grammar_context

    ctx = get_grammar_context()
    required = {
        "canonical_pattern",
        "segment_order",
        "template_rules",
        "vocabulary_sections",
        "segment_descriptions",
        "naming_guidance",
        "kind_definitions",
        "anti_patterns",
        "exclusive_pairs",
        "field_guidance",
        "tag_descriptions",
        "applicability",
        "quick_start",
        "common_patterns",
        "critical_distinctions",
        "vocabulary_usage_stats",
        "base_requirements",
        "type_specific_requirements",
        "documentation_guidance",
    }
    assert required <= set(ctx.keys()), f"Missing keys: {required - set(ctx.keys())}"


def test_grammar_context_types():
    """Verify key types from get_grammar_context()."""
    from imas_standard_names.grammar.context import get_grammar_context

    ctx = get_grammar_context()
    assert isinstance(ctx["canonical_pattern"], str)
    assert isinstance(ctx["segment_order"], str)
    assert isinstance(ctx["template_rules"], str)
    assert isinstance(ctx["vocabulary_sections"], list)
    assert isinstance(ctx["segment_descriptions"], dict)
    assert isinstance(ctx["exclusive_pairs"], list)


def test_standard_name_entry_import():
    """Verify StandardNameEntry is importable from models."""
    from imas_standard_names.models import StandardNameEntry, create_standard_name_entry

    assert StandardNameEntry is not None
    assert callable(create_standard_name_entry)


def test_build_compose_context_has_isn_keys():
    """Verify compose context includes ISN-provided keys."""
    from imas_codex.standard_names.context import (
        build_compose_context,
        clear_context_cache,
    )

    clear_context_cache()
    ctx = build_compose_context()
    # Core grammar keys
    assert "canonical_pattern" in ctx
    assert "segment_order" in ctx
    assert "vocabulary_sections" in ctx
    assert "segment_descriptions" in ctx
    # New ISN-provided keys
    assert "naming_guidance" in ctx
    assert "kind_definitions" in ctx
    assert "anti_patterns" in ctx
    # Additional ISN keys
    assert "quick_start" in ctx
    assert "common_patterns" in ctx
    assert "critical_distinctions" in ctx
    assert "vocabulary_usage_stats" in ctx
    assert "base_requirements" in ctx
    assert "type_specific_requirements" in ctx
    assert "documentation_guidance" in ctx
    # Codex-specific keys still present
    assert "examples" in ctx
    assert "tokamak_ranges" in ctx
    assert "field_guidance" in ctx
    assert "tag_descriptions" in ctx
    assert "applicability" in ctx
    assert "exclusive_pairs" in ctx
    clear_context_cache()
