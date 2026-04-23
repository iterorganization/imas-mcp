"""Tests for per-axis review model chain config (p39-3).

Covers:
- get_sn_review_names_models() / get_sn_review_docs_models() length validation
- get_sn_review_max_cycles() default
- get_sn_review_disagreement_threshold() default
- Axis independence (names vs docs lists don't cross-contaminate)
- ValueError on empty list or >3 entries
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_sn_review(monkeypatch, review_dict: dict):
    """Monkeypatch _get_section so sn.review returns *review_dict*."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": review_dict}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()


# ---------------------------------------------------------------------------
# names axis — length tests
# ---------------------------------------------------------------------------


def test_names_models_list_length_1_disabled_quorum(monkeypatch):
    """1 model → quorum disabled, returns list of len 1."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(
        monkeypatch, {"names": {"models": ["openrouter/anthropic/claude-opus-4.6"]}}
    )
    models = settings_mod.get_sn_review_names_models()
    assert len(models) == 1
    assert models[0] == "openrouter/anthropic/claude-opus-4.6"


def test_names_models_list_length_2_no_escalator(monkeypatch):
    """2 models → blind pair with no escalator, returns list of len 2."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(
        monkeypatch,
        {
            "names": {
                "models": [
                    "openrouter/anthropic/claude-opus-4.6",
                    "openrouter/openai/gpt-5.4",
                ]
            }
        },
    )
    models = settings_mod.get_sn_review_names_models()
    assert len(models) == 2


def test_names_models_list_length_3_full_quorum(monkeypatch):
    """3 models → full RD-quorum, returns list of len 3."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(
        monkeypatch,
        {
            "names": {
                "models": [
                    "openrouter/anthropic/claude-opus-4.6",
                    "openrouter/openai/gpt-5.4",
                    "openrouter/anthropic/claude-sonnet-4.6",
                ]
            }
        },
    )
    models = settings_mod.get_sn_review_names_models()
    assert len(models) == 3


def test_names_models_rejects_empty_list(monkeypatch):
    """0 models → ValueError."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {"names": {"models": []}})
    with pytest.raises(ValueError, match="at least 1 entry"):
        settings_mod.get_sn_review_names_models()


def test_names_models_rejects_over_three(monkeypatch):
    """4+ models → ValueError."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(
        monkeypatch,
        {"names": {"models": ["a", "b", "c", "d"]}},
    )
    with pytest.raises(ValueError, match="at most 3 entries"):
        settings_mod.get_sn_review_names_models()


# ---------------------------------------------------------------------------
# docs axis — independence test
# ---------------------------------------------------------------------------


def test_docs_models_independent_from_names(monkeypatch):
    """names and docs axes use separate model lists — no cross-contamination."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(
        monkeypatch,
        {
            "names": {"models": ["openrouter/anthropic/claude-opus-4.6"]},
            "docs": {
                "models": [
                    "openrouter/openai/gpt-5.4",
                    "openrouter/anthropic/claude-sonnet-4.6",
                ]
            },
        },
    )
    names_models = settings_mod.get_sn_review_names_models()
    docs_models = settings_mod.get_sn_review_docs_models()

    assert names_models == ["openrouter/anthropic/claude-opus-4.6"]
    assert docs_models == [
        "openrouter/openai/gpt-5.4",
        "openrouter/anthropic/claude-sonnet-4.6",
    ]
    # No overlap in content that would indicate cross-contamination
    assert names_models != docs_models


# ---------------------------------------------------------------------------
# Shared settings — defaults
# ---------------------------------------------------------------------------


def test_max_cycles_default_three(monkeypatch):
    """No max-cycles config → default is 3."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {})
    assert settings_mod.get_sn_review_max_cycles() == 3


def test_max_cycles_override(monkeypatch):
    """max-cycles in config is respected."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {"max-cycles": 2})
    assert settings_mod.get_sn_review_max_cycles() == 2


def test_disagreement_threshold_default(monkeypatch):
    """No config → default threshold is 0.15."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {})
    assert settings_mod.get_sn_review_disagreement_threshold() == pytest.approx(0.15)


def test_disagreement_threshold_override(monkeypatch):
    """disagreement-threshold in config is respected."""
    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {"disagreement-threshold": 0.25})
    assert settings_mod.get_sn_review_disagreement_threshold() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Validation — non-empty string entries
# ---------------------------------------------------------------------------


def test_names_models_rejects_empty_string_entry(monkeypatch):
    """Whitespace-only string entries in models list → ValueError on validation."""
    from imas_codex import settings as settings_mod

    # Whitespace-only strings pass the `if m` filter but fail the strip check
    _patch_sn_review(monkeypatch, {"names": {"models": ["  "]}})
    with pytest.raises(ValueError):
        settings_mod.get_sn_review_names_models()


# ---------------------------------------------------------------------------
# Warning for missing openrouter/ prefix
# ---------------------------------------------------------------------------


def test_names_models_warns_on_missing_openrouter_prefix(monkeypatch, caplog):
    """A model without 'openrouter/' prefix logs a warning but does not raise."""
    import logging

    from imas_codex import settings as settings_mod

    _patch_sn_review(monkeypatch, {"names": {"models": ["anthropic/claude-opus-4.6"]}})
    with caplog.at_level(logging.WARNING):
        models = settings_mod.get_sn_review_names_models()

    assert models == ["anthropic/claude-opus-4.6"]
    assert any("openrouter" in r.message for r in caplog.records)
