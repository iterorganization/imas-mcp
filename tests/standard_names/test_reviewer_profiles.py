"""Tests for reviewer profile selection (Phase D, plan 43).

Covers:
- get_sn_review_active_profile() env-var resolution
- get_sn_review_profile_models() for all four named profiles
- get_sn_review_profile_threshold() for all four named profiles
- Unknown profile raises ValueError
- Legacy accessors (get_sn_review_names_models, get_sn_review_disagreement_threshold)
  remain backward-compatible after refactor
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_profiles(monkeypatch, profiles_dict: dict) -> None:
    """Monkeypatch _get_section so sn.review.names.profiles = *profiles_dict*."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": {"names": {"profiles": profiles_dict}}}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()


def _patch_toplevel(monkeypatch, names_section: dict) -> None:
    """Monkeypatch _get_section without a profiles sub-table (legacy config)."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": {"names": names_section}}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()


# ---------------------------------------------------------------------------
# get_sn_review_active_profile
# ---------------------------------------------------------------------------


def test_active_profile_default_when_no_env_var(monkeypatch):
    """No env var → active profile is 'default'."""
    from imas_codex import settings as settings_mod

    monkeypatch.delenv("IMAS_CODEX_SN_REVIEW_PROFILE", raising=False)
    assert settings_mod.get_sn_review_active_profile() == "default"


def test_env_var_overrides_default_profile(monkeypatch):
    """IMAS_CODEX_SN_REVIEW_PROFILE env var is honoured."""
    from imas_codex import settings as settings_mod

    monkeypatch.setenv("IMAS_CODEX_SN_REVIEW_PROFILE", "quality-cost-balanced")
    assert settings_mod.get_sn_review_active_profile() == "quality-cost-balanced"


def test_env_var_opus_only(monkeypatch):
    """Env var set to 'opus-only' is returned verbatim."""
    from imas_codex import settings as settings_mod

    monkeypatch.setenv("IMAS_CODEX_SN_REVIEW_PROFILE", "opus-only")
    assert settings_mod.get_sn_review_active_profile() == "opus-only"


# ---------------------------------------------------------------------------
# get_sn_review_profile_models
# ---------------------------------------------------------------------------

_DEFAULT_MODELS = [
    "openrouter/anthropic/claude-opus-4.6",
    "openrouter/openai/gpt-5.4",
    "openrouter/anthropic/claude-sonnet-4.6",
]
_PILOT_MODELS = [
    "openrouter/anthropic/claude-sonnet-4.6",
    "openrouter/openai/gpt-5.4",
    "openrouter/anthropic/claude-opus-4.6",
]


def test_default_profile_loads_three_models(monkeypatch):
    """'default' profile returns 3 models when profiles section is present."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "default": {
                "models": _DEFAULT_MODELS,
                "disagreement-threshold": 0.20,
            }
        },
    )
    models = settings_mod.get_sn_review_profile_models("default")
    assert len(models) == 3
    assert models == _DEFAULT_MODELS


def test_pilot_profile_loads_haiku_primary(monkeypatch):
    """'quality-cost-balanced' profile is sonnet-primary + opus arbiter (no Haiku)."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "quality-cost-balanced": {
                "models": _PILOT_MODELS,
                "disagreement-threshold": 0.20,
            }
        },
    )
    models = settings_mod.get_sn_review_profile_models("quality-cost-balanced")
    assert len(models) == 3
    assert "sonnet" in models[0].lower()
    assert "opus" in models[2].lower()
    # HARD CONSTRAINT: no Haiku ever in reviewer chain
    assert not any("haiku" in m.lower() for m in models)


def test_opus_only_loads_one_model(monkeypatch):
    """'opus-only' profile returns exactly one model containing 'opus'."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "opus-only": {
                "models": ["openrouter/anthropic/claude-opus-4.6"],
                "disagreement-threshold": 1.0,
            }
        },
    )
    models = settings_mod.get_sn_review_profile_models("opus-only")
    assert len(models) == 1
    assert "opus" in models[0].lower()


def test_haiku_only_loads_one_model(monkeypatch):
    """Removed: haiku-only profile dropped (Sonnet 4.6 reviewer floor).

    Asserting it is now an unknown profile preserves the constraint.
    """
    from imas_codex import settings as settings_mod

    _patch_profiles(monkeypatch, {})
    with pytest.raises(ValueError, match="Unknown reviewer profile"):
        settings_mod.get_sn_review_profile_models("haiku-only")


def test_unknown_profile_raises_value_error(monkeypatch):
    """Requesting an unknown profile name raises ValueError."""
    from imas_codex import settings as settings_mod

    _patch_profiles(monkeypatch, {})
    with pytest.raises(ValueError, match="Unknown reviewer profile"):
        settings_mod.get_sn_review_profile_models("nonexistent-profile")


# ---------------------------------------------------------------------------
# get_sn_review_profile_threshold
# ---------------------------------------------------------------------------


def test_profile_threshold_default(monkeypatch):
    """Default profile threshold is 0.20."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {"default": {"models": _DEFAULT_MODELS, "disagreement-threshold": 0.20}},
    )
    assert settings_mod.get_sn_review_profile_threshold("default") == pytest.approx(
        0.20
    )


def test_profile_threshold_pilot(monkeypatch):
    """quality-cost-balanced profile threshold is 0.20."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "quality-cost-balanced": {
                "models": _PILOT_MODELS,
                "disagreement-threshold": 0.20,
            }
        },
    )
    assert settings_mod.get_sn_review_profile_threshold(
        "quality-cost-balanced"
    ) == pytest.approx(0.20)


def test_profile_threshold_opus_only(monkeypatch):
    """opus-only threshold is 1.0 (never escalate)."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "opus-only": {
                "models": ["openrouter/anthropic/claude-opus-4.6"],
                "disagreement-threshold": 1.0,
            }
        },
    )
    assert settings_mod.get_sn_review_profile_threshold("opus-only") == pytest.approx(
        1.0
    )


def test_profile_threshold_unknown_raises(monkeypatch):
    """Unknown profile → ValueError for threshold too."""
    from imas_codex import settings as settings_mod

    _patch_profiles(monkeypatch, {})
    with pytest.raises(ValueError, match="Unknown reviewer profile"):
        settings_mod.get_sn_review_profile_threshold("bogus")


# ---------------------------------------------------------------------------
# Legacy accessor backward-compat
# ---------------------------------------------------------------------------


def test_legacy_accessors_still_work_no_profiles_section(monkeypatch):
    """Legacy get_sn_review_names_models() works when no profiles key is present."""
    from imas_codex import settings as settings_mod

    # Old-style config: only top-level models, no profiles
    _patch_toplevel(monkeypatch, {"models": ["openrouter/anthropic/claude-opus-4.6"]})
    monkeypatch.delenv("IMAS_CODEX_SN_REVIEW_PROFILE", raising=False)

    models = settings_mod.get_sn_review_names_models()
    assert models == ["openrouter/anthropic/claude-opus-4.6"]


def test_legacy_threshold_still_works_no_profiles_section(monkeypatch):
    """Legacy get_sn_review_disagreement_threshold() works without profiles section."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": {"disagreement-threshold": 0.25}}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()
    monkeypatch.delenv("IMAS_CODEX_SN_REVIEW_PROFILE", raising=False)

    assert settings_mod.get_sn_review_disagreement_threshold() == pytest.approx(0.25)


def test_legacy_accessor_respects_active_env_var_profile(monkeypatch):
    """get_sn_review_names_models() uses active profile when env var is set."""
    from imas_codex import settings as settings_mod

    _patch_profiles(
        monkeypatch,
        {
            "quality-cost-balanced": {
                "models": _PILOT_MODELS,
                "disagreement-threshold": 0.20,
            }
        },
    )
    monkeypatch.setenv("IMAS_CODEX_SN_REVIEW_PROFILE", "quality-cost-balanced")

    models = settings_mod.get_sn_review_names_models()
    assert "sonnet" in models[0].lower()
