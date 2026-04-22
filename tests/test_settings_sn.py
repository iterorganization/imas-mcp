"""Tests for SN-related settings accessors in imas_codex.settings.

Verifies default values from pyproject.toml and environment variable overrides
for the five example-injection and retry tunables added in Wave 2.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Clear cached pyproject.toml settings between tests."""
    import imas_codex.settings as mod

    mod._load_pyproject_settings.cache_clear()
    yield
    mod._load_pyproject_settings.cache_clear()


# ── Default values (from pyproject.toml) ────────────────────────────────────


def test_example_target_scores_default():
    from imas_codex.settings import get_sn_example_target_scores

    result = get_sn_example_target_scores()
    assert isinstance(result, tuple)
    assert result == (1.0, 0.8, 0.65, 0.4)


def test_example_tolerance_default():
    from imas_codex.settings import get_sn_example_tolerance

    assert get_sn_example_tolerance() == pytest.approx(0.05)


def test_example_per_bucket_default():
    from imas_codex.settings import get_sn_example_per_bucket

    assert get_sn_example_per_bucket() == 1


def test_retry_attempts_default():
    from imas_codex.settings import get_sn_retry_attempts

    assert get_sn_retry_attempts() == 1


def test_retry_k_expansion_default():
    from imas_codex.settings import get_sn_retry_k_expansion

    assert get_sn_retry_k_expansion() == 12


# ── Environment variable overrides ──────────────────────────────────────────


def test_example_target_scores_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_TARGET_SCORES", "0.9,0.7,0.5")
    import imas_codex.settings as mod

    importlib.reload(mod)
    result = mod.get_sn_example_target_scores()
    assert result == (0.9, 0.7, 0.5)


def test_example_tolerance_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_TOLERANCE", "0.1")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_example_tolerance() == pytest.approx(0.1)


def test_example_per_bucket_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_EXAMPLE_PER_BUCKET", "3")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_example_per_bucket() == 3


def test_retry_attempts_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_RETRY_ATTEMPTS", "5")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_retry_attempts() == 5


def test_retry_k_expansion_env(monkeypatch):
    monkeypatch.setenv("IMAS_CODEX_SN_RETRY_K_EXPANSION", "20")
    import imas_codex.settings as mod

    importlib.reload(mod)
    assert mod.get_sn_retry_k_expansion() == 20
