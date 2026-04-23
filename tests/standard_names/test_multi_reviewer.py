"""Tests for the N-reviewer SN review pipeline.

Covers:
- Settings accessor get_sn_review_names_models() / get_sn_review_docs_models()
  read [sn.review.names/docs].models correctly
- StandardNameReviewState has new fields (review_models, review_records,
  canonical_review_model) and backward-compat secondary_models alias
- Pipeline helpers (_model_slug, _derive_model_family, _build_review_record)
  produce correct output without requiring a live Neo4j connection
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

# =============================================================================
# Settings tests
# =============================================================================


def test_get_sn_review_names_models_default():
    """get_sn_review_names_models() returns at least one string model."""
    from imas_codex.settings import get_sn_review_names_models

    models = get_sn_review_names_models()
    assert isinstance(models, list)
    assert 1 <= len(models) <= 3
    for m in models:
        assert isinstance(m, str)
        assert m  # non-empty


def test_get_sn_review_names_models_override(monkeypatch):
    """get_sn_review_names_models() reads from [sn.review.names].models override."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": {"names": {"models": ["a", "b", "c"]}}}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()
    models = settings_mod.get_sn_review_names_models()
    assert models == ["a", "b", "c"]


def test_get_sn_review_docs_models_override(monkeypatch):
    """get_sn_review_docs_models() reads from [sn.review.docs].models override."""
    from imas_codex import settings as settings_mod

    def fake_get_section(name: str) -> dict:
        if name == "sn":
            return {"review": {"docs": {"models": ["x", "y"]}}}
        return {}

    monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
    settings_mod._load_pyproject_settings.cache_clear()
    models = settings_mod.get_sn_review_docs_models()
    assert models == ["x", "y"]


# =============================================================================
# State tests
# =============================================================================


def test_state_has_review_models_default():
    """StandardNameReviewState initializes new fields with correct defaults."""
    from imas_codex.standard_names.review.state import StandardNameReviewState

    state = StandardNameReviewState(facility="dd")
    assert state.review_models == []
    assert state.canonical_review_model is None
    assert state.review_records == []
    # Backward-compat alias still present
    assert state.secondary_models == []


# =============================================================================
# Pipeline helper tests
# =============================================================================


def test_build_review_record_canonical_and_model_family():
    """_build_review_record returns a correct record for a canonical review."""
    from imas_codex.standard_names.review.pipeline import _build_review_record

    reviewed_at = datetime.now(UTC).isoformat()
    item = {
        "id": "electron_temperature",
        "reviewer_score": 0.85,
        "reviewer_scores": {"grammar": 0.9, "semantic": 0.8},
        "review_tier": "good",
        "reviewer_comments": "Well structured",
    }
    record = _build_review_record(
        item,
        model="openrouter/anthropic/claude-opus-4.6",
        is_canonical=True,
        reviewed_at=reviewed_at,
    )

    assert record["model_family"] == "anthropic"
    assert record["is_canonical"] is True
    assert record["id"].startswith("electron_temperature:")
    assert isinstance(record["scores_json"], str)
    parsed = json.loads(record["scores_json"])
    assert isinstance(parsed, dict)
    assert record["score"] == pytest.approx(0.85)
    assert record["tier"] == "good"
    assert record["standard_name_id"] == "electron_temperature"


@pytest.mark.parametrize(
    ("model_id", "expected_family"),
    [
        ("openai/gpt-5.4", "openai"),
        ("anthropic/claude-sonnet-4.6", "anthropic"),
        ("google/gemini-3-flash", "google"),
        ("mistralai/mixtral", "mistral"),
        ("meta-llama/llama-3", "meta"),
        ("x-ai/grok-2", "xai"),
        ("cohere/command-r", "cohere"),
        ("random/something", "other"),
    ],
)
def test_derive_model_family_mappings(model_id: str, expected_family: str):
    """_derive_model_family maps model ids to correct family labels."""
    from imas_codex.standard_names.review.pipeline import _derive_model_family

    assert _derive_model_family(model_id) == expected_family


def test_model_slug_normalizes():
    """_model_slug converts model ids to safe slugs."""
    from imas_codex.standard_names.review.pipeline import _model_slug

    slug = _model_slug("openrouter/anthropic/claude-opus-4.6")
    assert slug == "openrouter-anthropic-claude-opus-4-6"

    empty_slug = _model_slug("")
    assert empty_slug == "unknown"
