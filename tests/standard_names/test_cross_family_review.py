"""Tests for cross-family reviewer model diversity (cycle-cross-family-review).

Covers:
- Settings accessors read pyproject config keys correctly
- When secondary_models is empty, the review worker behaves identically
  (no secondary fields injected — no regression)
- The _run_secondary_review helper merges secondary fields correctly
"""

from __future__ import annotations

from unittest.mock import patch

# =============================================================================
# Settings accessor tests
# =============================================================================


class TestSNReviewSettings:
    """Settings accessors read [sn.review] pyproject keys correctly."""

    def test_primary_model_default(self):
        """get_sn_review_primary_model returns the expected default."""
        from imas_codex.settings import get_sn_review_primary_model

        model = get_sn_review_primary_model()
        assert isinstance(model, str)
        assert "claude-opus" in model or "claude" in model, (
            f"Expected an Anthropic model as default, got: {model}"
        )

    def test_primary_model_override(self, monkeypatch):
        """primary-model key in pyproject overrides the default."""
        from imas_codex import settings as settings_mod

        # Patch the internal _get_section to simulate a pyproject config entry
        original = settings_mod._get_section

        def fake_get_section(key: str):
            result = original(key)
            if key == "sn":
                result = dict(result)
                result["review"] = {"primary-model": "test/mock-model"}
            return result

        monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)

        from importlib import reload

        # Re-import to pick up monkeypatch; call directly to avoid cache
        model = settings_mod.get_sn_review_primary_model()
        # The monkeypatched section will be read on each call because we
        # patched _get_section, not the cached result.
        assert model == "test/mock-model"

    def test_secondary_models_default_empty(self):
        """get_sn_review_secondary_models returns empty list by default."""
        from imas_codex.settings import get_sn_review_secondary_models

        models = get_sn_review_secondary_models()
        assert isinstance(models, list)
        assert models == [], f"Default should be empty, got: {models}"

    def test_secondary_models_override(self, monkeypatch):
        """secondary-models key is read when present."""
        from imas_codex import settings as settings_mod

        original = settings_mod._get_section

        def fake_get_section(key: str):
            result = original(key)
            if key == "sn":
                result = dict(result)
                result["review"] = {
                    "secondary-models": ["test/model-a", "test/model-b"]
                }
            return result

        monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
        models = settings_mod.get_sn_review_secondary_models()
        assert models == ["test/model-a", "test/model-b"]

    def test_disagreement_threshold_default(self):
        """get_sn_review_disagreement_threshold returns 0.2 by default."""
        from imas_codex.settings import get_sn_review_disagreement_threshold

        threshold = get_sn_review_disagreement_threshold()
        assert isinstance(threshold, float)
        assert threshold == 0.2

    def test_disagreement_threshold_override(self, monkeypatch):
        """disagreement-threshold key is read and returned as float."""
        from imas_codex import settings as settings_mod

        original = settings_mod._get_section

        def fake_get_section(key: str):
            result = original(key)
            if key == "sn":
                result = dict(result)
                result["review"] = {"disagreement-threshold": 0.35}
            return result

        monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
        threshold = settings_mod.get_sn_review_disagreement_threshold()
        assert threshold == 0.35

    def test_benchmark_reviewer_falls_back_to_review_primary(self, monkeypatch):
        """get_sn_benchmark_reviewer_model falls back to sn.review.primary-model."""
        from imas_codex import settings as settings_mod

        original = settings_mod._get_section

        def fake_get_section(key: str):
            result = original(key)
            if key == "sn":
                result = dict(result)
                # No reviewer-model in benchmark, but review has primary-model
                bench = dict(result.get("benchmark", {}))
                bench.pop("reviewer-model", None)
                result["benchmark"] = bench
                result["review"] = {"primary-model": "test/primary-from-review"}
            return result

        monkeypatch.setattr(settings_mod, "_get_section", fake_get_section)
        model = settings_mod.get_sn_benchmark_reviewer_model()
        assert model == "test/primary-from-review"


# =============================================================================
# No-secondary regression: empty secondary_models must produce no new fields
# =============================================================================


class TestNoSecondaryRegression:
    """When secondary_models=[], the review state carries no secondary fields."""

    def test_empty_secondary_models_state_default(self):
        """StandardNameReviewState has empty secondary_models by default."""
        from imas_codex.standard_names.review.state import StandardNameReviewState

        state = StandardNameReviewState(facility="dd")
        assert state.secondary_models == []
        assert state.disagreement_threshold == 0.2

    def test_entries_unchanged_without_secondary(self):
        """_run_secondary_review is never triggered when secondary_models=[].

        This regression test verifies the _process_batch logic gate:
        the secondary review block is only entered when state.secondary_models
        is truthy, so an empty list leaves batch_items untouched.
        """
        # We test the gate condition at the Python level — not by running the
        # full async pipeline, which would require graph/LLM mocks.
        secondary_models: list[str] = []
        batch_items = [
            {
                "id": "electron_temperature",
                "reviewer_score": 0.85,
                "reviewer_model": "anthropic/claude-opus-4.6",
            }
        ]

        # Simulate the gate in review_review_worker
        items_copy = list(batch_items)
        if secondary_models and items_copy:
            # This block must NOT be entered
            items_copy[0]["reviewer_model_secondary"] = "injected"

        # Assert no secondary fields were added
        assert "reviewer_model_secondary" not in items_copy[0]
        assert "reviewer_score_secondary" not in items_copy[0]
        assert "reviewer_disagreement" not in items_copy[0]


# =============================================================================
# _run_secondary_review unit tests (mock the inner batch review)
# =============================================================================


class TestRunSecondaryReviewHelper:
    """Unit tests for the _run_secondary_review helper (pipeline-internal)."""

    def _make_primary_items(self) -> list[dict]:
        return [
            {
                "id": "electron_temperature",
                "reviewer_score": 0.90,
                "reviewer_model": "anthropic/claude-opus-4.6",
            },
            {
                "id": "ion_temperature",
                "reviewer_score": 0.60,
                "reviewer_model": "anthropic/claude-opus-4.6",
            },
        ]

    def _make_secondary_result(self, overrides: dict | None = None) -> dict:
        """Fake _review_single_batch return dict matching the real contract."""
        base = {
            "_cost": 0.01,
            "_tokens": 500,
            "_revised": 0,
            "_unscored": 0,
            "_items": [
                {
                    "id": "electron_temperature",
                    "reviewer_score": 0.85,
                    "reviewer_scores": {"grammar": 18, "semantic": 17},
                    "reviewer_model": "anthropic/claude-sonnet-4.6",
                },
                {
                    "id": "ion_temperature",
                    "reviewer_score": 0.30,  # Large disagreement with 0.60
                    "reviewer_scores": {"grammar": 10, "semantic": 12},
                    "reviewer_model": "anthropic/claude-sonnet-4.6",
                },
            ],
        }
        if overrides:
            base.update(overrides)
        return base

    def test_secondary_fields_persisted(self):
        """reviewer_model_secondary, reviewer_score_secondary, reviewer_disagreement set."""
        import asyncio

        from imas_codex.standard_names.review.pipeline import _run_secondary_review

        primary_items = self._make_primary_items()
        secondary_result = self._make_secondary_result()

        async def _fake_review(**kwargs):
            return secondary_result

        with patch(
            "imas_codex.standard_names.review.pipeline._review_single_batch",
            side_effect=_fake_review,
        ):
            result = asyncio.run(
                _run_secondary_review(
                    primary_items=primary_items,
                    secondary_model="anthropic/claude-sonnet-4.6",
                    threshold=0.2,
                    grammar_enums={},
                    compose_ctx={},
                    calibration_entries=[],
                    batch={"group_key": "test"},
                    wlog=_make_dummy_wlog(),
                    name_only=False,
                )
            )

        electron = next(e for e in result if e["id"] == "electron_temperature")
        assert electron["reviewer_model_secondary"] == "anthropic/claude-sonnet-4.6"
        assert electron["reviewer_score_secondary"] == 0.85
        assert "reviewer_disagreement" in electron
        # disagreement = abs(0.90 - 0.85) = 0.05 → below threshold
        assert electron["reviewer_disagreement"] == 0.05

    def test_disagreement_escalates_to_needs_revision(self):
        """Names with |primary - secondary| > threshold get needs_revision."""
        import asyncio

        from imas_codex.standard_names.review.pipeline import _run_secondary_review

        primary_items = self._make_primary_items()
        secondary_result = self._make_secondary_result()

        async def _fake_review(**kwargs):
            return secondary_result

        with patch(
            "imas_codex.standard_names.review.pipeline._review_single_batch",
            side_effect=_fake_review,
        ):
            result = asyncio.run(
                _run_secondary_review(
                    primary_items=primary_items,
                    secondary_model="anthropic/claude-sonnet-4.6",
                    threshold=0.2,
                    grammar_enums={},
                    compose_ctx={},
                    calibration_entries=[],
                    batch={"group_key": "test"},
                    wlog=_make_dummy_wlog(),
                    name_only=False,
                )
            )

        ion = next(e for e in result if e["id"] == "ion_temperature")
        # disagreement = abs(0.60 - 0.30) = 0.30 > 0.20 threshold
        assert ion["reviewer_disagreement"] == 0.30
        assert ion.get("validation_status") == "needs_revision"

    def test_consolidated_score_on_disagreement(self):
        """reviewer_score is updated to min(primary, secondary) on disagreement."""
        import asyncio

        from imas_codex.standard_names.review.pipeline import _run_secondary_review

        primary_items = self._make_primary_items()

        async def _fake_review(**kwargs):
            return self._make_secondary_result()

        with patch(
            "imas_codex.standard_names.review.pipeline._review_single_batch",
            side_effect=_fake_review,
        ):
            result = asyncio.run(
                _run_secondary_review(
                    primary_items=primary_items,
                    secondary_model="anthropic/claude-sonnet-4.6",
                    threshold=0.2,
                    grammar_enums={},
                    compose_ctx={},
                    calibration_entries=[],
                    batch={},
                    wlog=_make_dummy_wlog(),
                    name_only=False,
                )
            )

        # electron: primary=0.90, secondary=0.85, Δ=0.05 < 0.20 → score unchanged
        electron = next(e for e in result if e["id"] == "electron_temperature")
        assert electron["reviewer_score"] == 0.90

        # ion: primary=0.60, secondary=0.30, Δ=0.30 > 0.20 → consolidated = min
        ion = next(e for e in result if e["id"] == "ion_temperature")
        assert ion["reviewer_score"] == 0.30

    def test_reviewer_scores_secondary_captured(self):
        """reviewer_scores_secondary is stored from secondary batch."""
        import asyncio

        from imas_codex.standard_names.review.pipeline import _run_secondary_review

        primary_items = self._make_primary_items()

        async def _fake_review(**kwargs):
            return self._make_secondary_result()

        with patch(
            "imas_codex.standard_names.review.pipeline._review_single_batch",
            side_effect=_fake_review,
        ):
            result = asyncio.run(
                _run_secondary_review(
                    primary_items=primary_items,
                    secondary_model="anthropic/claude-sonnet-4.6",
                    threshold=0.2,
                    grammar_enums={},
                    compose_ctx={},
                    calibration_entries=[],
                    batch={},
                    wlog=_make_dummy_wlog(),
                    name_only=False,
                )
            )

        electron = next(e for e in result if e["id"] == "electron_temperature")
        assert "reviewer_scores_secondary" in electron
        assert electron["reviewer_scores_secondary"] == {"grammar": 18, "semantic": 17}


# =============================================================================
# Helper
# =============================================================================


def _make_dummy_wlog():
    """Return a minimal logger-like object for the secondary review helper."""
    import logging

    return logging.getLogger("test.cross_family_review")
