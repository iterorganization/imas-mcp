"""Verify reviewer_suggested_name flows from graph through fetch into compose context."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_fetch_review_feedback_includes_suggested_name():
    """Regression: reviewer_suggested_name and justification must be returned to compose."""
    from imas_codex.standard_names.graph_ops import fetch_review_feedback_for_sources

    fake_rows = [
        {
            "source_id": "dd:eq/profiles_1d/psi",
            "previous_name": "poloidal_flux_old",
            "previous_description": "old desc",
            "previous_documentation": "old doc",
            "reviewer_score": 0.45,
            "review_tier": "inadequate",
            "reviewer_comments": "Name lacks locus distinguisher.",
            "reviewer_scores_json": '{"grammar": 12, "semantic": 10}',
            "reviewer_suggested_name": "poloidal_magnetic_flux",
            "reviewer_suggestion_justification": "Cluster siblings use _magnetic_ qualifier.",
            "validation_status": "valid",
        }
    ]

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.query = MagicMock(return_value=fake_rows)

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient", return_value=fake_client
    ):
        result = fetch_review_feedback_for_sources(["dd:eq/profiles_1d/psi"])

    assert "dd:eq/profiles_1d/psi" in result
    entry = result["dd:eq/profiles_1d/psi"]
    assert entry["reviewer_suggested_name"] == "poloidal_magnetic_flux"
    assert (
        entry["reviewer_suggestion_justification"]
        == "Cluster siblings use _magnetic_ qualifier."
    )
    # Sanity: existing fields still present
    assert entry["previous_name"] == "poloidal_flux_old"
    assert entry["reviewer_comments"] == "Name lacks locus distinguisher."


def test_fetch_review_feedback_handles_null_suggested_name():
    """When verdict is accept, suggested_name + justification are null — fetch must not crash."""
    from imas_codex.standard_names.graph_ops import fetch_review_feedback_for_sources

    fake_rows = [
        {
            "source_id": "dd:cp/electrons/temperature",
            "previous_name": "electron_temperature",
            "previous_description": "good desc",
            "previous_documentation": None,
            "reviewer_score": 0.92,
            "review_tier": "outstanding",
            "reviewer_comments": "Name is excellent.",
            "reviewer_scores_json": None,
            "reviewer_suggested_name": None,
            "reviewer_suggestion_justification": None,
            "validation_status": "valid",
        }
    ]

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.query = MagicMock(return_value=fake_rows)

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient", return_value=fake_client
    ):
        result = fetch_review_feedback_for_sources(["dd:cp/electrons/temperature"])

    entry = result["dd:cp/electrons/temperature"]
    assert entry["reviewer_suggested_name"] is None
    assert entry["reviewer_suggestion_justification"] is None
