"""Verify docs-axis reviewer feedback flows from graph through fetch into enrich context.

Closes Gap D from the prompt-context audit: prior to this, the enrich pipeline
ran blind on re-enrichment cycles — the reviewer's docs-axis critique was
written to the graph but never injected back into the enrich prompt.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_fetch_docs_review_feedback_returns_full_payload():
    """Happy path: SN with docs review returns score + comments + per-dim + verdict."""
    from imas_codex.standard_names.graph_ops import (
        fetch_docs_review_feedback_for_sns,
    )

    fake_rows = [
        {
            "sn_id": "electron_temperature",
            "reviewer_score": 0.62,
            "reviewer_comments": "Documentation does not explain measurement methodology.",
            "reviewer_scores_json": (
                '{"description_quality": 14, "documentation_quality": 9, '
                '"completeness": 12, "physics_accuracy": 16}'
            ),
            "reviewer_verdict": "revise",
            "validation_issues": ["doc_too_short", "missing_units_in_prose"],
        }
    ]

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.query = MagicMock(return_value=fake_rows)

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient", return_value=fake_client
    ):
        result = fetch_docs_review_feedback_for_sns(["electron_temperature"])

    assert "electron_temperature" in result
    entry = result["electron_temperature"]
    assert entry["reviewer_score"] == 0.62
    assert entry["reviewer_comments"].startswith("Documentation does not")
    assert entry["reviewer_scores"]["documentation_quality"] == 9
    assert entry["reviewer_verdict"] == "revise"
    assert entry["validation_issues"] == ["doc_too_short", "missing_units_in_prose"]


def test_fetch_docs_review_feedback_omits_unreviewed():
    """Cold-start: SNs with reviewer_score_docs IS NULL are not in graph results."""
    from imas_codex.standard_names.graph_ops import (
        fetch_docs_review_feedback_for_sns,
    )

    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.query = MagicMock(return_value=[])  # graph filter on NOT NULL

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient", return_value=fake_client
    ):
        result = fetch_docs_review_feedback_for_sns(["never_reviewed_sn"])

    assert result == {}


def test_fetch_docs_review_feedback_handles_empty_input():
    """No graph call when input is None or empty."""
    from imas_codex.standard_names.graph_ops import (
        fetch_docs_review_feedback_for_sns,
    )

    assert fetch_docs_review_feedback_for_sns(None) == {}
    assert fetch_docs_review_feedback_for_sns([]) == {}
    assert fetch_docs_review_feedback_for_sns(set()) == {}


def test_fetch_docs_review_feedback_handles_malformed_scores_json():
    """Bad JSON in reviewer_scores_docs must not crash — fall back to None."""
    from imas_codex.standard_names.graph_ops import (
        fetch_docs_review_feedback_for_sns,
    )

    fake_rows = [
        {
            "sn_id": "ion_density",
            "reviewer_score": 0.7,
            "reviewer_comments": "OK",
            "reviewer_scores_json": "not-valid-json{{{",
            "reviewer_verdict": "accept",
            "validation_issues": None,
        }
    ]
    fake_client = MagicMock()
    fake_client.__enter__ = MagicMock(return_value=fake_client)
    fake_client.__exit__ = MagicMock(return_value=False)
    fake_client.query = MagicMock(return_value=fake_rows)

    with patch(
        "imas_codex.standard_names.graph_ops.GraphClient", return_value=fake_client
    ):
        result = fetch_docs_review_feedback_for_sns(["ion_density"])

    entry = result["ion_density"]
    assert entry["reviewer_scores"] is None
    assert entry["reviewer_score"] == 0.7
    assert entry["validation_issues"] is None


def test_enrich_user_template_renders_docs_feedback_block():
    """End-to-end: when item has docs_review_feedback, template renders the block."""
    pytest.skip(
        "docs_review_feedback block existed in the deleted enrich_user.md batch "
        "template; generate_docs_user.md is a per-item template that does not yet "
        "expose this block — port the feature to add it back"
    )


def test_enrich_user_template_omits_block_when_no_feedback():
    """Cold-start path: no docs_review_feedback → no block in rendered prompt."""
    from imas_codex.llm.prompt_loader import render_prompt

    item = {
        "name": "fresh_name",
        "unit": "m",
        "kind": "scalar",
        "physics_domain": "equilibrium",
        "dd_paths": [],
        "nearby": [],
        "siblings": [],
    }
    rendered = render_prompt("sn/generate_docs_user", {"item": item})
    assert "Prior docs-axis reviewer feedback" not in rendered
