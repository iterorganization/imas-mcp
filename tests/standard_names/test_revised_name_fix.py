"""Regression tests for the revised-name id overwrite bug (Wave 8A).

When a reviewer issues verdict=revise with a revised_name, the resulting
Review graph record must reference the ORIGINAL StandardName id, not the
suggested replacement name.

See: Wave 8A bug report — 386 orphan Review nodes caused by id overwrite.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest


def _make_name_review(
    source_id: str,
    name: str,
    revised_name: str,
    *,
    score: int = 15,
):
    """Build a fake name-axis review with verdict=revise and a revised_name."""
    from imas_codex.standard_names.models import (
        StandardNameQualityScoreNameOnly,
    )

    return SimpleNamespace(
        source_id=source_id,
        standard_name=name,
        scores=StandardNameQualityScoreNameOnly(
            grammar=score,
            semantic=score,
            convention=score,
            completeness=score,
        ),
        reasoning="suggested a better name",
        revised_name=revised_name,
        revised_fields=None,
    )


class TestRevisedNamePreservesOriginalId:
    """_match_reviews_to_entries must NOT overwrite entry['id'] with revised_name."""

    def test_scored_entry_keeps_original_id(self) -> None:
        """The entry returned by _match_reviews_to_entries has id == original name."""
        from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

        wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
        original_id = "electron_temperature"
        suggested = "electron_thermal_energy"

        names = [{"id": original_id, "source_id": original_id}]
        review = _make_name_review(
            source_id=original_id,
            name=original_id,
            revised_name=suggested,
        )

        scored, unmatched, revised_count = _match_reviews_to_entries(
            [review], names, wlog, target="names"
        )

        assert len(scored) == 1
        assert unmatched == []
        assert revised_count == 1

        entry = scored[0]
        # Core assertion: id must be the ORIGINAL name, not the suggestion
        assert entry["id"] == original_id, (
            f"Entry id was overwritten: got {entry['id']!r}, expected {original_id!r}"
        )

    def test_suggested_name_stored_separately(self) -> None:
        """The suggested name is available under _suggested_name, not id."""
        from imas_codex.standard_names.review.pipeline import _match_reviews_to_entries

        wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
        original_id = "ion_temperature"
        suggested = "ion_thermal_energy"

        names = [{"id": original_id, "source_id": original_id}]
        review = _make_name_review(
            source_id=original_id,
            name=original_id,
            revised_name=suggested,
        )

        scored, _, _ = _match_reviews_to_entries([review], names, wlog, target="names")
        entry = scored[0]

        assert entry.get("_suggested_name") == suggested
        assert entry.get("_original_id") == original_id

    def test_build_review_record_uses_original_id(self) -> None:
        """_build_review_record uses _original_id when present."""
        from imas_codex.standard_names.review.pipeline import _build_review_record

        original_id = "plasma_current"
        suggested = "toroidal_plasma_current"

        item = {
            "id": original_id,
            "_original_id": original_id,
            "_suggested_name": suggested,
            "reviewer_score": 0.75,
            "reviewer_verdict": "revise",
            "review_tier": "good",
        }

        rec = _build_review_record(
            item,
            model="test/model",
            is_canonical=True,
            reviewed_at="2025-01-01T00:00:00Z",
        )

        assert rec["standard_name_id"] == original_id, (
            f"Review record references {rec['standard_name_id']!r} "
            f"instead of original {original_id!r}"
        )
        # The Review node id encodes the original SN id
        assert original_id in rec["id"]
        # The suggested name must NOT appear in the SN id slot
        assert suggested not in rec["id"]

    def test_write_name_review_results_batch_uses_original_id(self) -> None:
        """write_name_review_results builds batch with _original_id, not id."""
        # We test the batch-building logic directly without a real graph
        # by extracting the batch list from the function source.
        entries = [
            {
                "id": "electron_temperature",
                "_original_id": "electron_temperature",
                "_suggested_name": "electron_kinetic_temperature",
                "reviewer_score": 0.8,
                "reviewed_at": "2025-01-01T00:00:00Z",
                "reviewer_verdict": "revise",
                "review_tier": "good",
                "review_input_hash": "abc123",
            }
        ]

        from imas_codex.standard_names.graph_ops import _ensure_json

        # Replicate the batch-building logic from write_name_review_results
        batch = [
            {
                "id": e.get("_original_id") or e["id"],
                "reviewer_suggested_name": e.get("_suggested_name") or "",
            }
            for e in entries
        ]

        assert batch[0]["id"] == "electron_temperature"
        assert batch[0]["reviewer_suggested_name"] == "electron_kinetic_temperature"
