"""Tests for divergence detection in export.

Plan 35 §3d: divergence-detection unit test — catalog-edited names
with modified protected fields are flagged.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.export import detect_divergence


def _make_candidate(
    name: str,
    origin: str = "pipeline",
    catalog_commit_sha: str | None = None,
    **kwargs,
) -> dict:
    return {
        "id": name,
        "origin": origin,
        "catalog_commit_sha": catalog_commit_sha,
        "description": "Some description",
        "documentation": "Some docs",
        "kind": "scalar",
        "links": [],
        **kwargs,
    }


class TestDetectDivergence:
    """detect_divergence flags catalog-edited names with lineage."""

    def test_pipeline_origin_no_divergence(self) -> None:
        """Pipeline-origin names are never flagged."""
        candidates = [_make_candidate("normal_name", origin="pipeline")]
        findings = detect_divergence(candidates)
        assert len(findings) == 0

    def test_catalog_edit_without_sha_no_divergence(self) -> None:
        """catalog_edit without catalog_commit_sha — nothing to compare."""
        candidates = [
            _make_candidate(
                "edited_name",
                origin="catalog_edit",
                catalog_commit_sha=None,
            )
        ]
        findings = detect_divergence(candidates)
        assert len(findings) == 0

    def test_catalog_edit_with_sha_flagged(self) -> None:
        """catalog_edit WITH catalog_commit_sha — flagged for verification."""
        candidates = [
            _make_candidate(
                "edited_name",
                origin="catalog_edit",
                catalog_commit_sha="abc123def456",
            )
        ]
        findings = detect_divergence(candidates)
        assert len(findings) == 1
        assert findings[0].name == "edited_name"
        assert "abc123de" in findings[0].detail

    def test_multiple_candidates_mixed(self) -> None:
        """Only catalog-edited names with SHA are flagged."""
        candidates = [
            _make_candidate("pipeline_name", origin="pipeline"),
            _make_candidate(
                "edited_a",
                origin="catalog_edit",
                catalog_commit_sha="sha_a",
            ),
            _make_candidate(
                "edited_b",
                origin="catalog_edit",
                catalog_commit_sha=None,
            ),
            _make_candidate(
                "edited_c",
                origin="catalog_edit",
                catalog_commit_sha="sha_c",
            ),
        ]
        findings = detect_divergence(candidates)
        flagged_names = {f.name for f in findings}
        assert flagged_names == {"edited_a", "edited_c"}

    def test_divergence_entry_has_hash(self) -> None:
        """Each finding includes a hash of protected field values."""
        candidates = [
            _make_candidate(
                "hashed_name",
                origin="catalog_edit",
                catalog_commit_sha="deadbeef",
            )
        ]
        findings = detect_divergence(candidates)
        assert len(findings) == 1
        assert findings[0].graph_hash  # Non-empty hash string
        assert len(findings[0].graph_hash) == 16  # Truncated SHA-256
