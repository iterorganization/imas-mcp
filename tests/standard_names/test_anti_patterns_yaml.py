"""Schema validation for ``imas_codex/standard_names/anti_patterns.yaml``.

The anti-patterns YAML is the static source of decomposition-failure
exemplars rendered into the SN compose system prompt.  These tests guard
the schema and minimum-content guarantees so regressions surface in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ANTI_PATTERNS_PATH = (
    Path(__file__).resolve().parents[2]
    / "imas_codex"
    / "standard_names"
    / "anti_patterns.yaml"
)

REQUIRED_KEYS = {
    "bad_name",
    "issue_category",
    "reviewer_comment",
    "absorbed_tokens",
    "correct_decomposition",
    "rewritten_name",
}

VALID_SEGMENTS = {
    "subject",
    "component",
    "coordinate",
    "device",
    "object",
    "geometry",
    "position",
    "process",
    "transformation",
    "geometric_base",
    "region",
    "physical_base",
}


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    assert ANTI_PATTERNS_PATH.exists(), f"missing {ANTI_PATTERNS_PATH}"
    with open(ANTI_PATTERNS_PATH) as f:
        data = yaml.safe_load(f)
    assert isinstance(data, list), "anti_patterns.yaml must be a top-level list"
    return data


class TestSchema:
    def test_minimum_entry_count(self, entries):
        assert len(entries) >= 6, (
            f"anti_patterns.yaml must contain ≥6 entries (got {len(entries)})"
        )

    def test_every_entry_has_required_keys(self, entries):
        for i, entry in enumerate(entries):
            missing = REQUIRED_KEYS - entry.keys()
            assert not missing, (
                f"entry #{i} ({entry.get('bad_name')!r}) missing keys: {missing}"
            )

    def test_issue_category_is_decomposition_failure(self, entries):
        for entry in entries:
            assert entry["issue_category"] == "decomposition_failure", (
                f"{entry['bad_name']!r}: issue_category must be 'decomposition_failure'"
            )

    def test_absorbed_tokens_well_formed(self, entries):
        for entry in entries:
            absorbed = entry["absorbed_tokens"]
            assert isinstance(absorbed, list) and absorbed, (
                f"{entry['bad_name']!r}: absorbed_tokens must be a non-empty list"
            )
            for at in absorbed:
                assert {"token", "segment"} <= at.keys()
                assert at["segment"] in VALID_SEGMENTS, (
                    f"{entry['bad_name']!r}: absorbed segment {at['segment']!r} unknown"
                )

    def test_correct_decomposition_uses_valid_segments(self, entries):
        for entry in entries:
            for seg in entry["correct_decomposition"]:
                assert seg in VALID_SEGMENTS, (
                    f"{entry['bad_name']!r}: correct_decomposition has unknown segment {seg!r}"
                )

    def test_reviewer_comment_length_bounded(self, entries):
        for entry in entries:
            comment = entry["reviewer_comment"].strip()
            assert 30 <= len(comment) <= 600, (
                f"{entry['bad_name']!r}: reviewer_comment length {len(comment)} "
                "out of bounds (30..600)"
            )

    def test_rewritten_name_differs_from_bad(self, entries):
        for entry in entries:
            assert entry["rewritten_name"] != entry["bad_name"], (
                f"{entry['bad_name']!r}: rewritten_name identical to bad_name"
            )

    def test_absorbed_tokens_appear_in_bad_name(self, entries):
        for entry in entries:
            bad = f"_{entry['bad_name']}_"
            for at in entry["absorbed_tokens"]:
                tok = at["token"]
                assert f"_{tok}_" in bad, (
                    f"{entry['bad_name']!r}: absorbed token {tok!r} not present "
                    "as underscore-delimited substring in the bad name"
                )


class TestLoaderIntegration:
    def test_loader_returns_entries(self):
        from imas_codex.standard_names.context import _load_decomposition_anti_patterns

        # lru_cache may have been primed by an earlier test — clear so we
        # exercise the fresh-load path.
        _load_decomposition_anti_patterns.cache_clear()
        entries = _load_decomposition_anti_patterns()
        assert len(entries) >= 6
        for e in entries:
            assert "bad_name" in e

    def test_context_builder_exposes_anti_patterns(self):
        from imas_codex.standard_names.context import (
            build_compose_context,
            clear_context_cache,
        )

        clear_context_cache()
        ctx = build_compose_context()
        assert "decomposition_anti_patterns" in ctx
        assert len(ctx["decomposition_anti_patterns"]) >= 6
