"""Tests for chain_history helpers.

All graph calls are mocked — no live Neo4j required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.chain_history import (
    _parse_comments,
    docs_chain_history,
    name_chain_history,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "imas_codex.standard_names.chain_history.GraphClient"


def _make_gc(rows: list[dict]) -> MagicMock:
    """Return a mock GraphClient context-manager whose .query() returns *rows*."""
    gc = MagicMock()
    gc.query.return_value = rows
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=gc)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# name_chain_history
# ---------------------------------------------------------------------------


class TestNameChainHistory:
    def test_oldest_first_ordering(self):
        """Graph rows come back ordered by chain_length ASC (oldest first).

        The function trusts the ORDER BY in Cypher and preserves row order,
        so we verify the output matches the row sequence.
        """
        rows = [
            {
                "name": "electron_T_lcfs",
                "model": "gpt-4o",
                "reviewer_score": 0.42,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": json.dumps(
                    {"grammar": "Base too abbreviated"}
                ),
                "generated_at": "2024-01-01T00:00:00Z",
            },
            {
                "name": "e_temp_at_separatrix",
                "model": "gpt-4o",
                "reviewer_score": 0.58,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": json.dumps(
                    {"convention": "Method token leaked in"}
                ),
                "generated_at": "2024-01-02T00:00:00Z",
            },
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = name_chain_history("electron_temperature_at_lcfs")

        assert len(result) == 2
        assert result[0]["name"] == "electron_T_lcfs"
        assert result[1]["name"] == "e_temp_at_separatrix"

    def test_respects_limit(self):
        """Only *limit* rows are forwarded to the caller (limit passed to Cypher)."""
        # Return exactly 3 rows to simulate LIMIT 3 from Cypher
        rows = [
            {
                "name": f"name_v{i}",
                "model": "gpt-4o",
                "reviewer_score": 0.4 + i * 0.05,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": None,
                "generated_at": None,
            }
            for i in range(3)
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = name_chain_history("some_sn", limit=3)

        assert len(result) == 3

    def test_empty_when_no_chain(self):
        """A fresh SN with no REFINED_FROM ancestors returns an empty list."""
        with patch(_MODULE, return_value=_make_gc([])):
            result = name_chain_history("brand_new_name")

        assert result == []

    def test_missing_fields_have_defaults(self):
        """None / missing graph properties are coerced to safe defaults."""
        rows = [
            {
                "name": "partial_sn",
                "model": None,
                "reviewer_score": None,
                "reviewer_verdict": None,
                "reviewer_comments_per_dim": None,
                "generated_at": None,
            }
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = name_chain_history("partial_sn")

        entry = result[0]
        assert entry["model"] == "unknown"
        assert entry["reviewer_score"] == 0.0
        assert entry["reviewer_verdict"] == "unknown"
        assert entry["reviewer_comments_per_dim"] == {}
        assert entry["generated_at"] is None

    def test_cypher_receives_sn_id_and_limit(self):
        """The correct sn_id and limit kwargs are forwarded to gc.query."""
        gc_mock = MagicMock()
        gc_mock.query.return_value = []
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=gc_mock)
        cm.__exit__ = MagicMock(return_value=False)

        with patch(_MODULE, return_value=cm):
            name_chain_history("test_sn_id", limit=4)

        call_kwargs = gc_mock.query.call_args[1]
        assert call_kwargs["sn_id"] == "test_sn_id"
        assert call_kwargs["limit"] == 4


# ---------------------------------------------------------------------------
# docs_chain_history
# ---------------------------------------------------------------------------


class TestDocsChainHistory:
    def test_oldest_first_ordering(self):
        """Rows are returned in created_at ASC order (as Cypher orders them)."""
        rows = [
            {
                "documentation": "First attempt docs.",
                "model": "claude-3",
                "reviewer_score": 0.55,
                "reviewer_comments_per_dim": json.dumps({"clarity": "Too terse"}),
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "documentation": "Second attempt docs.",
                "model": "claude-3",
                "reviewer_score": 0.68,
                "reviewer_comments_per_dim": json.dumps(
                    {"clarity": "Better but missing context"}
                ),
                "created_at": "2024-01-02T00:00:00Z",
            },
            {
                "documentation": "Third attempt docs.",
                "model": "gpt-4o",
                "reviewer_score": 0.72,
                "reviewer_comments_per_dim": None,
                "created_at": "2024-01-03T00:00:00Z",
            },
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = docs_chain_history("electron_temperature")

        assert len(result) == 3
        assert result[0]["documentation"] == "First attempt docs."
        assert result[2]["documentation"] == "Third attempt docs."

    def test_parses_json_comments(self):
        """reviewer_comments_per_dim stored as JSON string → decoded dict."""
        comments = {"clarity": "Needs more context", "accuracy": "Good"}
        rows = [
            {
                "documentation": "Some docs.",
                "model": "gpt-4o",
                "reviewer_score": 0.60,
                "reviewer_comments_per_dim": json.dumps(comments),
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = docs_chain_history("some_name")

        assert result[0]["reviewer_comments_per_dim"] == comments

    def test_accepts_already_decoded_dict_comments(self):
        """reviewer_comments_per_dim already a dict (e.g. from bolt driver) → passed through."""
        comments = {"physics": "Missing equation", "links": "Good"}
        rows = [
            {
                "documentation": "Some docs.",
                "model": "gpt-4o",
                "reviewer_score": 0.65,
                "reviewer_comments_per_dim": comments,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = docs_chain_history("some_name")

        assert result[0]["reviewer_comments_per_dim"] == comments

    def test_empty_documentation_coerced_to_empty_string(self):
        """None documentation from graph becomes empty string."""
        rows = [
            {
                "documentation": None,
                "model": "gpt-4o",
                "reviewer_score": 0.50,
                "reviewer_comments_per_dim": None,
                "created_at": None,
            }
        ]
        with patch(_MODULE, return_value=_make_gc(rows)):
            result = docs_chain_history("some_name")

        assert result[0]["documentation"] == ""

    def test_empty_when_no_revisions(self):
        """SN with no DocsRevision nodes returns empty list."""
        with patch(_MODULE, return_value=_make_gc([])):
            result = docs_chain_history("brand_new_name")

        assert result == []


# ---------------------------------------------------------------------------
# _parse_comments (unit tests for the internal helper)
# ---------------------------------------------------------------------------


class TestParseComments:
    def test_none_returns_empty_dict(self):
        assert _parse_comments(None) == {}

    def test_dict_passthrough(self):
        d = {"grammar": "ok", "semantic": "bad"}
        assert _parse_comments(d) is d

    def test_valid_json_string(self):
        assert _parse_comments('{"a": "1", "b": "2"}') == {"a": "1", "b": "2"}

    def test_invalid_json_returns_empty_dict(self):
        assert _parse_comments("not-json-at-all") == {}

    def test_json_non_dict_returns_empty_dict(self):
        assert _parse_comments("[1, 2, 3]") == {}

    def test_other_types_return_empty_dict(self):
        assert _parse_comments(42) == {}  # type: ignore[arg-type]
        assert _parse_comments(3.14) == {}  # type: ignore[arg-type]
