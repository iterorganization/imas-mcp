"""Tests for LLMCost deletion on `sn clear --force` and `clear_standard_names()`.

Verifies that both the full-wipe path (`sn clear --force` → `clear_sn_subsystem`)
and the partial-reset path (`sn run --reset-to extracted` → `clear_standard_names`)
both delete LLMCost nodes so the cost ledger is not left with stale rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_gc(counts: dict[str, int] | None = None) -> MagicMock:
    """Return a mock GraphClient context manager.

    ``query`` returns a count row for MATCH…RETURN count(n) calls, and a
    no-op empty list for DETACH DELETE / DELETE calls.
    """
    fake_gc = MagicMock()
    fake_gc.__enter__.return_value = fake_gc
    fake_gc.__exit__.return_value = None

    default_count = counts or {}

    def _query(cypher: str, **_kwargs):
        # Count queries: MATCH (n:Label) RETURN count(n) AS n
        for label, n in default_count.items():
            if f":{label}" in cypher and "count(n)" in cypher:
                return [{"n": n}]
        if "count(n)" in cypher or "count(r)" in cypher or "count(sn)" in cypher:
            return [{"n": 0}]
        # Delete / other queries return empty
        return []

    fake_gc.query = MagicMock(side_effect=_query)
    return fake_gc


# ---------------------------------------------------------------------------
# clear_sn_subsystem (sn clear --force path)
# ---------------------------------------------------------------------------


class TestClearSnSubsystemDeletesLLMCost:
    """``clear_sn_subsystem`` must include LLMCost in its deletion sweep."""

    def test_llmcost_in_returned_labels(self):
        """Dry-run result dict must include the LLMCost key."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc({"LLMCost": 3})

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_sn_subsystem(dry_run=True)

        assert "LLMCost" in result

    def test_llmcost_count_returned_in_dry_run(self):
        """Dry-run must report the correct LLMCost count without deleting."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc({"LLMCost": 7})

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_sn_subsystem(dry_run=True)

        assert result["LLMCost"] == 7

    def test_llmcost_detach_deleted_on_wipe(self):
        """Full wipe must issue DETACH DELETE on LLMCost nodes."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc(
            {
                "StandardName": 5,
                "Review": 2,
                "StandardNameSource": 3,
                "VocabGap": 1,
                "SNRun": 1,
                "LLMCost": 4,
            }
        )

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        delete_queries = [q for q in queries if "DETACH DELETE" in q]
        assert any("LLMCost" in q for q in delete_queries), (
            "Expected a DETACH DELETE query targeting LLMCost"
        )

    def test_all_six_pipeline_labels_deleted(self):
        """All six pipeline-output labels must have a DETACH DELETE query."""
        from imas_codex.standard_names import graph_ops

        expected = {
            "StandardName",
            "Review",
            "StandardNameSource",
            "VocabGap",
            "SNRun",
            "LLMCost",
        }
        fake_gc = _make_fake_gc(dict.fromkeys(expected, 1))

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=False)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        delete_queries = [q for q in queries if "DETACH DELETE" in q]
        for label in expected:
            assert any(label in q for q in delete_queries), (
                f"Missing DETACH DELETE for label: {label}"
            )

    def test_dry_run_no_detach_delete_for_llmcost(self):
        """Dry-run must never issue a DETACH DELETE for LLMCost."""
        from imas_codex.standard_names import graph_ops

        fake_gc = _make_fake_gc({"LLMCost": 10})

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_sn_subsystem(dry_run=True)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries), (
            "dry_run=True must not issue any DETACH DELETE"
        )


# ---------------------------------------------------------------------------
# clear_standard_names (sn run --reset-to extracted path)
# ---------------------------------------------------------------------------


class TestClearStandardNamesDeletesLLMCost:
    """``clear_standard_names`` must delete LLMCost rows on the reset path."""

    def _make_gc_for_clear_standard_names(self, sn_count: int = 5) -> MagicMock:
        """Build a fake GC that reports sn_count matching StandardName nodes."""
        fake_gc = MagicMock()
        fake_gc.__enter__.return_value = fake_gc
        fake_gc.__exit__.return_value = None

        def _query(cypher: str, **_kwargs):
            if "count(DISTINCT sn)" in cypher or "count(sn)" in cypher:
                return [{"n": sn_count}]
            if "count(r)" in cypher:
                return [{"n": 0}]
            return []

        fake_gc.query = MagicMock(side_effect=_query)
        return fake_gc

    def test_llmcost_detach_deleted_on_full_clear(self):
        """``clear_standard_names()`` must DETACH DELETE LLMCost nodes."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=3)

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "clear_standard_names must issue DETACH DELETE for LLMCost"
        )

    def test_llmcost_deleted_with_source_filter(self):
        """LLMCost must be deleted even when source_filter is applied."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=2)

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            graph_ops.clear_standard_names(source_filter="dd")

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "clear_standard_names with source_filter must still delete LLMCost"
        )

    def test_llmcost_not_deleted_in_dry_run(self):
        """Dry-run must not issue DETACH DELETE for LLMCost."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=5)

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_standard_names(dry_run=True)

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("LLMCost" in q and "DETACH DELETE" in q for q in queries), (
            "dry_run=True must not delete LLMCost"
        )
        # dry_run returns the count of SNs that would be deleted, not 0
        assert isinstance(result, int)

    def test_llmcost_not_deleted_when_no_matching_sn(self):
        """When count == 0, no deletions including LLMCost should occur."""
        from imas_codex.standard_names import graph_ops

        fake_gc = self._make_gc_for_clear_standard_names(sn_count=0)

        with patch.object(graph_ops, "GraphClient", return_value=fake_gc):
            result = graph_ops.clear_standard_names()

        queries = [c.args[0] for c in fake_gc.query.call_args_list]
        assert not any("DETACH DELETE" in q for q in queries), (
            "When no matching SNs, no DETACH DELETE should be issued"
        )
        assert result == 0
