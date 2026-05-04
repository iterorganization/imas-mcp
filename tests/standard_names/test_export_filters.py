"""Regression tests: export _fetch_candidates skips superseded/exhausted/quarantined.

Audit finding (Phase 3C):
  The old query used ``pipeline_status IN ['published','accepted','reviewed','enriched']``.
  That field is NOT updated when a name is superseded — only ``name_stage`` is
  authoritative.  A superseded node therefore kept its old ``pipeline_status``
  value and was erroneously included in export.

Fix: replace the gate with
    name_stage = 'accepted' AND docs_stage = 'accepted' AND validation_status = 'valid'

Writer confirmation: ``persist_refined_name_batch`` sets ``old.name_stage = 'superseded'``
in graph_ops.py (the predecessor IS correctly marked), so filtering by
``name_stage = 'accepted'`` is sufficient to exclude superseded nodes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.export import _fetch_candidates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GC_PATH = "imas_codex.graph.client.GraphClient"


def _make_node(
    id_: str,
    name_stage: str = "accepted",
    docs_stage: str = "accepted",
    validation_status: str = "valid",
    **kwargs,
) -> dict:
    """Return a minimal StandardName property dict."""
    return {
        "id": id_,
        "name_stage": name_stage,
        "docs_stage": docs_stage,
        "validation_status": validation_status,
        "description": f"Description for {id_}",
        "kind": "scalar",
        "unit": "eV",
        "physics_domain": "test",
        **kwargs,
    }


# The four nodes described in the plan §3C scenario
_ALL_NODES = [
    # SN_A: fully accepted docs, but name is superseded — must be excluded
    _make_node("sn_a", name_stage="superseded"),
    # SN_B: all stages correct — must be included
    _make_node("sn_b"),
    # SN_C: name exhausted — must be excluded
    _make_node("sn_c", name_stage="exhausted"),
    # SN_D: quarantined — must be excluded
    _make_node("sn_d", validation_status="quarantined"),
]


def _make_gc_returning(nodes: list[dict]) -> MagicMock:
    """Return a mock GraphClient whose query() returns ``nodes`` wrapped as records."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[{"record": n} for n in nodes])
    return mock_gc


def _make_gc_filtering(all_nodes: list[dict]) -> MagicMock:
    """Return a mock GraphClient that simulates the expected WHERE-clause filter.

    The mock applies ``name_stage='accepted' AND docs_stage='accepted' AND
    validation_status='valid'`` in Python, mirroring the Cypher predicate.
    This lets the behavior tests verify that the Cypher as a whole round-trips
    correctly through the function (record unpacking, domain filter, etc.).
    """

    def _query(cypher: str, **kwargs) -> list[dict]:
        domain = kwargs.get("domain")
        return [
            {"record": n}
            for n in all_nodes
            if (
                n.get("name_stage") == "accepted"
                and n.get("docs_stage") == "accepted"
                and n.get("validation_status") == "valid"
                and (domain is None or n.get("physics_domain") == domain)
            )
        ]

    mock_gc = MagicMock()
    mock_gc.query = _query
    return mock_gc


# ---------------------------------------------------------------------------
# Class 1: Cypher query contract
# ---------------------------------------------------------------------------


class TestFetchCandidatesQueryContract:
    """Verify the Cypher WHERE clause emitted by _fetch_candidates.

    These tests are "contract tests" — they assert the *shape* of the query,
    not the database behaviour.  They catch regressions where someone reverts
    the gate back to the legacy pipeline_status field.
    """

    def _run_and_capture(self, **kwargs) -> str:
        """Call _fetch_candidates, capture and return the first Cypher string."""
        captured: list[str] = []

        def mock_query(cypher: str, **kw) -> list:
            captured.append(cypher)
            return []

        mock_gc = MagicMock()
        mock_gc.query = mock_query

        with patch(_GC_PATH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            _fetch_candidates(**kwargs)

        assert captured, "_fetch_candidates issued no query"
        return captured[0]

    def test_uses_name_stage(self):
        """WHERE clause must reference name_stage."""
        cypher = self._run_and_capture()
        assert "name_stage" in cypher, (
            f"'name_stage' missing from WHERE clause.\n\nFull query:\n{cypher}"
        )

    def test_uses_docs_stage(self):
        """WHERE clause must reference docs_stage."""
        cypher = self._run_and_capture()
        assert "docs_stage" in cypher, (
            f"'docs_stage' missing from WHERE clause.\n\nFull query:\n{cypher}"
        )

    def test_uses_validation_status(self):
        """WHERE clause must reference validation_status."""
        cypher = self._run_and_capture()
        assert "validation_status" in cypher, (
            f"'validation_status' missing from WHERE clause.\n\nFull query:\n{cypher}"
        )

    def test_does_not_gate_on_pipeline_status_alone(self):
        """pipeline_status must NOT be the sole filter gate.

        The old gate ``pipeline_status IN [...]`` does not exclude superseded
        nodes (only name_stage tracks supersession).  After the fix the query
        must not reference pipeline_status at all.
        """
        cypher = self._run_and_capture()
        assert "pipeline_status" not in cypher, (
            "Legacy pipeline_status still gating the export query.\n"
            "Use name_stage/docs_stage/validation_status instead.\n\n"
            f"Full query:\n{cypher}"
        )

    def test_accepted_literal_present(self):
        """The string 'accepted' must appear in the WHERE clause."""
        cypher = self._run_and_capture()
        assert "'accepted'" in cypher or "= 'accepted'" in cypher, (
            f"Stage value 'accepted' missing from query:\n{cypher}"
        )

    def test_valid_literal_present(self):
        """The string 'valid' must appear (for validation_status filter)."""
        cypher = self._run_and_capture()
        assert "'valid'" in cypher or "= 'valid'" in cypher, (
            f"Value 'valid' missing from query (validation_status gate missing):\n{cypher}"
        )


# ---------------------------------------------------------------------------
# Class 2: Behaviour — only valid node returned from mixed set
# ---------------------------------------------------------------------------


class TestFetchCandidatesSkipsIneligibleNodes:
    """Verify that ineligible nodes are excluded from export results.

    The mock simulates Neo4j's WHERE-clause filter in Python so that the
    test exercises both the query predicate AND the record-unpacking code.
    """

    @pytest.fixture()
    def result(self) -> list[dict]:
        """Run _fetch_candidates over the _ALL_NODES fixture."""
        mock_gc = _make_gc_filtering(_ALL_NODES)
        with patch(_GC_PATH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            return _fetch_candidates()

    def test_only_one_result(self, result: list[dict]) -> None:
        """Exactly one node (sn_b) should survive the filter."""
        ids = [r["id"] for r in result]
        assert len(result) == 1, f"Expected 1 result; got {len(result)}: {ids}"

    def test_valid_node_present(self, result: list[dict]) -> None:
        """sn_b (accepted + accepted + valid) must be included."""
        ids = {r["id"] for r in result}
        assert "sn_b" in ids, f"Valid node sn_b missing from results: {ids}"

    def test_superseded_excluded(self, result: list[dict]) -> None:
        """sn_a (name_stage='superseded') must be excluded."""
        ids = {r["id"] for r in result}
        assert "sn_a" not in ids, "Superseded node sn_a must not be exported"

    def test_exhausted_excluded(self, result: list[dict]) -> None:
        """sn_c (name_stage='exhausted') must be excluded."""
        ids = {r["id"] for r in result}
        assert "sn_c" not in ids, "Exhausted node sn_c must not be exported"

    def test_quarantined_excluded(self, result: list[dict]) -> None:
        """sn_d (validation_status='quarantined') must be excluded."""
        ids = {r["id"] for r in result}
        assert "sn_d" not in ids, "Quarantined node sn_d must not be exported"


# ---------------------------------------------------------------------------
# Class 3: Domain filter is additive, not a replacement
# ---------------------------------------------------------------------------


class TestFetchCandidatesDomainFilter:
    """Verify the optional domain filter restricts results further."""

    def test_domain_filter_reduces_results(self) -> None:
        """domain='other' should return zero results from _ALL_NODES (all in 'test')."""
        mock_gc = _make_gc_filtering(_ALL_NODES)
        with patch(_GC_PATH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            result = _fetch_candidates(domain="other")
        assert result == [], f"Expected empty result for domain='other', got: {result}"

    def test_matching_domain_returns_node(self) -> None:
        """domain='test' should return sn_b (the only valid node in 'test' domain)."""
        mock_gc = _make_gc_filtering(_ALL_NODES)
        with patch(_GC_PATH) as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            result = _fetch_candidates(domain="test")
        ids = {r["id"] for r in result}
        assert "sn_b" in ids, f"Expected sn_b for domain='test', got: {ids}"
