"""Tests for ``imas_codex.standard_names.domain_ranking``."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import domain_ranking as dr


@pytest.fixture(autouse=True)
def _force_fallback_ranks(monkeypatch):
    """Force tests to use the deterministic FALLBACK_RANK_TABLE.

    Bypasses graph-derived priority index by stubbing it to ``{}``.
    """
    monkeypatch.setattr(dr, "get_domain_priority_index", lambda: {})


class TestDomainRank:
    def test_known_domain_returns_table_index(self):
        assert dr.domain_rank("equilibrium") == 0

    def test_more_central_has_lower_rank(self):
        assert dr.domain_rank("equilibrium") < dr.domain_rank("transport")
        assert dr.domain_rank("transport") < dr.domain_rank("general")

    def test_unknown_domain_returns_unknown_rank(self):
        assert dr.domain_rank("__not_a_real_domain__") == dr.UNKNOWN_RANK

    def test_none_returns_unknown_rank(self):
        assert dr.domain_rank(None) == dr.UNKNOWN_RANK

    def test_empty_string_returns_unknown_rank(self):
        assert dr.domain_rank("") == dr.UNKNOWN_RANK

    def test_general_is_lowest_priority(self):
        # 'general' is intentionally the most-generic domain → highest rank
        # of any *known* value.
        general_rank = dr.domain_rank("general")
        for d in dr.FALLBACK_RANK_TABLE:
            if d == "general":
                continue
            assert dr.domain_rank(d) < general_rank


class TestMaybePromoteDomain:
    @pytest.mark.parametrize(
        "current,candidate,expected",
        [
            # Empty handling
            (None, None, None),
            (None, "equilibrium", "equilibrium"),
            ("equilibrium", None, "equilibrium"),
            ("", "equilibrium", "equilibrium"),
            ("equilibrium", "", "equilibrium"),
            # Promotion: candidate has lower rank → promote
            ("transport", "equilibrium", "equilibrium"),
            ("general", "magnetohydrodynamics", "magnetohydrodynamics"),
            ("plasma_control", "equilibrium", "equilibrium"),
            # No promotion: candidate has higher rank → keep current
            ("equilibrium", "transport", "equilibrium"),
            ("magnetohydrodynamics", "general", "magnetohydrodynamics"),
            # Tie: keep current
            ("equilibrium", "equilibrium", "equilibrium"),
            # Unknown current, known candidate → promote
            ("__unknown__", "equilibrium", "equilibrium"),
            # Known current, unknown candidate → keep
            ("equilibrium", "__unknown__", "equilibrium"),
        ],
    )
    def test_promotion(self, current, candidate, expected):
        assert dr.maybe_promote_domain(current, candidate) == expected

    def test_promotion_is_idempotent(self):
        # Applying twice is a no-op.
        once = dr.maybe_promote_domain("transport", "equilibrium")
        twice = dr.maybe_promote_domain(once, "equilibrium")
        assert once == twice == "equilibrium"

    def test_promotion_is_monotonic(self):
        # A chain of attaches only ever moves toward more central domains.
        d = "general"
        d = dr.maybe_promote_domain(d, "transport")  # promote
        assert d == "transport"
        d = dr.maybe_promote_domain(d, "general")  # no-op
        assert d == "transport"
        d = dr.maybe_promote_domain(d, "equilibrium")  # promote further
        assert d == "equilibrium"
        d = dr.maybe_promote_domain(d, "transport")  # no-op
        assert d == "equilibrium"


class TestMergeSourceDomains:
    def test_empty_inputs(self):
        assert dr.merge_source_domains(None) == []
        assert dr.merge_source_domains([]) == []
        assert dr.merge_source_domains(None, None) == []

    def test_appends_new(self):
        assert dr.merge_source_domains(["equilibrium"], "transport") == [
            "equilibrium",
            "transport",
        ]

    def test_dedupes(self):
        assert dr.merge_source_domains(["equilibrium"], "equilibrium", "transport") == [
            "equilibrium",
            "transport",
        ]

    def test_preserves_existing_order(self):
        # Existing order is sticky; new items appear at the end.
        assert dr.merge_source_domains(["transport", "equilibrium"], "magnetics") == [
            "transport",
            "equilibrium",
            "magnetics",
        ]

    def test_skips_empty_and_none(self):
        assert dr.merge_source_domains(["equilibrium"], None, "", "transport") == [
            "equilibrium",
            "transport",
        ]


class TestGraphPriorityWins:
    """When the graph-derived index is non-empty, it wins over the fallback."""

    def test_graph_index_overrides_fallback(self, monkeypatch):
        # Reverse the table: 'general' is now most-central per graph.
        monkeypatch.setattr(
            dr,
            "get_domain_priority_index",
            lambda: {"general": 0, "equilibrium": 5},
        )
        assert dr.domain_rank("general") == 0
        assert dr.domain_rank("equilibrium") == 5
        # And promotion uses the graph-derived order.
        assert dr.maybe_promote_domain("equilibrium", "general") == "general"
