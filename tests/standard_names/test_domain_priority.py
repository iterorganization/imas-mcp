"""Tests for ``imas_codex.standard_names.domain_priority``.

The priority index is derived from ``Cluster.mapping_relevance`` weighted
counts. We mock ``GraphClient`` so the tests don't touch the live graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import domain_priority


@pytest.fixture(autouse=True)
def _reset_cache():
    domain_priority.reset_cache()
    yield
    domain_priority.reset_cache()


def _mock_graph_client(rows: list[dict]):
    """Return a context-manager mock for GraphClient yielding ``rows``."""
    gc = MagicMock()
    gc.query.return_value = iter(rows)
    cm = MagicMock()
    cm.__enter__.return_value = gc
    cm.__exit__.return_value = False
    return cm


def test_priority_high_outranks_medium():
    """A domain with HIGH-relevance clusters ranks above one with MEDIUM."""
    # equilibrium has 3 high (300) + 5 medium (50) = 350
    # transport has 0 high + 1 medium (10) = 10
    rows = [
        {"domain": "equilibrium", "importance": 350},
        {"domain": "transport", "importance": 10},
    ]
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client(rows)
    ):
        idx = domain_priority.get_domain_priority_index()

    assert idx["equilibrium"] == 0
    assert idx["transport"] == 1
    assert (
        domain_priority.pick_primary_domain(["transport", "equilibrium"])
        == "equilibrium"
    )


def test_unranked_domain_falls_back_to_alphabetical():
    """Domains absent from the index get rank 999 — alphabetical wins
    among unranked."""
    rows = [{"domain": "equilibrium", "importance": 100}]
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client(rows)
    ):
        # zoo and apple both unranked → alphabetical
        assert domain_priority.pick_primary_domain(["zoo", "apple"]) == "apple"
        # equilibrium known → wins over unranked
        assert (
            domain_priority.pick_primary_domain(["zoo", "equilibrium"]) == "equilibrium"
        )


def test_empty_graph_falls_back_to_alphabetical():
    """When no cluster has mapping_relevance, all domains tie at 999 and
    we fall through to alphabetical."""
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client([])
    ):
        assert (
            domain_priority.pick_primary_domain(["transport", "equilibrium"])
            == "equilibrium"
        )


def test_graph_unreachable_returns_empty_index():
    """If GraphClient raises, log a warning and return an empty dict."""
    with patch.object(
        domain_priority,
        "GraphClient",
        side_effect=RuntimeError("connection refused"),
    ):
        idx = domain_priority.get_domain_priority_index()
    assert idx == {}


def test_pick_primary_empty_list_raises():
    with pytest.raises(ValueError):
        domain_priority.pick_primary_domain([])


def test_cache_query_is_one_shot():
    """Subsequent calls do NOT re-query the graph (lru_cache)."""
    rows = [{"domain": "equilibrium", "importance": 100}]
    cm = _mock_graph_client(rows)
    with patch.object(domain_priority, "GraphClient", return_value=cm):
        domain_priority.get_domain_priority_index()
        # second call — cache hit, GraphClient should NOT be re-entered
        domain_priority.get_domain_priority_index()
    # __enter__ called exactly once
    assert cm.__enter__.call_count == 1


def test_tied_importance_breaks_alphabetical():
    """Two domains with equal importance — alphabetical wins as
    deterministic tie-break (the Cypher already orders by domain ASC)."""
    rows = [
        {"domain": "equilibrium", "importance": 100},
        {"domain": "magnetics", "importance": 100},
    ]
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client(rows)
    ):
        idx = domain_priority.get_domain_priority_index()
    # Cypher returned equilibrium first (alphabetical ASC tie-break)
    assert idx["equilibrium"] == 0
    assert idx["magnetics"] == 1
    assert (
        domain_priority.pick_primary_domain(["magnetics", "equilibrium"])
        == "equilibrium"
    )


# ---------------------------------------------------------------------------
# domain_key / domain_list helpers (multivalued physics_domain coercion)
# ---------------------------------------------------------------------------


def test_domain_key_none_returns_fallback():
    assert domain_priority.domain_key(None) == "unknown"
    assert domain_priority.domain_key(None, fallback="x") == "x"


def test_domain_key_empty_string_returns_fallback():
    assert domain_priority.domain_key("") == "unknown"
    assert domain_priority.domain_key("   ") == "unknown"


def test_domain_key_string_returns_trimmed():
    assert domain_priority.domain_key("equilibrium") == "equilibrium"
    assert domain_priority.domain_key("  transport  ") == "transport"


def test_domain_key_list_picks_priority(_reset_cache=None):
    rows = [
        {"domain": "equilibrium", "importance": 100},
        {"domain": "transport", "importance": 10},
    ]
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client(rows)
    ):
        # Higher-priority domain wins regardless of input order.
        assert domain_priority.domain_key(["transport", "equilibrium"]) == "equilibrium"


def test_domain_key_list_unranked_alphabetical_fallback():
    """When priority index is empty, lists fall back to alphabetical-first."""
    with patch.object(
        domain_priority, "GraphClient", return_value=_mock_graph_client([])
    ):
        assert (
            domain_priority.domain_key(["transport", "equilibrium", "magnetics"])
            == "equilibrium"
        )


def test_domain_key_list_with_blanks_returns_fallback():
    assert domain_priority.domain_key(["", None, "  "]) == "unknown"


def test_domain_key_unexpected_type_returns_fallback():
    assert domain_priority.domain_key(42) == "unknown"
    assert domain_priority.domain_key({"x": 1}) == "unknown"


def test_domain_list_none_empty():
    assert domain_priority.domain_list(None) == []
    assert domain_priority.domain_list("") == []
    assert domain_priority.domain_list("   ") == []


def test_domain_list_string_promoted_to_list():
    assert domain_priority.domain_list("transport") == ["transport"]
    assert domain_priority.domain_list("  equilibrium  ") == ["equilibrium"]


def test_domain_list_filters_blanks():
    assert domain_priority.domain_list(["a", "", "b", None, " ", "c"]) == [
        "a",
        "b",
        "c",
    ]
