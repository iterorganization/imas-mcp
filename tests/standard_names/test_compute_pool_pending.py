"""Unit tests for ``_compute_pool_pending``.

``_compute_pool_pending`` must mirror the ``claim_*_batch`` predicates exactly.
A drift between the watchdog query and the claim queries causes either
premature exit (undercount) or a stuck-idle watchdog (overcount — the smoke #2 bug).

The function signature is:

    _compute_pool_pending(gc, domains, rotation_cap, min_score) -> dict[str, int]

where *gc* is an open :class:`~imas_codex.graph.client.GraphClient` session.
All tests use a lightweight mock that intercepts ``gc.query``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.cli.sn import _compute_pool_pending

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_POOL_KEYS = frozenset(
    {
        "generate_name",
        "review_name",
        "refine_name",
        "generate_docs",
        "review_docs",
        "refine_docs",
    }
)


def _make_gc(row: dict | None) -> MagicMock:
    """Return a mock gc whose ``query`` returns *row* (or empty list)."""
    gc = MagicMock()
    gc.query.return_value = [row] if row is not None else []
    return gc


# ---------------------------------------------------------------------------
# Basic return-value tests
# ---------------------------------------------------------------------------


class TestReturnShape:
    """The function always returns a dict with exactly the 6 pool keys."""

    def test_keys_are_complete(self) -> None:
        row = dict.fromkeys(_ALL_POOL_KEYS, 0)
        result = _compute_pool_pending(
            _make_gc(row), domains=None, rotation_cap=3, min_score=0.75
        )
        assert set(result.keys()) == _ALL_POOL_KEYS

    def test_empty_query_returns_zeros(self) -> None:
        """When the DB returns no rows, every count must be 0."""
        gc = _make_gc(None)  # empty list
        result = _compute_pool_pending(gc, domains=None, rotation_cap=3, min_score=0.75)
        assert result == dict.fromkeys(_ALL_POOL_KEYS, 0)

    def test_values_are_ints(self) -> None:
        """Values must be plain Python ints (not neo4j Integer wrappers)."""
        row = dict.fromkeys(_ALL_POOL_KEYS, 1)
        result = _compute_pool_pending(
            _make_gc(row), domains=None, rotation_cap=3, min_score=0.75
        )
        for k, v in result.items():
            assert isinstance(v, int), f"Expected int for {k}, got {type(v)}"


class TestCountsPassthrough:
    """The function must faithfully forward the row values from the query."""

    def test_all_pools_non_zero(self) -> None:
        expected = {
            "generate_name": 3,
            "review_name": 2,
            "refine_name": 1,
            "generate_docs": 4,
            "review_docs": 5,
            "refine_docs": 6,
        }
        result = _compute_pool_pending(
            _make_gc(expected), domains=None, rotation_cap=3, min_score=0.75
        )
        assert result == expected

    def test_partial_row_defaults_missing_keys_to_zero(self) -> None:
        """If the DB row omits a key (unlikely but defensive), return 0 for it."""
        partial = {"generate_name": 7, "review_name": 3}
        result = _compute_pool_pending(
            _make_gc(partial), domains=None, rotation_cap=3, min_score=0.75
        )
        # Known keys are forwarded
        assert result["generate_name"] == 7
        assert result["review_name"] == 3
        # Missing keys must be 0, not KeyError
        for k in _ALL_POOL_KEYS - {"generate_name", "review_name"}:
            assert result[k] == 0


# ---------------------------------------------------------------------------
# Parameter-forwarding tests
# ---------------------------------------------------------------------------


class TestParamsForwarding:
    """rotation_cap, min_score, and domains must be forwarded to gc.query as
    keyword arguments so the Cypher parameters bind correctly."""

    def _call_and_get_kwargs(self, **kw) -> dict:
        gc = _make_gc(dict.fromkeys(_ALL_POOL_KEYS, 0))
        _compute_pool_pending(gc, **kw)
        # gc.query is called as: gc.query(cypher_string, **params)
        return gc.query.call_args.kwargs

    def test_rotation_cap_forwarded(self) -> None:
        kwargs = self._call_and_get_kwargs(domains=None, rotation_cap=5, min_score=0.75)
        assert kwargs["rotation_cap"] == 5

    def test_min_score_forwarded(self) -> None:
        kwargs = self._call_and_get_kwargs(domains=None, rotation_cap=3, min_score=0.8)
        assert kwargs["min_score"] == pytest.approx(0.8)

    def test_domains_forwarded_when_set(self) -> None:
        kwargs = self._call_and_get_kwargs(
            domains=["equilibrium", "magnetics"], rotation_cap=3, min_score=0.75
        )
        assert kwargs.get("domains") == ["equilibrium", "magnetics"]

    def test_domains_not_forwarded_when_none(self) -> None:
        """When domains is None, the Cypher template omits the IN clause, so the
        'domains' param must NOT appear in gc.query kwargs (it would be unbound)."""
        kwargs = self._call_and_get_kwargs(domains=None, rotation_cap=3, min_score=0.75)
        assert "domains" not in kwargs

    def test_domains_not_forwarded_when_empty_list(self) -> None:
        """Empty list is treated the same as None — no domain filter."""
        kwargs = self._call_and_get_kwargs(domains=[], rotation_cap=3, min_score=0.75)
        assert "domains" not in kwargs


# ---------------------------------------------------------------------------
# Cypher query structure smoke-test
# ---------------------------------------------------------------------------


class TestQueryStructure:
    """The Cypher passed to gc.query must contain the expected CALL sub-clauses
    and RETURN statement.  A broken edit to the query body would immediately
    surface here without needing a live Neo4j instance."""

    def _get_query_string(self, domains=None) -> str:
        gc = _make_gc(dict.fromkeys(_ALL_POOL_KEYS, 0))
        _compute_pool_pending(gc, domains=domains, rotation_cap=3, min_score=0.75)
        return gc.query.call_args.args[0]

    def test_query_contains_all_pool_returns(self) -> None:
        q = self._get_query_string()
        for key in _ALL_POOL_KEYS:
            assert key in q, f"Expected pool key '{key}' in Cypher query"

    def test_domain_filter_present_when_domains_supplied(self) -> None:
        q = self._get_query_string(domains=["equilibrium"])
        assert "$domains" in q, "Domain IN filter must reference $domains param"

    def test_domain_filter_absent_when_no_domains(self) -> None:
        q = self._get_query_string(domains=None)
        # The filter placeholder should NOT appear because it's only rendered
        # when domains is truthy
        assert "IN $domains" not in q
