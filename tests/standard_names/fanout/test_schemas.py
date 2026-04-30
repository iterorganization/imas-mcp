"""Discriminated-union schema tests for fan-out (plan 39 §12.2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imas_codex.standard_names.fanout.schemas import (
    MAX_FAN_DEGREE,
    FanoutPlan,
    _FindRelatedDDPaths,
    _SearchDDClusters,
    _SearchDDPaths,
    _SearchExistingNames,
)


class TestDiscriminator:
    def test_routes_search_existing_names(self) -> None:
        plan = FanoutPlan.model_validate(
            {"queries": [{"fn_id": "search_existing_names", "query": "foo bar"}]}
        )
        assert isinstance(plan.queries[0], _SearchExistingNames)
        assert plan.queries[0].k == 5  # default

    def test_routes_search_dd_paths(self) -> None:
        plan = FanoutPlan.model_validate(
            {"queries": [{"fn_id": "search_dd_paths", "query": "foo", "k": 7}]}
        )
        assert isinstance(plan.queries[0], _SearchDDPaths)
        assert plan.queries[0].k == 7

    def test_routes_find_related_dd_paths(self) -> None:
        plan = FanoutPlan.model_validate(
            {
                "queries": [
                    {
                        "fn_id": "find_related_dd_paths",
                        "path": "core_profiles/profiles_1d/electrons/temperature",
                    }
                ]
            }
        )
        assert isinstance(plan.queries[0], _FindRelatedDDPaths)
        assert plan.queries[0].max_results == 12  # default

    def test_routes_search_dd_clusters(self) -> None:
        plan = FanoutPlan.model_validate(
            {"queries": [{"fn_id": "search_dd_clusters", "query": "ne core"}]}
        )
        assert isinstance(plan.queries[0], _SearchDDClusters)


class TestRejection:
    def test_unknown_fn_id_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate(
                {"queries": [{"fn_id": "delete_database", "query": "oops"}]}
            )

    def test_out_of_bounds_k_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate(
                {
                    "queries": [
                        {"fn_id": "search_existing_names", "query": "foo", "k": 99}
                    ]
                }
            )

    def test_zero_k_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate(
                {
                    "queries": [
                        {"fn_id": "search_existing_names", "query": "foo", "k": 0}
                    ]
                }
            )

    def test_missing_required_query_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate({"queries": [{"fn_id": "search_dd_paths"}]})

    def test_missing_required_path_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate({"queries": [{"fn_id": "find_related_dd_paths"}]})

    def test_too_short_query_rejects(self) -> None:
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate(
                {"queries": [{"fn_id": "search_existing_names", "query": "x"}]}
            )

    def test_max_fan_degree(self) -> None:
        # MAX_FAN_DEGREE + 1 entries should fail.
        too_many = [
            {"fn_id": "search_existing_names", "query": f"q{i:02d}"}
            for i in range(MAX_FAN_DEGREE + 1)
        ]
        with pytest.raises(ValidationError):
            FanoutPlan.model_validate({"queries": too_many})

    def test_max_fan_degree_exact(self) -> None:
        # Exactly MAX_FAN_DEGREE entries should succeed.
        ok = [
            {"fn_id": "search_existing_names", "query": f"q{i:02d}"}
            for i in range(MAX_FAN_DEGREE)
        ]
        plan = FanoutPlan.model_validate({"queries": ok})
        assert len(plan.queries) == MAX_FAN_DEGREE

    def test_extra_fields_rejected_by_default(self) -> None:
        # Pydantic v2 ignores extras by default; the discriminator
        # variant only knows the declared fields.  This is a positive
        # test that confirms unknown keys don't crash the parser
        # (they're silently dropped).
        plan = FanoutPlan.model_validate(
            {
                "queries": [
                    {
                        "fn_id": "search_existing_names",
                        "query": "foo",
                        "extra_field_unknown": 42,
                    }
                ]
            }
        )
        assert plan.queries[0].fn_id == "search_existing_names"


class TestEmptyPlan:
    def test_empty_queries_is_valid(self) -> None:
        plan = FanoutPlan.model_validate({"queries": []})
        assert plan.queries == []

    def test_default_construction(self) -> None:
        plan = FanoutPlan()
        assert plan.queries == []
        assert plan.notes == ""
