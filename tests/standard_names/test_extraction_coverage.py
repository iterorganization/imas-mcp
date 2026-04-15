"""Tests for extraction coverage gap fixes."""

from __future__ import annotations

import inspect

import pytest

from imas_codex.standard_names.enrichment import (
    group_by_concept_and_unit,
    select_grouping_cluster,
    select_primary_cluster,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cluster(
    cluster_id: str = "c1",
    label: str = "Test Cluster",
    scope: str = "global",
    similarity_score: float | None = None,
) -> dict:
    d: dict = {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "cluster_description": f"{label} description",
        "scope": scope,
    }
    if similarity_score is not None:
        d["similarity_score"] = similarity_score
    return d


def _make_enriched(
    path: str = "core_profiles/profiles_1d/electrons/temperature",
    unit: str = "eV",
    ids_name: str = "core_profiles",
    grouping_cluster_id: str | None = "c1",
    grouping_cluster_label: str | None = "Electron temperature",
    primary_cluster_id: str | None = "c1",
    primary_cluster_label: str | None = "Electron temperature",
    parent_path: str = "core_profiles/profiles_1d/electrons",
    **extra,
) -> dict:
    """Build an enriched path dict suitable for group_by_concept_and_unit."""
    return {
        "path": path,
        "data_type": "FLT_1D",
        "unit": unit,
        "ids_name": ids_name,
        "description": "Test quantity",
        "primary_cluster_id": primary_cluster_id,
        "primary_cluster_label": primary_cluster_label,
        "primary_cluster_description": None,
        "grouping_cluster_id": grouping_cluster_id,
        "grouping_cluster_label": grouping_cluster_label,
        "parent_path": parent_path,
        "cluster_siblings": [],
        **extra,
    }


# ---------------------------------------------------------------------------
# Gap 4: Reversed grouping cluster priority (global-first)
# ---------------------------------------------------------------------------


class TestGroupingClusterSelection:
    """select_grouping_cluster prioritises global > domain > ids."""

    def test_global_preferred_over_ids(self):
        clusters = [
            _make_cluster("c1", "IDS Cluster", scope="ids"),
            _make_cluster("c2", "Global Cluster", scope="global"),
        ]
        result = select_grouping_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_global_preferred_over_domain(self):
        clusters = [
            _make_cluster("c1", "Domain", scope="domain"),
            _make_cluster("c2", "Global", scope="global"),
        ]
        result = select_grouping_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_domain_preferred_over_ids(self):
        clusters = [
            _make_cluster("c1", "IDS", scope="ids"),
            _make_cluster("c2", "Domain", scope="domain"),
        ]
        result = select_grouping_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_single_cluster_returned(self):
        clusters = [_make_cluster("c1", "Only", scope="ids")]
        assert select_grouping_cluster(clusters)["cluster_id"] == "c1"

    def test_empty_returns_none(self):
        assert select_grouping_cluster([]) is None

    def test_all_global_highest_similarity_wins(self):
        clusters = [
            _make_cluster("c1", "A", scope="global", similarity_score=0.7),
            _make_cluster("c2", "B", scope="global", similarity_score=0.9),
        ]
        result = select_grouping_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_scope_overrides_similarity(self):
        """Lower-scoring global cluster still beats higher-scoring IDS cluster."""
        clusters = [
            _make_cluster("c1", "High IDS", scope="ids", similarity_score=0.99),
            _make_cluster("c2", "Low Global", scope="global", similarity_score=0.1),
        ]
        result = select_grouping_cluster(clusters)
        assert result["cluster_id"] == "c2"


# ---------------------------------------------------------------------------
# Primary cluster selection: IDS-first (most specific)
# ---------------------------------------------------------------------------


class TestPrimaryClusterSelection:
    """select_primary_cluster prioritises ids > domain > global."""

    def test_ids_preferred_over_global(self):
        clusters = [
            _make_cluster("c1", "Global", scope="global"),
            _make_cluster("c2", "IDS", scope="ids"),
        ]
        result = select_primary_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_ids_preferred_over_domain(self):
        clusters = [
            _make_cluster("c1", "Domain", scope="domain"),
            _make_cluster("c2", "IDS", scope="ids"),
        ]
        result = select_primary_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_domain_preferred_over_global(self):
        clusters = [
            _make_cluster("c1", "Global", scope="global"),
            _make_cluster("c2", "Domain", scope="domain"),
        ]
        result = select_primary_cluster(clusters)
        assert result["cluster_id"] == "c2"

    def test_empty_returns_none(self):
        assert select_primary_cluster([]) is None

    def test_single_cluster_returned(self):
        c = _make_cluster("c1", "Solo", scope="domain")
        assert select_primary_cluster([c]) is c


# ---------------------------------------------------------------------------
# Gap 5: Enhanced unclustered grouping includes IDS name
# ---------------------------------------------------------------------------


class TestUnclusteredGrouping:
    """Unclustered paths use IDS name in the group key."""

    def test_unclustered_key_includes_ids(self):
        items = [
            _make_enriched(
                path="equilibrium/time_slice/profiles_1d/psi",
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                primary_cluster_id=None,
                primary_cluster_label=None,
                unit="Wb",
                ids_name="equilibrium",
                parent_path="equilibrium/time_slice/profiles_1d",
            )
        ]
        batches = group_by_concept_and_unit(items)
        assert len(batches) == 1
        assert "equilibrium" in batches[0].group_key

    def test_unclustered_key_includes_unclustered_prefix(self):
        items = [
            _make_enriched(
                path="magnetics/flux_loop/flux/data",
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                primary_cluster_id=None,
                primary_cluster_label=None,
                ids_name="magnetics",
                parent_path="magnetics/flux_loop/flux",
            )
        ]
        batches = group_by_concept_and_unit(items)
        assert batches[0].group_key.startswith("unclustered/")

    def test_different_ids_unclustered_same_parent_suffix_different_batches(self):
        """Two unclustered paths with different IDS names → different batches."""
        items = [
            _make_enriched(
                path="eq/profiles/psi",
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                primary_cluster_id=None,
                primary_cluster_label=None,
                ids_name="equilibrium",
                parent_path="eq/profiles",
            ),
            _make_enriched(
                path="cp/profiles/psi",
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                primary_cluster_id=None,
                primary_cluster_label=None,
                ids_name="core_profiles",
                parent_path="cp/profiles",
            ),
        ]
        batches = group_by_concept_and_unit(items)
        keys = {b.group_key for b in batches}
        assert len(keys) == 2


# ---------------------------------------------------------------------------
# Gap 2: Pre-LIMIT exclusion controlled by force flag
# ---------------------------------------------------------------------------


class TestExtractDdForceParameter:
    """extract_dd_candidates accepts a force parameter."""

    def test_force_parameter_exists(self):
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        sig = inspect.signature(extract_dd_candidates)
        assert "force" in sig.parameters

    def test_force_default_is_false(self):
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        sig = inspect.signature(extract_dd_candidates)
        assert sig.parameters["force"].default is False

    def test_limit_parameter_exists(self):
        """Extraction query has a LIMIT parameter to bound path count."""
        from imas_codex.standard_names.sources.dd import extract_dd_candidates

        sig = inspect.signature(extract_dd_candidates)
        assert "limit" in sig.parameters
