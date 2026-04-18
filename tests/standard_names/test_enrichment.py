"""Tests for DD enrichment layer.

Covers primary cluster selection, path enrichment with multi-cluster
deduplication, global grouping by (cluster × unit), and batch splitting.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.enrichment import (
    build_batch_context,
    enrich_paths,
    group_by_concept_and_unit,
    select_primary_cluster,
)
from imas_codex.standard_names.sources.base import ExtractionBatch

# ============================================================================
# Helpers
# ============================================================================


def _make_cluster(
    cluster_id: str = "c1",
    label: str = "Electron temperature",
    description: str = "Temperature of electrons",
    scope: str = "global",
    similarity_score: float | None = None,
) -> dict:
    """Build a cluster dict for testing."""
    d: dict = {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "cluster_description": description,
        "scope": scope,
    }
    if similarity_score is not None:
        d["similarity_score"] = similarity_score
    return d


def _make_row(
    path: str = "core_profiles/profiles_1d/electrons/temperature",
    data_type: str = "FLT_1D",
    unit: str | None = "eV",
    parent_type: str | None = "STRUCTURE",
    description: str = "Electron temperature",
    node_category: str | None = None,
    ids_name: str = "core_profiles",
    cluster_id: str | None = "c1",
    cluster_label: str | None = "Electron temperature",
    cluster_description: str | None = "Temperature of electrons",
    cluster_scope: str | None = "global",
    similarity_score: float | None = None,
    parent_path: str | None = "core_profiles/profiles_1d/electrons",
    parent_description: str | None = "Electrons node",
    cluster_siblings: list | None = None,
) -> dict:
    """Build a raw DD row for testing (as returned by dd.py enriched query)."""
    return {
        "path": path,
        "data_type": data_type,
        "unit": unit,
        "parent_type": parent_type,
        "description": description,
        "node_category": node_category,
        "ids_name": ids_name,
        "cluster_id": cluster_id,
        "cluster_label": cluster_label,
        "cluster_description": cluster_description,
        "cluster_scope": cluster_scope,
        "similarity_score": similarity_score,
        "parent_path": parent_path,
        "parent_description": parent_description,
        "cluster_siblings": cluster_siblings or [],
    }


# ============================================================================
# select_primary_cluster
# ============================================================================


class TestSelectPrimaryCluster:
    """Tests for primary cluster selection logic."""

    def test_empty_returns_none(self):
        assert select_primary_cluster([]) is None

    def test_single_cluster_returned(self):
        c = _make_cluster()
        assert select_primary_cluster([c]) is c

    def test_ids_scope_beats_domain(self):
        domain = _make_cluster("c1", "Domain cluster", scope="domain")
        ids = _make_cluster("c2", "IDS cluster", scope="ids")
        result = select_primary_cluster([domain, ids])
        assert result["cluster_id"] == "c2"

    def test_domain_scope_beats_global(self):
        global_c = _make_cluster("c1", "Global cluster", scope="global")
        domain_c = _make_cluster("c2", "Domain cluster", scope="domain")
        result = select_primary_cluster([global_c, domain_c])
        assert result["cluster_id"] == "c2"

    def test_ids_beats_global(self):
        global_c = _make_cluster("c1", "Global cluster", scope="global")
        ids_c = _make_cluster("c2", "IDS cluster", scope="ids")
        result = select_primary_cluster([global_c, ids_c])
        assert result["cluster_id"] == "c2"

    def test_same_scope_highest_similarity_wins(self):
        low = _make_cluster("c1", "Low score", scope="global", similarity_score=0.3)
        high = _make_cluster("c2", "High score", scope="global", similarity_score=0.9)
        result = select_primary_cluster([low, high])
        assert result["cluster_id"] == "c2"

    def test_same_scope_no_score_sorts_by_label(self):
        """When no similarity scores, deterministic sort by label."""
        beta = _make_cluster("c1", "Beta cluster", scope="global")
        alpha = _make_cluster("c2", "Alpha cluster", scope="global")
        result = select_primary_cluster([beta, alpha])
        assert result["cluster_id"] == "c2"  # Alpha < Beta

    def test_scope_overrides_similarity(self):
        """IDS-scope with low score beats global-scope with high score."""
        global_high = _make_cluster(
            "c1", "Global", scope="global", similarity_score=0.99
        )
        ids_low = _make_cluster("c2", "IDS-scoped", scope="ids", similarity_score=0.1)
        result = select_primary_cluster([global_high, ids_low])
        assert result["cluster_id"] == "c2"

    def test_missing_scope_treated_as_lowest_priority(self):
        no_scope = _make_cluster("c1", "No scope")
        no_scope["scope"] = ""  # simulate missing/empty scope
        global_c = _make_cluster("c2", "Global cluster", scope="global")
        result = select_primary_cluster([no_scope, global_c])
        assert result["cluster_id"] == "c2"

    def test_all_missing_scope_sorts_by_label(self):
        b = _make_cluster("c1", "Beta")
        b["scope"] = ""
        a = _make_cluster("c2", "Alpha")
        a["scope"] = ""
        result = select_primary_cluster([b, a])
        assert result["cluster_id"] == "c2"


# ============================================================================
# enrich_paths
# ============================================================================


class TestEnrichPaths:
    """Tests for path enrichment with multi-cluster deduplication."""

    def test_empty_input(self):
        assert enrich_paths([]) == []

    def test_filters_out_skip_paths(self):
        """STR_0D paths should be classified as skip and excluded."""
        rows = [_make_row(data_type="STR_0D", description="Some name string")]
        result = enrich_paths(rows)
        assert result == []

    def test_filters_out_metadata_paths(self):
        """'time' leaf paths are metadata and should be excluded."""
        rows = [
            _make_row(
                path="core_profiles/profiles_1d/time",
                data_type="FLT_0D",
                description="Time",
            )
        ]
        result = enrich_paths(rows)
        assert result == []

    def test_quantity_path_passes_through(self):
        rows = [_make_row()]
        result = enrich_paths(rows)
        assert len(result) == 1
        assert result[0]["path"] == "core_profiles/profiles_1d/electrons/temperature"

    def test_deduplicates_multi_cluster_rows(self):
        """Path in 2 clusters → 2 raw rows → 1 enriched row."""
        row1 = _make_row(cluster_id="c1", cluster_label="Cluster A")
        row2 = _make_row(cluster_id="c2", cluster_label="Cluster B")
        result = enrich_paths([row1, row2])
        assert len(result) == 1

    def test_all_clusters_collected(self):
        """Enriched row should have all_clusters with both memberships."""
        row1 = _make_row(cluster_id="c1", cluster_label="Cluster A")
        row2 = _make_row(cluster_id="c2", cluster_label="Cluster B")
        result = enrich_paths([row1, row2])
        assert len(result[0]["all_clusters"]) == 2
        ids = {c["cluster_id"] for c in result[0]["all_clusters"]}
        assert ids == {"c1", "c2"}

    def test_primary_cluster_attached(self):
        rows = [_make_row(cluster_id="c1", cluster_label="Electron temperature")]
        result = enrich_paths(rows)
        assert result[0]["primary_cluster_id"] == "c1"
        assert result[0]["primary_cluster_label"] == "Electron temperature"

    def test_no_cluster_gives_none_primary(self):
        rows = [_make_row(cluster_id=None, cluster_label=None)]
        result = enrich_paths(rows)
        assert len(result) == 1
        assert result[0]["primary_cluster_id"] is None
        assert result[0]["primary_cluster_label"] is None

    def test_preserves_enrichment_fields(self):
        """All original fields should be preserved in the enriched row."""
        rows = [
            _make_row(
                ids_name="core_profiles",
                parent_path="core_profiles/profiles_1d/electrons",
                parent_description="Electrons node",
            )
        ]
        result = enrich_paths(rows)
        assert result[0]["ids_name"] == "core_profiles"
        assert result[0]["parent_path"] == "core_profiles/profiles_1d/electrons"
        assert result[0]["parent_description"] == "Electrons node"

    def test_duplicate_cluster_id_not_double_counted(self):
        """Same cluster_id appearing twice (e.g. from coordinate join) → counted once."""
        row1 = _make_row(cluster_id="c1", cluster_label="Same cluster")
        row2 = _make_row(cluster_id="c1", cluster_label="Same cluster")
        result = enrich_paths([row1, row2])
        assert len(result[0]["all_clusters"]) == 1

    def test_mixed_quantity_and_skip(self):
        """Mix of quantity and skip rows: only quantities survive."""
        quantity = _make_row(
            path="equilibrium/time_slice/profiles_1d/psi",
            data_type="FLT_1D",
            unit="Wb",
            description="Poloidal flux",
            ids_name="equilibrium",
        )
        skip = _make_row(
            path="equilibrium/time_slice/profiles_1d/psi_error_upper",
            data_type="FLT_1D",
            unit="Wb",
            description="Upper error on psi",
            ids_name="equilibrium",
        )
        result = enrich_paths([quantity, skip])
        assert len(result) == 1
        assert result[0]["path"] == "equilibrium/time_slice/profiles_1d/psi"

    def test_rows_with_empty_path_skipped(self):
        rows = [_make_row(path=""), _make_row()]
        result = enrich_paths(rows)
        assert len(result) == 1


# ============================================================================
# group_by_concept_and_unit
# ============================================================================


class TestGroupByConceptAndUnit:
    """Tests for global grouping by (primary_cluster × unit)."""

    def _enriched(self, **overrides) -> dict:
        """Build an enriched path dict (post-enrich_paths)."""
        base = {
            "path": "core_profiles/profiles_1d/electrons/temperature",
            "data_type": "FLT_1D",
            "unit": "eV",
            "ids_name": "core_profiles",
            "description": "Electron temperature",
            "primary_cluster_id": "c1",
            "primary_cluster_label": "Electron temperature",
            "primary_cluster_description": "Temperature of electrons",
            "grouping_cluster_id": "c1",
            "grouping_cluster_label": "Electron temperature",
            "parent_path": "core_profiles/profiles_1d/electrons",
            "cluster_siblings": [],
        }
        base.update(overrides)
        return base

    def test_empty_input(self):
        assert group_by_concept_and_unit([]) == []

    def test_same_cluster_same_unit_one_batch(self):
        items = [
            self._enriched(path="a/b/temperature"),
            self._enriched(path="a/b/temp_fit"),
        ]
        batches = group_by_concept_and_unit(items)
        assert len(batches) == 1
        assert len(batches[0].items) == 2

    def test_same_cluster_different_unit_splits(self):
        items = [
            self._enriched(path="a/temperature", unit="eV"),
            self._enriched(path="b/temperature", unit="K"),
        ]
        batches = group_by_concept_and_unit(items)
        assert len(batches) == 2
        units = {b.items[0]["unit"] for b in batches}
        assert units == {"eV", "K"}

    def test_global_grouping_across_ids(self):
        """Paths from different IDSs with same cluster → same batch."""
        items = [
            self._enriched(
                path="core_profiles/profiles_1d/electrons/temperature",
                ids_name="core_profiles",
            ),
            self._enriched(
                path="core_transport/profiles_1d/electrons/temperature",
                ids_name="core_transport",
            ),
        ]
        batches = group_by_concept_and_unit(items)
        assert len(batches) == 1
        ids_names = {item["ids_name"] for item in batches[0].items}
        assert ids_names == {"core_profiles", "core_transport"}

    def test_unclustered_grouped_by_parent(self):
        items = [
            self._enriched(
                path="a/b/field1",
                primary_cluster_id=None,
                primary_cluster_label=None,
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                parent_path="a/b",
            ),
            self._enriched(
                path="a/b/field2",
                primary_cluster_id=None,
                primary_cluster_label=None,
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                parent_path="a/b",
            ),
            self._enriched(
                path="x/y/field3",
                primary_cluster_id=None,
                primary_cluster_label=None,
                grouping_cluster_id=None,
                grouping_cluster_label=None,
                parent_path="x/y",
            ),
        ]
        batches = group_by_concept_and_unit(items)
        # a/b and x/y → separate batches
        assert len(batches) == 2

    def test_oversized_group_split(self):
        items = [self._enriched(path=f"ids/path_{i}") for i in range(60)]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 3  # 60 / 25 → 3 chunks
        total = sum(len(b.items) for b in batches)
        assert total == 60

    def test_oversized_group_key_has_chunk_suffix(self):
        items = [self._enriched(path=f"ids/path_{i}") for i in range(30)]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 2
        assert batches[0].group_key.endswith("#0")
        assert batches[1].group_key.endswith("#1")

    def test_single_chunk_no_suffix(self):
        items = [self._enriched(path=f"ids/path_{i}") for i in range(10)]
        batches = group_by_concept_and_unit(items, max_batch_size=25)
        assert len(batches) == 1
        assert "#" not in batches[0].group_key

    def test_batch_is_extraction_batch(self):
        items = [self._enriched()]
        batches = group_by_concept_and_unit(items)
        assert isinstance(batches[0], ExtractionBatch)
        assert batches[0].source == "dd"

    def test_existing_names_passed_through(self):
        items = [self._enriched()]
        names = {"electron_temperature"}
        batches = group_by_concept_and_unit(items, existing_names=names)
        assert batches[0].existing_names == names

    def test_default_existing_names_empty(self):
        items = [self._enriched()]
        batches = group_by_concept_and_unit(items)
        assert batches[0].existing_names == set()

    def test_context_includes_cluster_label(self):
        items = [self._enriched()]
        batches = group_by_concept_and_unit(items)
        assert "Electron temperature" in batches[0].context

    def test_context_includes_unit(self):
        items = [self._enriched(unit="eV")]
        batches = group_by_concept_and_unit(items)
        assert "eV" in batches[0].context

    def test_context_includes_path_count(self):
        items = [self._enriched(path=f"a/b/{i}") for i in range(3)]
        batches = group_by_concept_and_unit(items)
        assert "3 paths" in batches[0].context

    def test_context_cross_ids_summary(self):
        items = [
            self._enriched(path="a/temp", ids_name="core_profiles"),
            self._enriched(path="b/temp", ids_name="core_transport"),
        ]
        batches = group_by_concept_and_unit(items)
        assert "Cross-IDS" in batches[0].context
        assert "core_profiles" in batches[0].context
        assert "core_transport" in batches[0].context

    def test_dimensionless_unit_label(self):
        items = [self._enriched(unit=None)]
        batches = group_by_concept_and_unit(items)
        assert "dimensionless" in batches[0].context

    def test_different_clusters_different_batches(self):
        items = [
            self._enriched(
                path="a/temperature",
                primary_cluster_label="Electron temperature",
                primary_cluster_id="c1",
            ),
            self._enriched(
                path="b/density",
                primary_cluster_label="Electron density",
                primary_cluster_id="c2",
                unit="m^-3",
            ),
        ]
        batches = group_by_concept_and_unit(items)
        assert len(batches) == 2


# ============================================================================
# build_batch_context
# ============================================================================


class TestBuildBatchContext:
    """Tests for batch context string generation."""

    def test_clustered_context(self):
        items = [
            {
                "unit": "eV",
                "ids_name": "core_profiles",
                "primary_cluster_description": "Temperature of electrons",
                "grouping_cluster_label": "Electron temperature",
                "cluster_siblings": [],
            }
        ]
        ctx = build_batch_context(items, "c1/eV")
        assert "Cluster: Electron temperature" in ctx
        assert "Authoritative unit: eV" in ctx
        assert "1 paths" in ctx
        assert "IDS: core_profiles" in ctx
        assert "Concept: Temperature of electrons" in ctx

    def test_unclustered_context(self):
        items = [
            {
                "unit": "eV",
                "ids_name": "core_profiles",
                "primary_cluster_description": None,
                "cluster_siblings": [],
            }
        ]
        ctx = build_batch_context(items, "unclustered/core_profiles/profiles_1d/eV")
        assert "Unclustered" in ctx
        assert "Parent structure: core_profiles/profiles_1d" in ctx

    def test_siblings_in_context(self):
        items = [
            {
                "unit": "eV",
                "ids_name": "core_profiles",
                "primary_cluster_description": None,
                "primary_cluster_label": "Cluster A",
                "cluster_siblings": [
                    {"path": "sibling/path", "unit": "eV"},
                ],
            }
        ]
        ctx = build_batch_context(items, "c_a/eV")
        assert "sibling/path" in ctx
        assert "Cross-IDS siblings" in ctx

    def test_dimensionless_unit(self):
        items = [
            {
                "unit": None,
                "ids_name": "equilibrium",
                "primary_cluster_description": None,
                "primary_cluster_label": "SomeCluster",
                "cluster_siblings": [],
            }
        ]
        ctx = build_batch_context(items, "c_some/dimensionless")
        assert "dimensionless" in ctx

    def test_multi_ids_context(self):
        items = [
            {
                "unit": "eV",
                "ids_name": "core_profiles",
                "primary_cluster_description": None,
                "primary_cluster_label": "Temperature",
                "cluster_siblings": [],
            },
            {
                "unit": "eV",
                "ids_name": "edge_profiles",
                "primary_cluster_description": None,
                "primary_cluster_label": "Temperature",
                "cluster_siblings": [],
            },
        ]
        ctx = build_batch_context(items, "c_temp/eV")
        assert "Cross-IDS" in ctx
        assert "core_profiles" in ctx
        assert "edge_profiles" in ctx


# ============================================================================
# Integration: full flow
# ============================================================================


class TestIntegration:
    """End-to-end flow: raw multi-cluster rows → batches."""

    def test_full_flow(self):
        """Multi-cluster rows → enrich_paths → group_by_concept_and_unit."""
        # Path in 2 clusters: "Electron temperature" (ids scope) and
        # "Kinetic profiles" (global scope).
        raw_rows = [
            _make_row(
                path="core_profiles/profiles_1d/electrons/temperature",
                cluster_id="c1",
                cluster_label="Electron temperature",
                cluster_scope="ids",
                unit="eV",
                ids_name="core_profiles",
            ),
            _make_row(
                path="core_profiles/profiles_1d/electrons/temperature",
                cluster_id="c2",
                cluster_label="Kinetic profiles",
                cluster_scope="global",
                unit="eV",
                ids_name="core_profiles",
            ),
            # Different path, same global cluster c2
            _make_row(
                path="core_transport/profiles_1d/electrons/temperature",
                cluster_id="c2",
                cluster_label="Kinetic profiles",
                cluster_scope="global",
                unit="eV",
                ids_name="core_transport",
            ),
            # Same path, also in IDS-scope cluster c1
            _make_row(
                path="core_transport/profiles_1d/electrons/temperature",
                cluster_id="c1",
                cluster_label="Electron temperature",
                cluster_scope="ids",
                unit="eV",
                ids_name="core_transport",
            ),
            # Skip path (error field)
            _make_row(
                path="core_profiles/profiles_1d/electrons/temperature_error_upper",
                data_type="FLT_1D",
                cluster_id="c1",
                cluster_label="Electron temperature",
                unit="eV",
                ids_name="core_profiles",
            ),
        ]

        enriched = enrich_paths(raw_rows)
        assert len(enriched) == 2  # 2 unique quantity paths

        # Primary cluster picks IDS-scope c1 (most specific)
        first = next(
            e
            for e in enriched
            if e["path"] == "core_profiles/profiles_1d/electrons/temperature"
        )
        assert first["primary_cluster_id"] == "c1"
        assert first["primary_cluster_label"] == "Electron temperature"
        # Grouping cluster picks global-scope c2 (widest)
        assert first["grouping_cluster_id"] == "c2"

        # Group globally
        batches = group_by_concept_and_unit(enriched)

        # Both paths share global cluster c2 / unit "eV" → single batch
        assert len(batches) == 1
        assert len(batches[0].items) == 2
        assert batches[0].source == "dd"
        # Group key uses global cluster c2 (not IDS-scope c1)
        assert "c2" in batches[0].group_key

    def test_mixed_units_split(self):
        """Same cluster, different units → separate batches."""
        raw_rows = [
            _make_row(
                path="a/temperature_ev",
                cluster_id="c1",
                cluster_label="Temperature",
                unit="eV",
            ),
            _make_row(
                path="b/temperature_k",
                cluster_id="c1",
                cluster_label="Temperature",
                unit="K",
            ),
        ]
        enriched = enrich_paths(raw_rows)
        batches = group_by_concept_and_unit(enriched)
        assert len(batches) == 2

    def test_unclustered_flow(self):
        """Paths without clusters → grouped by parent_path."""
        raw_rows = [
            _make_row(
                path="ids/struct/field_a",
                cluster_id=None,
                cluster_label=None,
                parent_path="ids/struct",
                unit="m",
            ),
            _make_row(
                path="ids/struct/field_b",
                cluster_id=None,
                cluster_label=None,
                parent_path="ids/struct",
                unit="m",
            ),
        ]
        enriched = enrich_paths(raw_rows)
        batches = group_by_concept_and_unit(enriched)
        assert len(batches) == 1
        assert "unclustered" in batches[0].group_key


# =============================================================================
# TestMagneticsDomainReclassification — Fix #4 from D.3 senior review §4.4
# =============================================================================


class TestMagneticsDomainReclassification:
    """Test that magnetics-IDS paths are reclassified to
    ``magnetic_field_diagnostics``."""

    def test_bpol_probe_reclassified(self):
        """magnetics/bpol_probe/* → magnetic_field_diagnostics."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/bpol_probe/field",
            ids_name="magnetics",
            data_type="FLT_1D",
            unit="T",
            description="Poloidal field probe measurement",
        )
        row["physics_domain"] = "equilibrium"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_flux_loop_reclassified(self):
        """magnetics/flux_loop/* → magnetic_field_diagnostics."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/flux_loop/flux/data",
            ids_name="magnetics",
            data_type="FLT_1D",
            unit="Wb",
            description="Flux loop measurement",
        )
        row["physics_domain"] = "general"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_rogowski_coil_reclassified(self):
        """magnetics/rogowski_coil/* → magnetic_field_diagnostics."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/rogowski_coil/current",
            ids_name="magnetics",
            data_type="FLT_0D",
            unit="A",
            description="Rogowski coil current",
        )
        row["physics_domain"] = "equilibrium"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_ip_reclassified(self):
        """magnetics/ip/* → magnetic_field_diagnostics."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/ip/data",
            ids_name="magnetics",
            data_type="FLT_0D",
            unit="A",
            description="Plasma current from magnetics",
        )
        row["physics_domain"] = "equilibrium"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_diamagnetic_flux_reclassified(self):
        """magnetics/diamagnetic_flux/* → magnetic_field_diagnostics."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/diamagnetic_flux/data",
            ids_name="magnetics",
            data_type="FLT_0D",
            unit="Wb",
            description="Diamagnetic flux",
        )
        row["physics_domain"] = "general"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_non_magnetics_ids_not_reclassified(self):
        """equilibrium IDS paths are NOT reclassified."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="equilibrium/time_slice/profiles_1d/psi",
            ids_name="equilibrium",
            data_type="FLT_1D",
            unit="Wb",
            description="Poloidal flux",
        )
        row["physics_domain"] = "equilibrium"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "equilibrium"

    def test_already_correct_domain_unchanged(self):
        """Paths already in magnetic_field_diagnostics stay unchanged."""
        from imas_codex.standard_names.enrichment import reclassify_magnetics_domain

        row = _make_row(
            path="magnetics/bpol_probe/field",
            ids_name="magnetics",
            data_type="FLT_1D",
            unit="T",
            description="Poloidal field probe measurement",
        )
        row["physics_domain"] = "magnetic_field_diagnostics"
        reclassify_magnetics_domain(row)
        assert row["physics_domain"] == "magnetic_field_diagnostics"

    def test_enrich_paths_reclassifies_magnetics(self):
        """enrich_paths() applies domain reclassification for magnetics."""
        rows = [
            _make_row(
                path="magnetics/bpol_probe/field",
                ids_name="magnetics",
                data_type="FLT_1D",
                unit="T",
                description="Poloidal field probe measurement",
            ),
        ]
        rows[0]["physics_domain"] = "equilibrium"

        enriched = enrich_paths(rows)
        assert len(enriched) == 1
        assert enriched[0]["physics_domain"] == "magnetic_field_diagnostics"
