"""Tests for plan 40: per-domain catalog layout with graph-hierarchy ordering.

Tests 1–15 from plan 40 §6:
1. Round-trip byte stability
2. Round-trip idempotence
3. Ordering — unary prefix
4. Ordering — projection
5. Ordering — binary
6. Ordering — uncertainty
7. Ordering — mixed
8. Ordering — orphan (cross-domain)
9. Stability under cluster reassignment
10. Stability under Neo4j property permutation
11. Computed-field ignored on import + INFO log
12. Partial-export publish safety + manifest mismatch abort
13. check_catalog + list-root parity
14. Legacy per-file rejection
15. Edge-model version guard
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ============================================================================
# Test 3: Ordering — unary prefix
# ============================================================================


class TestOrderingUnaryPrefix:
    """Unary prefix: base first, then alpha-sorted wrappers."""

    def test_unary_prefix_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "minimum_of_temperature"},
            {"name": "maximum_of_temperature"},
            {"name": "temperature"},
        ]
        # HAS_ARGUMENT: wrapper -> base (base is ordering-parent)
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_ARGUMENT"),
            ("minimum_of_temperature", "temperature", "HAS_ARGUMENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result]
        assert names == [
            "temperature",
            "maximum_of_temperature",
            "minimum_of_temperature",
        ]


# ============================================================================
# Test 4: Ordering — projection
# ============================================================================


class TestOrderingProjection:
    """Projection: base first, then alpha-sorted components."""

    def test_projection_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "z_component_of_magnetic_field"},
            {"name": "magnetic_field"},
            {"name": "x_component_of_magnetic_field"},
            {"name": "y_component_of_magnetic_field"},
        ]
        edges = [
            ("x_component_of_magnetic_field", "magnetic_field", "HAS_ARGUMENT"),
            ("y_component_of_magnetic_field", "magnetic_field", "HAS_ARGUMENT"),
            ("z_component_of_magnetic_field", "magnetic_field", "HAS_ARGUMENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result]
        assert names == [
            "magnetic_field",
            "x_component_of_magnetic_field",
            "y_component_of_magnetic_field",
            "z_component_of_magnetic_field",
        ]


# ============================================================================
# Test 5: Ordering — binary
# ============================================================================


class TestOrderingBinary:
    """Binary: both args as clean roots, ratio last."""

    def test_binary_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "ratio_of_pressure_to_density"},
            {"name": "pressure"},
            {"name": "density"},
        ]
        # Binary has two HAS_ARGUMENT edges
        edges = [
            ("ratio_of_pressure_to_density", "pressure", "HAS_ARGUMENT"),
            ("ratio_of_pressure_to_density", "density", "HAS_ARGUMENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result]
        # Alpha tie-break: density before pressure
        assert names == [
            "density",
            "pressure",
            "ratio_of_pressure_to_density",
        ]


# ============================================================================
# Test 6: Ordering — uncertainty
# ============================================================================


class TestOrderingUncertainty:
    """Uncertainty: base first, then alpha-sorted error siblings."""

    def test_uncertainty_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "upper_uncertainty_of_temperature"},
            {"name": "temperature"},
            {"name": "uncertainty_index_of_temperature"},
            {"name": "lower_uncertainty_of_temperature"},
        ]
        # HAS_ERROR: base -> error sibling (base is ordering-parent)
        edges = [
            ("temperature", "upper_uncertainty_of_temperature", "HAS_ERROR"),
            ("temperature", "lower_uncertainty_of_temperature", "HAS_ERROR"),
            ("temperature", "uncertainty_index_of_temperature", "HAS_ERROR"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result]
        assert names == [
            "temperature",
            "lower_uncertainty_of_temperature",
            "uncertainty_index_of_temperature",
            "upper_uncertainty_of_temperature",
        ]


# ============================================================================
# Test 7: Ordering — mixed
# ============================================================================


class TestOrderingMixed:
    """Mixed: base first, then alpha-sorted variants + components."""

    def test_mixed_order(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "upper_uncertainty_of_temperature"},
            {"name": "x_component_of_temperature"},
            {"name": "temperature"},
            {"name": "maximum_of_temperature"},
        ]
        edges = [
            ("temperature", "upper_uncertainty_of_temperature", "HAS_ERROR"),
            ("x_component_of_temperature", "temperature", "HAS_ARGUMENT"),
            ("maximum_of_temperature", "temperature", "HAS_ARGUMENT"),
        ]

        result = order_entries_by_hierarchy(entries, edges)
        names = [e["name"] for e in result]

        # temperature first (sole root), then all children alpha-sorted
        assert names[0] == "temperature"
        assert set(names[1:]) == {
            "maximum_of_temperature",
            "upper_uncertainty_of_temperature",
            "x_component_of_temperature",
        }
        # Alpha-sorted among children
        assert names[1:] == sorted(names[1:])


# ============================================================================
# Test 8: Ordering — orphan (cross-domain)
# ============================================================================


class TestOrderingOrphan:
    """Orphan: cross-domain wrapper lands after all clean-roots."""

    def test_orphan_after_clean_roots(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "alpha_base"},
            {"name": "cross_domain_wrapper"},
            {"name": "beta_base"},
        ]
        # No in-domain edges, but cross_domain_wrapper has a parent outside
        edges: list[tuple[str, str, str]] = []
        cross_domain = {"cross_domain_wrapper"}

        result = order_entries_by_hierarchy(
            entries, edges, cross_domain_parent_ids=cross_domain
        )
        names = [e["name"] for e in result]

        # Clean roots first (alpha_base, beta_base), then orphan
        assert names == ["alpha_base", "beta_base", "cross_domain_wrapper"]


# ============================================================================
# Test 9: Stability under cluster reassignment
# ============================================================================


class TestStabilityClusterReassignment:
    """Ordering is stable when primary_cluster_id changes."""

    def test_cluster_reassignment_no_effect(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries_v1 = [
            {"name": "temperature", "primary_cluster_id": "cluster_A"},
            {"name": "maximum_of_temperature", "primary_cluster_id": "cluster_A"},
        ]
        entries_v2 = [
            {"name": "temperature", "primary_cluster_id": "cluster_B"},
            {"name": "maximum_of_temperature", "primary_cluster_id": "cluster_B"},
        ]
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_ARGUMENT"),
        ]

        result_v1 = [e["name"] for e in order_entries_by_hierarchy(entries_v1, edges)]
        result_v2 = [e["name"] for e in order_entries_by_hierarchy(entries_v2, edges)]

        assert result_v1 == result_v2


# ============================================================================
# Test 10: Stability under Neo4j property permutation
# ============================================================================


class TestStabilityPropertyPermutation:
    """Ordering is stable regardless of dict key order."""

    def test_property_permutation_no_effect(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            order_entries_by_hierarchy,
        )

        entries_v1 = [
            {"name": "temperature", "kind": "scalar", "unit": "eV"},
            {"name": "maximum_of_temperature", "kind": "scalar", "unit": "eV"},
        ]
        # Same data, different insertion order
        entries_v2 = [
            {"unit": "eV", "name": "temperature", "kind": "scalar"},
            {"unit": "eV", "kind": "scalar", "name": "maximum_of_temperature"},
        ]
        edges = [
            ("maximum_of_temperature", "temperature", "HAS_ARGUMENT"),
        ]

        result_v1 = [e["name"] for e in order_entries_by_hierarchy(entries_v1, edges)]
        result_v2 = [e["name"] for e in order_entries_by_hierarchy(entries_v2, edges)]

        assert result_v1 == result_v2


# ============================================================================
# Test 1: Round-trip byte stability
# ============================================================================


class TestRoundTripByteStability:
    """Export entries, parse, re-emit → byte-identical."""

    def test_byte_stable_round_trip(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.canonical import (
            canonicalise_entry,
            reorder_entry_dict,
        )
        from imas_codex.standard_names.export import _write_domain_yaml

        entries = [
            {
                "name": "temperature",
                "kind": "scalar",
                "status": "draft",
                "description": "Temperature profile",
                "documentation": "A temperature measurement.",
                "unit": "eV",
                "physics_domain": "kinetics",
                "tags": ["time-dependent"],
                "links": [],
                "constraints": [],
            },
        ]

        # Write domain file
        _write_domain_yaml(tmp_path, "kinetics", entries, codex_sha="abc123")

        filepath = tmp_path / "standard_names" / "kinetics.yml"
        assert filepath.exists()

        # Parse back
        text = filepath.read_text(encoding="utf-8")
        # Skip header comments
        yaml_text = "\n".join(
            line for line in text.splitlines() if not line.startswith("#")
        )
        parsed = yaml.safe_load(yaml_text)
        assert isinstance(parsed, list)

        # Re-emit through canonical pipeline
        re_emitted = []
        for entry in parsed:
            canon = canonicalise_entry(entry)
            clean = {k: v for k, v in canon.items() if v is not None}
            ordered = reorder_entry_dict(clean)
            re_emitted.append(ordered)

        re_emitted_yaml = yaml.safe_dump(
            re_emitted, sort_keys=False, default_flow_style=False
        )

        # Extract YAML body from original (after header comments)
        original_yaml = yaml_text.strip() + "\n"
        assert re_emitted_yaml == original_yaml


# ============================================================================
# Test 2: Round-trip idempotence (mock-based)
# ============================================================================


class TestRoundTripIdempotence:
    """Export → parse → re-emit yields identical entries."""

    def test_idempotent_re_emit(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.canonical import (
            canonicalise_entry,
            reorder_entry_dict,
        )
        from imas_codex.standard_names.export import _write_domain_yaml

        entries = [
            {
                "name": "electron_temperature",
                "kind": "scalar",
                "status": "draft",
                "description": "Electron temperature",
                "documentation": "Te from Thomson scattering.",
                "unit": "eV",
                "physics_domain": "kinetics",
                "tags": ["core_profiles"],
                "links": ["name:ion_temperature"],
                "constraints": ["T_e > 0"],
                "validity_domain": "core plasma",
            },
        ]

        # First write
        _write_domain_yaml(tmp_path, "kinetics", entries, codex_sha="sha1")
        fp = tmp_path / "standard_names" / "kinetics.yml"
        first_bytes = fp.read_bytes()

        # Parse back and re-write
        text = fp.read_text(encoding="utf-8")
        yaml_text = "\n".join(
            line for line in text.splitlines() if not line.startswith("#")
        )
        parsed = yaml.safe_load(yaml_text)

        _write_domain_yaml(tmp_path, "kinetics", parsed, codex_sha="sha1")
        second_bytes = fp.read_bytes()

        assert first_bytes == second_bytes


# ============================================================================
# Test 11: Computed-field ignored on import + INFO log
# ============================================================================


class TestComputedFieldIgnoredOnImport:
    """Import ignores arguments/error_variants and logs INFO."""

    def test_computed_fields_stripped_with_log(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pytest.importorskip("imas_standard_names")

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        sn_dir = catalog_dir / "standard_names"
        sn_dir.mkdir()

        entry_with_computed = [
            {
                "name": "temperature",
                "description": "Temperature",
                "documentation": "A temperature.",
                "kind": "scalar",
                "unit": "eV",
                "status": "active",
                "tags": [],
                "links": [],
                "arguments": [{"name": "base", "operator": "identity"}],
                "error_variants": {"upper": "upper_uncertainty_of_temperature"},
            }
        ]
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump(entry_with_computed))

        from imas_codex.standard_names.catalog_import import run_import

        with caplog.at_level(
            logging.INFO, logger="imas_codex.standard_names.catalog_import"
        ):
            report = run_import(catalog_dir, dry_run=True)

        # Check that computed fields were logged
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            "computed field" in m.lower() or "Ignoring" in m for m in info_messages
        )

        # The entries should have been parsed (arguments stripped)
        if report.entries:
            for e in report.entries:
                assert "arguments" not in e
                assert "error_variants" not in e


# ============================================================================
# Test 12: Partial-export publish safety + manifest mismatch abort
# ============================================================================


class TestPartialExportPublishSafety:
    """Publish aborts on manifest mismatch."""

    def test_manifest_domain_mismatch_aborts(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()

        # Write one domain file
        (sn_dir / "transport.yml").write_text(yaml.safe_dump([{"name": "flux"}]))

        # Manifest claims full scope but only one domain
        manifest = {
            "catalog_name": "imas-standard-names-catalog",
            "export_scope": "full",
            "domains_included": ["transport", "magnetics"],
            "edge_model_version": "plan_39_v1",
        }
        (staging / "catalog.yml").write_text(yaml.safe_dump(manifest))

        # Create a fake ISNC git repo
        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()

        report = run_publish(staging, isnc)
        assert report.errors
        assert any("domain mismatch" in e.lower() for e in report.errors)

    def test_domain_subset_only_touches_listed(self, tmp_path: Path) -> None:
        """Domain-subset publish only copies listed domain files."""
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()

        (sn_dir / "transport.yml").write_text(yaml.safe_dump([{"name": "flux"}]))

        manifest = {
            "catalog_name": "imas-standard-names-catalog",
            "export_scope": "domain_subset",
            "domains_included": ["transport"],
            "edge_model_version": "plan_39_v1",
        }
        (staging / "catalog.yml").write_text(yaml.safe_dump(manifest))

        # Set up ISNC with existing domain files
        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()
        isnc_sn = isnc / "standard_names"
        isnc_sn.mkdir()
        (isnc_sn / "magnetics.yml").write_text("- name: b_field\n")

        # Mock git operations
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_publish(staging, isnc, dry_run=True)

        # magnetics.yml should still exist (not touched)
        assert (isnc_sn / "magnetics.yml").exists()


# ============================================================================
# Test 13: check_catalog + list-root parity
# ============================================================================


class TestCheckCatalogListRoot:
    """check_catalog handles list-root files correctly."""

    def test_check_catalog_list_root(self, tmp_path: Path) -> None:
        pytest.importorskip("imas_standard_names")

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        sn_dir = catalog_dir / "standard_names"
        sn_dir.mkdir()

        entries = [
            {
                "name": "temperature",
                "description": "Temperature",
                "documentation": "A temperature measurement.",
                "kind": "scalar",
                "unit": "eV",
                "status": "active",
                "tags": [],
                "links": [],
            },
        ]
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump(entries))

        from imas_codex.standard_names.catalog_import import check_catalog

        # GraphClient is imported inside check_catalog — patch at source module.
        with patch("imas_codex.graph.client.GraphClient") as mock_gc_cls:
            mock_gc = MagicMock()
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            mock_gc.query = MagicMock(return_value=[])
            mock_gc_cls.return_value = mock_gc

            result = check_catalog(catalog_dir)

        # Should have parsed the entry
        assert result.only_in_catalog == ["temperature"]


# ============================================================================
# Test 14: Legacy per-file rejection
# ============================================================================


class TestLegacyPerFileRejection:
    """Top-level dict YAML file rejected with migration error."""

    def test_dict_root_rejected(self, tmp_path: Path) -> None:
        pytest.importorskip("imas_standard_names")

        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()
        sn_dir = catalog_dir / "standard_names"
        sn_dir.mkdir()

        # Legacy per-file format: single dict
        legacy_entry = {
            "name": "temperature",
            "description": "Temperature",
            "documentation": "A temperature.",
            "kind": "scalar",
            "unit": "eV",
            "status": "active",
            "tags": [],
            "links": [],
        }
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump(legacy_entry))

        from imas_codex.standard_names.catalog_import import run_import

        report = run_import(catalog_dir, dry_run=True)

        assert report.errors
        assert any(
            "top-level YAML dict" in e or "per-file layout" in e.lower()
            for e in report.errors
        )
        assert report.imported == 0


# ============================================================================
# Test 15: Edge-model version guard
# ============================================================================


class TestEdgeModelVersionGuard:
    """Manifest with wrong edge_model_version rejected by publish."""

    def test_wrong_version_rejected(self, tmp_path: Path) -> None:
        from imas_codex.standard_names.publish import run_publish

        staging = tmp_path / "staging"
        staging.mkdir()
        sn_dir = staging / "standard_names"
        sn_dir.mkdir()
        (sn_dir / "kinetics.yml").write_text(yaml.safe_dump([{"name": "temperature"}]))

        manifest = {
            "catalog_name": "imas-standard-names-catalog",
            "export_scope": "full",
            "domains_included": ["kinetics"],
            "edge_model_version": "plan_39_v0",
        }
        (staging / "catalog.yml").write_text(yaml.safe_dump(manifest))

        isnc = tmp_path / "isnc"
        isnc.mkdir()
        (isnc / ".git").mkdir()

        report = run_publish(staging, isnc)
        assert report.errors
        assert any("edge_model_version" in e for e in report.errors)


# ============================================================================
# Canonical key order tests
# ============================================================================


class TestCanonicalKeyOrder:
    """CANONICAL_KEY_ORDER and reorder_entry_dict."""

    def test_reorder_known_keys(self) -> None:
        from imas_codex.standard_names.canonical import reorder_entry_dict

        entry = {
            "unit": "eV",
            "name": "temperature",
            "kind": "scalar",
            "status": "draft",
        }
        result = reorder_entry_dict(entry)
        assert list(result.keys()) == ["name", "kind", "status", "unit"]

    def test_unknown_key_raises(self) -> None:
        from imas_codex.standard_names.canonical import (
            UnknownCatalogKeyError,
            reorder_entry_dict,
        )

        entry = {"name": "temperature", "bogus_key": "value"}
        with pytest.raises(UnknownCatalogKeyError, match="bogus_key"):
            reorder_entry_dict(entry)

    def test_missing_keys_omitted(self) -> None:
        from imas_codex.standard_names.canonical import reorder_entry_dict

        entry = {"name": "temperature"}
        result = reorder_entry_dict(entry)
        assert result == {"name": "temperature"}


# ============================================================================
# Ordering error detection
# ============================================================================


class TestOrderingCycleDetection:
    """OrderingError raised for cycles."""

    def test_cycle_raises(self) -> None:
        from imas_codex.standard_names.catalog_ordering import (
            OrderingError,
            order_entries_by_hierarchy,
        )

        entries = [
            {"name": "a"},
            {"name": "b"},
        ]
        # Mutual dependency → cycle
        edges = [
            ("a", "b", "HAS_ARGUMENT"),
            ("b", "a", "HAS_ARGUMENT"),
        ]

        with pytest.raises(OrderingError, match="unemitted"):
            order_entries_by_hierarchy(entries, edges)
