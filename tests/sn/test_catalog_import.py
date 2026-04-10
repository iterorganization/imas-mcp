"""Tests for the catalog feedback import module.

Tests YAML parsing, grammar field derivation, tag filtering, dry-run
behavior, and graph write semantics — all mocked, no live Neo4j.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

imas_sn = pytest.importorskip("imas_standard_names")


# =============================================================================
# Fixtures
# =============================================================================

SAMPLE_CATALOG_ENTRY = {
    "name": "electron_temperature",
    "description": "Electron temperature",
    "documentation": "The electron temperature Te is measured by Thomson scattering.",
    "kind": "scalar",
    "unit": "eV",
    "tags": [],
    "links": [],
    "ids_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    "validity_domain": "core plasma",
    "constraints": ["T_e > 0"],
    "physics_domain": "core_plasma_physics",
    "status": "active",
}

SAMPLE_CATALOG_ENTRY_MINIMAL = {
    "name": "plasma_current",
    "description": "Plasma current",
    "documentation": "Total toroidal plasma current.",
    "kind": "scalar",
    "unit": "A",
    "tags": [],
    "links": [],
    "ids_paths": [],
    "validity_domain": "",
    "constraints": [],
    "physics_domain": "equilibrium",
    "status": "active",
}


@pytest.fixture()
def catalog_dir(tmp_path: Path) -> Path:
    """Create a temporary catalog directory with sample YAML files."""
    d = tmp_path / "catalog"
    d.mkdir()
    (d / "electron_temperature.yaml").write_text(yaml.safe_dump(SAMPLE_CATALOG_ENTRY))
    (d / "plasma_current.yaml").write_text(yaml.safe_dump(SAMPLE_CATALOG_ENTRY_MINIMAL))
    return d


@pytest.fixture()
def catalog_dir_with_tags(tmp_path: Path) -> Path:
    """Create a catalog directory with tagged entries."""
    d = tmp_path / "catalog_tagged"
    d.mkdir()

    entry_tagged = {**SAMPLE_CATALOG_ENTRY, "tags": ["spatial-profile"]}
    entry_untagged = {**SAMPLE_CATALOG_ENTRY_MINIMAL, "tags": []}

    (d / "electron_temperature.yaml").write_text(yaml.safe_dump(entry_tagged))
    (d / "plasma_current.yaml").write_text(yaml.safe_dump(entry_untagged))
    return d


# =============================================================================
# YAML parsing tests
# =============================================================================


class TestImportParsesYaml:
    """Test that import correctly parses YAML catalog files."""

    def test_parses_yaml_files(self, catalog_dir: Path) -> None:
        """Should parse all valid YAML files in the directory."""
        from imas_codex.sn.catalog_import import import_catalog

        with patch(
            "imas_codex.sn.catalog_import._write_catalog_entries", return_value=2
        ):
            result = import_catalog(catalog_dir, dry_run=False)

        assert result.imported == 2
        assert len(result.errors) == 0

    def test_parses_yml_extension(self, tmp_path: Path) -> None:
        """Should handle .yml file extension too."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "electron_temperature.yml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY)
        )

        from imas_codex.sn.catalog_import import import_catalog

        with patch(
            "imas_codex.sn.catalog_import._write_catalog_entries", return_value=1
        ):
            result = import_catalog(d, dry_run=False)

        assert result.imported == 1

    def test_recursive_subdirectories(self, tmp_path: Path) -> None:
        """Should walk subdirectories recursively."""
        d = tmp_path / "catalog"
        sub = d / "scalars"
        sub.mkdir(parents=True)
        (sub / "electron_temperature.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY)
        )

        from imas_codex.sn.catalog_import import import_catalog

        with patch(
            "imas_codex.sn.catalog_import._write_catalog_entries", return_value=1
        ):
            result = import_catalog(d, dry_run=False)

        assert result.imported == 1


# =============================================================================
# Grammar field derivation tests
# =============================================================================


class TestGrammarFields:
    """Test that grammar fields are derived from name parsing."""

    def test_derives_grammar_fields(self) -> None:
        """Should extract subject and physical_base from name."""
        from imas_codex.sn.catalog_import import _parse_grammar_fields

        fields = _parse_grammar_fields("electron_temperature")
        assert fields["subject"] == "electron"
        assert fields["physical_base"] == "temperature"

    def test_unparseable_name_returns_none(self) -> None:
        """Should return None fields for names that can't be parsed."""
        from imas_codex.sn.catalog_import import _parse_grammar_fields

        # Mock the grammar parser to raise, simulating an unparseable name
        with patch(
            "imas_standard_names.grammar.parse_standard_name",
            side_effect=ValueError("bad name"),
        ):
            fields = _parse_grammar_fields("__broken__")

        assert fields["physical_base"] is None
        assert fields["subject"] is None

    def test_grammar_fields_in_import_output(self, catalog_dir: Path) -> None:
        """Imported entries should have grammar fields populated."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        # Find the electron_temperature entry
        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["subject"] == "electron"
        assert et_entry["physical_base"] == "temperature"


# =============================================================================
# Import status and field mapping tests
# =============================================================================


class TestFieldMapping:
    """Test that catalog fields are correctly mapped to graph dict."""

    def test_sets_accepted_status(self, catalog_dir: Path) -> None:
        """All imported entries should have review_status='accepted'."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        for entry in result.entries:
            assert entry["review_status"] == "accepted"

    def test_maps_unit_to_units(self, catalog_dir: Path) -> None:
        """Catalog 'unit' field should map to graph 'units' key."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["units"] == "eV"
        assert "unit" not in et_entry  # should not have the catalog key

    def test_maps_ids_paths_to_imas_paths(self, catalog_dir: Path) -> None:
        """Catalog 'ids_paths' should map to graph 'imas_paths' key."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["imas_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]

    def test_source_type_dd_for_entries_with_paths(self, catalog_dir: Path) -> None:
        """Entries with ids_paths should have source_type='dd'."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["source_type"] == "dd"

    def test_source_type_manual_for_entries_without_paths(
        self, catalog_dir: Path
    ) -> None:
        """Entries without ids_paths should have source_type='manual'."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        pc_entry = next(e for e in result.entries if e["id"] == "plasma_current")
        assert pc_entry["source_type"] == "manual"

    def test_maps_kind(self, catalog_dir: Path) -> None:
        """Catalog 'kind' field should be mapped correctly."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        for entry in result.entries:
            assert entry["kind"] == "scalar"

    def test_empty_lists_become_none(self, catalog_dir: Path) -> None:
        """Empty catalog lists should become None for graph coalesce compatibility."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        # plasma_current has no ids_paths, empty tags, empty links
        pc_entry = next(e for e in result.entries if e["id"] == "plasma_current")
        assert pc_entry["imas_paths"] is None
        assert pc_entry["tags"] is None
        assert pc_entry["links"] is None

    def test_maps_physics_domain(self, catalog_dir: Path) -> None:
        """Catalog 'physics_domain' field should be mapped."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["physics_domain"] == "core_plasma_physics"


# =============================================================================
# Dry run tests
# =============================================================================


class TestDryRun:
    """Test that dry run mode doesn't write to graph."""

    def test_dry_run_no_write(self, catalog_dir: Path) -> None:
        """Dry run should not call the write function."""
        from imas_codex.sn.catalog_import import import_catalog

        with patch("imas_codex.sn.catalog_import._write_catalog_entries") as mock_write:
            result = import_catalog(catalog_dir, dry_run=True)

        mock_write.assert_not_called()
        assert result.imported == 2  # still reports count
        assert len(result.entries) == 2

    def test_non_dry_run_calls_write(self, catalog_dir: Path) -> None:
        """Non-dry-run should call the write function."""
        from imas_codex.sn.catalog_import import import_catalog

        with patch(
            "imas_codex.sn.catalog_import._write_catalog_entries", return_value=2
        ) as mock_write:
            result = import_catalog(catalog_dir, dry_run=False)

        mock_write.assert_called_once()
        assert result.imported == 2


# =============================================================================
# Tag filter tests
# =============================================================================


class TestTagFilter:
    """Test tag-based filtering of catalog entries."""

    def test_tag_filter_includes_matching(self, catalog_dir_with_tags: Path) -> None:
        """Should import entries whose tags match the filter."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(
            catalog_dir_with_tags, dry_run=True, tag_filter=["spatial-profile"]
        )

        assert result.imported == 1
        assert result.entries[0]["id"] == "electron_temperature"

    def test_tag_filter_skips_non_matching(self, catalog_dir_with_tags: Path) -> None:
        """Should skip entries that don't match the tag filter."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(
            catalog_dir_with_tags, dry_run=True, tag_filter=["spatial-profile"]
        )

        assert result.skipped == 1

    def test_no_tag_filter_imports_all(self, catalog_dir_with_tags: Path) -> None:
        """Without tag filter, all entries should be imported."""
        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(catalog_dir_with_tags, dry_run=True)

        assert result.imported == 2
        assert result.skipped == 0


# =============================================================================
# Error handling tests
# =============================================================================


class TestErrorHandling:
    """Test graceful error handling."""

    def test_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Should report errors for invalid YAML files without crashing."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "bad.yaml").write_text(": : : invalid yaml [[[")

        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "bad.yaml" in result.errors[0]

    def test_handles_non_mapping_yaml(self, tmp_path: Path) -> None:
        """Should report errors for YAML files that aren't dicts."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "list.yaml").write_text(yaml.safe_dump(["a", "b", "c"]))

        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "not a YAML mapping" in result.errors[0]

    def test_handles_incomplete_entry(self, tmp_path: Path) -> None:
        """Should report errors for entries missing required fields."""
        d = tmp_path / "catalog"
        d.mkdir()
        incomplete = {"name": "test", "kind": "scalar"}  # missing required fields
        (d / "incomplete.yaml").write_text(yaml.safe_dump(incomplete))

        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "incomplete.yaml" in result.errors[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty result."""
        d = tmp_path / "empty"
        d.mkdir()

        from imas_codex.sn.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert result.skipped == 0
        assert len(result.errors) == 0
        assert len(result.entries) == 0


# =============================================================================
# Graph write semantics tests
# =============================================================================


class TestWriteCatalogEntries:
    """Test that _write_catalog_entries produces correct Cypher."""

    def _call_write(self, entries: list[dict], mock_gc: MagicMock) -> int:
        """Call _write_catalog_entries with a mocked GraphClient."""
        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            from imas_codex.sn.catalog_import import _write_catalog_entries

            return _write_catalog_entries(entries)

    def test_catalog_fields_overwrite(self) -> None:
        """Catalog-owned fields should use direct SET, not coalesce."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "electron_temperature",
                "description": "Te profile",
                "documentation": "Rich docs here",
                "kind": "scalar",
                "units": "eV",
                "tags": ["core"],
                "links": None,
                "imas_paths": None,
                "validity_domain": "core",
                "constraints": None,
                "physics_domain": "core_plasma_physics",
                "review_status": "accepted",
                "source_type": "dd",
                "physical_base": "temperature",
                "subject": "electron",
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
            }
        ]
        self._call_write(entries, mock_gc)

        merge_call = mock_gc.query.call_args_list[0]
        cypher = merge_call[0][0]

        # Catalog-owned fields should NOT use coalesce — direct SET
        assert "sn.description = b.description" in cypher
        assert "sn.documentation = b.documentation" in cypher
        assert "sn.kind = b.kind" in cypher
        assert "sn.tags = b.tags" in cypher
        assert "sn.validity_domain = b.validity_domain" in cypher
        assert "sn.physical_base = b.physical_base" in cypher

        # review_status should be hardcoded to 'accepted'
        assert "sn.review_status = 'accepted'" in cypher

        # imported_at should be set
        assert "sn.imported_at = datetime()" in cypher

        # Graph-only fields should use coalesce (preserve existing)
        assert "coalesce(sn.embedding" in cypher
        assert "coalesce(sn.model" in cypher
        assert "coalesce(sn.generated_at" in cypher
        assert "coalesce(sn.created_at, datetime())" in cypher

    def test_unit_relationship_created(self) -> None:
        """Entries with units should create CANONICAL_UNITS relationship."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "electron_temperature",
                "description": "Te",
                "documentation": None,
                "kind": "scalar",
                "units": "eV",
                "tags": None,
                "links": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": None,
                "review_status": "accepted",
                "source_type": "dd",
                "physical_base": "temperature",
                "subject": "electron",
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
            }
        ]
        self._call_write(entries, mock_gc)

        unit_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "CANONICAL_UNITS" in str(call)
        ]
        assert len(unit_calls) >= 1
        unit_cypher = unit_calls[0][0][0]
        assert "MERGE (u:Unit" in unit_cypher
        assert "MERGE (sn)-[:CANONICAL_UNITS]->(u)" in unit_cypher

    def test_dd_relationship_from_imas_paths(self) -> None:
        """Entries with imas_paths should create HAS_STANDARD_NAME from IMASNode."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "electron_temperature",
                "description": "Te",
                "documentation": None,
                "kind": "scalar",
                "units": "eV",
                "tags": None,
                "links": None,
                "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                "validity_domain": None,
                "constraints": None,
                "physics_domain": None,
                "review_status": "accepted",
                "source_type": "dd",
                "physical_base": "temperature",
                "subject": "electron",
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
            }
        ]
        self._call_write(entries, mock_gc)

        dd_calls = [
            call
            for call in mock_gc.query.call_args_list
            if "HAS_STANDARD_NAME" in str(call)
        ]
        assert len(dd_calls) >= 1
        dd_cypher = dd_calls[0][0][0]
        assert "IMASNode" in dd_cypher
        assert "MERGE (src)-[:HAS_STANDARD_NAME]->(sn)" in dd_cypher

    def test_no_relationships_for_empty_fields(self) -> None:
        """Entries without units/imas_paths should not create those relationships."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "test_name",
                "description": "Test",
                "documentation": None,
                "kind": "scalar",
                "units": None,
                "tags": None,
                "links": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": None,
                "review_status": "accepted",
                "source_type": "manual",
                "physical_base": None,
                "subject": None,
                "component": None,
                "coordinate": None,
                "position": None,
                "process": None,
            }
        ]
        self._call_write(entries, mock_gc)

        # Should only have the MERGE query — no unit or relationship queries
        assert mock_gc.query.call_count == 1  # just the MERGE

    def test_empty_list_returns_zero(self) -> None:
        """Empty list should return 0 without touching the graph."""
        from imas_codex.sn.catalog_import import _write_catalog_entries

        result = _write_catalog_entries([])
        assert result == 0
