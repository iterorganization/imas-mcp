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
        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=2,
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

        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=1,
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

        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=1,
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
        from imas_codex.standard_names.catalog_import import _parse_grammar_fields

        fields = _parse_grammar_fields("electron_temperature")
        assert fields["subject"] == "electron"
        assert fields["physical_base"] == "temperature"

    def test_unparseable_name_returns_none(self) -> None:
        """Should return None fields for names that can't be parsed."""
        from imas_codex.standard_names.catalog_import import _parse_grammar_fields

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
        from imas_codex.standard_names.catalog_import import import_catalog

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
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        for entry in result.entries:
            assert entry["review_status"] == "accepted"

    def test_maps_unit_to_unit(self, catalog_dir: Path) -> None:
        """Catalog 'unit' field should pass through as graph unit key."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["unit"] == "eV"

    def test_maps_ids_paths_to_imas_paths(self, catalog_dir: Path) -> None:
        """Catalog 'ids_paths' should map to graph 'imas_paths' key."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["imas_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]

    def test_source_type_dd_for_entries_with_paths(self, catalog_dir: Path) -> None:
        """Entries with ids_paths should have source_type='dd'."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        et_entry = next(e for e in result.entries if e["id"] == "electron_temperature")
        assert et_entry["source_type"] == "dd"

    def test_source_type_manual_for_entries_without_paths(
        self, catalog_dir: Path
    ) -> None:
        """Entries without ids_paths should have source_type='manual'."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        pc_entry = next(e for e in result.entries if e["id"] == "plasma_current")
        assert pc_entry["source_type"] == "manual"

    def test_maps_kind(self, catalog_dir: Path) -> None:
        """Catalog 'kind' field should be mapped correctly."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        for entry in result.entries:
            assert entry["kind"] == "scalar"

    def test_empty_lists_become_none(self, catalog_dir: Path) -> None:
        """Empty catalog lists should become None for graph coalesce compatibility."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(catalog_dir, dry_run=True)

        # plasma_current has no ids_paths, empty tags, empty links
        pc_entry = next(e for e in result.entries if e["id"] == "plasma_current")
        assert pc_entry["imas_paths"] is None
        assert pc_entry["tags"] is None
        assert pc_entry["links"] is None

    def test_maps_physics_domain(self, catalog_dir: Path) -> None:
        """Catalog 'physics_domain' field should be mapped."""
        from imas_codex.standard_names.catalog_import import import_catalog

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
        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries"
        ) as mock_write:
            result = import_catalog(catalog_dir, dry_run=True)

        mock_write.assert_not_called()
        assert result.imported == 2  # still reports count
        assert len(result.entries) == 2

    def test_non_dry_run_calls_write(self, catalog_dir: Path) -> None:
        """Non-dry-run should call the write function."""
        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=2,
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
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(
            catalog_dir_with_tags, dry_run=True, tag_filter=["spatial-profile"]
        )

        assert result.imported == 1
        assert result.entries[0]["id"] == "electron_temperature"

    def test_tag_filter_skips_non_matching(self, catalog_dir_with_tags: Path) -> None:
        """Should skip entries that don't match the tag filter."""
        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(
            catalog_dir_with_tags, dry_run=True, tag_filter=["spatial-profile"]
        )

        assert result.skipped == 1

    def test_no_tag_filter_imports_all(self, catalog_dir_with_tags: Path) -> None:
        """Without tag filter, all entries should be imported."""
        from imas_codex.standard_names.catalog_import import import_catalog

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

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "bad.yaml" in result.errors[0]

    def test_handles_non_mapping_yaml(self, tmp_path: Path) -> None:
        """Should report errors for YAML files that aren't dicts."""
        d = tmp_path / "catalog"
        d.mkdir()
        (d / "list.yaml").write_text(yaml.safe_dump(["a", "b", "c"]))

        from imas_codex.standard_names.catalog_import import import_catalog

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

        from imas_codex.standard_names.catalog_import import import_catalog

        result = import_catalog(d, dry_run=True)

        assert result.imported == 0
        assert len(result.errors) == 1
        assert "incomplete.yaml" in result.errors[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty result."""
        d = tmp_path / "empty"
        d.mkdir()

        from imas_codex.standard_names.catalog_import import import_catalog

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

            from imas_codex.standard_names.catalog_import import _write_catalog_entries

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
                "unit": "eV",
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
                "unit": "eV",
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
                "unit": "eV",
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
                "unit": None,
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
        from imas_codex.standard_names.catalog_import import _write_catalog_entries

        result = _write_catalog_entries([])
        assert result == 0


# =============================================================================
# Phase 2: Version tracking tests
# =============================================================================


class TestResolveCatalogSha:
    """Tests for _resolve_catalog_sha()."""

    def test_returns_sha_in_git_repo(self, tmp_path: Path) -> None:
        """Should return a 40-char SHA when run in a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        # Use the project repo itself as the catalog dir
        project_root = Path(__file__).resolve().parents[2]
        sha = _resolve_catalog_sha(project_root)
        assert sha is not None
        assert len(sha) == 40
        assert all(c in "0123456789abcdef" for c in sha)

    def test_returns_none_for_non_git_dir(self, tmp_path: Path) -> None:
        """Should return None for a directory that isn't a git repo."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        sha = _resolve_catalog_sha(tmp_path)
        assert sha is None

    def test_returns_none_when_git_not_found(self, tmp_path: Path) -> None:
        """Should return None when git binary is missing."""
        from imas_codex.standard_names.catalog_import import _resolve_catalog_sha

        with patch(
            "imas_codex.standard_names.catalog_import.subprocess.run"
        ) as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            sha = _resolve_catalog_sha(tmp_path)
            assert sha is None


class TestVersionTracking:
    """Tests for catalog_commit_sha propagation through the import pipeline."""

    def test_sha_in_cypher_batch(self) -> None:
        """_write_catalog_entries should inject catalog_commit_sha into each entry."""
        from imas_codex.standard_names.catalog_import import _write_catalog_entries

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "electron_temperature",
                "description": "Te",
                "documentation": None,
                "kind": "scalar",
                "unit": "eV",
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

        test_sha = "abc123def456" * 3 + "abcd"  # 40 chars

        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            _write_catalog_entries(entries, catalog_commit_sha=test_sha)

        # Verify the SHA was injected into the entry dicts
        assert entries[0]["catalog_commit_sha"] == test_sha

        # Verify the Cypher includes catalog_commit_sha
        merge_call = mock_gc.query.call_args_list[0]
        cypher = merge_call[0][0]
        assert "catalog_commit_sha" in cypher

    def test_sha_none_when_not_provided(self) -> None:
        """When no SHA is provided, entries should get catalog_commit_sha=None."""
        from imas_codex.standard_names.catalog_import import _write_catalog_entries

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        entries = [
            {
                "id": "test_name",
                "description": "Test",
                "documentation": None,
                "kind": "scalar",
                "unit": None,
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

        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            _write_catalog_entries(entries)

        assert entries[0]["catalog_commit_sha"] is None

    def test_import_result_contains_sha(self, catalog_dir: Path) -> None:
        """import_catalog() should populate catalog_commit_sha on the result."""
        from imas_codex.standard_names.catalog_import import import_catalog

        test_sha = "a" * 40

        with (
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=test_sha,
            ),
            patch(
                "imas_codex.standard_names.catalog_import._write_catalog_entries",
                return_value=2,
            ),
        ):
            result = import_catalog(catalog_dir=catalog_dir)

        assert result.catalog_commit_sha == test_sha

    def test_import_result_sha_none_for_non_git(self, catalog_dir: Path) -> None:
        """import_catalog() should have sha=None when dir is not a git repo."""
        from imas_codex.standard_names.catalog_import import import_catalog

        with (
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
            patch(
                "imas_codex.standard_names.catalog_import._write_catalog_entries",
                return_value=2,
            ),
        ):
            result = import_catalog(catalog_dir=catalog_dir)

        assert result.catalog_commit_sha is None


class TestImportIdempotency:
    """Tests that re-importing the same catalog produces identical results."""

    def test_double_import_same_entries(self, catalog_dir: Path) -> None:
        """Importing the same catalog twice should produce same entry count."""
        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=2,
        ) as mock_write:
            r1 = import_catalog(catalog_dir=catalog_dir)
            r2 = import_catalog(catalog_dir=catalog_dir)

        assert r1.imported == r2.imported
        assert len(r1.entries) == len(r2.entries)
        # Both calls should produce identical entry dicts
        for e1, e2 in zip(r1.entries, r2.entries, strict=False):
            assert e1["id"] == e2["id"]
            assert e1["description"] == e2["description"]
        assert mock_write.call_count == 2

    def test_idempotent_entry_dicts(self, catalog_dir: Path) -> None:
        """Entry dicts from two imports of the same catalog should be identical."""
        from imas_codex.standard_names.catalog_import import import_catalog

        with patch(
            "imas_codex.standard_names.catalog_import._write_catalog_entries",
            return_value=2,
        ):
            r1 = import_catalog(catalog_dir=catalog_dir)
            r2 = import_catalog(catalog_dir=catalog_dir)

        # Compare each field (excluding mutable fields like catalog_commit_sha)
        for e1, e2 in zip(r1.entries, r2.entries, strict=False):
            for key in (
                "id",
                "description",
                "documentation",
                "kind",
                "unit",
                "tags",
                "imas_paths",
                "validity_domain",
                "constraints",
                "physics_domain",
                "review_status",
                "source_type",
                "physical_base",
                "subject",
                "component",
                "coordinate",
                "position",
                "process",
            ):
                assert e1[key] == e2[key], f"Mismatch on field '{key}'"


class TestCheckMode:
    """Tests for check_catalog() — the --check sync comparison."""

    def test_all_in_sync(self, catalog_dir: Path) -> None:
        """When graph matches catalog exactly, in_sync should equal entry count."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Build graph rows that match catalog exactly
        graph_rows = [
            {
                "id": "electron_temperature",
                "description": "Electron temperature",
                "documentation": "The electron temperature Te is measured by Thomson scattering.",
                "kind": "scalar",
                "unit": "eV",
                "tags": None,
                "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "physics_domain": "core_plasma_physics",
                "catalog_commit_sha": "a" * 40,
            },
            {
                "id": "plasma_current",
                "description": "Plasma current",
                "documentation": "Total toroidal plasma current.",
                "kind": "scalar",
                "unit": "A",
                "tags": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": "equilibrium",
                "catalog_commit_sha": "a" * 40,
            },
        ]

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=graph_rows)

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value="a" * 40,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert cr.in_sync == 2
        assert cr.only_in_catalog == []
        assert cr.only_in_graph == []
        assert cr.diverged == []
        assert cr.catalog_commit_sha == "a" * 40
        assert cr.graph_commit_sha == "a" * 40

    def test_only_in_catalog(self, catalog_dir: Path) -> None:
        """Entries in catalog but not graph should appear in only_in_catalog."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Graph has no entries
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert set(cr.only_in_catalog) == {"electron_temperature", "plasma_current"}
        assert cr.in_sync == 0
        assert cr.only_in_graph == []

    def test_only_in_graph(self, catalog_dir: Path) -> None:
        """Entries in graph but not catalog should appear in only_in_graph."""
        from imas_codex.standard_names.catalog_import import check_catalog

        graph_rows = [
            {
                "id": "electron_temperature",
                "description": "Electron temperature",
                "documentation": "The electron temperature Te is measured by Thomson scattering.",
                "kind": "scalar",
                "unit": "eV",
                "tags": None,
                "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "physics_domain": "core_plasma_physics",
                "catalog_commit_sha": None,
            },
            {
                "id": "plasma_current",
                "description": "Plasma current",
                "documentation": "Total toroidal plasma current.",
                "kind": "scalar",
                "unit": "A",
                "tags": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": "equilibrium",
                "catalog_commit_sha": None,
            },
            {
                "id": "ion_density",
                "description": "Ion density",
                "documentation": "Total ion density.",
                "kind": "scalar",
                "unit": "m^-3",
                "tags": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": "core_plasma_physics",
                "catalog_commit_sha": None,
            },
        ]

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=graph_rows)

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert cr.only_in_graph == ["ion_density"]
        assert cr.in_sync == 2  # electron_temperature and plasma_current match

    def test_diverged_entries(self, catalog_dir: Path) -> None:
        """Entries with different field values should appear in diverged."""
        from imas_codex.standard_names.catalog_import import check_catalog

        graph_rows = [
            {
                "id": "electron_temperature",
                "description": "WRONG description",  # differs from catalog
                "documentation": "The electron temperature Te is measured by Thomson scattering.",
                "kind": "scalar",
                "unit": "keV",  # differs from catalog (eV)
                "tags": None,
                "imas_paths": ["core_profiles/profiles_1d/electrons/temperature"],
                "validity_domain": "core plasma",
                "constraints": ["T_e > 0"],
                "physics_domain": "core_plasma_physics",
                "catalog_commit_sha": None,
            },
            {
                "id": "plasma_current",
                "description": "Plasma current",
                "documentation": "Total toroidal plasma current.",
                "kind": "scalar",
                "unit": "A",
                "tags": None,
                "imas_paths": None,
                "validity_domain": None,
                "constraints": None,
                "physics_domain": "equilibrium",
                "catalog_commit_sha": None,
            },
        ]

        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=graph_rows)

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(catalog_dir=catalog_dir)

        assert cr.in_sync == 1  # plasma_current matches
        assert len(cr.diverged) == 1
        assert cr.diverged[0]["name"] == "electron_temperature"
        assert "description" in cr.diverged[0]["fields"]
        assert "unit" in cr.diverged[0]["fields"]

    def test_check_with_tag_filter(self, tmp_path: Path) -> None:
        """Tag filter should limit which catalog entries are checked."""
        from imas_codex.standard_names.catalog_import import check_catalog

        # Create catalog with tagged entry
        d = tmp_path / "catalog"
        d.mkdir()
        entry_with_tag = dict(SAMPLE_CATALOG_ENTRY)
        entry_with_tag["tags"] = ["spatial-profile"]
        (d / "electron_temperature.yaml").write_text(yaml.safe_dump(entry_with_tag))
        (d / "plasma_current.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY_MINIMAL)
        )

        # Graph has no entries
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        with (
            patch("imas_codex.graph.client.GraphClient") as MockGC,
            patch(
                "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
                return_value=None,
            ),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)

            cr = check_catalog(
                catalog_dir=d,
                tag_filter=["spatial-profile"],
            )

        # Only electron_temperature has the tag — plasma_current should be filtered out
        assert cr.only_in_catalog == ["electron_temperature"]
        assert cr.in_sync == 0

    def test_check_empty_catalog(self, tmp_path: Path) -> None:
        """Empty catalog directory should return empty CheckResult."""
        from imas_codex.standard_names.catalog_import import check_catalog

        d = tmp_path / "empty_catalog"
        d.mkdir()

        with patch(
            "imas_codex.standard_names.catalog_import._resolve_catalog_sha",
            return_value=None,
        ):
            cr = check_catalog(catalog_dir=d)

        assert cr.in_sync == 0
        assert cr.only_in_catalog == []
        assert cr.only_in_graph == []
        assert cr.diverged == []


class TestNormalizeField:
    """Tests for _normalize_field() comparison normalization."""

    def test_none(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(None) is None

    def test_empty_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("") is None
        assert _normalize_field("  ") is None

    def test_normal_string(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field("hello") == "hello"
        assert _normalize_field("  hello  ") == "hello"

    def test_empty_list(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field([]) is None

    def test_list_sorted(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(["b", "a"]) == ("a", "b")
        assert _normalize_field(["a", "b"]) == ("a", "b")

    def test_numeric_passthrough(self) -> None:
        from imas_codex.standard_names.catalog_import import _normalize_field

        assert _normalize_field(42) == 42
        assert _normalize_field(3.14) == 3.14


class TestPublishImportRoundTrip:
    """Test that published entries can be reviewed, imported, and re-imported."""

    def test_published_entry_importable_after_review(self, tmp_path: Path) -> None:
        """A published entry enriched with catalog fields should import cleanly."""
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.models import SNProvenance, SNPublishEntry
        from imas_codex.standard_names.publish import generate_yaml_entry

        # 1. Generate a published YAML entry (what `sn publish` produces)
        published = SNPublishEntry(
            name="electron_temperature",
            kind="physical",
            unit="eV",
            tags=["core_profiles"],
            status="drafted",
            description="Electron temperature",
            provenance=SNProvenance(
                source="dd",
                source_id="core_profiles/profiles_1d/electrons/temperature",
                ids_name="core_profiles",
                confidence=0.95,
            ),
        )
        published_yaml = generate_yaml_entry(published)
        assert "electron_temperature" in published_yaml

        # 2. Simulate reviewer enriching entry into catalog format
        reviewed = {
            "name": "electron_temperature",
            "description": "Electron temperature",
            "documentation": "The electron temperature Te is measured by Thomson.",
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

        catalog_dir = tmp_path / "reviewed_catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(yaml.safe_dump(reviewed))

        # 3. Import the reviewed entry
        result = import_catalog(catalog_dir=catalog_dir, dry_run=True)

        assert result.imported == 1
        entry = result.entries[0]

        # 4. Verify all catalog fields map correctly into graph dict shape
        assert entry["id"] == "electron_temperature"
        assert entry["unit"] == "eV"
        assert entry["imas_paths"] == [
            "core_profiles/profiles_1d/electrons/temperature"
        ]
        assert entry["review_status"] == "accepted"
        assert entry["source_type"] == "dd"
        assert entry["physics_domain"] == "core_plasma_physics"
        assert entry["validity_domain"] == "core plasma"
        assert entry["constraints"] == ["T_e > 0"]
        # Grammar-parsed fields should be populated
        assert entry["physical_base"] == "temperature"
        assert entry["subject"] == "electron"

    def test_round_trip_preserves_all_fields(self, tmp_path: Path) -> None:
        """Importing the same catalog entry twice should yield identical dicts."""
        from imas_codex.standard_names.catalog_import import import_catalog

        catalog_dir = tmp_path / "rt_catalog"
        catalog_dir.mkdir()
        (catalog_dir / "electron_temperature.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY)
        )
        (catalog_dir / "plasma_current.yaml").write_text(
            yaml.safe_dump(SAMPLE_CATALOG_ENTRY_MINIMAL)
        )

        r1 = import_catalog(catalog_dir=catalog_dir, dry_run=True)
        r2 = import_catalog(catalog_dir=catalog_dir, dry_run=True)

        assert len(r1.entries) == len(r2.entries) == 2
        for e1, e2 in zip(r1.entries, r2.entries, strict=False):
            assert e1 == e2, f"Mismatch for {e1.get('id')}: {e1} != {e2}"

    def test_graph_records_reimport_consistency(self, tmp_path: Path) -> None:
        """graph_records_to_entries output can be re-published and re-imported."""
        from imas_codex.standard_names.catalog_import import import_catalog
        from imas_codex.standard_names.publish import (
            generate_catalog_files,
            graph_records_to_entries,
        )

        # Simulate graph records from a first import
        graph_records = [
            {
                "name": "electron_temperature",
                "source_type": "dd",
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "unit": "eV",
                "description": "Electron temperature",
                "ids_name": "core_profiles",
                "confidence": 0.95,
            }
        ]

        # Convert to publish entries and write YAML
        publish_entries = graph_records_to_entries(graph_records)
        assert len(publish_entries) == 1
        assert publish_entries[0].name == "electron_temperature"

        publish_dir = tmp_path / "published"
        written = generate_catalog_files(publish_entries, publish_dir)
        assert len(written) == 1

        # Now create a "reviewed" catalog version from the published YAML
        catalog_dir = tmp_path / "catalog_reviewed"
        catalog_dir.mkdir()
        reviewed = {
            "name": "electron_temperature",
            "description": "Electron temperature",
            "documentation": "Te from core profiles.",
            "kind": "scalar",
            "unit": "eV",
            "tags": [],
            "links": [],
            "ids_paths": ["core_profiles/profiles_1d/electrons/temperature"],
            "validity_domain": "",
            "constraints": [],
            "physics_domain": "core_plasma_physics",
            "status": "active",
        }
        (catalog_dir / "electron_temperature.yaml").write_text(yaml.safe_dump(reviewed))

        # Import the reviewed entry
        result = import_catalog(catalog_dir=catalog_dir, dry_run=True)
        assert result.imported == 1
        entry = result.entries[0]

        # Verify key fields survive the full publish→review→import cycle
        assert entry["id"] == "electron_temperature"
        assert entry["unit"] == "eV"
        assert entry["source_type"] == "dd"
        assert entry["review_status"] == "accepted"
        assert entry["documentation"] == "Te from core profiles."
