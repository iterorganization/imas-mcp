"""Tests for the standard name publish module.

Tests YAML generation, batching, duplicate checking, model validation,
and graph record conversion — all pure-function tests that don't require
a live Neo4j connection.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.sn.models import SNProvenance, SNPublishBatch, SNPublishEntry
from imas_codex.sn.publish import (
    batch_by_group,
    check_catalog_duplicates,
    confidence_tier,
    generate_catalog_files,
    generate_yaml_entry,
    graph_records_to_entries,
    make_publish_batches,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def sample_provenance() -> SNProvenance:
    return SNProvenance(
        source="dd",
        source_id="equilibrium/time_slice/profiles_1d/electrons/temperature",
        ids_name="equilibrium",
        confidence=0.95,
        generated_by="imas-codex",
    )


@pytest.fixture()
def sample_entry(sample_provenance: SNProvenance) -> SNPublishEntry:
    return SNPublishEntry(
        name="electron_temperature",
        kind="scalar",
        unit="eV",
        tags=["equilibrium", "core_profiles"],
        status="drafted",
        description="Electron temperature profile",
        provenance=sample_provenance,
    )


@pytest.fixture()
def sample_entries() -> list[SNPublishEntry]:
    """A diverse set of entries for batching / grouping tests."""
    return [
        SNPublishEntry(
            name="electron_temperature",
            kind="scalar",
            unit="eV",
            tags=["equilibrium"],
            description="Electron temperature",
            provenance=SNProvenance(
                source="dd",
                source_id="equilibrium/time_slice/profiles_1d/electrons/temperature",
                ids_name="equilibrium",
                confidence=0.95,
            ),
        ),
        SNPublishEntry(
            name="electron_density",
            kind="scalar",
            unit="m^-3",
            tags=["core_profiles"],
            description="Electron density",
            provenance=SNProvenance(
                source="dd",
                source_id="core_profiles/profiles_1d/electrons/density",
                ids_name="core_profiles",
                confidence=0.88,
            ),
        ),
        SNPublishEntry(
            name="plasma_current",
            kind="scalar",
            unit="A",
            tags=["equilibrium"],
            description="Plasma current",
            provenance=SNProvenance(
                source="dd",
                source_id="equilibrium/time_slice/global_quantities/ip",
                ids_name="equilibrium",
                confidence=0.45,
            ),
        ),
        SNPublishEntry(
            name="major_radius",
            kind="vector",
            unit="m",
            tags=["equilibrium"],
            description="Major radius",
            provenance=SNProvenance(
                source="signals",
                source_id="tcv:rmajor",
                ids_name=None,
                confidence=0.72,
            ),
        ),
    ]


# =============================================================================
# Model validation tests
# =============================================================================


class TestSNProvenance:
    def test_valid_provenance(self) -> None:
        p = SNProvenance(
            source="dd",
            source_id="equilibrium/time_slice/profiles_1d/psi",
            confidence=0.9,
        )
        assert p.source == "dd"
        assert p.generated_by == "imas-codex"

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            SNProvenance(source="dd", source_id="x", confidence=1.5)
        with pytest.raises(ValueError):
            SNProvenance(source="dd", source_id="x", confidence=-0.1)

    def test_optional_ids_name(self) -> None:
        p = SNProvenance(source="signals", source_id="sig:x", confidence=0.5)
        assert p.ids_name is None


class TestSNPublishEntry:
    def test_defaults(self, sample_provenance: SNProvenance) -> None:
        entry = SNPublishEntry(
            name="test_name",
            provenance=sample_provenance,
        )
        assert entry.kind == "scalar"
        assert entry.status == "drafted"
        assert entry.tags == []
        assert entry.unit is None
        assert entry.documentation is None
        assert entry.links == []
        assert entry.ids_paths == []
        assert entry.constraints == []
        assert entry.validity_domain is None

    def test_all_fields(self, sample_entry: SNPublishEntry) -> None:
        assert sample_entry.name == "electron_temperature"
        assert sample_entry.kind == "scalar"
        assert sample_entry.unit == "eV"
        assert "equilibrium" in sample_entry.tags
        assert sample_entry.provenance.confidence == 0.95


class TestSNPublishBatch:
    def test_batch_creation(self, sample_entry: SNPublishEntry) -> None:
        batch = SNPublishBatch(
            group_key="equilibrium",
            entries=[sample_entry],
            confidence_tier="high",
        )
        assert batch.group_key == "equilibrium"
        assert len(batch.entries) == 1
        assert batch.confidence_tier == "high"


# =============================================================================
# Confidence tier tests
# =============================================================================


class TestConfidenceTier:
    def test_high(self) -> None:
        assert confidence_tier(0.8) == "high"
        assert confidence_tier(0.95) == "high"
        assert confidence_tier(1.0) == "high"

    def test_medium(self) -> None:
        assert confidence_tier(0.5) == "medium"
        assert confidence_tier(0.79) == "medium"

    def test_low(self) -> None:
        assert confidence_tier(0.0) == "low"
        assert confidence_tier(0.49) == "low"


# =============================================================================
# YAML generation tests
# =============================================================================


class TestGenerateYamlEntry:
    def test_format(self, sample_entry: SNPublishEntry) -> None:
        content = generate_yaml_entry(sample_entry)
        doc = yaml.safe_load(content)

        assert doc["name"] == "electron_temperature"
        assert doc["kind"] == "scalar"
        assert doc["unit"] == "eV"
        assert doc["status"] == "drafted"
        assert doc["description"] == "Electron temperature profile"
        assert doc["provenance"]["source"] == "dd"
        assert doc["provenance"]["confidence"] == 0.95
        assert doc["provenance"]["generated_by"] == "imas-codex"

    def test_all_fields_present(self, sample_entry: SNPublishEntry) -> None:
        content = generate_yaml_entry(sample_entry)
        doc = yaml.safe_load(content)

        # Required fields
        assert "name" in doc
        assert "kind" in doc
        assert "status" in doc
        assert "provenance" in doc

        # Provenance sub-fields
        prov = doc["provenance"]
        assert "source" in prov
        assert "source_id" in prov
        assert "confidence" in prov
        assert "generated_by" in prov

    def test_optional_unit_omitted(self, sample_provenance: SNProvenance) -> None:
        entry = SNPublishEntry(
            name="test_name",
            provenance=sample_provenance,
        )
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)
        assert "unit" not in doc

    def test_optional_ids_name_omitted(self) -> None:
        prov = SNProvenance(source="signals", source_id="sig:x", confidence=0.5)
        entry = SNPublishEntry(name="test_name", provenance=prov)
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)
        assert "ids_name" not in doc["provenance"]

    def test_tags_in_output(self, sample_entry: SNPublishEntry) -> None:
        content = generate_yaml_entry(sample_entry)
        doc = yaml.safe_load(content)
        assert doc["tags"] == ["equilibrium", "core_profiles"]

    def test_empty_tags_omitted(self, sample_provenance: SNProvenance) -> None:
        entry = SNPublishEntry(
            name="test_name",
            provenance=sample_provenance,
        )
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)
        assert "tags" not in doc

    def test_roundtrip_yaml(self, sample_entry: SNPublishEntry) -> None:
        """YAML output should parse back to the same values."""
        content = generate_yaml_entry(sample_entry)
        doc = yaml.safe_load(content)
        assert doc["name"] == sample_entry.name
        assert doc["unit"] == sample_entry.unit
        assert doc["provenance"]["confidence"] == sample_entry.provenance.confidence


class TestGenerateCatalogFiles:
    def test_writes_files(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        written = generate_catalog_files(sample_entries, tmp_path)
        assert len(written) == len(sample_entries)
        for path in written:
            assert path.exists()
            assert path.suffix == ".yaml"

    def test_filenames(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        written = generate_catalog_files(sample_entries, tmp_path)
        names = {p.stem for p in written}
        expected = {e.name for e in sample_entries}
        assert names == expected

    def test_directory_structure_by_tag(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        """Files should be grouped into tag-based subdirectories."""
        written = generate_catalog_files(sample_entries, tmp_path)
        # All entries have tags — check subdirs were created
        subdirs = {p.parent.name for p in written}
        assert "equilibrium" in subdirs
        assert "core_profiles" in subdirs
        # equilibrium tag entries go into equilibrium/
        eq_files = [p for p in written if p.parent.name == "equilibrium"]
        eq_names = {p.stem for p in eq_files}
        assert "electron_temperature" in eq_names
        assert "plasma_current" in eq_names
        assert "major_radius" in eq_names

    def test_untagged_goes_to_unscoped(
        self, tmp_path: Path, sample_provenance: SNProvenance
    ) -> None:
        """Entries without tags go into 'unscoped/' subdirectory."""
        entry = SNPublishEntry(name="untagged_quantity", provenance=sample_provenance)
        written = generate_catalog_files([entry], tmp_path)
        assert len(written) == 1
        assert written[0].parent.name == "unscoped"

    def test_file_content_valid_yaml(
        self, tmp_path: Path, sample_entry: SNPublishEntry
    ) -> None:
        written = generate_catalog_files([sample_entry], tmp_path)
        assert len(written) == 1
        with open(written[0]) as f:
            doc = yaml.safe_load(f)
        assert doc["name"] == "electron_temperature"

    def test_creates_output_dir(
        self, tmp_path: Path, sample_entry: SNPublishEntry
    ) -> None:
        out = tmp_path / "nested" / "dir"
        assert not out.exists()
        generate_catalog_files([sample_entry], out)
        assert out.is_dir()


# =============================================================================
# Batching tests
# =============================================================================


class TestBatchByGroup:
    def test_batch_by_ids(self, sample_entries: list[SNPublishEntry]) -> None:
        groups = batch_by_group(sample_entries, group_by="ids")
        assert "equilibrium" in groups
        assert "core_profiles" in groups
        # major_radius has no ids_name → "unscoped"
        assert "unscoped" in groups
        assert len(groups["equilibrium"]) == 2  # electron_temperature + plasma_current

    def test_batch_by_domain(self, sample_entries: list[SNPublishEntry]) -> None:
        groups = batch_by_group(sample_entries, group_by="domain")
        assert "equilibrium" in groups
        assert "core_profiles" in groups

    def test_batch_by_confidence(self, sample_entries: list[SNPublishEntry]) -> None:
        groups = batch_by_group(sample_entries, group_by="confidence")
        assert "high" in groups  # 0.95, 0.88
        assert "medium" in groups  # 0.72
        assert "low" in groups  # 0.45

    def test_empty_entries(self) -> None:
        groups = batch_by_group([], group_by="ids")
        assert groups == {}


class TestMakePublishBatches:
    def test_creates_batches(self, sample_entries: list[SNPublishEntry]) -> None:
        batches = make_publish_batches(sample_entries, group_by="ids")
        assert len(batches) > 0
        assert all(isinstance(b, SNPublishBatch) for b in batches)

    def test_batch_confidence_tier(self, sample_entries: list[SNPublishEntry]) -> None:
        batches = make_publish_batches(sample_entries, group_by="ids")
        for batch in batches:
            assert batch.confidence_tier in ("high", "medium", "low")

    def test_all_entries_accounted(self, sample_entries: list[SNPublishEntry]) -> None:
        batches = make_publish_batches(sample_entries, group_by="ids")
        total = sum(len(b.entries) for b in batches)
        assert total == len(sample_entries)


# =============================================================================
# Duplicate checking tests
# =============================================================================


class TestCheckCatalogDuplicates:
    def test_no_catalog_dir(self, sample_entries: list[SNPublishEntry]) -> None:
        new, dupes = check_catalog_duplicates(sample_entries, catalog_dir=None)
        assert len(new) == len(sample_entries)
        assert len(dupes) == 0

    def test_no_duplicates(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        new, dupes = check_catalog_duplicates(sample_entries, catalog_dir=tmp_path)
        assert len(new) == len(sample_entries)
        assert len(dupes) == 0

    def test_finds_catalog_duplicates(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        # Write one existing catalog entry in a subdirectory (tag-based layout)
        subdir = tmp_path / "equilibrium"
        subdir.mkdir()
        (subdir / "electron_temperature.yaml").write_text(
            yaml.safe_dump({"name": "electron_temperature", "kind": "scalar"})
        )
        new, dupes = check_catalog_duplicates(sample_entries, catalog_dir=tmp_path)
        assert len(dupes) == 1
        assert dupes[0].name == "electron_temperature"
        assert len(new) == len(sample_entries) - 1

    def test_finds_catalog_duplicates_top_level(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        # Also detect duplicates in flat (top-level) YAML files
        (tmp_path / "electron_temperature.yaml").write_text(
            yaml.safe_dump({"name": "electron_temperature", "kind": "scalar"})
        )
        new, dupes = check_catalog_duplicates(sample_entries, catalog_dir=tmp_path)
        assert len(dupes) == 1
        assert dupes[0].name == "electron_temperature"

    def test_finds_within_batch_duplicates(
        self, sample_provenance: SNProvenance
    ) -> None:
        entries = [
            SNPublishEntry(name="dup_name", provenance=sample_provenance),
            SNPublishEntry(name="dup_name", provenance=sample_provenance),
            SNPublishEntry(name="unique_name", provenance=sample_provenance),
        ]
        new, dupes = check_catalog_duplicates(entries, catalog_dir=None)
        assert len(new) == 2  # first dup_name + unique_name
        assert len(dupes) == 1  # second dup_name

    def test_nonexistent_catalog_dir(
        self, tmp_path: Path, sample_entries: list[SNPublishEntry]
    ) -> None:
        """Non-existent catalog dir should be treated as empty."""
        fake_dir = tmp_path / "nonexistent"
        new, dupes = check_catalog_duplicates(sample_entries, catalog_dir=fake_dir)
        assert len(new) == len(sample_entries)


# =============================================================================
# Graph record conversion tests
# =============================================================================


class TestGraphRecordsToEntries:
    def test_schema_canonical_fields(self) -> None:
        """Test conversion from schema-canonical property names."""
        records = [
            {
                "name": "electron_temperature",
                "description": "Te profile",
                "source": "dd",
                "source_path": "equilibrium/time_slice/profiles_1d/electrons/temperature",
                "canonical_units": "eV",
                "confidence": 0.9,
                "ids_name": "equilibrium",
            }
        ]
        entries = graph_records_to_entries(records)
        assert len(entries) == 1
        e = entries[0]
        assert e.name == "electron_temperature"
        assert e.unit == "eV"
        assert e.provenance.source == "dd"
        assert e.provenance.source_id == (
            "equilibrium/time_slice/profiles_1d/electrons/temperature"
        )
        assert e.provenance.confidence == 0.9
        assert e.provenance.ids_name == "equilibrium"

    def test_legacy_field_names(self) -> None:
        """Test conversion from legacy write property names."""
        records = [
            {
                "name": "plasma_current",
                "description": "Ip",
                "source_type": "dd",
                "source_id": "equilibrium/global/ip",
                "units": "A",
                "confidence": None,
                "ids_name": None,
            }
        ]
        entries = graph_records_to_entries(records)
        assert len(entries) == 1
        e = entries[0]
        assert e.name == "plasma_current"
        assert e.unit == "A"
        assert e.provenance.source == "dd"
        assert e.provenance.confidence == 1.0  # default for validated

    def test_empty_records(self) -> None:
        assert graph_records_to_entries([]) == []

    def test_skips_nameless_records(self) -> None:
        records = [{"description": "orphan", "source": "dd"}]
        entries = graph_records_to_entries(records)
        assert len(entries) == 0

    def test_tags_include_ids_name(self) -> None:
        records = [
            {
                "name": "foo",
                "source": "dd",
                "source_path": "x",
                "confidence": 0.7,
                "ids_name": "magnetics",
            }
        ]
        entries = graph_records_to_entries(records)
        assert "magnetics" in entries[0].tags

    def test_tags_empty_when_no_ids(self) -> None:
        records = [
            {
                "name": "bar",
                "source": "signals",
                "source_path": "y",
                "confidence": 0.7,
                "ids_name": None,
            }
        ]
        entries = graph_records_to_entries(records)
        assert entries[0].tags == []

    def test_rich_fields_carried_through(self) -> None:
        """Rich fields (documentation, links, ids_paths, constraints, validity_domain, kind) are preserved."""
        records = [
            {
                "name": "ion_temperature",
                "description": "Ion temperature",
                "documentation": "Ion temperature $T_i$ in eV. See also electron_temperature.",
                "kind": "scalar",
                "source": "dd",
                "source_path": "core_profiles/profiles_1d/ion/temperature",
                "canonical_units": "eV",
                "confidence": 0.9,
                "ids_name": "core_profiles",
                "tags": ["core_profiles", "kinetics"],
                "links": ["electron_temperature", "ion_density"],
                "ids_paths": ["core_profiles/profiles_1d/ion/temperature"],
                "constraints": ["T_i > 0"],
                "validity_domain": "core plasma",
            }
        ]
        entries = graph_records_to_entries(records)
        assert len(entries) == 1
        e = entries[0]
        assert (
            e.documentation
            == "Ion temperature $T_i$ in eV. See also electron_temperature."
        )
        assert e.kind == "scalar"
        assert e.links == ["electron_temperature", "ion_density"]
        assert e.ids_paths == ["core_profiles/profiles_1d/ion/temperature"]
        assert e.constraints == ["T_i > 0"]
        assert e.validity_domain == "core plasma"
        assert e.tags == ["core_profiles", "kinetics"]

    def test_kind_defaults_to_scalar(self) -> None:
        """kind field defaults to 'scalar' when not present in record."""
        records = [
            {"name": "test_q", "source": "dd", "source_path": "x", "confidence": 0.8}
        ]
        entries = graph_records_to_entries(records)
        assert entries[0].kind == "scalar"


# =============================================================================
# Rich-field YAML round-trip tests
# =============================================================================


class TestRichFieldRoundTrip:
    def test_all_rich_fields_in_yaml(self, sample_provenance: SNProvenance) -> None:
        """Full round-trip: create entry with all rich fields → YAML → parse back."""
        entry = SNPublishEntry(
            name="ion_temperature",
            kind="scalar",
            unit="eV",
            tags=["core_profiles", "kinetics"],
            status="drafted",
            description="Ion temperature",
            documentation="Ion temperature $T_i$ in eV. Typical range 0.1–20 keV.",
            links=["electron_temperature", "ion_density"],
            ids_paths=["core_profiles/profiles_1d/ion/temperature"],
            constraints=["T_i > 0"],
            validity_domain="core plasma",
            provenance=sample_provenance,
        )
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)

        assert doc["name"] == "ion_temperature"
        assert doc["kind"] == "scalar"
        assert doc["unit"] == "eV"
        assert (
            doc["documentation"]
            == "Ion temperature $T_i$ in eV. Typical range 0.1–20 keV."
        )
        assert doc["links"] == [
            {"name": "electron_temperature"},
            {"name": "ion_density"},
        ]
        assert doc["ids_paths"] == ["core_profiles/profiles_1d/ion/temperature"]
        assert doc["constraints"] == ["T_i > 0"]
        assert doc["validity_domain"] == "core plasma"
        assert doc["tags"] == ["core_profiles", "kinetics"]

    def test_empty_rich_fields_omitted(self, sample_provenance: SNProvenance) -> None:
        """Empty optional rich fields should not appear in YAML output."""
        entry = SNPublishEntry(
            name="bare_quantity",
            provenance=sample_provenance,
        )
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)

        assert "documentation" not in doc
        assert "links" not in doc
        assert "ids_paths" not in doc
        assert "constraints" not in doc
        assert "validity_domain" not in doc

    def test_links_formatted_as_name_dicts(
        self, sample_provenance: SNProvenance
    ) -> None:
        """links list should be serialized as [{name: ...}] objects."""
        entry = SNPublishEntry(
            name="test_quantity",
            links=["alpha", "beta"],
            provenance=sample_provenance,
        )
        content = generate_yaml_entry(entry)
        doc = yaml.safe_load(content)
        assert doc["links"] == [{"name": "alpha"}, {"name": "beta"}]
