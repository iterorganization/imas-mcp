"""Tests for document store."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.search.document_store import (
    Document,
    DocumentMetadata,
    DocumentStore,
    SearchIndex,
    Units,
)


def create_mock_document(
    path_id: str,
    ids_name: str = "core_profiles",
    documentation: str = "",
    units: str = "m",
) -> Document:
    """Create a mock document for testing."""
    metadata = DocumentMetadata(
        path_id=path_id,
        ids_name=ids_name,
        path_name=path_id.split("/")[-1],
        units=units,
        data_type="float",
        coordinates=("rho_tor_norm",),
        physics_domain="transport",
    )
    doc = Document(
        metadata=metadata,
        documentation=documentation or f"Documentation for {path_id}",
        relationships={},
        raw_data={"data_type": "float", "units": units},
    )
    doc.set_units()
    return doc


class TestUnits:
    """Tests for Units dataclass."""

    def test_from_unit_string_basic(self):
        """Test creating Units from basic unit string."""
        units = Units.from_unit_string("eV")

        assert units.unit_str == "eV"
        assert units.has_meaningful_units() is True

    def test_from_unit_string_dimensionless(self):
        """Test creating Units from dimensionless string."""
        units = Units.from_unit_string("1")

        assert units.has_meaningful_units() is False

    def test_from_unit_string_empty(self):
        """Test creating Units from empty string."""
        units = Units.from_unit_string("")

        assert units.has_meaningful_units() is False

    def test_has_meaningful_units(self):
        """Test meaningful units detection."""
        assert Units.from_unit_string("eV").has_meaningful_units() is True
        assert Units.from_unit_string("m^-3").has_meaningful_units() is True
        assert Units.from_unit_string("1").has_meaningful_units() is False
        assert Units.from_unit_string("none").has_meaningful_units() is False
        assert Units.from_unit_string("").has_meaningful_units() is False

    def test_get_embedding_components(self):
        """Test embedding component generation."""
        units = Units.from_unit_string("eV")

        components = units.get_embedding_components()

        assert len(components) > 0
        assert any("eV" in c for c in components)

    def test_get_embedding_components_empty(self):
        """Test embedding components for dimensionless units."""
        units = Units.from_unit_string("1")

        components = units.get_embedding_components()

        assert len(components) == 0


class TestDocumentMetadata:
    """Tests for DocumentMetadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata creation with all fields."""
        metadata = DocumentMetadata(
            path_id="core_profiles/temperature",
            ids_name="core_profiles",
            path_name="temperature",
            units="eV",
            data_type="FLT_1D",
            coordinates=("rho_tor_norm",),
            physics_domain="transport",
            physics_phenomena=("transport", "heating"),
        )

        assert metadata.path_id == "core_profiles/temperature"
        assert metadata.physics_domain == "transport"
        assert len(metadata.physics_phenomena) == 2

    def test_metadata_immutable(self):
        """Test that metadata is immutable (frozen)."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test",
            path_name="path",
        )

        with pytest.raises(AttributeError):
            metadata.path_id = "new/path"


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self):
        """Test document creation."""
        doc = create_mock_document("core_profiles/temperature", units="eV")

        assert doc.metadata.path_id == "core_profiles/temperature"
        assert doc.units is not None

    def test_set_units(self):
        """Test unit setting."""
        metadata = DocumentMetadata(
            path_id="test/path",
            ids_name="test",
            path_name="path",
            units="eV",
        )
        doc = Document(metadata=metadata, documentation="Test")

        doc.set_units()

        assert doc.units is not None
        assert doc.units.unit_str == "eV"

    def test_embedding_text(self):
        """Test embedding text generation."""
        doc = create_mock_document(
            "core_profiles/temperature",
            ids_name="core_profiles",
            documentation="Electron temperature profile",
            units="eV",
        )

        text = doc.embedding_text

        assert "core profiles" in text or "core_profiles" in text
        assert "temperature" in text
        assert "eV" in text or "electron_volt" in text

    def test_to_datapath(self):
        """Test conversion to IdsNode."""
        doc = create_mock_document("core_profiles/temperature", units="eV")
        doc.raw_data = {
            "data_type": "FLT_1D",
            "lifecycle": "active",
            "units": "eV",
        }

        datapath = doc.to_datapath()

        assert datapath.path == "core_profiles/temperature"
        assert datapath.units == "eV"

    def test_to_datapath_with_identifier_schema(self):
        """Test conversion with identifier schema."""
        doc = create_mock_document("core_profiles/type")
        doc.raw_data = {
            "identifier_schema": {
                "schema_path": "test/schema.xml",
                "documentation": "Test schema",
                "options": [],
                "metadata": {},
            }
        }

        datapath = doc.to_datapath()

        assert datapath.identifier_schema is not None

    def test_to_datapath_with_validation_rules(self):
        """Test conversion with validation rules."""
        doc = create_mock_document("core_profiles/density")
        doc.raw_data = {
            "validation_rules": {
                "min_value": 0,
                "max_value": 1e25,
                "units_required": True,
            }
        }

        datapath = doc.to_datapath()

        assert datapath.validation_rules is not None
        assert datapath.validation_rules.min_value == 0


class TestSearchIndex:
    """Tests for SearchIndex."""

    def test_add_document(self):
        """Test adding a document to the index."""
        index = SearchIndex()
        doc = create_mock_document("core_profiles/temperature", units="eV")

        index.add_document(doc)

        assert "core_profiles/temperature" in index.by_path_id
        assert "core_profiles" in index.by_ids_name
        assert index.total_documents == 1

    def test_add_document_physics_domain(self):
        """Test physics domain indexing."""
        index = SearchIndex()
        doc = create_mock_document("core_profiles/temperature")

        index.add_document(doc)

        assert "transport" in index.by_physics_domain
        assert "core_profiles/temperature" in index.by_physics_domain["transport"]

    def test_add_document_units_indexing(self):
        """Test units indexing."""
        index = SearchIndex()
        doc = create_mock_document("core_profiles/temperature", units="eV")

        index.add_document(doc)

        assert "eV" in index.by_units

    def test_add_document_coordinate_indexing(self):
        """Test coordinate indexing."""
        index = SearchIndex()
        doc = create_mock_document("core_profiles/temperature")

        index.add_document(doc)

        assert "rho_tor_norm" in index.by_coordinates

    def test_add_document_documentation_indexing(self):
        """Test documentation word indexing."""
        index = SearchIndex()
        doc = create_mock_document(
            "core_profiles/temperature", documentation="Electron temperature profile"
        )

        index.add_document(doc)

        assert "electron" in index.documentation_words
        assert "temperature" in index.documentation_words

    def test_add_document_path_segment_indexing(self):
        """Test path segment indexing."""
        index = SearchIndex()
        metadata = DocumentMetadata(
            path_id="core_profiles/profiles_1d/temperature",
            ids_name="core_profiles",
            path_name="core_profiles/profiles_1d/temperature",
            units="m",
            data_type="float",
            coordinates=("rho_tor_norm",),
            physics_domain="transport",
        )
        doc = Document(
            metadata=metadata,
            documentation="Temperature profile",
            relationships={},
            raw_data={"data_type": "float", "units": "m"},
        )

        index.add_document(doc)

        assert "temperature" in index.path_segments
        assert "profiles_1d" in index.path_segments


class TestDocumentStore:
    """Tests for DocumentStore."""

    @pytest.fixture
    def mock_resources_path(self, tmp_path):
        """Create mock resources directory."""
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        detailed_dir = schemas_dir / "detailed"
        detailed_dir.mkdir()

        catalog = {
            "ids_catalog": {
                "core_profiles": {"description": "Core profiles IDS"},
                "equilibrium": {"description": "Equilibrium IDS"},
            },
            "metadata": {"version": "4.0.0"},
        }
        with open(schemas_dir / "ids_catalog.json", "w") as f:
            json.dump(catalog, f)

        core_profiles_data = {
            "paths": {
                "temperature": {
                    "documentation": "Temperature",
                    "units": "eV",
                    "data_type": "FLT_1D",
                }
            }
        }
        with open(detailed_dir / "core_profiles.json", "w") as f:
            json.dump(core_profiles_data, f)

        return tmp_path

    def test_is_available(self, mock_resources_path):
        """Test availability check."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                assert store.is_available() is True

    def test_get_available_ids(self, mock_resources_path):
        """Test getting available IDS list."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                ids = store.get_available_ids()

                assert "core_profiles" in ids
                assert "equilibrium" in ids

    def test_get_available_ids_with_filter(self, mock_resources_path):
        """Test getting available IDS with filter by testing internal _get_available_ids."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore(ids_set={"core_profiles"})
                store._available_ids = None  # Reset cache to use real implementation
                ids = store._get_available_ids()

                assert "core_profiles" in ids
                assert "equilibrium" not in ids

    def test_generate_db_filename(self, mock_resources_path):
        """Test database filename generation."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                filename = store._generate_db_filename()

                assert filename == "imas_fts.db"

    def test_generate_db_filename_with_ids_set(self, mock_resources_path):
        """Test database filename with IDS set."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore(ids_set={"core_profiles"})
                filename = store._generate_db_filename()

                assert filename.startswith("imas_fts_")
                assert filename.endswith(".db")

    def test_get_dd_version(self, mock_resources_path):
        """Test getting data dictionary version."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                version = store._get_dd_version()

                assert version == "4.0.0"

    def test_preprocess_fts_query_minus_term(self, mock_resources_path):
        """Test FTS query preprocessing with minus terms."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()

                processed = store._preprocess_fts_query("plasma -wall")
                assert "NOT wall" in processed
                assert "-wall" not in processed

    def test_preprocess_fts_query_preserves_quotes(self, mock_resources_path):
        """Test FTS query preprocessing preserves quoted strings."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()

                processed = store._preprocess_fts_query('"exact phrase"')
                assert '"exact phrase"' in processed

    def test_preprocess_fts_query_removes_isolated_dash(self, mock_resources_path):
        """Test FTS query preprocessing removes isolated dashes."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()

                processed = store._preprocess_fts_query("plasma - temperature")
                assert " - " not in processed

    def test_context_manager(self, mock_resources_path):
        """Test context manager protocol."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                with DocumentStore() as store:
                    assert store is not None

    def test_get_statistics_structure(self, mock_resources_path):
        """Test that SearchIndex provides statistics when called directly."""
        index = SearchIndex()
        index.by_physics_domain = {"transport": set()}
        index.by_units = {"eV": set()}
        index.by_coordinates = {"rho": set()}
        index.documentation_words = {"test": set()}
        index.path_segments = {"path": set()}

        assert index.total_documents == 0
        assert index.total_ids == 0
        assert len(index.by_physics_domain) == 1
        assert len(index.by_units) == 1

    def test_get_physics_domains(self, mock_resources_path):
        """Test getting physics domains."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                store._index = SearchIndex()
                store._index.by_physics_domain = {"transport": set(), "mhd": set()}

                domains = store.get_physics_domains()

                assert "transport" in domains
                assert "mhd" in domains

    def test_get_available_units(self, mock_resources_path):
        """Test getting available units."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()
                store._index = SearchIndex()
                store._index.by_units = {"eV": set(), "m^-3": set()}

                units = store.get_available_units()

                assert "eV" in units
                assert "m^-3" in units

    def test_create_document(self, mock_resources_path):
        """Test document creation from raw data."""
        with patch.object(
            DocumentStore,
            "_get_resources_path",
            return_value=mock_resources_path / "schemas",
        ):
            with patch.object(
                DocumentStore,
                "_get_sqlite_dir",
                return_value=mock_resources_path / "db",
            ):
                store = DocumentStore()

                path_data = {
                    "documentation": "Test doc",
                    "units": "eV",
                    "data_type": "FLT_1D",
                    "coordinates": ["rho_tor_norm"],
                    "physics_context": {
                        "domain": "transport",
                        "phenomena": ["heating"],
                    },
                }

                doc = store._create_document("test_ids", "temperature", path_data)

                assert doc.metadata.path_id == "test_ids/temperature"
                assert doc.metadata.units == "eV"
                assert doc.metadata.physics_domain == "transport"
