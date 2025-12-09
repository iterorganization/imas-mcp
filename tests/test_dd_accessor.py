"""Tests for data dictionary accessor classes."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from packaging.version import Version

from imas_mcp.dd_accessor import (
    CompositeDataDictionaryAccessor,
    DataDictionaryAccessor,
    EnvironmentDataDictionaryAccessor,
    ImasDataDictionariesAccessor,
    ImasDataDictionaryAccessor,
    IndexNameDataDictionaryAccessor,
    MetadataDataDictionaryAccessor,
    create_dd_accessor,
    save_index_metadata,
)


class TestEnvironmentDataDictionaryAccessor:
    """Tests for EnvironmentDataDictionaryAccessor."""

    def test_get_version_with_env_var(self, monkeypatch):
        """Test getting version from environment variable."""
        monkeypatch.setenv("IMAS_DD_VERSION", "4.0.0")

        accessor = EnvironmentDataDictionaryAccessor()
        version = accessor.get_version()

        assert version == Version("4.0.0")

    def test_get_version_missing_env_var(self, monkeypatch):
        """Test error when environment variable is missing."""
        monkeypatch.delenv("IMAS_DD_VERSION", raising=False)

        accessor = EnvironmentDataDictionaryAccessor()

        with pytest.raises(ValueError):
            accessor.get_version()

    def test_is_available_with_env(self, monkeypatch):
        """Test availability when environment variable is set."""
        monkeypatch.setenv("IMAS_DD_VERSION", "4.0.0")

        accessor = EnvironmentDataDictionaryAccessor()

        assert accessor.is_available() is True

    def test_is_available_without_env(self, monkeypatch):
        """Test availability when environment variable is not set."""
        monkeypatch.delenv("IMAS_DD_VERSION", raising=False)

        accessor = EnvironmentDataDictionaryAccessor()

        assert accessor.is_available() is False

    def test_get_xml_tree_not_supported(self):
        """Test that get_xml_tree raises NotImplementedError."""
        accessor = EnvironmentDataDictionaryAccessor()

        with pytest.raises(NotImplementedError):
            accessor.get_xml_tree()

    def test_custom_env_var(self, monkeypatch):
        """Test using custom environment variable name."""
        monkeypatch.setenv("MY_DD_VERSION", "3.42.0")

        accessor = EnvironmentDataDictionaryAccessor(env_var="MY_DD_VERSION")
        version = accessor.get_version()

        assert version == Version("3.42.0")


class TestIndexNameDataDictionaryAccessor:
    """Tests for IndexNameDataDictionaryAccessor."""

    def test_get_version_basic(self):
        """Test version extraction from basic index name."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="lexicographic_4.0.1", index_prefix="lexicographic"
        )
        version = accessor.get_version()

        assert version == Version("4.0.1")

    def test_get_version_with_hash(self):
        """Test version extraction with commit hash suffix."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="lexicographic_4.0.1.dev164-9dbb96e3",
            index_prefix="lexicographic",
        )
        version = accessor.get_version()

        assert version == Version("4.0.1.dev164")

    def test_get_version_invalid_format(self):
        """Test error with invalid index name format."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="invalid_format", index_prefix="lexicographic"
        )

        with pytest.raises(ValueError):
            accessor.get_version()

    def test_is_available_valid(self):
        """Test availability with valid index name."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="lexicographic_4.0.1", index_prefix="lexicographic"
        )

        assert accessor.is_available() is True

    def test_is_available_invalid(self):
        """Test availability with invalid index name."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="invalid", index_prefix="lexicographic"
        )

        assert accessor.is_available() is False

    def test_get_xml_tree_not_supported(self):
        """Test that get_xml_tree raises NotImplementedError."""
        accessor = IndexNameDataDictionaryAccessor(
            index_name="lexicographic_4.0.1", index_prefix="lexicographic"
        )

        with pytest.raises(NotImplementedError):
            accessor.get_xml_tree()


class TestMetadataDataDictionaryAccessor:
    """Tests for MetadataDataDictionaryAccessor."""

    @pytest.fixture
    def metadata_dir(self, tmp_path):
        """Create a temporary metadata directory."""
        metadata = {
            "dd_version": "4.0.0",
            "build_timestamp": "2024-01-01T00:00:00",
            "ids_names": ["core_profiles", "equilibrium"],
            "total_documents": 1000,
        }

        metadata_file = tmp_path / "test_index.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)

        return tmp_path

    def test_get_version(self, metadata_dir):
        """Test getting version from metadata."""
        accessor = MetadataDataDictionaryAccessor(metadata_dir)
        version = accessor.get_version()

        assert version == Version("4.0.0")

    def test_is_available(self, metadata_dir):
        """Test availability with metadata file."""
        accessor = MetadataDataDictionaryAccessor(metadata_dir)

        assert accessor.is_available() is True

    def test_is_available_no_files(self, tmp_path):
        """Test availability without metadata files."""
        accessor = MetadataDataDictionaryAccessor(tmp_path)

        assert accessor.is_available() is False

    def test_get_version_missing_version(self, tmp_path):
        """Test error when version is missing in metadata."""
        metadata_file = tmp_path / "test.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({"other_field": "value"}, f)

        accessor = MetadataDataDictionaryAccessor(tmp_path)

        with pytest.raises(ValueError):
            accessor.get_version()

    def test_get_xml_tree_not_supported(self, metadata_dir):
        """Test that get_xml_tree raises NotImplementedError."""
        accessor = MetadataDataDictionaryAccessor(metadata_dir)

        with pytest.raises(NotImplementedError):
            accessor.get_xml_tree()

    def test_uses_most_recent_file(self, tmp_path):
        """Test that the most recent metadata file is used."""
        import time

        old_metadata = tmp_path / "old.metadata.json"
        with open(old_metadata, "w") as f:
            json.dump({"dd_version": "3.0.0"}, f)

        time.sleep(0.1)

        new_metadata = tmp_path / "new.metadata.json"
        with open(new_metadata, "w") as f:
            json.dump({"dd_version": "4.0.0"}, f)

        accessor = MetadataDataDictionaryAccessor(tmp_path)
        version = accessor.get_version()

        assert version == Version("4.0.0")


class TestImasDataDictionaryAccessor:
    """Tests for ImasDataDictionaryAccessor."""

    def test_load_imas_dd_import_error(self):
        """Test error when imas-data-dictionary is not installed."""
        with patch.dict("sys.modules", {"imas_data_dictionary": None}):
            with patch("builtins.__import__", side_effect=ImportError("Not installed")):
                with pytest.raises(ImportError):
                    ImasDataDictionaryAccessor()

    @pytest.mark.skipif(
        True, reason="Requires imas-data-dictionary package to be installed"
    )
    def test_is_available_with_package(self):
        """Test availability when package is loaded."""
        accessor = ImasDataDictionaryAccessor()
        assert accessor.is_available() is True

    @pytest.mark.skipif(
        True, reason="Requires imas-data-dictionary package to be installed"
    )
    def test_get_schema_success(self):
        """Test successful schema retrieval."""
        accessor = ImasDataDictionaryAccessor()
        _result = accessor.get_schema("test/schema.xml")

    def test_get_schema_not_available(self):
        """Test schema retrieval when method not available."""
        with patch("imas_mcp.dd_accessor.ImasDataDictionaryAccessor._load_imas_dd"):
            accessor = ImasDataDictionaryAccessor.__new__(ImasDataDictionaryAccessor)
            accessor._imas_dd = MagicMock()
            del accessor._imas_dd.get_schema

            result = accessor.get_schema("test/schema.xml")

            assert result is None


class TestImasDataDictionariesAccessor:
    """Tests for ImasDataDictionariesAccessor (PyPI package)."""

    def test_init_import_error(self):
        """Test error when imas_data_dictionaries is not installed."""
        with patch.dict("sys.modules", {"imas_data_dictionaries": None}):
            with patch("builtins.__import__", side_effect=ImportError("Not installed")):
                with pytest.raises(ImportError):
                    ImasDataDictionariesAccessor("4.0.0")

    def test_get_version(self):
        """Test version retrieval."""
        with patch("imas_data_dictionaries.get_dd_xml") as mock_get_xml:
            mock_get_xml.return_value = b"<root></root>"

            accessor = ImasDataDictionariesAccessor("4.0.0")

            assert accessor.get_version() == Version("4.0.0")


class TestCompositeDataDictionaryAccessor:
    """Tests for CompositeDataDictionaryAccessor."""

    def test_uses_first_available(self):
        """Test that first available accessor is used."""
        unavailable = MagicMock()
        unavailable.is_available.return_value = False

        available = MagicMock()
        available.is_available.return_value = True
        available.get_version.return_value = Version("4.0.0")

        composite = CompositeDataDictionaryAccessor([unavailable, available])
        version = composite.get_version()

        assert version == Version("4.0.0")
        available.get_version.assert_called_once()

    def test_caches_primary_accessor(self):
        """Test that primary accessor is cached."""
        available = MagicMock()
        available.is_available.return_value = True
        available.get_version.return_value = Version("4.0.0")

        composite = CompositeDataDictionaryAccessor([available])

        composite.get_version()
        composite.get_version()

        assert available.is_available.call_count == 2

    def test_no_available_accessor(self):
        """Test error when no accessor is available."""
        unavailable = MagicMock()
        unavailable.is_available.return_value = False

        composite = CompositeDataDictionaryAccessor([unavailable])

        with pytest.raises(RuntimeError):
            composite.get_version()

    def test_is_available(self):
        """Test availability check."""
        available = MagicMock()
        available.is_available.return_value = True

        composite = CompositeDataDictionaryAccessor([available])

        assert composite.is_available() is True

    def test_is_not_available(self):
        """Test availability when none available."""
        unavailable = MagicMock()
        unavailable.is_available.return_value = False

        composite = CompositeDataDictionaryAccessor([unavailable])

        assert composite.is_available() is False

    def test_get_xml_tree_skips_unsupported(self):
        """Test XML tree access skips unsupported accessors."""
        unsupported = MagicMock()
        unsupported.is_available.return_value = True
        unsupported.get_xml_tree.side_effect = NotImplementedError()

        supported = MagicMock()
        supported.is_available.return_value = True
        supported.get_xml_tree.return_value = MagicMock()

        composite = CompositeDataDictionaryAccessor([unsupported, supported])
        result = composite.get_xml_tree()

        assert result is not None

    def test_get_xml_tree_no_support(self):
        """Test error when no accessor supports XML tree."""
        unsupported = MagicMock()
        unsupported.is_available.return_value = True
        unsupported.get_xml_tree.side_effect = NotImplementedError()

        composite = CompositeDataDictionaryAccessor([unsupported])

        with pytest.raises(RuntimeError):
            composite.get_xml_tree()


class TestCreateDDAccessor:
    """Tests for create_dd_accessor factory function."""

    def test_creates_composite(self, monkeypatch):
        """Test that factory creates composite accessor."""
        monkeypatch.delenv("IMAS_DD_VERSION", raising=False)

        with patch(
            "imas_mcp.dd_accessor.ImasDataDictionaryAccessor",
            side_effect=ImportError(),
        ):
            accessor = create_dd_accessor()

            assert isinstance(accessor, CompositeDataDictionaryAccessor)

    def test_includes_environment_accessor(self, monkeypatch):
        """Test that environment accessor is included."""
        monkeypatch.setenv("IMAS_DD_VERSION", "4.0.0")

        with patch(
            "imas_mcp.dd_accessor.ImasDataDictionaryAccessor",
            side_effect=ImportError(),
        ):
            accessor = create_dd_accessor()

            assert accessor.is_available() is True

    def test_includes_metadata_accessor(self, tmp_path, monkeypatch):
        """Test that metadata accessor is included when dir provided."""
        monkeypatch.delenv("IMAS_DD_VERSION", raising=False)

        metadata_file = tmp_path / "test.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({"dd_version": "4.0.0"}, f)

        with patch(
            "imas_mcp.dd_accessor.ImasDataDictionaryAccessor",
            side_effect=ImportError(),
        ):
            accessor = create_dd_accessor(metadata_dir=tmp_path)

            assert accessor.is_available() is True

    def test_includes_index_name_accessor(self, monkeypatch):
        """Test that index name accessor is included when provided."""
        monkeypatch.delenv("IMAS_DD_VERSION", raising=False)

        with patch(
            "imas_mcp.dd_accessor.ImasDataDictionaryAccessor",
            side_effect=ImportError(),
        ):
            accessor = create_dd_accessor(
                index_name="lexicographic_4.0.0", index_prefix="lexicographic"
            )

            assert accessor.is_available() is True


class TestSaveIndexMetadata:
    """Tests for save_index_metadata function."""

    def test_creates_metadata_file(self, tmp_path):
        """Test metadata file creation."""
        save_index_metadata(
            metadata_dir=tmp_path,
            index_name="test_index",
            dd_version=Version("4.0.0"),
            ids_names=["core_profiles", "equilibrium"],
            total_documents=1000,
            index_type="lexicographic",
        )

        metadata_file = tmp_path / "test_index.metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            data = json.load(f)

        assert data["dd_version"] == "4.0.0"
        assert data["total_documents"] == 1000
        assert "core_profiles" in data["ids_names"]

    def test_creates_directory(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_subdir"

        save_index_metadata(
            metadata_dir=new_dir,
            index_name="test_index",
            dd_version=Version("4.0.0"),
            ids_names=[],
            total_documents=0,
            index_type="test",
        )

        assert new_dir.exists()

    def test_includes_build_metadata(self, tmp_path):
        """Test that additional build metadata is included."""
        save_index_metadata(
            metadata_dir=tmp_path,
            index_name="test_index",
            dd_version=Version("4.0.0"),
            ids_names=[],
            total_documents=0,
            index_type="test",
            build_metadata={"custom_field": "custom_value"},
        )

        metadata_file = tmp_path / "test_index.metadata.json"
        with open(metadata_file) as f:
            data = json.load(f)

        assert data["custom_field"] == "custom_value"
