import pytest
import whoosh.fields
import whoosh.index

from imas_mcp_server.indexer import DataDictionaryEntry, WhooshIndex, IndexableDocument


def test_whoosh_index_with_dir(tmp_path):
    """Test the WhooshIndex class."""
    dirname = tmp_path / "whoosh_index"
    dirname.mkdir(parents=True, exist_ok=True)

    # Create a Whoosh schema
    schema = whoosh.fields.Schema(
        title=whoosh.fields.TEXT(stored=True),
    )

    # create a whoosh index
    whoosh.index.create_in(dirname, schema)

    # Create an instance of WhooshIndex
    whoosh_index = WhooshIndex(dirname)

    assert whoosh_index.schema == schema

    # Check if the writer is not initialized
    assert whoosh_index._writer is None


def test_whoosh_index_without_dir(tmp_path):
    """Test the WhooshIndex class without a directory."""
    assert tmp_path.is_dir()
    assert not whoosh.index.exists_in(tmp_path)

    # Create an instance of WhooshIndex with a directory
    whoosh_index = WhooshIndex(tmp_path)
    assert whoosh_index._index is not None
    assert whoosh_index.schema is not None

    # Create an instance of WhooshIndex without a directory
    new_dir = tmp_path / "new_dir"
    assert not new_dir.is_dir()
    assert not whoosh.index.exists_in(new_dir)
    whoosh_index = WhooshIndex(new_dir)
    assert whoosh_index._index is not None
    assert whoosh_index.schema is not None


def test_with_schema(tmp_path):
    """Test the WhooshIndex class with a schema."""

    # Create a Whoosh schema
    schema = whoosh.fields.Schema(
        path=whoosh.fields.TEXT(stored=True),
        documentation=whoosh.fields.TEXT(stored=True),
    )

    # Create a matching document model
    class DocumentModel(IndexableDocument):
        """IMAS Data Dictionary document model."""

        path: str
        documentation: str

    # Create an instance of WhooshIndex with a schema
    whoosh_index = WhooshIndex(tmp_path, schema=schema, document_model=DocumentModel)

    assert whoosh_index.schema == schema
    assert whoosh_index.document_model == DocumentModel


def test_iadd_document(tmp_path):
    """Test the WhooshIndex class with __iadd__ method."""
    dirname = tmp_path

    # Create an instance of WhooshIndex
    whoosh_index = WhooshIndex(dirname)

    # Add a document to the index
    with whoosh_index.writer():
        whoosh_index += {
            "path": "equilibrium/ids_properties",
            "documentation": "properties of the equilibrium ids",
        }
        whoosh_index += {"path": "pf_active/coil/name", "documentation": "coil name"}
        assert whoosh_index._writer is not None

    # Check if the writer is None after committing
    assert whoosh_index._writer is None

    # Check for the added documents
    with whoosh_index._index.searcher() as searcher:
        results = list(searcher.documents())
        assert len(results) == 2
        assert results[0]["path"] == "equilibrium/ids_properties"
        assert results[1]["documentation"] == "coil name"

    # Check the length of the index
    assert len(whoosh_index) == 2

    # Search for 'another' in the index
    with whoosh_index.searcher() as searcher:
        results = list(searcher.find("documentation", "equilibrium properties"))
        assert len(results) == 1
        assert results[0]["documentation"] == "properties of the equilibrium ids"


def test_add_without_context(tmp_path):
    """Test the WhooshIndex class with __add__ method without context."""
    dirname = tmp_path

    # Create an instance of WhooshIndex
    whoosh_index = WhooshIndex(dirname)

    # Add a document to the index
    with pytest.raises(ValueError):
        whoosh_index += {"title": "Test Title"}


def test_units_validator():
    dd_entry = DataDictionaryEntry(path="ids/path", documentation="docs", units="m/s^2")
    assert dd_entry.units == "m.s^-2"


def test_no_units_validator():
    dd_entry = DataDictionaryEntry(path="ids/path", documentation="docs")
    assert dd_entry.units == ""


if __name__ == "__main__":
    pytest.main([__file__])
