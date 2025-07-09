import pytest
import importlib.util
from pydantic import ValidationError

# Skip all tests in this file since they require whoosh (legacy dependency)
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("whoosh") is None,
    reason="Skipping legacy tests that require whoosh",
)

# Only import whoosh modules if available
if importlib.util.find_spec("whoosh") is not None:
    import whoosh.fields
    import whoosh.index
    from imas_mcp.whoosh_index import DataDictionaryEntry, SearchResult, WhooshIndex


def test_whoosh_index_with_dir(tmp_path):
    """Test the WhooshIndex class."""
    dirname = tmp_path
    dirname.mkdir(parents=True, exist_ok=True)

    # Create a Whoosh schema
    schema = whoosh.fields.Schema(
        title=whoosh.fields.TEXT(stored=True),
    )

    # create a whoosh index
    whoosh.index.create_in(dirname, schema)

    # Create an instance of WhooshIndex
    whoosh_index = WhooshIndex(dirname)
    _ = whoosh_index._index  # Access the index to ensure it is created

    # Check if the index is created with the correct schema
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

    # Create an instance of WhooshIndex with a schema
    whoosh_index = WhooshIndex(
        tmp_path, schema=schema, document_model=DataDictionaryEntry
    )

    assert whoosh_index.schema == schema
    assert whoosh_index.document_model == DataDictionaryEntry


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


def test_page_size(whoosh_index):
    """Test the page size of the WhooshIndex."""

    query_string = "(equilibrium OR core_profiles OR magnetics)"
    result = whoosh_index.search_by_keywords(query_string, page_size=10)
    print("***", len(result))
    assert len(result) == 4

    result = whoosh_index.search_by_keywords(query_string, page_size=2)
    assert len(result) == 2

    result = whoosh_index.search_by_keywords(query_string, page_size=1, page=2)
    assert len(result) == 1
    assert result[0].path == "equilibrium/time_slice/profiles_1d/q_safety_factor"


def test_search_by_keywords(whoosh_index):
    """Test the search_by_keywords method using Whoosh."""
    # Test search for safety factor
    results = whoosh_index.search_by_keywords("safety factor profile")
    assert len(results) >= 1
    paths = [result.path for result in results]
    assert "equilibrium/time_slice/profiles_1d/q_safety_factor" in paths

    # Test search for electron density
    results = whoosh_index.search_by_keywords("electron density profile")
    assert len(results) >= 1
    paths = [result.path for result in results]
    assert "core_profiles/time_slice/electrons/density" in paths

    # Test search for plasma current
    results = whoosh_index.search_by_keywords("plasma current measurement")
    assert len(results) >= 1
    paths = [result.path for result in results]
    assert "magnetics/time_trace/ip" in paths

    results = whoosh_index.search_by_keywords("documentation:ion")
    assert results[0].path == "core_profiles/time_slice/ions/1/density"

    # Test fuzzy search with intentional typos
    fuzzy_results = whoosh_index.search_by_keywords("saftey~2 factor profile")
    assert len(fuzzy_results) >= 1
    fuzzy_paths = [result.path for result in fuzzy_results]
    assert "equilibrium/time_slice/profiles_1d/q_safety_factor" in fuzzy_paths

    # Test fuzzy search for electron density with typo
    fuzzy_results = whoosh_index.search_by_keywords("electrn~ densty~")
    assert len(fuzzy_results) >= 1
    fuzzy_paths = [result.path for result in fuzzy_results]
    assert "core_profiles/time_slice/electrons/density" in fuzzy_paths

    # Test fuzzy search for plasma current with typo
    fuzzy_results = whoosh_index.search_by_keywords("plazma curent", fuzzy=True)
    assert len(fuzzy_results) >= 1
    fuzzy_paths = [result.path for result in fuzzy_results]
    assert "magnetics/time_trace/ip" in fuzzy_paths


def test_unit_error(tmp_path):
    """Test the WhooshIndex class with a unit error."""
    dirname = tmp_path

    # Create an instance of WhooshIndex
    whoosh_index = WhooshIndex(dirname, skip_unit_parsing=False)

    # Add a document to the index
    with whoosh_index.writer():
        whoosh_index += {
            "path": "pf_active/coil/velocity",
            "documentation": "coil velocity",
            "units": "invalid unit",
        }
    assert whoosh_index._unit_error["invalid unit"] == ["pf_active/coil/velocity"]


def test_invalid_imas_units(tmp_path):
    whoosh_index = WhooshIndex(tmp_path)

    with whoosh_index.writer():
        whoosh_index += {
            "path": "pf_active/something/with/an/invalid/imas/unit",
            "documentation": "unit wrangling attribute",
            "units": "m^dimension",  # we need to allow this for compliance with the DD
        }
    assert not whoosh_index._unit_error


def test_document_model_errors(tmp_path):
    whoosh_index = WhooshIndex(tmp_path)

    with whoosh_index.writer():
        with pytest.raises(ValidationError):
            # Attempt to add a document without documentation
            whoosh_index += {
                "path": "path/to/document",
            }

        with pytest.raises(ValidationError):
            whoosh_index += {
                "path": 123,  # Invalid path type
                "documentation": "This is a test document.",
            }

        with pytest.raises(ValidationError):
            whoosh_index += {
                "path": "a/sample/path",
                "documentation": ["document as list"],  # invalid documentation type
            }


def test_search_by_exact_path(whoosh_index):
    """Test the search_by_exact_path method."""
    # Test exact path search
    result = whoosh_index.search_by_exact_path(
        "equilibrium/time_slice/profiles_1d/q_safety_factor"
    )
    assert isinstance(result, SearchResult)
    assert result.path == "equilibrium/time_slice/profiles_1d/q_safety_factor"

    # Test non-existing path
    result = whoosh_index.search_by_exact_path("non/existing/path")
    assert result is None

    # Test with a different path
    result = whoosh_index.search_by_exact_path(
        "core_profiles/time_slice/electrons/density"
    )
    assert isinstance(result, SearchResult)
    assert result.path == "core_profiles/time_slice/electrons/density"


def test_search_by_path_prefix(whoosh_index):
    """Test the search_by_path_prefix method."""
    # Test path prefix search
    results = whoosh_index.search_by_path_prefix("equilibrium/time_slice")
    assert len(results) >= 1
    assert all(isinstance(result, SearchResult) for result in results)
    assert any(
        result.path == "equilibrium/time_slice/profiles_1d/q_safety_factor"
        for result in results
    )

    # Test non-existing prefix
    results = whoosh_index.search_by_path_prefix("non/existing/prefix")
    assert len(results) == 0

    # Test with a different prefix
    results = whoosh_index.search_by_path_prefix("core_profiles/time_slice")
    assert len(results) >= 1
    assert any(
        result.path == "core_profiles/time_slice/electrons/density"
        for result in results
    )


def test_filter_search_results(whoosh_index):
    """Test the filter_search_results method."""
    # Test filtering by path prefix
    results = whoosh_index.search_by_keywords("equilibrium")

    # Test no filter case
    filtered_results = whoosh_index.filter_search_results(results, filters={})
    assert filtered_results == results

    filtered_results = whoosh_index.filter_search_results(
        results, filters={"path": "^equilibrium/time_slice"}
    )
    assert len(filtered_results) >= 1
    assert all(
        result.path.startswith("equilibrium/time_slice") for result in filtered_results
    )

    # Test filtering by documentation keyword
    filtered_results = whoosh_index.filter_search_results(
        results,
        filters={"documentation": r".*safety factor.*"},  # ignore case
    )
    assert len(filtered_results) >= 1
    assert all(
        "safety factor" in result.documentation.lower() for result in filtered_results
    )

    # Test empty regex filter
    filtered_results = whoosh_index.filter_search_results(
        results,
        filters={"documentation": r".*safety car.*"},  # ignore case
    )
    assert len(filtered_results) == 0

    # Test invalid regex pattern
    filtered_results = whoosh_index.filter_search_results(
        results,
        filters={"documentation": r"[a-z"},  # invalid regex
    )
    assert len(filtered_results) == 0

    # Test exact unit match
    filtered_results = whoosh_index.filter_search_results(
        results,
        filters={"units": ""},  # dimensionless units
    )
    assert len(filtered_results) >= 1
    assert all(result.units == "" for result in filtered_results)

    # Test simple string no match
    filtered_results = whoosh_index.filter_search_results(
        results,
        filters={"units": "m^2"},
        regex=False,  # non-matching units
    )
    assert len(filtered_results) == 0


if __name__ == "__main__":
    pytest.main([__file__])
