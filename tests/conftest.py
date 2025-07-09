import pytest
import importlib.util


@pytest.fixture(scope="session")
def whoosh_index(tmp_path_factory):
    """Fixture to create a test WhooshIndex with sample paths and documentation."""
    # Check if whoosh is available
    if importlib.util.find_spec("whoosh") is None:
        pytest.skip("Whoosh not available - skipping legacy whoosh_index fixture")

    # Import here to avoid import errors at module level
    from imas_mcp.whoosh_index import WhooshIndex

    # Create a temporary directory for the index
    tmp_dir = tmp_path_factory.mktemp("whoosh_index")
    index = WhooshIndex(dirname=tmp_dir)

    # Add some sample paths with documentation
    index.add_document(
        {
            "path": "equilibrium/time_slice/profiles_1d/q_safety_factor",
            "documentation": "Safety factor q profile as a function of normalized poloidal flux.",
        }
    )

    document_batch = [
        {
            "path": "equilibrium/time_slice/profiles_1d/q_safety_factor",
            "documentation": "Safety factor q profile as a function of normalized poloidal flux.",
            "units": "",
        },
        {
            "path": "core_profiles/time_slice/electrons/density",
            "documentation": "Electron density profile as a function of normalized poloidal flux.",
        },
        {
            "path": "core_profiles/time_slice/ions/1/density",
            "documentation": "Ion density profile for the first ion species as a function of normalized poloidal flux.",
        },
        {
            "path": "magnetics/time_trace/ip",
            "documentation": "Plasma current time trace measured by magnetic diagnostics.",
        },
    ]
    index.add_document_batch(document_batch)

    yield index
