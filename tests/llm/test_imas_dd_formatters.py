"""Direct-call tests for graph-backed IMAS DD formatters."""

from imas_codex.core.data_model import IdsNode
from imas_codex.llm.search_formatters import (
    format_cluster_report,
    format_export_domain_report,
    format_fetch_paths_report,
    format_search_imas_report,
)
from imas_codex.models.constants import SearchMode
from imas_codex.models.error_models import ToolError
from imas_codex.models.result_models import FetchPathsResult, SearchPathsResult
from imas_codex.search.search_strategy import SearchHit


def test_format_search_imas_report_handles_tool_error():
    result = ToolError(
        error="Unexpected error: backend unavailable",
        suggestions=["Retry the operation"],
        fallback_data={"message": "Search failed"},
    )

    formatted = format_search_imas_report(result)

    assert "Error: Unexpected error: backend unavailable" in formatted
    assert "Suggestions:" in formatted
    assert "Retry the operation" in formatted
    assert "Fallback: Search failed" in formatted


def test_format_cluster_report_handles_tool_error():
    result = ToolError(error="Unexpected error: cluster search failed")

    formatted = format_cluster_report(result)

    assert formatted == "Error: Unexpected error: cluster search failed"


def test_format_fetch_paths_report_handles_string_cluster_labels():
    result = FetchPathsResult(
        nodes=[
            IdsNode(
                path="equilibrium/time_slice/boundary/psi",
                documentation="Boundary flux",
                ids_name="equilibrium",
                data_type="FLT_1D",
                units="Wb",
                cluster_labels=["Equilibrium Boundary", "Flux Surfaces"],
            )
        ],
        summary={"total_requested": 1, "fetched": 1, "not_found": 0},
    )

    formatted = format_fetch_paths_report(result)

    assert "## IMAS Path Details (1 fetched)" in formatted
    assert 'Clusters: "Equilibrium Boundary", "Flux Surfaces"' in formatted


def test_format_export_domain_report_handles_tool_error():
    result = ToolError(error="Unexpected error: domain export failed")

    formatted = format_export_domain_report(result)

    assert formatted == "Error: Unexpected error: domain export failed"


def test_format_export_domain_report_shows_resolution_details():
    formatted = format_export_domain_report(
        {
            "domain": "magnetics",
            "resolved_domains": ["magnetic_field_diagnostics"],
            "resolution": "ids_name:magnetics",
            "total_paths": 1,
            "ids_count": 1,
            "by_ids": {
                "magnetics": [
                    {
                        "path": "magnetics/flux_loop/flux/data",
                        "documentation": "Flux loop data",
                        "units": "Wb",
                    }
                ]
            },
        }
    )

    assert "Resolved domains: magnetic_field_diagnostics" in formatted
    assert "Resolution: ids_name:magnetics" in formatted
    assert "magnetics/flux_loop/flux/data" in formatted


def test_format_export_domain_report_shows_no_match_reason():
    formatted = format_export_domain_report(
        {
            "domain": "unknown_domain",
            "resolved_domains": [],
            "resolution": "no_match",
            "total_paths": 0,
            "ids_count": 0,
            "by_ids": {},
            "error": "No physics domain found matching 'unknown_domain'.",
        }
    )

    assert "Resolution: no_match" in formatted
    assert "No physics domain found matching 'unknown_domain'." in formatted


def test_format_search_imas_report_handles_success_result():
    hit = SearchHit(
        score=0.92,
        rank=1,
        search_mode=SearchMode.AUTO,
        highlights="",
        path="core_profiles/profiles_1d/electrons/temperature",
        documentation="Electron temperature profile",
        ids_name="core_profiles",
        units="eV",
        data_type="FLT_1D",
        coordinates=["core_profiles/profiles_1d/grid/rho_tor_norm"],
    )
    result = SearchPathsResult(
        hits=[hit],
        summary={"hits_returned": 1},
        query="electron temperature",
        search_mode=SearchMode.AUTO,
        physics_domains=["core_transport"],
    )

    formatted = format_search_imas_report(result)

    assert "## IMAS Paths (1 matches)" in formatted
    assert "core_profiles/profiles_1d/electrons/temperature" in formatted
    assert "Coordinates: core_profiles/profiles_1d/grid/rho_tor_norm" in formatted
