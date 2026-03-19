"""Direct formatter tests for DD version and error-field reports."""

from imas_codex.llm.server import (
    _format_dd_versions_report,
    _format_error_fields_report,
    _format_version_context_report,
)


def test_format_dd_versions_report():
    formatted = _format_dd_versions_report(
        {
            "current_version": "4.1.0",
            "version_range": "3.42.0 - 4.1.0",
            "version_count": 3,
            "versions": ["3.42.0", "4.0.0", "4.1.0"],
        }
    )

    assert "DD Version Metadata" in formatted
    assert "Current version: 4.1.0" in formatted
    assert "Version chain: 3.42.0 -> 4.0.0 -> 4.1.0" in formatted


def test_format_version_context_report_distinguishes_not_found_and_no_changes():
    formatted = _format_version_context_report(
        {
            "paths": {
                "equilibrium/time_slice/profiles_1d/psi": {
                    "change_count": 0,
                    "notable_changes": [],
                }
            },
            "total_paths": 2,
            "paths_found": ["equilibrium/time_slice/profiles_1d/psi"],
            "paths_without_changes": ["equilibrium/time_slice/profiles_1d/psi"],
            "paths_with_changes": 0,
            "graph_change_nodes_seen": 0,
            "not_found": ["fake/path"],
        }
    )

    assert "2 paths queried, 1 found, 0 with changes" in formatted
    assert (
        "**equilibrium/time_slice/profiles_1d/psi**: no metadata changes recorded"
        in formatted
    )
    assert "Not found: fake/path" in formatted
    assert (
        "Paths without notable changes: equilibrium/time_slice/profiles_1d/psi"
        in formatted
    )


def test_format_error_fields_report():
    formatted = _format_error_fields_report(
        {
            "path": "equilibrium/time_slice/profiles_1d/psi",
            "count": 1,
            "not_found": False,
            "error_fields": [
                {
                    "path": "equilibrium/time_slice/profiles_1d/psi_error_upper",
                    "error_type": "upper",
                    "documentation": "Upper error bound",
                }
            ],
        }
    )

    assert "Error fields for equilibrium/time_slice/profiles_1d/psi:" in formatted
    assert "psi_error_upper (upper) - Upper error bound" in formatted
