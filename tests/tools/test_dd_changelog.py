"""Tests for the DD changelog tool and its report formatter.

``VersionTool.get_dd_changelog`` ranks IMASNode paths by how volatile they
have been across DD versions. These tests pin the query parameter contract,
result shaping, and error handling against a mocked GraphClient, and the
markdown formatter's header, row, and truncation-hint behaviour.

No live Neo4j required — all graph access goes through ``MagicMock``.
"""

from unittest.mock import MagicMock

import pytest

from imas_codex.llm.search_formatters import format_dd_changelog_report
from imas_codex.tools.version_tool import VersionTool


def _row(
    path: str = "equilibrium/time_slice/profiles_1d/psi",
    ids: str = "equilibrium",
    **overrides,
) -> dict:
    """Build one changelog row in the shape returned by the Cypher query."""
    row = {
        "path": path,
        "ids": ids,
        "lifecycle_status": "active",
        "change_count": 3,
        "type_variety": 2,
        "change_types": ["units_changed", "documentation_changed"],
        "was_renamed": 0,
        "volatility_score": 7,
    }
    row.update(overrides)
    return row


# ============================================================================
# VersionTool.get_dd_changelog
# ============================================================================


class TestGetDDChangelog:
    """Query contract and result shaping of the changelog tool."""

    @pytest.mark.asyncio
    async def test_default_call_passes_null_filters_and_limit_50(self):
        gc = MagicMock()
        gc.query.return_value = []

        result = await VersionTool(gc).get_dd_changelog()

        gc.query.assert_called_once()
        _, kwargs = gc.query.call_args
        assert kwargs == {
            "ids_filter": None,
            "from_version": None,
            "to_version": None,
            "limit": 50,
        }
        assert result == {
            "results": [],
            "total": 0,
            "ids_filter": None,
            "version_range": None,
            "limit": 50,
        }

    @pytest.mark.asyncio
    async def test_filters_are_forwarded_as_query_parameters(self):
        gc = MagicMock()
        gc.query.return_value = []

        result = await VersionTool(gc).get_dd_changelog(
            ids_filter="equilibrium",
            from_version="3.30.0",
            to_version="3.39.0",
            limit=10,
        )

        _, kwargs = gc.query.call_args
        assert kwargs == {
            "ids_filter": "equilibrium",
            "from_version": "3.30.0",
            "to_version": "3.39.0",
            "limit": 10,
        }
        assert result["ids_filter"] == "equilibrium"
        assert result["limit"] == 10
        assert result["version_range"] == {"from": "3.30.0", "to": "3.39.0"}

    @pytest.mark.asyncio
    async def test_open_ended_version_range_uses_empty_string(self):
        gc = MagicMock()
        gc.query.return_value = []

        lower_only = await VersionTool(gc).get_dd_changelog(from_version="3.30.0")
        upper_only = await VersionTool(gc).get_dd_changelog(to_version="4.0.0")

        assert lower_only["version_range"] == {"from": "3.30.0", "to": ""}
        assert upper_only["version_range"] == {"from": "", "to": "4.0.0"}

    @pytest.mark.asyncio
    async def test_query_is_parameterised_not_interpolated(self):
        gc = MagicMock()
        gc.query.return_value = []

        await VersionTool(gc).get_dd_changelog(ids_filter="equilibrium", limit=5)

        (cypher,), _ = gc.query.call_args
        for placeholder in ("$ids_filter", "$from_version", "$to_version", "$limit"):
            assert placeholder in cypher
        assert "equilibrium" not in cypher
        assert "IMASNodeChange" in cypher
        assert "ORDER BY volatility_score DESC" in cypher

    @pytest.mark.asyncio
    async def test_rows_are_copied_into_results_in_order(self):
        gc = MagicMock()
        rows = [
            _row(volatility_score=9),
            _row(path="equilibrium/time", volatility_score=4),
        ]
        gc.query.return_value = rows

        result = await VersionTool(gc).get_dd_changelog()

        assert result["results"] == rows
        assert result["total"] == 2
        # Results are fresh dicts, not the driver's record objects.
        assert result["results"][0] is not rows[0]

    @pytest.mark.asyncio
    async def test_none_rows_yield_empty_results(self):
        gc = MagicMock()
        gc.query.return_value = None

        result = await VersionTool(gc).get_dd_changelog()

        assert result["results"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_query_failure_is_returned_not_raised(self):
        gc = MagicMock()
        gc.query.side_effect = RuntimeError("bolt connection refused")

        result = await VersionTool(gc).get_dd_changelog()

        assert result == {"error": "Failed to query changelog: bolt connection refused"}


# ============================================================================
# format_dd_changelog_report
# ============================================================================


class TestFormatDDChangelogReport:
    """Markdown rendering of the changelog result."""

    def test_error_result_renders_error_line(self):
        assert format_dd_changelog_report({"error": "boom"}) == "Error: boom"

    def test_header_without_filters(self):
        report = format_dd_changelog_report(
            {
                "results": [_row()],
                "total": 1,
                "ids_filter": None,
                "version_range": None,
                "limit": 50,
            }
        )

        assert report.splitlines()[0] == "## DD Changelog — 1 most volatile paths"
        assert "(IDS:" not in report
        assert "Version range" not in report

    def test_header_with_ids_and_closed_version_range(self):
        report = format_dd_changelog_report(
            {
                "results": [],
                "total": 0,
                "ids_filter": "equilibrium",
                "version_range": {"from": "3.30.0", "to": "3.39.0"},
                "limit": 50,
            }
        )

        assert "## DD Changelog — 0 most volatile paths (IDS: equilibrium)" in report
        assert "Version range: 3.30.0 → 3.39.0" in report

    def test_open_ended_range_labels(self):
        def render(version_range: dict) -> str:
            return format_dd_changelog_report(
                {
                    "results": [],
                    "total": 0,
                    "version_range": version_range,
                    "limit": 50,
                }
            )

        assert "Version range: 3.30.0 → latest" in render({"from": "3.30.0", "to": ""})
        assert "Version range: earliest → 4.0.0" in render({"from": "", "to": "4.0.0"})
        assert "Version range" not in render({"from": "", "to": ""})

    def test_empty_results_still_render_table_header(self):
        report = format_dd_changelog_report({"results": [], "total": 0, "limit": 50})

        assert (
            "| Rank | Path | IDS | Lifecycle | Changes | Types | Renamed | Score |"
            in report
        )
        table_lines = [ln for ln in report.splitlines() if ln.startswith("|")]
        assert len(table_lines) == 2  # header + separator only

    def test_row_rendering(self):
        renamed = _row(
            path="core_profiles/time",
            ids="core_profiles",
            was_renamed=1,
            lifecycle_status="obsolescent",
            change_types=["renamed"],
            change_count=1,
            volatility_score=6,
        )
        report = format_dd_changelog_report(
            {"results": [_row(), renamed], "total": 2, "limit": 50}
        )

        assert (
            "| 1 | `equilibrium/time_slice/profiles_1d/psi` | equilibrium |  | 3 "
            "| units_changed, documentation_changed |  | 7 |"
        ) in report
        assert (
            "| 2 | `core_profiles/time` | core_profiles | obsolescent | 1 "
            "| renamed | ✓ | 6 |"
        ) in report

    def test_missing_lifecycle_and_types_are_tolerated(self):
        report = format_dd_changelog_report(
            {
                "results": [_row(lifecycle_status=None, change_types=None)],
                "total": 1,
                "limit": 50,
            }
        )

        assert (
            "| 1 | `equilibrium/time_slice/profiles_1d/psi` | equilibrium |  | 3 "
            "|  |  | 7 |"
        ) in report

    def test_truncation_hint_only_when_limit_reached(self):
        at_limit = format_dd_changelog_report(
            {"results": [_row(), _row(path="x")], "total": 2, "limit": 2}
        )
        under_limit = format_dd_changelog_report(
            {"results": [_row()], "total": 1, "limit": 50}
        )

        assert "*Showing top 2 — use `limit` to see more.*" in at_limit
        assert "Showing top" not in under_limit
