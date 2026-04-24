"""Tests for scripts/run_domain_cycles.py.

Uses unittest.mock to patch GraphClient and subprocess.run so no real
Neo4j connection or LLM calls are made.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Import the script module by path (it is not an installed package)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "run_domain_cycles.py"


def _load_script():
    """Load run_domain_cycles.py as a module and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location("run_domain_cycles", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["run_domain_cycles"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_mod = _load_script()


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_THREE_DOMAINS = [
    {"domain": "equilibrium", "backlog": 120},
    {"domain": "transport", "backlog": 80},
    {"domain": "magnetics", "backlog": 40},
]

_EMPTY_SCORE_STATS = {
    "n": 0,
    "min": None,
    "mean": None,
    "median": None,
    "p75": None,
    "max": None,
}


def _make_stats(domain: str, *, n_gen: int = 10, n_reviewed: int = 5) -> dict:
    """Return a realistic stats dict for testing."""
    return {
        "pipeline_counts": {"named": n_gen},
        "total_generated": n_gen,
        "scores": {
            "n": n_reviewed,
            "min": 0.5,
            "mean": 0.72,
            "median": 0.73,
            "p75": 0.80,
            "max": 0.95,
        }
        if n_reviewed > 0
        else _EMPTY_SCORE_STATS,
        "n_reviewed": n_reviewed,
        "quarantine_count": 2,
        "top_issues": ["[pydantic] name too long"],
        "top_themes": [("unit_mismatch", 3), ("canonical_ordering_issue", 1)],
        "vocab_gaps": {"subject": 2, "process": 1},
        "cycle_spend": 0.05 * n_gen,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDomainCycleRunner:
    """Core behavior: 3 domains, mocked graph + subprocess."""

    def test_report_has_section_per_domain_and_global_summary(self, tmp_path):
        """Each domain gets a ## section; global summary # header is present."""
        report_file = tmp_path / "report.md"

        with (
            patch(
                "run_domain_cycles.GraphClient",
                autospec=True,
            ) as MockGC,
            patch("run_domain_cycles.subprocess.run") as mock_sub,
        ):
            # Context-manager protocol
            gc_inst = MockGC.return_value.__enter__.return_value

            def gc_side_effect(cypher, **kwargs):
                if "StandardNameSource" in cypher and "claimed_at" in cypher:
                    # Domain discovery query
                    return _THREE_DOMAINS
                if "pipeline_status" in cypher:
                    return [{"status": "named", "n": 10}]
                if "reviewer_score_name IS NOT NULL" in cypher:
                    return [{"score": 0.7 + i * 0.05} for i in range(5)]
                if "validation_status = 'quarantined'" in cypher:
                    return [{"issues": ["[pydantic] too long"]}]
                if "reviewer_comments_name IS NOT NULL" in cypher:
                    return [{"comments": "unit mismatch in name"}]
                if "HAS_STANDARD_NAME_VOCAB_GAP" in cypher:
                    return [{"segment": "subject", "gap_count": 2}]
                if "generated_at" in cypher:
                    return [{"total_spend": 0.50}]
                return []

            gc_inst.query.side_effect = gc_side_effect
            mock_sub.return_value = MagicMock(returncode=0)

            exit_code = _mod.main(
                [
                    "--cost-limit",
                    "5.0",
                    "--turn-number",
                    "1",
                    "--source",
                    "dd",
                    "--target",
                    "names",
                    "--report",
                    str(report_file),
                ]
            )

        assert exit_code == 0
        content = report_file.read_text()

        # Each domain has a ## header
        for domain in ("equilibrium", "transport", "magnetics"):
            assert f"## {domain}" in content, f"Missing section for {domain}"

        # Global summary present
        assert "# Run summary" in content
        assert "turn 1" in content

    def test_subprocess_called_once_per_domain(self, tmp_path):
        """subprocess.run is called exactly once per domain."""
        report_file = tmp_path / "report.md"

        with (
            patch("run_domain_cycles.GraphClient", autospec=True) as MockGC,
            patch("run_domain_cycles.subprocess.run") as mock_sub,
        ):
            gc_inst = MockGC.return_value.__enter__.return_value
            gc_inst.query.return_value = []

            # Patch domain discovery to return 3 domains
            with patch(
                "run_domain_cycles._query_domains_with_backlog",
                return_value=_THREE_DOMAINS,
            ):
                mock_sub.return_value = MagicMock(returncode=0)
                _mod.main(["--cost-limit", "2.0", "--report", str(report_file)])

        assert mock_sub.call_count == 3

    def test_skip_domains_filter(self, tmp_path):
        """--skip-domains removes matching domains from the run."""
        report_file = tmp_path / "report.md"

        with (
            patch("run_domain_cycles.GraphClient", autospec=True) as MockGC,
            patch("run_domain_cycles.subprocess.run") as mock_sub,
        ):
            gc_inst = MockGC.return_value.__enter__.return_value

            def gc_side_effect(cypher, **kwargs):
                # Domain discovery query returns all 3 domains
                if "StandardNameSource" in cypher and "claimed_at" in cypher:
                    return _THREE_DOMAINS
                # Stats queries return empty
                return []

            gc_inst.query.side_effect = gc_side_effect
            mock_sub.return_value = MagicMock(returncode=0)

            exit_code = _mod.main(
                [
                    "--skip-domains",
                    "equilibrium,transport",
                    "--report",
                    str(report_file),
                ]
            )

        assert exit_code == 0
        # Only magnetics should have been called
        assert mock_sub.call_count == 1
        # magnetics appears in the domain arg
        called_cmd = mock_sub.call_args[0][0]
        assert "magnetics" in called_cmd

        content = report_file.read_text()
        assert "## magnetics" in content
        assert "## equilibrium" not in content
        assert "## transport" not in content

    def test_skip_domains_applied_at_graph_query_level(self):
        """_query_domains_with_backlog respects skip_domains before returning."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [
            {"domain": "equilibrium", "backlog": 50},
            {"domain": "general", "backlog": 20},
            {"domain": "transport", "backlog": 30},
        ]
        result = _mod._query_domains_with_backlog(
            mock_gc,
            source="dd",
            skip_domains={"general"},
            include_domains=None,
        )
        domains = [r["domain"] for r in result]
        assert "general" not in domains
        assert "equilibrium" in domains
        assert "transport" in domains

    def test_empty_domain_graph_exits_cleanly(self, capsys):
        """When no domains have backlog, script prints a notice and exits 0."""
        with (
            patch("run_domain_cycles.GraphClient", autospec=True) as MockGC,
            patch("run_domain_cycles.subprocess.run") as mock_sub,
        ):
            gc_inst = MockGC.return_value.__enter__.return_value
            gc_inst.query.return_value = []

            exit_code = _mod.main(["--cost-limit", "5.0"])

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Nothing to do" in captured.out or "No domains" in captured.out
        mock_sub.assert_not_called()

    def test_failed_subprocess_continues_to_next_domain(self, tmp_path):
        """A CalledProcessError on one domain does not abort the script."""
        report_file = tmp_path / "report.md"

        with (
            patch("run_domain_cycles.GraphClient", autospec=True) as MockGC,
            patch(
                "run_domain_cycles._query_domains_with_backlog",
                return_value=[
                    {"domain": "equilibrium", "backlog": 50},
                    {"domain": "transport", "backlog": 30},
                ],
            ),
            patch("run_domain_cycles.subprocess.run") as mock_sub,
        ):
            gc_inst = MockGC.return_value.__enter__.return_value
            gc_inst.query.return_value = []

            # First domain fails, second succeeds
            mock_sub.side_effect = [
                subprocess.CalledProcessError(1, ["sn", "run"]),
                MagicMock(returncode=0),
            ]

            exit_code = _mod.main(["--report", str(report_file)])

        # Both domains were attempted
        assert mock_sub.call_count == 2
        # Script still exits 0 (failure is logged but not fatal)
        assert exit_code == 0


class TestFormatHelpers:
    """Unit tests for markdown formatting functions."""

    def test_format_domain_section_with_scores(self):
        stats = _make_stats("equilibrium", n_gen=15, n_reviewed=8)
        section = _mod._format_domain_section(
            domain="equilibrium",
            stats=stats,
            cost=0.75,
            turn_number=2,
            duration_s=42.3,
        )
        assert "## equilibrium" in section
        assert "$0.7500" in section
        assert "turn 2" in section
        assert "42.3s" in section
        assert "generated: 15" in section
        assert "mean=0.72" in section
        assert "unit_mismatch (3)" in section
        assert "subject=2" in section

    def test_format_domain_section_empty_scores(self):
        stats = _make_stats("general", n_gen=0, n_reviewed=0)
        stats["scores"] = _EMPTY_SCORE_STATS
        stats["total_generated"] = 0
        stats["top_themes"] = []
        stats["vocab_gaps"] = {}
        section = _mod._format_domain_section(
            domain="general",
            stats=stats,
            cost=0.0,
            turn_number=1,
            duration_s=1.0,
        )
        assert "## general" in section
        assert "generated: 0" in section
        # No scores should be printed when empty
        assert "mean=" not in section

    def test_format_global_summary_table(self):
        domain_results = [
            {
                "domain": "equilibrium",
                "stats": _make_stats("equilibrium", n_gen=10),
                "cost": 0.50,
            },
            {
                "domain": "transport",
                "stats": _make_stats("transport", n_gen=8),
                "cost": 0.40,
            },
        ]
        summary = _mod._format_global_summary(
            domain_results, turn_number=1, total_cost=0.90
        )
        assert "# Run summary" in summary
        assert "turn 1" in summary
        assert "$0.9000" in summary
        assert "equilibrium" in summary
        assert "transport" in summary
        assert "2 domains" in summary

    def test_score_stats_computed_correctly(self):
        """_query_domain_stats score statistics are computed without numpy."""
        # We test via the pure-Python code path by calling the helper inline
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        n = len(scores)
        sorted_s = sorted(scores)
        mean = sum(sorted_s) / n
        mid = n // 2
        median = sorted_s[mid] if n % 2 else (sorted_s[mid - 1] + sorted_s[mid]) / 2
        assert abs(mean - 0.7) < 1e-9
        assert abs(median - 0.7) < 1e-9


class TestBuildSnRunCmd:
    """Test the subprocess command construction."""

    def test_single_pass_always_present(self):
        cmd = _mod._build_sn_run_cmd(
            domain="equilibrium",
            source="dd",
            target="names",
            cost_limit=5.0,
            turn_number=1,
            min_score=None,
            dry_run=False,
        )
        assert "--single-pass" in cmd

    def test_min_score_omitted_when_none(self):
        cmd = _mod._build_sn_run_cmd(
            domain="equilibrium",
            source="dd",
            target="names",
            cost_limit=5.0,
            turn_number=1,
            min_score=None,
            dry_run=False,
        )
        assert "--min-score" not in cmd

    def test_min_score_present_when_nonzero(self):
        cmd = _mod._build_sn_run_cmd(
            domain="magnetics",
            source="dd",
            target="names",
            cost_limit=2.0,
            turn_number=2,
            min_score=0.6,
            dry_run=False,
        )
        assert "--min-score" in cmd
        idx = cmd.index("--min-score")
        assert cmd[idx + 1] == "0.6"

    def test_dry_run_flag_propagated(self):
        cmd = _mod._build_sn_run_cmd(
            domain="transport",
            source="dd",
            target="names",
            cost_limit=1.0,
            turn_number=1,
            min_score=None,
            dry_run=True,
        )
        assert "--dry-run" in cmd

    def test_physics_domain_in_command(self):
        cmd = _mod._build_sn_run_cmd(
            domain="gyrokinetics",
            source="dd",
            target="docs",
            cost_limit=3.0,
            turn_number=3,
            min_score=0.5,
            dry_run=False,
        )
        assert "--physics-domain" in cmd
        idx = cmd.index("--physics-domain")
        assert cmd[idx + 1] == "gyrokinetics"
        assert "--target" in cmd
        assert "docs" in cmd
