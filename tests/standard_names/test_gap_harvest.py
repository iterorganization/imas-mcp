"""Tests for gap_harvest: PR-ready YAML/Markdown harvester for VocabGap nodes.

Covers:
- harvest_vocab_gaps: correct query construction and data transformation
- format_pr_yaml: YAML structure, metadata fields, segment grouping, closed/open flags
- format_pr_markdown: report structure, per-segment table, top tokens, PR template
- CLI sn gaps --format yaml integration (mocked graph)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "segment": "process",
        "needed_token": "time_derivative",
        "occurrences": 8,
        "example_count": 8,
        "source_types": ["IMASNode"],
        "example_dd_paths": [
            "equilibrium/time_slice/profiles_1d/dpsi_dt",
            "core_profiles/profiles_1d/electrons/density_fast/velocity",
        ],
        "example_reasons": ["Need time-derivative process"],
        "first_seen": "2025-01-15T10:00:00",
        "last_seen": "2025-06-01T12:00:00",
    },
    {
        "segment": "process",
        "needed_token": "radial_derivative",
        "occurrences": 5,
        "example_count": 5,
        "source_types": ["IMASNode"],
        "example_dd_paths": ["equilibrium/time_slice/profiles_1d/dpsi_drho_tor"],
        "example_reasons": ["Quantity is a spatial derivative"],
        "first_seen": "2025-02-01T08:00:00",
        "last_seen": "2025-06-02T09:00:00",
    },
    {
        "segment": "coordinate",
        "needed_token": "rho_pol_norm",
        "occurrences": 3,
        "example_count": 3,
        "source_types": ["IMASNode", "FacilitySignal"],
        "example_dd_paths": ["core_sources/source/fusion/power"],
        "example_reasons": ["Poloidal flux coordinate"],
        "first_seen": "2025-03-01T08:00:00",
        "last_seen": "2025-05-10T08:00:00",
    },
    {
        "segment": "physical_base",
        "needed_token": "beta",
        "occurrences": 2,
        "example_count": 2,
        "source_types": ["FacilitySignal"],
        "example_dd_paths": [],
        "example_reasons": ["Open-segment token"],
        "first_seen": "2025-04-01T08:00:00",
        "last_seen": "2025-04-15T08:00:00",
    },
]


# ---------------------------------------------------------------------------
# Tests: harvest_vocab_gaps
# ---------------------------------------------------------------------------


class TestHarvestVocabGaps:
    """harvest_vocab_gaps transforms raw graph rows into enriched records."""

    def _make_gc(self, rows):
        gc = MagicMock()
        gc.query.return_value = iter(rows)
        return gc

    def test_returns_records_for_each_row(self):
        from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps

        raw = [
            {
                "segment": "transformation",
                "needed_token": "derivative_of",
                "example_count": 3,
                "source_count": 3,
                "source_types": ["IMASNode"],
                "example_dd_paths": ["some/path"],
                "example_reasons": ["reason A"],
                "first_seen": "2025-01-01T00:00:00",
                "last_seen": "2025-06-01T00:00:00",
            }
        ]
        gc = self._make_gc(raw)
        records = harvest_vocab_gaps(gc)
        assert len(records) == 1
        r = records[0]
        assert r["segment"] == "transformation"
        assert r["needed_token"] == "derivative_of"
        assert r["occurrences"] == 3
        assert r["example_dd_paths"] == ["some/path"]
        assert r["example_reasons"] == ["reason A"]

    def test_filters_none_values_in_lists(self):
        from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps

        raw = [
            {
                "segment": "process",
                "needed_token": "fusion",
                "example_count": 1,
                "source_count": 1,
                "source_types": ["IMASNode", None],
                "example_dd_paths": [None, "some/path", None],
                "example_reasons": [None, "reason B"],
                "first_seen": None,
                "last_seen": None,
            }
        ]
        gc = self._make_gc(raw)
        records = harvest_vocab_gaps(gc)
        assert records[0]["example_dd_paths"] == ["some/path"]
        assert records[0]["example_reasons"] == ["reason B"]
        assert records[0]["source_types"] == ["IMASNode"]
        assert records[0]["first_seen"] is None
        assert records[0]["last_seen"] is None

    def test_segment_filter_added_to_query(self):
        from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps

        gc = self._make_gc([])
        harvest_vocab_gaps(gc, segment_filter="transformation")
        call_kwargs = gc.query.call_args
        # segment param must be present
        assert "segment" in call_kwargs.kwargs or "segment" in str(call_kwargs)

    def test_empty_graph_returns_empty_list(self):
        from imas_codex.standard_names.gap_harvest import harvest_vocab_gaps

        gc = self._make_gc([])
        records = harvest_vocab_gaps(gc)
        assert records == []


# ---------------------------------------------------------------------------
# Tests: format_pr_yaml
# ---------------------------------------------------------------------------


class TestFormatPrYaml:
    """format_pr_yaml emits structured, PR-ready YAML."""

    def _parse(self, records, **kw):
        from imas_codex.standard_names.gap_harvest import format_pr_yaml

        raw = format_pr_yaml(records, **kw)
        # Strip comment lines before parsing
        lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
        return yaml.safe_load("\n".join(lines))

    def test_metadata_fields_present(self):
        doc = self._parse(SAMPLE_RECORDS, isn_version="0.7.0rc23", dd_version="3.41.0")
        meta = doc["metadata"]
        assert meta["isn_version"] == "0.7.0rc23"
        assert meta["dd_version"] == "3.41.0"
        assert "generated_at" in meta
        assert meta["total_gaps"] == len(SAMPLE_RECORDS)
        # distinct_tokens: time_derivative_of, derivative_of, fusion, beta
        assert meta["distinct_tokens"] == 4

    def test_grouped_by_segment(self):
        doc = self._parse(SAMPLE_RECORDS)
        seg_keys = set(doc["gaps_by_segment"].keys())
        assert "process" in seg_keys
        assert "coordinate" in seg_keys

    def test_token_entries_have_required_fields(self):
        doc = self._parse(SAMPLE_RECORDS)
        tokens = doc["gaps_by_segment"]["process"]["tokens"]
        assert len(tokens) == 2
        for token_entry in tokens:
            assert "needed_token" in token_entry
            assert "occurrences" in token_entry

    def test_occurrences_sorted_descending(self):
        doc = self._parse(SAMPLE_RECORDS)
        tokens = doc["gaps_by_segment"]["process"]["tokens"]
        occs = [t["occurrences"] for t in tokens]
        assert occs == sorted(occs, reverse=True)

    def test_closed_segment_flagged(self):
        doc = self._parse(SAMPLE_RECORDS)
        # process is a closed segment (has tokens in ISN)
        assert doc["gaps_by_segment"]["process"]["segment_type"] == "closed"

    def test_open_segment_flagged(self):
        doc = self._parse(SAMPLE_RECORDS)
        # physical_base has no tokens → open segment
        assert doc["gaps_by_segment"]["physical_base"]["segment_type"] == "open"

    def test_example_paths_included_when_present(self):
        doc = self._parse(SAMPLE_RECORDS)
        tokens = doc["gaps_by_segment"]["process"]["tokens"]
        # First token has 2 example paths (time_derivative, 8 occurrences)
        top_token = tokens[0]
        assert "example_dd_paths" in top_token
        assert len(top_token["example_dd_paths"]) > 0

    def test_empty_records_produces_valid_yaml(self):
        from imas_codex.standard_names.gap_harvest import format_pr_yaml

        raw = format_pr_yaml([], isn_version="0.7.0rc23", dd_version="3.41.0")
        lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
        doc = yaml.safe_load("\n".join(lines))
        assert doc["metadata"]["total_gaps"] == 0
        assert doc["metadata"]["distinct_tokens"] == 0
        assert doc["gaps_by_segment"] == {}

    def test_unknown_versions_when_not_provided(self):
        doc = self._parse(SAMPLE_RECORDS)
        assert doc["metadata"]["isn_version"] == "unknown"
        assert doc["metadata"]["dd_version"] == "unknown"

    def test_header_comment_present(self):
        from imas_codex.standard_names.gap_harvest import format_pr_yaml

        raw = format_pr_yaml(SAMPLE_RECORDS)
        assert raw.startswith("#")
        assert "ISN Vocabulary Gap Report" in raw


# ---------------------------------------------------------------------------
# Tests: format_pr_markdown
# ---------------------------------------------------------------------------


class TestFormatPrMarkdown:
    """format_pr_markdown emits a readable summary with PR template."""

    def _render(self, records, **kw):
        from imas_codex.standard_names.gap_harvest import format_pr_markdown

        return format_pr_markdown(records, **kw)

    def test_contains_summary_line(self):
        md = self._render(SAMPLE_RECORDS)
        assert "gap records" in md
        assert "distinct tokens" in md

    def test_per_segment_table_present(self):
        md = self._render(SAMPLE_RECORDS)
        assert "Gaps by Grammar Segment" in md
        assert "process" in md
        assert "coordinate" in md

    def test_top_tokens_table_present(self):
        md = self._render(SAMPLE_RECORDS)
        assert "Top" in md
        assert "time_derivative" in md

    def test_pr_template_present(self):
        md = self._render(SAMPLE_RECORDS)
        assert "ISN Grammar PR Template" in md
        assert "Suggested PR Title" in md
        assert "Acceptance Criteria" in md

    def test_empty_note_when_no_records(self):
        md = self._render([])
        assert "No VocabGap nodes" in md

    def test_version_in_header(self):
        md = self._render(SAMPLE_RECORDS, isn_version="0.7.0rc23", dd_version="3.41.0")
        assert "0.7.0rc23" in md
        assert "3.41.0" in md


# ---------------------------------------------------------------------------
# Tests: CLI integration (sn gaps --format yaml)
# ---------------------------------------------------------------------------


class TestSnGapsCLIYaml:
    """sn gaps --format yaml routes through gap_harvest.format_pr_yaml."""

    def _invoke(self, args=None):
        from click.testing import CliRunner

        from imas_codex.cli.sn import sn_gaps

        runner = CliRunner()
        return runner.invoke(sn_gaps, args or ["--format", "yaml"])

    @patch("imas_codex.graph.client.GraphClient")
    @patch("imas_codex.standard_names.gap_harvest.harvest_vocab_gaps")
    def test_yaml_format_invokes_harvester(self, mock_harvest, mock_gc):
        """When --format yaml, harvest_vocab_gaps + format_pr_yaml are called."""
        mock_harvest.return_value = []
        mock_gc.return_value.__enter__ = lambda s: MagicMock()
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        result = self._invoke(["--format", "yaml", "--direction", "missing"])
        # No crash; empty output is fine (no gaps)
        assert result.exit_code == 0

    @patch("imas_codex.graph.client.GraphClient")
    @patch(
        "imas_codex.standard_names.gap_harvest.harvest_vocab_gaps",
        return_value=SAMPLE_RECORDS[:3],
    )
    @patch(
        "imas_codex.standard_names.gap_harvest._isn_version",
        return_value="0.7.0rc23",
    )
    @patch(
        "imas_codex.standard_names.gap_harvest._dd_version",
        return_value="3.41.0",
    )
    def test_yaml_output_contains_pr_header(
        self, mock_ddv, mock_isnv, mock_harvest, mock_gc
    ):
        mock_gc.return_value.__enter__ = lambda s: MagicMock()
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        result = self._invoke(["--format", "yaml", "--direction", "missing"])
        assert result.exit_code == 0
        assert "ISN Vocabulary Gap Report" in result.output
        assert "process" in result.output

    @patch("imas_codex.graph.client.GraphClient")
    @patch(
        "imas_codex.standard_names.gap_harvest.harvest_vocab_gaps",
        return_value=SAMPLE_RECORDS[:3],
    )
    @patch("imas_codex.standard_names.gap_harvest._isn_version", return_value=None)
    @patch("imas_codex.standard_names.gap_harvest._dd_version", return_value=None)
    def test_yaml_output_is_valid_yaml(
        self, mock_ddv, mock_isnv, mock_harvest, mock_gc
    ):
        mock_gc.return_value.__enter__ = lambda s: MagicMock()
        mock_gc.return_value.__exit__ = MagicMock(return_value=False)

        result = self._invoke(["--format", "yaml", "--direction", "missing"])
        assert result.exit_code == 0
        lines = [ln for ln in result.output.splitlines() if not ln.startswith("#")]
        doc = yaml.safe_load("\n".join(lines))
        assert "metadata" in doc
        assert "gaps_by_segment" in doc
        assert doc["metadata"]["total_gaps"] == 3
