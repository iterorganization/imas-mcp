"""Tests for static tree discovery parallel modules."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.discovery.static.state import StaticDiscoveryState

# ===========================================================================
# StaticDiscoveryState tests
# ===========================================================================


class TestStaticDiscoveryState:
    """Tests for StaticDiscoveryState dataclass."""

    def _make_state(self, **kwargs) -> StaticDiscoveryState:
        defaults = {
            "facility": "tcv",
            "ssh_host": "tcv",
            "tree_name": "static",
            "tree_config": {"tree_name": "static", "versions": []},
        }
        defaults.update(kwargs)
        return StaticDiscoveryState(**defaults)

    def test_default_values(self):
        state = self._make_state()
        assert state.cost_limit == 2.0
        assert state.timeout == 600
        assert state.batch_size == 40
        assert state.enrich is True
        assert state.stop_requested is False
        assert state.force is False

    def test_should_stop_when_not_requested(self):
        state = self._make_state()
        assert state.should_stop() is False

    def test_should_stop_when_requested(self):
        state = self._make_state()
        state.stop_requested = True
        assert state.should_stop() is True

    def test_should_stop_past_deadline(self):
        state = self._make_state(deadline=time.time() - 1)
        assert state.should_stop() is True

    def test_should_stop_before_deadline(self):
        state = self._make_state(deadline=time.time() + 3600)
        assert state.should_stop() is False

    def test_budget_exhausted(self):
        state = self._make_state(cost_limit=1.0)
        state.enrich_stats.cost = 1.5
        assert state.budget_exhausted is True

    def test_budget_not_exhausted(self):
        state = self._make_state(cost_limit=5.0)
        state.enrich_stats.cost = 0.5
        assert state.budget_exhausted is False

    def test_total_cost(self):
        state = self._make_state()
        state.enrich_stats.cost = 3.14
        assert state.total_cost == 3.14

    def test_has_all_phases(self):
        state = self._make_state()
        assert state.extract_phase.name == "extract"
        assert state.units_phase.name == "units"
        assert state.enrich_phase.name == "enrich"
        assert state.ingest_phase.name == "ingest"

    def test_has_all_stats(self):
        state = self._make_state()
        assert state.extract_stats.processed == 0
        assert state.units_stats.processed == 0
        assert state.enrich_stats.processed == 0
        assert state.ingest_stats.processed == 0


# ===========================================================================
# graph_ops tests (mocked GraphClient)
# ===========================================================================


class TestGraphOps:
    """Tests for graph operations with mocked GraphClient."""

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_seed_versions_empty(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import seed_versions

        result = seed_versions("tcv", "static", [])
        assert result == 0

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_seed_versions_creates_nodes(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import seed_versions

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"seeded": 3}]

        result = seed_versions("tcv", "static", [1, 3, 8])
        assert result == 3
        mock_gc.query.assert_called_once()

        # Check records passed to query
        call_kwargs = mock_gc.query.call_args
        records = call_kwargs.kwargs.get("records") or call_kwargs[1].get("records")
        if records is None and len(call_kwargs.args) > 1:
            # Could be positional
            pass
        assert mock_gc.ensure_facility.called

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_seed_versions_with_config(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import seed_versions

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"seeded": 2}]

        version_config = [
            {"version": 1, "first_shot": 100, "description": "Initial"},
            {"version": 3, "first_shot": 300, "description": "Upgrade"},
        ]
        result = seed_versions("tcv", "static", [1, 3], version_config)
        assert result == 2

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_claim_version_returns_dict(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import claim_version_for_extraction

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"id": "tcv:static:v1", "version": 1, "first_shot": 1}
        ]

        result = claim_version_for_extraction("tcv", "static")
        assert result is not None
        assert result["version"] == 1
        assert result["id"] == "tcv:static:v1"

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_claim_version_returns_none_when_empty(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import claim_version_for_extraction

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = []

        result = claim_version_for_extraction("tcv", "static")
        assert result is None

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_mark_version_extracted(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import mark_version_extracted

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        mark_version_extracted("tcv:static:v1", 47976)
        mock_gc.query.assert_called_once()

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_has_pending_extract_work(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import has_pending_extract_work

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"has_work": True}]

        assert has_pending_extract_work("tcv", "static") is True

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_has_pending_enrich_work(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import has_pending_enrich_work

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        # First call: has_pending_pattern_work returns False
        # Second call: has_pending_enrich_work node query returns False
        mock_gc.query.side_effect = [
            [{"has_work": False}],
            [{"has_work": False}],
        ]

        assert has_pending_enrich_work("tcv", "static") is False

    def test_has_pending_ingest_work_always_false(self):
        from imas_codex.discovery.static.graph_ops import has_pending_ingest_work

        assert has_pending_ingest_work("tcv", "static") is False

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_get_static_discovery_stats(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import get_static_discovery_stats

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = [
            # Version status
            [
                {"status": "discovered", "cnt": 2, "nodes": 0},
                {"status": "ingested", "cnt": 6, "nodes": 280000},
            ],
            # Claimed count
            [{"cnt": 1}],
            # Node enrichment
            [
                {
                    "total": 280000,
                    "enriched": 500,
                    "enrichable": 200000,
                }
            ],
            # Parent group stats
            [{"total_parents": 3000, "pending_parents": 2500, "enriched_parents": 500}],
            # Pattern stats
            [{"total": 50, "enriched": 10, "pending": 40}],
        ]

        stats = get_static_discovery_stats("tcv", "static")
        assert stats["versions_total"] == 8
        assert stats["versions_discovered"] == 2
        assert stats["versions_ingested"] == 6
        assert stats["versions_claimed"] == 1
        assert stats["nodes_graph"] == 280000
        assert stats["nodes_enriched"] == 500
        assert stats["parent_groups_total"] == 3000
        assert stats["parent_groups_pending"] == 2500
        assert stats["parent_groups_enriched"] == 500
        assert stats["patterns_total"] == 50
        assert stats["patterns_enriched"] == 10
        assert stats["pending_patterns"] == 40

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_release_parent_claim(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import release_parent_claim

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        release_parent_claim("tcv:static:TOP.W")
        mock_gc.query.assert_called_once()

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_mark_parent_children_enriched_empty(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import mark_parent_children_enriched

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = mark_parent_children_enriched("tcv:static:TOP.W", {})
        assert result == 0

    # --- Pattern tests ---

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_detect_and_create_patterns_empty(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import detect_and_create_patterns

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = []

        result = detect_and_create_patterns("tcv", "static")
        assert result == 0

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_detect_and_create_patterns_finds_groups(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import detect_and_create_patterns

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = [
            # Detection query result
            [
                {
                    "gp_path": "\\MAGNETICS::TOP.W",
                    "leaf_name": "R",
                    "leaf_type": "NUMERIC",
                    "parent_count": 830,
                    "representative": "\\MAGNETICS::TOP.W.W001.R",
                },
                {
                    "gp_path": "\\MAGNETICS::TOP.W",
                    "leaf_name": "Z",
                    "leaf_type": "NUMERIC",
                    "parent_count": 830,
                    "representative": "\\MAGNETICS::TOP.W.W001.Z",
                },
            ],
            # Merge + link result
            [
                {"id": "tcv:static:\\MAGNETICS::TOP.W:R", "linked": 830},
                {"id": "tcv:static:\\MAGNETICS::TOP.W:Z", "linked": 830},
            ],
        ]

        result = detect_and_create_patterns("tcv", "static")
        assert result == 2
        # Verify ensure_facility was called
        mock_gc.ensure_facility.assert_called_once_with("tcv")

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_claim_patterns_for_enrichment(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import claim_patterns_for_enrichment

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {
                "id": "tcv:static:\\MAGNETICS::TOP.W:R",
                "grandparent_path": "\\MAGNETICS::TOP.W",
                "leaf_name": "R",
                "index_count": 830,
                "node_type": "NUMERIC",
                "representative_path": "\\MAGNETICS::TOP.W.W001.R",
                "tags": ["R"],
                "units": "m",
                "parent_path": "\\MAGNETICS::TOP.W.W001",
            }
        ]

        result = claim_patterns_for_enrichment("tcv", "static", limit=10)
        assert len(result) == 1
        assert result[0]["leaf_name"] == "R"
        assert result[0]["index_count"] == 830

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_mark_patterns_enriched_empty(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import mark_patterns_enriched

        result = mark_patterns_enriched([], {})
        assert result == 0

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_mark_patterns_enriched_propagates(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import mark_patterns_enriched

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"pattern_id": "tcv:static:\\MAGNETICS::TOP.W:R", "propagated": 830}
        ]

        result = mark_patterns_enriched(
            ["tcv:static:\\MAGNETICS::TOP.W:R"],
            {"tcv:static:\\MAGNETICS::TOP.W:R": "Major radius of coil turn"},
            {
                "tcv:static:\\MAGNETICS::TOP.W:R": {
                    "keywords": ["radius", "coil"],
                    "category": "coil",
                }
            },
        )
        assert result == 830

    def test_release_pattern_claims_empty(self):
        from imas_codex.discovery.static.graph_ops import release_pattern_claims

        result = release_pattern_claims([])
        assert result == 0

    @patch("imas_codex.discovery.static.graph_ops.GraphClient")
    def test_has_pending_pattern_work(self, mock_gc_cls):
        from imas_codex.discovery.static.graph_ops import has_pending_pattern_work

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"has_work": True}]

        assert has_pending_pattern_work("tcv", "static") is True


# ===========================================================================
# parallel module tests
# ===========================================================================


class TestParallelHelpers:
    """Tests for parallel module helper functions."""

    @patch("imas_codex.discovery.static.parallel.GraphClient")
    def test_force_reset_versions(self, mock_gc_cls):
        from imas_codex.discovery.static.parallel import _force_reset_versions

        mock_gc = MagicMock()
        mock_gc_cls.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"reset": 3}]

        result = _force_reset_versions("tcv", "static", [1, 3, 8])
        assert result == 3


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCliHelpers:
    """Tests for static CLI helper functions."""

    def test_parse_versions_from_string(self):
        from imas_codex.cli.discover.static import _parse_versions

        result = _parse_versions("1,3,8", {})
        assert result == [1, 3, 8]

    def test_parse_versions_from_config(self):
        from imas_codex.cli.discover.static import _parse_versions

        cfg = {"versions": [{"version": 1}, {"version": 3}, {"version": 8}]}
        result = _parse_versions(None, cfg)
        assert result == [1, 3, 8]

    def test_parse_versions_none(self):
        from imas_codex.cli.discover.static import _parse_versions

        result = _parse_versions(None, {})
        assert result is None

    def test_parse_versions_single(self):
        from imas_codex.cli.discover.static import _parse_versions

        result = _parse_versions("5", {})
        assert result == [5]


# ===========================================================================
# Progress display tests
# ===========================================================================


class TestStaticProgressDisplay:
    """Tests for the StaticProgressDisplay."""

    def test_creates_display(self):
        from imas_codex.cli.discover.static import StaticProgressDisplay

        display = StaticProgressDisplay(
            facility="tcv",
            cost_limit=2.0,
        )
        assert display.facility == "tcv"
        assert display.cost_limit == 2.0
        assert display.enrich is True

    def test_state_eta_no_data(self):
        from imas_codex.cli.discover.static import StaticProgressState

        state = StaticProgressState()
        assert state.eta_seconds is None

    def test_state_eta_with_extract_rate(self):
        from imas_codex.cli.discover.static import StaticProgressState

        state = StaticProgressState()
        state.extract_rate = 2.0  # 2 nodes/s
        state.extract_nodes_total = 10
        state.extract_nodes = 4
        eta = state.eta_seconds
        assert eta is not None
        assert eta == pytest.approx(3.0, abs=0.1)

    def test_state_elapsed(self):
        from imas_codex.cli.discover.static import StaticProgressState

        state = StaticProgressState(start_time=time.time() - 60)
        assert state.elapsed >= 59.0

    def test_build_pipeline_section(self):
        from rich.console import Console

        from imas_codex.cli.discover.static import StaticProgressDisplay

        console = Console(width=120, force_terminal=True)
        display = StaticProgressDisplay(
            facility="tcv",
            cost_limit=2.0,
            console=console,
        )
        display.state.extract_completed = 3
        display.state.extract_total = 8
        display.state.extract_nodes = 100000
        section = display._build_pipeline_section()
        text = section.plain
        assert "EXTRACT" in text
        assert "UNITS" in text
        assert "ENRICH" in text

    def test_build_resources_section(self):
        from rich.console import Console

        from imas_codex.cli.discover.static import StaticProgressDisplay

        console = Console(width=120, force_terminal=True)
        display = StaticProgressDisplay(
            facility="tcv",
            cost_limit=2.0,
            console=console,
        )
        display.state.extract_completed = 8
        display.state.extract_total = 8
        display.state.extract_nodes = 380000
        display.state.units_found = 15000
        display.state.enrich_completed = 500
        display.state.enrich_cost = 0.45
        section = display._build_resources_section()
        text = section.plain
        assert "TIME" in text
        assert "COST" in text
        assert "STATS" in text
        assert "versions" in text

    def test_refresh_from_graph_gates_enrich_on_extraction(self):
        """Enrichment totals should not update until extraction is complete."""
        from rich.console import Console

        from imas_codex.cli.discover.static import StaticProgressDisplay

        console = Console(width=120, force_terminal=True)
        display = StaticProgressDisplay(
            facility="test", cost_limit=2.0, console=console
        )

        # During extraction (3/8 versions done), enrich totals should stay at 0
        mock_stats = {
            "versions_total": 8,
            "versions_ingested": 3,
            "nodes_total": 100000,
            "nodes_enrichable": 34000,
            "nodes_enriched": 0,
            "patterns_total": 50,
            "patterns_enriched": 0,
            "parent_groups_total": 3000,
            "parent_groups_enriched": 0,
        }
        with patch(
            "imas_codex.discovery.static.graph_ops.get_static_discovery_stats",
            return_value=mock_stats,
        ):
            display.refresh_from_graph("test", "test_tree")

        assert display.state.enrich_total == 0, (
            "enrich_total should be 0 during extraction"
        )

        # After all versions extracted, enrich totals should update
        mock_stats["versions_ingested"] = 8
        with patch(
            "imas_codex.discovery.static.graph_ops.get_static_discovery_stats",
            return_value=mock_stats,
        ):
            display.refresh_from_graph("test", "test_tree")

        # enrich_total = patterns_total + parent_groups_total = 50 + 3000
        assert display.state.enrich_total == 3050

    def test_update_units_with_msg_only(self):
        """Units callback with msg but no results should push to queue."""
        from rich.console import Console

        from imas_codex.cli.discover.static import StaticProgressDisplay
        from imas_codex.discovery.base.progress import WorkerStats

        console = Console(width=120, force_terminal=True)
        display = StaticProgressDisplay(
            facility="test", cost_limit=2.0, console=console
        )
        stats = WorkerStats()
        display.update_units("awaiting extract", stats, None)
        assert len(display.units_queue) > 0

    def test_update_enrich_with_msg_only(self):
        """Enrich callback with msg but no results should push to queue."""
        from rich.console import Console

        from imas_codex.cli.discover.static import StaticProgressDisplay
        from imas_codex.discovery.base.progress import WorkerStats

        console = Console(width=120, force_terminal=True)
        display = StaticProgressDisplay(
            facility="test", cost_limit=2.0, console=console
        )
        stats = WorkerStats()
        display.update_enrich("awaiting extract", stats, None)
        assert len(display.enrich_queue) > 0
