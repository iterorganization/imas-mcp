"""Tests for the unified build_worker_status_section() in base/progress.py.

Verifies that the shared worker section renderer correctly:
- Groups workers by their ``group`` field
- Shows budget-stopped annotations for sensitive groups
- Dims all groups when paused
- Falls back to name-based grouping for legacy workers
- Appends extra indicators (e.g. embedding source)
"""

from imas_codex.discovery.base.progress import build_worker_status_section
from imas_codex.discovery.base.supervision import (
    SupervisedWorkerGroup,
    WorkerState,
)


def _make_group(workers: list[tuple[str, str, WorkerState]]) -> SupervisedWorkerGroup:
    """Create a SupervisedWorkerGroup from (name, group, state) tuples."""
    wg = SupervisedWorkerGroup()
    for name, group, wstate in workers:
        status = wg.create_status(name, group=group)
        status.state = wstate
    return wg


class TestBuildWorkerStatusSection:
    """Tests for build_worker_status_section()."""

    def test_no_worker_group(self):
        """No worker group shows 'starting...'."""
        text = build_worker_status_section(None)
        plain = text.plain
        assert "WORKERS" in plain
        assert "starting..." in plain

    def test_groups_by_group_field(self):
        """Workers are grouped by their group field, not name."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.running),
                ("score_worker_1", "score", WorkerState.running),
                ("ingest_worker_0", "ingest", WorkerState.running),
                ("artifact_score_worker", "score", WorkerState.running),
                ("artifact_worker", "ingest", WorkerState.running),
                ("image_score_worker", "score", WorkerState.running),
            ]
        )
        text = build_worker_status_section(wg)
        plain = text.plain
        assert "score:4" in plain
        assert "ingest:2" in plain

    def test_budget_stopped_annotation(self):
        """Budget-sensitive groups show '(budget)' when all stopped."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.stopped),
                ("score_worker_1", "score", WorkerState.stopped),
                ("ingest_worker_0", "ingest", WorkerState.running),
            ]
        )
        text = build_worker_status_section(
            wg,
            budget_exhausted=True,
            budget_sensitive_groups={"score"},
        )
        plain = text.plain
        assert "(budget)" in plain

    def test_no_budget_annotation_when_not_exhausted(self):
        """No budget annotation when budget is not exhausted."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.stopped),
            ]
        )
        text = build_worker_status_section(
            wg,
            budget_exhausted=False,
            budget_sensitive_groups={"score"},
        )
        plain = text.plain
        assert "(budget)" not in plain

    def test_backoff_annotation(self):
        """Workers in backoff show count."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.running),
                ("score_worker_1", "score", WorkerState.backoff),
            ]
        )
        text = build_worker_status_section(wg)
        plain = text.plain
        assert "1 backoff" in plain
        assert "1 active" in plain

    def test_crashed_annotation(self):
        """Crashed workers show count."""
        wg = _make_group(
            [
                ("ingest_worker_0", "ingest", WorkerState.crashed),
            ]
        )
        text = build_worker_status_section(wg)
        plain = text.plain
        assert "1 failed" in plain

    def test_paused_dims_everything(self):
        """When paused, all worker groups are dimmed."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.running),
                ("ingest_worker_0", "ingest", WorkerState.running),
            ]
        )
        text = build_worker_status_section(wg, is_paused=True)
        # All spans should use "dim" style
        for span in text._spans:
            assert span.style in ("dim", "bold green", "dim italic") or "dim" in str(
                span.style
            ), f"Expected dim style, got {span.style}"

    def test_extra_indicators(self):
        """Extra indicators are appended after worker groups."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.running),
            ]
        )
        text = build_worker_status_section(
            wg,
            extra_indicators=[("embed:remote", "green"), ("auth:vpn", "cyan")],
        )
        plain = text.plain
        assert "embed:remote" in plain
        assert "auth:vpn" in plain

    def test_fallback_group_from_name(self):
        """Workers without group fall back to name-based grouping."""
        wg = SupervisedWorkerGroup()
        # Create workers without group (legacy path)
        s1 = wg.create_status("scan_worker_0")
        s1.state = WorkerState.running
        s2 = wg.create_status("enrich_worker_0")
        s2.state = WorkerState.running

        text = build_worker_status_section(wg)
        plain = text.plain
        assert "scan:1" in plain
        assert "enrich:1" in plain

    def test_multiple_groups_ordered(self):
        """Groups appear in dict insertion order (score first, then ingest)."""
        wg = _make_group(
            [
                ("score_worker_0", "score", WorkerState.running),
                ("ingest_worker_0", "ingest", WorkerState.running),
            ]
        )
        text = build_worker_status_section(wg)
        plain = text.plain
        score_pos = plain.index("score:")
        ingest_pos = plain.index("ingest:")
        assert score_pos < ingest_pos
