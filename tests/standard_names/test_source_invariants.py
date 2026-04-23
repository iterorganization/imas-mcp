"""Tests for the FROM_DD_PATH source invariant (rc22 B5).

Invariant: every non-stale ``StandardNameSource`` with
``source_type='dd'`` MUST have a ``FROM_DD_PATH`` edge to an
``IMASNode``.  Prior bootstrap produced 499 orphan sources (RD §5).
We prevent recurrence with two complementary mechanisms:

1. **Birth invariant** — ``merge_standard_name_sources()`` rejects any
   ``'dd'`` source whose ``dd_path`` is ``None`` or missing (never
   orphan-from-birth).

2. **Reconcile** — ``reconcile_standard_name_sources()`` marks as stale
   any ``'dd'`` source that lost its ``FROM_DD_PATH`` edge
   (orphan-after-birth).  Reconcile runs automatically as the **first
   phase** of every ``sn run`` turn.

All tests are mock-based — no live Neo4j required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_merge(sources, mock_gc, *, force=False):
    """Call merge_standard_name_sources with a mocked GraphClient."""
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import merge_standard_name_sources

        return merge_standard_name_sources(sources, force=force)


def _call_reconcile(
    mock_gc, *, source_type="dd", stale_ids=None, revived=0, relinked=0
):
    """Call reconcile_standard_name_sources with controlled mock returns."""
    stale_rows = [{"id": sid} for sid in (stale_ids or [])]
    mock_gc.query.side_effect = [
        stale_rows,  # first query: find orphan candidates
        [{"count": revived}],  # second query: revive stale
        [{"count": relinked}],  # third query: re-link
    ]
    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
        patch(
            "imas_codex.standard_names.graph_ops.mark_sources_stale"
        ) as mock_mark_stale,
    ):
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        mock_mark_stale.return_value = len(stale_ids or [])

        from imas_codex.standard_names.graph_ops import reconcile_standard_name_sources

        result = reconcile_standard_name_sources(source_type)
    return result, mock_mark_stale


def _dd_source(i: int, *, dd_path: str | None = "auto") -> dict:
    """Return a minimal dd source dict.  Pass dd_path=None to omit the field."""
    path = f"core_profiles/test_field_{i}" if dd_path == "auto" else dd_path
    src: dict = {
        "id": f"dd:core_profiles/test_field_{i}",
        "source_type": "dd",
        "source_id": f"core_profiles/test_field_{i}",
        "batch_key": "test-batch",
        "status": "extracted",
        "description": f"Test field {i}",
    }
    if path is not None:
        src["dd_path"] = path
    return src


# ---------------------------------------------------------------------------
# Birth-invariant: merge must never create orphan dd sources
# ---------------------------------------------------------------------------


class TestBirthInvariant:
    """merge_standard_name_sources must enforce the birth invariant."""

    def test_dd_source_with_dd_path_written(self):
        """A dd source WITH a valid dd_path is forwarded to the graph."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 1}]

        result = _call_merge([_dd_source(0)], mock_gc)

        # Graph was called (source was not filtered)
        mock_gc.query.assert_called_once()
        assert result == 1

    def test_cypher_includes_from_dd_path(self):
        """The MERGE Cypher must create FROM_DD_PATH for dd sources."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 5}]

        _call_merge([_dd_source(i) for i in range(5)], mock_gc)

        cypher = mock_gc.query.call_args[0][0]
        assert "FROM_DD_PATH" in cypher, "Cypher must create FROM_DD_PATH edge"
        assert "IMASNode" in cypher, "Cypher must reference IMASNode"

    def test_dd_source_with_dd_path_none_rejected(self):
        """A dd source with dd_path=None must be silently dropped (returns 0)."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 0}]

        result = _call_merge([_dd_source(0, dd_path=None)], mock_gc)

        # No graph write — birth invariant enforced
        mock_gc.query.assert_not_called()
        assert result == 0

    def test_dd_source_with_no_dd_path_key_rejected(self):
        """A dd source with the dd_path key absent entirely must be dropped."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 0}]

        src = _dd_source(0)
        src.pop("dd_path", None)  # ensure key is absent

        result = _call_merge([src], mock_gc)

        mock_gc.query.assert_not_called()
        assert result == 0

    def test_mixed_batch_only_valid_written(self):
        """In a mixed batch, only sources with dd_path reach the graph."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 3}]

        sources = [
            _dd_source(0),  # valid
            _dd_source(1, dd_path=None),  # orphan — must be dropped
            _dd_source(2),  # valid
            _dd_source(3),  # valid
        ]
        result = _call_merge(sources, mock_gc)

        # Graph is still called (3 valid sources remain)
        mock_gc.query.assert_called_once()
        # Only valid sources sent
        sent = mock_gc.query.call_args.kwargs.get(
            "sources", mock_gc.query.call_args[1].get("sources", [])
        )
        assert all(s.get("dd_path") for s in sent), (
            "All sources forwarded to the graph must have a dd_path"
        )
        assert result == 3

    def test_all_orphan_batch_returns_zero_without_graph_call(self):
        """If every source in the batch would be an orphan, return 0 immediately."""
        mock_gc = MagicMock()

        sources = [_dd_source(i, dd_path=None) for i in range(5)]
        result = _call_merge(sources, mock_gc)

        mock_gc.query.assert_not_called()
        assert result == 0

    def test_signals_source_without_dd_path_not_filtered(self):
        """Signal sources are not subject to the dd_path birth-invariant check."""
        mock_gc = MagicMock()
        mock_gc.query.return_value = [{"affected": 1}]

        src = {
            "id": "signals:tcv:ip",
            "source_type": "signals",
            "source_id": "tcv:ip",
            "batch_key": "test-batch",
            "status": "extracted",
            # no dd_path — fine for signals
        }
        result = _call_merge([src], mock_gc)

        mock_gc.query.assert_called_once()
        assert result == 1


# ---------------------------------------------------------------------------
# Reconcile marks stale when FROM_DD_PATH is missing (orphan-after-birth)
# ---------------------------------------------------------------------------


class TestReconcileMarksStale:
    """reconcile_standard_name_sources marks orphaned dd sources stale."""

    def test_reconcile_marks_source_stale_when_dd_node_gone(self):
        """When reconcile finds dd sources without FROM_DD_PATH, it marks them stale."""
        mock_gc = MagicMock()
        orphan_ids = ["dd:path/a", "dd:path/b"]

        result, mock_mark_stale = _call_reconcile(mock_gc, stale_ids=orphan_ids)

        mock_mark_stale.assert_called_once_with(orphan_ids)
        assert result["stale_marked"] == len(orphan_ids)

    def test_reconcile_no_orphans_no_stale_call(self):
        """When no orphans exist, mark_sources_stale should be called with empty list."""
        mock_gc = MagicMock()

        result, mock_mark_stale = _call_reconcile(mock_gc, stale_ids=[])

        # mark_sources_stale([]) returns 0 without a graph call (early-exit)
        mock_mark_stale.assert_called_once_with([])
        assert result["stale_marked"] == 0

    def test_reconcile_returns_revived_count(self):
        """reconcile returns the count of stale sources that were revived."""
        mock_gc = MagicMock()

        result, _ = _call_reconcile(mock_gc, stale_ids=[], revived=3)

        assert result["revived"] == 3

    def test_reconcile_returns_relinked_count(self):
        """reconcile returns the count of non-stale sources that were re-linked."""
        mock_gc = MagicMock()

        result, _ = _call_reconcile(mock_gc, stale_ids=[], relinked=7)

        assert result["relinked"] == 7

    def test_reconcile_stale_query_excludes_already_stale(self):
        """The Cypher used to find orphan candidates must exclude status='stale'."""
        mock_gc = MagicMock()
        mock_gc.query.side_effect = [
            [],
            [{"count": 0}],
            [{"count": 0}],
        ]
        with (
            patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC,
            patch("imas_codex.standard_names.graph_ops.mark_sources_stale"),
        ):
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            from imas_codex.standard_names.graph_ops import (
                reconcile_standard_name_sources,
            )

            reconcile_standard_name_sources("dd")

        # First query is the "find orphans" query — must exclude 'stale'
        first_cypher = mock_gc.query.call_args_list[0][0][0]
        assert "stale" in first_cypher, (
            "Find-orphans Cypher must filter out already-stale sources"
        )


# ---------------------------------------------------------------------------
# Reconcile runs at sn-run start (first phase)
# ---------------------------------------------------------------------------


class TestReconcileRunsAtTurnStart:
    """reconcile must be the first phase in every sn-run turn."""

    def test_reconcile_is_first_phase(self):
        """The turn phases list must begin with 'reconcile'."""
        # We don't need to execute the turn — just inspect the phase
        # ordering by checking that TURN_PHASES starts with 'reconcile'.
        from imas_codex.standard_names.turn import TURN_PHASES, run_turn

        assert TURN_PHASES[0] == "reconcile", (
            f"TURN_PHASES must start with 'reconcile', got {TURN_PHASES[0]!r}"
        )

    def test_reconcile_before_generate_in_turn_phases(self):
        """'reconcile' must appear before 'generate' is active in the phases tuple."""
        from imas_codex.standard_names.turn import TURN_PHASES

        # generate is represented as 'extract'/'compose'/'validate'/'persist' in CLI
        # but the internal phase name is 'generate'.  TURN_PHASES lists the CLI names,
        # so just confirm 'reconcile' is first.
        assert TURN_PHASES.index("reconcile") < TURN_PHASES.index("extract"), (
            "'reconcile' must precede 'extract' (generate) in TURN_PHASES"
        )

    def test_reconcile_phase_key_in_only_mapping(self):
        """--only reconcile must map to the reconcile internal phase."""
        from imas_codex.standard_names.turn import _ONLY_TO_ACTIVE

        assert "reconcile" in _ONLY_TO_ACTIVE, (
            "'reconcile' must be a valid --only value"
        )
        assert "reconcile" in _ONLY_TO_ACTIVE["reconcile"], (
            "--only reconcile must activate the reconcile internal phase"
        )


# ---------------------------------------------------------------------------
# Producer contract: extract worker must emit dd_path for DD sources
# ---------------------------------------------------------------------------


class TestExtractWorkerEmitsDdPath:
    """Regression: workers.sn_extract_worker must populate the dd_path field
    so the B5 birth invariant in merge_standard_name_sources accepts the
    source.  Caught live in C2 canary: 100% rejection (497/497 sources) when
    workers.py emitted only source_id."""

    def test_extract_worker_dd_source_includes_dd_path(self):
        """For source_type='dd', the source dict passed to merge must
        include dd_path == path."""
        # Read workers.py source to verify the key is in the constructed dict.
        # This is a structural assertion; full async integration is covered
        # by test_dd_source_with_dd_path_written above.
        from pathlib import Path

        workers_py = (
            Path(__file__).parent.parent.parent
            / "imas_codex"
            / "standard_names"
            / "workers.py"
        )
        text = workers_py.read_text()
        # The source-dict construction in sn_extract_worker must include
        # a 'dd_path' key for source_type=='dd'.
        assert '"dd_path": path if source_type == "dd"' in text, (
            "workers.sn_extract_worker must populate dd_path for dd sources "
            "to satisfy the B5 birth invariant"
        )
