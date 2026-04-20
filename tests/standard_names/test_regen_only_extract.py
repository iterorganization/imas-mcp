"""Regression tests for --regen-only regen-only extraction.

Covers the architectural bug where --regen-only only gated
prompt injection without narrowing the extraction scope. The fix adds a
"regen-only" mode that re-queues exactly the sources whose linked
StandardName is in ``validation_status='needs_revision'``.

Tests cover:

1. ``test_regen_only_extract_selects_only_needs_revision_sources`` —
   a graph with 5 needs_revision SNs + 20 valid SNs yields exactly 5
   extract items, each carrying review_feedback.
2. ``test_domain_filter_narrows_regen_scope`` — ``--domain`` passed
   through ``state.domain_filter`` reaches the Cypher query.
3. ``test_broad_extract_when_not_regen_mode`` — without
   ``--regen-only``, the old DD extraction path is used.
4. ``test_is_regen_only_mode_gating`` — ``paths_list`` / ``from_model``
   short-circuit regen-only even with ``--regen-only``.
5. ``test_fetch_needs_revision_sources_cypher_shape`` — helper builds
   the expected Cypher and params.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.models import StandardNameValidationStatus
from imas_codex.standard_names.graph_ops import fetch_needs_revision_sources
from imas_codex.standard_names.state import StandardNameBuildState
from imas_codex.standard_names.workers import extract_worker

# ---------------------------------------------------------------------------
# State.is_regen_only_mode
# ---------------------------------------------------------------------------


class TestIsRegenOnlyMode:
    def test_regen_on_with_plain_flag(self) -> None:
        st = StandardNameBuildState(facility="dd", regen_only=True)
        assert st.is_regen_only_mode() is True

    def test_off_without_flag(self) -> None:
        st = StandardNameBuildState(facility="dd", regen_only=False)
        assert st.is_regen_only_mode() is False

    def test_paths_list_short_circuits(self) -> None:
        st = StandardNameBuildState(
            facility="dd",
            regen_only=True,
            paths_list=["equilibrium/time_slice/profiles_1d/psi"],
        )
        assert st.is_regen_only_mode() is False

    def test_from_model_short_circuits(self) -> None:
        st = StandardNameBuildState(
            facility="dd",
            regen_only=True,
            from_model="anthropic/claude-sonnet-4.6",
        )
        assert st.is_regen_only_mode() is False

    def test_domain_filter_is_narrowing_not_overriding(self) -> None:
        """--domain narrows regen scope, it does NOT disable regen-only mode."""
        st = StandardNameBuildState(
            facility="dd",
            regen_only=True,
            domain_filter="equilibrium",
        )
        assert st.is_regen_only_mode() is True


# ---------------------------------------------------------------------------
# graph_ops.fetch_needs_revision_sources — Cypher shape
# ---------------------------------------------------------------------------


class TestFetchNeedsRevisionSources:
    def _mock_graph(self, rows: list[dict]) -> MagicMock:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=rows)
        return mock_gc

    def test_cypher_uses_enum_value_no_hardcoded_string(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources()

        cypher, kwargs = mock_gc.query.call_args[0][0], mock_gc.query.call_args[1]
        # Cypher must not hardcode the status string literal
        assert "'needs_revision'" not in cypher
        assert "validation_status = $status" in cypher
        # Enum value is used as the bound param
        assert kwargs["status"] == StandardNameValidationStatus.needs_revision.value

    def test_dd_default_walks_imasnode(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources()
        cypher = mock_gc.query.call_args[0][0]
        assert "IMASNode" in cypher
        assert "HAS_STANDARD_NAME" in cypher

    def test_signals_walks_facilitysignal(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources(source_type="signals")
        cypher = mock_gc.query.call_args[0][0]
        assert "FacilitySignal" in cypher
        assert "IMASNode" not in cypher

    def test_invalid_source_type_raises(self) -> None:
        with pytest.raises(ValueError, match="source_type"):
            fetch_needs_revision_sources(source_type="bogus")

    def test_domain_filter_reaches_cypher(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources(domain="equilibrium")
        cypher, kwargs = mock_gc.query.call_args[0][0], mock_gc.query.call_args[1]
        assert "sn.physics_domain = $domain" in cypher
        assert kwargs["domain"] == "equilibrium"

    def test_ids_filter_joins_ids_node(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources(ids="equilibrium")
        cypher, kwargs = mock_gc.query.call_args[0][0], mock_gc.query.call_args[1]
        assert "IN_IDS" in cypher
        assert "ids_node.id = $ids" in cypher
        assert kwargs["ids"] == "equilibrium"

    def test_limit_applied(self) -> None:
        mock_gc = self._mock_graph([])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_needs_revision_sources(limit=7)
        cypher, kwargs = mock_gc.query.call_args[0][0], mock_gc.query.call_args[1]
        assert "LIMIT $limit" in cypher
        assert kwargs["limit"] == 7

    def test_returns_review_feedback_shape(self) -> None:
        mock_gc = self._mock_graph(
            [
                {
                    "source_id": "equilibrium/time_slice/profiles_1d/psi",
                    "previous_name": "mediocre_name",
                    "previous_description": "d",
                    "previous_documentation": None,
                    "reviewer_score": 0.35,
                    "review_tier": "poor",
                    "reviewer_comments": "drop prefix",
                    "reviewer_scores_json": None,
                    "validation_status": "needs_revision",
                },
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_needs_revision_sources(domain="equilibrium")
        assert len(out) == 1
        entry = out[0]
        assert entry["source_id"] == "equilibrium/time_slice/profiles_1d/psi"
        assert entry["source_type"] == "dd"
        fb = entry["review_feedback"]
        assert fb["previous_name"] == "mediocre_name"
        assert fb["review_tier"] == "poor"
        assert fb["reviewer_comments"] == "drop prefix"
        assert fb["validation_status"] == "needs_revision"

    def test_dedup_keeps_lowest_score_feedback(self) -> None:
        mock_gc = self._mock_graph(
            [
                {
                    "source_id": "a/b/c",
                    "previous_name": "okay",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.55,
                    "review_tier": "adequate",
                    "reviewer_comments": "",
                    "reviewer_scores_json": None,
                    "validation_status": "needs_revision",
                },
                {
                    "source_id": "a/b/c",
                    "previous_name": "awful",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.10,
                    "review_tier": "poor",
                    "reviewer_comments": "rewrite",
                    "reviewer_scores_json": None,
                    "validation_status": "needs_revision",
                },
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_needs_revision_sources()
        assert len(out) == 1
        assert out[0]["review_feedback"]["previous_name"] == "awful"
        assert out[0]["review_feedback"]["reviewer_score"] == 0.10


# ---------------------------------------------------------------------------
# extract_worker — regen-only routing
# ---------------------------------------------------------------------------


def _make_batch(path: str) -> SimpleNamespace:
    """Build a minimal ExtractionBatch-like object the worker can consume."""
    return SimpleNamespace(
        source="dd",
        group_key="equilibrium",
        items=[{"path": path, "description": "desc", "ids_name": "equilibrium"}],
        context={},
        existing_names=set(),
    )


class TestExtractWorkerRegenOnly:
    def test_regen_only_selects_only_needs_revision_sources(self) -> None:
        """Given 5 needs_revision SNs (the worker must ignore the 20 valid SNs),
        extract must return exactly 5 items, each with review_feedback."""
        # 5 needs_revision sources returned by the helper
        nr_paths = [f"equilibrium/time_slice/profiles_1d/x{i}" for i in range(5)]
        fake_sources = [
            {
                "source_id": p,
                "source_type": "dd",
                "review_feedback": {
                    "previous_name": f"prev_{i}",
                    "reviewer_comments": "bad prefix",
                    "review_tier": "poor",
                    "reviewer_score": 0.2,
                    "reviewer_scores": None,
                    "validation_status": "needs_revision",
                    "previous_description": None,
                    "previous_documentation": None,
                },
            }
            for i, p in enumerate(nr_paths)
        ]
        fake_feedback_map = {s["source_id"]: s["review_feedback"] for s in fake_sources}
        fake_batches = [_make_batch(p) for p in nr_paths]

        state = StandardNameBuildState(
            facility="dd",
            source="dd",
            regen_only=True,
            force=True,
            domain_filter="equilibrium",
            dry_run=True,
        )

        with (
            patch(
                "imas_codex.standard_names.graph_ops.get_existing_standard_names",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.get_named_source_ids",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.fetch_needs_revision_sources",
                return_value=fake_sources,
            ) as mock_fetch_needs,
            patch(
                "imas_codex.standard_names.sources.dd.extract_specific_paths",
                return_value=fake_batches,
            ) as mock_extract_specific,
            patch(
                "imas_codex.standard_names.sources.dd.extract_dd_candidates",
            ) as mock_extract_broad,
            patch(
                "imas_codex.standard_names.graph_ops.fetch_review_feedback_for_sources",
                return_value=fake_feedback_map,
            ),
            patch(
                "imas_codex.standard_names.graph_ops.get_source_name_mapping",
                return_value={},
            ),
        ):
            asyncio.run(extract_worker(state))

        # Broad DD extract must NOT be called — regen-only replaces it
        mock_extract_broad.assert_not_called()
        mock_fetch_needs.assert_called_once()
        mock_extract_specific.assert_called_once()
        # Verify the paths passed to extract_specific_paths are exactly the 5 NR ones
        call_paths = mock_extract_specific.call_args.kwargs["paths"]
        assert call_paths == nr_paths

        # State now holds exactly 5 items, each with review_feedback attached
        all_items = [it for b in state.extracted for it in b.items]
        assert len(all_items) == 5
        for item in all_items:
            assert "review_feedback" in item
            assert item["review_feedback"]["review_tier"] == "poor"

    def test_domain_filter_narrows_regen_scope(self) -> None:
        """--domain propagates to the needs_revision source query."""
        state = StandardNameBuildState(
            facility="dd",
            source="dd",
            regen_only=True,
            force=True,
            domain_filter="equilibrium",
            ids_filter="equilibrium",
            limit=42,
            dry_run=True,
        )

        with (
            patch(
                "imas_codex.standard_names.graph_ops.get_existing_standard_names",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.fetch_needs_revision_sources",
                return_value=[],
            ) as mock_fetch_needs,
            patch(
                "imas_codex.standard_names.sources.dd.extract_specific_paths",
                return_value=[],
            ),
            patch(
                "imas_codex.standard_names.sources.dd.extract_dd_candidates",
            ) as mock_extract_broad,
        ):
            asyncio.run(extract_worker(state))

        mock_extract_broad.assert_not_called()
        mock_fetch_needs.assert_called_once()
        kwargs = mock_fetch_needs.call_args.kwargs
        assert kwargs["domain"] == "equilibrium"
        assert kwargs["ids"] == "equilibrium"
        assert kwargs["limit"] == 42
        assert kwargs["source_type"] == "dd"

    def test_broad_extract_when_not_regen_mode(self) -> None:
        """Without --regen-only the broad DD path is still used."""
        state = StandardNameBuildState(
            facility="dd",
            source="dd",
            regen_only=False,
            domain_filter="equilibrium",
            dry_run=True,
        )

        with (
            patch(
                "imas_codex.standard_names.graph_ops.get_existing_standard_names",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.get_named_source_ids",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.fetch_needs_revision_sources",
            ) as mock_fetch_needs,
            patch(
                "imas_codex.standard_names.sources.dd.extract_dd_candidates",
                return_value=[],
            ) as mock_extract_broad,
        ):
            asyncio.run(extract_worker(state))

        # Regen-only must NOT be engaged
        mock_fetch_needs.assert_not_called()
        mock_extract_broad.assert_called_once()

    def test_regen_only_skipped_when_paths_list_present(self) -> None:
        """--paths + --regen-only still uses targeted path extraction,
        not the needs_revision query (paths_list is a stricter scope)."""
        state = StandardNameBuildState(
            facility="dd",
            source="dd",
            regen_only=True,
            force=True,
            paths_list=["equilibrium/time_slice/profiles_1d/psi"],
            dry_run=True,
        )

        with (
            patch(
                "imas_codex.standard_names.graph_ops.get_existing_standard_names",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.get_named_source_ids",
                return_value=set(),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.fetch_needs_revision_sources",
            ) as mock_fetch_needs,
            patch(
                "imas_codex.standard_names.sources.dd.extract_specific_paths",
                return_value=[],
            ) as mock_extract_specific,
            patch(
                "imas_codex.standard_names.graph_ops.fetch_review_feedback_for_sources",
                return_value={},
            ),
            patch(
                "imas_codex.standard_names.graph_ops.get_source_name_mapping",
                return_value={},
            ),
        ):
            asyncio.run(extract_worker(state))

        mock_fetch_needs.assert_not_called()
        mock_extract_specific.assert_called_once()
        # The paths argument came from state.paths_list, not from a NR query
        call_paths = mock_extract_specific.call_args.kwargs["paths"]
        assert call_paths == ["equilibrium/time_slice/profiles_1d/psi"]
