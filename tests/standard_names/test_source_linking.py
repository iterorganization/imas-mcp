"""Guardrail tests for StandardNameSource → StandardName linking.

These tests enforce the invariant introduced by the linking fix
(commit: ``fix(sn): gate source linking on StandardName match``):

1. The Cypher used by ``_update_sources_after_compose`` /
   ``_update_sources_after_attach`` and by ``mark_sources_composed`` /
   ``mark_sources_attached`` must MATCH the target ``StandardName`` before
   it SETs the ``status`` / ``composed_at`` fields, so a missing SN never
   produces a composed/attached source without a ``PRODUCED_NAME`` edge.

2. The same Cypher must write a scalar ``produced_sn_id`` mirror property
   on the source for recoverability.

All tests use mocked ``GraphClient`` — no Neo4j instance is required.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_ops import (
    mark_sources_attached,
    mark_sources_composed,
)
from imas_codex.standard_names.workers import (
    _update_sources_after_attach,
    _update_sources_after_compose,
)


def _captured_query(mock_gc: MagicMock) -> str:
    """Return the Cypher string passed to ``gc.query`` on the mock."""
    assert mock_gc.return_value.__enter__.return_value.query.called, (
        "GraphClient.query was not invoked"
    )
    call = mock_gc.return_value.__enter__.return_value.query.call_args
    # The positional arg is the query string; support keyword form too.
    if call.args:
        return call.args[0]
    return call.kwargs["query"]


def _assert_match_before_set(cypher: str) -> None:
    """The sn MATCH must appear before the sns SET so status gates on sn."""
    stripped = " ".join(cypher.split())
    sn_match_idx = stripped.find("MATCH (sn:StandardName")
    sns_set_idx = stripped.find("SET sns.status")
    assert sn_match_idx != -1, f"missing MATCH (sn:StandardName in Cypher:\n{cypher}"
    assert sns_set_idx != -1, f"missing SET sns.status in Cypher:\n{cypher}"
    assert sn_match_idx < sns_set_idx, (
        "SET sns.status must not run before MATCH (sn:StandardName) — "
        "otherwise rows with no matching SN still bump status and leave "
        f"orphan sources. Cypher:\n{cypher}"
    )


def _assert_writes_produced_sn_id(cypher: str) -> None:
    assert "produced_sn_id" in cypher, (
        f"Cypher must set sns.produced_sn_id for recoverability:\n{cypher}"
    )


def _assert_merges_produced_edge(cypher: str) -> None:
    assert "PRODUCED_NAME" in cypher, (
        f"Cypher must MERGE the (:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName) "
        f"edge:\n{cypher}"
    )


class TestWorkerHelpers:
    """Cover _update_sources_after_compose and _update_sources_after_attach."""

    def _fake_candidates(self) -> list[dict]:
        return [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "id": "poloidal_magnetic_flux",
            },
            {
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "id": "electron_temperature",
            },
        ]

    def test_compose_cypher_matches_sn_before_set(self) -> None:
        log = logging.getLogger("test")
        adapter = logging.LoggerAdapter(log, {})
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"linked": 2}
            ]
            _update_sources_after_compose(self._fake_candidates(), "dd", adapter)

        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)
        assert "'composed'" in cypher

    def test_attach_cypher_matches_sn_before_set(self) -> None:
        log = logging.getLogger("test")
        adapter = logging.LoggerAdapter(log, {})

        class A:
            def __init__(self, sid: str, name: str) -> None:
                self.source_id = sid
                self.standard_name = name

        attachments = [
            A("equilibrium/time_slice/global_quantities/ip", "plasma_current")
        ]
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"linked": 1}
            ]
            _update_sources_after_attach(attachments, "dd", adapter)

        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)
        assert "'attached'" in cypher

    def test_compose_warns_on_partial_linking(self, caplog) -> None:
        """If the Cypher reports fewer linked sources than the batch size,
        the helper must log a warning so operators can spot lost edges."""
        log = logging.getLogger("test_linking")
        adapter = logging.LoggerAdapter(log, {})
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"linked": 1}
            ]
            with caplog.at_level(logging.WARNING, logger="test_linking"):
                _update_sources_after_compose(self._fake_candidates(), "dd", adapter)

        assert any("Compose-linking gap" in r.message for r in caplog.records), (
            "Expected a WARNING about partial linking when the DB reports "
            f"fewer linked sources than batch size. Got: {[r.message for r in caplog.records]}"
        )


class TestGraphOpsHelpers:
    """Cover the token-verified ``mark_sources_(composed|attached)`` helpers."""

    def test_mark_sources_composed_cypher_invariants(self) -> None:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"affected": 2}
            ]
            affected = mark_sources_composed(
                token="tok-123",
                source_ids=["dd:a", "dd:b"],
                standard_name_id="electron_temperature",
            )
        assert affected == 2
        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)

    def test_mark_sources_attached_cypher_invariants(self) -> None:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"affected": 1}
            ]
            affected = mark_sources_attached(
                token="tok-abc",
                source_ids=["dd:x"],
                standard_name_id="plasma_current",
            )
        assert affected == 1
        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)
