"""Telemetry node writer tests (plan 39 §8.2, §12.2)."""

from __future__ import annotations

from typing import Any

from imas_codex.standard_names.fanout.telemetry import write_fanout_node


class _MockGraphClient:
    """Captures every ``query`` call for inspection."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._next_return: list[dict[str, Any]] = []

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append({"cypher": cypher, "params": params})
        # Mirror the CREATE…RETURN id contract.
        return [{"id": params.get("id")}]


class _RaisingGraphClient:
    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        raise RuntimeError("graph unavailable")


class TestWriteFanoutNode:
    def test_writes_node_with_correct_shape(self) -> None:
        gc = _MockGraphClient()
        rid = write_fanout_node(
            gc,
            run_id="abc-123",
            sn_id="electron_temperature",
            site="refine_name",
            outcome="ok",
            plan_size=2,
            hits_count=15,
            evidence_tokens=412,
            arm="on",
            escalate=False,
        )
        assert rid == "abc-123"
        assert len(gc.calls) == 1
        c = gc.calls[0]
        # Cypher creates a Fanout node with all required properties.
        assert ":Fanout" in c["cypher"]
        for key in (
            "id",
            "sn_id",
            "site",
            "outcome",
            "plan_size",
            "hits_count",
            "evidence_tokens",
            "arm",
            "escalate",
            "created_at",
        ):
            assert key in c["params"], f"missing param: {key}"
        assert c["params"]["id"] == "abc-123"
        assert c["params"]["sn_id"] == "electron_temperature"
        assert c["params"]["outcome"] == "ok"
        assert c["params"]["plan_size"] == 2
        assert c["params"]["hits_count"] == 15
        assert c["params"]["arm"] == "on"
        assert c["params"]["escalate"] is False

    def test_skipped_when_gc_is_none(self) -> None:
        # The disabled path passes gc=None — no write should happen.
        rid = write_fanout_node(
            None,
            run_id="abc-123",
            sn_id="x",
            site="refine_name",
            outcome="ok",
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
        )
        assert rid is None

    def test_off_arm_outcome_recorded(self) -> None:
        gc = _MockGraphClient()
        write_fanout_node(
            gc,
            run_id="off-1",
            sn_id="x",
            site="refine_name",
            outcome="off_arm",
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
            arm="off",
        )
        assert gc.calls[0]["params"]["outcome"] == "off_arm"
        assert gc.calls[0]["params"]["arm"] == "off"

    def test_failure_logged_not_raised(self) -> None:
        # Telemetry failures must never break the parent refine cycle.
        gc = _RaisingGraphClient()
        rid = write_fanout_node(
            gc,
            run_id="boom",
            sn_id="x",
            site="refine_name",
            outcome="ok",
            plan_size=0,
            hits_count=0,
            evidence_tokens=0,
        )
        assert rid is None
