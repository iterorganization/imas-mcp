"""Renderer tests for fan-out evidence block (plan 39 §12.2)."""

from __future__ import annotations

from imas_codex.standard_names.fanout.render import format_results
from imas_codex.standard_names.fanout.schemas import FanoutHit, FanoutResult


def _hit(label: str, score: float | None = 0.9) -> FanoutHit:
    return FanoutHit(
        kind="standard_name",
        id=label.replace(" ", "_"),
        label=label,
        score=score,
    )


class TestFormatResults:
    def test_empty_inputs_return_empty_string(self) -> None:
        assert format_results([], result_hit_cap=8, evidence_token_cap=2000) == ""

    def test_all_failed_return_empty_string(self) -> None:
        results = [
            FanoutResult(fn_id="search_dd_paths", ok=False, error="timeout"),
            FanoutResult(fn_id="search_dd_clusters", ok=False, error="boom"),
        ]
        assert format_results(results, result_hit_cap=8, evidence_token_cap=2000) == ""

    def test_all_empty_hits_return_empty_string(self) -> None:
        results = [
            FanoutResult(fn_id="search_dd_paths", ok=True, hits=[]),
        ]
        assert format_results(results, result_hit_cap=8, evidence_token_cap=2000) == ""

    def test_per_hit_cap(self) -> None:
        result = FanoutResult(
            fn_id="search_existing_names",
            args={"fn_id": "search_existing_names", "query": "T_e", "k": 5},
            ok=True,
            hits=[_hit(f"name_{i}") for i in range(20)],
        )
        out = format_results([result], result_hit_cap=3, evidence_token_cap=2000)
        # Only 3 hit lines should be present.
        assert out.count("- name_") == 3
        assert "name_0" in out
        assert "name_2" in out
        assert "name_3" not in out

    def test_total_token_cap_baseline_vs_escalation(self) -> None:
        # 30 long-label hits → 30 lines.  Baseline cap should fit
        # them all (well under 2000 tokens).  Escalation cap (800)
        # is large enough too — so to make this test discriminating
        # we use a deliberately small escalation cap and assert
        # truncation marker.
        long = "x" * 200
        result = FanoutResult(
            fn_id="search_dd_paths",
            args={"fn_id": "search_dd_paths", "query": "ne", "k": 8},
            ok=True,
            hits=[_hit(f"{long}_{i}") for i in range(30)],
        )
        baseline = format_results([result], result_hit_cap=30, evidence_token_cap=2000)
        escalation = format_results([result], result_hit_cap=30, evidence_token_cap=80)
        assert "..." not in baseline
        assert "..." in escalation
        # Escalation cap must produce strictly less output than baseline.
        assert len(escalation) < len(baseline)

    def test_header_records_query_and_error_count(self) -> None:
        results = [
            FanoutResult(
                fn_id="search_dd_paths",
                args={"fn_id": "search_dd_paths", "query": "ne"},
                ok=True,
                hits=[_hit("hit1")],
            ),
            FanoutResult(fn_id="search_dd_clusters", ok=False, error="timeout"),
        ]
        out = format_results(results, result_hit_cap=8, evidence_token_cap=2000)
        assert "queries=2" in out
        assert "errors=1" in out

    def test_score_formatting(self) -> None:
        result = FanoutResult(
            fn_id="search_existing_names",
            args={"fn_id": "search_existing_names", "query": "T_e"},
            ok=True,
            hits=[_hit("electron_temperature", score=0.9134)],
        )
        out = format_results([result], result_hit_cap=8, evidence_token_cap=2000)
        # Score formatted to 2dp.
        assert "0.91" in out

    def test_drops_failed_runners_from_render(self) -> None:
        results = [
            FanoutResult(
                fn_id="search_existing_names",
                args={"fn_id": "search_existing_names", "query": "T_e"},
                ok=True,
                hits=[_hit("ok_name")],
            ),
            FanoutResult(fn_id="search_dd_paths", ok=False, error="timeout"),
        ]
        out = format_results(results, result_hit_cap=8, evidence_token_cap=2000)
        # Successful runner's section appears.
        assert "search_existing_names" in out
        assert "ok_name" in out
        # Failed runner's header does NOT appear (the format strips errors).
        assert "search_dd_paths(" not in out
