"""Tests for the 6-pool streaming display module.

13 tests covering rendering, color, ETA, ETC, throttle, deque rotation,
and per-item line format for all 6 pool types.

All tests exercise **pure rendering functions** — no ``Live`` context,
no async loop, no graph connection required.
"""

from __future__ import annotations

import time

import pytest
from rich.text import Text

from imas_codex.standard_names.display import (
    BAR_WIDTH,
    POOL_LABELS,
    POOL_ORDER,
    STREAM_MAXLEN,
    PoolDisplayState,
    compute_eta,
    compute_etc,
    format_item_generate_docs,
    format_item_generate_name,
    format_item_refine_docs,
    format_item_refine_name,
    format_item_review_docs,
    format_item_review_name,
    make_bar,
    render_footer,
    render_pool_panel,
    score_color,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _pool(
    name: str = "review_name",
    completed: int = 0,
    total: int = 100,
    cost: float = 0.0,
    throttled: bool = False,
    throttle_reason: str = "",
    items: list | None = None,
) -> PoolDisplayState:
    """Convenience factory for PoolDisplayState."""
    p = PoolDisplayState(
        name=name,
        completed=completed,
        total=total,
        cost=cost,
        start_time=time.time() - 10.0,  # 10s ago
        throttled=throttled,
        throttle_reason=throttle_reason,
    )
    for it in items or []:
        p.add_item(it)
    return p


def _text(t: Text) -> str:
    """Extract plain text from a Rich Text object."""
    return t.plain


# ═══════════════════════════════════════════════════════════════════════
# 1. Pool panel renders progress bar
# ═══════════════════════════════════════════════════════════════════════


class TestPoolPanelRendersProgressBar:
    def test_bar_at_25_percent(self):
        """Pool with 50/200 completed renders bar with ~25% fill."""
        state = _pool("generate_name", completed=50, total=200)
        rendered = _text(render_pool_panel(state))
        assert "GENERATE_NAME" in rendered
        assert "50/200" in rendered
        assert "25%" in rendered

    def test_bar_at_zero(self):
        state = _pool("review_docs", completed=0, total=100)
        rendered = _text(render_pool_panel(state))
        assert "0/100" in rendered
        assert "0%" in rendered

    def test_bar_at_100(self):
        state = _pool("refine_docs", completed=100, total=100)
        rendered = _text(render_pool_panel(state))
        assert "100/100" in rendered
        assert "100%" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 2. Review score colors
# ═══════════════════════════════════════════════════════════════════════


class TestReviewScoreColor:
    def test_green_at_0_90(self):
        assert score_color(0.90) == "green"

    def test_green_at_threshold(self):
        assert score_color(0.85) == "green"

    def test_yellow_at_0_70(self):
        assert score_color(0.70) == "yellow"

    def test_yellow_at_lower_threshold(self):
        assert score_color(0.65) == "yellow"

    def test_red_at_0_40(self):
        assert score_color(0.40) == "red"

    def test_red_at_0_64(self):
        assert score_color(0.64) == "red"


# ═══════════════════════════════════════════════════════════════════════
# 3. Per-item line format: REVIEW_NAME
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemReviewName:
    def test_includes_name_score_comment(self):
        item = {
            "name": "e_temp_core",
            "score": 0.83,
            "comment": "Good grammar; documentation could be more specific about the core region",
        }
        rendered = _text(format_item_review_name(item))
        assert "e_temp_core" in rendered
        assert "0.83" in rendered
        assert "Good grammar" in rendered

    def test_score_two_decimals(self):
        item = {"name": "test", "score": 0.7, "comment": "ok"}
        rendered = _text(format_item_review_name(item))
        assert "0.70" in rendered

    def test_comment_clipped_at_80(self):
        long_comment = "A" * 200
        item = {"name": "test", "score": 0.5, "comment": long_comment}
        rendered = _text(format_item_review_name(item))
        # The comment should be clipped, so full 200 A's should not appear
        assert "A" * 100 not in rendered

    def test_score_color_applied(self):
        """High score uses green style, low uses red."""
        item_high = {"name": "test", "score": 0.90, "comment": ""}
        text_high = format_item_review_name(item_high)
        # Find the span containing the score
        score_spans = [
            span
            for span in text_high._spans
            if "0.90" in text_high.plain[span.start : span.end]
        ]
        assert any(span.style == "green" for span in score_spans)

        item_low = {"name": "test", "score": 0.40, "comment": ""}
        text_low = format_item_review_name(item_low)
        score_spans_low = [
            span
            for span in text_low._spans
            if "0.40" in text_low.plain[span.start : span.end]
        ]
        assert any(span.style == "red" for span in score_spans_low)


# ═══════════════════════════════════════════════════════════════════════
# 4. Per-item line format: GENERATE_NAME
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemGenerateName:
    def test_source_arrow_name(self):
        item = {
            "source": "equilibrium/time_slice/profiles_1d/psi",
            "name": "poloidal_flux_in_equilibrium",
        }
        rendered = _text(format_item_generate_name(item))
        assert "→" in rendered
        assert "poloidal_flux_in_equilibrium" in rendered

    def test_dd_path_fallback(self):
        item = {"dd_path": "core_profiles/profiles_1d/electrons/temperature"}
        rendered = _text(format_item_generate_name(item))
        assert "core_profiles" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 5. Per-item line format: REFINE_NAME
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemRefineName:
    def test_old_arrow_new_with_chain(self):
        item = {
            "old_name": "separatrix_n_i",
            "new_name": "ion_density_at_lcfs",
            "chain_length": 1,
            "escalated": False,
        }
        rendered = _text(format_item_refine_name(item))
        assert "separatrix_n_i" in rendered
        assert "(chain=1)" in rendered
        assert "ion_density_at_lcfs" in rendered
        assert "→" in rendered

    def test_escalation_flag(self):
        item = {
            "old_name": "e_temp_pedestal",
            "chain_length": 2,
            "escalated": True,
            "model": "opus-4.6",
        }
        rendered = _text(format_item_refine_name(item))
        assert "escalating to opus-4.6" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 6. Per-item line format: GENERATE_DOCS
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemGenerateDocs:
    def test_name_and_description_preview(self):
        item = {
            "name": "e_temp_core",
            "description": "The temperature of electrons in the core plasma region",
        }
        rendered = _text(format_item_generate_docs(item))
        assert "e_temp_core" in rendered
        assert "temperature of electrons" in rendered

    def test_description_clipped_at_100(self):
        long_desc = "X" * 200
        item = {"name": "test", "description": long_desc}
        rendered = _text(format_item_generate_docs(item))
        assert "X" * 150 not in rendered


# ═══════════════════════════════════════════════════════════════════════
# 7. Per-item line format: REVIEW_DOCS
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemReviewDocs:
    def test_same_format_as_review_name(self):
        item = {
            "name": "e_temp_core",
            "score": 0.78,
            "comment": "Description is clear but lacks SI units",
        }
        rendered = _text(format_item_review_docs(item))
        assert "0.78" in rendered
        assert "SI units" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 8. Per-item line format: REFINE_DOCS
# ═══════════════════════════════════════════════════════════════════════


class TestPerItemRefineDocs:
    def test_name_rev_description(self):
        item = {
            "name": "e_temp_core",
            "revision": 1,
            "description": "The mean kinetic energy per electron in the core region",
        }
        rendered = _text(format_item_refine_docs(item))
        assert "e_temp_core" in rendered
        assert "(rev=1)" in rendered
        assert "kinetic energy" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 9. Throttled pool label
# ═══════════════════════════════════════════════════════════════════════


class TestThrottledPoolLabel:
    def test_paused_suffix_and_backlog_count(self):
        state = _pool(
            "generate_name",
            completed=720,
            total=2000,
            throttled=True,
            throttle_reason="review_name backlog 207>200",
        )
        rendered = _text(render_pool_panel(state))
        assert "GENERATE_NAME" in rendered
        assert "[paused:" in rendered
        assert "review_name backlog 207>200" in rendered

    def test_not_throttled_no_paused(self):
        state = _pool("generate_name", completed=720, total=2000)
        rendered = _text(render_pool_panel(state))
        assert "[paused" not in rendered


# ═══════════════════════════════════════════════════════════════════════
# 10. Per-pool cost accumulator
# ═══════════════════════════════════════════════════════════════════════


class TestPerPoolCostAccumulator:
    def test_sum_of_costs(self):
        state = _pool("review_name", completed=10, total=100, cost=0.0)
        # Simulate feeding 5 cost events
        costs = [0.10, 0.15, 0.20, 0.25, 0.30]
        for c in costs:
            state.cost += c

        assert abs(state.cost - 1.00) < 1e-9

        rendered = _text(render_pool_panel(state))
        assert "$1.00" in rendered


# ═══════════════════════════════════════════════════════════════════════
# 11. ETA calculation
# ═══════════════════════════════════════════════════════════════════════


class TestETACalculation:
    def test_eta_from_throughput(self):
        """Given throughput and remaining work, ETA is computed correctly."""
        # 50 items done in 10s → 5/s; 150 remaining → 30s ETA
        pools = [_pool("generate_name", completed=50, total=200)]
        # Force start_time to 10s ago
        pools[0].start_time = time.time() - 10.0
        eta = compute_eta(pools)
        assert eta is not None
        assert abs(eta - 30.0) < 2.0  # allow small timing variance

    def test_eta_none_when_no_progress(self):
        pools = [_pool("generate_name", completed=0, total=200)]
        eta = compute_eta(pools)
        assert eta is None

    def test_eta_zero_when_all_done(self):
        pools = [_pool("generate_name", completed=200, total=200)]
        eta = compute_eta(pools)
        assert eta is not None
        assert eta == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 12. ETC projection
# ═══════════════════════════════════════════════════════════════════════


class TestETCProjection:
    def test_etc_equals_cost_per_item_times_remaining_plus_current(self):
        """ETC = current_cost + cost_per_item × remaining."""
        # 10 items done at $0.10/item = $1.00; 90 remaining → ETC = $1 + 90*0.10 = $10
        pool = _pool("generate_name", completed=10, total=100, cost=1.00)
        etc = compute_etc([pool])
        assert etc is not None
        assert abs(etc - 10.0) < 1e-6

    def test_etc_none_when_no_items(self):
        pool = _pool("generate_name", completed=0, total=100, cost=0.0)
        etc = compute_etc([pool])
        assert etc is None


# ═══════════════════════════════════════════════════════════════════════
# 13. Streamed items deque rotates
# ═══════════════════════════════════════════════════════════════════════


class TestStreamedItemsDequeRotates:
    def test_maxlen_3_keeps_last_3(self):
        """Feeding 5 items into a maxlen=3 deque keeps only the last 3."""
        state = _pool("review_name")
        for i in range(5):
            state.add_item({"name": f"sn_{i}", "score": 0.5, "comment": f"comment {i}"})

        assert len(state.items) == STREAM_MAXLEN
        names = [it["name"] for it in state.items]
        assert names == ["sn_2", "sn_3", "sn_4"]

    def test_items_appear_in_rendered_output(self):
        state = _pool("review_name", completed=5, total=100)
        state.add_item({"name": "alpha", "score": 0.90, "comment": "good"})
        state.add_item({"name": "beta", "score": 0.60, "comment": "needs work"})

        rendered = _text(render_pool_panel(state))
        assert "alpha" in rendered
        assert "beta" in rendered


# ═══════════════════════════════════════════════════════════════════════
# Footer rendering
# ═══════════════════════════════════════════════════════════════════════


class TestFooterRendering:
    def test_time_row_shows_elapsed(self):
        pools = [_pool("generate_name", completed=50, total=200)]
        footer = _text(render_footer(pools))
        assert "TIME" in footer

    def test_cost_row_shows_total_and_cap(self):
        pools = [_pool("generate_name", completed=50, total=200, cost=1.50)]
        footer = _text(render_footer(pools, cost_limit=10.0))
        assert "COST" in footer
        assert "$1.50" in footer
        assert "CAP $10.00" in footer

    def test_servers_row(self):
        pools = [_pool("generate_name", completed=50, total=200)]
        footer = _text(
            render_footer(
                pools,
                graph_latency_ms=8.0,
                llm_latency_s=1.2,
                graph_host="graph:titan",
                llm_host="llm:iter",
            )
        )
        assert "SERVERS" in footer
        assert "graph:titan (avg 8ms)" in footer
        assert "llm:iter (avg 1.2s)" in footer

    def test_pool_labels_are_correct(self):
        """All 6 pool labels use the canonical GENERATE_NAME etc. form."""
        expected_labels = {
            "GENERATE_NAME",
            "REVIEW_NAME",
            "REFINE_NAME",
            "GENERATE_DOCS",
            "REVIEW_DOCS",
            "REFINE_DOCS",
        }
        assert set(POOL_LABELS.values()) == expected_labels
        # No old labels
        for old_label in ("DRAFT", "REVISE", "DESCRIBE", "DOCUMENTATION", "ENRICH"):
            assert old_label not in POOL_LABELS.values()

    def test_pool_order_is_six(self):
        assert len(POOL_ORDER) == 6
