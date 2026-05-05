"""Tests for refine_name context richness: chain history + prior reviews.

Verifies that when a low-scoring SN cycles through refine_name, the returned
item and rendered prompt carry the full history of prior reviews.

Plan §3B.  Two classes of assertions:

1. **Passing** — chain_history is populated from REFINED_FROM ancestors and the
   rendered prompt contains the ancestors' per-dim comments verbatim.
2. **Xfail (known bugs)** — the current node's own review comment is not
   surfaced as a ``prior_reviews`` key on the returned item, and it does not
   appear in the rendered refine prompt.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GC_PATH = "imas_codex.standard_names.graph_ops.GraphClient"
# claim_refine_name_batch does a local import so patch the source symbol
_CHAIN_HIST_PATH = "imas_codex.standard_names.chain_history.name_chain_history"

# Fixture comments — chosen to be distinctive search strings
_A_PER_DIM: dict[str, str] = {"grammar": "missing subject"}
_B_PER_DIM: dict[str, str] = {"convention": "still ambiguous"}

# ---------------------------------------------------------------------------
# Mock helpers (mirrors test_refine_name_chain.py pattern)
# ---------------------------------------------------------------------------


def _mock_gc_tx() -> tuple[MagicMock, MagicMock]:
    """Return (gc, tx) mock pair wired for transactional claim queries."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


@contextmanager
def _patch_gc(gc: MagicMock):
    with patch(_GC_PATH, return_value=gc):
        yield


@contextmanager
def _patch_chain_history(return_value: list[dict] | None = None):
    with patch(_CHAIN_HIST_PATH, return_value=return_value or []):
        yield


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _ancestor_a_chain_entry() -> dict[str, Any]:
    """Simulate a name_chain_history entry for predecessor node A.

    Represents an earlier draft that was reviewed and scored 0.55 (below
    threshold), then superseded when B was generated.
    """
    return {
        "name": "temperature_too_vague_name",
        "model": "gpt-4o-mini",
        "reviewer_score": 0.55,
        "reviewer_comments_per_dim": _A_PER_DIM,
        "generated_at": "2024-01-01T00:00:00Z",
    }


def _b_item_row() -> dict[str, Any]:
    """Simulate the raw graph row for current node B.

    B has name_stage='reviewed' (→ 'refining' after claim) and chain_length=1,
    meaning it was already generated as one refinement of A.  Its reviewer
    score (0.60) is still below threshold, so it is eligible for another cycle.
    """
    return {
        "id": "temperature_still_ambiguous",
        "description": "Electron temperature near the separatrix",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": 0.60,
        "reviewer_comments_per_dim_name": json.dumps(_B_PER_DIM),
        "chain_length": 1,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
    }


# ---------------------------------------------------------------------------
# Helper: minimal prompt context (mirrors workers.process_refine_name_batch)
# ---------------------------------------------------------------------------


def _make_prompt_context(
    chain_history: list[dict],
    item_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the prompt context dict that workers.py passes to render_prompt."""
    item: dict[str, Any] = {
        "id": "temperature_still_ambiguous",
        "description": "Electron temperature near the separatrix",
        "documentation": None,
        "kind": "scalar",
        "unit": "eV",
        "cluster_id": None,
        "physics_domain": ["core_profiles"],
        "validation_status": "valid",
        "reviewer_score_name": 0.60,
        "reviewer_comments_per_dim_name": json.dumps(_B_PER_DIM),
        "chain_length": 1,
        "name_stage": "refining",
        "source_paths": ["core_profiles/profiles_1d/electrons/temperature"],
        "ids_name": "core_profiles",
        "parent_path": None,
        "parent_description": None,
        "data_type": "FLT_1D",
    }
    if item_overrides:
        item.update(item_overrides)
    return {
        "item": item,
        "chain_history": chain_history,
        "chain_length": len(chain_history),
        "hybrid_neighbours": [],
        "fanout_evidence": "",
    }


# =============================================================================
# 1.  claim_refine_name_batch — chain_history enrichment
# =============================================================================


class TestClaimRefineChainHistory:
    """claim_refine_name_batch enriches returned items with REFINED_FROM ancestors."""

    def test_chain_history_is_populated(self):
        """Returned item carries chain_history matching name_chain_history output."""
        from imas_codex.standard_names.graph_ops import claim_refine_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed query
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                # read-back query
                [_b_item_row()],
            ]
        )

        ancestor_chain = [_ancestor_a_chain_entry()]
        with _patch_gc(gc), _patch_chain_history(return_value=ancestor_chain):
            items = claim_refine_name_batch(min_score=0.7, batch_size=1)

        assert len(items) == 1
        assert items[0]["chain_history"] == ancestor_chain

    def test_chain_history_entry_has_per_dim_comments(self):
        """Ancestor entry in chain_history exposes reviewer_comments_per_dim dict."""
        from imas_codex.standard_names.graph_ops import claim_refine_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                [_b_item_row()],
            ]
        )

        ancestor_chain = [_ancestor_a_chain_entry()]
        with _patch_gc(gc), _patch_chain_history(return_value=ancestor_chain):
            items = claim_refine_name_batch(min_score=0.7, batch_size=1)

        entry = items[0]["chain_history"][0]
        assert isinstance(entry["reviewer_comments_per_dim"], dict)
        assert entry["reviewer_comments_per_dim"]["grammar"] == "missing subject"

    def test_chain_history_entry_has_reviewer_score(self):
        """Ancestor entry in chain_history exposes a numeric reviewer_score."""
        from imas_codex.standard_names.graph_ops import claim_refine_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                [_b_item_row()],
            ]
        )

        ancestor_chain = [_ancestor_a_chain_entry()]
        with _patch_gc(gc), _patch_chain_history(return_value=ancestor_chain):
            items = claim_refine_name_batch(min_score=0.7, batch_size=1)

        entry = items[0]["chain_history"][0]
        assert entry["reviewer_score"] == pytest.approx(0.55)

    # -------------------------------------------------------------------------
    # BUG: chain_history contains only REFINED_FROM ancestors; the current
    # node's own review is never merged into a unified prior_reviews list.
    # -------------------------------------------------------------------------

    def test_prior_reviews_key_present_with_full_chain(self):
        """Returned item should carry prior_reviews with EVERY review in the chain.

        Expected: prior_reviews = [A's review, B's review] (oldest first).
        Actual:   key is absent; only chain_history (ancestors) is returned.
        """
        from imas_codex.standard_names.graph_ops import claim_refine_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": "eV", "_physics_domain": "cp"}],
                [_b_item_row()],
            ]
        )

        ancestor_chain = [_ancestor_a_chain_entry()]
        with _patch_gc(gc), _patch_chain_history(return_value=ancestor_chain):
            items = claim_refine_name_batch(min_score=0.7, batch_size=1)

        item = items[0]
        # Should be a merged list containing both A's and B's reviews
        assert "prior_reviews" in item, (
            "'prior_reviews' key missing from claimed item — "
            "see plan §3B: full chain history not surfaced"
        )
        names_in_reviews = [r["name"] for r in item["prior_reviews"]]
        assert "temperature_too_vague_name" in names_in_reviews  # ancestor A
        assert "temperature_still_ambiguous" in names_in_reviews  # current B


# =============================================================================
# 2.  refine_name_user prompt rendering
# =============================================================================


class TestRefinePromptRendering:
    """Rendered refine_name_user prompt surfaces prior reviewer comments."""

    def test_prompt_contains_ancestor_per_dim_comment(self):
        """Rendered prompt includes ancestor A's per-dim comment verbatim."""
        from imas_codex.llm.prompt_loader import render_prompt

        chain_history = [_ancestor_a_chain_entry()]
        ctx = _make_prompt_context(chain_history)

        rendered = render_prompt("sn/refine_name_user", ctx)

        assert "missing subject" in rendered, (
            "Expected ancestor A's per-dim comment 'missing subject' in rendered "
            "refine prompt — check chain_history template rendering"
        )

    def test_prompt_contains_ancestor_reviewer_score(self):
        """Rendered prompt includes ancestor A's numeric reviewer score."""
        from imas_codex.llm.prompt_loader import render_prompt

        chain_history = [_ancestor_a_chain_entry()]
        ctx = _make_prompt_context(chain_history)

        rendered = render_prompt("sn/refine_name_user", ctx)

        assert "0.55" in rendered, (
            "Expected ancestor A's reviewer score '0.55' in rendered refine prompt"
        )

    def test_prompt_contains_ancestor_name(self):
        """Rendered prompt names the specific prior attempt that was rejected."""
        from imas_codex.llm.prompt_loader import render_prompt

        chain_history = [_ancestor_a_chain_entry()]
        ctx = _make_prompt_context(chain_history)

        rendered = render_prompt("sn/refine_name_user", ctx)

        assert "temperature_too_vague_name" in rendered

    def test_prompt_without_chain_history_shows_fallback(self):
        """Empty chain_history renders the 'no prior history' fallback message."""
        from imas_codex.llm.prompt_loader import render_prompt

        ctx = _make_prompt_context(chain_history=[])

        rendered = render_prompt("sn/refine_name_user", ctx)

        # Template: "_(no prior refinement history — this is the first refine attempt)_"
        assert "no prior refinement history" in rendered

    # -------------------------------------------------------------------------
    # BUG: current node B's own review comment does not appear in the prompt.
    # The template iterates chain_history (ancestors only); item's own
    # reviewer_comments_per_dim_name is never injected into the history section.
    # -------------------------------------------------------------------------

    def test_prompt_contains_current_node_review_comment(self):
        """Rendered prompt must include B's reviewer comment 'still ambiguous'.

        Expected:   'still ambiguous' appears in the Refinement history section.
        Actual:     only chain_history (A's data) is rendered; B's per-dim
                    comment is silently dropped.
        """
        from imas_codex.llm.prompt_loader import render_prompt

        chain_history = [_ancestor_a_chain_entry()]
        ctx = _make_prompt_context(chain_history)

        rendered = render_prompt("sn/refine_name_user", ctx)

        assert "still ambiguous" in rendered, (
            "Expected current node B's review comment 'still ambiguous' in rendered "
            "refine prompt — the template must also render item.reviewer_comments_per_dim_name"
        )
