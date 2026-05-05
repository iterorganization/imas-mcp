"""Unit tests for the sn/refine_docs_user.md prompt template.

Verifies that the template renders without error across the range of
context shapes produced by ``process_refine_docs_batch`` in workers.py:

* Minimal context (no reviewer data, no DD paths, no chain history)
* Empty chain_history list
* With full reviewer feedback (score + per-dimension JSON comments)
* Missing optional fields (unit, physics_domain, description)
* reviewer_score_docs / reviewer_comments_per_dim_docs present as
  top-level context vars (not under ``item.*``) — regression guard for
  the 296aa4e6 / bug-refine-docs-render-fail fix where the template
  referenced ``item.reviewer_score_docs`` but the context never passes
  an ``item`` key.

No LLM calls are made.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.llm.prompt_loader import render_prompt

# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

_SN_ID = "toroidal_component_of_pfirsch_schlueter_current_density"
_HISTORY_ENTRY = {
    "documentation": "The Pfirsch-Schlüter current density toroidal component.",
    "model": "gpt-4o",
    "reviewer_score": 0.52,
    "reviewer_comments_per_dim": {
        "clarity": "Add typical value ranges.",
        "physics": "Missing LaTeX notation.",
    },
    "created_at": "2025-01-01T00:00:00Z",
}


def _minimal_ctx(**overrides) -> dict:
    """Context matching what workers.py builds — no reviewer data."""
    base: dict = {
        "sn_name": _SN_ID,
        "description": "Toroidal component of the Pfirsch-Schlüter current density.",
        "documentation": "Some existing docs.",
        "kind": "scalar",
        "unit": "A/m^2",
        "physics_domain": "equilibrium",
        "docs_chain_length": 0,
        "docs_chain_history": [],
        "reviewer_score_docs": None,
        "reviewer_comments_per_dim_docs": None,
        "dd_paths": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Core render tests
# ---------------------------------------------------------------------------


def test_renders_with_minimal_context():
    """Full workers.py context shape renders without exception."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx())
    assert result
    assert _SN_ID in result


def test_renders_with_empty_chain_history():
    """docs_chain_history=[] produces the 'no prior revision' message."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx(docs_chain_history=[]))
    assert result
    assert "no prior revision history" in result


def test_renders_with_chain_history():
    """Chain history entries are rendered with score and comments."""
    ctx = _minimal_ctx(
        docs_chain_history=[_HISTORY_ENTRY],
        docs_chain_length=1,
    )
    result = render_prompt("sn/refine_docs_user", ctx)
    assert "Revision 1" in result
    assert "0.52" in result
    assert "clarity" in result
    assert "Add typical value ranges" in result


def test_reviewer_section_present_with_score():
    """When reviewer_score_docs is set the 'Current node docs review' block appears."""
    ctx = _minimal_ctx(
        reviewer_score_docs=0.63,
        reviewer_comments_per_dim_docs=json.dumps(
            {"completeness": "Missing measurement context."}
        ),
    )
    result = render_prompt("sn/refine_docs_user", ctx)
    assert "Current node docs review" in result
    assert "0.63" in result
    assert "completeness" in result


def test_reviewer_section_absent_when_no_data():
    """Without reviewer data the 'Current node docs review' block is skipped."""
    result = render_prompt(
        "sn/refine_docs_user",
        _minimal_ctx(reviewer_score_docs=None, reviewer_comments_per_dim_docs=None),
    )
    assert "Current node docs review" not in result


def test_renders_without_item_key_in_context():
    """Context with no 'item' key must not raise UndefinedError.

    Regression guard: 296aa4e6 introduced ``item.reviewer_score_docs`` in
    the template, but workers.py never passes ``item``.  The fix replaces
    ``item.reviewer_score_docs`` with the top-level ``reviewer_score_docs``.
    """
    ctx = _minimal_ctx()
    assert "item" not in ctx  # confirm item is absent
    result = render_prompt("sn/refine_docs_user", ctx)
    assert result  # must not raise


# ---------------------------------------------------------------------------
# Missing / None optional fields
# ---------------------------------------------------------------------------


def test_renders_with_missing_unit():
    """unit=None / '' renders gracefully (falls back to '—')."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx(unit=None))
    assert result
    assert "—" in result


def test_renders_with_missing_physics_domain():
    """physics_domain='' renders gracefully."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx(physics_domain=""))
    assert result


def test_renders_with_missing_description():
    """description='' skips the one-line description line."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx(description=""))
    assert result
    assert "One-line description" not in result


def test_renders_with_empty_dd_paths():
    """dd_paths=[] renders the '(no linked DD paths)' placeholder."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx(dd_paths=[]))
    assert "(no linked DD paths)" in result


def test_renders_with_dd_paths_present():
    """dd_paths list is rendered correctly."""
    ctx = _minimal_ctx(
        dd_paths=[
            {
                "path": "equilibrium/time_slice/profiles_1d/j_tor",
                "ids": "equilibrium",
                "unit": "A/m^2",
                "documentation": "Toroidal current density.",
            }
        ]
    )
    result = render_prompt("sn/refine_docs_user", ctx)
    assert "j_tor" in result
    assert "Toroidal current density" in result


def test_sn_name_in_task_section():
    """The task section footer includes the SN name."""
    result = render_prompt("sn/refine_docs_user", _minimal_ctx())
    assert f"Produce updated documentation for `{_SN_ID}`" in result


def test_chain_length_displayed():
    """docs_chain_length is shown in the history section header."""
    result = render_prompt(
        "sn/refine_docs_user",
        _minimal_ctx(docs_chain_length=3, docs_chain_history=[_HISTORY_ENTRY] * 3),
    )
    assert "docs chain length so far: 3" in result


def test_reviewer_comments_none_renders_gracefully():
    """reviewer_comments_per_dim_docs=None but score set — no crash."""
    ctx = _minimal_ctx(
        reviewer_score_docs=0.45,
        reviewer_comments_per_dim_docs=None,
    )
    result = render_prompt("sn/refine_docs_user", ctx)
    assert "Current node docs review" in result
    assert "0.45" in result
