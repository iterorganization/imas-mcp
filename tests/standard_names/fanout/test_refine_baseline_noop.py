"""Refine-name prompt baseline: empty fan-out evidence is byte-identical.

Plan 39 §6.3 + §11 acceptance #1: when ``fanout_evidence == ""`` the
rendered ``sn/refine_name_user`` prompt is byte-identical to the
prompt rendered with ``fanout_evidence`` absent.

This guards against accidental Jinja whitespace bleed when the
``{{ fanout_evidence }}`` placeholder is empty — a frequent regression
when conditional sections are introduced.

The "golden" fixture is generated dynamically at first run from the
no-evidence render so the test stays robust to unrelated edits to the
prompt body (the *invariant* is byte-identical, not byte-stable
across edits).
"""

from __future__ import annotations

from imas_codex.llm.prompt_loader import render_prompt


def _ctx(*, fanout_evidence: str | None = None) -> dict:
    base = {
        "item": {
            "path": "core_profiles/profiles_1d/electrons/temperature",
            "ids_name": "core_profiles",
            "description": "Electron temperature",
            "unit": "eV",
            "data_type": "FLT_1D",
            "physics_domain": "kinetics",
            "parent_path": "core_profiles/profiles_1d/electrons",
            "parent_description": "Electron-related profiles.",
        },
        "chain_history": [
            {
                "name": "electron_temp",
                "model": "test/model",
                "reviewer_score": 0.6,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": {
                    "clarity": "name is unclear; consider electron_temperature",
                },
            }
        ],
        "chain_length": 1,
        "hybrid_neighbours": [],
    }
    if fanout_evidence is not None:
        base["fanout_evidence"] = fanout_evidence
    return base


def test_empty_evidence_byte_identical_to_omitted() -> None:
    """``fanout_evidence=""`` renders identically to omitting the var."""
    rendered_omitted = render_prompt("sn/refine_name_user", _ctx())
    rendered_empty = render_prompt("sn/refine_name_user", _ctx(fanout_evidence=""))
    assert rendered_empty == rendered_omitted, (
        "fanout_evidence='' must render byte-identical to baseline "
        "(no extra whitespace, no headers, no separators)."
    )


def test_evidence_appears_when_non_empty() -> None:
    """Sanity: a non-empty evidence string is injected into the prompt."""
    marker = "## Fan-out evidence (queries=2, errors=0)\n- electron_temperature_at_axis"
    rendered = render_prompt("sn/refine_name_user", _ctx(fanout_evidence=marker))
    assert marker in rendered


def test_baseline_does_not_contain_evidence_header() -> None:
    """When evidence is empty, the literal evidence header must not appear."""
    rendered = render_prompt("sn/refine_name_user", _ctx(fanout_evidence=""))
    assert "Fan-out evidence" not in rendered
