"""Tests for refine_name_user.md and refine_docs_user.md prompt rendering.

Uses ``render_prompt()`` from ``imas_codex.llm.prompt_loader`` — the same
function used by workers — to verify that key sections are present and that
chain history is rendered correctly.

No LLM calls are made; this is purely a template-rendering test.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import render_prompt

# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def _make_name_context(
    *,
    chain_history: list[dict] | None = None,
    hybrid_neighbours: list[dict] | None = None,
    chain_length: int = 2,
) -> dict:
    """Build a minimal context dict suitable for refine_name_user.md."""
    if chain_history is None:
        chain_history = [
            {
                "name": "electron_T_lcfs",
                "model": "gpt-4o",
                "reviewer_score": 0.42,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": {
                    "grammar": "Base token abbreviated — use 'temperature' not 'T'.",
                    "convention": "Locus suffix should be '_at_lcfs' per grammar.",
                },
                "generated_at": "2024-01-01T00:00:00Z",
            },
            {
                "name": "e_temp_at_separatrix",
                "model": "claude-3",
                "reviewer_score": 0.61,
                "reviewer_verdict": "revise",
                "reviewer_comments_per_dim": {
                    "semantic": "Method device leaked into name.",
                },
                "generated_at": "2024-01-02T00:00:00Z",
            },
        ]
    if hybrid_neighbours is None:
        hybrid_neighbours = [
            {
                "path": "core_profiles/profiles_1d/electrons/temperature",
                "description": "Electron temperature on 1D profiles",
            }
        ]
    return {
        "item": {
            "path": "equilibrium/time_slice/boundary/lcfs_electron_temp",
            "ids_name": "equilibrium",
            "description": "Electron temperature at the last closed flux surface",
            "unit": "eV",
            "data_type": "FLT_0D",
            "physics_domain": "equilibrium",
            "parent_path": "equilibrium/time_slice/boundary",
            "parent_description": "Plasma boundary",
        },
        "hybrid_neighbours": hybrid_neighbours,
        "chain_history": chain_history,
        "chain_length": chain_length,
    }


def _make_docs_context(
    *,
    docs_chain_history: list[dict] | None = None,
    docs_chain_length: int = 2,
) -> dict:
    """Build a minimal context dict suitable for refine_docs_user.md."""
    if docs_chain_history is None:
        docs_chain_history = [
            {
                "documentation": "The electron temperature Te at LCFS.",
                "model": "gpt-4o",
                "reviewer_score": 0.55,
                "reviewer_comments_per_dim": {
                    "clarity": "Too terse — add typical value ranges and measurement context.",
                    "physics": "Missing LaTeX notation for Te.",
                },
                "created_at": "2024-01-01T00:00:00Z",
            },
            {
                "documentation": "Electron temperature $T_e$ measured at the LCFS.",
                "model": "claude-3",
                "reviewer_score": 0.68,
                "reviewer_comments_per_dim": {
                    "links": "Missing link to ion_temperature.",
                },
                "created_at": "2024-01-02T00:00:00Z",
            },
        ]
    return {
        "sn_name": "electron_temperature_at_lcfs",
        "unit": "eV",
        "kind": "scalar",
        "physics_domain": "equilibrium",
        "description": "Electron temperature at the last closed flux surface",
        "dd_paths": [
            {
                "path": "equilibrium/time_slice/boundary/lcfs_electron_temp",
                "ids": "equilibrium",
                "unit": "eV",
                "documentation": "Electron temperature at boundary",
            }
        ],
        "docs_chain_history": docs_chain_history,
        "docs_chain_length": docs_chain_length,
    }


# ---------------------------------------------------------------------------
# refine_name_user.md
# ---------------------------------------------------------------------------


class TestRefineNamePrompt:
    @pytest.fixture()
    def rendered(self) -> str:
        return render_prompt("sn/refine_name_user", _make_name_context())

    def test_refinement_history_section_present(self, rendered: str):
        assert "Refinement history" in rendered

    def test_attempt_numbers_rendered(self, rendered: str):
        assert "Attempt 1" in rendered
        assert "Attempt 2" in rendered

    def test_reviewer_scores_rendered(self, rendered: str):
        # 0.42 and 0.61 should appear formatted
        assert "0.42" in rendered
        assert "0.61" in rendered

    def test_per_dimension_comments_rendered(self, rendered: str):
        assert "grammar" in rendered
        assert "abbreviated" in rendered
        assert "semantic" in rendered

    def test_prior_names_rendered(self, rendered: str):
        assert "electron_T_lcfs" in rendered
        assert "e_temp_at_separatrix" in rendered

    def test_chain_length_rendered(self, rendered: str):
        assert "chain length so far: 2" in rendered

    def test_path_context_rendered(self, rendered: str):
        assert "equilibrium/time_slice/boundary/lcfs_electron_temp" in rendered
        assert "eV" in rendered

    def test_hybrid_neighbours_rendered(self, rendered: str):
        assert "core_profiles/profiles_1d/electrons/temperature" in rendered

    def test_task_instruction_present(self, rendered: str):
        assert "Your task" in rendered
        assert "lowest-scoring dimensions" in rendered

    def test_do_not_repeat_history_rule(self, rendered: str):
        # The prompt uses markdown bold: "Do **not** repeat any name"
        assert "repeat" in rendered and "name" in rendered

    def test_empty_history_renders_gracefully(self):
        ctx = _make_name_context(chain_history=[], chain_length=0)
        rendered = render_prompt("sn/refine_name_user", ctx)
        assert "Refinement history" in rendered
        # Should not crash and should show the empty-history note
        assert "first refine attempt" in rendered

    def test_empty_neighbours_renders_gracefully(self):
        ctx = _make_name_context(hybrid_neighbours=[])
        rendered = render_prompt("sn/refine_name_user", ctx)
        assert "none available" in rendered

    def test_no_per_dim_comments_renders_gracefully(self):
        ctx = _make_name_context(
            chain_history=[
                {
                    "name": "some_name",
                    "model": "gpt-4o",
                    "reviewer_score": 0.50,
                    "reviewer_verdict": "revise",
                    "reviewer_comments_per_dim": {},
                    "generated_at": None,
                }
            ]
        )
        rendered = render_prompt("sn/refine_name_user", ctx)
        assert "no per-dimension comments recorded" in rendered


# ---------------------------------------------------------------------------
# refine_docs_user.md
# ---------------------------------------------------------------------------


class TestRefineDocsPrompt:
    @pytest.fixture()
    def rendered(self) -> str:
        return render_prompt("sn/refine_docs_user", _make_docs_context())

    def test_docs_revision_history_section_present(self, rendered: str):
        assert "Docs revision history" in rendered

    def test_revision_numbers_rendered(self, rendered: str):
        assert "Revision 1" in rendered
        assert "Revision 2" in rendered

    def test_reviewer_scores_rendered(self, rendered: str):
        assert "0.55" in rendered
        assert "0.68" in rendered

    def test_per_dimension_comments_rendered(self, rendered: str):
        assert "clarity" in rendered
        assert "Too terse" in rendered
        assert "links" in rendered

    def test_prior_documentation_rendered(self, rendered: str):
        assert "The electron temperature Te at LCFS" in rendered

    def test_docs_chain_length_rendered(self, rendered: str):
        assert "docs chain length so far: 2" in rendered

    def test_sn_name_rendered(self, rendered: str):
        assert "electron_temperature_at_lcfs" in rendered

    def test_unit_and_kind_rendered(self, rendered: str):
        assert "eV" in rendered
        assert "scalar" in rendered

    def test_dd_paths_rendered(self, rendered: str):
        assert "equilibrium/time_slice/boundary/lcfs_electron_temp" in rendered

    def test_task_instruction_present(self, rendered: str):
        assert "Your task" in rendered
        assert "lowest-scoring dimensions" in rendered

    def test_output_fields_specified(self, rendered: str):
        assert "documentation" in rendered
        assert "links" in rendered

    def test_empty_history_renders_gracefully(self):
        ctx = _make_docs_context(docs_chain_history=[], docs_chain_length=0)
        rendered = render_prompt("sn/refine_docs_user", ctx)
        assert "Docs revision history" in rendered
        assert "first docs refine attempt" in rendered

    def test_no_per_dim_comments_renders_gracefully(self):
        ctx = _make_docs_context(
            docs_chain_history=[
                {
                    "documentation": "Minimal docs.",
                    "model": "gpt-4o",
                    "reviewer_score": 0.60,
                    "reviewer_comments_per_dim": {},
                    "created_at": None,
                }
            ]
        )
        rendered = render_prompt("sn/refine_docs_user", ctx)
        assert "no per-dimension comments recorded" in rendered
