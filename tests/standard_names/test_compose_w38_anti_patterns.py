"""W38 compose-prompt anti-pattern hardening tests.

Verifies that the three W38 anti-patterns (instrument prefix carry-over,
suffix-form for component, compound hardware identifiers) are present in
both the system prompts (generate_name_system.md, generate_name_system_lean.md) and
the user prompts (generate_name_dd.md, generate_name_dd_names.md).

These are content/structural assertions — they do not call the LLM.
"""

from __future__ import annotations

import pytest

from imas_codex.llm.prompt_loader import PROMPTS_DIR, render_prompt


def _load(name: str) -> str:
    return (PROMPTS_DIR / "sn" / name).read_text(encoding="utf-8")


# Concrete exemplars from the W37 rotation (real bad/good pairs).
W38_BAD_EXAMPLES = (
    "x_ray_crystal_spectrometer_pixel_photon_energy_lower_bound",
    "halo_region_parallel_energy_due_to_heat_flux",
    "z_coordinate_of_sensor_direction_unit_vector",
)
W38_GOOD_EXAMPLES = (
    "photon_energy_lower_bound",
    "parallel_component_of_halo_energy",
    "z_component_of_direction_unit_vector",
)
W38_HARDWARE_PROPERTY_EXEMPLAR = "cross_sectional_area_of_rogowski_coil"


@pytest.mark.parametrize(
    "filename", ["generate_name_system.md", "generate_name_system_lean.md"]
)
class TestSystemPromptW38Gallery:
    """The W38 anti-pattern entries must appear in both system prompts."""

    def test_gallery_heading(self, filename: str) -> None:
        raw = _load(filename)
        assert "ANTI-PATTERN REFERENCE" in raw or "W38 ANTI-PATTERN GALLERY" in raw

    def test_three_anti_pattern_headers(self, filename: str) -> None:
        raw = _load(filename)
        assert "W38-A1" in raw
        assert "W38-A2" in raw
        assert "W38-A3" in raw

    @pytest.mark.parametrize("bad", W38_BAD_EXAMPLES)
    def test_bad_example_present(self, filename: str, bad: str) -> None:
        assert bad in _load(filename)

    @pytest.mark.parametrize("good", W38_GOOD_EXAMPLES)
    def test_good_example_present(self, filename: str, good: str) -> None:
        assert good in _load(filename)

    def test_hardware_property_exception(self, filename: str) -> None:
        # A1 must explicitly carve out the hardware-property exception so the
        # generator does not over-strip instrument tokens.
        raw = _load(filename)
        assert W38_HARDWARE_PROPERTY_EXEMPLAR in raw
        assert "intrinsic" in raw.lower()

    def test_decision_rule_present_per_block(self, filename: str) -> None:
        raw = _load(filename)
        # Each block ends with a "Decision rule:" line.
        assert raw.count("Decision rule") >= 3


@pytest.mark.parametrize(
    "filename", ["generate_name_dd.md", "generate_name_dd_names.md"]
)
class TestUserPromptW38TableRows:
    """The user-facing per-batch tables must include W38 row entries."""

    def test_w38_tags_in_table(self, filename: str) -> None:
        raw = _load(filename)
        assert "W38-A1" in raw
        assert "W38-A2" in raw
        assert "W38-A3" in raw

    @pytest.mark.parametrize(
        "bad,good",
        list(zip(W38_BAD_EXAMPLES, W38_GOOD_EXAMPLES, strict=True)),
    )
    def test_concrete_pairs(self, filename: str, bad: str, good: str) -> None:
        raw = _load(filename)
        assert bad in raw
        assert good in raw


class TestSystemPromptW38Placement:
    """Anti-patterns must come AFTER the static schema/grammar block (includes)
    but BEFORE per-item context — i.e. inside the static system prompt cache
    layer, not inside the dynamic user prompt."""

    def test_compose_system_after_includes(self) -> None:
        raw = _load("generate_name_system.md")
        # W38 must appear after the prelude includes; trailing tail includes
        # (e.g. _coordinate_conventions.md) may legitimately appear later.
        prelude_end = raw.index('{% include "sn/_exemplars_name_only.md" %}')
        w38_pos = raw.index("W38-A1")
        assert w38_pos > prelude_end

    def test_emw_before_w38_in_reference(self) -> None:
        # EMW entries appear before W38 entries in the consolidated reference.
        raw = _load("generate_name_system.md")
        emw_pos = raw.index("EMW-1")
        w38_pos = raw.index("W38-A1")
        assert w38_pos > emw_pos

    def test_compose_system_before_curated_examples(self) -> None:
        raw = _load("generate_name_system.md")
        w38_pos = raw.index("W38-A1")
        examples_pos = raw.index("## Curated Examples")
        assert w38_pos < examples_pos


class TestSystemPromptRendersWithDefaultContext:
    """Render the system prompt with a representative compose context and
    confirm the W38 block survives Jinja rendering (no template error,
    block visible at runtime)."""

    @pytest.fixture
    def context(self) -> dict:
        from imas_codex.standard_names.context import build_compose_context

        return build_compose_context()

    def test_compose_system_renders(self, context: dict) -> None:
        rendered = render_prompt("sn/generate_name_system", context)
        assert "W38-A1" in rendered
        for bad in W38_BAD_EXAMPLES:
            assert bad in rendered
        for good in W38_GOOD_EXAMPLES:
            assert good in rendered

    def test_compose_system_lean_renders(self, context: dict) -> None:
        rendered = render_prompt("sn/generate_name_system_lean", context)
        assert "W38 ANTI-PATTERN GALLERY" in rendered
        for bad in W38_BAD_EXAMPLES:
            assert bad in rendered
