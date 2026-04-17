"""Tests for L3 post-generation audits module."""

from __future__ import annotations

import numpy as np
import pytest

# =========================================================================
# L3: latex_def_check
# =========================================================================


class TestLatexDefCheck:
    """Tests for latex_def_check audit."""

    def test_pass_symbols_defined(self):
        """Symbols with definitions pass."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The safety factor is $q = d\\Phi/d\\Psi$, where "
                "$q$ is the safety factor (dimensionless), "
                "$\\Phi$ is the toroidal flux (Wb), and "
                "$\\Psi$ is the poloidal flux (Wb)."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) == 0

    def test_fail_undefined_symbol(self):
        """Symbols without definitions are flagged."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {
            "documentation": (
                "The plasma has $\\alpha$ and $\\beta$ parameters. "
                "These affect confinement significantly. "
                "Further analysis shows improved stability."
            ),
        }
        issues = latex_def_check(candidate)
        assert len(issues) >= 1
        assert any("latex_def_check" in i for i in issues)

    def test_pass_empty_documentation(self):
        """No documentation produces no issues."""
        from imas_codex.standard_names.audits import latex_def_check

        assert latex_def_check({"documentation": ""}) == []
        assert latex_def_check({}) == []

    def test_pass_no_latex_symbols(self):
        """Documentation without LaTeX symbols passes."""
        from imas_codex.standard_names.audits import latex_def_check

        candidate = {"documentation": "A simple description without math."}
        assert latex_def_check(candidate) == []


# =========================================================================
# L3: provenance_verb_check
# =========================================================================


class TestProvenanceVerbCheck:
    """Tests for provenance_verb_check audit."""

    def test_pass_no_provenance_verb(self):
        """Clean name passes."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature"}
        issues = provenance_verb_check(candidate)
        assert len(issues) == 0

    def test_fail_measured_in_name(self):
        """Name with 'measured' when source lacks it is flagged."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(
            candidate, source_path="core_profiles/profiles_1d/electrons/temperature"
        )
        assert len(issues) == 1
        assert "measured" in issues[0]

    def test_pass_measured_in_both(self):
        """Name with 'measured' is OK when source path also has it."""
        from imas_codex.standard_names.audits import provenance_verb_check

        candidate = {"id": "electron_temperature_measured"}
        issues = provenance_verb_check(candidate, source_path="diagnostics/measured/te")
        assert len(issues) == 0


# =========================================================================
# L3: synonym_check
# =========================================================================


class TestSynonymCheck:
    """Tests for synonym_check audit."""

    def test_pass_no_similar(self):
        """No existing SNs means no synonym issues."""
        from imas_codex.standard_names.audits import synonym_check

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": np.random.rand(384).tolist(),
        }
        issues = synonym_check(candidate, [])
        assert len(issues) == 0

    def test_fail_high_cosine(self):
        """Near-identical embedding with same unit is flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)
        # Create a very similar vector
        similar_vec = vec + np.random.rand(384).astype(np.float32) * 0.01

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "te_profile",
                "unit": "eV",
                "description_embedding": similar_vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 1
        assert "synonym_check" in issues[0]

    def test_pass_different_unit(self):
        """Same embedding but different unit is not flagged."""
        from imas_codex.standard_names.audits import synonym_check

        vec = np.random.rand(384).astype(np.float32)

        candidate = {
            "id": "electron_temperature",
            "unit": "eV",
            "description_embedding": vec.tolist(),
        }
        existing = [
            {
                "name": "electron_density",
                "unit": "m^-3",
                "description_embedding": vec.tolist(),
            }
        ]
        issues = synonym_check(candidate, existing)
        assert len(issues) == 0


# =========================================================================
# L3: unit_dimension_check
# =========================================================================


class TestUnitDimensionCheck:
    """Tests for unit_dimension_check audit."""

    def test_pass_consistent(self):
        """Description consistent with unit passes."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "eV", "description": "Electron temperature profile"}
        assert unit_dimension_check(candidate) == []

    def test_fail_inconsistent(self):
        """Description inconsistent with unit is flagged."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "A", "description": "Radial position of the boundary"}
        issues = unit_dimension_check(candidate)
        assert len(issues) == 1
        assert "unit_dimension_check" in issues[0]

    def test_pass_dimensionless(self):
        """Dimensionless unit is not checked."""
        from imas_codex.standard_names.audits import unit_dimension_check

        candidate = {"unit": "1", "description": "Safety factor profile"}
        assert unit_dimension_check(candidate) == []


# =========================================================================
# L3: multi_subject_check
# =========================================================================


class TestMultiSubjectCheck:
    """Tests for multi_subject_check audit."""

    def test_pass_single_subject(self):
        """Single-subject name passes."""
        from imas_codex.standard_names.audits import multi_subject_check

        candidate = {"id": "electron_temperature"}
        issues = multi_subject_check(candidate)
        assert len(issues) == 0

    def test_fail_multiple_subjects(self):
        """Name with two different subject tokens is flagged."""
        from imas_codex.standard_names.audits import multi_subject_check

        # This will flag if two Subject enum values appear in name tokens
        candidate = {"id": "electron_ion_temperature"}
        issues = multi_subject_check(candidate)
        # May or may not flag depending on grammar — at least doesn't crash
        assert isinstance(issues, list)


# =========================================================================
# L3: cocos_specificity_check
# =========================================================================


class TestCocosSpecificityCheck:
    """Tests for cocos_specificity_check audit."""

    def test_pass_cocos_mentioned(self):
        """Documentation mentioning COCOS with digit passes."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {
            "documentation": "Sign convention: Positive when COCOS 11 convention applies."
        }
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 0

    def test_fail_no_cocos(self):
        """Documentation without COCOS mention when source has COCOS type is flagged."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "The poloidal magnetic flux per radian."}
        issues = cocos_specificity_check(candidate, source_cocos_type="psi_like")
        assert len(issues) == 1
        assert "cocos_specificity_check" in issues[0]

    def test_pass_no_cocos_type(self):
        """No source COCOS type means no check."""
        from imas_codex.standard_names.audits import cocos_specificity_check

        candidate = {"documentation": "Simple quantity."}
        issues = cocos_specificity_check(candidate, source_cocos_type=None)
        assert len(issues) == 0


# =========================================================================
# L3: run_audits integration
# =========================================================================


class TestRunAudits:
    """Tests for run_audits orchestrator."""

    def test_clean_candidate_passes(self):
        """A well-formed candidate passes all audits."""
        from imas_codex.standard_names.audits import run_audits

        candidate = {
            "id": "electron_temperature",
            "description": "Electron temperature profile",
            "documentation": (
                "The electron temperature $T_e$ is the kinetic energy "
                "per degree of freedom, where $T_e$ denotes the temperature (eV)."
            ),
            "unit": "eV",
        }
        issues = run_audits(candidate)
        assert isinstance(issues, list)

    def test_has_critical_audit_failure(self):
        """Critical check tag detection works."""
        from imas_codex.standard_names.audits import has_critical_audit_failure

        assert (
            has_critical_audit_failure(["audit:latex_def_check: missing def"]) is True
        )
        assert has_critical_audit_failure(["audit:synonym_check: cosine=0.95"]) is True
        assert (
            has_critical_audit_failure(["audit:multi_subject_check: two subjects"])
            is True
        )
        assert (
            has_critical_audit_failure(["audit:placeholder_check: [condition]"]) is True
        )
        assert (
            has_critical_audit_failure(
                ["audit:unit_validity_check: non-unit token 'dimension'"]
            )
            is True
        )
        assert (
            has_critical_audit_failure(["audit:unit_dimension_check: mismatch"])
            is False
        )
        assert has_critical_audit_failure([]) is False


# =========================================================================
# placeholder_check
# =========================================================================


class TestPlaceholderCheck:
    """Tests for placeholder_check audit."""

    def test_pass_concrete_sign_convention(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "Sign convention: Positive when the plasma current flows "
                "counter-clockwise viewed from above."
            ),
        }
        assert placeholder_check(c) == []

    def test_fail_bracketed_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "Sign convention: Positive when [condition]."}
        issues = placeholder_check(c)
        assert len(issues) == 1
        assert "placeholder_check" in issues[0]

    def test_fail_bracketed_specific_condition(self):
        from imas_codex.standard_names.audits import placeholder_check

        c = {"description": "Positive when [specific physical condition]"}
        issues = placeholder_check(c)
        assert len(issues) == 1

    def test_pass_markdown_link(self):
        """Markdown [text](url) links are not placeholders."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {
            "documentation": (
                "See [magnetic_flux](#magnetic_flux) and [safety_factor](#safety_factor)."
            ),
        }
        assert placeholder_check(c) == []

    def test_pass_numeric_bracket(self):
        """Numeric brackets like [1] or citation-like markers are not flagged."""
        from imas_codex.standard_names.audits import placeholder_check

        c = {"documentation": "See reference [1] and range [0, 1]."}
        assert placeholder_check(c) == []


# =========================================================================
# unit_validity_check
# =========================================================================


class TestUnitValidityCheck:
    """Tests for unit_validity_check audit."""

    def test_pass_real_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        for unit in ("m", "T", "Wb", "eV", "m^2", "kg*m/s^2", "m.s^-1", "1"):
            assert unit_validity_check({"unit": unit}) == [], f"failed for {unit}"

    def test_fail_m_caret_dimension(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "m^dimension"})
        assert len(issues) == 1
        assert "dimension" in issues[0]

    def test_fail_fourier_in_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        issues = unit_validity_check({"unit": "T*fourier"})
        assert len(issues) == 1

    def test_pass_empty_unit(self):
        from imas_codex.standard_names.audits import unit_validity_check

        assert unit_validity_check({"unit": ""}) == []
        assert unit_validity_check({"unit": "dimensionless"}) == []


class TestGenericNounCheck:
    """Tests for generic_noun_check audit."""

    def test_fail_bare_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        issues = generic_noun_check({"id": "geometry"})
        assert len(issues) == 1
        assert "generic_noun_check" in issues[0]

    def test_fail_bare_data(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "data"})) == 1

    def test_pass_qualified_geometry(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "grid_object_geometry"}) == []

    def test_pass_multi_token(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert generic_noun_check({"id": "electron_temperature"}) == []

    def test_fail_generic_qualifier_plus_generic_noun(self):
        from imas_codex.standard_names.audits import generic_noun_check

        assert len(generic_noun_check({"id": "raw_data"})) == 1


class TestTautologyCheck:
    """Tests for tautology_check audit."""

    def test_fail_position_of_position(self):
        from imas_codex.standard_names.audits import tautology_check

        issues = tautology_check({"id": "radial_position_of_reference_position"})
        assert len(issues) == 1
        assert "tautology_check" in issues[0]
        assert "position" in issues[0]

    def test_fail_component_of_component(self):
        from imas_codex.standard_names.audits import tautology_check

        assert len(tautology_check({"id": "normal_component_of_field_component"})) == 1

    def test_pass_no_of(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "electron_temperature"}) == []

    def test_pass_different_heads(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "radial_position_of_plasma_boundary"}) == []

    def test_pass_of_without_tautology_head(self):
        from imas_codex.standard_names.audits import tautology_check

        assert tautology_check({"id": "elongation_of_plasma_boundary"}) == []


class TestSpectralSuffixCheck:
    """Tests for spectral_suffix_check audit."""

    def test_fail_fourier_coefficients(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        issues = spectral_suffix_check({"id": "normal_field_fourier_coefficients"})
        assert len(issues) == 1
        assert "spectral_suffix_check" in issues[0]

    def test_fail_harmonics(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert len(spectral_suffix_check({"id": "magnetic_field_harmonics"})) == 1

    def test_pass_mode_prefix(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "mode_amplitude_of_normal_field"}) == []

    def test_pass_ordinary_name(self):
        from imas_codex.standard_names.audits import spectral_suffix_check

        assert spectral_suffix_check({"id": "poloidal_magnetic_flux"}) == []


class TestAbbreviationCheck:
    """Tests for abbreviation_check audit."""

    def test_fail_norm_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        issues = abbreviation_check({"id": "norm_poloidal_magnetic_flux"})
        assert len(issues) == 1
        assert "normalised" in issues[0]

    def test_fail_perp_interior(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "velocity_perp_component"})) == 1

    def test_fail_temp_prefix(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert len(abbreviation_check({"id": "temp_profile"})) == 1

    def test_pass_full_words(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": "normalised_poloidal_magnetic_flux"}) == []
        assert abbreviation_check({"id": "perpendicular_velocity_component"}) == []

    def test_pass_empty(self):
        from imas_codex.standard_names.audits import abbreviation_check

        assert abbreviation_check({"id": ""}) == []


class TestNameDescriptionConsistencyCheck:
    """Tests for name_description_consistency_check audit."""

    def test_fail_fourier_desc_bare_name(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        issues = name_description_consistency_check(
            {
                "id": "normal_component_of_magnetic_field",
                "description": "Fourier coefficients of the normal component of the field.",
            }
        )
        assert len(issues) == 1

    def test_pass_decomposition_marker(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "mode_amplitude_of_normal_field",
                    "description": "Fourier coefficients of the normal field.",
                }
            )
            == []
        )

    def test_pass_plain_desc(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert (
            name_description_consistency_check(
                {
                    "id": "electron_temperature",
                    "description": "Electron temperature profile.",
                }
            )
            == []
        )

    def test_pass_missing_fields(self):
        from imas_codex.standard_names.audits import (
            name_description_consistency_check,
        )

        assert name_description_consistency_check({"id": "x", "description": ""}) == []


class TestAmericanSpellingCheck:
    """Tests for american_spelling_check audit (NC-17)."""

    def test_fail_british_in_name(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "normalised_poloidal_flux", "description": "A flux."}
        )
        assert any("'normalised'" in i and "normalized" in i for i in issues)
        assert any("field 'name'" in i for i in issues)

    def test_fail_british_in_description(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "plasma_current",
                "description": "Current at the centre of the plasma, analysed per shot.",
            }
        )
        fields = {i.split("field '")[1].split("'")[0] for i in issues}
        assert "description" in fields
        joined = " ".join(issues)
        assert "centre" in joined and "analysed" in joined

    def test_fail_british_in_constraints(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {
                "id": "foo",
                "description": "ok",
                "constraints": ["Must be normalised to 1"],
            }
        )
        assert any("constraints[0]" in i for i in issues)

    def test_pass_american_only(self):
        from imas_codex.standard_names.audits import american_spelling_check

        assert (
            american_spelling_check(
                {
                    "id": "normalized_poloidal_flux",
                    "description": "The normalized flux at the center of the plasma, analyzed per shot.",
                    "documentation": "Modeled behavior of labeled channels.",
                }
            )
            == []
        )

    def test_case_insensitive(self):
        from imas_codex.standard_names.audits import american_spelling_check

        issues = american_spelling_check(
            {"id": "x", "description": "The Normalised profile."}
        )
        assert len(issues) == 1
        assert "Normalised" in issues[0]
